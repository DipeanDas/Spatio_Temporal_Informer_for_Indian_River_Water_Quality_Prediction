import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from utils.tools import metric, save_checkpoint
from utils.data_loader import get_dataloader
from models.informer import Informer 

class Exp_Informer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.val_loader, self.test_loader, self.train_dataset = get_dataloader(
            data_path=config['data_path'],
            seq_len=config['seq_len'],
            label_len=config['label_len'],
            pred_len=config['pred_len'],
            batch_size=config['batch_size'],
            target=config['target'],
            features=config['features'],            
            shuffle=config.get('shuffle', True),
            drop_last=config.get('drop_last', True),
            split_ratio=config.get('split_ratio', (0.7, 0.1, 0.2))
        )       

        self.model = Informer(
            enc_in=config['enc_in'],
            dec_in=config['dec_in'],
            c_out=config['c_out'],
            seq_len=config['seq_len'],
            label_len=config['label_len'],
            pred_len=config['pred_len'],
            factor=config['factor'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            e_layers=config['e_layers'],
            d_layers=config['d_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            attn=config['attn'],
            embed=config['embed'],
            freq=config['freq'],
            activation=config['activation'],
            output_attention=config['output_attention'],
            distil=config['distil']
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.save_path = os.path.join('./checkpoints', config['model_id'])
        os.makedirs(self.save_path, exist_ok=True)

    def train(self):
        min_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_location_x, batch_location_y in self.train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.long().to(self.device)   # convert to long for embeddings
                batch_y_mark = batch_y_mark.long().to(self.device)
                batch_location_x = batch_location_x.long().to(self.device)
                batch_location_y = batch_location_y.long().to(self.device)

                dec_inp = torch.zeros(batch_y.size(0), self.config['label_len'] + self.config['pred_len'], self.config['enc_in']).to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_x[:, -self.config['label_len']:, :]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_location_x, batch_location_y)

                outputs = outputs[:, -self.config['pred_len']:, :]
                loss = self.criterion(outputs, batch_y[:, -self.config['pred_len']:, :])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            val_loss = self.validate()

            print(f"Epoch {epoch + 1}/{self.config['epochs']} - "
                f"Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(self.model, self.save_path, f"{self.config['model_id']}.pth")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_location_x, batch_location_y in self.val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.long().to(self.device)
                batch_y_mark = batch_y_mark.long().to(self.device)
                batch_location_x = batch_location_x.long().to(self.device)
                batch_location_y = batch_location_y.long().to(self.device)

                dec_inp = torch.zeros(batch_y.size(0), self.config['label_len'] + self.config['pred_len'], self.config['enc_in']).to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_x[:, -self.config['label_len']:, :]

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_location_x, batch_location_y)
                loss = self.criterion(outputs, batch_y[:, -self.config['pred_len']:, :])

                total_loss += loss.item()

        return total_loss / len(self.val_loader)   

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, f"{self.config['model_id']}.pth")))
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_location_x, batch_location_y in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.to(self.device).long()
                batch_y_mark = batch_y_mark.to(self.device).long()
                batch_location_x = batch_location_x.to(self.device).long()  # Location indices for encoder
                batch_location_y = batch_location_y.to(self.device).long()  # Location indices for decoder

                # Prepare decoder input (same as training loop)
                dec_inp = torch.zeros(batch_y.size(0), self.config['label_len'] + self.config['pred_len'], self.config['enc_in']).to(self.device)
                dec_inp[:, :self.config['label_len'], :] = batch_x[:, -self.config['label_len']:, :]

                # Pass data to the model, including the spatial (location) embeddings
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_location_x, batch_location_y)
                outputs = outputs[:, -self.config['pred_len']:, :]  # Get only prediction part

                # Store the predictions and true values for evaluation
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y[:, -self.config['pred_len']:, :].cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        trues = np.concatenate(trues, axis=0).reshape(-1, 1)

        # === Inverse transform only target column ===
        scaler = joblib.load('scaler_bod.pkl')
        col_order = joblib.load('column_order.pkl')
        target_idx = col_order.index(self.config['target'])

        dummy_pred = np.zeros((preds.shape[0], len(col_order)))
        dummy_true = np.zeros((trues.shape[0], len(col_order)))

        dummy_pred[:, target_idx] = preds[:, 0]
        dummy_true[:, target_idx] = trues[:, 0]

        preds_inv = scaler.inverse_transform(dummy_pred)[:, target_idx]
        trues_inv = scaler.inverse_transform(dummy_true)[:, target_idx]

        # === Metrics ===
        mse, mae, rmse, r2, plcc, srcc, krcc = metric(preds_inv, trues_inv)

        print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"Correlation - PLCC: {plcc:.4f}, SRCC: {srcc:.4f}, KRCC: {krcc:.4f}")

        # === Save Results ===
        from datetime import datetime
        import matplotlib.pyplot as plt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs("results", exist_ok=True)
        os.makedirs("scores", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        pd.DataFrame({
            'Actual BOD': np.round(trues_inv.reshape(-1), 4),
            'Predicted BOD': np.round(preds_inv.reshape(-1), 4)
        }).to_csv(f"results/pred_vs_true_{timestamp}.csv", index=False)

        with open(f"scores/metrics_{timestamp}.txt", 'w') as f:
            f.write(f"MSE:  {mse:.4f}\n")
            f.write(f"MAE:  {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2:   {r2:.4f}\n")
            f.write(f"PLCC: {plcc:.4f}\n")
            f.write(f"SRCC: {srcc:.4f}\n")
            f.write(f"KRCC: {krcc:.4f}\n")

        plt.figure(figsize=(10, 5))
        x = np.arange(len(trues_inv))
        plt.scatter(x, trues_inv.reshape(-1), color='green', label='Actual BOD', alpha=0.7)
        plt.scatter(x + 0.2, preds_inv.reshape(-1), color='blue', label='Predicted BOD', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('BOD Value')
        plt.title('Actual vs Predicted BOD - Scatter Plot')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/scatter_{timestamp}.png")
        plt.close()

        return preds_inv, trues_inv




