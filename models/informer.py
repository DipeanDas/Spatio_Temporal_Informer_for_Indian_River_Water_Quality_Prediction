import torch
import torch.nn as nn
from models.embed import DataEmbedding
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import ProbAttention, FullAttention, AttentionLayer

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=2048, dropout=0.0, attn='prob', embed='timeF',
                 freq='m', activation='gelu', output_attention=False, distil=True, mix=True, n_locations=44):
        super(Informer, self).__init__()

        self.pred_len = pred_len
        self.output_attention = output_attention

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed_type=embed, dropout=dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed_type=embed, dropout=dropout)
        self.location_embedding = nn.Embedding(n_locations, d_model)  # Spatial embedding for locations

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            conv_layer=ConvLayer(d_model) if distil else None
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, location_enc, location_dec):
        """
        Forward pass through the model.

        x_enc: Encoder input (batch_x)
        x_mark_enc: Temporal features for the encoder (batch_x_mark)
        x_dec: Decoder input (dec_inp, typically previous predictions)
        x_mark_dec: Temporal features for the decoder (batch_y_mark)
        location_enc: Spatial features for the encoder (batch_location_x)
        location_dec: Spatial features for the decoder (batch_location_y)
        """
        
        # Encoding phase
        enc_out = self.enc_embedding(x_enc, x_mark_enc, location_enc)  # Add location embeddings to encoder input
        enc_out = enc_out + self.location_embedding(location_enc)  # Add spatial embeddings to encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoding phase
        dec_out = self.dec_embedding(x_dec, x_mark_dec, location_dec)  # Add location embeddings to decoder input
        dec_out = dec_out + self.location_embedding(location_dec)  # Add spatial embeddings to decoder
        dec_out, _ = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # Return output based on attention configuration
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
