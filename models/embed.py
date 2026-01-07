import torch
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].clone().detach()

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [Batch, Seq Len, Features]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.tokenConv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # [B, L, D]

class SpatialEmbedding(nn.Module):
    def __init__(self, n_locations, d_model):
        super(SpatialEmbedding, self).__init__()
        self.location_embedding = nn.Embedding(n_locations, d_model)  # n_locations = 44 (for 44 rivers)

    def forward(self, locations):
        # locations: [Batch, Seq Len] with location indices
        return self.location_embedding(locations)

# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF'):
#         super().__init__()
#         self.month_embed = nn.Embedding(12, d_model)  # 12 months (0 to 11)
#         self.year_embed = nn.Embedding(3, d_model)    # For year encoding (22, 23, 24)
#         self.embed_type = embed_type

#     def forward(self, x):
#         # x: [Batch, Seq Len, Time Features]
#         month_x = self.month_embed(x[:, :, 0].long())  # Month feature (0-11)
#         year_x = self.year_embed(x[:, :, 1].long() - 22)  # Year feature (0 for 22, 1 for 23, 2 for 24)

#         # Apply sin-cos encoding for cyclical features (month and year)
#         month_sin = torch.sin(2 * np.pi * month_x / 12)  # Sine function for months (0-11)
#         month_cos = torch.cos(2 * np.pi * month_x / 12)  # Cosine function for months

#         year_sin = torch.sin(2 * np.pi * year_x / 3)  # Sine for year (0 for 22, 1 for 23, 2 for 24)
#         year_cos = torch.cos(2 * np.pi * year_x / 3)  # Cosine for year (0 for 22, 1 for 23, 2 for 24)

#         return month_sin + month_cos + year_sin + year_cos
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF'):
        super().__init__()
        self.month_embed = nn.Embedding(12, d_model)  # 12 months (0 to 11)
        self.year_embed = nn.Embedding(3, d_model)    # 3 years (0 for 22, 1 for 23, 2 for 24)
        self.embed_type = embed_type

    def forward(self, x):
        # x: [Batch, Seq Len, Time Features]
        month_x = self.month_embed(x[:, :, 0].long())  # Month feature (0-11)
        
        # Year feature, converted to [0, 1, 2] for 22, 23, 24
        year_x = torch.clamp(x[:, :, 1].long() - 22, min=0, max=2)  # Ensure year is in [0, 1, 2]

        year_x = self.year_embed(year_x)  # Apply embedding to the year

        # Apply sin-cos encoding for cyclical features (month and year)
        month_sin = torch.sin(2 * np.pi * month_x / 12)  # Sine function for months (0-11)
        month_cos = torch.cos(2 * np.pi * month_x / 12)  # Cosine function for months

        year_sin = torch.sin(2 * np.pi * year_x / 3)  # Sine for year (0 for 22, 1 for 23, 2 for 24)
        year_cos = torch.cos(2 * np.pi * year_x / 3)  # Cosine for year (0 for 22, 1 for 23, 2 for 24)

        return month_sin + month_cos + year_sin + year_cos




class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', dropout=0.1, n_locations=44):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.spatial_embedding = SpatialEmbedding(n_locations=n_locations, d_model=d_model)  # Adding spatial embeddings
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, location_idx):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        spatial_x = self.spatial_embedding(location_idx)  # Adding spatial information
        return self.dropout(x + spatial_x)  # Combine temporal and spatial embeddings

