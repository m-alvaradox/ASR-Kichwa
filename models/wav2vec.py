import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 2, 2), (512, 2, 2),
        ]
        self.layers = nn.ModuleList()
        in_dim = 1
        for out_dim, kernel, stride in conv_layers:
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel, stride, bias=False),
                nn.GroupNorm(out_dim, out_dim) if in_dim == 1 else nn.Identity(),
                nn.ReLU()
            ))
            in_dim = out_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Wav2Vec2Partial(nn.Module):
    def __init__(self, num_transformer_layers=6, dim=768):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.feature_projection = nn.Linear(512, dim)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim) for _ in range(num_transformer_layers)
        ])

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.feature_projection(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return x