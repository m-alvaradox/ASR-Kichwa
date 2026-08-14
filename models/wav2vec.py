import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 2, 2), (512, 2, 2),
        ]
        self.conv_layers = nn.ModuleList()
        in_dim = 1
        for i, (out_dim, kernel, stride) in enumerate(conv_layers):
            layer = nn.ModuleDict({
                "conv": nn.Conv1d(in_dim, out_dim, kernel, stride, bias=False)
            })
            if i == 0:
                layer["layer_norm"] = nn.GroupNorm(out_dim, out_dim)
            self.conv_layers.append(layer)
            in_dim = out_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer["conv"](x)
            if "layer_norm" in layer:
                x = layer["layer_norm"](x)
            x = nn.functional.gelu(x)
        return x.transpose(1, 2)


class PositionalConvEmbedding(nn.Module):
    def __init__(self, dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.kernel_size % 2 == 0:
            x = x[:, :, :-1]
        x = nn.functional.gelu(x)
        return x.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = nn.ModuleDict({
            "q_proj": nn.Linear(dim, dim),
            "k_proj": nn.Linear(dim, dim),
            "v_proj": nn.Linear(dim, dim),
            "out_proj": nn.Linear(dim, dim),
        })
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.ModuleDict({
            "intermediate_dense": nn.Linear(dim, ff_dim),
            "output_dense": nn.Linear(ff_dim, dim),
        })
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, x):
        B, T, D = x.shape
        q = self.attention["q_proj"](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attention["k_proj"](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attention["v_proj"](x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.attention["out_proj"](out)

    def forward(self, x):
        residual = x
        x = self._attn(x)
        x = residual + self.dropout(x)
        x = self.layer_norm(x)

        residual = x
        x = self.feed_forward["intermediate_dense"](x)
        x = nn.functional.gelu(x)
        x = self.feed_forward["output_dense"](x)
        x = residual + self.dropout(x)
        x = self.final_layer_norm(x)
        return x


class Wav2Vec2Partial(nn.Module):
    def __init__(self, num_transformer_layers=6, dim=768):
        super().__init__()
        self.feature_extractor = FeatureEncoder()
        self.feature_projection = nn.Linear(512, dim)
        self.pos_conv_embed = PositionalConvEmbedding(dim=dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim) for _ in range(num_transformer_layers)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.feature_projection(x)
        x = x + self.pos_conv_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x