import torch.nn as nn


class KichwaDecoder1D(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, vocab_size=50, dropout=0.25):
        super(KichwaDecoder1D, self).__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, padding=12, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)
        logits = self.classifier(x)
        return logits
