import torch
import torch.nn as nn

class KichwaDecoder1D(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, vocab_size=50):
        super(KichwaDecoder1D, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),   
            nn.ReLU(),  
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),  
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),  
            nn.Dropout(0.1)
        )

        # Inyección de Memoria Temporal 
        # hidden_size = 256 para minimizar consumo computacional. 
        # Al ser bidireccional, la salida volverá a ser 512.
        self.rnn = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = x.transpose(1, 2) # Forma resultante: [batch, tiempo, 512]
        
        self.rnn.flatten_parameters() 
        
        x, _ = self.rnn(x) 
        
        logits = self.classifier(x)
        return logits