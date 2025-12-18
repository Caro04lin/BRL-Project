# Imports/Models/light_cnn_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUCNNGRU(nn.Module):
    def __init__(self, input_size=27, cnn_channels=128, gru_hidden=512, gru_layers=2, dropout=0.4, proj_size=512):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU temporal encoder
        self.gru = nn.GRU(input_size=cnn_channels,
                          hidden_size=gru_hidden,
                          num_layers=gru_layers,
                          batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0)

        # Projection head
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T, features)
        x = x.transpose(1, 2)                 # → (B, features, T)
        x = self.cnn(x)                       # → (B, C, T)
        x = x.transpose(1, 2)                 # → (B, T, C)
        out, _ = self.gru(x)                  # → (B, T, H)
        out = out[:, -1, :]                   # last time step
        return self.fc(out)                   # (B, proj_size)
