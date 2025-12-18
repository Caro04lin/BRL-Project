# Imports/Models/light_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUGRU(nn.Module):
    """
    GRU encoder for IMU sequences.
    Input: (batch, seq_len, input_size)
    Output: (batch, hidden_size) -> feature vector for fusion
    """
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=False):
        super(IMUGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU (batch_first=True keeps input as (B, T, C))
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)

        # After GRU we project to a compact feature (keep size = hidden_size for fusion)
        self.fc_proj = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, T, input_size)
        # Initialize hidden state (num_layers * num_directions, B, hidden_size)
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size, device=x.device)

        out, hn = self.gru(x, h0)  # out: (B, T, hidden_size * num_directions)
        # use last time-step
        last = out[:, -1, :]  # (B, hidden_size * num_directions)
        last = self.dropout(last)
        feat = F.relu(self.fc_proj(last))  # (B, hidden_size)
        return feat
