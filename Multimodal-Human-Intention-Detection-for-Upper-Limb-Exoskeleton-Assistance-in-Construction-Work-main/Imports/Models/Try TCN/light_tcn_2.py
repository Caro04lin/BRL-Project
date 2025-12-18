import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Chomp1d pour enlever padding ----
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

# ---- Temporal Block avec skip connection ----
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.LayerNorm(out_channels)  # stable training
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.LayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch, channels, seq_len)
        """
        out = self.conv1(x)
        out = self.chomp1(out)
        out = out.transpose(1, 2)         # (B, seq_len, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)         # (B, channels, seq_len)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# ---- Temporal Conv Net ----
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1,
                dilation=dilation,
                padding=(kernel_size-1)*dilation,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ---- IMU TCN Model ----
class IMUTCN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, [hidden_size]*num_layers, kernel_size=3, dropout=dropout)
        self.fc_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)          # (batch, input_size, seq_len) pour Conv1d
        out = self.tcn(x)              # (batch, hidden_size, seq_len)
        out = out[:, :, -1]            # dernier pas de temps → (batch, hidden_size)
        out = self.fc_proj(out)        # (batch, hidden_size)
        return out
