import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Temporal Block ----
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Residual 1x1 conv if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        
        return F.relu(out + res)


# ---- TCN ----
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            
            padding = ((kernel_size - 1) * dilation) // 2

            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size,
                              stride=1, dilation=dilation, padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class IMUTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        channels = [hidden_size] * num_layers
        self.tcn = TemporalConvNet(input_size, channels, kernel_size=3, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        out = self.tcn(x)        # [batch, hidden, seq_len]
        out = out[:, :, -1]      # last time step
        out = F.relu(self.fc1(out))
        return out

