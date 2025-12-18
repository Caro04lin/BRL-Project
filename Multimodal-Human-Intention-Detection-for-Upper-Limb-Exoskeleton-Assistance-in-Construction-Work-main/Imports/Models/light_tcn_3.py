# Imports/Models/light_tcn_3.py        
import torch
import torch.nn as nn

# ---- Chomp1d ----
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


# ---- TemporalBlock ----
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return self.relu(out + res)


# TemporalConvNet
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.4):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation, padding=(kernel_size - 1) * dilation,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# IMU TCN with Global Avg Pooling
class IMUTCN(nn.Module):
    def __init__(self, input_size=27, hidden_size=512, num_layers=6, dropout=0.4, proj_size=512):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, [hidden_size] * num_layers,
                                   kernel_size=3, dropout=dropout)
        out_proj_size = proj_size if proj_size is not None else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, out_proj_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T, features)
        x = x.transpose(1, 2)             # (B, features, T)
        out = self.tcn(x)                 # (B, hidden, T)
        out = out.mean(dim=2)             # Global Average Pooling over temporal dim -> (B, hidden)
        return self.fc(out)               # (B, out_proj_size)