# Imports/Models/fusion_cnn_gru.py
import torch
import torch.nn as nn
from Imports.Models.MoViNet.models import MoViNet, ConvBlock3D, TemporalCGAvgPool3D
from Imports.Models.MoViNet.config import _C as config
from Imports.Models.light_cnn_gru import IMUCNNGRU

class ModifiedMoViNet(nn.Module):
    def __init__(self, cfg, causal=False, pretrained=False):
        super().__init__()
        self.movinet = MoViNet(cfg, causal=causal, pretrained=pretrained)
        self.movinet.classifier = nn.Identity()  # return feature maps

    def forward(self, x):
        return self.movinet(x)

class FusionModelCNNGRU(nn.Module):
    def __init__(self, movinet_config, num_classes,
                 imu_input_size=27, cnn_channels=128,
                 gru_hidden=512, gru_layers=2, dropout=0.4,
                 proj_imu_size=512, freeze_backbone=False, causal=False):
        super().__init__()

        # video backbone
        self.movinet = ModifiedMoViNet(movinet_config, causal=causal, pretrained=True)
        if freeze_backbone:
            for p in self.movinet.parameters():
                p.requires_grad = False

        # IMU CNN+GRU encoder
        self.imu_encoder = IMUCNNGRU(
            input_size=imu_input_size,
            cnn_channels=cnn_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
            proj_size=proj_imu_size
        )

        # fusion classifier 
        in_ch = 992  
        self.classifier = nn.Sequential(
            ConvBlock3D(in_ch, 256, kernel_size=(1,1,1), tf_like=True, causal=causal, conv_type='3d', bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            ConvBlock3D(256, num_classes, kernel_size=(1,1,1), tf_like=True, causal=causal, conv_type='3d', bias=True)
        )

        if causal:
            self.cgap = TemporalCGAvgPool3D()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, video_frames, imu_data):
        B = video_frames.size(0)
        video_features = self.movinet(video_frames)
        imu_features = self.imu_encoder(imu_data)

        fused = torch.cat([video_features, imu_features], dim=1)
        fused = self.dropout(fused)
        x = fused.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        out = self.classifier(x)
        if hasattr(self, 'cgap'):
            out = self.cgap(out)
        return out.view(B, -1)
