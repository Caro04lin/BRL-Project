# Imports/Models/fusion_tcn_3.py
import torch
import torch.nn as nn
from Imports.Models.MoViNet.models import MoViNet, ConvBlock3D, TemporalCGAvgPool3D
from Imports.Models.MoViNet.config import _C as config
from Imports.Models.light_tcn_3 import IMUTCN

# Modified MoViNet that returns features (remove final classifier)
class ModifiedMoViNet(nn.Module):
    def __init__(self, cfg, causal=False, pretrained=False):
        super().__init__()
        self.movinet = MoViNet(cfg, causal=causal, pretrained=pretrained)
        # remove final classifier to get features
        self.movinet.classifier = nn.Identity()

    def forward(self, x):
        return self.movinet(x)  # expected (B, feat_dim)


class FusionModelTCN(nn.Module):
    def __init__(self, movinet_config, num_classes,
                 tcn_input_size=27, tcn_hidden_size=512, tcn_num_layers=6,
                 tcn_dropout=0.4, proj_imu_size=512, freeze_backbone=False, causal=False):
        super().__init__()
        # video backbone
        self.movinet = ModifiedMoViNet(movinet_config, causal=causal, pretrained=True)
        if freeze_backbone:
            for p in self.movinet.parameters():
                p.requires_grad = False

        # imu tcn -> project to a size that matches movinet feature magnitude
        self.tcn = IMUTCN(input_size=tcn_input_size,
                          hidden_size=tcn_hidden_size,
                          num_layers=tcn_num_layers,
                          dropout=tcn_dropout,
                          proj_size=proj_imu_size)

        self.dropout = nn.Dropout(p=tcn_dropout)
        self.num_classes = num_classes
        self.causal = causal
        if causal:
            self.cgap = TemporalCGAvgPool3D()

        # Statically constructed classifier
        in_ch = 992  # MoViNet features + TCN features concat (like for training)
        self.classifier_conv = nn.Sequential(
            ConvBlock3D(in_ch, 256, kernel_size=(1,1,1), tf_like=True, causal=self.causal, conv_type='3d', bias=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            ConvBlock3D(256, self.num_classes, kernel_size=(1,1,1), tf_like=True, causal=self.causal, conv_type='3d', bias=True)
        )

    def forward(self, video_frames, imu_data):
        B = video_frames.size(0)

        video_features = self.movinet(video_frames)
        imu_features = self.tcn(imu_data)

        combined = torch.cat([video_features, imu_features], dim=1)
        combined = self.dropout(combined)

        x = combined.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        out = self.classifier_conv(x)
        if hasattr(self, 'cgap'):
            out = self.cgap(out)
        out = out.view(B, self.num_classes)
        return out
