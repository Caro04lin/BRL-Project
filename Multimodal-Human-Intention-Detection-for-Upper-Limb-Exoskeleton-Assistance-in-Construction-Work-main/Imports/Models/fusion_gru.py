# Imports/Models/fusion_gru.py
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from Imports.Models.MoViNet.models import MoViNet
from Imports.Models.light_gru import IMUGRU
from Imports.Models.MoViNet.config import _C as config
from Imports.Models.MoViNet.models import ConvBlock3D, TemporalCGAvgPool3D

class ModifiedMoViNet(nn.Module):
    def __init__(self, cfg, causal=False, pretrained=False):
        super().__init__()
        self.movinet = MoViNet(cfg, causal=causal, pretrained=pretrained)
        # remove classifier to get features
        self.movinet.classifier = nn.Identity()

    def forward(self, x):
        return self.movinet(x)

class FusionModelGRU(nn.Module):
    def __init__(self, movinet_config, num_classes, imu_input_size, gru_hidden_size=512, gru_num_layers=2, gru_dropout=0.3, causal=False):
        super().__init__()
        self.movinet = ModifiedMoViNet(movinet_config, causal=causal, pretrained=True)
        self.gru = IMUGRU(input_size=imu_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers, dropout=gru_dropout, bidirectional=False)
        
        movinet_feat_size = 480  
       
        combined_feature_size = movinet_feat_size + gru_hidden_size

        self.classifier = nn.Sequential(
            ConvBlock3D(combined_feature_size, 512, kernel_size=(1,1,1), tf_like=True, causal=causal, conv_type='3d', bias=True),
            nn.Dropout(p=0.2),
            ConvBlock3D(512, num_classes, kernel_size=(1,1,1), tf_like=True, causal=causal, conv_type='3d', bias=True)
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()

    def forward(self, video_frames, imu_data):
        # video_features: expected (B, movinet_feat)
        video_features = self.movinet(video_frames)  # shape depends on movinet
        if video_features.ndim > 2:
            # If movinet returns (B, C, T?, H?, W?) -> flatten spatial/time dims as needed
            video_features = torch.flatten(video_features, start_dim=1)

        imu_features = self.gru(imu_data)  # (B, gru_hidden_size)
        combined = torch.cat((video_features, imu_features), dim=1)  # (B, combined_feature_size)
        # reshape to (B, C, T, H, W) -> use 1x1x1 conv, so expand dims
        combined_features = combined.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        out = self.classifier(combined_features)
        if hasattr(self, 'cgap'):
            out = self.cgap(out)
        return out.squeeze()
