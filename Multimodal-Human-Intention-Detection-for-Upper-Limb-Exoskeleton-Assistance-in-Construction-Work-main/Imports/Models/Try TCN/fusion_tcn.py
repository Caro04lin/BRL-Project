from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from Imports.Models.MoViNet.models import MoViNet
from Imports.Models.light_tcn2 import IMUTCN
from Imports.Models.MoViNet.config import _C as config
from Imports.Models.MoViNet.models import ConvBlock3D, TemporalCGAvgPool3D

class ModifiedMoViNet(nn.Module):
    def __init__(self, cfg, causal=False, pretrained=False):
        super().__init__()
        self.movinet = MoViNet(cfg, causal=causal, pretrained=pretrained)
        self.movinet.classifier = nn.Identity()  # 将分类器替换为恒等映射，直接输出前一层的特征

    def forward(self, x):
        return self.movinet(x)

class FusionModelTCN(nn.Module):
    def __init__(self, movinet_config, num_classes, tcn_input_size, tcn_hidden_size, tcn_num_layers, causal=False):
        super().__init__()
        self.movinet = ModifiedMoViNet(movinet_config, causal=causal, pretrained=True)
        self.tcn = IMUTCN(input_size=tcn_input_size, hidden_size=tcn_hidden_size,
                          num_layers=tcn_num_layers, num_classes=num_classes)
        combined_feature_size = 992  # adjust if needed

        self.classifier = nn.Sequential(
            ConvBlock3D(combined_feature_size, 512, kernel_size=(1, 1, 1),
                        tf_like=True, causal=causal, conv_type='3d', bias=True),
            nn.Dropout(p=0.2),
            ConvBlock3D(512, num_classes, kernel_size=(1, 1, 1),
                        tf_like=True, causal=causal, conv_type='3d', bias=True)
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()

    def forward(self, video_frames, imu_data):
        video_features = self.movinet(video_frames)   # [batch, 992]
        imu_features = self.tcn(imu_data)             # [batch, hidden_size]
        combined_features = torch.cat((video_features, imu_features), dim=1)
        combined_features = combined_features.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        output = self.classifier(combined_features)
        if hasattr(self, 'cgap'):
            output = self.cgap(output)
        return output.squeeze()

movinet_config = config.MODEL.MoViNetA0
num_classes = 10
tcn_input_size = 27
tcn_hidden_size = 512
tcn_num_layers = 2

model = FusionModelTCN(movinet_config, num_classes, tcn_input_size, tcn_hidden_size, tcn_num_layers)

