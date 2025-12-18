import torch
import torch.nn as nn
from Imports.Models.MoViNet.models import MoViNet, ConvBlock3D, TemporalCGAvgPool3D
from Imports.Models.MoViNet.config import _C as config
from Imports.Models.light_tcn_2 import IMUTCN

# ---- MoViNet modifié pour extraire les features ----
class ModifiedMoViNet(nn.Module):
    def __init__(self, cfg, causal=False, pretrained=False):
        super().__init__()
        self.movinet = MoViNet(cfg, causal=causal, pretrained=pretrained)
        self.movinet.classifier = nn.Identity()  # enlever la classification

    def forward(self, x):
        # x: (batch, 3, T, H, W)
        return self.movinet(x)  # (batch, feat_dim)

# ---- Fusion Model: Video + IMU ----
class FusionModelTCN(nn.Module):
    def __init__(self, movinet_config, num_classes,
                 tcn_input_size, tcn_hidden_size=128, tcn_num_layers=2, causal=False):
        super().__init__()
        self.movinet = ModifiedMoViNet(movinet_config, causal=causal, pretrained=True)
        self.tcn = IMUTCN(tcn_input_size,
                          hidden_size=tcn_hidden_size,
                          num_layers=tcn_num_layers)

        self.classifier_dropout = nn.Dropout(p=0.2)
        self.causal = causal
        self.num_classes = num_classes

        # Placeholder pour la première couche Conv3D, sera ajusté dynamiquement
        self.classifier_conv = None
        if causal:
            self.cgap = TemporalCGAvgPool3D()

    def forward(self, video_frames, imu_data):
        """
        video_frames: (batch, 3, T, H, W)
        imu_data: (batch, T, imu_features)
        """
        # --- Extraire features vidéo ---
        video_features = self.movinet(video_frames)      # (batch, C1)
        # --- Extraire features IMU ---
        imu_features = self.tcn(imu_data)               # (batch, tcn_hidden)

        # --- Fusion ---
        combined_features = torch.cat((video_features, imu_features), dim=1)  # (batch, C1 + hidden)

        # --- Reshape pour Conv3D ---
        combined_features = combined_features.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, C, 1, 1, 1)

        # --- Créer le Conv3D dynamiquement si pas encore fait ---
        if self.classifier_conv is None:
            in_channels = combined_features.shape[1]
            self.classifier_conv = nn.Sequential(
                ConvBlock3D(in_channels, 512, kernel_size=(1, 1, 1),
                            tf_like=True, causal=self.causal, conv_type='3d', bias=True),
                self.classifier_dropout,
                ConvBlock3D(512, self.num_classes, kernel_size=(1, 1, 1),
                            tf_like=True, causal=self.causal, conv_type='3d', bias=True)
            )
            self.classifier_conv.to(combined_features.device)

        # --- Classification ---
        output = self.classifier_conv(combined_features)
        if hasattr(self, 'cgap'):
            output = self.cgap(output)

        # output: (batch, num_classes)
        return output.squeeze()





