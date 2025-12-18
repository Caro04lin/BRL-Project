# Imports/dataloader_3.py for TCN, CNN-GRU and GRU models
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import re
from PIL import Image

# The names of samples collected after data collection (with safe_imu_data_collection.py code) are of the type ‘Sample_1’, “Sample_2”, ..., ‘Sample_N’
class HARDataSet(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10, require_complete_imu=True):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.require_complete_imu = require_complete_imu
        self.samples = []

        all_samples = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Sample_")]
        # sort numerically
        all_samples = sorted(all_samples, key=lambda x: int(re.sub(r"[^\d]", "", x) or 0))

        for sample_name in all_samples:
            sample_path = os.path.join(root_dir, sample_name)
            frames = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.lower().endswith('.jpg')]
            if not frames:
                continue
            frames.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
            imu_path = os.path.join(sample_path, "imu_new.csv")
            if not os.path.isfile(imu_path):
                if self.require_complete_imu:
                    continue
                else:
                    # skip sample if imu missing
                    continue
            if len(frames) < sequence_length:
                continue
            for i in range(0, len(frames) - sequence_length + 1):
                self.samples.append((frames[i:i+sequence_length], imu_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, imu_path, start_idx = self.samples[idx]
        frames = [Image.open(f).convert("RGB") for f in frames_path]
        if self.transform:
            frames = [self.transform(im) for im in frames]
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

        imu_df = pd.read_csv(imu_path)
        imu_rows = imu_df.iloc[start_idx:start_idx + len(frames_path)].values
        imu_tensor = torch.tensor(imu_rows, dtype=torch.float32)
        if imu_tensor.shape[0] < len(frames_path):
            pad_len = len(frames_path) - imu_tensor.shape[0]
            imu_tensor = torch.cat([imu_tensor, torch.zeros((pad_len, imu_tensor.shape[1]))], dim=0)

        return frames_tensor, imu_tensor

# The names of samples in the Action dataset (with safe_imu_data_collection.py code) are of the type ‘1’, “2”, ..., ‘N’

class HARDataSetTrain(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = []
        self.label_map = {}

        action_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for label_idx, action_name in enumerate(sorted(action_folders)):
            self.label_map[action_name] = label_idx
            action_path = os.path.join(root_dir, action_name)

            for sample_id in os.listdir(action_path):
                sample_path = os.path.join(action_path, sample_id)
                if not os.path.isdir(sample_path):
                    continue

                frames = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.jpg')]
                frames.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

                imu_path = os.path.join(sample_path, 'imu_new.csv')
                if not os.path.isfile(imu_path):
                    continue

                if len(frames) >= sequence_length:
                    for i in range(0, len(frames) - sequence_length + 1):
                        self.samples.append((frames[i:i + sequence_length], imu_path, i, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_path, imu_path, start_idx, label = self.samples[idx]

        frames = [Image.open(frame).convert("RGB") for frame in frames_path]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C,T,H,W)

        imu_data = pd.read_csv(imu_path).iloc[start_idx:start_idx + self.sequence_length].values
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32)

        if imu_tensor.shape[0] < self.sequence_length:
            pad_len = self.sequence_length - imu_tensor.shape[0]
            imu_tensor = torch.cat([imu_tensor, torch.zeros((pad_len, imu_tensor.shape[1]))], dim=0)

        return frames_tensor, imu_tensor, label