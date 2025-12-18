#dataloader for LSTM
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader


class HARDataSet(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.samples = []

        # List of validated files
        all_samples = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Sample_")]
        total_samples = len(all_samples)

        for sample_name in all_samples:
            sample_path = os.path.join(root_dir, sample_name)

            frames = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith('.jpg')]
            frames.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

            imu_path = os.path.join(sample_path, 'imu_new.csv')
            if not os.path.isfile(imu_path):
                # Extract the numero of the sample
                num = int(sample_name.replace("Sample_", ""))
                # Display the warning only if this is not the last sample (The last file is not completely filled which is normal because we are continuously collecting data)
                if num < total_samples:
                    print(f"Warning: imu_new.csv missing in {sample_name}")
                continue

            if len(frames) >= sequence_length:
                for i in range(0, len(frames) - sequence_length + 1):
                    self.samples.append((frames[i:i + sequence_length], imu_path, i))
            
        # Define SampleNumber once
        if self.samples:
            first_image_path = self.samples[0][0][0]
            self.SampleNumber = os.path.basename(os.path.dirname(first_image_path))
        else:
            self.SampleNumber = ''

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        try:
            frames_path, imu_path, start_idx = self.samples[idx]
            sample_folder = os.path.basename(os.path.dirname(frames_path[0]))
            self.SampleNumber = sample_folder

            frames = [Image.open(frame).convert("RGB") for frame in frames_path]
            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            imu_data = pd.read_csv(imu_path)
            imu_data = imu_data.iloc[start_idx:start_idx + len(frames_path)].reset_index(drop=True)

            if len(imu_data) != len(frames_path):
                raise ValueError(f"Mismatch: {len(imu_data)} imu rows vs {len(frames_path)} frames")

            frames_tensor = torch.stack(frames)
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)

            return frames_tensor, torch.tensor(imu_data.values, dtype=torch.float32)

        except Exception as e:
            print(f"Error at sample index {idx}, path: {frames_path[0]}, error: {e}")
            raise

