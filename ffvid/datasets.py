import os
from math import pi
from glob import glob

import torch
from PIL import Image
from torchvision import transforms as TF
from torch.utils.data import Dataset, DataLoader

from ffvid.utils import get_grid


class FramesDataset(Dataset):
    def __init__(self, folder, resolution=256, phase_mod=False):
        super().__init__()
        self.files = sorted(glob(os.path.join(folder, '*.png')))
        self.num_frames = len(self.files)
        self.resolution = resolution
        self.phase_mod = phase_mod
        
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = TF.Resize(self.resolution)(img)
        img = TF.ToTensor()(img)
        coords = get_grid(*img.shape[-2:], 0, True, 'cpu')
        if self.phase_mod:
            timestamp = int(self.files[idx].split('/')[-1].split('.')[0].split('_')[-1])
            timestamp_float = timestamp / self.num_frames
            coords = [coords, torch.cat([coords, coords[[0]] * 0. + timestamp_float], dim=0)]
        else:
            timestamp = int(self.files[idx].split('/')[-1].split('.')[0].split('_')[-1])
            timestamp_float = timestamp / self.num_frames
            coords = torch.cat([coords, coords[[0]] * 0. + timestamp_float], dim=0)
        return img, coords