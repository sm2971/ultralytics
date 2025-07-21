# MTL project: Dataset for joint segmentation and landmark detection
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MTLDataset(Dataset):
    def __init__(self, img_dir, seg_dir, lm_dir, transform=None, size=(640, 640)):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.lm_dir = lm_dir
        self.transform = transform
        self.size = size
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        seg_path = os.path.join(self.seg_dir, img_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
        lm_path = os.path.join(self.lm_dir, img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        # Load image
        img = Image.open(img_path).convert('RGB').resize(self.size)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CxHxW

        # Load segmentation mask
        seg = Image.open(seg_path).resize(self.size)
        seg = np.array(seg).astype(np.float32) / 255.0
        if len(seg.shape) == 3:
            seg = seg[..., 0]  # Use first channel if mask is RGB
        seg = np.expand_dims(seg, 0)  # 1xHxW

        # Load landmarks
        with open(lm_path, 'r') as f:
            lm_coords = [list(map(float, line.strip().split(','))) for line in f.readlines()]
        lm_coords = np.array(lm_coords)
        # Optionally, convert to heatmaps or keep as coordinates

        # Apply transform if provided
        if self.transform:
            img, seg, lm_coords = self.transform(img, seg, lm_coords)

        return {
            'image': torch.tensor(img, dtype=torch.float32),
            'segmentation': torch.tensor(seg, dtype=torch.float32),
            'landmarks': torch.tensor(lm_coords, dtype=torch.float32)
        } 