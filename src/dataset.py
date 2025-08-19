# COMPLETELY FIXED src/dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from src.utils import rgb_to_lab_norm, create_random_hints, build_model_input

class ColorizeDataset(Dataset):
    """
    Loads RGB images, converts to LAB.
    Input for model: (4,H,W) as [L, hint_mask, hint_ab(2)] with randomized training hints.
    Target: AB (2,H,W)
    """
    def __init__(self, root_dir: str, size: int = 256, max_hints: int = 20):
        self.root_dir = root_dir
        self.size = size
        self.max_hints = max_hints
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(exts)]
        self.files.sort()
        print(f"Found {len(self.files)} images in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert('RGB').resize((self.size, self.size), Image.BICUBIC)
        rgb = np.array(img, dtype=np.uint8)  # HxWx3 RGB
        L, AB = rgb_to_lab_norm(rgb)  # L in [0,1] shape (H,W,1), AB in [-1,1] shape (H,W,2)
        k = random.randint(0, self.max_hints)
        hint_mask, hint_ab = create_random_hints(AB, k=k, sigma=7)
        x = build_model_input(L, hint_mask, hint_ab)          # (4,H,W) float32
        y = torch.from_numpy(AB.transpose(2, 0, 1)).float()   # (2,H,W)
        return x, y, rgb  # include original for visualization