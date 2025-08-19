# Fixed src/losses.py
import torch
import torch.nn as nn

class ColorL1Loss(nn.Module):
    """Simple L1 on AB channels."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_ab, target_ab):
        return self.l1(pred_ab, target_ab)