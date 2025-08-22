# src/__init__.py
"# Package init" 
"""
Image Colorization Package

A deep learning package for conditional image colorization using U-Net architecture.
"""

__version__ = "1.0.0"
__author__ = "Sita Ganesh"

# Import main components for easy access
from .model import UNet
from .dataset import ColorizeDataset
from .losses import ColorL1Loss
from .utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8

__all__ = [
    'UNet', 
    'ColorizeDataset', 
    'ColorL1Loss',
    'rgb_to_lab_norm',
    'lab_norm_to_rgb_uint8'
]