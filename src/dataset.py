# src/dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import logging
from typing import Tuple, List

from src.utils import rgb_to_lab_norm, create_random_hints, build_model_input

logger = logging.getLogger(__name__)

class ColorizeDataset(Dataset):
    """
    Dataset for image colorization with random hints.
    
    Loads RGB images, converts to LAB color space.
    Input: (4,H,W) tensor [L, hint_mask, hint_ab(2)] with randomized training hints
    Target: AB channels (2,H,W)
    """
    
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    def __init__(self, root_dir: str, size: int = 256, max_hints: int = 20):
        """
        Initialize dataset.
        
        Args:
            root_dir: Path to image directory
            size: Target image size (will be resized to size x size)
            max_hints: Maximum number of color hints per image
        """
        self.root_dir = root_dir
        self.size = size
        self.max_hints = max_hints
        
        # Validate parameters
        if not os.path.isdir(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")
        
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        
        if max_hints < 0:
            raise ValueError(f"max_hints must be non-negative, got {max_hints}")
        
        # Find all valid image files
        self.files = self._find_image_files()
        
        if len(self.files) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
        
        logger.info(f"Found {len(self.files)} images in {root_dir}")
    
    def _find_image_files(self) -> List[str]:
        """Find all valid image files in the root directory."""
        files = []
        
        try:
            for filename in os.listdir(self.root_dir):
                if filename.lower().endswith(self.VALID_EXTENSIONS):
                    filepath = os.path.join(self.root_dir, filename)
                    if os.path.isfile(filepath):
                        files.append(filepath)
        except Exception as e:
            logger.error(f"Error scanning directory {self.root_dir}: {e}")
            raise
        
        return sorted(files)
    
    def _load_and_preprocess_image(self, filepath: str) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            # Open and convert image
            with Image.open(filepath) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img_resized = img.resize((self.size, self.size), Image.BICUBIC)
                
                # Convert to numpy array
                rgb_array = np.array(img_resized, dtype=np.uint8)
                
                return rgb_array
                
        except Exception as e:
            logger.error(f"Error loading image {filepath}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            x: Model input tensor (4,H,W)
            y: Target AB channels (2,H,W)
            rgb: Original RGB image for visualization
        """
        try:
            if idx < 0 or idx >= len(self.files):
                raise IndexError(f"Index {idx} out of range [0, {len(self.files)})")
            
            # Load and preprocess image
            filepath = self.files[idx]
            rgb = self._load_and_preprocess_image(filepath)
            
            # Convert to LAB color space
            L, AB = rgb_to_lab_norm(rgb)
            
            # Generate random hints
            num_hints = random.randint(0, self.max_hints)
            hint_mask, hint_ab = create_random_hints(AB, k=num_hints, sigma=7)
            
            # Build model input
            x = build_model_input(L, hint_mask, hint_ab)
            
            # Target AB channels as tensor
            y = torch.from_numpy(AB.transpose(2, 0, 1)).float()
            
            return x, y, rgb
            
        except Exception as e:
            logger.error(f"Error getting item {idx} from dataset: {e}")
            # Return a dummy sample to prevent training crashes
            dummy_x = torch.zeros(4, self.size, self.size)
            dummy_y = torch.zeros(2, self.size, self.size)
            dummy_rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            return dummy_x, dummy_y, dummy_rgb