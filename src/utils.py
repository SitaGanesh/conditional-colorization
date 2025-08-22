# src/utils.py
import numpy as np
import cv2
import torch
import os
import logging
from typing import Tuple, Optional
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image_array(img: np.ndarray, expected_dtype: type = np.uint8, expected_channels: Optional[int] = None) -> bool:
    """Validate image array format and properties."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")
    if img.dtype != expected_dtype:
        raise ValueError(f"Expected dtype {expected_dtype}, got {img.dtype}")
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array (H,W,C), got {img.ndim}D")
    if expected_channels is not None and img.shape[2] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {img.shape[asset:1]}")
    return True


def rgb_to_lab_norm(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB to normalized LAB color space.
    
    Args:
        rgb_uint8: HxWx3 uint8 RGB array
        
    Returns:
        L: Lightness channel in [0,1] shape (H,W,1)
        AB: Color channels in [-1,1] shape (H,W,2)
    """
    try:
        validate_image_array(rgb_uint8, np.uint8, 3)
        
        # Convert to LAB
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Normalize L channel to [0,1]
        L = lab[..., 0:1] / 100.0
        L = np.clip(L, 0, 1)  # Ensure valid range
        
        # Normalize AB channels to [-1,1]
        AB = (lab[..., 1:3] - 128.0) / 127.0
        AB = np.clip(AB, -1, 1)  # Ensure valid range
        
        return L, AB
        
    except Exception as e:
        logger.error(f"Error in rgb_to_lab_norm: {e}")
        raise

def lab_norm_to_rgb_uint8(L: np.ndarray, AB: np.ndarray) -> np.ndarray:
    """
    Convert normalized LAB to RGB uint8.
    
    Args:
        L: Lightness in [0,1] shape (H,W,1) or (H,W)
        AB: Color channels in [-1,1] shape (H,W,2)
        
    Returns:
        RGB uint8 array shape (H,W,3)
    """
    try:
        # Handle L channel dimensions
        if L.ndim == 2:
            L = L[..., np.newaxis]
        
        if L.shape[-1] != 1:
            raise ValueError(f"L channel should have 1 dimension, got {L.shape[-1]}")
        
        if AB.shape[-1] != 2:
            raise ValueError(f"AB channels should have 2 dimensions, got {AB.shape[-1]}")
        
        # Ensure same spatial dimensions
        if L.shape[:2] != AB.shape[:2]:
            raise ValueError(f"L and AB spatial dimensions don't match: {L.shape[:2]} vs {AB.shape[:2]}")
        
        # Denormalize and clip
        L_denorm = np.clip(L * 100.0, 0, 100).astype(np.uint8)
        AB_denorm = np.clip(AB * 127.0 + 128.0, 0, 255).astype(np.uint8)
        
        # Concatenate LAB channels
        lab_uint8 = np.concatenate([L_denorm, AB_denorm], axis=2)
        
        # Convert to RGB
        rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
        
        return rgb
        
    except Exception as e:
        logger.error(f"Error in lab_norm_to_rgb_uint8: {e}")
        raise

def hex_to_ab(hex_color: str) -> np.ndarray:
    """
    Convert hex color to AB channels in LAB space.
    
    Args:
        hex_color: Hex color string like '#RRGGBB'
        
    Returns:
        AB values as (2,) array in [-1,1] range
    """
    try:
        # Clean and validate hex string
        hex_clean = hex_color.lstrip('#')
        if len(hex_clean) != 6:
            raise ValueError(f"Invalid hex color format: {hex_color}")
        
        # Parse RGB values
        try:
            r = int(hex_clean[0:2], 16)
            g = int(hex_clean[2:4], 16)
            b = int(hex_clean[4:6], 16)
        except ValueError as e:
            raise ValueError(f"Invalid hex color values: {hex_color}")
        
        # Convert to LAB
        rgb_array = np.uint8([[[r, g, b]]])
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Extract and normalize AB
        ab = (lab[0, 0, 1:3] - 128.0) / 127.0
        ab = np.clip(ab, -1, 1)
        
        return ab.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in hex_to_ab with color {hex_color}: {e}")
        raise

def create_random_hints(AB: np.ndarray, k: int = 20, sigma: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random color hints as soft blobs.
    
    Args:
        AB: Color channels (H,W,2) in [-1,1]
        k: Number of hint points
        sigma: Gaussian blur kernel size
        
    Returns:
        hint_mask: (H,W,1) hint mask
        hint_ab: (H,W,2) hint colors
    """
    try:
        if AB.shape[-1] != 2:
            raise ValueError(f"AB should have 2 channels, got {AB.shape[-1]}")
        
        H, W, _ = AB.shape
        hint_mask = np.zeros((H, W, 1), dtype=np.float32)
        hint_ab = np.zeros((H, W, 2), dtype=np.float32)
        
        if k <= 0:
            return hint_mask, hint_ab
        
        # Generate random hint locations
        k = min(k, H * W)  # Don't exceed image size
        ys = np.random.randint(0, H, size=(k,))
        xs = np.random.randint(0, W, size=(k,))
        
        # Place hints
        for y, x in zip(ys, xs):
            hint_mask[y, x, 0] = 1.0
            hint_ab[y, x, :] = AB[y, x, :]
        
        # Apply Gaussian blur if sigma > 0
        if sigma > 0:
            # Ensure odd kernel size
            if sigma % 2 == 0:
                sigma += 1
            
            # Blur hint mask
            hint_mask_2d = hint_mask[:, :, 0]
            hint_mask_blurred = cv2.GaussianBlur(hint_mask_2d, (sigma, sigma), 0)
            hint_mask = hint_mask_blurred[:, :, np.newaxis]
            
            # Blur hint colors
            hint_ab[:, :, 0] = cv2.GaussianBlur(hint_ab[:, :, 0], (sigma, sigma), 0)
            hint_ab[:, :, 1] = cv2.GaussianBlur(hint_ab[:, :, 1], (sigma, sigma), 0)
        
        return hint_mask, hint_ab
        
    except Exception as e:
        logger.error(f"Error in create_random_hints: {e}")
        raise

def build_model_input(L: np.ndarray, hint_mask: np.ndarray, hint_ab: np.ndarray) -> torch.Tensor:
    """
    Build model input by concatenating L, hint_mask, and hint_ab.
    
    Args:
        L: Lightness (H,W,1)
        hint_mask: Hint mask (H,W,1)
        hint_ab: Hint colors (H,W,2)
        
    Returns:
        Model input tensor (4,H,W)
    """
    try:
        # Validate shapes
        if L.shape[-1] != 1:
            raise ValueError(f"L should have shape (H,W,1), got {L.shape}")
        if hint_mask.shape[-1] != 1:
            raise ValueError(f"hint_mask should have shape (H,W,1), got {hint_mask.shape}")
        if hint_ab.shape[-1] != 2:
            raise ValueError(f"hint_ab should have shape (H,W,2), got {hint_ab.shape}")
        
        # Check spatial dimensions match
        if not (L.shape[:2] == hint_mask.shape[:2] == hint_ab.shape[:2]):
            raise ValueError("Spatial dimensions of L, hint_mask, and hint_ab must match")
        
        # Concatenate channels
        x = np.concatenate([L, hint_mask, hint_ab], axis=2)  # (H,W,4)
        
        # Convert to tensor and transpose to CHW format
        x = x.transpose(2, 0, 1).astype(np.float32)  # (4,H,W)
        
        return torch.from_numpy(x)
        
    except Exception as e:
        logger.error(f"Error in build_model_input: {e}")
        raise

def compute_metrics(rgb_pred_uint8: np.ndarray, rgb_gt_uint8: np.ndarray) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM metrics.
    
    Args:
        rgb_pred_uint8: Predicted RGB image
        rgb_gt_uint8: Ground truth RGB image
        
    Returns:
        PSNR and SSIM values
    """
    try:
        validate_image_array(rgb_pred_uint8, np.uint8, 3)
        validate_image_array(rgb_gt_uint8, np.uint8, 3)
        
        if rgb_pred_uint8.shape != rgb_gt_uint8.shape:
            raise ValueError("Predicted and ground truth images must have same shape")
        
        psnr = psnr_metric(rgb_gt_uint8, rgb_pred_uint8, data_range=255)
        ssim = ssim_metric(rgb_gt_uint8, rgb_pred_uint8, channel_axis=2, data_range=255)
        
        return psnr, ssim
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise

def save_rgb(path: str, rgb_uint8: np.ndarray) -> None:
    """
    Save RGB image to file.
    
    Args:
        path: Output file path
        rgb_uint8: RGB image array
    """
    try:
        validate_image_array(rgb_uint8, np.uint8, 3)
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(path))
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        
        # Save image
        success = cv2.imwrite(path, bgr)
        if not success:
            raise RuntimeError(f"Failed to save image to {path}")
        
        logger.info(f"Image saved to {path}")
        
    except Exception as e:
        logger.error(f"Error saving image to {path}: {e}")
        raise

def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    """
    try:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise