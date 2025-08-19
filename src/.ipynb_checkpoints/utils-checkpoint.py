# COMPLETELY FIXED src/utils.py
import numpy as np
import cv2
import torch
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

# Color space helpers
def rgb_to_lab_norm(rgb_uint8: np.ndarray):
    """
    rgb_uint8: HxWx3 uint8 RGB
    Returns:
      L in [0,1] shape (H,W,1)
      AB in [-1,1] shape (H,W,2)
    """
    assert rgb_uint8.dtype == np.uint8 and rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0:1] / 100.0  # Keep as (H,W,1)
    AB = (lab[..., 1:3] - 128.0) / 127.0  # Keep as (H,W,2)
    return L, AB

def lab_norm_to_rgb_uint8(L: np.ndarray, AB: np.ndarray):
    """
    L in [0,1], AB in [-1,1] → returns RGB uint8 HxWx3
    """
    L_u = (np.clip(L, 0, 1) * 100.0).astype(np.uint8)
    AB_u = (np.clip(AB, -1, 1) * 127.0 + 128.0).astype(np.uint8)
    lab_u = np.concatenate([L_u, AB_u], axis=2)
    rgb = cv2.cvtColor(lab_u, cv2.COLOR_LAB2RGB)
    return rgb

def hex_to_ab(hex_color: str) -> np.ndarray:
    """
    '#RRGGBB' -> ab (2,) in [-1,1]
    """
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    rgb = np.uint8([[[r, g, b]]])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ab = (lab[0, 0, 1:3] - 128.0) / 127.0
    return ab.astype(np.float32)

# Hint helpers
def create_random_hints(AB: np.ndarray, k: int = 20, sigma: int = 7):
    """
    AB: (H,W,2) in [-1,1]
    Returns hint_mask (H,W,1), hint_ab (H,W,2) as soft blobs from k random points.
    """
    H, W, _ = AB.shape
    hint_mask = np.zeros((H, W, 1), dtype=np.float32)
    hint_ab = np.zeros((H, W, 2), dtype=np.float32)
    
    if k <= 0:
        return hint_mask, hint_ab
    
    ys = np.random.randint(0, H, size=(k,))
    xs = np.random.randint(0, W, size=(k,))
    
    for y, x in zip(ys, xs):
        hint_mask[y, x, 0] = 1.0
        hint_ab[y, x, :] = AB[y, x, :]
    
    # Gaussian blur to make soft blobs - FIX: handle dimensions properly
    if sigma % 2 == 0:
        sigma += 1  # kernel size must be odd
    
    # For hint_mask (H,W,1) - squeeze to 2D, blur, then add dimension back
    hint_mask_2d = hint_mask[:, :, 0]  # (H,W)
    hint_mask_blurred = cv2.GaussianBlur(hint_mask_2d, (sigma, sigma), 0)
    hint_mask = hint_mask_blurred[:, :, np.newaxis]  # Back to (H,W,1)
    
    # For hint_ab (H,W,2) - blur each channel separately
    hint_ab[:, :, 0] = cv2.GaussianBlur(hint_ab[:, :, 0], (sigma, sigma), 0)
    hint_ab[:, :, 1] = cv2.GaussianBlur(hint_ab[:, :, 1], (sigma, sigma), 0)
    
    return hint_mask, hint_ab

def build_model_input(L: np.ndarray, hint_mask: np.ndarray, hint_ab: np.ndarray):
    """
    Stack to shape (H,W,4) then return CHW float32 tensor (4,H,W)
    L: (H,W,1), hint_mask: (H,W,1), hint_ab: (H,W,2)
    """
    # Ensure all inputs have 3 dimensions and correct shapes
    assert L.shape[-1] == 1, f"L should have shape (H,W,1), got {L.shape}"
    assert hint_mask.shape[-1] == 1, f"hint_mask should have shape (H,W,1), got {hint_mask.shape}"
    assert hint_ab.shape[-1] == 2, f"hint_ab should have shape (H,W,2), got {hint_ab.shape}"
    
    x = np.concatenate([L, hint_mask, hint_ab], axis=2)  # (H,W,4)
    x = x.transpose(2, 0, 1).astype(np.float32)  # (4,H,W)
    return torch.from_numpy(x)

# Metrics & IO
def compute_metrics(rgb_pred_uint8: np.ndarray, rgb_gt_uint8: np.ndarray):
    """Return PSNR and SSIM for convenience (Y-channel SSIM could be better; here use RGB)."""
    psnr = psnr_metric(rgb_gt_uint8, rgb_pred_uint8, data_range=255)
    ssim = ssim_metric(rgb_gt_uint8, rgb_pred_uint8, channel_axis=2)
    return psnr, ssim

def save_rgb(path: str, rgb_uint8: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))  # save as BGR for cv2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)