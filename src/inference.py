# COMPLETELY FIXED src/inference.py
import argparse
import numpy as np
from PIL import Image
import torch
import cv2
import os
from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input, ensure_dir

def load_model(ckpt_path, device="cpu"):
    # 1) Load checkpoint dict first
    ck = torch.load(ckpt_path, map_location=device)
    # 2) Read `base` and `size` from checkpoint (defaults as fallback)
    base = ck.get("base", 64)
    size = ck.get("size", 256)
    # 3) Instantiate model using the saved `base`
    model = UNet(in_channels=4, out_channels=2, base=base).to(device)
    # 4) Load its state dict
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, size



def run_inference(img_path, ckpt, color_hex=None, mask_path=None, out_path="out.png", device="cpu"):
    model, size = load_model(ckpt, device=device)
    img = Image.open(img_path).convert("RGB").resize((size, size))
    rgb = np.array(img, dtype=np.uint8)
    L, _AB = rgb_to_lab_norm(rgb)
    
    # hints
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L").resize((size, size))
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        hint_mask = mask_np[..., None]  # Add channel dimension (H,W,1)
        if color_hex:
            ab = hex_to_ab(color_hex)
            hint_ab = np.tile(ab.reshape(1,1,2), (size, size, 1)).astype(np.float32) * hint_mask
        else:
            hint_ab = np.zeros((size, size, 2), dtype=np.float32)
    else:
        hint_mask = np.zeros((size, size, 1), dtype=np.float32)
        hint_ab = np.zeros((size, size, 2), dtype=np.float32)
    
    x = build_model_input(L, hint_mask, hint_ab).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ab = model(x)[0].cpu().numpy().transpose(1,2,0)
    
    rgb_pred = lab_norm_to_rgb_uint8(L, pred_ab)
    ensure_dir(os.path.dirname(out_path) if os.path.dirname(out_path) else ".")
    Image.fromarray(rgb_pred).save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to input RGB image")
    ap.add_argument("--ckpt", default="checkpoints/ckpt_epoch_1.pth")
    ap.add_argument("--mask", default=None, help="optional mask image (white = apply color)")
    ap.add_argument("--color", default=None, help="hex color like #00aaff")
    ap.add_argument("--out", default="out.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_inference(args.img, args.ckpt, args.color, args.mask, args.out, device)