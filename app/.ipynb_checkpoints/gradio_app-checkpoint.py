# --- path bootstrap (do not remove) ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

import gradio as gr
import numpy as np
from PIL import Image
import torch
from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input

device = "cuda" if torch.cuda.is_available() else "cpu"

# Default values
size = 256
base = 64
ckpt_path = None

# Find latest checkpoint
ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
if os.path.isdir(ckpt_dir):
    ckpts = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("ckpt_epoch_") and f.endswith(".pth")
    ]
    if ckpts:
        ckpt_name = sorted(
            ckpts,
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

# Load checkpoint before model creation
if ckpt_path:
    ck = torch.load(ckpt_path, map_location=device)
    base = ck.get("base", base)
    size = ck.get("size", size)

# Instantiate model with correct base
model = UNet(in_channels=4, out_channels=2, base=base).to(device)

# Load weights if checkpoint found
if ckpt_path:
    model.load_state_dict(ck["model_state_dict"])

model.eval()

def predict(img_pil, mask_pil, color_hex):
    # accept RGB or grayscale
    img = img_pil.convert("RGB").resize((size, size))
    rgb = np.array(img, dtype=np.uint8)
    L, _ = rgb_to_lab_norm(rgb)

    if mask_pil is not None:
        mask = mask_pil.convert("L").resize((size, size))
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        hint_mask = mask_np[..., None]
        ab = hex_to_ab(color_hex)
        hint_ab = (np.tile(ab.reshape(1,1,2), (size, size, 1)).astype(np.float32)
                   * hint_mask)
    else:
        hint_mask = np.zeros((size, size, 1), dtype=np.float32)
        hint_ab = np.zeros((size, size, 2), dtype=np.float32)

    x = build_model_input(L, hint_mask, hint_ab).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ab = model(x)[0].cpu().numpy().transpose(1,2,0)

    rgb_pred = lab_norm_to_rgb_uint8(L, pred_ab)
    return Image.fromarray(rgb_pred)

with gr.Blocks(title="Conditional Colorizer") as demo:
    gr.Markdown(
        "## 🎨 Conditional Image Colorizer\n"
        "Upload a grayscale or color image, paint a mask over a region, pick a color, and colorize it."
    )
    with gr.Row():
        inp_img = gr.Image(label="Input (gray or color)", type="pil")
        mask    = gr.Image(label="Mask (paint white where to colorize)", type="pil")
    color = gr.ColorPicker(label="Color", value="#00aaff")
    out = gr.Image(label="Output")
    btn = gr.Button("Colorize")
    btn.click(fn=predict, inputs=[inp_img, mask, color], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
