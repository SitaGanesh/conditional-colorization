# --- path bootstrap (do not remove) ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import uvicorn, io
import numpy as np
from PIL import Image
import torch

from src.model import UNet
from src.utils import rgb_to_lab_norm, lab_norm_to_rgb_uint8, hex_to_ab, build_model_input

app = FastAPI(title="Conditional Colorizer API")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Defaults
size = 256
base = 64
ckpt_path = None

# Find latest checkpoint
ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
if os.path.isdir(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest_ckpt = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

# Load checkpoint dict before model creation
if ckpt_path:
    ck = torch.load(ckpt_path, map_location=device)
    base = ck.get("base", base)
    size = ck.get("size", size)

# Instantiate model with correct base
model = UNet(in_channels=4, out_channels=2, base=base).to(device)

# Load its weights
if ckpt_path:
    model.load_state_dict(ck["model_state_dict"])

model.eval()

@app.get("/health")
def health():
    return {"ok": True, "device": device, "size": size}

@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    color_hex: str = Form("#00aaff"),
    mask: Optional[UploadFile] = File(None)
):
    # Read and preprocess input image
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((size, size))
    rgb = np.array(img, dtype=np.uint8)
    L, _ = rgb_to_lab_norm(rgb)

    # Optional mask
    if mask is not None:
        m_bytes = await mask.read()
        mimg = Image.open(io.BytesIO(m_bytes)).convert("L").resize((size, size))
        mask_np = np.array(mimg, dtype=np.float32) / 255.0
        hint_mask = mask_np[..., None]
        ab = hex_to_ab(color_hex)
        hint_ab = np.tile(ab.reshape(1,1,2), (size, size, 1)).astype(np.float32) * hint_mask
    else:
        hint_mask = np.zeros((size, size, 1), dtype=np.float32)
        hint_ab = np.zeros((size, size, 2), dtype=np.float32)

    x = build_model_input(L, hint_mask, hint_ab).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ab = model(x)[0].cpu().numpy().transpose(1,2,0)

    rgb_pred = lab_norm_to_rgb_uint8(L, pred_ab)
    out_img = Image.fromarray(rgb_pred)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("app.api_main:app", host="0.0.0.0", port=8000, reload=True)
