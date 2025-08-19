# Fixed src/train.py - Update imports to absolute
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torchvision.utils as vutils
from src.dataset import ColorizeDataset
from src.model import UNet
from src.losses import ColorL1Loss
from src.utils import lab_norm_to_rgb_uint8, save_rgb, ensure_dir

def save_sample_grid(L, pred_ab, gt_ab, out_path):
    """
    L: (B,1,H,W) in [0,1]
    pred_ab, gt_ab: (B,2,H,W) in [-1,1]
    Save a small grid for quick visual inspection.
    """
    L_np = L.detach().cpu().numpy().transpose(0,2,3,1)
    P_np = pred_ab.detach().cpu().numpy().transpose(0,2,3,1)
    G_np = gt_ab.detach().cpu().numpy().transpose(0,2,3,1)
    imgs = []
    for i in range(min(4, L_np.shape[0])):
        rgb_pred = lab_norm_to_rgb_uint8(L_np[i], P_np[i])
        rgb_gt   = lab_norm_to_rgb_uint8(L_np[i], G_np[i])
        # stack pred | gt horizontally
        pair = np.concatenate([rgb_pred, rgb_gt], axis=1)
        imgs.append(pair)
    grid = np.concatenate(imgs, axis=0)
    save_rgb(out_path, grid)

def train(data_root, val_root, epochs, batch_size, lr, device, size, base, save_every):
    ensure_dir("checkpoints")
    ds = ColorizeDataset(data_root, size=size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_ds = ColorizeDataset(val_root, size=size) if val_root and os.path.isdir(val_root) else None
    if val_ds:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = UNet(in_channels=4, out_channels=2, base=base).to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)
    criterion = ColorL1Loss()

    global_step = 0
    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}")
        running = 0.0
        for (inp, target, _rgb) in pbar:
            inp = inp.to(device)             # (B,4,H,W)
            target = target.to(device)       # (B,2,H,W)
            optimz.zero_grad()
            pred = model(inp)                # (B,2,H,W)
            loss = criterion(pred, target)
            loss.backward()
            optimz.step()
            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=running / (pbar.n + 1))

        # save checkpoint
        ckpt_path = f"checkpoints/ckpt_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimz.state_dict(),
            "size": size,
            "base": base,
        }, ckpt_path)

        # save a small sample grid from the last batch
        try:
            save_sample_grid(inp[:, :1, ...], pred, target, f"checkpoints/samples_epoch_{epoch}.png")
        except Exception as e:
            print("Sample save failed:", str(e))

        # simple val loop (optional)
        if val_ds:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for (vinp, vtarget, _vrgb) in val_dl:
                    vinp = vinp.to(device)
                    vtarget = vtarget.to(device)
                    vpred = model(vinp)
                    vloss = criterion(vpred, vtarget)
                    val_running += vloss.item()
            print(f"[VAL] epoch {epoch} loss: {val_running / len(val_dl):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train")
    parser.add_argument("--val", type=str, default="data/val")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--base", type=int, default=64)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.data, args.val, args.epochs, args.bs, args.lr, device, args.size, args.base, save_every=1)