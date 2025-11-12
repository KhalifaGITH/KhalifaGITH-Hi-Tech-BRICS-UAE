
#!/usr/bin/env python3
"""
Drone AI Task 2: Single-File End-to-End Pipeline
------------------------------------------------
What this script does:
1) PREP  : Tile a large GeoTIFF and rasterize GeoJSON labels onto tiles.
2) TRAIN : Train a compact U-Net on the tiles and save weights.
3) INFER : Run sliding-window inference on the full orthomosaic.
4) EVAL  : (Optional) Compute per-class IoU on tiles.
5) VIZ   : Save quicklook visualizations (random tiles and full-mosaic preview) and training curves.

No UI required. Tested with Python 3.10+.
"""

import os
import sys
import math
import json
import yaml
import random
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import rasterio
from rasterio.windows import Window
from rasterio import features

import geopandas as gpd
from shapely.geometry import shape

# --------------------
# Defaults & Config
# --------------------

DEFAULT_CLASSMAP = {
    "mapping": {
        "cropland": 1,
        "road": 2,
        "forest_belt": 3,
        "water": 4
    },
    "palette": {
        0: [0, 0, 0],
        1: [200, 200, 0],
        2: [160, 160, 160],
        3: [0, 120, 0],
        4: [0, 0, 200]
    }
}

@dataclass
class Config:
    # Data
    raw_tif: str = "data/raw/ortho.tif"
    raw_geojson: str = "data/raw/ortho.geojson"
    workdir: str = "data/working"

    # Tiling
    tile_size: int = 1024
    tile_stride: int = 1024

    # Training
    in_channels: int = 3
    num_classes: int = 5  # background + 4 classes
    base_channels: int = 32
    epochs: int = 25
    batch_size: int = 2
    learning_rate: float = 3e-4
    val_split: float = 0.2
    num_workers: int = 2
    random_seed: int = 1337
    augment: bool = True

    # Inference
    sliding_window: int = 2048
    overlap: int = 256
    soft_blend: bool = True
    weights_path: str = "data/working/model_final.pt"
    pred_geotiff: str = "data/working/prediction.tif"
    pred_geojson: str = "data/working/prediction_polygons.geojson"

    # Visualization / logs
    out_plots_dir: str = "data/working/plots"
    history_path: str = "data/working/train_history.json"

def ensure_dirs(cfg: Config):
    os.makedirs(cfg.workdir, exist_ok=True)
    os.makedirs(os.path.join(cfg.workdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(cfg.workdir, "masks"), exist_ok=True)
    os.makedirs(cfg.out_plots_dir, exist_ok=True)

# --------------------
# Raster & Vector utils
# --------------------

def iter_windows(width: int, height: int, size: int, stride: Optional[int] = None):
    if stride is None:
        stride = size
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            yield Window(x, y, size, size), (x, y)

def load_training_polygons(geojson_path: str, classmap: Dict[str, int]):
    gdf = gpd.read_file(geojson_path)
    if 'class' not in gdf.columns:
        raise ValueError("GeoJSON must contain a 'class' field for labels.")
    gdf = gdf[gdf['class'].isin(classmap)].copy()
    gdf['class_id'] = gdf['class'].map(classmap)
    return gdf

def rasterize_labels(gdf, transform, out_shape, crs, default_bg=0):
    if gdf.crs is None:
        raise ValueError("GeoJSON lacks CRS. Please define it before training.")
    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    shapes = [(geom, v) for geom, v in zip(gdf.geometry, gdf['class_id'])]
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=default_bg,
        all_touched=True,
        dtype=np.uint8
    )
    return mask

def gaussian_weight(h: int, w: int):
    y = np.linspace(-1, 1, h); x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    w2 = np.exp(-(xx**2 + yy**2))
    return w2.astype('float32')

# --------------------
# Model (U-Net)
# --------------------

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=5, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bott = conv_block(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = conv_block(base*2, base)

        self.outc = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b = self.bott(p4)

        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)

        return self.outc(d1)

# --------------------
# Dataset
# --------------------

class NPZDataset(Dataset):
    def __init__(self, img_dir, msk_dir, augment=False):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.npy')])
        self.msk_paths = [p.replace('images', 'masks') for p in self.img_paths]
        self.augment = augment

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32) / 255.0
        msk = np.load(self.msk_paths[idx]).astype(np.int64)
        if self.augment and img.shape[-1] > 0:
            if random.random() < 0.5:
                img = img[..., ::-1].copy()
                msk = msk[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[..., ::-1, :].transpose(0, 2, 1).copy()
                msk = msk.T.copy()
        return torch.from_numpy(img), torch.from_numpy(msk)

# --------------------
# Step A: PREP (tiling + rasterize labels)
# --------------------

def step_prep(cfg: Config, classmap: Dict[str, int]):
    ensure_dirs(cfg)
    img_dir = os.path.join(cfg.workdir, "images")
    msk_dir = os.path.join(cfg.workdir, "masks")

    gdf = load_training_polygons(cfg.raw_geojson, classmap)

    with rasterio.open(cfg.raw_tif) as ds:
        W, H = ds.width, ds.height
        for win, (x, y) in list(iter_windows(W, H, cfg.tile_size, cfg.tile_stride)):
            arr = ds.read((1,2,3), window=win)
            if arr.mean() < 1:
                continue
            t_win = rasterio.windows.transform(win, ds.transform)
            mask = rasterize_labels(gdf, t_win, (cfg.tile_size, cfg.tile_size), ds.crs)
            if mask.sum() == 0:
                continue
            np.save(os.path.join(img_dir, f"{y}_{x}.npy"), arr.astype('uint8'))
            np.save(os.path.join(msk_dir, f"{y}_{x}.npy"), mask.astype('uint8'))
    print("PREP: tiles saved to", img_dir, "and", msk_dir)

# --------------------
# Step B: TRAIN
# --------------------

def step_train(cfg: Config) -> Dict[str, List[float]]:
    ensure_dirs(cfg)
    img_dir = os.path.join(cfg.workdir, "images")
    msk_dir = os.path.join(cfg.workdir, "masks")

    ds = NPZDataset(img_dir, msk_dir, augment=cfg.augment)
    if len(ds) == 0:
        raise RuntimeError("No training tiles were found. Run PREP first or check your label coverage.")

    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.random_seed))

    loader_tr = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    loader_va = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    history = {"loss": [], "val_mIoU": []}
    best = 0.0

    for ep in range(cfg.epochs):
        model.train()
        run_loss = 0.0
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item())
        history["loss"].append(run_loss / max(1, len(loader_tr)))

        # Validation IoU
        model.eval()
        inter = torch.zeros(cfg.num_classes, device=device)
        union = torch.zeros(cfg.num_classes, device=device)
        with torch.no_grad():
            for x, y in loader_va:
                x, y = x.to(device), y.to(device)
                pr = model(x).argmax(1)
                for c in range(cfg.num_classes):
                    inter[c] += ((pr == c) & (y == c)).sum()
                    union[c] += ((pr == c) | (y == c)).sum()
        miou = float((inter / (union + 1e-6)).mean().item())
        history["val_mIoU"].append(miou)
        print(f"Epoch {ep+1}/{cfg.epochs} - loss: {history['loss'][-1]:.4f} - val mIoU: {miou:.4f}")
        if miou > best:
            best = miou
            os.makedirs(os.path.dirname(cfg.weights_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.weights_path)

    # Save history for plots
    with open(cfg.history_path, "w") as f:
        json.dump(history, f)
    print("TRAIN: done. Best mIoU:", best, "Weights:", cfg.weights_path)
    return history

# --------------------
# Step C: INFER (full mosaic, sliding window)
# --------------------

@torch.no_grad()
def step_infer(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels).to(device)
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    win = cfg.sliding_window
    overlap = cfg.overlap
    step = win - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than sliding_window.")

    with rasterio.open(cfg.raw_tif) as ds:
        W, H = ds.width, ds.height
        profile = ds.profile.copy(); profile.update({"count": 1, "dtype": "uint8"})
        transform = ds.transform; crs = ds.crs

        prob_acc = np.zeros((cfg.num_classes, H, W), dtype="float32")
        w_acc = np.zeros((H, W), dtype="float32")
        wmask = gaussian_weight(win, win) if cfg.soft_blend else np.ones((win, win), dtype="float32")

        ys = list(range(0, max(1, H - win + 1), step))
        xs = list(range(0, max(1, W - win + 1), step))
        if ys[-1] != H - win: ys.append(H - win)
        if xs[-1] != W - win: xs.append(W - win)

        for y in ys:
            for x in xs:
                window = Window(x, y, win, win)
                patch = ds.read((1,2,3), window=window).astype("float32") / 255.0
                xt = torch.from_numpy(patch[None, ...]).to(device)
                probs = torch.softmax(model(xt), dim=1)[0].cpu().numpy()
                for c in range(cfg.num_classes):
                    prob_acc[c, y:y+win, x:x+win] += probs[c] * wmask
                w_acc[y:y+win, x:x+win] += wmask

        prob_acc /= (w_acc[None, ...] + 1e-6)
        pred = prob_acc.argmax(0).astype("uint8")

        with rasterio.open(cfg.pred_geotiff, "w", **profile) as dst:
            dst.write(pred[None, ...])
            dst.transform = transform

        # Optional polygonization to GeoJSON
        if cfg.pred_geojson:
            geoms = []
            for val in np.unique(pred):
                if val == 0:  # skip background to reduce size
                    continue
                for geom, v in features.shapes(pred, mask=(pred == val), transform=transform):
                    geoms.append((shape(geom), int(val)))
            if geoms:
                gdf = gpd.GeoDataFrame(
                    {"class_id": [v for _, v in geoms],
                     "geometry": [g for g, _ in geoms]},
                    crs=crs
                )
                # Add readable class names if possible
                if os.path.exists("classmap.yaml"):
                    m = yaml.safe_load(open("classmap.yaml", "r"))["mapping"]
                    inv = {v: k for k, v in m.items()}
                    gdf["class"] = gdf["class_id"].map(inv)
                gdf.to_file(cfg.pred_geojson, driver="GeoJSON")

        print("INFER: saved", cfg.pred_geotiff, "and", cfg.pred_geojson)

# --------------------
# Step D: EVAL (optional, on all tiles you have)

# --------------------

@torch.no_grad()
def step_eval(cfg: Config):
    img_dir = os.path.join(cfg.workdir, "images")
    msk_dir = os.path.join(cfg.workdir, "masks")
    ds = NPZDataset(img_dir, msk_dir, augment=False)
    if len(ds) == 0:
        raise RuntimeError("No tiles found for evaluation.")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(cfg.in_channels, cfg.num_classes, cfg.base_channels).to(device)
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    inter = torch.zeros(cfg.num_classes, device=device)
    union = torch.zeros(cfg.num_classes, device=device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pr = model(x).argmax(1)
        for c in range(cfg.num_classes):
            inter[c] += ((pr == c) & (y == c)).sum()
            union[c] += ((pr == c) | (y == c)).sum()

    iou = (inter / (union + 1e-6)).cpu().numpy()
    miou = float(iou.mean())
    print("EVAL: per-class IoU:", iou, "mIoU:", miou)
    return iou, miou

# --------------------
# Step E: VIZ (plots & quicklooks)
# --------------------

def step_viz(cfg: Config, classmap: Dict[int, List[int]]):
    os.makedirs(cfg.out_plots_dir, exist_ok=True)

    # 1) Training curves
    if os.path.exists(cfg.history_path):
        hist = json.load(open(cfg.history_path, "r"))
        plt.figure()
        plt.plot(hist.get("loss", []), label="train loss")
        plt.plot(hist.get("val_mIoU", []), label="val mIoU")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training History")
        out_path = os.path.join(cfg.out_plots_dir, "training_history.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print("VIZ: saved", out_path)

    # 2) Random training tile preview (RGB, mask, overlay)
    img_dir = os.path.join(cfg.workdir, "images")
    if os.path.isdir(img_dir):
        imgs = [f for f in os.listdir(img_dir) if f.endswith(".npy")]
        if imgs:
            choice = random.choice(imgs)
            img = np.load(os.path.join(img_dir, choice))
            msk = np.load(os.path.join(cfg.workdir, "masks", choice))
            # Normalize for display
            rgb = np.transpose(img, (1, 2, 0)).astype(np.uint8)

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(rgb); plt.title("Tile RGB"); plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(msk); plt.title("Label mask"); plt.axis("off")
            # overlay (simple alpha)
            overlay = rgb.copy()
            plt.subplot(1,3,3); plt.imshow(rgb); plt.imshow(msk, alpha=0.4); plt.title("Overlay"); plt.axis("off")
            out_path = os.path.join(cfg.out_plots_dir, "tile_preview.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print("VIZ: saved", out_path)

    # 3) Full-mosaic quicklook of prediction (downsampled)
    if os.path.exists(cfg.pred_geotiff):
        with rasterio.open(cfg.pred_geotiff) as ds:
            pred = ds.read(1)
            # coarse preview to keep memory small
            factor = max(1, int(max(pred.shape) / 2048))
            pred_small = pred[::factor, ::factor]
            plt.figure()
            plt.imshow(pred_small)
            plt.title("Prediction (downsampled quicklook)")
            plt.axis("off")
            out_path = os.path.join(cfg.out_plots_dir, "prediction_quicklook.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print("VIZ: saved", out_path)

# --------------------
# Helper: load classmap (from classmap.yaml if present)
# --------------------

def load_classmap() -> Dict[str, int]:
    if os.path.exists("classmap.yaml"):
        cm = yaml.safe_load(open("classmap.yaml", "r"))
        return cm.get("mapping", DEFAULT_CLASSMAP["mapping"])
    return DEFAULT_CLASSMAP["mapping"]

# --------------------
# CLI
# --------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drone AI Task 2 | Single-file pipeline")
    parser.add_argument("--config", type=str, default=None, help="Optional path to a YAML config file.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prep", help="Tile GeoTIFF and rasterize GeoJSON labels.")
    sub.add_parser("train", help="Train the U-Net model on tiles.")
    sub.add_parser("infer", help="Run sliding-window inference on full mosaic.")
    sub.add_parser("eval", help="Evaluate IoU on available tiles.")
    sub.add_parser("viz", help="Create training curves and quicklook images.")

    args = parser.parse_args()

    # Load config
    cfg = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    classmap = load_classmap()

    if args.cmd == "prep":
        step_prep(cfg, classmap)
    elif args.cmd == "train":
        hist = step_train(cfg)
    elif args.cmd == "infer":
        step_infer(cfg)
    elif args.cmd == "eval":
        step_eval(cfg)
    elif args.cmd == "viz":
        step_viz(cfg, DEFAULT_CLASSMAP["palette"])
    else:
        print("Unknown command. Use one of: prep | train | infer | viz | eval")

if __name__ == "__main__":
    main()
