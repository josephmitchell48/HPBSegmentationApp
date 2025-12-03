#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vsnet_multitask_v6.py

Key changes vs v5:
- Split transforms into pre_cache (deterministic, heavy I/O/geometry) and runtime (random aug)
- Use PersistentDataset cached to --cache_dir (NVMe) so caching persists across runs
- Use DataLoader pin_memory=True, prefetch_factor, persistent_workers for faster input pipeline
- Honors --workers for both PersistentDataset hashing and DataLoader
- Safe channels_last handling (applied to tensors, not to modules using LayerNorm)
"""
import numpy as np
from monai.transforms import MapTransform
from typing import Dict
try:
    from skimage.morphology import skeletonize_3d
except Exception:
    skeletonize_3d = None
try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
except Exception:
    distance_transform_edt = None
    binary_erosion = None

import os, json, argparse, random, math, time
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from monai.config import print_config
from monai.data import (
    PersistentDataset,
    Dataset,
    CacheDataset,  # kept for reference/option
    DataLoader,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    AsDiscreted,
    ToTensord,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

seg_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

edge_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

mse = torch.nn.MSELoss()

# weights from paper defaults (can be CLI args)
alpha = args.center_w   # default 1.0
beta  = args.edge_w     # default 1.0
lam2  = args.deep_w2    # default 0.1
lam3  = args.deep_w3    # default 0.1

# -----------------------------
# Model helpers (keeps your VSNet import flexible)
# -----------------------------
def build_model(num_classes: int = 2):
    """
    Try several VSNet import paths; fall back to a simple MONAI UNet if needed.
    Adjust this block if your VSNet repo uses a different entrypoint.
    """
    tried = []
    # Try common VSNet import patterns
    candidates = [
        ("vsnet.model", "VSNet"),
        ("vsnet", "VSNet"),
        ("VSNet.model", "VSNet"),
    ]
    for module, cls in candidates:
        try:
            mod = __import__(module, fromlist=[cls])
            VSNet = getattr(mod, cls)
            return VSNet(num_classes=num_classes)
        except Exception as e:
            tried.append(f"{module}.{cls}: {e}")

    # Fallback: light UNet so script still runs
    from monai.networks.nets import UNet
    print("[warn] Could not import VSNet; falling back to MONAI UNet.\nTried:\n - " + "\n - ".join(tried))
    return UNet(
        dimensions=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )

# -----------------------------
# Data discovery and splits
# -----------------------------
def pair_images_labels(root: str):
    img_dir = Path(root) / "imagesTr"
    lab_dir = Path(root) / "labelsTr"
    assert img_dir.is_dir() and lab_dir.is_dir(), f"Expect {img_dir} and {lab_dir}"

    images = sorted(glob(str(img_dir / "*.nii*")))
    labels = sorted(glob(str(lab_dir / "*.nii*")))
    # pair by basename (without extension)
    label_map = {Path(p).stem.split(".")[0]: p for p in labels}
    pairs = []
    for ip in images:
        key = Path(ip).stem.split(".")[0]
        if key in label_map:
            pairs.append({"image": ip, "label": label_map[key]})
    return images, labels, pairs

def make_splits(root: str, val_frac=0.2, seed=13):
    images, labels, pairs = pair_images_labels(root)
    assert len(images) > 0 and len(pairs) > 0, "No training pairs found in imagesTr/labelsTr."

    # deterministic split
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    val_n = max(1, int(len(pairs) * val_frac))
    val_idx = set(idxs[:val_n])
    train_data = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_data = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    print(f"[data] found images: {len(images)}, labels: {len(labels)}, paired: {len(pairs)}")
    print(f"[data] train pairs: {len(train_data)}, val pairs: {len(val_data)}")
    return train_data, val_data

class AddAuxTargetsD(MapTransform):
    """
    From multi-class label (0=bg, 1=HV, 2=PV), build:
      - 'edge': 2-class target [bg/edge] (int64)
      - 'center': float in [0,1] distance-to-centerline inside vessels (else 0)
    """
    def __init__(self, label_key="label", allow_missing_keys=False):
        super().__init__([label_key], allow_missing_keys)
        self.label_key = label_key

    def __call__(self, data: Dict):
        d = dict(data)
        lab = d[self.label_key]  # (1, Z, Y, X) or (Z, Y, X)
        arr = lab[0] if lab.ndim == 4 else lab  # to (Z,Y,X)

        vessel = (arr > 0).astype(np.uint8)

        # --- Edge (binary boundary of vessel) ---
        if binary_erosion is None:
            # simple morphological gradient fallback
            er = (vessel & (np.pad(vessel, 1)[1:-1,1:-1,1:-1] == 1)).astype(np.uint8)  # no-op fallback
            edge = (vessel ^ er).astype(np.uint8)
        else:
            er = binary_erosion(vessel, iterations=1)
            edge = (vessel ^ er).astype(np.uint8)  # 1 at contour, 0 elsewhere

        # class ids: 0=bg, 1=edge
        edge_2class = edge.astype(np.int64)

        # --- Centerline distance (normalized inside vessels) ---
        center = np.zeros_like(vessel, dtype=np.float32)
        if skeletonize_3d is not None and distance_transform_edt is not None and vessel.any():
            sk = skeletonize_3d(vessel.astype(bool)).astype(np.uint8)
            # distance to *nearest skeleton voxel*, but zero outside vessel
            dist_to_skel = distance_transform_edt(1 - sk)
            # mask to vessel only, then min-max normalize to (0,1]
            inside = vessel.astype(bool)
            vals = dist_to_skel[inside]
            if vals.size > 0:
                mx = vals.max()
                if mx > 0:
                    center[inside] = vals / mx  # (0,1]
        # pack back to (1,Z,Y,X)
        d["edge"] = edge_2class[None, ...]
        d["center"] = center[None, ...]
        return d

# -----------------------------
# Transforms (split)
# -----------------------------
def build_transforms(roi, pixdim, patches_per_vol, for_train=True):
    # Deterministic, heavy steps (pre-cache)
    pre_cache = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=tuple(pixdim), mode=("bilinear", "nearest")),
        # Hepatic windowing (adjust if you have a different strategy)
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
        AddAuxTargetsD(label_key="label"),  # <--- add here
    ]

    if not for_train:
        return Compose(pre_cache), Compose([])  # no runtime for val

    # Runtime (random) augmentations — applied each epoch/iteration
    runtime = [
        # Patch sampler – positive/negative sampling around label foreground
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(roi),
            pos=1,
            neg=1,
            num_samples=patches_per_vol,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.15,
            rotate_range=(math.radians(10), math.radians(10), math.radians(10)),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.02),
        ToTensord(keys=["image", "label"]),
    ]
    return Compose(pre_cache), Compose(runtime)


# -----------------------------
# Training / validation
# -----------------------------
def train_one_epoch(
    model, loader, optimizer, scaler, device, amp, accum_steps, channels_last
):
    model.train()
    dice = DiceMetric(include_background=False, reduction="mean")
    # Default multi-task: use Dice+CE on single head unless your VSNet exposes more heads.
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)

    optimizer.zero_grad(set_to_none=True)
    total_loss, step_count = 0.0, 0

    for batch_idx, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)

        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last_3d)

        with autocast(enabled=amp):
            logits = model(img)  # (N, C, D, H, W)
            loss = criterion(logits, lab.float())

        scaler.scale(loss / accum_steps).backward()
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        step_count += 1

        # quick running dice on-thresholded preds (for sanity)
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            dice(y_pred=pred, y=lab)

    mean_loss = total_loss / max(1, step_count)
    mean_dice = dice.aggregate().item() if step_count > 0 else 0.0
    dice.reset()
    return mean_loss, mean_dice


@torch.no_grad()
def validate(model, loader, device, amp, channels_last):
    model.eval()
    dice = DiceMetric(include_background=False, reduction="mean")
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)

    total_loss, step_count = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)

        if channels_last:
            img = img.contiguous(memory_format=torch.channels_last_3d)

        with autocast(enabled=amp):
            logits = model(img)
            loss = criterion(logits, lab.float())

        total_loss += loss.item()
        step_count += 1

        pred = (torch.sigmoid(logits) > 0.5).float()
        dice(y_pred=pred, y=lab)

    mean_loss = total_loss / max(1, step_count)
    mean_dice = dice.aggregate().item() if step_count > 0 else 0.0
    dice.reset()
    return mean_loss, mean_dice


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--roi", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--pixdim", nargs=3, type=float, default=[1.5, 1.5, 1.5])
    parser.add_argument("--patches_per_vol", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--sw_batch", type=int, default=1)  # kept for CLI compatibility
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="/mnt/nvme/monai_cache")
    # Keep your custom weights for multitask heads for compatibility (unused in this generic loop)
    parser.add_argument("--deep_w2", type=float, default=0.3)
    parser.add_argument("--deep_w3", type=float, default=0.1)
    parser.add_argument("--edge_w", type=float, default=0.1)
    parser.add_argument("--center_w", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_cfg.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("[dbg] entered main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gpu] device={device.type}, name={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    # Data
    train_data, val_data = make_splits(args.data_root, val_frac=0.2, seed=13)

    pre_cache_train, runtime_train = build_transforms(args.roi, args.pixdim, args.patches_per_vol, for_train=True)
    pre_cache_val, _ = build_transforms(args.roi, args.pixdim, args.patches_per_vol, for_train=False)

    # PersistentDataset builds a content-addressed cache on disk at --cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    train_pre = PersistentDataset(data=train_data, transform=pre_cache_train, cache_dir=args.cache_dir)
    val_pre   = PersistentDataset(data=val_data,   transform=pre_cache_val,   cache_dir=args.cache_dir)

    # Wrap with runtime (random) transforms for training; validation gets identity runtime
    train_ds = Dataset(data=train_pre, transform=runtime_train)
    val_ds   = Dataset(data=val_pre,   transform=Compose([]))

    # DataLoaders
    # NOTE: persistent_workers True needs workers > 0
    persistent_workers = args.workers > 0
    # prefetch_factor only used when workers > 0; keep a safe value
    prefetch_factor = 2 if args.workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # patches_per_vol controls effective batch via RandCropByPosNegLabeld
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, args.workers),
        pin_memory=True,
        persistent_workers=(min(2, args.workers) > 0),
        prefetch_factor=(2 if min(2, args.workers) > 0 else None),
    )

    # Model
    # NOTE: If your VSNet needs different constructor args, adjust build_model()
    model = build_model(num_classes=1).to(device)

    if args.channels_last:
        # do NOT set modules to channels_last (some use LayerNorm over spatial dims);
        # instead, keep inputs channels_last_3d to help cudnn when safe
        pass

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.fp16)

    best_dice, best_path = -1.0, None
    log_path = os.path.join(args.out, "train_log.jsonl")
    with open(log_path, "a") as lf:
        lf.write(json.dumps({"event": "start", "time": time.time()}, ensure_ascii=False) + "\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp=args.fp16,
            accum_steps=max(1, args.accum), channels_last=args.channels_last
        )
        log_rec = {"epoch": epoch, "train_loss": tr_loss, "train_dice": tr_dice}

        if epoch % max(1, args.val_every) == 0:
            va_loss, va_dice = validate(
                model, val_loader, device, amp=args.fp16, channels_last=args.channels_last
            )
            log_rec.update({"val_loss": va_loss, "val_dice": va_dice})

            # save best
            if va_dice > best_dice:
                best_dice = va_dice
                best_path = os.path.join(args.out, "best.pt")
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, best_path)

        # save periodic checkpoints
        if epoch % 20 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out, f"epoch_{epoch:04d}.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_path)

        # log
        log_rec["time"] = time.time()
        with open(log_path, "a") as lf:
            lf.write(json.dumps(log_rec, ensure_ascii=False) + "\n")

        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} dice={tr_dice:.4f} "
              + (f"| val_loss={log_rec.get('val_loss', float('nan')):.4f} dice={log_rec.get('val_dice', float('nan')):.4f}" if "val_loss" in log_rec else ""))

    print(f"[done] best_val_dice={best_dice:.4f} best_path={best_path}")

if __name__ == "__main__":
    main()
