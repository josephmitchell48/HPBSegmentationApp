#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSNet multitask trainer (centerline + edge + deep supervision), robust (v5):
- list_data_collate for RandSpatialCropSamplesd lists
- fixed-size RandAffined to keep 96^3 spatial dims
- aux heads resized to GT spatial size
- deep heads: if C==2, train against binary vessel (y>0) target
- centerline regression aligned to (B,1,D,H,W)
- AsDiscrete in validation (no SWI) to avoid permute/shape issues
- channels_last_3d applied to **inputs only** (not model), to avoid rank errors
- strong early logging
"""

import os, sys, json, time, random, traceback
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

def log_early(outdir, msg: str):
    """Log early messages (before torch, etc) to stdout and an early.log file."""
    print(msg, flush=True)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "early.log"), "a") as f:
            f.write(msg + "\n")

@dataclass
class Args:
    data_root: str
    out: str
    epochs: int = 200
    workers: int = 4
    roi: Tuple[int,int,int] = (96,96,96)
    pixdim: Tuple[float,float,float] = (1.5,1.5,1.5)
    patches_per_vol: int = 4
    accum: int = 1
    sw_batch: int = 1
    val_every: int = 2
    in_channels: int = 1
    out_channels: int = 3
    center_w: float = 0.2
    edge_w: float = 0.2
    deep_w2: float = 0.0
    deep_w3: float = 0.0
    downs: int = 4
    csa_norm_spatial: int = 6
    fp16: bool = False
    channels_last: bool = False
    cache_dir: Optional[str] = None
    seed: int = 42

def parse_args(argv=None) -> Args:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--roi", nargs=3, type=int, default=(96,96,96))
    p.add_argument("--pixdim", nargs=3, type=float, default=(1.5,1.5,1.5))
    p.add_argument("--patches_per_vol", type=int, default=4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--sw_batch", type=int, default=1)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=3)
    p.add_argument("--center_w", type=float, default=0.2)
    p.add_argument("--edge_w", type=float, default=0.2)
    p.add_argument("--deep_w2", type=float, default=0.0)
    p.add_argument("--deep_w3", type=float, default=0.0)
    p.add_argument("--downs", type=int, default=4)
    p.add_argument("--csa_norm_spatial", type=int, default=6)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args_ns = p.parse_args(argv)
    return Args(**vars(args_ns))

def main(argv=None):
    args = parse_args(argv)
    out = args.out
    os.makedirs(out, exist_ok=True)

    log_early(out, f"[cfg] {json.dumps(asdict(args), indent=2)}")

    # ROI <-> CSA LayerNorm bottleneck check
    ds = 2 ** args.downs
    bx, by, bz = args.roi[0]//ds, args.roi[1]//ds, args.roi[2]//ds
    if (bx,by,bz) != (args.csa_norm_spatial,)*3:
        msg = (f"ROI {tuple(args.roi)} incompatible with CSA LayerNorm: bottleneck {(bx,by,bz)} "
               f"but expected {(args.csa_norm_spatial,)*3}. Try --roi {args.csa_norm_spatial*ds} "
               f"{args.csa_norm_spatial*ds} {args.csa_norm_spatial*ds}")
        log_early(out, "[err] " + msg); raise RuntimeError(msg)

    # heavy imports
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
            ScaleIntensityRanged, EnsureTyped, ToTensord, AsDiscrete,
            RandSpatialCropSamplesd, RandCropByPosNegLabeld, RandFlipd, RandAffined,
        )
        from monai.metrics import DiceMetric
        from monai.losses import DiceCELoss
        from monai.utils import set_determinism

        try:
            from monai.data import PersistentDataset
            have_persist = True
        except Exception:
            have_persist = False

        try:
            from VSNet import VSNet
        except Exception as e:
            log_early(out, f"[err] cannot import VSNet: {e}")
            raise

        try:
            from skimage.morphology import skeletonize_3d
        except Exception:
            skeletonize_3d = None
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion
        except Exception:
            distance_transform_edt, binary_erosion = None, None

    except Exception as e:
        log_early(out, f"[err] heavy imports failed: {e}")
        traceback.print_exc()
        return 1

    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_early(out, f"[dbg] device={device}, cuda={torch.cuda.is_available()}")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    data_root = args.data_root
    imagesTr = os.path.join(data_root, "imagesTr")
    labelsTr = os.path.join(data_root, "labelsTr")

    img_files = sorted([f for f in os.listdir(imagesTr) if f.endswith(".nii.gz") and not f.startswith("._")])
    label_files = sorted([f for f in os.listdir(labelsTr) if f.endswith(".nii.gz") and not f.startswith("._")])

    assert len(img_files) == len(label_files), "Image/label count mismatch"
    log_early(out, f"[data] found images: {len(img_files)}, labels: {len(label_files)}")

    files = []
    for img, lab in zip(img_files, label_files):
        if img != lab:
            log_early(out, f"[warn] filename mismatch: {img} vs {lab}")
        files.append({
            "image": os.path.join(imagesTr, img),
            "label": os.path.join(labelsTr, lab),
        })

    # simple split: 80/20
    random.Random(args.seed).shuffle(files)
    n = len(files)
    n_train = int(n * 0.8)
    train_files = files[:n_train]
    val_files = files[n_train:]
    log_early(out, f"[data] train files: {len(train_files)}, val files: {len(val_files)}")

    # -------------------------------------------------------------------------
    # Auxiliary target creation (edge + centerline)
    # -------------------------------------------------------------------------
    class AddAuxTargetsD:
        """
        Given dict with 'label' (C=1 or C=num_classes), produce:
        - 'edge': binary edge map (thin boundary of vessel)
        - 'center': (if skeletonize_3d & distance_transform_edt available) centerline regression target
        """

        def __init__(self, label_key="label"):
            self.label_key = label_key

        def __call__(self, d):
            lab = d[self.label_key]  # expecting torch or numpy, but pre-Transforms still on numpy
            # We postpone conversion to tensor -> do this in numpy
            if isinstance(lab, torch.Tensor):
                lab_np = lab.cpu().numpy()
            else:
                lab_np = lab

            # lab_np: (C,D,H,W) or (D,H,W)
            if lab_np.ndim == 4:
                lab_np = lab_np[0]  # assume first channel is semantic label

            # vessel mask (y>0)
            vessel = (lab_np > 0).astype(np.uint8)

            # edge map: vessel minus eroded vessel
            if binary_erosion is not None:
                eroded = binary_erosion(vessel, iterations=1)
                edge = (vessel & ~eroded).astype(np.uint8)
            else:
                edge = vessel.copy()

            # centerline via skeleton + distance transform
            if skeletonize_3d is not None and distance_transform_edt is not None:
                skel = skeletonize_3d(vessel).astype(np.uint8)
                dist = distance_transform_edt(vessel)
                center = (skel * dist).astype(np.float32)
            else:
                skel = vessel
                center = vessel.astype(np.float32)

            # We keep edge as binary 0/1; center is float
            # For multi-class (C>2), we'll train center against vessel>0 anyway.

            # put back into dict as channels-first
            edge_2class = edge
            d["edge"] = edge_2class[None, ...]
            d["center"] = center[None, ...]
            return d

    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------
    pre = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=args.pixdim, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image","label"]),
        AddAuxTargetsD(label_key="label"),
        ToTensord(keys=["image","label","edge","center"]),
    ])
    # Label-aware patch sampling for training
    aug = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label", "edge", "center"],
            label_key="label",                 # treat label > 0 as foreground
            spatial_size=tuple(args.roi),
            pos=1.0,
            neg=1.0,
            num_samples=args.patches_per_vol,
        ),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[0]),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[1]),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[2]),
        RandAffined(
            keys=["image","label","edge","center"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
            mode=("bilinear","nearest","nearest","bilinear"),
        ),
    ])
    valpre = pre
    valcrop = Compose([
        RandSpatialCropSamplesd(
            keys=["image","label","edge","center"],
            roi_size=args.roi,
            num_samples=1,
            random_center=True,
            random_size=False,
        ),
    ])

    if args.cache_dir:
        try:
            from monai.data import PersistentDataset
            train_cache = PersistentDataset(data=train_files, transform=pre, cache_dir=args.cache_dir)
            val_cache   = PersistentDataset(data=val_files,  transform=valpre, cache_dir=args.cache_dir)
            train_ds = Dataset(data=train_cache, transform=aug)
            val_ds   = Dataset(data=val_cache,   transform=valcrop)
            log_early(out, "[dbg] using PersistentDataset cache")
        except Exception as e:
            log_early(out, f"[warn] PersistentDataset unavailable: {e}")
            train_ds = Dataset(data=train_files, transform=Compose([pre, aug]))
            val_ds   = Dataset(data=val_files,  transform=Compose([valpre, valcrop]))
            log_early(out, "[dbg] using in-memory Dataset")
    else:
        train_ds = Dataset(data=train_files, transform=Compose([pre, aug]))
        val_ds   = Dataset(data=val_files,  transform=Compose([valpre, valcrop]))
        log_early(out, "[dbg] using in-memory Dataset")

    from monai.data import list_data_collate
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=list_data_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=list_data_collate)

    # -------------------------------------------------------------------------
    # Model, loss, metrics
    # -------------------------------------------------------------------------
    model = VSNet(
        spatial_dim=3,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        downs=args.downs,
    ).to(device)

    if args.channels_last:
        log_early(out, "[dbg] enabling channels_last on inputs")
    if torch.cuda.device_count() > 1:
        log_early(out, f"[dbg] using DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    main_loss = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    edge_loss = DiceCELoss(
        include_background=True,
        to_onehot_y=False,
        softmax=False,
        sigmoid=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    center_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = 0.0
    best_epoch = -1

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def move_to(x):
        if isinstance(x, torch.Tensor):
            if args.channels_last and x.ndim == 5:
                x = x.to(memory_format=torch.channels_last_3d)
            return x.to(device)
        return x

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss_main = 0.0
        epoch_loss_edge = 0.0
        epoch_loss_cent = 0.0
        epoch_loss_deep = 0.0
        epoch_loss_total = 0.0

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = move_to(batch["image"])
            labs = move_to(batch["label"])
            edge = move_to(batch["edge"])
            center = move_to(batch["center"])

            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(imgs)
                # out expected: dict with keys: "seg", "edge", "center", "deep2", "deep3"
                seg_logits = out["seg"]
                edge_logits = out["edge"]
                center_pred = out["center"]
                deep2 = out.get("deep2", None)
                deep3 = out.get("deep3", None)

                # main loss: multi-class (C=args.out_channels)
                loss_main = main_loss(seg_logits, labs)

                # edge loss: treat edge as binary map
                loss_edge = edge_loss(edge_logits, edge)

                # center loss: regression vs center
                # ensure shapes align
                if center_pred.shape[1] != 1:
                    # reduce to single-channel by mean
                    center_pred_reg = center_pred.mean(dim=1, keepdim=True)
                else:
                    center_pred_reg = center_pred
                loss_cent = center_loss(center_pred_reg, center)

                loss_deep = 0.0
                if deep2 is not None and args.deep_w2 > 0:
                    if args.out_channels == 2:
                        # binary vessel target
                        vessel = (labs > 0).float()
                        loss_deep2 = main_loss(deep2, vessel)
                    else:
                        loss_deep2 = main_loss(deep2, labs)
                    loss_deep = loss_deep + args.deep_w2 * loss_deep2
                if deep3 is not None and args.deep_w3 > 0:
                    if args.out_channels == 2:
                        vessel = (labs > 0).float()
                        loss_deep3 = main_loss(deep3, vessel)
                    else:
                        loss_deep3 = main_loss(deep3, labs)
                    loss_deep = loss_deep + args.deep_w3 * loss_deep3

                loss_total = loss_main + args.edge_w * loss_edge + args.center_w * loss_cent + loss_deep

            scaler.scale(loss_total).backward()

            if (step + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss_main  += loss_main.item()
            epoch_loss_edge  += loss_edge.item()
            epoch_loss_cent  += loss_cent.item()
            epoch_loss_deep  += float(loss_deep)
            epoch_loss_total += loss_total.item()

        t1 = time.time()
        n_steps = len(train_loader)
        Lm = epoch_loss_main / n_steps
        Le = epoch_loss_edge / n_steps
        Lc = epoch_loss_cent / n_steps
        Ld = epoch_loss_deep / n_steps
        Lt = epoch_loss_total / n_steps

        log_early(out, f"[epoch {epoch:03d}] train time={t1-t0:.1f}s "
                       f"L_main={Lm:.4f} L_edge={Le:.4f} L_cent={Lc:.4f} L_deep={Ld:.4f} L_total={Lt:.4f}")

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------
        if epoch % args.val_every == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = move_to(batch["image"])
                    labs = move_to(batch["label"])

                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        out = model(imgs)
                        seg_logits = out["seg"]
                        # convert to prob and one-hot for metric
                        probs = torch.softmax(seg_logits, dim=1)
                        preds = torch.argmax(probs, dim=1, keepdim=True)

                    # AsDiscrete for labels and preds
                    to_onehot = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
                    y_pred = to_onehot(preds)
                    y_true = to_onehot(labs)

                    dice_metric(y_pred, y_true)

                mean_dice = dice_metric.aggregate().item()
                log_early(out, f"[epoch {epoch:03d}] val dice={mean_dice:.4f}")

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(out, "best.pt"))
                    log_early(out, f"[ckpt] new best at epoch {epoch:03d} dice={best_dice:.4f}")

        # save last model
        torch.save(model.state_dict(), os.path.join(out, "last.pt"))

    log_early(out, f"[done] training complete, best dice={best_dice:.4f} at epoch {best_epoch:03d}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
