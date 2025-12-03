#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSNet multitask trainer (centerline + edge + deep supervision), robust:
- list_data_collate for RandSpatialCropSamplesd lists
- fixed-size RandAffined to keep 96^3 spatial dims
- aux heads resized to GT spatial size
- deep heads: if C==2, train against binary vessel (y>0) target
- centerline regression aligned to (B,1,D,H,W)
- AsDiscrete in validation (no SWI) to avoid permute/shape issues
- strong early logging
"""

import os, sys, json, time, random, traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import argparse
def get_args():
    p = argparse.ArgumentParser("VSNet multitask training — robust")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--roi", nargs=3, type=int, default=[96,96,96])
    p.add_argument("--pixdim", nargs=3, type=float, default=[1.5,1.5,1.5])
    p.add_argument("--patches_per_vol", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--sw_batch", type=int, default=1)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--cache_dir", type=str, default=None)
    # VSNet ctor
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=3)
    # Loss weights
    p.add_argument("--center_w", type=float, default=1.0)
    p.add_argument("--edge_w", type=float, default=1.0)
    p.add_argument("--deep_w2", type=float, default=0.1)
    p.add_argument("--deep_w3", type=float, default=0.1)
    # ROI/bottleneck compatibility
    p.add_argument("--downs", type=int, default=4)
    p.add_argument("--csa_norm_spatial", type=int, default=6)
    # Opt
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    # Debug
    p.add_argument("--debug_shapes", action="store_true")
    return p.parse_args()

def log_early(out_dir: Path, msg: str):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train.log", "a") as f:
            f.write(msg.rstrip()+"\n")
    except Exception as e:
        sys.stderr.write(f"[early-log-fail] {e}\n")

def find_task08_pairs(root: Path):
    root = Path(root)
    cand = []
    for sub in ["imagesTr", "imagesVal", "imagesTs"]:
        p = root / sub
        if p.is_dir():
            for img in sorted(p.glob("*.nii*")):
                stem = img.name.replace("_0000", "")
                lab = (root / sub.replace("images", "labels") / stem)
                if not lab.exists():
                    lab = (root / "labelsTr" / stem)
                if lab.exists():
                    cand.append({"image": str(img), "label": str(lab)})
    random.shuffle(cand)
    n = max(1, int(0.8 * len(cand))) if len(cand) > 1 else len(cand)
    return cand[:n], cand[n:]

def main():
    args = get_args()
    out = Path(args.out)
    log_early(out, "[dbg] entered main")
    log_early(out, f"[dbg] args: {json.dumps(vars(args))}")

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
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        from monai.data import Dataset, list_data_collate
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
            ScaleIntensityRanged, EnsureTyped, ToTensord, AsDiscrete,
            RandSpatialCropSamplesd, RandFlipd, RandAffined,
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
            distance_transform_edt = None; binary_erosion = None

    except Exception as e:
        log_early(out, "[err] heavy import failed")
        log_early(out, traceback.format_exc())
        raise

    set_determinism(seed=args.seed)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    class AddAuxTargetsD:
        def __init__(self, label_key="label"):
            self.label_key = label_key
        def __call__(self, data: Dict):
            d = dict(data)
            lab = d[self.label_key]
            if isinstance(lab, torch.Tensor):
                lab_np = lab.detach().cpu().numpy()
            else:
                lab_np = lab
            arr = lab_np[0] if lab_np.ndim == 4 else lab_np
            vessel = (arr > 0).astype(np.uint8)

            if binary_erosion is not None:
                er = binary_erosion(vessel, iterations=1)
                edge = (vessel ^ er).astype(np.uint8)
            else:
                edge = np.zeros_like(vessel, dtype=np.uint8)
                Z,Y,X = vessel.shape
                for z in range(1,Z-1):
                    for y in range(1,Y-1):
                        for x in range(1,X-1):
                            if vessel[z,y,x] and vessel[z-1:z+2,y-1:y+2,x-1:x+2].min() == 0:
                                edge[z,y,x] = 1
            edge_2class = edge.astype(np.int64)

            center = np.zeros_like(vessel, dtype=np.float32)
            if skeletonize_3d is not None and distance_transform_edt is not None and vessel.any():
                sk = skeletonize_3d(vessel.astype(bool)).astype(np.uint8)
                dist_to_skel = distance_transform_edt(1 - sk)
                inside = vessel.astype(bool)
                vals = dist_to_skel[inside]
                if vals.size > 0:
                    mx = vals.max()
                    if mx > 0:
                        center[inside] = vals / mx

            d["edge"] = edge_2class[None, ...]
            d["center"] = center[None, ...]
            return d

    train_files, val_files = find_task08_pairs(Path(args.data_root))
    log_early(out, f"[dbg] found train files: {len(train_files)}, val files: {len(val_files)}")
    if len(train_files) == 0:
        raise RuntimeError(f"No Task08 pairs found under {args.data_root}")

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
    aug = Compose([
        RandSpatialCropSamplesd(keys=["image","label","edge","center"], roi_size=args.roi, num_samples=args.patches_per_vol, random_center=True, random_size=False),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[0]),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[1]),
        RandFlipd(keys=["image","label","edge","center"], prob=0.2, spatial_axis=[2]),
        RandAffined(keys=["image","label","edge","center"], prob=0.15, rotate_range=(0.15,0.15,0.15), scale_range=(0.1,0.1,0.1), spatial_size=args.roi, padding_mode="border", mode=("bilinear","nearest","nearest","bilinear")),
    ])
    valpre = pre
    valcrop = Compose([
        RandSpatialCropSamplesd(keys=["image","label","edge","center"], roi_size=args.roi, num_samples=1, random_center=True, random_size=False),
        RandAffined(keys=["image","label","edge","center"], prob=1.0, rotate_range=(0.0,0.0,0.0), scale_range=(0.0,0.0,0.0), spatial_size=args.roi, padding_mode="border", mode=("bilinear","nearest","nearest","bilinear")),
    ])

    if args.cache_dir and 'PersistentDataset' in globals():
        train_cache = PersistentDataset(data=train_files, transform=pre, cache_dir=args.cache_dir)
        val_cache   = PersistentDataset(data=val_files,  transform=valpre, cache_dir=args.cache_dir)
        train_ds = Dataset(data=train_cache, transform=aug)
        val_ds   = Dataset(data=val_cache,   transform=valcrop)
        log_early(out, "[dbg] using PersistentDataset cache")
    else:
        train_ds = Dataset(data=train_files, transform=Compose([pre, aug]))
        val_ds   = Dataset(data=val_files,  transform=Compose([valpre, valcrop]))
        log_early(out, "[dbg] using in-memory Dataset")

    from monai.data import list_data_collate
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=list_data_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=list_data_collate)

    # Model
    try:
        model = VSNet(args.in_channels, args.out_channels)
    except TypeError:
        model = VSNet(in_channels=args.in_channels, out_channels=args.out_channels)
    log_early(out, "[dbg] model constructed")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # sanity forward
    model.train()
    with torch.no_grad():
        dummy = torch.zeros(1, args.in_channels, *args.roi, device=device)
        outs = model(dummy)
        if not isinstance(outs, (list, tuple)) or len(outs) != 5:
            msg = f"VSNet forward must return 5 tensors; got {type(outs)}"
            log_early(out, "[err] " + msg); raise RuntimeError(msg)

    def _resize_to(inp, spatial, mode="trilinear"):
        if inp is None or not torch.is_tensor(inp) or inp.dim()!=5:
            return inp
        _, _, d, h, w = inp.shape
        if (d, h, w) == tuple(spatial):
            return inp
        return torch.nn.functional.interpolate(inp, size=tuple(spatial), mode=mode, align_corners=False if mode!="nearest" else None)

    def _maybe_log_shapes(step, seg_v, seg_e, deep2, deep3, reg, y, ed, cl):
        if not args.debug_shapes: return
        shp = lambda t: tuple(t.shape) if torch.is_tensor(t) else str(type(t))
        log_early(out, f"[dbg][{step}] seg_v={shp(seg_v)} seg_e={shp(seg_e)} d2={shp(deep2)} d3={shp(deep3)} reg={shp(reg)} y={shp(y)} ed={shp(ed)} cl={shp(cl)}")

    seg_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    edge_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    mse = nn.MSELoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    @dataclass
    class TrainState:
        epoch: int = 0
        best_dice: float = 0.0
    state = TrainState()

    log_early(out, "[dbg] starting training loop")

    for epoch in range(1, args.epochs + 1):
        model.train()
        state.epoch = epoch
        t0 = time.time()
        running = {"main":0.0, "edge":0.0, "cent":0.0, "deep":0.0, "total":0.0}
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader, 1):
            x  = batch["image"].to(device)
            y  = batch["label"].to(device).long()      # (B,1,D,H,W)
            ed = batch["edge"].to(device).long()       # (B,1,D,H,W)
            cl = batch["center"].to(device).float()    # (B,1,D,H,W)

            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type="cuda", enabled=args.fp16):
                seg_v, reg, seg_e, deep2, deep3 = model(x)

                # resize to target spatial
                tgt_spatial = y.shape[-3:]
                seg_v  = _resize_to(seg_v,  tgt_spatial, "trilinear")
                seg_e  = _resize_to(seg_e,  ed.shape[-3:], "trilinear")
                deep2  = _resize_to(deep2,  tgt_spatial, "trilinear")
                deep3  = _resize_to(deep3,  tgt_spatial, "trilinear")
                reg    = _resize_to(reg,    cl.shape[-3:], "trilinear")

                _maybe_log_shapes(f"ep{epoch}-it{it}", seg_v, seg_e, deep2, deep3, reg, y, ed, cl)

                L_main = seg_loss(seg_v, y)
                L_edge = edge_loss(seg_e, ed)

                reg_pred = reg if (reg is not None and reg.dim()==5) else reg.unsqueeze(1)
                L_cent = mse(reg_pred, cl)

                L_deep = 0.0
                if deep2 is not None:
                    if deep2.shape[1] == 2:
                        y_bin = (y > 0).long()
                        L_deep = L_deep + args.deep_w2 * edge_loss(deep2, y_bin)
                    else:
                        L_deep = L_deep + args.deep_w2 * seg_loss(deep2, y)
                if deep3 is not None:
                    if deep3.shape[1] == 2:
                        y_bin = (y > 0).long()
                        L_deep = L_deep + args.deep_w3 * edge_loss(deep3, y_bin)
                    else:
                        L_deep = L_deep + args.deep_w3 * seg_loss(deep3, y)

                loss = (L_main + args.center_w*L_cent + args.edge_w*L_edge + L_deep) / args.accum

            scaler.scale(loss).backward()
            if it % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            running["main"] += float(L_main.item())
            running["edge"] += float(L_edge.item())
            running["cent"] += float(L_cent.item())
            running["deep"] += float(L_deep if isinstance(L_deep,float) else getattr(L_deep,"item",lambda:0.0)())
            running["total"] += float(loss.item() * args.accum)

        if (len(train_loader) % args.accum) != 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        dur = time.time() - t0
        log_early(out, f"[epoch {epoch:03d}] train time={dur:.1f}s "
                       f"L_main={running['main']/max(1,len(train_loader)):.4f} "
                       f"L_edge={running['edge']/max(1,len(train_loader)):.4f} "
                       f"L_cent={running['cent']/max(1,len(train_loader)):.4f} "
                       f"L_deep={running['deep']/max(1,len(train_loader)):.4f} "
                       f"L_total={running['total']/max(1,len(train_loader)):.4f}")

        # Validation (no SWI; crops already 96^3)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval(); dice_metric.reset()
            post_pred  = AsDiscrete(argmax=True, to_onehot=args.out_channels)
            post_label = AsDiscrete(to_onehot=args.out_channels)
            with torch.no_grad():
                for vb in val_loader:
                    vx = vb["image"].to(device)
                    vy = vb["label"].to(device).long()
                    if args.channels_last:
                        vx = vx.contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", enabled=args.fp16):
                        logits = model(vx)[0]
                    y_pred = post_pred(logits)
                    y_true = post_label(vy.squeeze(1))
                    dice_metric(y_pred=y_pred.float(), y=y_true.float())
                mean_dice = float(dice_metric.aggregate().item())
            log_early(out, f"[epoch {epoch:03d}] val dice={mean_dice:.4f}")

    log_early(out, "[done] training complete")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            argv = sys.argv[1:]
            out_dir = None
            for i,a in enumerate(argv):
                if a == "--out" and i+1 < len(argv):
                    out_dir = Path(argv[i+1]); break
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "train.log", "a") as f:
                    f.write("[err] unhandled exception\n")
                    f.write(traceback.format_exc()+"\n")
        except Exception as e:
            sys.stderr.write(f"[log-fail] {e}\n")
        raise
