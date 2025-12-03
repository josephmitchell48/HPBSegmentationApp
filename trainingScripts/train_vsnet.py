#!/usr/bin/env python3
import os, glob, time, argparse, json, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import numpy as np

from monai.data import PersistentDataset, Dataset, pad_list_data_collate, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd,
    RandCropByLabelClassesd, EnsureTyped, RandFlipd, RandRotate90d, RandShiftIntensityd,
    MapTransform, DeleteItemsd, SelectItemsd
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# -------------------------
# Speed knobs (global)
# -------------------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.0+
except Exception:
    pass

# Prefer modern AMP API
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.0+
except Exception:
    autocast = torch.cuda.amp.autocast
    GradScaler = torch.cuda.amp.GradScaler

# -------------------------
# Model factory
# -------------------------
def build_model(in_ch=1, out_ch=3, base_ch=32):
    import inspect
    try:
        from VSNet import VSNet
    except Exception as e:
        raise RuntimeError(f"VSNet import failed: {e}. Ensure VSNet.py is on PYTHONPATH or in ~.") from e

    sig = inspect.signature(VSNet)
    params = sig.parameters

    in_keys  = ["in_chans", "in_channels", "in_ch", "in_channel", "input_channels"]
    out_keys = ["out_chans", "out_channels", "out_ch", "n_classes", "num_classes", "classes", "n_outputs"]

    kwargs = {}
    for k in in_keys:
        if k in params:
            kwargs[k] = in_ch; break
    for k in out_keys:
        if k in params:
            kwargs[k] = out_ch; break

    try:
        model = VSNet(**kwargs) if kwargs else VSNet(in_ch, out_ch)
        return model
    except TypeError as e:
        raise TypeError(
            "Could not construct VSNet with inferred args. "
            f"Tried kwargs={kwargs} and positional (in_ch={in_ch}, out_ch={out_ch}). "
            f"VSNet signature is: {sig}"
        ) from e

# -------------------------
# Geometry sync helpers
# -------------------------
class SyncLabelMetaD(MapTransform):
    def __init__(self, image_key="image", label_key="label", allow_missing_keys=False):
        super().__init__([image_key, label_key], allow_missing_keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        lab = d[self.label_key]
        for k in ("affine", "original_affine", "space", "pixdim", "direction", "origin", "spacing"):
            try:
                if hasattr(img, "meta") and k in img.meta:
                    lab.meta[k] = img.meta[k]
            except Exception:
                pass
        return d

def _sanity_affine_pair(sample):
    Ai = sample["image"].meta.get("affine"); Al = sample["label"].meta.get("affine")
    if Ai is None or Al is None:
        return sample
    import numpy as np
    if not np.allclose(np.asarray(Ai), np.asarray(Al), atol=1e-5, rtol=0):
        raise RuntimeError(
            "Affine mismatch after orientation/spacing:\n"
            f"image affine:\n{Ai}\nlabel affine:\n{Al}"
        )
    return sample

# -------------------------
# Data loaders
# -------------------------
def make_loaders(
    root:str,
    workers:int=8,
    roi=(96,96,96),
    pixdim=(1.5,1.5,1.5),
    patches_per_vol:int=8,
    train_limit:int=0,
    val_limit:int=0,
    cache_dir:str="~/monai_cache_task08_p1p2_roi96",
    seed:int=1337
):
    set_determinism(seed=seed)
    random.seed(seed); np.random.seed(seed)

    img_dir = os.path.join(root, "imagesTr")
    lab_dir = os.path.join(root, "labelsTr")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    files = [{"image":p, "label": os.path.join(lab_dir, os.path.basename(p))} for p in imgs]

    if train_limit > 0:
        train_files = files[:train_limit]
    else:
        n = len(files); split = max(1, int(0.8 * n))
        train_files = files[:split]
    if val_limit > 0:
        val_files = files[-val_limit:]
    else:
        val_files = files[len(train_files):]

    keys = ["image", "label"]
    roi = tuple(roi); pixdim = tuple(pixdim)

    # ---------- cached, deterministic pre-pipeline ----------
    pre = Compose([
        LoadImaged(keys),
        SyncLabelMetaD("image", "label"),
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes="RAS", labels=None),
        Spacingd(keys, pixdim=pixdim, mode=("bilinear","nearest")),
        CropForegroundd(keys, source_key="image", margin=8, return_coords=False),
        # (belt-and-suspenders, in case a MONAI build still emits coords)
        DeleteItemsd(keys=[
            "foreground_start_coord","foreground_end_coord",
            "image_foreground_start_coord","image_foreground_end_coord",
            "label_foreground_start_coord","label_foreground_end_coord",
        ]),
        SpatialPadd(keys, spatial_size=roi),
        EnsureTyped(keys, dtype=(torch.float32, torch.uint8)),
        ScaleIntensityRanged("image", -100, 300, 0.0, 1.0, clip=True),
        # Keep only what we train on
        SelectItemsd(keys=["image","label"]),
    ])

    # ---------- training-time augmentations ----------
    post = Compose([
        RandFlipd(keys, prob=0.5, spatial_axis=[0,1,2]),
        RandRotate90d(keys, prob=0.25, max_k=3),
        RandShiftIntensityd("image", offsets=0.1, prob=0.25),
        RandCropByLabelClassesd(
            keys=keys, label_key="label",
            spatial_size=roi, num_samples=patches_per_vol,
            ratios=[0, 1, 1], num_classes=3, image_key="image"
        ),
        EnsureTyped("image", dtype=torch.float32),
        EnsureTyped("label", dtype=torch.uint8),
        SelectItemsd(keys=["image","label"]),
    ])

    # ---------- validation transforms (no aug) ----------
    tvl = Compose([
        LoadImaged(keys),
        SyncLabelMetaD("image", "label"),
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes="RAS", labels=None),
        Spacingd(keys, pixdim=pixdim, mode=("bilinear","nearest")),
        CropForegroundd(keys, source_key="image", margin=8, return_coords=False),
        DeleteItemsd(keys=[
            "foreground_start_coord","foreground_end_coord",
            "image_foreground_start_coord","image_foreground_end_coord",
            "label_foreground_start_coord","label_foreground_end_coord",
        ]),
        SpatialPadd(keys, spatial_size=roi),
        EnsureTyped(keys, dtype=(torch.float32, torch.uint8)),
        ScaleIntensityRanged("image", -100, 300, 0.0, 1.0, clip=True),
        SelectItemsd(keys=["image","label"]),
    ])

    cache_dir = os.path.expanduser(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    train_pre = PersistentDataset(train_files, pre, cache_dir=cache_dir)
    train_ds  = Dataset(train_pre, transform=post)

    tvl_check = Compose([tvl, _sanity_affine_pair])
    val_pre   = PersistentDataset(val_files, tvl_check, cache_dir=cache_dir)
    val_ds    = Dataset(val_pre)

    pin = True
    pw  = workers > 0
    pf  = 2 if workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=pw, prefetch_factor=(pf or 2),
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, workers // 2), pin_memory=pin,
        persistent_workers=pw, prefetch_factor=(pf or 2),
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader, len(train_files), len(val_files)

# -------------------------
# Training helpers
# -------------------------
def save_ckpt(state, path:str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def validate(model, loader, device, sw_roi, sw_batch, num_classes=3):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="none")
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            y = b["label"].to(device).squeeze(1).long()
            logits = sliding_window_inference(
                x, roi_size=sw_roi, sw_batch_size=sw_batch,
                predictor=lambda z: forward_logits(model, z), overlap=0.5, mode="gaussian",
                device=device, progress=False
            )
            pr = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            yp = torch.nn.functional.one_hot(pr, num_classes).permute(0,4,1,2,3).float()
            yy = torch.nn.functional.one_hot(y,  num_classes).permute(0,4,1,2,3).float()
            dice_metric(y_pred=yp, y=yy)
        d = dice_metric.aggregate()
        dice_metric.reset()
    return float(d.mean().item())

def forward_logits(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        for k in ("logits", "out", "y", "pred", "seg"):
            if k in out:
                out = out[k]; break
    return out

# -------------------------
# Main train
# -------------------------
def main():
    print("[dbg] entered main", flush=True)
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", type=str, required=True, help="Task08_HepaticVessel root (with imagesTr/labelsTr)")
    p.add_argument("--out", type=str, default="./ckpts_vsnet", help="output folder")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=40, help="early stopping patience (epochs)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--accum", type=int, default=4, help="grad accumulation steps")
    p.add_argument("--clip", type=float, default=0.0, help="grad clip norm (0 = off)")
    p.add_argument("--roi", type=int, nargs=3, default=(96,96,96))
    p.add_argument("--pixdim", type=float, nargs=3, default=(1.5,1.5,1.5))
    p.add_argument("--patches_per_vol", type=int, default=8)
    p.add_argument("--sw_batch", type=int, default=2, help="sliding window batch size at validation")
    p.add_argument("--train_limit", type=int, default=0, help="use first N cases for training (0=all)")
    p.add_argument("--val_limit", type=int, default=0, help="use last N cases for val (0=auto)")
    p.add_argument("--cache_dir", type=str, default="~/monai_cache_task08_p1p2_roi96")
    p.add_argument("--val_every", type=int, default=2, help="validate every N epochs (>=1)")
    p.add_argument("--speed_log", type=int, default=20, help="print speed every N iters (0=off)")
    p.add_argument("--resume", type=str, default="", help="path to a checkpoint to resume")
    p.add_argument("--fp16", action="store_true", help="mixed precision (amp)")
    p.add_argument("--channels_last", action="store_true", help="use channels_last_3d memory format if available")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, n_tr, n_vl = make_loaders(
        root=args.data_root, workers=args.workers, roi=tuple(args.roi),
        pixdim=tuple(args.pixdim), patches_per_vol=args.patches_per_vol,
        train_limit=args.train_limit, val_limit=args.val_limit,
        cache_dir=args.cache_dir
    )
    print(f"[data] train cases: {n_tr}, val cases: {n_vl}")
    print(f"[gpu] device={device}, name={torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    model = build_model(in_ch=1, out_ch=3).to(device)
    if args.channels_last and hasattr(torch, "channels_last_3d"):
        model = model.to(memory_format=torch.channels_last_3d)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = GradScaler('cuda', enabled=args.fp16)

    start_epoch = 0
    best_dice = -1.0
    iters_per_epoch = len(train_loader)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        except Exception:
            pass
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", best_dice)
        print(f"[resume] loaded {args.resume} @ epoch {start_epoch-1}, best_dice={best_dice:.4f}")

    no_improve = 0
    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, "train_cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[cfg] wrote {cfg_path}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        last_speed_t = time.time()
        it_window = 0

        for it, batch in enumerate(train_loader):
            if not torch.is_tensor(batch["image"]):
                raise TypeError(f"Batch 'image' is not a tensor (got {type(batch['image'])}).")
            if not torch.is_tensor(batch["label"]):
                raise TypeError(f"Batch 'label' is not a tensor (got {type(batch['label'])}).")

            x = batch["image"].to(device)                # [B,1,D,H,W]
            y = batch["label"].to(device).long()         # [B,1,D,H,W] ints
            if args.channels_last and hasattr(torch, "channels_last_3d"):
                x = x.to(memory_format=torch.channels_last_3d)

            with autocast('cuda', enabled=args.fp16):
                logits = forward_logits(model, x)
                loss = loss_fn(logits, y)
                loss = loss / max(1, args.accum)

            scaler.scale(loss).backward()

            if (it + 1) % args.accum == 0:
                if args.clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach().item()) * max(1, args.accum)

            if args.speed_log > 0:
                it_window += 1
                if it_window >= args.speed_log:
                    now = time.time()
                    it_s = it_window / max(1e-9, (now - last_speed_t))
                    bs = batch["image"].shape[0]
                    print(f"[speed] epoch {epoch:03d} it {it+1:05d}: {it_s:.2f} it/s, ~{it_s*bs:.2f} samples/s")
                    last_speed_t = now
                    it_window = 0

        if (iters_per_epoch % args.accum) != 0:
            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / max(1, iters_per_epoch)

        do_val = (args.val_every <= 1) or ((epoch % args.val_every) == 0)
        val_dice = float('nan')
        if do_val:
            val_dice = validate(
                model, val_loader, device,
                sw_roi=tuple(args.roi), sw_batch=args.sw_batch, num_classes=3
            )

        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] loss={train_loss:.4f}  val_dice={val_dice:.4f}  ({dt:.1f}s)")

        last_path = os.path.join(args.out, "last.ckpt")
        save_ckpt({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_dice": best_dice
        }, last_path)

        if do_val:
            if val_dice > best_dice:
                best_dice = val_dice
                no_improve = 0
                best_path = os.path.join(args.out, "best.ckpt")
                save_ckpt({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice
                }, best_path)
                print(f"  ↳ new best! saved {best_path} (dice={best_dice:.4f})")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[early-stop] no improvement for {args.patience} val-epochs. Best={best_dice:.4f}")
                    break

    print(f"[done] best_dice={best_dice:.4f}")

if __name__ == "__main__":
    main()
