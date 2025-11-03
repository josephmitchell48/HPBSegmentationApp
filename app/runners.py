import glob
import json
import os
import shutil
from pathlib import Path

from .utils import run


def _first_existing(*paths: Path | str | None) -> Path | None:
  for candidate in paths:
    if not candidate:
      continue
    path = Path(candidate)
    if path.exists():
      return path
  return None


def find_ts_liver(out_dir: Path) -> Path:
  candidates = [
    out_dir / "liver.nii.gz",
    out_dir / "segmentations" / "liver.nii.gz",
  ]
  match = _first_existing(*candidates)
  if match:
    return match

  hits = glob.glob(str(out_dir / "**" / "liver.nii*"), recursive=True)
  if hits:
    return Path(hits[0])

  raise FileNotFoundError("TotalSegmentator liver output not found")


def find_ts_multilabel(out_dir: Path) -> Path:
  candidates = [
    out_dir / "segmentation.nii.gz",
    out_dir / "segmentations.nii.gz",
    out_dir / "segmentations" / "segmentation.nii.gz",
  ]
  match = _first_existing(*candidates)
  if match:
    return match

  hits = glob.glob(str(out_dir / "**" / "segmentation*.nii*"), recursive=True)
  if hits:
    return Path(hits[0])

  raise FileNotFoundError("TotalSegmentator multi-label output not found")


def find_nnunet_out(out_dir: Path, case_id: str) -> Path:
  match = _first_existing(out_dir / f"{case_id}.nii.gz")
  if match:
    return match

  hits = sorted(glob.glob(str(out_dir / "*.nii*")))
  if hits:
    return Path(hits[0])

  raise FileNotFoundError(f"nnU-Net output not found in {out_dir}")


def nnunet_v1_task008(in_dir: Path, out_dir: Path, *, case_id: str, folds: str = "0") -> Path:
  env = os.environ.copy()
  env.setdefault("RESULTS_FOLDER", "/models/nnunet_v1")
  cmd = (
    "nnUNet_predict "
    f"-i {in_dir} "
    f"-o {out_dir} "
    "-t Task008_HepaticVessel "
    "-m 3d_fullres "
    f"-f {folds} "
    "--disable_tta "
    "--num_threads_preprocessing 1 --num_threads_nifti_save 1 "
    "-chk model_final_checkpoint"
  )
  run(cmd, env=env)

  try:
    return find_nnunet_out(out_dir, case_id)
  except FileNotFoundError as exc:
    hits = sorted(Path(p).name for p in glob.glob(str(out_dir / "*.nii*")))
    raise RuntimeError(f"Task008: expected output not found for {case_id}; saw {hits}") from exc


def totalseg_liver_only(in_path: Path, out_dir: Path, *, fast: bool = False) -> Path:
  flags = ["--fast"] if fast else []
  flag_str = " ".join(flags)
  cmd = (
    "TotalSegmentator "
    f"-i {in_path} "
    f"-o {out_dir} "
    "--roi_subset liver "
    f"{flag_str}"
  ).strip()
  run(cmd)
  try:
    return find_ts_liver(out_dir)
  except FileNotFoundError as exc:
    hits = sorted(Path(p).relative_to(out_dir).as_posix() for p in glob.glob(str(out_dir / "**" / "*.nii*"), recursive=True))
    raise RuntimeError(f"TotalSegmentator liver: output not found; saw {hits}") from exc


def totalseg_multilabel(in_path: Path, out_dir: Path, *, fast: bool = False) -> Path:
  flags = "--ml --fast" if fast else "--ml"
  cmd = f"TotalSegmentator -i {in_path} -o {out_dir} {flags}"
  run(cmd)
  try:
    return find_ts_multilabel(out_dir)
  except FileNotFoundError as exc:
    hits = sorted(Path(p).relative_to(out_dir).as_posix() for p in glob.glob(str(out_dir / "**" / "*.nii*"), recursive=True))
    raise RuntimeError(f"TotalSegmentator multi-label output missing; saw {hits}") from exc


def prepare_package(case_root: Path, *, liver_mask: Path, task008_mask: Path, metadata: dict) -> Path:
  pkg_dir = case_root / "package"
  pkg_dir.mkdir(parents=True, exist_ok=True)

  target_liver = pkg_dir / "liver.nii.gz"
  target_task8 = pkg_dir / "task008.nii.gz"
  shutil.copy2(liver_mask, target_liver)
  shutil.copy2(task008_mask, target_task8)

  meta_path = pkg_dir / "meta.json"
  meta_path.write_text(json.dumps(metadata, indent=2))

  return pkg_dir
