import json
import os
import shutil
from pathlib import Path

from .utils import run


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

  expected = out_dir / f"{case_id}.nii.gz"
  if expected.exists():
    return expected

  for candidate in out_dir.glob("*.nii.gz"):
    if candidate.name not in {"plans.pkl", "postprocessing.json"}:
      return candidate

  raise RuntimeError(f"Task008: expected output not found for {case_id}")


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
  out_path = out_dir / "liver.nii.gz"
  if out_path.exists():
    return out_path

  found = [p.name for p in out_dir.glob("*.nii.gz")]
  raise RuntimeError(f"TotalSegmentator liver: expected liver.nii.gz, found {found}")


def totalseg_multilabel(in_path: Path, out_dir: Path, *, fast: bool = False) -> Path:
  flags = "--ml --fast" if fast else "--ml"
  cmd = f"TotalSegmentator -i {in_path} -o {out_dir} {flags}"
  run(cmd)
  for name in ("segmentation.nii.gz", "segmentations.nii.gz"):
    candidate = out_dir / name
    if candidate.exists():
      return candidate
  raise RuntimeError("TotalSegmentator multi-label output missing")


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
