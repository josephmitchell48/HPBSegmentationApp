import glob
import json
import os
import shutil
import time
from collections.abc import Callable
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


def find_output(out_dir: Path, pattern: str = "*.nii*") -> Path:
  candidates = list(out_dir.rglob(pattern))
  if not candidates:
    raise FileNotFoundError(f"No outputs matching {pattern} under {out_dir}")
  candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
  return candidates[0]


def run_with_verified_output(
  cmd: str,
  out_dir: Path,
  *,
  env: dict[str, str] | None = None,
  expected_hint: str | None = None,
  finder: Callable[[Path], Path] | None = None,
) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  start = time.time()
  run(cmd, env=env)
  duration = time.time() - start
  print(f"[done] {cmd} ({duration:.1f}s)", flush=True)
  # give filesystem a moment to flush outputs on network mounts
  time.sleep(0.25)
  try:
    if finder:
      path = finder(out_dir)
    else:
      path = find_output(out_dir)
    print(f"[found] {path}", flush=True)
    return path
  except FileNotFoundError as exc:
    hits = [
      Path(p).relative_to(out_dir).as_posix()
      for p in glob.glob(str(out_dir / "**" / "*"), recursive=True)
    ]
    raise RuntimeError(
      f"Expected output not found (hint={expected_hint}); saw {hits}"
    ) from exc


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
  return run_with_verified_output(
    cmd,
    out_dir,
    env=env,
    expected_hint=case_id,
    finder=lambda path: find_nnunet_out(path, case_id),
  )


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
  return run_with_verified_output(
    cmd,
    out_dir,
    expected_hint="liver.nii.gz",
    finder=find_ts_liver,
  )


def totalseg_multilabel(in_path: Path, out_dir: Path, *, fast: bool = False) -> Path:
  flags = "--ml --fast" if fast else "--ml"
  cmd = f"TotalSegmentator -i {in_path} -o {out_dir} {flags}"
  return run_with_verified_output(
    cmd,
    out_dir,
    expected_hint="segmentation.nii.gz",
    finder=find_ts_multilabel,
  )


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
