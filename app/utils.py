import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional

import numpy as np
import SimpleITK as sitk
from skimage import measure

from fastapi import UploadFile

from .config import get_settings


def run(
  cmd: str,
  *,
  env: Optional[Dict[str, str]] = None,
  cwd: Optional[Path] = None,
  timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
  settings = get_settings()
  print(f"[cmd] {cmd}", flush=True)
  full_env = os.environ.copy()
  full_env.setdefault("OMP_NUM_THREADS", "1")
  full_env.setdefault("MKL_NUM_THREADS", "1")
  if env:
    full_env.update(env)

  try:
    processed = subprocess.run(
      shlex.split(cmd),
      capture_output=True,
      text=True,
      cwd=str(cwd) if cwd else None,
      env=full_env,
      timeout=timeout,
    )
  except subprocess.TimeoutExpired as exc:
    print(f"[timeout] {cmd} exceeded {timeout}s", flush=True)
    raise

  if processed.returncode != 0:
    print("---- STDOUT ----\n" + processed.stdout, flush=True)
    print("---- STDERR ----\n" + processed.stderr, flush=True)
    raise subprocess.CalledProcessError(
      processed.returncode,
      cmd,
      output=processed.stdout,
      stderr=processed.stderr,
    )
  return processed


def unique_case_id(prefix: str = "case") -> str:
  return f"{prefix}_{uuid4_hex(8)}"


def uuid4_hex(length: int = 8) -> str:
  import uuid

  return uuid.uuid4().hex[:length]


@contextmanager
def temp_case_dirs(case_id: str, *, cleanup: bool = True) -> Iterator[Dict[str, Path]]:
  settings = get_settings()
  in_dir = settings.in_root / case_id
  out_dir = settings.out_root / case_id
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)
  dirs: Dict[str, Path] = {"in": in_dir, "out": out_dir, "root": out_dir}
  try:
    yield dirs
  finally:
    if cleanup and not settings.keep_intermediate:
      shutil.rmtree(in_dir, ignore_errors=True)
      shutil.rmtree(out_dir, ignore_errors=True)


def make_case_logger(case_root: Path) -> Callable[[str], None]:
  log_path = case_root / "case.log"

  def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
      fh.write(f"{timestamp} {message}\n")

  return _log


def list_tree(path: Path, max_files: int = 100) -> list[str]:
  if not path.exists():
    return []
  entries: list[str] = []
  for candidate in path.rglob("*"):
    if candidate.is_file():
      entries.append(candidate.relative_to(path).as_posix())
      if len(entries) >= max_files:
        break
  return sorted(entries)


def stage_artifact(src: Path, dest_dir: Path, filename: str) -> Path:
  dest_dir.mkdir(parents=True, exist_ok=True)
  dest_path = dest_dir / filename
  tmp_path = dest_dir / f".{filename}.part"
  with src.open("rb") as infile, tmp_path.open("wb") as outfile:
    shutil.copyfileobj(infile, outfile)
    outfile.flush()
    os.fsync(outfile.fileno())
  tmp_path.replace(dest_path)
  return dest_path


def read_upload(file: UploadFile, target: Path) -> None:
  data = file.file.read()
  target.write_bytes(data)


def package_outputs(source_dir: Path, *, base_name: str) -> Path:
  temp_dir = Path(tempfile.gettempdir())
  archive_path = temp_dir / f"{base_name}_results.zip"
  if archive_path.exists():
    archive_path.unlink()
  shutil.make_archive(archive_path.with_suffix(""), "zip", source_dir)
  return archive_path


def write_metadata(destination: Path, payload: Dict) -> Path:
  destination.write_text(json.dumps(payload, indent=2))
  return destination


def extract_archive(upload: UploadFile, work_dir: Path) -> Iterable[Path]:
  """
  Supports .zip or .tar(.gz) archives. Returns directories for each case.
  """
  tmp_path = work_dir / upload.filename
  with tmp_path.open("wb") as f:
    shutil.copyfileobj(upload.file, f)

  case_dirs: list[Path] = []
  if zipfile.is_zipfile(tmp_path):
    with zipfile.ZipFile(tmp_path) as zf:
      zf.extractall(work_dir)
    case_dirs = [d for d in (work_dir).iterdir() if d.is_dir()]
  elif tarfile.is_tarfile(tmp_path):
    with tarfile.open(tmp_path) as tf:
      tf.extractall(work_dir)
    case_dirs = [d for d in (work_dir).iterdir() if d.is_dir()]
  else:
    raise ValueError("Unsupported archive type; provide .zip or .tar.gz")
  tmp_path.unlink(missing_ok=True)
  return case_dirs


class Timer:
  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.duration = time.time() - self.start


def log_execution(label: str, seconds: float) -> None:
  print(f"[{label}] {seconds:.2f}s", flush=True)


def _write_vtp(vertices: np.ndarray, faces: np.ndarray, destination: Path) -> Path:
  destination.parent.mkdir(parents=True, exist_ok=True)
  offsets = np.arange(3, 3 * len(faces) + 1, 3, dtype=np.int32)
  connectivity = faces.reshape(-1)
  verts_flat = vertices.reshape(-1)

  with destination.open("w") as fh:
    fh.write('<?xml version="1.0"?>\n')
    fh.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
    fh.write("  <PolyData>\n")
    fh.write(f'    <Piece NumberOfPoints="{len(vertices)}" NumberOfPolys="{len(faces)}">\n')
    fh.write("      <Points>\n")
    fh.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
    fh.write("          " + " ".join(f"{v:.5f}" for v in verts_flat) + "\n")
    fh.write("        </DataArray>\n")
    fh.write("      </Points>\n")
    fh.write("      <Polys>\n")
    fh.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
    fh.write("          " + " ".join(str(int(v)) for v in connectivity) + "\n")
    fh.write("        </DataArray>\n")
    fh.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
    fh.write("          " + " ".join(str(int(v)) for v in offsets) + "\n")
    fh.write("        </DataArray>\n")
    fh.write("      </Polys>\n")
    fh.write("    </Piece>\n")
    fh.write("  </PolyData>\n")
    fh.write("</VTKFile>\n")

  return destination


def generate_meshes(
  mask_path: Path,
  label_map: Optional[Dict[int, str]],
  output_dir: Path,
  logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Path]:
  """
  Convert labeled mask into VTP meshes for the provided labels.
  label_map maps label value -> descriptive name.
  """
  image = sitk.ReadImage(str(mask_path))
  array = sitk.GetArrayFromImage(image)  # z, y, x
  spacing = np.array(image.GetSpacing(), dtype=np.float32)  # (x, y, z)
  origin = np.array(image.GetOrigin(), dtype=np.float32)
  direction = np.array(image.GetDirection(), dtype=np.float32).reshape(3, 3)

  unique_labels = [int(v) for v in np.unique(array) if int(v) > 0]
  if not unique_labels:
    if logger:
      logger("no non-zero labels found; skipping mesh generation")
    return {}

  effective_map: Dict[int, str]
  if label_map:
    effective_map = {int(label): name for label, name in label_map.items() if int(label) in unique_labels}
  else:
    effective_map = {label: f"label_{label}" for label in unique_labels}

  meshes: Dict[str, Path] = {}
  for label_value in unique_labels:
    if label_value not in effective_map:
      continue
    label_name = effective_map[label_value]
    binary = array == int(label_value)
    if not np.any(binary):
      if logger:
        logger(f"label {label_name} (value={label_value}) empty; skipping")
      continue

    verts, faces, _, _ = measure.marching_cubes(binary.astype(np.uint8), level=0.5)
    ijk = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)
    scaled = ijk * spacing
    physical = (direction @ scaled.T).T + origin
    vtp_path = output_dir / f"{label_name}.vtp"
    _write_vtp(physical.astype(np.float32), faces.astype(np.int32), vtp_path)
    meshes[label_name] = vtp_path
    if logger:
      logger(
        f"label {label_name} mesh points={len(physical)} faces={len(faces)} path={vtp_path}"
      )

  return meshes
