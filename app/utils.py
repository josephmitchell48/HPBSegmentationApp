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
from typing import Dict, Iterable, Iterator, Optional

from fastapi import UploadFile

from .config import get_settings


def run(cmd: str, *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
  settings = get_settings()
  print(f"[cmd] {cmd}", flush=True)
  full_env = os.environ.copy()
  full_env.setdefault("OMP_NUM_THREADS", "1")
  full_env.setdefault("MKL_NUM_THREADS", "1")
  if env:
    full_env.update(env)

  processed = subprocess.run(
    shlex.split(cmd),
    capture_output=True,
    text=True,
    cwd=str(cwd) if cwd else None,
    env=full_env,
  )
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
def temp_case_dirs(case_id: str) -> Iterator[Dict[str, Path]]:
  settings = get_settings()
  in_dir = settings.in_root / case_id
  out_dir = settings.out_root / case_id
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)
  try:
    yield {"in": in_dir, "out": out_dir}
  finally:
    if not settings.keep_intermediate:
      shutil.rmtree(in_dir, ignore_errors=True)
      shutil.rmtree(out_dir, ignore_errors=True)


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
