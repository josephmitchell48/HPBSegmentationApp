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
