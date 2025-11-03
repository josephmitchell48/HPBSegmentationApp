from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .config import get_settings
from .runners import nnunet_v1_task008, prepare_package, totalseg_liver_only, totalseg_multilabel
from .utils import (
  Timer,
  extract_archive,
  make_case_logger,
  log_execution,
  package_outputs,
  stage_artifact,
  temp_case_dirs,
  unique_case_id,
  write_metadata,
)

settings = get_settings()
app = FastAPI(title=settings.app_name)
ARTIFACT_ROOT = Path("/opt/hpb-seg/artifacts")
WORK_ROOT = ARTIFACT_ROOT / "work"
SEND_ROOT = ARTIFACT_ROOT / "send"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
SEND_ROOT.mkdir(parents=True, exist_ok=True)
MAX_CONCURRENCY = int(os.getenv("HPB_MAX_CONCURRENCY", "1"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("HPB_MODEL_TIMEOUT_SECONDS", "1800"))
SEGMENT_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
RESULTS_FOLDER = Path(os.getenv("RESULTS_FOLDER", "/models/nnunet_v1"))
TOTALSEG_HOME = Path(os.getenv("TOTALSEG_HOME", "/models/totalseg"))


def _check_writable(path: Path) -> bool:
  probe = path / ".write_probe"
  try:
    path.mkdir(parents=True, exist_ok=True)
    with probe.open("w") as fh:
      fh.write("ok")
    probe.unlink(missing_ok=True)
    return True
  except Exception:
    return False


@app.get("/healthz")
def health() -> PlainTextResponse:
  return PlainTextResponse("ok")


@app.get("/version")
def version() -> JSONResponse:
  payload: dict[str, object] = {
    "artifacts": {
      "work_dir": str(WORK_ROOT),
      "send_dir": str(SEND_ROOT),
      "work_writable": _check_writable(WORK_ROOT),
      "send_writable": _check_writable(SEND_ROOT),
    },
    "executables": {
      "nnUNet_predict": shutil.which("nnUNet_predict") is not None,
      "TotalSegmentator": shutil.which("TotalSegmentator") is not None,
    },
    "models": {
      "results_folder": str(RESULTS_FOLDER),
      "totalseg_home": str(TOTALSEG_HOME),
      "results_present": RESULTS_FOLDER.exists(),
      "totalseg_present": TOTALSEG_HOME.exists(),
    },
  }
  try:
    import torch  # type: ignore

    payload["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
  except Exception as exc:  # pragma: no cover
    payload["torch_error"] = str(exc)
  try:
    import nnunet  # type: ignore

    payload["nnunet_v1"] = getattr(nnunet, "__version__", "unknown")
  except Exception as exc:  # pragma: no cover
    payload["nnunet_v1_error"] = str(exc)
  try:
    import totalsegmentator as ts  # type: ignore

    payload["totalseg"] = getattr(ts, "__version__", "unknown")
  except Exception as exc:  # pragma: no cover
    payload["totalseg_error"] = str(exc)

  return JSONResponse(payload)


@app.get("/readinessz")
def readiness() -> JSONResponse:
  checks: dict[str, bool] = {}
  errors: list[str] = []

  def add_check(name: str, condition: bool, error_message: str) -> None:
    checks[name] = bool(condition)
    if not condition:
      errors.append(error_message)

  add_check("nnUNet_predict", shutil.which("nnUNet_predict") is not None, "nnUNet_predict missing from PATH")
  add_check("TotalSegmentator", shutil.which("TotalSegmentator") is not None, "TotalSegmentator missing from PATH")
  add_check("results_folder", RESULTS_FOLDER.exists(), f"{RESULTS_FOLDER} missing")
  add_check("totalseg_home", TOTALSEG_HOME.exists(), f"{TOTALSEG_HOME} missing")
  add_check("artifacts_work_writable", _check_writable(WORK_ROOT), f"Cannot write to {WORK_ROOT}")
  add_check("artifacts_send_writable", _check_writable(SEND_ROOT), f"Cannot write to {SEND_ROOT}")
  add_check("max_concurrency", MAX_CONCURRENCY > 0, "HPB_MAX_CONCURRENCY must be >=1")

  status_code = 200 if not errors else 503
  payload = {"ok": not errors, "checks": checks}
  if errors:
    payload["errors"] = errors
  return JSONResponse(payload, status_code=status_code)


@app.post("/segment/task008")
async def segment_task008(ct: UploadFile = File(...), folds: str = "0"):
  case_id = unique_case_id()
  request_id = unique_case_id(prefix="req")
  case_root = WORK_ROOT / case_id
  in_dir = case_root / "in"
  out_dir = case_root / "out"
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)

  case_logger = make_case_logger(case_root)
  data = await ct.read()
  case_logger(
    f"request_id={request_id} task=task008 filename={ct.filename or 'upload'} bytes={len(data)} folds={folds}"
  )

  in_path = in_dir / f"{case_id}_0000.nii.gz"
  in_path.write_bytes(data)
  case_logger(f"saved input to {in_path}")

  try:
    async with SEGMENT_SEMAPHORE:
      with Timer() as timer:
        output_path = nnunet_v1_task008(
          in_dir,
          out_dir,
          case_id=case_id,
          folds=folds,
          logger=case_logger,
          timeout=MODEL_TIMEOUT_SECONDS,
        )
    log_execution(f"task008:{case_id}", timer.duration)
    case_logger(f"nnUNet duration={timer.duration:.2f}s output={output_path}")
  except Exception as exc:
    case_logger(f"error task008: {exc}")
    raise HTTPException(
      status_code=500,
      detail={"error": f"Task008 failed: {exc}", "case_id": case_id, "request_id": request_id},
    ) from exc

  stable_path = stage_artifact(output_path, SEND_ROOT, f"{case_id}_task008.nii.gz")
  case_logger(f"copied output to {stable_path} size={stable_path.stat().st_size}")
  return FileResponse(
    stable_path,
    media_type="application/gzip",
    filename=stable_path.name,
    headers={"X-Request-ID": request_id, "X-Case-ID": case_id},
  )


@app.post("/segment/liver")
async def segment_liver(ct: UploadFile = File(...), fast: bool = False):
  case_id = unique_case_id()
  request_id = unique_case_id(prefix="req")
  case_root = WORK_ROOT / case_id
  in_dir = case_root / "in"
  out_dir = case_root / "out"
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)

  case_logger = make_case_logger(case_root)
  data = await ct.read()
  case_logger(
    f"request_id={request_id} task=liver filename={ct.filename or 'upload'} bytes={len(data)} fast={fast}"
  )

  in_path = in_dir / f"{case_id}.nii.gz"
  in_path.write_bytes(data)
  case_logger(f"saved input to {in_path}")

  try:
    async with SEGMENT_SEMAPHORE:
      with Timer() as timer:
        output_path = totalseg_liver_only(
          in_path,
          out_dir,
          fast=fast,
          logger=case_logger,
          timeout=MODEL_TIMEOUT_SECONDS,
        )
    log_execution(f"liver:{case_id}", timer.duration)
    case_logger(f"TotalSegmentator duration={timer.duration:.2f}s output={output_path}")
  except Exception as exc:
    case_logger(f"error liver: {exc}")
    raise HTTPException(
      status_code=500,
      detail={"error": f"TotalSegmentator liver failed: {exc}", "case_id": case_id, "request_id": request_id},
    ) from exc

  stable_path = stage_artifact(output_path, SEND_ROOT, f"{case_id}_liver.nii.gz")
  case_logger(f"copied output to {stable_path} size={stable_path.stat().st_size}")
  return FileResponse(
    stable_path,
    media_type="application/gzip",
    filename=stable_path.name,
    headers={"X-Request-ID": request_id, "X-Case-ID": case_id},
  )


@app.post("/segment/totalseg")
async def segment_totalseg(ct: UploadFile = File(...), fast: bool = False):
  case_id = unique_case_id()
  request_id = unique_case_id(prefix="req")
  case_root = WORK_ROOT / case_id
  in_dir = case_root / "in"
  out_dir = case_root / "out"
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)

  case_logger = make_case_logger(case_root)
  data = await ct.read()
  case_logger(
    f"request_id={request_id} task=totalseg filename={ct.filename or 'upload'} bytes={len(data)} fast={fast}"
  )

  in_path = in_dir / f"{case_id}.nii.gz"
  in_path.write_bytes(data)
  case_logger(f"saved input to {in_path}")

  try:
    async with SEGMENT_SEMAPHORE:
      with Timer() as timer:
        output_path = totalseg_multilabel(
          in_path,
          out_dir,
          fast=fast,
          logger=case_logger,
          timeout=MODEL_TIMEOUT_SECONDS,
        )
    log_execution(f"totalseg:{case_id}", timer.duration)
    case_logger(f"TotalSegmentator-ml duration={timer.duration:.2f}s output={output_path}")
  except Exception as exc:
    case_logger(f"error totalseg: {exc}")
    raise HTTPException(
      status_code=500,
      detail={"error": f"TotalSegmentator multi-label failed: {exc}", "case_id": case_id, "request_id": request_id},
    ) from exc

  stable_path = stage_artifact(output_path, SEND_ROOT, f"{case_id}_totalseg.nii.gz")
  case_logger(f"copied output to {stable_path} size={stable_path.stat().st_size}")
  return FileResponse(
    stable_path,
    media_type="application/gzip",
    filename=stable_path.name,
    headers={"X-Request-ID": request_id, "X-Case-ID": case_id},
  )


@app.post("/segment/both")
async def segment_both(
  ct: UploadFile = File(...),
  folds: str = "0",
  fast: bool = True,
):
  case_id = unique_case_id()
  request_id = unique_case_id(prefix="req")
  case_root = WORK_ROOT / case_id
  in_dir = case_root / "in"
  out_dir = case_root / "out"
  in_dir.mkdir(parents=True, exist_ok=True)
  out_dir.mkdir(parents=True, exist_ok=True)

  case_logger = make_case_logger(case_root)
  raw_ct = in_dir / f"{case_id}.nii.gz"
  ct_v1 = in_dir / f"{case_id}_0000.nii.gz"
  data = await ct.read()
  case_logger(
    f"request_id={request_id} task=both filename={ct.filename or 'upload'} bytes={len(data)} fast={fast} folds={folds}"
  )
  raw_ct.write_bytes(data)
  ct_v1.write_bytes(data)
  case_logger(f"saved inputs to {raw_ct} and {ct_v1}")

  liver_dir = out_dir / "totalseg"
  task_dir = out_dir / "task008"
  liver_dir.mkdir(parents=True, exist_ok=True)
  task_dir.mkdir(parents=True, exist_ok=True)

  try:
    async with SEGMENT_SEMAPHORE:
      with Timer() as timer_liver:
        liver_path = totalseg_liver_only(
          raw_ct,
          liver_dir,
          fast=fast,
          logger=case_logger,
          timeout=MODEL_TIMEOUT_SECONDS,
        )
      log_execution(f"liver:{case_id}", timer_liver.duration)
      case_logger(f"liver mask duration={timer_liver.duration:.2f}s output={liver_path}")

      with Timer() as timer_task008:
        task008_path = nnunet_v1_task008(
          in_dir,
          task_dir,
          case_id=case_id,
          folds=folds,
          logger=case_logger,
          timeout=MODEL_TIMEOUT_SECONDS,
        )
    log_execution(f"task008:{case_id}", timer_task008.duration)
    case_logger(f"task008 mask duration={timer_task008.duration:.2f}s output={task008_path}")
  except Exception as exc:
    case_logger(f"error both: {exc}")
    raise HTTPException(
      status_code=500,
      detail={"error": f"Pipeline failed: {exc}", "case_id": case_id, "request_id": request_id},
    ) from exc

  metadata = {
    "case_id": case_id,
    "labels_task008": {"1": "hepatic_vessels", "2": "liver_tumors"},
    "liver_seconds": round(timer_liver.duration, 2),
    "task008_seconds": round(timer_task008.duration, 2),
    "timestamp": time.time(),
  }
  case_logger(f"metadata {metadata}")

  pkg_dir = prepare_package(out_dir, liver_mask=liver_path, task008_mask=task008_path, metadata=metadata)
  archive_path = package_outputs(pkg_dir, base_name=case_id)
  stable_archive = stage_artifact(archive_path, SEND_ROOT, f"{case_id}_results.zip")
  case_logger(f"copied archive to {stable_archive} size={stable_archive.stat().st_size}")
  return FileResponse(
    stable_archive,
    media_type="application/zip",
    filename=stable_archive.name,
    headers={"X-Request-ID": request_id, "X-Case-ID": case_id},
  )


@app.post("/segment/batch")
async def segment_batch(
  bundle: UploadFile = File(...),
  folds: str = "0",
  fast: bool = True,
):
  batch_id = unique_case_id(prefix="batch")
  request_id = unique_case_id(prefix="req")
  batch_root = settings.out_root / batch_id
  batch_root.mkdir(parents=True, exist_ok=True)
  batch_logger = make_case_logger(batch_root)
  batch_logger(
    f"request_id={request_id} task=batch filename={bundle.filename or 'upload'} folds={folds} fast={fast}"
  )
  try:
    case_dirs = extract_archive(bundle, batch_root)
  except ValueError as exc:
    batch_logger(f"error archive: {exc}")
    raise HTTPException(
      status_code=400,
      detail={"error": str(exc), "batch_id": batch_id, "request_id": request_id},
    ) from exc

  if len(case_dirs) > settings.max_batch_cases:
    batch_logger(f"error too many cases: {len(case_dirs)}")
    raise HTTPException(
      status_code=400,
      detail={
        "error": f"Too many cases (>{settings.max_batch_cases})",
        "batch_id": batch_id,
        "request_id": request_id,
      },
    )

  manifest: List[dict] = []

  for case_dir in case_dirs:
    case_id = case_dir.name
    case_logger = make_case_logger(batch_root / case_id)
    case_logger(f"batch case start fast={fast} folds={folds}")
    with temp_case_dirs(case_id) as dirs:
      in_dir, out_dir = dirs["in"], dirs["out"]

      raw_source = case_dir / "raw.nii.gz"
      raw0000_source = case_dir / "raw_0000.nii.gz"
      if not raw_source.exists() and raw0000_source.exists():
        raw_source = raw0000_source
      if not raw_source.exists():
        case_logger("missing raw.nii.gz in input bundle")
        raise HTTPException(status_code=400, detail=f"Case {case_id} missing raw.nii.gz")

      raw_ct = in_dir / f"{case_id}.nii.gz"
      ct_v1 = in_dir / f"{case_id}_0000.nii.gz"
      data = raw_source.read_bytes()
      raw_ct.write_bytes(data)
      ct_v1.write_bytes(data)
      case_logger(f"copied inputs from bundle size={len(data)}")

      liver_dir = out_dir / "totalseg"
      task_dir = out_dir / "task008"
      liver_dir.mkdir(parents=True, exist_ok=True)
      task_dir.mkdir(parents=True, exist_ok=True)

      try:
        async with SEGMENT_SEMAPHORE:
          with Timer() as timer_liver:
            liver_path = totalseg_liver_only(
              raw_ct,
              liver_dir,
              fast=fast,
              logger=case_logger,
              timeout=MODEL_TIMEOUT_SECONDS,
            )
          with Timer() as timer_task008:
            task008_path = nnunet_v1_task008(
              in_dir,
              task_dir,
              case_id=case_id,
              folds=folds,
              logger=case_logger,
              timeout=MODEL_TIMEOUT_SECONDS,
            )
      except Exception as exc:
        case_logger(f"error batch case: {exc}")
        raise HTTPException(
          status_code=500,
          detail={
            "error": f"Batch case {case_id} failed: {exc}",
            "batch_id": batch_id,
            "case_id": case_id,
            "request_id": request_id,
          },
        ) from exc

      metadata = {
        "case_id": case_id,
        "labels_task008": {"1": "hepatic_vessels", "2": "liver_tumors"},
        "liver_seconds": round(timer_liver.duration, 2),
        "task008_seconds": round(timer_task008.duration, 2),
        "timestamp": time.time(),
      }
      case_logger(f"metadata {metadata}")

      pkg_dir = prepare_package(out_dir, liver_mask=liver_path, task008_mask=task008_path, metadata=metadata)
      dest_dir = batch_root / case_id
      if dest_dir.exists():
        shutil.rmtree(dest_dir)
      shutil.copytree(pkg_dir, dest_dir)
      case_logger(f"copied package to {dest_dir}")
      manifest.append(metadata)

  manifest_path = batch_root / "manifest.json"
  write_metadata(manifest_path, {"batch_id": batch_id, "cases": manifest})

  consolidated = package_outputs(batch_root, base_name=batch_id)
  stable_archive = stage_artifact(consolidated, SEND_ROOT, f"{batch_id}_batch.zip")
  batch_logger(f"copied archive to {stable_archive} size={stable_archive.stat().st_size}")
  return FileResponse(
    stable_archive,
    media_type="application/zip",
    filename=stable_archive.name,
    headers={"X-Request-ID": request_id, "X-Batch-ID": batch_id},
  )
