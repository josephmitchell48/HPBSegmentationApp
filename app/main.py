from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .config import get_settings
from .runners import nnunet_v1_task008, prepare_package, totalseg_liver_only, totalseg_multilabel
from .utils import (
  Timer,
  extract_archive,
  log_execution,
  package_outputs,
  temp_case_dirs,
  unique_case_id,
  write_metadata,
)

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/healthz")
def health() -> PlainTextResponse:
  return PlainTextResponse("ok")


@app.get("/version")
def version() -> JSONResponse:
  info = {}
  try:
    import torch  # type: ignore

    info["torch"] = torch.__version__
    info["cuda"] = torch.cuda.is_available()
  except Exception as exc:  # pragma: no cover - informational only
    info["torch_error"] = str(exc)
  try:
    import nnunet  # type: ignore

    info["nnunet_v1"] = getattr(nnunet, "__version__", "unknown")
  except Exception as exc:
    info["nnunet_v1_error"] = str(exc)
  try:
    import totalsegmentator as ts  # type: ignore

    info["totalseg"] = getattr(ts, "__version__", "unknown")
  except Exception as exc:
    info["totalseg_error"] = str(exc)
  return JSONResponse(info)


@app.post("/segment/task008")
async def segment_task008(ct: UploadFile = File(...), folds: str = "0"):
  case_id = unique_case_id()
  with temp_case_dirs(case_id) as dirs:
    in_dir, out_dir = dirs["in"], dirs["out"]
    in_path = in_dir / f"{case_id}_0000.nii.gz"
    in_path.write_bytes(await ct.read())

    try:
      with Timer() as timer:
        output_path = nnunet_v1_task008(in_dir, out_dir, case_id=case_id, folds=folds)
      log_execution(f"task008:{case_id}", timer.duration)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"Task008 failed: {exc}") from exc

    return FileResponse(output_path, media_type="application/gzip", filename=f"{case_id}_task008.nii.gz")


@app.post("/segment/liver")
async def segment_liver(ct: UploadFile = File(...), fast: bool = False):
  case_id = unique_case_id()
  with temp_case_dirs(case_id) as dirs:
    in_dir, out_dir = dirs["in"], dirs["out"]
    in_path = in_dir / f"{case_id}.nii.gz"
    in_path.write_bytes(await ct.read())

    try:
      with Timer() as timer:
        output_path = totalseg_liver_only(in_path, out_dir, fast=fast)
      log_execution(f"liver:{case_id}", timer.duration)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"TotalSegmentator liver failed: {exc}") from exc

    return FileResponse(output_path, media_type="application/gzip", filename=f"{case_id}_liver.nii.gz")


@app.post("/segment/totalseg")
async def segment_totalseg(ct: UploadFile = File(...), fast: bool = False):
  case_id = unique_case_id()
  with temp_case_dirs(case_id) as dirs:
    in_dir, out_dir = dirs["in"], dirs["out"]
    in_path = in_dir / f"{case_id}.nii.gz"
    in_path.write_bytes(await ct.read())

    try:
      with Timer() as timer:
        output_path = totalseg_multilabel(in_path, out_dir, fast=fast)
      log_execution(f"totalseg:{case_id}", timer.duration)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"TotalSegmentator multi-label failed: {exc}") from exc

    return FileResponse(output_path, media_type="application/gzip", filename=f"{case_id}_totalseg.nii.gz")


@app.post("/segment/both")
async def segment_both(
  ct: UploadFile = File(...),
  folds: str = "0",
  fast: bool = True,
  background_tasks: BackgroundTasks | None = None,
):
  case_id = unique_case_id()
  with temp_case_dirs(case_id) as dirs:
    in_dir, out_dir = dirs["in"], dirs["out"]
    case_root = out_dir

    raw_ct = in_dir / f"{case_id}.nii.gz"
    ct_v1 = in_dir / f"{case_id}_0000.nii.gz"
    data = await ct.read()
    raw_ct.write_bytes(data)
    ct_v1.write_bytes(data)

    liver_dir = case_root / "totalseg"
    task_dir = case_root / "task008"
    liver_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
      with Timer() as timer_liver:
        liver_path = totalseg_liver_only(raw_ct, liver_dir, fast=fast)
      log_execution(f"liver:{case_id}", timer_liver.duration)

      with Timer() as timer_task008:
        task008_path = nnunet_v1_task008(in_dir, task_dir, case_id=case_id, folds=folds)
      log_execution(f"task008:{case_id}", timer_task008.duration)
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    metadata = {
      "case_id": case_id,
      "labels_task008": {"1": "hepatic_vessels", "2": "liver_tumors"},
      "liver_seconds": round(timer_liver.duration, 2),
      "task008_seconds": round(timer_task008.duration, 2),
      "timestamp": time.time(),
    }

    pkg_dir = prepare_package(case_root, liver_mask=liver_path, task008_mask=task008_path, metadata=metadata)
    archive_path = package_outputs(pkg_dir, base_name=case_id)

    if background_tasks:
      background_tasks.add_task(shutil.rmtree, pkg_dir.parent, ignore_errors=True)

    return FileResponse(archive_path, media_type="application/zip", filename=f"{case_id}_results.zip")


@app.post("/segment/batch")
async def segment_batch(
  bundle: UploadFile = File(...),
  folds: str = "0",
  fast: bool = True,
):
  batch_id = unique_case_id(prefix="batch")
  batch_root = settings.out_root / batch_id
  batch_root.mkdir(parents=True, exist_ok=True)
  try:
    case_dirs = extract_archive(bundle, batch_root)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  if len(case_dirs) > settings.max_batch_cases:
    raise HTTPException(status_code=400, detail=f"Too many cases (>{settings.max_batch_cases})")

  manifest: List[dict] = []

  for case_dir in case_dirs:
    case_id = case_dir.name
    with temp_case_dirs(case_id) as dirs:
      in_dir, out_dir = dirs["in"], dirs["out"]

      raw_source = case_dir / "raw.nii.gz"
      raw0000_source = case_dir / "raw_0000.nii.gz"
      if not raw_source.exists() and raw0000_source.exists():
        raw_source = raw0000_source
      if not raw_source.exists():
        raise HTTPException(status_code=400, detail=f"Case {case_id} missing raw.nii.gz")

      raw_ct = in_dir / f"{case_id}.nii.gz"
      ct_v1 = in_dir / f"{case_id}_0000.nii.gz"
      data = raw_source.read_bytes()
      raw_ct.write_bytes(data)
      ct_v1.write_bytes(data)

      liver_dir = out_dir / "totalseg"
      task_dir = out_dir / "task008"
      liver_dir.mkdir(parents=True, exist_ok=True)
      task_dir.mkdir(parents=True, exist_ok=True)

      try:
        with Timer() as timer_liver:
          liver_path = totalseg_liver_only(raw_ct, liver_dir, fast=fast)
        with Timer() as timer_task008:
          task008_path = nnunet_v1_task008(in_dir, task_dir, case_id=case_id, folds=folds)
      except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch case {case_id} failed: {exc}") from exc

      metadata = {
        "case_id": case_id,
        "labels_task008": {"1": "hepatic_vessels", "2": "liver_tumors"},
        "liver_seconds": round(timer_liver.duration, 2),
        "task008_seconds": round(timer_task008.duration, 2),
        "timestamp": time.time(),
      }

      pkg_dir = prepare_package(out_dir, liver_mask=liver_path, task008_mask=task008_path, metadata=metadata)
      dest_dir = batch_root / case_id
      if dest_dir.exists():
        shutil.rmtree(dest_dir)
      shutil.copytree(pkg_dir, dest_dir)
      manifest.append(metadata)

  manifest_path = batch_root / "manifest.json"
  write_metadata(manifest_path, {"batch_id": batch_id, "cases": manifest})

  consolidated = package_outputs(batch_root, base_name=batch_id)
  return FileResponse(consolidated, media_type="application/zip", filename=f"{batch_id}_batch.zip")
