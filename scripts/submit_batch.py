#!/usr/bin/env python3
"""Example client for the /segment/batch endpoint."""

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path

import requests


def build_archive(cases_dir: Path) -> bytes:
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for case_dir in cases_dir.iterdir():
      if not case_dir.is_dir():
        continue
      for file_path in case_dir.glob("*.nii.gz"):
        arcname = f"{case_dir.name}/{file_path.name}"
        zf.write(file_path, arcname)
  buf.seek(0)
  return buf.read()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("cases", type=Path, help="Directory with case subfolders")
  parser.add_argument("--endpoint", default="http://localhost:8080/segment/batch")
  parser.add_argument("--fast", action="store_true", help="Enable TotalSegmentator fast mode")
  args = parser.parse_args()

  archive_bytes = build_archive(args.cases)
  files = {"bundle": ("batch.zip", archive_bytes, "application/zip")}
  data = {"fast": str(args.fast).lower()}
  response = requests.post(args.endpoint, files=files, data=data, timeout=3600)
  response.raise_for_status()

  output_path = Path("batch_results.zip")
  output_path.write_bytes(response.content)
  print(f"Results saved to {output_path.resolve()}")


if __name__ == "__main__":
  main()
