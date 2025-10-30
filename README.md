# HPB Segmentation API

FastAPI microservice for hepatopancreatobiliary (HPB) imaging segmentation. The service exposes endpoints that run nnU-Net Task008 and TotalSegmentator models on uploaded CT scans, generating liver, vascular, and tumor masks suitable for the LearnHPB viewer.

## Highlights

- `POST /segment/task008` – nnU-Net v1 Task008 hepatic vessel + tumor model
- `POST /segment/liver` – TotalSegmentator liver-only ROI
- `POST /segment/totalseg` – TotalSegmentator multi-label (optional)
- `POST /segment/both` – Runs both pipelines and returns a packaged ZIP (liver + task008 + metadata)
- `POST /segment/batch` – Accepts a tar/zip archive with multiple NIfTI cases and processes them sequentially
- Health/version endpoints for ops visibility
- Disk cleanup hooks and throttled threading env vars for deterministic EC2 performance

## Directory Layout

```
API app/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment + path management
│   ├── main.py                # FastAPI application + routes
│   ├── runners.py             # nnUNet + TotalSegmentator helpers
│   └── utils.py               # Common helpers (subprocess, temp dirs, packaging)
├── requirements.txt
├── Dockerfile
├── scripts/
│   ├── bootstrap.sh           # Install models + service dependencies on EC2
│   ├── submit_batch.py        # Example client for /segment/batch
│   └── systemd-service-example.service
└── README.md
```

## Quick Start (Local GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Export model paths (adjust for local checkout)
export RESULTS_FOLDER=/models/nnunet_v1
export TOTALSEGMENTATOR_FORCE_CUDA=1

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Upload a CT NIfTI file:

```bash
curl -X POST \
  -F "ct=@/path/to/case_0000.nii.gz" \
  http://localhost:8080/segment/both --output results.zip
```

## EC2 Deployment Notes

1. Launch a GPU instance (e.g., `g4dn.xlarge`) using an Ubuntu 22.04 AMI.
2. Copy this directory to the instance and run `scripts/bootstrap.sh` (installs CUDA libs, TotalSegmentator, nnU-Net weights placeholder, creates `/models`).
3. Start the API with Uvicorn or Gunicorn; example `systemd` unit in `scripts/systemd-service-example.service`.
4. Point the LearnHPB ingestion pipeline at the `/segment/*` endpoints.

## Batch Processing

`POST /segment/batch` accepts a `.zip` or `.tar.gz` containing one subdirectory per case. Each subdirectory must contain either `raw.nii.gz` or both `raw.nii.gz` and `raw_0000.nii.gz`. The server writes segmentation outputs to `/tmp/out/<case>/package` and streams a consolidated ZIP back. See `scripts/submit_batch.py` for an end-to-end example that also uploads results to S3.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HPB_IN_ROOT` | `/tmp/hpb_in` | Input scratch directory |
| `HPB_OUT_ROOT` | `/tmp/hpb_out` | Output scratch directory |
| `RESULTS_FOLDER` | *(nnUNet default)* | Location of nnU-Net v1 checkpoints |
| `AWS_REGION` | `us-east-1` | Used by `scripts/submit_batch.py` if uploading to S3 |
| `HPB_S3_BUCKET` | *(unset)* | Optional S3 bucket for results |

## Model Assets

- nnU-Net Task008 weights should be placed under `$RESULTS_FOLDER/Task008_HepaticVessel/nnUNetTrainerV2__nnUNetPlansv2.1/`.
- TotalSegmentator automatically downloads its checkpoints on first use; cache them in `/models/totalseg` for offline use.

## Next Steps

- Hook into AWS Batch/Step Functions for high-volume processing.
- Publish run metrics to CloudWatch (`app.utils.log_execution`).
- Extend metadata to include HU ranges, derived mesh statistics, etc.
