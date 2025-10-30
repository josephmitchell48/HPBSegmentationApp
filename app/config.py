from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
  app_name: str = Field(default="HPB Segmentation API", alias="APP_NAME")
  in_root: Path = Field(default=Path("/tmp/hpb_in"), alias="HPB_IN_ROOT")
  out_root: Path = Field(default=Path("/tmp/hpb_out"), alias="HPB_OUT_ROOT")
  keep_intermediate: bool = Field(default=False, alias="HPB_KEEP_INTERMEDIATE")
  aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
  s3_bucket: Optional[str] = Field(default=None, alias="HPB_S3_BUCKET")
  max_batch_cases: int = Field(default=10, alias="HPB_MAX_BATCH")

  class Config:
    populate_by_name = True
    extra = "ignore"


@lru_cache
def get_settings() -> Settings:
  settings = Settings()
  settings.in_root.mkdir(parents=True, exist_ok=True)
  settings.out_root.mkdir(parents=True, exist_ok=True)
  return settings
