"""Shared data contracts for all Mini workers (Schema v1)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mini.paths import SCHEMA_VERSION


class LanguageCode(str, Enum):
    EN = "en"
    MR = "mr"
    HI = "hi"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DataSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"
    UNASSIGNED = "unassigned"


class Category(str, Enum):
    SOIL = "soil"
    WEATHER = "weather"
    CROP = "crop"
    DISEASE = "disease"
    PEST = "pest"
    FERTILIZER = "fertilizer"
    IRRIGATION = "irrigation"
    SCHEME = "scheme"
    MARKET = "market"
    FINANCE = "finance"
    MACHINERY = "machinery"
    GENERAL = "general"
    SEED = "seed"
    ADVISORY = "advisory"


class Region(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    zone: Optional[str] = None
    country: str = "India"


class StandardRecord(BaseModel):
    """Canonical training/eval record — all workers that emit QA must use this."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    category: Category
    subcategory: Optional[str] = None
    crop: Optional[str] = None
    region: Optional[Region] = None
    language: LanguageCode = LanguageCode.EN
    question: str
    answer: str
    source: str
    source_url: Optional[str] = None
    verified: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    license: str = "educational-open-compilation"
    split: DataSplit = DataSplit.UNASSIGNED
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "answer")
    @classmethod
    def non_empty_text(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("question/answer must be non-empty")
        return v

    def to_training_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class WorkerResult(BaseModel):
    """Standard result envelope returned by every worker run."""

    worker_id: str
    ok: bool
    dry_run: bool = False
    message: str = ""
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class PipelineResult(BaseModel):
    pipeline: str
    ok: bool
    dry_run: bool = False
    run_id: str
    steps: list[WorkerResult] = Field(default_factory=list)
    message: str = ""


class ManifestEntry(BaseModel):
    path: str
    sha256: Optional[str] = None
    bytes: Optional[int] = None
    domain: Optional[str] = None
    source: Optional[str] = None


class BatchManifest(BaseModel):
    """Ingest/process batch manifest written beside lake artifacts."""

    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    worker_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    schema_version: str = SCHEMA_VERSION
    entries: list[ManifestEntry] = Field(default_factory=list)
    notes: str = ""
