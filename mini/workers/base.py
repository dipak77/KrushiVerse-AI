"""Base worker contract for automated Mini factory jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from mini.contracts import WorkerResult

WORKER_REGISTRY: dict[str, type["BaseWorker"]] = {}


def register_worker(cls: type["BaseWorker"]) -> type["BaseWorker"]:
    WORKER_REGISTRY[cls.worker_id] = cls
    return cls


def list_workers() -> list[dict[str, Any]]:
    rows = []
    for wid, cls in sorted(WORKER_REGISTRY.items(), key=lambda x: x[0]):
        rows.append(
            {
                "worker_id": wid,
                "name": cls.name,
                "description": cls.description,
                "epic": cls.epic,
                "status": cls.status,
            }
        )
    return rows


def get_worker(worker_id: str) -> "BaseWorker":
    if worker_id not in WORKER_REGISTRY:
        raise KeyError(f"Unknown worker: {worker_id}. Known: {sorted(WORKER_REGISTRY)}")
    return WORKER_REGISTRY[worker_id]()


class BaseWorker(ABC):
    worker_id: str = "W-ABSTRACT"
    name: str = "Abstract Worker"
    description: str = ""
    epic: str = "E0"
    status: str = "stub"  # stub | partial | ready

    def run(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            result = self.execute(dry_run=dry_run, **kwargs)
        except Exception as e:
            return WorkerResult(
                worker_id=self.worker_id,
                ok=False,
                dry_run=dry_run,
                message=f"Worker failed: {e}",
                errors=[str(e)],
                started_at=started,
                finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        result.worker_id = self.worker_id
        result.dry_run = dry_run
        result.started_at = result.started_at or started
        result.finished_at = result.finished_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return result

    @abstractmethod
    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        ...
