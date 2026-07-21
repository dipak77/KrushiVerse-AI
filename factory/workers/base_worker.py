"""Base class — prod-ready v2 with config upgrade support, real heartbeat, PROGRESS.json.

Fixes vs original:
- Writes PROGRESS.json for monitor.py (step/pct/tokens_per_sec) — was missing, caused 5% stuck bug
- Passes task_args (config_path, block_size, resume) to mini workers — enables new 12M config upgrade
- Heartbeat robust with try/except, never crashes worker
- GPU release guaranteed in finally
- _fail includes traceback + artifacts
"""
from __future__ import annotations
import argparse, json, traceback, os, time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from factory.gpu_manager import GPUManager
from factory.state import TaskStore, atomic_write_json, utc_now

class FactoryWorker(ABC):
    needs_gpu = False

    def __init__(self, task_id: str, *, factory_dir: str | Path = "factory", gpu_reserved: bool = False) -> None:
        self.task_id = task_id
        self.factory_dir = Path(factory_dir)
        self.store = TaskStore(self.factory_dir)
        self.gpu = GPUManager(self.factory_dir)
        self.gpu_reserved = gpu_reserved
        self.task = self.store.task(task_id)
        self.task_args = dict(self.task.get("task_args") or {})

    def heartbeat(self, *, step: int | None = None, loss: float | None = None, pct: float | None = None, tokens_per_sec: float | None = None, message: str = "") -> None:
        try:
            payload = {
                "task_id": self.task_id,
                "worker": self.task.get("worker"),
                "step": step,
                "loss": loss,
                "progress_pct": pct,
                "tokens_per_sec": tokens_per_sec,
                "message": message,
                "time": utc_now(),
            }
            # Remove None to keep file small
            payload = {k:v for k,v in payload.items() if v is not None}
            atomic_write_json(self.factory_dir / "heartbeats" / f"{self.task_id}.json", payload)
            # Also update PROGRESS.json for monitor.py pretrain live tracking
            if self.task_id == "pretrain_10k" and step is not None:
                prog_path = Path.cwd() / "mini" / "models" / "v0.6-base" / "PROGRESS.json"
                prog_path.parent.mkdir(parents=True, exist_ok=True)
                total = int(self.task_args.get("max_steps") or 10000)
                pct_calc = round(100.0 * step / max(1,total), 1)
                prog = {
                    "step": step,
                    "steps": total,
                    "pct": pct_calc,
                    "train_loss": loss,
                    "val_ppl": 0,
                    "tokens_per_sec": tokens_per_sec or 0,
                    "eta_human": message,
                    "updated_at": utc_now()
                }
                try:
                    atomic_write_json(prog_path, prog)
                except Exception:
                    pass
        except Exception:
            pass  # heartbeat must never crash worker

    def _complete(self, artifacts: list[str], metrics: dict[str, Any]) -> None:
        def mutate(task: dict[str, Any]) -> None:
            task["status"] = "COMPLETED"
            task["finished_at"] = utc_now()
            task["artifacts"] = artifacts
            task["metrics"] = metrics
            task.pop("error", None)
        self.store.update_task(self.task_id, mutate)
        self.heartbeat(message="completed", pct=100.0)

    def _fail(self, error: str) -> None:
        def mutate(task: dict[str, Any]) -> None:
            task["status"] = "FAILED"
            task["finished_at"] = utc_now()
            task["error"] = error[:5000]
        self.store.update_task(self.task_id, mutate)
        self.heartbeat(message=f"failed: {error[:200]}")

    @abstractmethod
    def execute(self, *, dry_run: bool) -> tuple[bool, list[str], dict[str, Any], str]:
        """Return (ok, artifacts, metrics, message)."""

    def run(self, *, dry_run: bool = False) -> int:
        self.heartbeat(message="started")
        if self.needs_gpu and not self.gpu_reserved:
            if not self.gpu.acquire(self.task_id):
                self._fail("GPU reservation unavailable; planner must reserve GPU before launch")
                return 1
        if self.needs_gpu and not self.gpu.held_by(self.task_id):
            self._fail("GPU lock not held by this task")
            return 1
        try:
            ok, artifacts, metrics, message = self.execute(dry_run=dry_run)
            if ok:
                self._complete(artifacts, metrics)
                return 0
            self._fail(message)
            return 1
        except Exception as exc:
            self._fail(f"{exc}\n{traceback.format_exc(limit=10)}")
            return 1
        finally:
            if self.needs_gpu:
                try:
                    self.gpu.release(self.task_id)
                except Exception:
                    pass


class MiniAdapterWorker(FactoryWorker):
    """Adapter to existing Mini workers — now passes config_path, block_size, resume."""

    mini_worker_id: str | None = None
    pipeline: str | None = None

    def execute(self, *, dry_run: bool) -> tuple[bool, list[str], dict[str, Any], str]:
        if self.pipeline:
            from mini.orchestrator.dag import run_pipeline
            result = run_pipeline(self.pipeline, dry_run=dry_run)
            metrics = result.model_dump()
            artifacts = [artifact for step in result.steps for artifact in step.artifacts]
            return result.ok, artifacts, metrics, result.message

        if not self.mini_worker_id:
            raise RuntimeError("Factory adapter needs mini_worker_id or pipeline")

        # Pass through task_args including new config upgrade keys
        from mini.workers.base import get_worker
        # Ensure config_path default to fixed 12M config if not provided
        if "config_path" not in self.task_args and self.mini_worker_id in ("W-PRETRAIN", "W-TOKEN"):
            # Prefer fixed config if exists
            for cand in ["configs/config_v2_12M_fixed.json", "models/config_v2_15M.json", "models/config_v2_15M_FIXED.json"]:
                if Path(cand).exists():
                    self.task_args["config_path"] = cand
                    break

        result = get_worker(self.mini_worker_id).run(dry_run=dry_run, **self.task_args)
        return result.ok, result.artifacts, result.metrics, result.message


def module_main(worker_type: type[FactoryWorker]) -> int:
    parser = argparse.ArgumentParser(description=f"Run {worker_type.__name__} under factory")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--factory-dir", default="factory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-reserved", action="store_true")
    args = parser.parse_args()
    return worker_type(args.task_id, factory_dir=args.factory_dir, gpu_reserved=args.gpu_reserved).run(dry_run=args.dry_run)
