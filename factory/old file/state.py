"""Small, dependency-free persistent state helpers for the factory."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON file so monitor reads never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(temporary, path)
            return
        except (PermissionError, OSError):
            time.sleep(0.05)
    try:
        os.replace(temporary, path)
    except Exception:
        pass


@contextmanager
def file_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Cross-process lock based on exclusive file creation (works on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}".encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for state lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class TaskStore:
    """Concurrency-safe reader/updater for a factory task DAG."""

    def __init__(self, factory_dir: str | Path = "factory") -> None:
        self.root = Path(factory_dir)
        self.path = self.root / "TASK_DAG.json"
        self.lock_path = self.root / ".task_dag.lock"

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Factory DAG not initialized: {self.path}. Run `python -m factory.planner init`."
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("tasks"), list):
            raise ValueError(f"Invalid factory DAG (missing tasks list): {self.path}")
        return payload

    def initialize(self, template_path: str | Path, *, overwrite: bool = False) -> Path:
        template = Path(template_path)
        if self.path.exists() and not overwrite:
            return self.path
        payload = json.loads(template.read_text(encoding="utf-8"))
        payload.setdefault("created_at", utc_now())
        atomic_write_json(self.path, payload)
        return self.path

    def update_task(self, task_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with file_lock(self.lock_path):
            dag = self.read()
            for task in dag["tasks"]:
                if task.get("id") == task_id:
                    mutate(task)
                    task["updated_at"] = utc_now()
                    atomic_write_json(self.path, dag)
                    return task
        raise KeyError(f"Unknown factory task: {task_id}")

    def task(self, task_id: str) -> dict[str, Any]:
        return next(task for task in self.read()["tasks"] if task.get("id") == task_id)

    def ready_tasks(self) -> list[dict[str, Any]]:
        tasks = self.read()["tasks"]
        completed = {task.get("id") for task in tasks if task.get("status") == "COMPLETED"}
        ready = [
            task
            for task in tasks
            if task.get("status") == "PENDING" and set(task.get("deps") or []).issubset(completed)
        ]
        return sorted(ready, key=lambda task: (-int(task.get("priority") or 0), str(task.get("id"))))

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.read()["tasks"]:
            status = str(task.get("status") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
        return counts
