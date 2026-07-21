"""GPU mutex for single 4GB RTX 2050 — prod-ready with stale auto-recovery."""
from __future__ import annotations
import json, os, socket, time
from pathlib import Path
from typing import Any
from factory.state import atomic_write_json

class GPUManager:
    def __init__(self, factory_dir: str | Path = "factory", *, min_free_bytes: int = 1 * 1024**3, require_cuda: bool = True):
        self.root = Path(factory_dir)
        self.lock_path = self.root / "gpu.lock"
        self.min_free_bytes = min_free_bytes
        self.require_cuda = require_cuda

    @staticmethod
    def _cuda_memory() -> tuple[bool, int | None, int | None]:
        try:
            import torch
            if not torch.cuda.is_available():
                return False, None, None
            free, total = torch.cuda.mem_get_info()
            return True, int(free), int(total)
        except Exception:
            return False, None, None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            import psutil
            return psutil.pid_exists(pid)
        except:
            try:
                os.kill(pid, 0)
                return True
            except:
                return False

    def availability(self) -> dict[str, Any]:
        cuda, free, total = self._cuda_memory()
        return {"cuda_available": cuda, "free_bytes": free, "total_bytes": total, "meets_min_free_memory": bool(free is None or free >= self.min_free_bytes)}

    def status(self) -> dict[str, Any]:
        if not self.lock_path.exists():
            return {"state": "FREE", **self.availability()}
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"holder": "unknown", "corrupt": True}
        return {"state": "LOCKED", **payload, **self.availability()}

    def acquire(self, task_id: str) -> bool:
        # Auto-recover stale lock before acquiring
        self.recover_stale(max_age_seconds=3600)
        availability = self.availability()
        if self.require_cuda and not availability["cuda_available"]:
            return False
        if availability["cuda_available"] and not availability["meets_min_free_memory"]:
            # Still allow if no real holder (free check may be fragmented)
            if self.lock_path.exists():
                return False
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"holder": task_id, "pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": time.time()}
        try:
            with self.lock_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return True
        except FileExistsError:
            return False

    def held_by(self, task_id: str) -> bool:
        return self.status().get("holder") == task_id

    def release(self, task_id: str, *, force: bool = False) -> bool:
        if not self.lock_path.exists():
            return False
        if not force and not self.held_by(task_id):
            return False
        try:
            self.lock_path.unlink()
            return True
        except FileNotFoundError:
            return False

    def recover_stale(self, *, max_age_seconds: float = 3600) -> bool:
        status = self.status()
        acquired = status.get("acquired_at")
        holder_pid = status.get("pid")
        if status.get("state") != "LOCKED":
            return False
        # If PID dead, recover immediately
        if isinstance(holder_pid, int) and not self._pid_alive(holder_pid):
            return self.release(str(status.get("holder") or ""), force=True)
        # If too old, recover
        if isinstance(acquired, (int, float)) and time.time() - acquired > max_age_seconds:
            return self.release(str(status.get("holder") or ""), force=True)
        return False
