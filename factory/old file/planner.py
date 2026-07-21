"""Dependency-aware, one-GPU-at-a-time planner for the v3 factory."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from factory.gpu_manager import GPUManager
from factory.monitor import write_status
from factory.state import TaskStore, utc_now


WORKER_MODULES = {
    "data": "data_worker",
    "tokenizer": "tokenizer_worker",
    "knowledge": "knowledge_worker",
    "training": "training_worker",
    "synthetic": "synthetic_worker",
    "retrieval": "retrieval_worker",
    "eval": "eval_worker",
    "safety": "safety_worker",
    "quant": "quant_worker",
    "tester": "tester_worker",
    "deploy": "deploy_worker",
}


def pid_is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        # os.kill(pid, 0) is supported by modern Windows Python and POSIX.
        try:
            os.kill(pid, 0)
            return True
        except (OSError, PermissionError):
            return False


class FactoryPlanner:
    def __init__(self, factory_dir: str | Path = "factory", *, max_cpu_workers: int = 4) -> None:
        self.root = Path(factory_dir)
        self.store = TaskStore(self.root)
        self.gpu = GPUManager(self.root)
        self.max_cpu_workers = max(1, max_cpu_workers)

    def recover_finished_processes(self) -> list[str]:
        """Mark orphaned RUNNING processes failed; completed workers update their own state."""
        recovered: list[str] = []
        for task in self.store.read()["tasks"]:
            if task.get("status") != "RUNNING" or pid_is_running(task.get("pid")):
                continue

            task_id = str(task["id"])
            if task.get("gpu"):
                self.gpu.release(task_id)

            def mutate(current: dict[str, Any]) -> None:
                if current.get("status") == "RUNNING":
                    current["status"] = "FAILED"
                    current["finished_at"] = utc_now()
                    current["error"] = "Worker process exited before marking task complete"

            self.store.update_task(task_id, mutate)
            recovered.append(task_id)
        return recovered

    def build_command(self, task: dict[str, Any], *, dry_run: bool) -> list[str]:
        worker_name = str(task.get("worker"))
        module = WORKER_MODULES.get(worker_name)
        if not module:
            raise ValueError(f"No implemented factory adapter for worker '{worker_name}'")
        
        python_bin = sys.executable
        repo_venv_python = Path(__file__).resolve().parent.parent / "venv" / "Scripts" / "python.exe"
        if repo_venv_python.exists():
            python_bin = str(repo_venv_python)

        command = [
            python_bin,
            "-m",
            f"factory.workers.{module}",
            "--task-id",
            str(task["id"]),
            "--factory-dir",
            str(self.root),
        ]
        if task.get("gpu"):
            command.append("--gpu-reserved")
        if dry_run:
            command.append("--dry-run")
        return command

    def _cpu_slots(self) -> int:
        running_cpu = sum(
            1
            for task in self.store.read()["tasks"]
            if task.get("status") == "RUNNING" and not task.get("gpu")
        )
        return max(0, self.max_cpu_workers - running_cpu)

    def launch(self, task: dict[str, Any], *, dry_run: bool = False) -> int | None:
        task_id = str(task["id"])
        if task.get("gpu") and not self.gpu.acquire(task_id):
            return None
        # Mark RUNNING before spawning. A fast worker may complete before Popen
        # returns; setting the state first prevents a later planner write from
        # accidentally reverting COMPLETED back to RUNNING.
        def reserve(current: dict[str, Any]) -> None:
            current["status"] = "RUNNING"
            current["started_at"] = utc_now()
            current["attempts"] = int(current.get("attempts") or 0) + 1
            current.pop("error", None)

        self.store.update_task(task_id, reserve)
        try:
            command = self.build_command(task, dry_run=dry_run)
            process = subprocess.Popen(command, cwd=Path.cwd())
        except Exception:
            if task.get("gpu"):
                self.gpu.release(task_id)
            self.store.update_task(
                task_id,
                lambda current: current.update(status="FAILED", error="Planner could not start worker"),
            )
            raise

        def mutate(current: dict[str, Any]) -> None:
            if current.get("status") == "RUNNING":
                current["pid"] = process.pid

        self.store.update_task(task_id, mutate)
        return process.pid

    def run_once(self, *, execute: bool = False, dry_run: bool = False) -> dict[str, Any]:
        recovered = self.recover_finished_processes()
        ready = self.store.ready_tasks()
        launched: list[str] = []
        deferred: list[str] = []
        cpu_slots = self._cpu_slots()
        for task in ready:
            if not execute:
                continue
            if not task.get("gpu") and cpu_slots <= 0:
                deferred.append(str(task["id"]))
                continue
            try:
                pid = self.launch(task, dry_run=dry_run)
            except Exception as exc:
                task_id = str(task["id"])
                self.store.update_task(
                    task_id,
                    lambda current: current.update(status="FAILED", error=f"Planner launch failed: {exc}"),
                )
                continue
            if pid is None:
                deferred.append(str(task["id"]))
                continue
            launched.append(str(task["id"]))
            if not task.get("gpu"):
                cpu_slots -= 1
        status = write_status(self.root)
        return {
            "ready": [str(task["id"]) for task in ready],
            "launched": launched,
            "deferred": deferred,
            "recovered": recovered,
            "summary": status["summary"],
        }


def template_path() -> Path:
    return Path(__file__).with_name("TASK_DAG.template.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KrushiVerseAI v3 autonomous factory planner")
    parser.add_argument("command", choices=["init", "status", "run"], nargs="?", default="run")
    parser.add_argument("--factory-dir", default="factory")
    parser.add_argument("--auto", action="store_true", help="Keep scheduling until interrupted")
    parser.add_argument("--execute", action="store_true", help="Actually launch workers (default only reports state)")
    parser.add_argument("--dry-run", action="store_true", help="Launch Mini workers in dry-run mode")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite TASK_DAG.json when initializing")
    args = parser.parse_args(argv)
    planner = FactoryPlanner(args.factory_dir, max_cpu_workers=args.max_cpu_workers)
    if args.command == "init":
        path = planner.store.initialize(template_path(), overwrite=args.overwrite)
        print(path)
        return 0
    if args.command == "status":
        import json

        print(json.dumps(write_status(args.factory_dir), indent=2))
        return 0
    while True:
        result = planner.run_once(execute=args.execute, dry_run=args.dry_run)
        print(result)
        if not args.auto:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
