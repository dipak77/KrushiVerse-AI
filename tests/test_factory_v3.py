"""Focused contract tests for the v3 autonomous factory foundation."""

from __future__ import annotations

import json

from factory.gpu_manager import GPUManager
from factory.planner import FactoryPlanner
from factory.state import TaskStore


def _write_dag(tmp_path, tasks):
    (tmp_path / "TASK_DAG.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")


def test_task_store_selects_only_dependency_ready_tasks_in_priority_order(tmp_path):
    _write_dag(
        tmp_path,
        [
            {"id": "complete", "status": "COMPLETED", "deps": [], "priority": 1},
            {"id": "low", "status": "PENDING", "deps": ["complete"], "priority": 1},
            {"id": "high", "status": "PENDING", "deps": ["complete"], "priority": 9},
            {"id": "blocked", "status": "PENDING", "deps": ["missing"], "priority": 10},
        ],
    )
    store = TaskStore(tmp_path)
    assert [task["id"] for task in store.ready_tasks()] == ["high", "low"]
    store.update_task("high", lambda task: task.update(status="COMPLETED", artifacts=["result.json"]))
    assert store.task("high")["artifacts"] == ["result.json"]


def test_gpu_manager_is_single_holder_and_owner_only_release(tmp_path):
    gpu = GPUManager(tmp_path, require_cuda=False, min_free_bytes=0)
    assert gpu.acquire("train-a") is True
    assert gpu.acquire("train-b") is False
    assert gpu.release("train-b") is False
    assert gpu.held_by("train-a") is True
    assert gpu.release("train-a") is True
    assert gpu.status()["state"] == "FREE"


def test_planner_preview_does_not_launch_or_mutate_ready_task(tmp_path):
    _write_dag(
        tmp_path,
        [
            {
                "id": "data_v2",
                "worker": "data",
                "status": "PENDING",
                "deps": [],
                "gpu": False,
                "priority": 10,
            }
        ],
    )
    planner = FactoryPlanner(tmp_path)
    result = planner.run_once(execute=False)
    assert result["ready"] == ["data_v2"]
    assert result["launched"] == []
    assert TaskStore(tmp_path).task("data_v2")["status"] == "PENDING"
    command = planner.build_command(TaskStore(tmp_path).task("data_v2"), dry_run=True)
    assert command[-1] == "--dry-run"
    assert "factory.workers.data_worker" in command
