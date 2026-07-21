"""Deploy Expert: package production model release bundle and update registry."""

from __future__ import annotations

from typing import Any
from factory.workers.base_worker import MiniAdapterWorker, module_main


class DeployWorker(MiniAdapterWorker):
    mini_worker_id = "W-DEPLOY"
    needs_gpu = False

    def execute(self, *, dry_run: bool = False) -> tuple[bool, list[str], dict[str, Any], str]:
        # ensure force=True by default for automated factory pipeline deployments
        self.task_args.setdefault("force", True)
        return super().execute(dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(module_main(DeployWorker))
