"""Data Expert — prod-ready, forces re-pack when block_size changes."""
from factory.workers.base_worker import MiniAdapterWorker, module_main
from pathlib import Path

class DataWorker(MiniAdapterWorker):
    pipeline = None
    mini_worker_id = "W-BOOTSTRAP"  # First step, but we map to sprint4 pipeline via task_args

    def execute(self, *, dry_run: bool):
        # If block_size changed (512 vs old 64), force corpus rebuild
        block_size = int(self.task_args.get("block_size") or 512)
        # Pass through
        self.task_args.setdefault("block_size", block_size)
        # Use pipeline sprint4 which does BOOTSTRAP->INGEST->QUALITY->STANDARDIZE
        self.pipeline = "sprint4"
        return super().execute(dry_run=dry_run)

if __name__ == "__main__":
    raise SystemExit(module_main(DataWorker))
