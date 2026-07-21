"""Synthetic Expert — prod 25k QA with dedup."""
from factory.workers.base_worker import MiniAdapterWorker, module_main
class SyntheticWorker(MiniAdapterWorker):
    mini_worker_id = "W-QASYNTH"
    needs_gpu = True
    def execute(self, *, dry_run: bool):
        self.task_args.setdefault("target_qa", 25000)
        self.task_args.setdefault("dedup_threshold", 0.92)
        return super().execute(dry_run=dry_run)
if __name__ == "__main__":
    raise SystemExit(module_main(SyntheticWorker))
