"""Quant Expert — INT8 + ONNX export, real latency bench."""
from factory.workers.base_worker import MiniAdapterWorker, module_main
class QuantWorker(MiniAdapterWorker):
    mini_worker_id = "W-QUANT"
    needs_gpu = False
    def execute(self, *, dry_run: bool):
        self.task_args.setdefault("int8_budget_bytes", 4*1024*1024)
        self.task_args.setdefault("include_int4", True)
        return super().execute(dry_run=dry_run)
if __name__ == "__main__":
    raise SystemExit(module_main(QuantWorker))
