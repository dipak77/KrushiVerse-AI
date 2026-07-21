"""Eval Expert: benchmark model performance, accuracy, grounding, F1 score."""

from factory.workers.base_worker import MiniAdapterWorker, module_main


class EvalWorker(MiniAdapterWorker):
    mini_worker_id = "W-EVAL"
    needs_gpu = True


if __name__ == "__main__":
    raise SystemExit(module_main(EvalWorker))
