"""Retrieval Expert — BM25 + Dense 1200 docs Recall@5."""
from factory.workers.base_worker import MiniAdapterWorker, module_main
class RetrievalWorker(MiniAdapterWorker):
    mini_worker_id = "W-RAG"
    needs_gpu = False
    def execute(self, *, dry_run: bool):
        self.task_args.setdefault("top_k", 2)
        self.task_args.setdefault("enable_checklist_filter", True)
        return super().execute(dry_run=dry_run)
if __name__ == "__main__":
    raise SystemExit(module_main(RetrievalWorker))
