"""Knowledge Expert — prod build KG with 15,200 triples."""
from factory.workers.base_worker import MiniAdapterWorker, module_main
class KnowledgeWorker(MiniAdapterWorker):
    mini_worker_id = "W-KGBUILD"
    def execute(self, *, dry_run: bool):
        self.task_args.setdefault("config_path", "configs/config_v2_12M_fixed.json")
        return super().execute(dry_run=dry_run)
if __name__ == "__main__":
    raise SystemExit(module_main(KnowledgeWorker))
