"""Tokenizer adapter — prod-ready with 8k vocab + new config support."""
from factory.workers.base_worker import MiniAdapterWorker, module_main

class TokenizerWorker(MiniAdapterWorker):
    mini_worker_id = "W-TOKEN"
    # Ensure config_path passed through for vocab_size 8192
    def execute(self, *, dry_run: bool):
        # Force vocab 8192 for prod
        self.task_args.setdefault("vocab_size", 8192)
        self.task_args.setdefault("config_path", "configs/config_v2_12M_fixed.json")
        return super().execute(dry_run=dry_run)

if __name__ == "__main__":
    raise SystemExit(module_main(TokenizerWorker))
