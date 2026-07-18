"""Automated worker modules for the Mini factory DAG."""

from mini.workers.base import BaseWorker, WORKER_REGISTRY, get_worker, list_workers
from mini.workers import registry  # noqa: F401 — side-effect: register all workers
from mini.workers import ingest  # noqa: F401
from mini.workers import quality  # noqa: F401
from mini.workers import standardize  # noqa: F401
from mini.workers import analyze  # noqa: F401
from mini.workers import qa_synth  # noqa: F401
from mini.workers import kg_build  # noqa: F401
from mini.workers import token  # noqa: F401
from mini.workers import pretrain  # noqa: F401

__all__ = ["BaseWorker", "WORKER_REGISTRY", "get_worker", "list_workers"]
