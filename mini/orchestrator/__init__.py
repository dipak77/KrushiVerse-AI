"""Mini factory orchestrator — local DAG runner (Prefect/Airflow-compatible later)."""

from mini.orchestrator.dag import PIPELINES, run_pipeline

__all__ = ["PIPELINES", "run_pipeline"]
