"""Post-v1.0 scale roadmap decision (Sprint 17)."""

from __future__ import annotations

from typing import Any

SCALE_ROADMAP: dict[str, Any] = {
    "decision": "proceed_with_family_scale",
    "decision_summary": (
        "After Mini v1.0 (~1M), scale along the same worker/dataset/tokenizer family: "
        "10M → 50M → 100M → 300M → 1B. Do not fork pipelines; grow emb/layers/vocab carefully."
    ),
    "owners": {
        "program": "KrushiVerseAI Mini program",
        "data": "W-INGEST / W-QASYNTH maintainers",
        "train": "W-PRETRAIN / W-SFT maintainers",
        "eval": "W-EVAL maintainers",
        "serve": "platform FastAPI / W-DEPLOY",
    },
    "ladder": [
        {
            "target_params": "10M",
            "focus": "Wider emb (256–384), more layers; same DomainTokenizer family",
            "depends_on": ["stable lake v1", "eval gates green", "1M serve stable"],
            "horizon": "post-v1.0 Q1",
        },
        {
            "target_params": "50M",
            "focus": "Longer context, better MR/HI; optional LoRA SFT",
            "depends_on": ["10M base", "≥100k QA", "GPU budget"],
            "horizon": "post-v1.0 Q2–Q3",
        },
        {
            "target_params": "100M+",
            "focus": "Multi-GPU pretrain; distillation from larger teacher optional",
            "depends_on": ["50M", "distributed train job", "human eval panel"],
            "horizon": "later",
        },
        {
            "target_params": "300M–1B",
            "focus": "Program-scale; separate funding/infra track",
            "depends_on": ["product PMF", "K8s serve", "curated gold growth"],
            "horizon": "future",
        },
    ],
    "reuse": [
        "StandardRecord schema + lake layout",
        "W-* workers and orchestrator DAG",
        "Eval gold/probes/gates",
        "Mini inference chain + USE_MINI_LLM flag",
        "Version registry + model cards",
    ],
    "explicit_non_goals_now": [
        "Vision-language Mini at v1.0",
        "Neo4j production cluster",
        "Multi-GPU 100M+ training in S17",
    ],
}


def build_scale_report() -> dict[str, Any]:
    return {
        "title": "Scale roadmap decision (post Mini v1.0)",
        "sprint": "S17",
        **SCALE_ROADMAP,
    }
