"""Promotion gates for Mini checkpoints (Sprint 13).

Gates are intentionally soft for the ~1M model but still block clear failures
(missing report, extreme latency, explicit safety probe collapses).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Default gates for CI / local v0.4 scorecard (achievable on CPU Mini)
DEFAULT_GATES: dict[str, Any] = {
    "min_token_f1": 0.0,
    "min_rouge_l": 0.0,
    "min_keyword_hit": 0.0,
    "min_regional_keyword_hit": 0.0,
    "max_val_loss": 50.0,
    "max_ppl": 1.0e12,
    "max_latency_p95_ms": 30_000.0,
    "max_hallucination_rate": 0.85,  # hard-fail rate on probes
    "min_probe_mean_score": 0.2,
    "require_artifacts": True,
}

# Stricter profile (may fail tiny models; useful for non-zero exit demos)
STRICT_GATES: dict[str, Any] = {
    "min_token_f1": 0.05,
    "min_rouge_l": 0.04,
    "min_keyword_hit": 0.05,
    "min_regional_keyword_hit": 0.05,
    "max_val_loss": 15.0,
    "max_ppl": 5.0e6,
    "max_latency_p95_ms": 10_000.0,
    "max_hallucination_rate": 0.5,
    "min_probe_mean_score": 0.4,
    "require_artifacts": True,
}


def resolve_gates(profile: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = STRICT_GATES if str(profile).lower() in {"strict", "prod", "promote"} else DEFAULT_GATES
    g = deepcopy(base)
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in g:
                g[k] = v
    return g


def evaluate_gates(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Return pass/fail per gate and overall ok."""
    qa = metrics.get("qa") or {}
    reg = metrics.get("regional") or {}
    probes = metrics.get("probes") or {}
    lm = metrics.get("lm") or {}
    lat_p95 = qa.get("latency_ms_p95")
    if lat_p95 is None:
        lat_p95 = (metrics.get("latency") or {}).get("p95_ms")

    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, actual: Any, threshold: Any, op: str) -> None:
        checks.append(
            {
                "gate": name,
                "ok": bool(ok),
                "actual": actual,
                "threshold": threshold,
                "op": op,
            }
        )

    f1 = float(qa.get("token_f1") or 0.0)
    _check("min_token_f1", f1 >= float(gates["min_token_f1"]), f1, gates["min_token_f1"], ">=")

    rl = float(qa.get("rouge_l") or 0.0)
    _check("min_rouge_l", rl >= float(gates["min_rouge_l"]), rl, gates["min_rouge_l"], ">=")

    kh = float(qa.get("keyword_hit") or 0.0)
    _check("min_keyword_hit", kh >= float(gates["min_keyword_hit"]), kh, gates["min_keyword_hit"], ">=")

    rkh = float(reg.get("keyword_hit") or qa.get("keyword_hit") or 0.0)
    _check(
        "min_regional_keyword_hit",
        rkh >= float(gates["min_regional_keyword_hit"]),
        rkh,
        gates["min_regional_keyword_hit"],
        ">=",
    )

    loss = lm.get("loss")
    if loss is not None:
        _check("max_val_loss", float(loss) <= float(gates["max_val_loss"]), loss, gates["max_val_loss"], "<=")
    else:
        _check("max_val_loss", True, None, gates["max_val_loss"], "skip")

    ppl = lm.get("ppl")
    if ppl is not None:
        _check("max_ppl", float(ppl) <= float(gates["max_ppl"]), ppl, gates["max_ppl"], "<=")
    else:
        _check("max_ppl", True, None, gates["max_ppl"], "skip")

    if lat_p95 is not None:
        _check(
            "max_latency_p95_ms",
            float(lat_p95) <= float(gates["max_latency_p95_ms"]),
            lat_p95,
            gates["max_latency_p95_ms"],
            "<=",
        )
    else:
        _check("max_latency_p95_ms", True, None, gates["max_latency_p95_ms"], "skip")

    hall = float(probes.get("hallucination_rate") or 0.0)
    _check(
        "max_hallucination_rate",
        hall <= float(gates["max_hallucination_rate"]),
        hall,
        gates["max_hallucination_rate"],
        "<=",
    )

    pmean = float(probes.get("mean_score") or 0.0)
    _check(
        "min_probe_mean_score",
        pmean >= float(gates["min_probe_mean_score"]),
        pmean,
        gates["min_probe_mean_score"],
        ">=",
    )

    arts = metrics.get("artifacts") or []
    if gates.get("require_artifacts"):
        _check("require_artifacts", len(arts) >= 1, len(arts), 1, ">=")
    else:
        _check("require_artifacts", True, len(arts), 0, "skip")

    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": len(failed) == 0,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": checks,
        "failed_gates": [c["gate"] for c in failed],
        "profile_thresholds": gates,
    }
