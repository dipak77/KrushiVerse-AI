"""Load / smoke checks for Mini serve path (Sprint 17)."""

from __future__ import annotations

import time
from typing import Any


def run_load_smoke(
    *,
    queries: list[str] | None = None,
    rounds: int = 3,
    enable_web: bool = False,
    enable_agents: bool = False,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Repeated Mini chat calls; log latency stats (CPU-friendly)."""
    from app.llm.mini_bridge import run_mini_chat
    from mini.eval.metrics import percentile

    queries = queries or [
        "How do I manage pink bollworm in cotton with IPM?",
        "What is a basic soil-test fertilizer tip for onion?",
        "Should I double pesticide dose before rain?",
    ]
    latencies: list[float] = []
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    t_all = time.perf_counter()
    for r in range(rounds):
        for q in queries:
            t0 = time.perf_counter()
            try:
                out = run_mini_chat(
                    q,
                    language="en",
                    mode="grounded",
                    enable_web=enable_web,
                    enable_agents=enable_agents,
                    use_platform_rag=False,
                    max_new_tokens=max_new_tokens,
                    seed=42 + r,
                )
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                results.append(
                    {
                        "round": r,
                        "query": q[:80],
                        "ok": bool(out.get("ok")),
                        "n_sources": out.get("n_sources"),
                        "engine": out.get("engine"),
                        "latency_ms": round(dt, 2),
                    }
                )
                if not out.get("ok") and out.get("n_sources", 0) == 0:
                    errors.append(f"no sources: {q[:40]}")
            except Exception as e:
                errors.append(str(e))
                results.append({"round": r, "query": q[:80], "ok": False, "error": str(e)})
    total_ms = (time.perf_counter() - t_all) * 1000.0
    ok_n = sum(1 for x in results if x.get("ok"))
    return {
        "ok": ok_n >= max(1, len(results) // 2) and len(errors) < len(results),
        "rounds": rounds,
        "n_calls": len(results),
        "n_ok": ok_n,
        "errors": errors[:10],
        "latency_ms": {
            "mean": round(sum(latencies) / max(1, len(latencies)), 2) if latencies else None,
            "p50": round(percentile(latencies, 50) or 0, 2) if latencies else None,
            "p95": round(percentile(latencies, 95) or 0, 2) if latencies else None,
            "total_wall_ms": round(total_ms, 2),
        },
        "samples": results[:6],
    }
