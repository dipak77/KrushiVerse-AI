"""Eval metrics: token-F1, EM, ROUGE-L, PPL/loss, latency, memory (Sprint 13)."""

from __future__ import annotations

import re
import time
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def token_f1(pred: str, gold: str) -> float:
    pt = set(_tokens(pred))
    gt = set(_tokens(gold))
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    inter = len(pt & gt)
    if inter == 0:
        return 0.0
    prec = inter / len(pt)
    rec = inter / len(gt)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def exact_match(pred: str, gold: str) -> float:
    p = re.sub(r"\s+", " ", (pred or "").strip().lower())
    g = re.sub(r"\s+", " ", (gold or "").strip().lower())
    return 1.0 if p == g and p else 0.0


def keyword_hit_rate(pred: str, must_keywords: list[str] | None) -> float:
    if not must_keywords:
        return 1.0
    pred_l = (pred or "").lower()
    hits = sum(1 for k in must_keywords if k.lower() in pred_l)
    return hits / max(1, len(must_keywords))


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 via LCS over tokens (no external deps)."""
    p = _tokens(pred)
    g = _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # DP LCS length
    n, m = len(p), len(g)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def process_rss_mb() -> float | None:
    try:
        import os

        import psutil  # type: ignore

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        try:
            import resource

            # Linux: ru_maxrss is KB; macOS is bytes — best-effort
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Heuristic: if huge, treat as bytes
            if rss > 10_000_000:
                return round(rss / (1024 * 1024), 2)
            return round(rss / 1024.0, 2)
        except Exception:
            return None


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


def aggregate_qa_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: list of {f1, em, rouge_l, keyword_hit, latency_ms}."""
    if not rows:
        return {
            "n": 0,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "keyword_hit": 0.0,
            "latency_ms_mean": None,
            "latency_ms_p95": None,
        }
    f1s = [float(r.get("f1") or 0) for r in rows]
    ems = [float(r.get("em") or 0) for r in rows]
    rls = [float(r.get("rouge_l") or 0) for r in rows]
    khs = [float(r.get("keyword_hit") or 0) for r in rows]
    lats = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    return {
        "n": len(rows),
        "token_f1": round(sum(f1s) / len(f1s), 4),
        "exact_match": round(sum(ems) / len(ems), 4),
        "rouge_l": round(sum(rls) / len(rls), 4),
        "keyword_hit": round(sum(khs) / len(khs), 4),
        "latency_ms_mean": round(sum(lats) / len(lats), 2) if lats else None,
        "latency_ms_p95": round(percentile(lats, 95) or 0, 2) if lats else None,
    }
