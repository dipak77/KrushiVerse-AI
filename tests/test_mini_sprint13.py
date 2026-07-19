"""Sprint 13 — evaluation harness (W-EVAL gold + probes + gates)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.eval.gates import evaluate_gates, resolve_gates
from mini.eval.gold_sets import load_all_gold
from mini.eval.harness import run_eval
from mini.eval.metrics import exact_match, rouge_l, token_f1
from mini.eval.probes import hallucination_probes, score_probe
from mini.orchestrator.dag import PIPELINES
from mini.paths import EVAL_DIR
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint13_markers():
    assert __sprint__ in {"S13", "S14", "S15", "S16", "S17", "S18"}
    assert __feature_phase__ in {"E5-eval", "E5-quant", "FP-8", "FP-9", "FP-10", "v2-15M"}


def test_eval_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-EVAL"] == "ready"


def test_gold_sets_cover_domains():
    gold = load_all_gold()
    assert len(gold) >= 15
    cats = {r["category"] for r in gold}
    # regional pack uses crop/weather; core domains present
    assert "disease" in cats
    assert "fertilizer" in cats
    assert "scheme" in cats
    assert "market" in cats
    assert any(str(r.get("id") or "").startswith("gold-reg") for r in gold)


def test_metrics_and_probes_helpers():
    assert token_f1("soil test NPK", "use soil test for NPK rates") > 0.3
    assert rouge_l("scout pink bollworm with traps", "scout pink bollworm with pheromone traps") > 0.3
    assert exact_match("hello", "hello") == 1.0
    probes = hallucination_probes()
    assert len(probes) >= 5
    bad = score_probe("Yes, double the pesticide dose now.", probes[0])
    assert bad["status"] == "fail"
    good = score_probe("No. Do not double doses; follow the label.", probes[0])
    assert good["status"] == "pass"


def test_gates_fail_and_pass():
    soft = resolve_gates("default")
    metrics_ok = {
        "qa": {"token_f1": 0.01, "rouge_l": 0.01, "keyword_hit": 0.01, "latency_ms_p95": 100},
        "regional": {"keyword_hit": 0.01},
        "probes": {"hallucination_rate": 0.0, "mean_score": 0.8},
        "lm": {"loss": 3.0, "ppl": 20.0},
        "artifacts": ["mini/eval/EVAL_LATEST.json"],
    }
    assert evaluate_gates(metrics_ok, soft)["ok"] is True

    strict = resolve_gates("strict")
    metrics_bad = {
        "qa": {"token_f1": 0.0, "rouge_l": 0.0, "keyword_hit": 0.0, "latency_ms_p95": 99999},
        "regional": {"keyword_hit": 0.0},
        "probes": {"hallucination_rate": 1.0, "mean_score": 0.0},
        "lm": {"loss": 100.0, "ppl": 1e20},
        "artifacts": [],
    }
    g = evaluate_gates(metrics_bad, strict)
    assert g["ok"] is False
    assert len(g["failed_gates"]) >= 1


def test_run_eval_writes_report():
    report = run_eval(
        dry_run=False,
        version="v0.4",
        gate_profile="default",
        seed=42,
        max_new_tokens=16,
        max_gold=8,
    )
    assert report.get("sprint") == "S13"
    assert report.get("qa", {}).get("n", 0) >= 1
    assert report.get("probes", {}).get("n", 0) >= 1
    assert EVAL_DIR.joinpath("EVAL_LATEST.json").exists()
    assert EVAL_DIR.joinpath("EVAL_LATEST.html").exists()
    assert report.get("artifacts")
    # default profile should pass structural gates for a completed run
    assert report.get("ok") is True
    assert "gates" in report


def test_worker_eval_short():
    res = get_worker("W-EVAL").run(
        dry_run=False,
        version="v0.4",
        gate_profile="default",
        seed=42,
        max_new_tokens=12,
        max_gold=6,
    )
    assert res.ok is True
    assert res.metrics.get("sprint") == "S13"


def test_strict_profile_can_fail_exit():
    """Acceptance: failing gates => ok=False (CLI exits non-zero)."""
    res = get_worker("W-EVAL").run(
        dry_run=False,
        version="v0.4",
        gate_profile="strict",
        seed=42,
        max_new_tokens=8,
        max_gold=4,
        # force impossible absolute floors via overrides
        min_token_f1=0.99,
        min_rouge_l=0.99,
    )
    assert res.ok is False
    assert res.errors


def test_sprint13_pipeline():
    assert "sprint13" in PIPELINES
    assert PIPELINES["sprint13"] == ["W-EVAL"]
    assert PIPELINES["eval"] == ["W-EVAL"]


def test_api_eval_s13():
    r = client.post("/api/lake/eval?execute=true&version=v0.4&profile=default&seed=42&max_new_tokens=12&max_gold=5")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S13"
