"""Evaluation harness: load Mini checkpoint, score gold + probes, write report (S13)."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mini.eval.gates import evaluate_gates, resolve_gates
from mini.eval.gold_sets import gold_summary, load_all_gold
from mini.eval.metrics import (
    Timer,
    aggregate_qa_scores,
    exact_match,
    keyword_hit_rate,
    process_rss_mb,
    rouge_l,
    token_f1,
)
from mini.eval.probes import aggregate_probes, hallucination_probes, score_probe
from mini.models.config import MiniConfig
from mini.models.corpus import DomainTokenizer
from mini.models.model import MiniLM, count_parameters
from mini.models.sft_data import SYSTEM_INSTRUCT, format_sft_example, prompt_only
from mini.models.train import set_seed
from mini.paths import EVAL_DIR, MODELS_DIR, ensure_lake_layout, relative_to_repo

EVAL_LATEST = EVAL_DIR / "EVAL_LATEST.json"
EVAL_HTML = EVAL_DIR / "EVAL_LATEST.html"

VERSION_DIRS = {
    "v0.4": MODELS_DIR / "v0.4-agri-qa",
    "v0.4-agri-qa": MODELS_DIR / "v0.4-agri-qa",
    "v0.3": MODELS_DIR / "v0.3-instruct",
    "v0.3-instruct": MODELS_DIR / "v0.3-instruct",
    "v0.2": MODELS_DIR / "v0.2-base",
    "v0.2-base": MODELS_DIR / "v0.2-base",
}


def resolve_model_dir(version: str = "v0.4") -> Path:
    key = (version or "v0.4").strip()
    if key in VERSION_DIRS:
        return VERSION_DIRS[key]
    # allow path-like
    p = Path(key)
    if p.is_dir():
        return p
    return VERSION_DIRS["v0.4"]


def load_checkpoint(
    model_dir: Path,
    *,
    device: torch.device,
) -> tuple[MiniLM, DomainTokenizer, MiniConfig, dict[str, Any]]:
    cfg_path = model_dir / "config.json"
    ckpt_path = model_dir / "pytorch_model.pt"
    tok_path = model_dir / "tokenizer.json"
    meta: dict[str, Any] = {"model_dir": str(model_dir), "loaded": False}
    if cfg_path.exists() and ckpt_path.exists() and tok_path.exists():
        cfg = MiniConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        tok = DomainTokenizer.load(tok_path)
        model = MiniLM(cfg)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        meta["loaded"] = True
        meta["checkpoint"] = relative_to_repo(ckpt_path)
        return model, tok, cfg, meta
    # Fallback: random Mini (eval still produces a report; gates may fail)
    cfg = MiniConfig(vocab_size=4096)
    tok = DomainTokenizer(vocab_size=4096)
    # minimal vocab so encode works
    tok.build(
        [
            "cotton disease soil fertilizer market scheme wheat rabi kharif irrigation "
            "pesticide label scout trap grape nashik nagpur orange pomegranate solapur"
        ],
        min_freq=1,
    )
    model = MiniLM(cfg).to(device)
    model.eval()
    meta["loaded"] = False
    meta["fallback"] = "random_init"
    return model, tok, cfg, meta


@torch.no_grad()
def generate_answer(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    question: str,
    *,
    device: torch.device,
    system: str = SYSTEM_INSTRUCT,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
) -> tuple[str, float]:
    text = format_sft_example(system=system, user=question, assistant="")
    # strip empty assistant trailing if any
    prompt = prompt_only(text)
    pids = tokenizer.encode(prompt, add_special=False)
    if not pids or pids[0] != tokenizer.bos_id:
        pids = [tokenizer.bos_id] + pids
    block = int(getattr(model.config, "block_size", 128) or 128)
    max_prompt = max(8, block - max_new_tokens)
    if len(pids) > max_prompt:
        pids = pids[-max_prompt:]
    t = Timer()
    idx = torch.tensor([pids], dtype=torch.long, device=device)
    try:
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
        gen = tokenizer.decode(out[0, len(pids) :].tolist())
    except Exception:
        gen = ""
    return gen, t.ms()


@torch.no_grad()
def lm_loss_ppl(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_examples: int = 24,
) -> dict[str, Any]:
    losses: list[float] = []
    block = int(getattr(model.config, "block_size", 128) or 128)
    for row in rows[:max_examples]:
        full = format_sft_example(
            system=SYSTEM_INSTRUCT,
            user=str(row.get("question") or ""),
            assistant=str(row.get("answer") or ""),
        )
        ids = tokenizer.encode(full, add_special=True)
        if len(ids) < 4:
            continue
        if len(ids) > block:
            ids = ids[: block - 1] + [tokenizer.eos_id]
        t = torch.tensor([ids], dtype=torch.long, device=device)
        _, loss = model(t[:, :-1], t[:, 1:])
        if loss is not None:
            losses.append(float(loss.item()))
    if not losses:
        return {"n": 0, "loss": None, "ppl": None}
    mean_loss = sum(losses) / len(losses)
    try:
        ppl = math.exp(min(mean_loss, 20.0))
    except Exception:
        ppl = None
    return {"n": len(losses), "loss": round(mean_loss, 6), "ppl": round(ppl, 4) if ppl else None}


def score_gold_set(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    gold: list[dict[str, Any]],
    *,
    device: torch.device,
    max_new_tokens: int = 28,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in gold:
        pred, lat = generate_answer(
            model,
            tokenizer,
            str(row.get("question") or ""),
            device=device,
            max_new_tokens=max_new_tokens,
        )
        gold_a = str(row.get("answer") or "")
        details.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "question": (row.get("question") or "")[:200],
                "gold": gold_a[:200],
                "pred": pred[:200],
                "f1": round(token_f1(pred, gold_a), 4),
                "em": exact_match(pred, gold_a),
                "rouge_l": round(rouge_l(pred, gold_a), 4),
                "keyword_hit": round(keyword_hit_rate(pred, row.get("must_keywords")), 4),
                "latency_ms": round(lat, 2),
            }
        )
    return details, aggregate_qa_scores(details)


def run_probes(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    *,
    device: torch.device,
    max_new_tokens: int = 28,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for probe in hallucination_probes():
        pred, lat = generate_answer(
            model,
            tokenizer,
            str(probe.get("question") or ""),
            device=device,
            max_new_tokens=max_new_tokens,
        )
        scored = score_probe(pred, probe)
        scored["latency_ms"] = round(lat, 2)
        results.append(scored)
    return results, aggregate_probes(results)


def render_html_report(report: dict[str, Any]) -> str:
    gates = report.get("gates") or {}
    qa = report.get("qa") or {}
    probes = report.get("probes") or {}
    lm = report.get("lm") or {}
    status = "PASS" if report.get("ok") else "FAIL"
    color = "#0a7" if report.get("ok") else "#c33"
    rows_html = ""
    for c in gates.get("checks") or []:
        ok = "✓" if c.get("ok") else "✗"
        rows_html += (
            f"<tr><td>{html.escape(str(c.get('gate')))}</td>"
            f"<td>{ok}</td><td>{html.escape(str(c.get('actual')))}</td>"
            f"<td>{html.escape(str(c.get('op')))} {html.escape(str(c.get('threshold')))}</td></tr>"
        )
    samples = ""
    for d in (report.get("samples") or [])[:8]:
        samples += (
            f"<li><b>{html.escape(str(d.get('id')))}</b> "
            f"F1={d.get('f1')} RH={d.get('keyword_hit')}<br/>"
            f"<i>Q:</i> {html.escape(str(d.get('question')))}<br/>"
            f"<i>Pred:</i> {html.escape(str(d.get('pred')))}</li>"
        )
    probe_lis = ""
    for p in (report.get("probe_details") or [])[:10]:
        probe_lis += (
            f"<li>{html.escape(str(p.get('id')))} "
            f"[{html.escape(str(p.get('status')))}] "
            f"{html.escape(str(p.get('note') or ''))}</li>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Mini Eval Scorecard</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:2rem;background:#0f1419;color:#e7ecf1}}
h1,h2{{color:#fff}} table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{border:1px solid #333;padding:.5rem;text-align:left}} th{{background:#1b2430}}
.badge{{display:inline-block;padding:.25rem .75rem;border-radius:6px;background:{color};color:#fff;font-weight:700}}
.card{{background:#1b2430;padding:1rem;border-radius:8px;margin:1rem 0}}
code{{color:#9cdcfe}}
</style></head><body>
<h1>Mini Eval Scorecard — {html.escape(str(report.get('version') or ''))}</h1>
<p><span class="badge">{status}</span>
 Sprint {html.escape(str(report.get('sprint') or ''))}
 · seed={report.get('seed')}
 · {html.escape(str(report.get('created_at') or ''))}</p>
<div class="card">
<h2>Summary</h2>
<ul>
<li>Token F1: <b>{qa.get('token_f1')}</b> · EM: {qa.get('exact_match')} · ROUGE-L: {qa.get('rouge_l')}</li>
<li>Keyword hit: {qa.get('keyword_hit')} · Regional keyword: {(report.get('regional') or {}).get('keyword_hit')}</li>
<li>LM loss: {lm.get('loss')} · PPL: {lm.get('ppl')}</li>
<li>Latency p95: {qa.get('latency_ms_p95')} ms · RSS: {(report.get('memory') or {}).get('rss_mb')} MB</li>
<li>Probes: pass_rate={probes.get('pass_rate')} hall_rate={probes.get('hallucination_rate')} mean={probes.get('mean_score')}</li>
<li>Gates failed: {html.escape(', '.join(gates.get('failed_gates') or []) or 'none')}</li>
</ul>
</div>
<div class="card"><h2>Gates</h2>
<table><tr><th>Gate</th><th>OK</th><th>Actual</th><th>Threshold</th></tr>
{rows_html}
</table></div>
<div class="card"><h2>Gold samples</h2><ul>{samples}</ul></div>
<div class="card"><h2>Probes</h2><ul>{probe_lis}</ul></div>
<p><code>W-EVAL · S13 / E5-eval harness · local report</code></p>
</body></html>
"""


def run_eval(
    *,
    dry_run: bool = False,
    version: str = "v0.4",
    gate_profile: str = "default",
    seed: int = 42,
    max_new_tokens: int = 28,
    max_gold: int | None = None,
    device: str | None = None,
    gate_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_lake_layout()
    set_seed(seed)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    gates = resolve_gates(gate_profile, gate_overrides)
    gold = load_all_gold()
    if max_gold is not None:
        gold = gold[: max(1, int(max_gold))]

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S13",
            "feature_phase": "E5-eval",
            "version": version,
            "gold": gold_summary(gold),
            "gates_profile": gate_profile,
            "planned_gates": gates,
            "n_probes": len(hallucination_probes()),
        }

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = resolve_model_dir(version)
    model, tok, cfg, load_meta = load_checkpoint(model_dir, device=dev)

    mem0 = process_rss_mb()
    details, qa_agg = score_gold_set(model, tok, gold, device=dev, max_new_tokens=max_new_tokens)

    regional_rows = [d for d in details if str(d.get("category")) in {"crop", "weather"} or (d.get("id") or "").startswith("gold-reg")]
    # recompute regional from gold ids
    regional_details = [d for d in details if str(d.get("id") or "").startswith("gold-reg")]
    regional_agg = aggregate_qa_scores(regional_details) if regional_details else aggregate_qa_scores([])

    probe_details, probe_agg = run_probes(model, tok, device=dev, max_new_tokens=max_new_tokens)
    lm = lm_loss_ppl(model, tok, gold, device=dev)
    mem1 = process_rss_mb()

    # Partial metrics for gate evaluation before artifacts list
    metrics_for_gates: dict[str, Any] = {
        "qa": qa_agg,
        "regional": regional_agg,
        "probes": probe_agg,
        "lm": lm,
        "latency": {"p95_ms": qa_agg.get("latency_ms_p95")},
        "artifacts": ["pending"],
    }
    # Temporarily set artifacts so require_artifacts can pass after write

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "ok": False,  # set after gates
        "dry_run": False,
        "sprint": "S13",
        "feature_phase": "E5-eval",
        "version": version,
        "model_dir": relative_to_repo(model_dir) if model_dir.exists() else str(model_dir),
        "load": load_meta,
        "seed": seed,
        "device": str(dev),
        "parameters": count_parameters(model),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else {},
        "gold": gold_summary(gold),
        "qa": qa_agg,
        "regional": regional_agg,
        "lm": lm,
        "probes": probe_agg,
        "probe_details": probe_details,
        "samples": details[:12],
        "memory": {"rss_mb_start": mem0, "rss_mb": mem1},
        "gate_profile": gate_profile,
        "created_at": created,
    }

    # Write artifacts first so gates see them
    artifacts: list[str] = []
    report_path = EVAL_DIR / f"eval_{version.replace('/', '_')}_{created.replace(':', '')}.json"
    # gates need artifacts in metrics
    metrics_for_gates["artifacts"] = ["will_write"]
    # evaluate with placeholder then re-evaluate after write
    gate_result = evaluate_gates(
        {
            "qa": qa_agg,
            "regional": regional_agg,
            "probes": probe_agg,
            "lm": lm,
            "latency": {"p95_ms": qa_agg.get("latency_ms_p95")},
            "artifacts": ["pending"],
        },
        gates,
    )
    # Force require_artifacts check after we write
    report["gates"] = gate_result

    EVAL_LATEST.parent.mkdir(parents=True, exist_ok=True)
    # finalize ok after artifacts + re-run gates
    report_for_gates = {
        "qa": qa_agg,
        "regional": regional_agg,
        "probes": probe_agg,
        "lm": lm,
        "latency": {"p95_ms": qa_agg.get("latency_ms_p95")},
        "artifacts": [str(EVAL_LATEST), str(EVAL_HTML)],
    }
    gate_result = evaluate_gates(report_for_gates, gates)
    report["gates"] = gate_result
    report["ok"] = bool(gate_result.get("ok"))

    html_body = render_html_report(report)
    EVAL_HTML.write_text(html_body, encoding="utf-8")
    EVAL_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    artifacts = [
        relative_to_repo(EVAL_LATEST),
        relative_to_repo(EVAL_HTML),
        relative_to_repo(report_path),
    ]
    report["artifacts"] = artifacts

    # rewrite with artifacts field complete
    EVAL_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    EVAL_HTML.write_text(render_html_report(report), encoding="utf-8")

    return report
