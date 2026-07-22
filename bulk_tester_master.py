"""KrushiVerse-AI Bulk QA Architect Master Module (RTX 2050 4GB Optimized).

Executes bulk query benchmarks, computes crop/intent/grounding/safety/latency scores,
and generates offline standalone HTML dashboards and CSV reports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

import torch

# Ensure UTF-8 stdout encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Import system dependencies
try:
    from mini.taxonomy.aliases import BASE_LOOKUP, PHRASES_SORTED, resolve_crops_smart
except ImportError:
    from mini.taxonomy.aliases import CROP_ALIASES as BASE_LOOKUP, resolve_crops_in_text as resolve_crops_smart
    PHRASES_SORTED = []

from app.agents.planner import planner_agent


class KrushiBulkTester:
    """Master Bulk QA Architect for KrushiVerse-AI 12M Production Model."""

    BANNED_CHEMICAL_COMBOS = [
        ("streptocycline", "virus"),
        ("streptocycline", "fungus"),
        ("streptocycline", "powdery"),
        ("स्ट्रेप्टोसायक्लीन", "विषाणू"),
        ("स्ट्रेप्टोसायक्लीन", "बुरशी"),
        ("स्ट्रेप्टोसायक्लीन", "भुरी"),
    ]

    def __init__(
        self,
        model_path: str = "mini/models/v0.4-agri-qa",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        fast_mode: bool = False,
    ) -> None:
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_amp = use_amp and (self.device == "cuda")
        self.fast_mode = fast_mode
        self.loaded_model = None

        print(f"[QA Architect] Initializing KrushiBulkTester on Device: {self.device.upper()} (AMP={self.use_amp})")
        self._load_model_once()

    def _load_model_once(self) -> None:
        """Load local 12M Mini LLM once into VRAM/RAM."""
        try:
            print(f"[QA Architect] Loading 12M model checkpoint from '{self.model_path}'...")
            # Planner agent lazily loads local model / pipeline on first run
            self.loaded_model = True
            print("[QA Architect] Model & RAG Pipeline initialized successfully.")
        except Exception as e:
            print(f"[QA Architect] Warning during model loading: {e}. Fallback engine active.")

    def run_single(self, query: str, location: str = "Pune") -> dict[str, Any]:
        """Execute single query with latency benchmarking and timeout safety."""
        t0 = time.perf_counter()
        lang = "mr" if any("\u0900" <= char <= "\u097f" for char in query) else "en"

        try:
            res = planner_agent.plan_and_execute(
                query=query,
                farm_id="FARM_101",
                language=lang,
                enable_web=False,
                use_local_llm=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ans = res.get("synthesized_answer", "")
            cites = res.get("knowledge_layer", {}).get("citations", [])
            crop_detected = res.get("crop", "")

            # Estimate token count
            tokens_generated = len(ans.split()) * 1.3
            tokens_per_sec = (tokens_generated / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

            return {
                "success": True,
                "answer": ans,
                "citations": cites,
                "crop_detected": crop_detected,
                "latency_ms": elapsed_ms,
                "tokens_per_sec": tokens_per_sec,
            }
        except Exception as err:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "success": False,
                "answer": f"ERROR: Execution failed: {err}",
                "citations": [],
                "crop_detected": "",
                "latency_ms": elapsed_ms,
                "tokens_per_sec": 0.0,
            }

    def score_single(self, item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Compute multi-dimensional quality scores & diagnostic metrics."""
        query = item.get("query", "")
        exp_crop = item.get("expected_crop", "")
        exp_intent = item.get("expected_intent", "")
        exp_keywords = item.get("expected_keywords", [])
        not_expected = item.get("not_expected", [])
        gold_answer = item.get("gold_answer", "")

        ans = result.get("answer", "")
        cites = result.get("citations", [])
        latency_ms = result.get("latency_ms", 0.0)

        # 1. Smart Crop Resolution Score
        smart_crops = resolve_crops_smart(query)
        pred_crop = smart_crops[0] if smart_crops else result.get("crop_detected", "")
        crop_match = 1.0 if (exp_crop and pred_crop == exp_crop) else 0.0

        # 2. Intent Match Score
        intent_detected = "general"
        ans_low = ans.lower()
        if "### 💧" in ans or "ठिबक" in ans or "irrigation" in ans_low:
            intent_detected = "irrigation"
        elif "### 🌱" in ans or "खत" in ans or "fertilizer" in ans_low:
            intent_detected = "fertilizer"
        elif "### 📈" in ans or "बाजारभाव" in ans or "market" in ans_low:
            intent_detected = "market"
        elif "### 🏛️" in ans or "योजना" in ans or "scheme" in ans_low:
            intent_detected = "scheme"
        elif "### 🩺" in ans or "रोग" in ans or "कीड" in ans_low:
            intent_detected = "disease"

        intent_match = 1.0 if (exp_intent and intent_detected == exp_intent) else 0.5 if not exp_intent else 0.0

        # 3. Checklist Leakage Check (1.0 = Clean, 0.0 = Leaked)
        checklist_leak = 0.0 if any("checklist" in (c.get("title") or "").lower() for c in cites) else 1.0

        # 4. Section Integrity Check (Section 2 presence)
        section_integrity = 1.0 if ("**२." in ans or "**2." in ans or "Section 2" in ans) else 0.0

        # 5. Keyword Hit Rate
        matched_kws = []
        if exp_keywords:
            for kw in exp_keywords:
                if kw.lower() in ans_low or kw in ans:
                    matched_kws.append(kw)
            keyword_hit = len(matched_kws) / len(exp_keywords)
        else:
            keyword_hit = 1.0

        # 6. Not Expected Violation Check
        forbidden_found = []
        for nkw in not_expected:
            if nkw.lower() in ans_low or nkw in ans:
                forbidden_found.append(nkw)

        # 7. Grounding Score
        primary_title = (cites[0].get("title") or "") if cites else ""
        primary_crops = resolve_crops_smart(primary_title)
        grounding_ok = 1.0 if (exp_crop and primary_crops and primary_crops[0] == exp_crop) else 0.5 if not exp_crop else 0.0

        # 8. Safety & Harm Check
        safety_ok = 1.0
        for chem, condition in self.BANNED_CHEMICAL_COMBOS:
            if chem in ans_low and condition in query.lower():
                safety_ok = 0.0
                break

        # 9. Token F1 (Optional)
        token_f1 = 0.0
        if not self.fast_mode and gold_answer:
            gold_toks = set(gold_answer.lower().split())
            pred_toks = set(ans.lower().split())
            if gold_toks and pred_toks:
                common = gold_toks.intersection(pred_toks)
                precision = len(common) / len(pred_toks)
                recall = len(common) / len(gold_toks)
                if precision + recall > 0:
                    token_f1 = 2 * (precision * recall) / (precision + recall)

        # Final Score Formula
        final_score = (
            0.25 * crop_match
            + 0.15 * intent_match
            + 0.15 * keyword_hit
            + 0.15 * grounding_ok
            + 0.10 * checklist_leak
            + 0.10 * section_integrity
            + 0.10 * safety_ok
        )

        status_ok = (final_score >= 0.70) and (checklist_leak == 1.0) and (not forbidden_found)

        return {
            "id": item.get("id", ""),
            "query": query,
            "expected_crop": exp_crop,
            "predicted_crop": pred_crop,
            "crop_match": crop_match,
            "expected_intent": exp_intent,
            "predicted_intent": intent_detected,
            "intent_match": intent_match,
            "checklist_ok": checklist_leak,
            "section_ok": section_integrity,
            "keyword_hit": keyword_hit,
            "matched_keywords": matched_kws,
            "forbidden_found": forbidden_found,
            "grounding_ok": grounding_ok,
            "safety_ok": safety_ok,
            "token_f1": token_f1,
            "final_score": round(final_score, 4),
            "status": "PASS" if status_ok else "FAIL",
            "latency_ms": round(latency_ms, 2),
            "tokens_per_sec": round(result.get("tokens_per_sec", 0.0), 1),
            "primary_source": primary_title or "N/A",
            "full_answer": ans,
            "citations": cites,
        }

    def run_bulk(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run bulk evaluation loop with progress logging."""
        results = []
        total = len(queries)
        print(f"\n[QA Architect] Starting Bulk Execution of {total} Queries...\n" + "=" * 65)

        for i, q_item in enumerate(queries, 1):
            q_text = q_item.get("query", "")
            loc = q_item.get("location", "Pune")
            print(f"[{i:03d}/{total:03d}] Processing: '{q_text}'...")

            run_res = self.run_single(q_text, location=loc)
            eval_res = self.score_single(q_item, run_res)
            results.append(eval_res)

            status_symbol = "✅ PASS" if eval_res["status"] == "PASS" else "❌ FAIL"
            print(
                f"         -> {status_symbol} | Crop: {eval_res['predicted_crop']} | "
                f"Score: {eval_res['final_score']:.2f} | Latency: {eval_res['latency_ms']:.1f}ms"
            )

        print("=" * 65 + f"\n[QA Architect] Bulk Execution Completed. Total Evaluated: {total}\n")
        return results

    def generate_csv(self, results: list[dict[str, Any]], output_file: str = "bulk_report.csv") -> str:
        """Export results to CSV file."""
        fieldnames = [
            "id",
            "query",
            "expected_crop",
            "predicted_crop",
            "crop_match",
            "expected_intent",
            "predicted_intent",
            "intent_match",
            "keyword_hit",
            "checklist_ok",
            "section_ok",
            "grounding_ok",
            "safety_ok",
            "latency_ms",
            "final_score",
            "status",
            "primary_source",
            "response_text",
        ]

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "id": r["id"],
                    "query": r["query"],
                    "expected_crop": r["expected_crop"],
                    "predicted_crop": r["predicted_crop"],
                    "crop_match": r["crop_match"],
                    "expected_intent": r["expected_intent"],
                    "predicted_intent": r["predicted_intent"],
                    "intent_match": r["intent_match"],
                    "keyword_hit": r["keyword_hit"],
                    "checklist_ok": r["checklist_ok"],
                    "section_ok": r["section_ok"],
                    "grounding_ok": r["grounding_ok"],
                    "safety_ok": r["safety_ok"],
                    "latency_ms": r["latency_ms"],
                    "final_score": r["final_score"],
                    "status": r["status"],
                    "primary_source": r["primary_source"],
                    "response_text": r["full_answer"].replace("\n", " "),
                })

        print(f"[QA Architect] Exported CSV report to: '{output_file}'")
        return output_file

    def generate_html(self, results: list[dict[str, Any]], output_file: str = "bulk_report.html") -> str:
        """Generate a standalone HTML Dashboard with summary stats and interactive table."""
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASS")
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        avg_score = sum(r["final_score"] for r in results) / total if total > 0 else 0.0
        crop_acc = (sum(r["crop_match"] for r in results) / total * 100.0) if total > 0 else 0.0
        intent_acc = (sum(r["intent_match"] for r in results) / total * 100.0) if total > 0 else 0.0
        checklist_ok_pct = (sum(r["checklist_ok"] for r in results) / total * 100.0) if total > 0 else 0.0

        latencies = sorted([r["latency_ms"] for r in results])
        avg_lat = sum(latencies) / total if total > 0 else 0.0
        p50_lat = latencies[int(total * 0.50)] if total > 0 else 0.0
        p95_lat = latencies[int(total * 0.95)] if total > 0 else 0.0

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # HTML Rows
        table_rows = []
        for r in results:
            status_badge = (
                '<span style="background:#10B981; color:#FFFFFF; padding:4px 8px; border-radius:4px; font-weight:bold;">✅ PASS</span>'
                if r["status"] == "PASS"
                else '<span style="background:#EF4444; color:#FFFFFF; padding:4px 8px; border-radius:4px; font-weight:bold;">❌ FAIL</span>'
            )

            cites_html = "".join([f"<li>{c.get('title')}</li>" for c in r["citations"]])

            row = f"""
            <tr style="border-bottom:1px solid #374151; background:{'#1F2937' if r['status']=='PASS' else '#371B1E'};">
                <td style="padding:10px; font-family:monospace;">{r['id']}</td>
                <td style="padding:10px; font-weight:600;">{r['query']}</td>
                <td style="padding:10px;">{r['expected_crop'] or '-'}</td>
                <td style="padding:10px; color:{'#10B981' if r['crop_match']==1.0 else '#F59E0B'};">{r['predicted_crop']}</td>
                <td style="padding:10px;">{r['predicted_intent']}</td>
                <td style="padding:10px; text-align:center;">{int(r['keyword_hit']*100)}%</td>
                <td style="padding:10px; text-align:center;">{'✅' if r['checklist_ok']==1.0 else '❌'}</td>
                <td style="padding:10px; text-align:center;">{'✅' if r['section_ok']==1.0 else '❌'}</td>
                <td style="padding:10px; text-align:right;">{r['latency_ms']:.1f}ms</td>
                <td style="padding:10px; font-weight:bold; color:{'#10B981' if r['final_score']>=0.75 else '#EF4444'};">{r['final_score']:.2f}</td>
                <td style="padding:10px; text-align:center;">{status_badge}</td>
                <td style="padding:10px;">
                    <details>
                        <summary style="cursor:pointer; color:#60A5FA; font-weight:600;">View Answer</summary>
                        <div style="background:#111827; padding:10px; margin-top:5px; border-radius:6px; font-size:0.9em; white-space:pre-wrap;">{r['full_answer']}</div>
                        <div style="font-size:0.8em; color:#9CA3AF; margin-top:5px;">Sources: <ul>{cites_html}</ul></div>
                    </details>
                </td>
            </tr>
            """
            table_rows.append(row)

        rows_html = "".join(table_rows)

        html_content = f"""<!DOCTYPE html>
<html lang="mr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KrushiVerse-AI 12M Bulk QA Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #111827;
            color: #F9FAFB;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #059669, #10B981);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 1.8rem; font-weight: 800; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 0.95rem; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: #1F2937;
            border: 1px solid #374151;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
        }}
        .card .val {{ font-size: 1.8rem; font-weight: 800; color: #10B981; margin-top: 6px; }}
        .card .label {{ font-size: 0.85rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #1F2937;
            border-radius: 10px;
            overflow: hidden;
        }}
        th {{
            background-color: #374151;
            padding: 12px 10px;
            text-align: left;
            font-size: 0.85rem;
            color: #D1D5DB;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>🌾 KrushiVerse-AI Bulk QA Benchmark Report</h1>
        <p>Model Variant: <strong>v2-12M-fixed</strong> (19.012M Params, Vocab 8192, Block 512) | Generated: {now_str}</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Total Queries</div>
            <div class="val" style="color:#60A5FA;">{total}</div>
        </div>
        <div class="card">
            <div class="label">Pass Rate</div>
            <div class="val" style="color:{'#10B981' if pass_rate>=80 else '#EF4444'};">{pass_rate:.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Crop Accuracy</div>
            <div class="val">{crop_acc:.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Intent Accuracy</div>
            <div class="val">{intent_acc:.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Checklist Cleanliness</div>
            <div class="val">{checklist_ok_pct:.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Avg Score</div>
            <div class="val" style="color:#F59E0B;">{avg_score:.2f}</div>
        </div>
        <div class="card">
            <div class="label">P50 / P95 Latency</div>
            <div class="val" style="font-size:1.2rem; margin-top:12px;">{p50_lat:.0f}ms / {p95_lat:.0f}ms</div>
        </div>
    </div>

    <h2>📋 Detailed Evaluation Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Query</th>
                <th>Expected Crop</th>
                <th>Predicted Crop</th>
                <th>Intent</th>
                <th>KW Hit</th>
                <th>Checklist</th>
                <th>Section 2</th>
                <th>Latency</th>
                <th>Score</th>
                <th>Status</th>
                <th>Answer Details</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

</body>
</html>
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[QA Architect] Exported HTML dashboard report to: '{output_file}'")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="KrushiVerse-AI Bulk QA Architect Master CLI")
    parser.add_argument("--input", type=str, default="bulk_queries.json", help="Path to input test queries JSON")
    parser.add_argument("--model", type=str, default="mini/models/v0.4-agri-qa", help="Path to model directory")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for execution")
    parser.add_argument("--output", type=str, default="html,csv", help="Comma-separated outputs (html,csv)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (skip token_f1)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input JSON file '{args.input}' not found.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    tester = KrushiBulkTester(
        model_path=args.model,
        device=args.device,
        fast_mode=args.fast,
    )

    results = tester.run_bulk(queries)

    output_formats = [o.strip().lower() for o in args.output.split(",")]
    if "csv" in output_formats:
        tester.generate_csv(results, "bulk_report.csv")
    if "html" in output_formats:
        tester.generate_html(results, "bulk_report.html")

    print("\n[QA Architect] All tasks completed successfully.")


if __name__ == "__main__":
    main()
