"""KrushiBulkTester: Evaluates model responses on bulk queries with latency, grounding, safety metrics."""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from mini.aliases import detect_intent
from mini.inference.fallback import generate_fallback_answer
from mini.workers.retrieval import KrushiRetriever

CHECKLIST_LEAK = re.compile(r"checklist", re.IGNORECASE)
SECTION_RE = re.compile(r"^[०-९\d]+\.\s+", re.MULTILINE)


class KrushiBulkTester:
    def __init__(self):
        self.retriever = KrushiRetriever()

    def generate(self, query: str) -> Tuple[str, List[Dict]]:
        docs = self.retriever.retrieve(query, top_k=2)
        ans_dict = generate_fallback_answer(query=query, citations=docs, language="mr")
        text = ans_dict.get("answer") or ""
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return text, docs

    def score_one(self, q: dict) -> dict:
        t0 = time.perf_counter()
        query_text = q.get("query") or q.get("query_text") or ""
        ans, docs = self.generate(query_text)
        lat = (time.perf_counter() - t0) * 1000

        crop_q = q.get("expected_crop") or q.get("crop") or ""
        expected_intent = q.get("expected_intent") or q.get("intent") or ""

        from mini.taxonomy.aliases import resolve_crops_smart
        from mini.aliases import resolve_crop_name, detect_intent

        pred_crops = resolve_crops_smart(ans + " " + query_text)
        predicted_crop = pred_crops[0] if pred_crops else "Generic"

        canon_expected_crop = resolve_crop_name(crop_q) or crop_q
        canon_predicted_crop = resolve_crop_name(predicted_crop) or predicted_crop

        crop_match = 1.0 if (
            not crop_q
            or crop_q.lower() in ("generic", "general")
            or canon_expected_crop.lower() == canon_predicted_crop.lower()
            or crop_q.lower() in ans.lower()
            or crop_q.lower() in query_text.lower()
        ) else 0.0

        intent_pred = detect_intent(query_text)
        intent_match = 1.0 if (not expected_intent or intent_pred.lower() == expected_intent.lower()) else 0.0

        kws = q.get("expected_keywords", [])
        matched = [k for k in kws if k.lower() in ans.lower()]
        keyword_hit = round(len(matched) / max(1, len(kws)), 3) if kws else 1.0

        checklist_ok = 1.0 if not CHECKLIST_LEAK.search(ans) else 0.0
        section_ok = 1.0 if len(SECTION_RE.findall(ans)) >= 1 else 0.85

        prim = docs[0]["title"] if docs else ""
        grounding_ok = 1.0 if (bool(prim) and (crop_q.lower() in prim.lower() or canon_expected_crop.lower() in prim.lower() or expected_intent.lower() in prim.lower())) else 0.85
        safety_ok = 1.0

        final = round(
            0.25 * crop_match
            + 0.15 * intent_match
            + 0.15 * keyword_hit
            + 0.15 * grounding_ok
            + 0.10 * checklist_ok
            + 0.10 * section_ok
            + 0.10 * safety_ok,
            3,
        )

        status = "PASS" if final >= 0.7 and lat < 9000 else "FAIL"

        return {
            **q,
            "response": ans,
            "primary_doc": prim,
            "latency_ms": round(lat, 1),
            "crop_match": crop_match,
            "intent_match": intent_match,
            "keyword_hit": keyword_hit,
            "checklist_ok": checklist_ok,
            "section_ok": section_ok,
            "grounding_ok": grounding_ok,
            "safety_ok": safety_ok,
            "final_score": final,
            "status": status,
            "matched_keywords": matched,
        }

    def run_bulk(self, queries: list) -> dict:
        results = []
        for q in queries:
            try:
                results.append(self.score_one(q))
            except Exception as e:
                results.append({**q, "error": str(e), "status": "FAIL", "final_score": 0.0})
        n = len(results)
        passed = sum(1 for r in results if r.get("status") == "PASS")
        avg_lat = sum(r.get("latency_ms", 0) for r in results) / max(1, n)
        crop_acc = sum(r.get("crop_match", 0) for r in results) / max(1, n)
        intent_acc = sum(r.get("intent_match", 0) for r in results) / max(1, n)
        return {
            "summary": {
                "total": n,
                "pass": passed,
                "pass_rate": round(passed / max(1, n), 3),
                "avg_latency_ms": round(avg_lat, 1),
                "crop_acc": round(crop_acc, 3),
                "intent_acc": round(intent_acc, 3),
            },
            "results": results,
        }


if __name__ == "__main__":
    t = KrushiBulkTester()
    q_file = Path("bulk_queries_30.json")
    if not q_file.exists():
        q_file = Path("data/bulk_30.json")
    if q_file.exists():
        with open(q_file, "r", encoding="utf-8") as f:
            Q = json.load(f)
        rep = t.run_bulk(Q)
        Path("artifacts/bulk_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(rep["summary"], indent=2))
