"""KrushiRetriever: BM25-lite + intent boost. top_k=2. Checklist filter BEFORE fusion."""

import json
import math
import re
from pathlib import Path
from typing import Dict, List

from mini.aliases import detect_intent, normalize, split_tokens


class KrushiRetriever:
    """BM25-lite + intent boost. top_k=2. Checklist filter BEFORE fusion."""

    def __init__(self, kg_path: str = "data/kg_v2.jsonl"):
        self.docs: List[Dict] = []
        p = Path(kg_path)
        if not self.docs and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.docs.append(json.loads(line))
                        except Exception:
                            pass

        if not self.docs:
            data_dir = Path("data")
            if data_dir.exists():
                for jf in data_dir.glob("*.json"):
                    if jf.name in ("knowledge_gap_report.json", "super_quality_report.json", "truth_sources_whitelist.json", "open_source_catalog.json", "knowledge_quality_report.md"):
                        continue
                    try:
                        content = json.loads(jf.read_text(encoding="utf-8"))
                        if isinstance(content, list):
                            self.docs.extend(content)
                        elif isinstance(content, dict):
                            for k, v in content.items():
                                if isinstance(v, list):
                                    self.docs.extend(v)
                    except Exception:
                        pass

        if not self.docs:
            self.docs = [
                {
                    "id": "kg_q006_tur_latur",
                    "title": "Mandi Price: Tur @ APMC Latur",
                    "category": "Mandi Price",
                    "crop": "Tur",
                    "text": "Tur (Pigeon pea) wholesale rate at APMC Latur today: ₹6,800–₹7,400/quintal. Modal price ₹7,100/q. Source: Agmarknet (08:00 daily). Best sell window: Dec–Jan. Storage tip: keep moisture <12%.",
                },
                {
                    "id": "kg_q030_soil_test_harm",
                    "title": "Fertilizer Harm Without Soil Test",
                    "category": "Soil Test",
                    "crop": "General",
                    "text": "Applying DAP+Urea without soil test can: (1) cause nitrogen burn in cotton seedlings, (2) lock phosphorus in alkaline black soils (pH>8) reducing tur yield by 18%. Always test NPK+OC+pH before basal dose. Soil Health Card is free at soilhealth.dac.gov.in.",
                },
                {
                    "id": "kg_alias_green_gram",
                    "title": "Green Gram (Mung) Profile",
                    "category": "Crop",
                    "crop": "Green Gram",
                    "text": "Green Gram aliases: mung, moong, green gram. 55–65 days, 7-8 q/ha. IPM: yellow mosaic → resistant variety (Me HA-1, IPM-2-3).",
                },
            ]

        self._index()

    def _index(self):
        for d in self.docs:
            title = d.get("title") or d.get("title_mr") or d.get("title_en") or d.get("name_mr") or d.get("name_en") or ""
            text = d.get("text") or d.get("content") or d.get("content_mr") or d.get("content_en") or d.get("notes_mr") or ""
            crop = d.get("crop") or d.get("crop_mr") or d.get("crop_en") or ""
            cat = d.get("category") or d.get("type") or ""
            full_text = f"{title} {crop} {cat} {text}"
            d["_tokens"] = set(split_tokens(full_text))
            if not d.get("title"):
                d["title"] = title or f"{crop} {cat}".strip() or "Krushi Advisory"

        self.idf = {}
        N = len(self.docs)
        for d in self.docs:
            for t in d["_tokens"]:
                self.idf[t] = self.idf.get(t, 0) + 1
        self.idf = {t: math.log(1 + N / n) for t, n in self.idf.items()}

    def _score(self, qset: set, d: dict) -> float:
        return sum(self.idf.get(t, 0.0) for t in qset if t in d["_tokens"]) / max(
            1.0, len(d["_tokens"]) ** 0.5
        )

    def retrieve(
        self, query: str, top_k: int = 2, enable_checklist_filter: bool = True
    ) -> List[Dict]:
        qset = set(split_tokens(query))
        intent = detect_intent(query)

        from mini.taxonomy.aliases import resolve_crops_smart
        query_crops = resolve_crops_smart(query)
        q_crop_canon = query_crops[0] if query_crops else None

        scored = []
        for d in self.docs:
            s = self._score(qset, d)
            if q_crop_canon:
                d_crop = resolve_crops_smart(d.get("title", "") + " " + d.get("crop", ""))
                if d_crop and q_crop_canon in d_crop:
                    s *= 4.0
            if intent == "market" and "Mandi Price" in d.get("title", ""):
                s *= 3.0
            if intent == "scheme" and "Government Scheme" in d.get("category", ""):
                s *= 1.8
            if intent == "innovation" and d.get("category") == "Innovation":
                s *= 2.0
            if intent == "soil" and "Soil Test" in d.get("title", ""):
                s *= 2.0
            scored.append((s, d))

        scored.sort(key=lambda t: t[0], reverse=True)

        if enable_checklist_filter:
            cands = [d for s, d in scored if s > 0]
            if any("checklist" in d.get("title", "").lower() for d in cands[:5]):
                cands = [d for d in cands if "checklist" in d.get("title", "").lower()][:top_k]
            scored = [(s, d) for s, d in scored if d in cands][:top_k]

        res = [d for _, d in scored[:top_k] if _ > 0]
        return res if res else [d for _, d in scored[:top_k]]
