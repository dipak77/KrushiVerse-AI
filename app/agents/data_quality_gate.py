"""Data Quality Gate Agent for KrushiVerse-AI.

Enforces zero-hallucination, agronomic safety, non-harmful treatment validation,
crop matching, dose conversion, and truth source verification.
"""

import json
import os
import re
from typing import Any

WHITELIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "truth_sources_whitelist.json"))


class DataQualityGate:
    """Enforces strict agronomic, medical, and source quality rules across knowledge items."""

    def __init__(self):
        self.whitelist = self._load_whitelist()

    def _load_whitelist(self) -> list[str]:
        if os.path.exists(WHITELIST_PATH):
            try:
                with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [w["url"] for w in data.get("whitelist_domains", [])]
            except Exception:
                pass
        return [
            "https://nrcgrapes.icar.gov.in",
            "https://nrcp.icar.gov.in",
            "https://iivr.icar.gov.in",
            "https://iihr.res.in",
            "https://drmr.icar.gov.in",
            "https://dgr.icar.gov.in",
            "https://iipr.icar.gov.in",
            "https://cicr.org.in",
            "https://iisr.icar.gov.in",
            "https://mpkv.ac.in",
            "https://api.data.gov.in",
            "https://agriwelfare.gov.in",
            "https://krishikosh.egranth.ac.in"
        ]

    def validate_truth_source(self, source_url_or_tag: str) -> bool:
        """Rule 1: Must be ICAR / SAU / Gov / Open Source tagged, reject unverified blogs."""
        if not source_url_or_tag:
            return False
        tag_low = source_url_or_tag.lower()
        if any(b in tag_low for b in ["blog", "youtube", "quora", "sales"]):
            return False
        return any(w in tag_low for w in ["icar", "sau", "mpkv", "gov", "agmarknet", "open"]) or any(w in tag_low for w in self.whitelist)

    def validate_treatment_safety(self, disease_name: str, disease_type: str, treatment_text: str) -> tuple[bool, str]:
        """Rule 2 & 3: Harm Check & Chemical Appropriateness."""
        t_low = treatment_text.lower()
        d_type = disease_type.lower()
        d_name_low = disease_name.lower()

        is_viral = "virus" in d_type or "virus" in d_name_low or "बोकड्या" in d_name_low or "चुरडा" in d_name_low or "विषाणू" in d_name_low
        is_fungal = "fungus" in d_type or "भुरी" in d_name_low or "करपा" in d_name_low or "केवडा" in d_name_low or "तांबेरा" in d_name_low or "mildew" in d_name_low or "rot" in d_name_low
        is_bacterial = "bacteria" in d_type or "तेल्या" in d_name_low or "कॅंकर" in d_name_low or "blight" in d_name_low

        # Viral Safety: NO Streptocycline or Copper alone! MUST have vector control
        if is_viral:
            if "streptocycline" in t_low or "स्ट्रेप्टोसायक्लीन" in t_low:
                return False, "Harm Check Failed: Streptocycline antibiotic prescribed for viral disease."
            if not any(k in t_low for k in ["imidacloprid", "इमिडाक्लोप्रिड", "acetamiprid", "ॲसिटामिप्रीड", "sticky", "सापळे", "उपटून", "जाळा", "vector", "माशी"]):
                return False, "Harm Check Failed: Viral disease treatment missing vector control."

        # Fungal Safety: NO Streptocycline! MUST have antifungal agent
        if is_fungal and not is_bacterial:
            if "streptocycline" in t_low or "स्ट्रेप्टोसायक्लीन" in t_low:
                return False, "Harm Check Failed: Streptocycline antibiotic prescribed for fungal disease."
            if not any(k in t_low for k in ["sulphur", "गंधक", "copper", "कॉपर", "mancozeb", "मँकोझेब", "tebuconazole", "टेबुकॉनाझोल", "myclobutanil", "मायक्लोब्युटॅनिल", "carbendazim", "कार्बेन्डाझिम", "bordeaux", "बोर्डो"]):
                return False, "Harm Check Failed: Fungal disease treatment missing appropriate fungicide."

        return True, "Passed treatment safety check."

    def validate_dosage_conversion(self, treatment_text: str) -> tuple[bool, str]:
        """Rule 4: Dose Check - MUST specify both per-liter rate and 10L bucket format."""
        if not treatment_text:
            return False, "Missing treatment text."
        has_rate = any(k in treatment_text for k in ["g/L", "ml/L", "ग्रॅम/लिटर", "मिली/लिटर", "ग्रॅम", "मिली"])
        has_bucket = "10" in treatment_text or "१०" in treatment_text or "पाण्यात" in treatment_text
        if not (has_rate and has_bucket):
            return False, "Dose Check Failed: Must contain 10L bucket dosage conversion in Marathi."
        return True, "Passed dosage check."

    def filter_crop_mismatch(self, query_crop: str, doc_crop: str, doc_title: str) -> bool:
        """Rule 5: Hallucination & Crop Mismatch Check."""
        if not query_crop or not doc_crop:
            return True
        q_crop_low = query_crop.lower()
        d_crop_low = doc_crop.lower()
        title_low = doc_title.lower()

        if q_crop_low in d_crop_low or d_crop_low in q_crop_low:
            return True

        # Check title explicit mismatch (e.g., Mustard doc for Grapes query)
        if q_crop_low in title_low:
            return True
        if any(c in title_low for c in ["grapes", "pomegranate", "chilli", "cotton", "tomato", "soybean", "mustard", "banana"]) and q_crop_low not in title_low:
            return False

        return True


data_quality_gate = DataQualityGate()
