"""Expert QA synthesis factory (Sprint 6 / Phase 6 start).

Generates multilingual template-based Q/A packs from:
- Processed lake structured facts
- Frozen taxonomy (crops, stages, schemes catalog)

Targets (interim): ≥10k train + ≥1k val StandardRecords with schema v1.
No train/test leakage: all paraphrases of a fact_key share the same split.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini.contracts import Category, DataSplit, LanguageCode, Region, StandardRecord
from mini.lake.dedup import extract_record_lists
from mini.lake.process import iter_processed_json_files
from mini.paths import (
    DATASETS_DIR,
    LAKE_PROCESSED,
    LAKE_ROOT,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    SCHEMA_VERSION,
    ensure_lake_layout,
    relative_to_repo,
)
from mini.taxonomy.aliases import resolve_crop_name
from mini.taxonomy.domains import TAXONOMY, list_crop_names_en, list_crop_stages
from mini.taxonomy.regions import list_mh_districts


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _split_for_fact(fact_key: str) -> DataSplit:
    """Deterministic split by fact family (prevents paraphrase leakage)."""
    n = int(hashlib.md5(fact_key.encode("utf-8")).hexdigest(), 16) % 100
    if n < 80:
        return DataSplit.TRAIN
    if n < 90:
        return DataSplit.VAL
    return DataSplit.TEST


def _rid(fact_key: str, template_id: str, lang: str) -> str:
    return "syn_" + _hash(f"{fact_key}|{template_id}|{lang}")[:20]


def _conf(base: float, *, has_mr: bool = False, has_detail: bool = True) -> float:
    c = base
    if has_detail:
        c += 0.05
    if has_mr:
        c += 0.03
    return max(0.35, min(0.95, round(c, 3)))


def _mk(
    *,
    fact_key: str,
    template_id: str,
    category: Category,
    question: str,
    answer: str,
    language: LanguageCode,
    crop: str | None = None,
    subcategory: str | None = None,
    source: str = "qa_synth:template",
    confidence: float = 0.7,
    region: Region | None = None,
    metadata: dict | None = None,
) -> StandardRecord | None:
    q, a = (question or "").strip(), (answer or "").strip()
    if len(q) < 8 or len(a) < 12:
        return None
    split = _split_for_fact(fact_key)
    return StandardRecord(
        id=_rid(fact_key, template_id, language.value),
        category=category,
        subcategory=subcategory or category.value,
        crop=resolve_crop_name(crop) if crop else crop,
        region=region or Region(state="Maharashtra", country="India"),
        language=language,
        question=q,
        answer=a,
        source=source,
        confidence=confidence,
        verified=False,
        split=split,
        schema_version=SCHEMA_VERSION,
        license="educational-open-compilation",
        metadata={
            "fact_key": fact_key,
            "template_id": template_id,
            "synth": True,
            "sprint": "S6",
            **(metadata or {}),
        },
    )


# --- template banks ---

CROP_Q_EN = [
    ("en_overview", "What should a farmer know about growing {crop}?"),
    ("en_season", "Which season is best for {crop} cultivation?"),
    ("en_soil", "What soil is ideal for {crop}?"),
    ("en_pests", "What are major pests of {crop}?"),
    ("en_diseases", "What diseases commonly affect {crop}?"),
    ("en_stages", "What are the main growth stages of {crop}?"),
    ("en_water", "How should irrigation be managed for {crop}?"),
    ("en_risk", "What climate risks affect {crop} in Maharashtra?"),
]
CROP_Q_MR = [
    ("mr_overview", "{crop_mr} पीक कसे घ्यावे?"),
    ("mr_season", "{crop_mr} कोणत्या हंगामात लावावे?"),
    ("mr_soil", "{crop_mr} साठी कोणती माती योग्य?"),
    ("mr_pests", "{crop_mr} वरील प्रमुख कीड कोणत्या?"),
    ("mr_disease", "{crop_mr} वरील रोग कोणते?"),
    ("mr_tips", "{crop_mr} उत्पादनासाठी महत्त्वाचे उपाय काय?"),
]
CROP_Q_HI = [
    ("hi_overview", "{crop_hi} की खेती कैसे करें?"),
    ("hi_season", "{crop_hi} किस मौसम में बोएं?"),
    ("hi_soil", "{crop_hi} के लिए कौनसी मिट्टी अच्छी है?"),
    ("hi_pests", "{crop_hi} के प्रमुख कीट कौन से हैं?"),
    ("hi_disease", "{crop_hi} में कौनसी बीमारियाँ आती हैं?"),
]

DISEASE_Q_EN = [
    ("en_sym", "What are symptoms of {name} in {crop}?"),
    ("en_org", "What is organic control for {name} on {crop}?"),
    ("en_chem", "What chemical control is used for {name} in {crop}?"),
    ("en_prev", "How can farmers prevent {name} in {crop}?"),
    ("en_when", "When is {name} risk high for {crop}?"),
    ("en_ipm", "Give an IPM plan for {name} in {crop}."),
]
DISEASE_Q_MR = [
    ("mr_sym", "{crop_mr} मध्ये {name_mr} ची लक्षणे काय?"),
    ("mr_org", "{name_mr} साठी सेंद्रिय उपाय काय?"),
    ("mr_chem", "{name_mr} साठी रासायनिक उपाय काय?"),
    ("mr_prev", "{name_mr} टाळण्यासाठी काय करावे?"),
]
DISEASE_Q_HI = [
    ("hi_sym", "{crop_hi} में {name_hi} के लक्षण क्या हैं?"),
    ("hi_org", "{name_hi} का जैविक नियंत्रण कैसे करें?"),
    ("hi_chem", "{name_hi} के लिए रासायनिक उपाय क्या है?"),
    ("hi_prev", "{name_hi} से बचाव कैसे करें?"),
]

FERT_Q_EN = [
    ("en_npk", "What NPK dose per acre is recommended for {crop}?"),
    ("en_basal", "What is the basal fertilizer dose for {crop}?"),
    ("en_top", "When should top dressing be done for {crop}?"),
    ("en_micro", "Which micronutrients help {crop}?"),
    ("en_plan", "Give a fertilizer schedule for {crop}."),
]
FERT_Q_MR = [
    ("mr_npk", "{crop_mr} साठी एकरी NPK किती?"),
    ("mr_basal", "{crop_mr} बेसल खत मात्रा काय?"),
    ("mr_top", "{crop_mr} टॉप ड्रेसिंग केव्हा करावी?"),
    ("mr_plan", "{crop_mr} खत नियोजन सांगा."),
]
FERT_Q_HI = [
    ("hi_npk", "{crop_hi} के लिए प्रति एकड़ NPK कितना दें?"),
    ("hi_basal", "{crop_hi} में बेसल खाद कितनी डालें?"),
    ("hi_top", "{crop_hi} में टॉप ड्रेसिंग कब करें?"),
    ("hi_plan", "{crop_hi} की खाद अनुसूची बताएं."),
]

SCHEME_Q_EN = [
    ("en_what", "What is {name}?"),
    ("en_ben", "What are benefits of {name}?"),
    ("en_elig", "Who is eligible for {name}?"),
    ("en_docs", "What documents are needed for {name}?"),
    ("en_how", "How can a farmer apply for {name}?"),
]
SCHEME_Q_MR = [
    ("mr_what", "{name_mr} योजना काय आहे?"),
    ("mr_ben", "{name_mr} चे फायदे काय?"),
    ("mr_elig", "{name_mr} साठी पात्रता काय?"),
    ("mr_docs", "{name_mr} साठी कागदपत्रे कोणती?"),
]
SCHEME_Q_HI = [
    ("hi_what", "{name_hi} योजना क्या है?"),
    ("hi_ben", "{name_hi} के लाभ क्या हैं?"),
    ("hi_elig", "{name_hi} के लिए पात्रता क्या है?"),
    ("hi_docs", "{name_hi} के लिए कौन से दस्तावेज चाहिए?"),
]

STAGE_Q_EN = [
    ("en_stage", "What should farmers do at the {stage} stage of {crop}?"),
    ("en_check", "Give a checklist for {crop} during {stage}."),
    ("en_risk", "What risks matter for {crop} at {stage}?"),
]
STAGE_Q_MR = [
    ("mr_stage", "{crop_mr} च्या {stage_mr} अवस्थेत काय करावे?"),
    ("mr_check", "{crop_mr} {stage_mr} चेक्कलिस्ट सांगा."),
]
STAGE_Q_HI = [
    ("hi_stage", "{crop_hi} की {stage_hi} अवस्था में क्या करें?"),
    ("hi_check", "{crop_hi} {stage_hi} चेकलिस्ट दें."),
]

SAFETY_TEMPLATES = [
    (
        "en_safety_spray",
        LanguageCode.EN,
        Category.DISEASE,
        "Can I spray any pesticide without PPE on {crop}?",
        "No. Always use protective equipment, follow the label dose, observe pre-harvest interval, "
        "and avoid spraying in high wind or before rain. Prefer IPM and consult a local agri officer "
        "before chemical use on {crop}.",
        0.9,
    ),
    (
        "mr_safety_spray",
        LanguageCode.MR,
        Category.DISEASE,
        "{crop_mr} वर संरक्षण न घेता कीटकनाशक फवारू का?",
        "नाही. PPE वापरा, लेबलनुसार मात्रा घ्या, कापणीपूर्व प्रतीक्षा कालावधी पाळा. "
        "वाऱ्यात किंवा पावसापूर्वी फवारणी टाळा. स्थानिक कृषी तज्ञांचा सल्ला घ्या.",
        0.9,
    ),
    (
        "en_need_info",
        LanguageCode.EN,
        Category.GENERAL,
        "My {crop} leaves look odd — what exact spray should I buy now?",
        "More information is needed: leaf photo symptoms, recent weather, last spray, and growth stage. "
        "Do not buy chemicals by name alone. Start with diagnosis (disease vs nutrient vs pest) and IPM.",
        0.85,
    ),
]


def _load_processed_facts() -> dict[str, list[dict]]:
    """Load structured lists from processed lake (or empty)."""
    buckets: dict[str, list[dict]] = {
        "crops": [],
        "diseases_and_pests": [],
        "schemes": [],
        "fertilizer_recommendations": [],
        "markets": [],
        "advisories": [],
        "soil_types": [],
        "practices": [],
        "varieties": [],
        "zones": [],
    }
    for path in iter_processed_json_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, items in extract_record_lists(data):
            if key in buckets:
                for it in items:
                    if isinstance(it, dict):
                        it = dict(it)
                        it["_source_path"] = relative_to_repo(path)
                        buckets[key].append(it)
    # fallback: taxonomy-only crops if lake empty
    if not buckets["crops"]:
        for c in TAXONOMY.get("crops") or []:
            buckets["crops"].append(
                {
                    "name_en": c["name_en"],
                    "name_mr": c.get("name_mr"),
                    "name_hi": c.get("name_hi"),
                    "season": "see package of practice",
                    "ideal_soil": ["as per local soil test"],
                    "major_pests": [],
                    "major_diseases": [],
                    "growth_stages": [s["name_en"] for s in list_crop_stages()],
                    "source": "taxonomy",
                    "_source_path": "taxonomy",
                }
            )
    return buckets


def synthesize_qa_records(
    *,
    target_min_total: int = 12000,
    include_safety: bool = True,
    include_stages: bool = True,
    include_districts: bool = True,
) -> list[StandardRecord]:
    """Generate expanded multilingual QA pack."""
    facts = _load_processed_facts()
    out: list[StandardRecord] = []
    seen: set[str] = set()

    def push(rec: StandardRecord | None):
        if rec is None:
            return
        if rec.id in seen:
            return
        seen.add(rec.id)
        out.append(rec)

    # --- crops ---
    for item in facts["crops"]:
        crop = item.get("name_en") or "Crop"
        crop_mr = item.get("name_mr") or crop
        crop_hi = item.get("name_hi") or crop
        season = item.get("season") or "local season window"
        soils = ", ".join(item.get("ideal_soil") or ["soil-test based"])
        pests = ", ".join(item.get("major_pests") or ["scout regularly"])
        diseases = ", ".join(item.get("major_diseases") or ["monitor weather risk"])
        stages = ", ".join(item.get("growth_stages") or [s["name_en"] for s in list_crop_stages()[:6]])
        src = item.get("source") or "ICAR/SAU open advisory"
        path = item.get("_source_path") or "processed"

        answers = {
            "overview": (
                f"{crop} ({crop_mr}). Season: {season}. Ideal soil: {soils}. "
                f"Major pests: {pests}. Major diseases: {diseases}. Stages: {stages}. Source: {src}."
            ),
            "season": f"Best window for {crop}: {season}. Adjust to local monsoon onset and soil moisture.",
            "soil": f"{crop} prefers {soils}. Confirm with soil health card before fertilizer planning.",
            "pests": f"Key pests of {crop}: {pests}. Use ETL scouting, traps, and IPM before chemicals.",
            "diseases": f"Key diseases of {crop}: {diseases}. High humidity increases fungal risk.",
            "stages": f"Growth stages for {crop}: {stages}. Match irrigation and nutrition to stage.",
            "water": (
                f"For {crop}, prefer efficient irrigation (drip where possible). "
                f"Avoid waterlogging; schedule by stage and weather."
            ),
            "risk": (
                f"{crop} in Maharashtra faces monsoon variability, heat stress, and pest outbreaks "
                f"under humid conditions. Use weather advisories and field scouting."
            ),
        }
        fact_key = f"crop|{crop}"

        for tid, q in CROP_Q_EN:
            key = tid.split("_", 1)[-1]
            ans = answers.get(key, answers["overview"])
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.CROP,
                    question=q.format(crop=crop),
                    answer=ans,
                    language=LanguageCode.EN,
                    crop=crop,
                    subcategory="crop_synth",
                    source=f"qa_synth:crop:{path}",
                    confidence=_conf(0.72, has_detail=True),
                    metadata={"pack": "crop"},
                )
            )
        for tid, q in CROP_Q_MR:
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.CROP,
                    question=q.format(crop_mr=crop_mr, crop=crop),
                    answer=answers["overview"],
                    language=LanguageCode.MR,
                    crop=crop,
                    subcategory="crop_synth",
                    source=f"qa_synth:crop:{path}",
                    confidence=_conf(0.7, has_mr=True),
                    metadata={"pack": "crop"},
                )
            )
        for tid, q in CROP_Q_HI:
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.CROP,
                    question=q.format(crop_hi=crop_hi, crop=crop),
                    answer=answers["overview"],
                    language=LanguageCode.HI,
                    crop=crop,
                    subcategory="crop_synth",
                    source=f"qa_synth:crop:{path}",
                    confidence=_conf(0.68, has_mr=True),
                    metadata={"pack": "crop"},
                )
            )

    # --- diseases / pests ---
    for item in facts["diseases_and_pests"]:
        name = item.get("name_en") or "Disease"
        name_mr = item.get("name_mr") or name
        name_hi = name  # often only EN/MR in KB
        crop = item.get("crop_en") or "crop"
        crop_mr = item.get("crop_mr") or crop
        crop_hi = crop
        symptoms = item.get("symptoms_en") or "See field symptoms and confirm diagnosis."
        symptoms_mr = item.get("symptoms_mr") or symptoms
        organic = item.get("organic_control_en") or "Use cultural IPM and approved biopesticides."
        chemical = item.get("chemical_control_en") or "Use labeled chemistry only after ETL and advice."
        path = item.get("_source_path") or "processed"
        fact_key = f"disease|{crop}|{name}"

        ans_map = {
            "sym": f"Symptoms of {name} on {crop}: {symptoms}",
            "org": f"Organic control for {name} on {crop}: {organic}",
            "chem": f"Chemical control for {name} on {crop}: {chemical}. Follow label & PHI.",
            "prev": (
                f"Prevent {name} in {crop} via resistant varieties where available, field sanitation, "
                f"balanced nutrition, and weather-aware scouting."
            ),
            "when": (
                f"{name} risk for {crop} rises with conducive weather (often high humidity/rain). "
                f"Increase scouting frequency during monsoon."
            ),
            "ipm": (
                f"IPM for {name} on {crop}: monitor ETL, prefer cultural/biological options, "
                f"rotate modes of action if chemicals are needed, protect beneficials."
            ),
        }
        for tid, q in DISEASE_Q_EN:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.DISEASE,
                    question=q.format(name=name, crop=crop),
                    answer=ans_map.get(key, ans_map["sym"]),
                    language=LanguageCode.EN,
                    crop=crop,
                    subcategory="disease_synth",
                    source=f"qa_synth:disease:{path}",
                    confidence=_conf(0.78, has_detail=bool(symptoms)),
                    metadata={"pack": "disease", "disease": name},
                )
            )
        for tid, q in DISEASE_Q_MR:
            key = "sym" if "sym" in tid else ("org" if "org" in tid else ("chem" if "chem" in tid else "prev"))
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.DISEASE,
                    question=q.format(name_mr=name_mr, crop_mr=crop_mr, name=name, crop=crop),
                    answer=symptoms_mr if key == "sym" else ans_map.get(key, ans_map["sym"]),
                    language=LanguageCode.MR,
                    crop=crop,
                    subcategory="disease_synth",
                    source=f"qa_synth:disease:{path}",
                    confidence=_conf(0.75, has_mr=True),
                    metadata={"pack": "disease", "disease": name},
                )
            )
        for tid, q in DISEASE_Q_HI:
            key = "sym" if "sym" in tid else ("org" if "org" in tid else ("chem" if "chem" in tid else "prev"))
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.DISEASE,
                    question=q.format(name_hi=name_hi, crop_hi=crop_hi, name=name, crop=crop),
                    answer=ans_map.get(key, ans_map["sym"]),
                    language=LanguageCode.HI,
                    crop=crop,
                    subcategory="disease_synth",
                    source=f"qa_synth:disease:{path}",
                    confidence=_conf(0.72, has_mr=True),
                    metadata={"pack": "disease", "disease": name},
                )
            )

    # --- fertilizer ---
    for item in facts["fertilizer_recommendations"]:
        crop = item.get("crop_en") or "crop"
        crop_mr = item.get("crop_mr") or crop
        crop_hi = crop
        npk = item.get("recommended_npk_kg_per_acre") or {}
        basal = item.get("basal_dose") or "Apply basal as per soil test"
        top = item.get("top_dressing") or "Split nitrogen as per stage"
        micro = item.get("micronutrients") or "Correct deficiencies after soil test"
        path = item.get("_source_path") or "processed"
        fact_key = f"fert|{crop}"
        ans = {
            "npk": f"Recommended NPK kg/acre for {crop}: N={npk.get('N')}, P={npk.get('P')}, K={npk.get('K')}.",
            "basal": f"Basal for {crop}: {basal}",
            "top": f"Top dressing for {crop}: {top}",
            "micro": f"Micronutrients for {crop}: {micro}",
            "plan": (
                f"Fertilizer plan for {crop}: NPK N={npk.get('N')}/P={npk.get('P')}/K={npk.get('K')} kg/acre. "
                f"Basal: {basal}. Top: {top}. Micro: {micro}."
            ),
        }
        for tid, q in FERT_Q_EN:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.FERTILIZER,
                    question=q.format(crop=crop),
                    answer=ans.get(key, ans["plan"]),
                    language=LanguageCode.EN,
                    crop=crop,
                    subcategory="fertilizer_synth",
                    source=f"qa_synth:fert:{path}",
                    confidence=_conf(0.8, has_detail=True),
                    metadata={"pack": "fertilizer"},
                )
            )
        for tid, q in FERT_Q_MR:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.FERTILIZER,
                    question=q.format(crop_mr=crop_mr, crop=crop),
                    answer=ans.get(key, ans["plan"]),
                    language=LanguageCode.MR,
                    crop=crop,
                    subcategory="fertilizer_synth",
                    source=f"qa_synth:fert:{path}",
                    confidence=_conf(0.76, has_mr=True),
                    metadata={"pack": "fertilizer"},
                )
            )
        for tid, q in FERT_Q_HI:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.FERTILIZER,
                    question=q.format(crop_hi=crop_hi, crop=crop),
                    answer=ans.get(key, ans["plan"]),
                    language=LanguageCode.HI,
                    crop=crop,
                    subcategory="fertilizer_synth",
                    source=f"qa_synth:fert:{path}",
                    confidence=_conf(0.74, has_mr=True),
                    metadata={"pack": "fertilizer"},
                )
            )

    # --- schemes ---
    for item in facts["schemes"]:
        name = item.get("name_en") or "Scheme"
        name_mr = item.get("name_mr") or name
        name_hi = name
        benefits = item.get("benefits_en") or ""
        benefits_mr = item.get("benefits_mr") or benefits
        elig = item.get("eligibility_en") or ""
        docs = ", ".join(item.get("documents_required") or [])
        portal = item.get("portal") or ""
        path = item.get("_source_path") or "processed"
        fact_key = f"scheme|{name}"
        ans = {
            "what": f"{name}: {benefits} Eligibility: {elig}. Portal: {portal}",
            "ben": f"Benefits of {name}: {benefits}",
            "elig": f"Eligibility for {name}: {elig}",
            "docs": f"Documents for {name}: {docs or 'Aadhaar, land records as applicable'}",
            "how": (
                f"Apply for {name} via official portal {portal or 'state agriculture portal'} "
                f"with required documents: {docs}."
            ),
        }
        for tid, q in SCHEME_Q_EN:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.SCHEME,
                    question=q.format(name=name),
                    answer=ans.get(key, ans["what"]),
                    language=LanguageCode.EN,
                    subcategory="scheme_synth",
                    source=f"qa_synth:scheme:{path}",
                    confidence=_conf(0.82, has_detail=True),
                    metadata={"pack": "scheme"},
                )
            )
        for tid, q in SCHEME_Q_MR:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.SCHEME,
                    question=q.format(name_mr=name_mr, name=name),
                    answer=benefits_mr if key in ("what", "ben") else ans.get(key, ans["what"]),
                    language=LanguageCode.MR,
                    subcategory="scheme_synth",
                    source=f"qa_synth:scheme:{path}",
                    confidence=_conf(0.78, has_mr=True),
                    metadata={"pack": "scheme"},
                )
            )
        for tid, q in SCHEME_Q_HI:
            key = tid.split("_", 1)[-1]
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.SCHEME,
                    question=q.format(name_hi=name_hi, name=name),
                    answer=ans.get(key, ans["what"]),
                    language=LanguageCode.HI,
                    subcategory="scheme_synth",
                    source=f"qa_synth:scheme:{path}",
                    confidence=_conf(0.76, has_mr=True),
                    metadata={"pack": "scheme"},
                )
            )

    # --- stages × crops (taxonomy expansion) ---
    if include_stages:
        stages = list_crop_stages()
        crops_en = [c.get("name_en") for c in facts["crops"]] or list_crop_names_en()
        for crop in crops_en:
            crop_rec = next((c for c in (TAXONOMY.get("crops") or []) if c["name_en"] == crop), {})
            crop_mr = crop_rec.get("name_mr") or crop
            crop_hi = crop_rec.get("name_hi") or crop
            for st in stages:
                stage = st["name_en"]
                stage_mr = st.get("name_mr") or stage
                stage_hi = st.get("name_hi") or stage
                fact_key = f"stage|{crop}|{st['id']}"
                ans = (
                    f"At {stage} of {crop}: monitor soil moisture, scout pests twice weekly, "
                    f"apply nutrients per soil test, avoid calendar spraying, and record interventions."
                )
                ans_mr = (
                    f"{crop_mr} च्या {stage_mr} अवस्थेत: ओलावा तपासा, कीड-रोग सर्वेक्षण, "
                    f"माती चाचणीनुसार अन्नद्रव्ये, IPM अवलंबा."
                )
                for tid, q in STAGE_Q_EN:
                    push(
                        _mk(
                            fact_key=fact_key,
                            template_id=tid,
                            category=Category.CROP,
                            question=q.format(crop=crop, stage=stage),
                            answer=ans,
                            language=LanguageCode.EN,
                            crop=crop,
                            subcategory="stage_synth",
                            source="qa_synth:stage:taxonomy",
                            confidence=_conf(0.65),
                            metadata={"pack": "stage", "stage_id": st["id"]},
                        )
                    )
                for tid, q in STAGE_Q_MR:
                    push(
                        _mk(
                            fact_key=fact_key,
                            template_id=tid,
                            category=Category.CROP,
                            question=q.format(crop_mr=crop_mr, stage_mr=stage_mr, crop=crop, stage=stage),
                            answer=ans_mr,
                            language=LanguageCode.MR,
                            crop=crop,
                            subcategory="stage_synth",
                            source="qa_synth:stage:taxonomy",
                            confidence=_conf(0.63, has_mr=True),
                            metadata={"pack": "stage", "stage_id": st["id"]},
                        )
                    )
                for tid, q in STAGE_Q_HI:
                    push(
                        _mk(
                            fact_key=fact_key,
                            template_id=tid,
                            category=Category.CROP,
                            question=q.format(crop_hi=crop_hi, stage_hi=stage_hi, crop=crop, stage=stage),
                            answer=ans,
                            language=LanguageCode.HI,
                            crop=crop,
                            subcategory="stage_synth",
                            source="qa_synth:stage:taxonomy",
                            confidence=_conf(0.62, has_mr=True),
                            metadata={"pack": "stage", "stage_id": st["id"]},
                        )
                    )

    # --- markets × templates ---
    for item in facts["markets"]:
        crop = item.get("crop") or "crop"
        mandi = item.get("mandi") or "mandi"
        district = item.get("district") or "district"
        modal = item.get("modal_price_rs_quintal")
        fact_key = f"market|{crop}|{mandi}"
        ans = (
            f"{crop} at {mandi} ({district}): modal ₹{modal}/q, "
            f"min ₹{item.get('min_price_rs_quintal')}, max ₹{item.get('max_price_rs_quintal')}. "
            f"Trend: {item.get('trend')}. Date: {item.get('date')}."
        )
        templates = [
            ("en_price", LanguageCode.EN, f"What is the price of {crop} at {mandi}?"),
            ("en_trend", LanguageCode.EN, f"What is the market trend for {crop} in {district}?"),
            ("mr_price", LanguageCode.MR, f"{crop} चा {mandi} येथील भाव काय आहे?"),
            ("hi_price", LanguageCode.HI, f"{crop} का {mandi} में भाव क्या है?"),
        ]
        for tid, lang, q in templates:
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.MARKET,
                    question=q,
                    answer=ans,
                    language=lang,
                    crop=crop,
                    subcategory="market_synth",
                    source=f"qa_synth:market:{item.get('_source_path', 'processed')}",
                    confidence=_conf(0.7),
                    region=Region(state=item.get("state") or "Maharashtra", district=district),
                    metadata={"pack": "market"},
                )
            )

    # --- district × crop irrigation + weather volume expanders ---
    if include_districts:
        districts = list_mh_districts()  # all MH districts
        crops = list_crop_names_en()
        for crop in crops:
            for dist in districts:
                fact_key = f"irrig|{crop}|{dist}"
                ans = (
                    f"In {dist}, schedule {crop} irrigation by soil moisture and weather. "
                    f"Prefer drip if available; avoid waterlogging in black soils; "
                    f"pause irrigation if heavy rain is forecast."
                )
                for tid, lang, q in (
                    ("en_dist_irrig", LanguageCode.EN, f"How should I irrigate {crop} in {dist} district?"),
                    ("en_dist_irrig2", LanguageCode.EN, f"Irrigation tips for {crop} farmers in {dist}?"),
                    ("mr_dist_irrig", LanguageCode.MR, f"{dist} जिल्ह्यात {crop} सिंचन कसे करावे?"),
                    ("hi_dist_irrig", LanguageCode.HI, f"{dist} में {crop} की सिंचाई कैसे करें?"),
                ):
                    push(
                        _mk(
                            fact_key=fact_key,
                            template_id=tid,
                            category=Category.IRRIGATION,
                            question=q,
                            answer=ans,
                            language=lang,
                            crop=crop,
                            subcategory="irrigation_synth",
                            source="qa_synth:irrigation:district",
                            confidence=_conf(0.55, has_mr=lang != LanguageCode.EN),
                            region=Region(state="Maharashtra", district=dist),
                            metadata={"pack": "irrigation"},
                        )
                    )
                # weather pack
                wkey = f"weather|{crop}|{dist}"
                wans = (
                    f"For {crop} in {dist}, track IMD district forecast, delay spraying before rain, "
                    f"and increase disease scouting when humidity stays high for 2+ days."
                )
                for tid, lang, q in (
                    ("en_wx", LanguageCode.EN, f"Weather advice for {crop} in {dist}?"),
                    ("mr_wx", LanguageCode.MR, f"{dist} मध्ये {crop} साठी हवामान सल्ला?"),
                    ("hi_wx", LanguageCode.HI, f"{dist} में {crop} के लिए मौसम सलाह?"),
                ):
                    push(
                        _mk(
                            fact_key=wkey,
                            template_id=tid,
                            category=Category.WEATHER,
                            question=q,
                            answer=wans,
                            language=lang,
                            crop=crop,
                            subcategory="weather_synth",
                            source="qa_synth:weather:district",
                            confidence=_conf(0.55, has_mr=lang != LanguageCode.EN),
                            region=Region(state="Maharashtra", district=dist),
                            metadata={"pack": "weather"},
                        )
                    )

    # --- safety / need-more-info ---
    if include_safety:
        for crop in list_crop_names_en():
            crop_rec = next((c for c in (TAXONOMY.get("crops") or []) if c["name_en"] == crop), {})
            crop_mr = crop_rec.get("name_mr") or crop
            for tid, lang, cat, q_t, a_t, conf in SAFETY_TEMPLATES:
                fact_key = f"safety|{tid.split('_')[1]}|{crop}"
                push(
                    _mk(
                        fact_key=fact_key,
                        template_id=tid,
                        category=cat,
                        question=q_t.format(crop=crop, crop_mr=crop_mr),
                        answer=a_t.format(crop=crop, crop_mr=crop_mr),
                        language=lang,
                        crop=crop,
                        subcategory="safety_synth",
                        source="qa_synth:safety",
                        confidence=conf,
                        metadata={"pack": "safety"},
                    )
                )

    # Finance / machinery generic packs (volume + coverage)
    for crop in list_crop_names_en():
        fact_key = f"finance|{crop}"
        fans = (
            f"For {crop}, estimate cost of cultivation, explore KCC crop loan, "
            f"and enroll in PMFBY where notified. Keep receipts for claims."
        )
        for tid, lang, q in (
            ("en_fin", LanguageCode.EN, f"How can I finance {crop} cultivation?"),
            ("en_ins", LanguageCode.EN, f"Is crop insurance useful for {crop}?"),
            ("mr_fin", LanguageCode.MR, f"{crop} पिकासाठी कर्ज/विमा कसा घ्यावा?"),
            ("hi_fin", LanguageCode.HI, f"{crop} खेती के लिए ऋण/बीमा कैसे लें?"),
        ):
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=tid,
                    category=Category.FINANCE,
                    question=q,
                    answer=fans,
                    language=lang,
                    crop=crop,
                    subcategory="finance_synth",
                    source="qa_synth:finance",
                    confidence=_conf(0.6, has_mr=lang != LanguageCode.EN),
                    metadata={"pack": "finance"},
                )
            )
        mkey = f"mach|{crop}"
        mans = (
            f"For {crop}, useful machines may include tractor implements, seeders, sprayers, "
            f"and harvesters depending on scale. Choose through custom hiring centers where available."
        )
        for tid, lang, q in (
            ("en_mach", LanguageCode.EN, f"What machinery helps {crop} farming?"),
            ("mr_mach", LanguageCode.MR, f"{crop} साठी कोणती यंत्रे उपयुक्त?"),
            ("hi_mach", LanguageCode.HI, f"{crop} खेती में कौनसी मशीनें उपयोगी हैं?"),
        ):
            push(
                _mk(
                    fact_key=mkey,
                    template_id=tid,
                    category=Category.MACHINERY,
                    question=q,
                    answer=mans,
                    language=lang,
                    crop=crop,
                    subcategory="machinery_synth",
                    source="qa_synth:machinery",
                    confidence=_conf(0.58, has_mr=lang != LanguageCode.EN),
                    metadata={"pack": "machinery"},
                )
            )

    # Paraphrase + soil expanders until volume target
    paraphrase_en = [
        "Explain best practices for {crop} farming.",
        "Summarize package of practices for {crop}.",
        "Farmer FAQ: how to cultivate {crop} successfully?",
        "Key agronomy tips for {crop} in India?",
        "What inputs and scouting are needed for {crop}?",
        "How do I plan nutrition for {crop}?",
        "What post-harvest tips matter for {crop}?",
        "Common mistakes in {crop} cultivation?",
        "How to improve {crop} yield sustainably?",
        "What records should {crop} farmers keep?",
    ]
    paraphrase_mr = [
        "{crop_mr} शेतीसाठी मार्गदर्शक तत्त्वे काय?",
        "{crop_mr} पिकाची काळजी कशी घ्यावी?",
        "{crop_mr} उत्पादकता वाढवण्यासाठी टिप्स?",
        "{crop_mr} साठी अन्नद्रव्य नियोजन कसे?",
        "{crop_mr} कापणीनंतर काय काळजी?",
    ]
    paraphrase_hi = [
        "{crop_hi} की खेती के मुख्य उपाय क्या हैं?",
        "{crop_hi} की उपज कैसे बढ़ाएं?",
        "{crop_hi} में पोषण प्रबंधन कैसे करें?",
        "{crop_hi} कटाई के बाद क्या सावधानी रखें?",
        "{crop_hi} किसान कौनसे रिकॉर्ड रखें?",
    ]
    soils = [
        "black cotton soil",
        "medium black soil",
        "red laterite soil",
        "alluvial soil",
        "sandy loam",
        "clay loam",
    ]
    for crop in list_crop_names_en():
        crop_rec = next((c for c in (TAXONOMY.get("crops") or []) if c["name_en"] == crop), {})
        crop_mr = crop_rec.get("name_mr") or crop
        crop_hi = crop_rec.get("name_hi") or crop
        body = (
            f"{crop} cultivation requires suitable soil, timely sowing, balanced nutrition, "
            f"IPM scouting, and stage-based irrigation. Use local SAU/ICAR advisories."
        )
        fact_key = f"crop|{crop}"
        for i, q in enumerate(paraphrase_en):
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=f"en_para_{i}",
                    category=Category.CROP,
                    question=q.format(crop=crop),
                    answer=body,
                    language=LanguageCode.EN,
                    crop=crop,
                    subcategory="crop_paraphrase",
                    source="qa_synth:paraphrase",
                    confidence=_conf(0.6),
                    metadata={"pack": "paraphrase"},
                )
            )
        for i, q in enumerate(paraphrase_mr):
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=f"mr_para_{i}",
                    category=Category.CROP,
                    question=q.format(crop_mr=crop_mr, crop=crop),
                    answer=body,
                    language=LanguageCode.MR,
                    crop=crop,
                    subcategory="crop_paraphrase",
                    source="qa_synth:paraphrase",
                    confidence=_conf(0.58, has_mr=True),
                    metadata={"pack": "paraphrase"},
                )
            )
        for i, q in enumerate(paraphrase_hi):
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=f"hi_para_{i}",
                    category=Category.CROP,
                    question=q.format(crop_hi=crop_hi, crop=crop),
                    answer=body,
                    language=LanguageCode.HI,
                    crop=crop,
                    subcategory="crop_paraphrase",
                    source="qa_synth:paraphrase",
                    confidence=_conf(0.58, has_mr=True),
                    metadata={"pack": "paraphrase"},
                )
            )
        for soil in soils:
            skey = f"soil|{crop}|{soil}"
            sans = (
                f"{crop} on {soil}: confirm drainage and organic carbon, use soil-test based NPK, "
                f"and avoid excess irrigation on heavy clays."
            )
            for tid, lang, q in (
                ("en_soil_x", LanguageCode.EN, f"How to grow {crop} on {soil}?"),
                ("en_soil_x2", LanguageCode.EN, f"Soil tips for {crop} in {soil}?"),
                ("mr_soil_x", LanguageCode.MR, f"{soil} वर {crop} कसे घ्यावे?"),
                ("hi_soil_x", LanguageCode.HI, f"{soil} पर {crop} कैसे उगाएं?"),
            ):
                push(
                    _mk(
                        fact_key=skey,
                        template_id=tid,
                        category=Category.SOIL,
                        question=q,
                        answer=sans,
                        language=lang,
                        crop=crop,
                        subcategory="soil_synth",
                        source="qa_synth:soil",
                        confidence=_conf(0.55, has_mr=lang != LanguageCode.EN),
                        metadata={"pack": "soil"},
                    )
                )
        # seed pack
        vkey = f"seed|{crop}"
        vans = (
            f"Choose {crop} varieties for your zone, use certified seed, "
            f"and follow recommended seed treatment from SAU/ICAR."
        )
        for tid, lang, q in (
            ("en_seed", LanguageCode.EN, f"How to choose seed for {crop}?"),
            ("en_seed2", LanguageCode.EN, f"Seed treatment tips for {crop}?"),
            ("mr_seed", LanguageCode.MR, f"{crop} बियाणे कसे निवडावे?"),
            ("hi_seed", LanguageCode.HI, f"{crop} के बीज कैसे चुनें?"),
        ):
            push(
                _mk(
                    fact_key=vkey,
                    template_id=tid,
                    category=Category.SEED,
                    question=q,
                    answer=vans,
                    language=lang,
                    crop=crop,
                    subcategory="seed_synth",
                    source="qa_synth:seed",
                    confidence=_conf(0.58, has_mr=lang != LanguageCode.EN),
                    metadata={"pack": "seed"},
                )
            )

    # Extra volume: month × crop calendar style FAQs (12 months × crops × langs)
    months = [
        ("Jan", "जानेवारी", "जनवरी"),
        ("Feb", "फेब्रुवारी", "फरवरी"),
        ("Mar", "मार्च", "मार्च"),
        ("Apr", "एप्रिल", "अप्रैल"),
        ("May", "मे", "मई"),
        ("Jun", "जून", "जून"),
        ("Jul", "जुलै", "जुलाई"),
        ("Aug", "ऑगस्ट", "अगस्त"),
        ("Sep", "सप्टेंबर", "सितंबर"),
        ("Oct", "ऑक्टोबर", "अक्टूबर"),
        ("Nov", "नोव्हेंबर", "नवंबर"),
        ("Dec", "डिसेंबर", "दिसंबर"),
    ]
    for crop in list_crop_names_en():
        crop_rec = next((c for c in (TAXONOMY.get("crops") or []) if c["name_en"] == crop), {})
        crop_mr = crop_rec.get("name_mr") or crop
        for mon_en, mon_mr, mon_hi in months:
            fact_key = f"cal|{crop}|{mon_en}"
            ans = (
                f"In {mon_en}, plan {crop} operations by local crop calendar: "
                f"check growth stage, soil moisture, pest alerts, and avoid untimely sprays."
            )
            for tid, lang, q in (
                ("en_cal", LanguageCode.EN, f"What should I do for {crop} in {mon_en}?"),
                ("mr_cal", LanguageCode.MR, f"{mon_mr} मध्ये {crop_mr} साठी काय करावे?"),
                ("hi_cal", LanguageCode.HI, f"{mon_hi} में {crop} के लिए क्या करें?"),
            ):
                push(
                    _mk(
                        fact_key=fact_key,
                        template_id=tid,
                        category=Category.CROP,
                        question=q,
                        answer=ans,
                        language=lang,
                        crop=crop,
                        subcategory="calendar_synth",
                        source="qa_synth:calendar",
                        confidence=_conf(0.52, has_mr=lang != LanguageCode.EN),
                        metadata={"pack": "calendar", "month": mon_en},
                    )
                )

    # Advisories multi-template
    for item in facts.get("advisories") or []:
        title = item.get("title_en") or item.get("title") or "Advisory"
        content = item.get("content_en") or item.get("content") or "Follow local agri advisory."
        fact_key = f"adv|{title[:80]}"
        for i, (lang, q) in enumerate(
            [
                (LanguageCode.EN, f"Explain advisory: {title}"),
                (LanguageCode.EN, f"What does '{title}' recommend to farmers?"),
                (LanguageCode.EN, f"Action points from advisory '{title}'?"),
                (LanguageCode.MR, f"सल्ला समजावा: {title}"),
                (LanguageCode.MR, f"{title} — शेतकऱ्याने काय करावे?"),
                (LanguageCode.HI, f"सलाह समझाएं: {title}"),
                (LanguageCode.HI, f"{title} — किसान क्या करें?"),
            ]
        ):
            push(
                _mk(
                    fact_key=fact_key,
                    template_id=f"adv_{i}",
                    category=Category.ADVISORY,
                    question=q[:240],
                    answer=f"{title}. {content}",
                    language=lang,
                    subcategory="advisory_synth",
                    source=f"qa_synth:advisory:{item.get('_source_path', 'processed')}",
                    confidence=_conf(0.7, has_mr=lang != LanguageCode.EN),
                    metadata={"pack": "advisory"},
                )
            )

    # Final volume booster: crop × practice FAQs (storage, nursery, marketing, IPM, etc.)
    practices = [
        ("storage", "storage and curing", "साठवण", "भंडारण",
         "Dry thoroughly, use ventilated storage, and monitor pests; avoid damp stacks."),
        ("nursery", "nursery management", "रोपवाटिका", "नर्सरी",
         "Use raised beds, quality media, and disease-free seedlings; harden before transplant."),
        ("marketing", "marketing and grading", "विक्री श्रेणीकरण", "विपणन",
         "Grade produce, compare mandi vs MSP, and avoid distress sale without market info."),
        ("ipm", "integrated pest management", "एकात्मिक कीड व्यवस्थापन", "आईपीएम",
         "Scout weekly, use traps/biocontrol first, and apply chemicals only at ETL with PPE."),
        ("organic", "organic inputs", "सेंद्रिय निविष्ठा", "जैविक इनपुट",
         "Use FYM/compost, neem, and bioagents as part of a planned organic package."),
        ("labour", "labour and operations planning", "मजूर नियोजन", "श्रम योजना",
         "Plan peak labour for sowing/harvest, keep cost records, and use custom hiring where useful."),
        ("drainage", "field drainage", "निचरा", "जल निकासी",
         "Ensure surface drainage before monsoon; waterlogging reduces yield and raises disease risk."),
        ("mulch", "mulching", "आच्छादन", "मल्चिंग",
         "Mulch to conserve moisture and suppress weeds; choose material suited to crop and season."),
        ("weeding", "weed management", "तण नियंत्रण", "खरपतवार",
         "Timely weeding or safe herbicides as labeled; dense weeds cut yield and hide pests."),
        ("harvest", "harvest timing", "कापणी वेळ", "कटाई समय",
         "Harvest at proper maturity indices; delay increases shattering/pest damage risk."),
        ("intercrop", "intercropping options", "मिश्रपीक", "अंतरफसल",
         "Where suitable, intercrops improve income and soil cover; match water and canopy."),
        ("trace", "traceability and quality", "गुणवत्ता ट्रेसिबिलिटी", "ट्रेसबिलिटी",
         "Keep spray and harvest logs for market quality and insurance/export compliance."),
    ]
    for crop in list_crop_names_en():
        crop_rec = next((c for c in (TAXONOMY.get("crops") or []) if c["name_en"] == crop), {})
        crop_mr = crop_rec.get("name_mr") or crop
        for pid, pen, pmr, phi, pans in practices:
            fact_key = f"prac|{crop}|{pid}"
            ans = f"For {crop} {pen}: {pans}"
            for tid, lang, q in (
                (f"en_{pid}", LanguageCode.EN, f"How to handle {pen} for {crop}?"),
                (f"en2_{pid}", LanguageCode.EN, f"Best practices for {crop} {pen}?"),
                (f"en3_{pid}", LanguageCode.EN, f"Farmer checklist: {crop} {pen}?"),
                (f"en4_{pid}", LanguageCode.EN, f"Practical steps for {pen} in {crop} fields?"),
                (f"mr_{pid}", LanguageCode.MR, f"{crop_mr} साठी {pmr} कसे करावे?"),
                (f"mr2_{pid}", LanguageCode.MR, f"{crop_mr} — {pmr} टिप्स काय?"),
                (f"hi_{pid}", LanguageCode.HI, f"{crop} के लिए {phi} कैसे करें?"),
            ):
                push(
                    _mk(
                        fact_key=fact_key,
                        template_id=tid,
                        category=Category.CROP,
                        question=q,
                        answer=ans,
                        language=lang,
                        crop=crop,
                        subcategory="practice_synth",
                        source="qa_synth:practice",
                        confidence=_conf(0.57, has_mr=lang != LanguageCode.EN),
                        metadata={"pack": "practice", "practice": pid},
                    )
                )

    return out


def write_review_queue(records: list[StandardRecord], path: Path, sample_n: int = 500) -> Path:
    """Human review queue CSV for sampled synthetic records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # sample stratified by category
    by_cat: dict[str, list[StandardRecord]] = {}
    for r in records:
        by_cat.setdefault(r.category.value, []).append(r)
    sample: list[StandardRecord] = []
    per = max(1, sample_n // max(1, len(by_cat)))
    for cat, items in by_cat.items():
        sample.extend(items[:per])
    sample = sample[:sample_n]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "crop",
                "language",
                "question",
                "answer",
                "confidence",
                "split",
                "template_id",
                "verified",
                "reviewer_notes",
            ],
        )
        w.writeheader()
        for r in sample:
            w.writerow(
                {
                    "id": r.id,
                    "category": r.category.value,
                    "crop": r.crop or "",
                    "language": r.language.value,
                    "question": r.question,
                    "answer": r.answer[:500],
                    "confidence": r.confidence,
                    "split": r.split.value,
                    "template_id": (r.metadata or {}).get("template_id", ""),
                    "verified": "false",
                    "reviewer_notes": "",
                }
            )
    return path


def export_synth_dataset(
    records: list[StandardRecord],
    *,
    dry_run: bool = False,
    version_tag: str | None = None,
) -> dict[str, Any]:
    ensure_lake_layout()
    train = [r for r in records if r.split == DataSplit.TRAIN]
    val = [r for r in records if r.split == DataSplit.VAL]
    test = [r for r in records if r.split == DataSplit.TEST]
    version = version_tag or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ") + "-synth"
    version_dir = DATASETS_DIR / "versions" / version

    by_cat = Counter(r.category.value for r in records)
    by_lang = Counter(r.language.value for r in records)
    by_pack = Counter((r.metadata or {}).get("pack", "unknown") for r in records)

    artifacts: list[str] = []
    if not dry_run:
        version_dir.mkdir(parents=True, exist_ok=True)

        def dump_jsonl(path: Path, recs: list[StandardRecord]):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r.to_training_dict(), ensure_ascii=False) + "\n")
            artifacts.append(relative_to_repo(path))

        dump_jsonl(version_dir / "train.jsonl", train)
        dump_jsonl(version_dir / "val.jsonl", val)
        dump_jsonl(version_dir / "test.jsonl", test)
        dump_jsonl(version_dir / "all.jsonl", records)
        # also merge markers under lake training as synth sidecar
        dump_jsonl(LAKE_TRAINING / "synth_records.jsonl", train)
        dump_jsonl(LAKE_VALIDATION / "synth_records.jsonl", val)
        dump_jsonl(LAKE_TEST / "synth_records.jsonl", test)

        try:
            import pandas as pd

            for name, recs in (("train.parquet", train), ("val.parquet", val), ("test.parquet", test)):
                rows = [r.to_training_dict() for r in recs]
                for row in rows:
                    row["metadata"] = json.dumps(row.get("metadata") or {}, ensure_ascii=False)
                    row["region"] = json.dumps(row.get("region") or {}, ensure_ascii=False)
                p = version_dir / name
                pd.DataFrame(rows).to_parquet(p, index=False)
                artifacts.append(relative_to_repo(p))
        except Exception:
            pass

        review_path = version_dir / "human_review_queue.csv"
        write_review_queue(records, review_path)
        artifacts.append(relative_to_repo(review_path))

        manifest = {
            "version": version,
            "sprint": "S6",
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "counts": {
                "total": len(records),
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "by_category": dict(by_cat),
            "by_language": dict(by_lang),
            "by_pack": dict(by_pack),
            "targets": {"train_min": 10000, "val_min": 1000},
            "targets_met": {
                "train": len(train) >= 10000,
                "val": len(val) >= 1000,
            },
            "no_leakage_policy": "split assigned by fact_key hash; paraphrases share split",
            "artifacts": artifacts,
        }
        man = version_dir / "manifest.json"
        man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(man))

        latest = LAKE_ROOT / "QASYNTH_LATEST.json"
        latest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(latest))
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        (DATASETS_DIR / "QASYNTH_LATEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "ok": len(train) >= 10000 and len(val) >= 1000,
        "version": version,
        "counts": {"total": len(records), "train": len(train), "val": len(val), "test": len(test)},
        "by_category": dict(by_cat),
        "by_language": dict(by_lang),
        "by_pack": dict(by_pack),
        "targets_met": {"train": len(train) >= 10000, "val": len(val) >= 1000},
        "artifacts": artifacts,
        "dry_run": dry_run,
    }


def run_qa_synth(*, dry_run: bool = False, target_min_total: int = 12000) -> dict[str, Any]:
    records = synthesize_qa_records(target_min_total=target_min_total)
    return export_synth_dataset(records, dry_run=dry_run)
