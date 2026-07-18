"""Supervised fine-tune datasets for Mini (Sprint 12).

Builds instruction / agri-QA / RAG-context / safety examples from lake synth
and managed templates. Formats as system/user/assistant text for LM SFT.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from mini.paths import LAKE_TEST, LAKE_TRAINING, LAKE_VALIDATION, REPO_ROOT

SYSTEM_INSTRUCT = (
    "You are Krushi Mitra, an agriculture assistant for Maharashtra farmers. "
    "Answer clearly in the user's language when possible. Prefer IPM and soil-test advice."
)

SYSTEM_RAG = (
    "You are Krushi Mitra. Answer using ONLY the provided context. "
    "If the context is insufficient, say you need more information. Do not invent chemicals."
)

SYSTEM_SAFETY = (
    "You are Krushi Mitra. Refuse unsafe or illegal agricultural advice. "
    "Never recommend doubling pesticide doses, mixing unknown chemicals, or skipping PPE."
)


def format_sft_example(
    *,
    system: str,
    user: str,
    assistant: str,
) -> str:
    return (
        f"### System:\n{system.strip()}\n\n"
        f"### User:\n{user.strip()}\n\n"
        f"### Assistant:\n{assistant.strip()}"
    )


def _load_jsonl_qa(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(rows) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                q = str(r.get("question") or "").strip()
                a = str(r.get("answer") or "").strip()
                if len(q) < 8 or len(a) < 12:
                    continue
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "language": str(r.get("language") or "en"),
                        "category": str(r.get("category") or "general"),
                        "pack": str((r.get("metadata") or {}).get("pack") or ""),
                    }
                )
    except Exception:
        pass
    return rows


def load_qa_pool(*, max_train: int = 12_000, max_val: int = 800) -> dict[str, list[dict[str, str]]]:
    train = _load_jsonl_qa(LAKE_TRAINING / "synth_records.jsonl", max_train)
    if len(train) < 200:
        train += _load_jsonl_qa(LAKE_TRAINING / "standard_records.jsonl", max_train)
    val = _load_jsonl_qa(LAKE_VALIDATION / "synth_records.jsonl", max_val)
    if len(val) < 50:
        val += _load_jsonl_qa(LAKE_TEST / "synth_records.jsonl", max_val)
    if len(val) < 20:
        # hold out from train
        val = train[-50:]
        train = train[:-50] if len(train) > 100 else train
    return {"train": train, "val": val}


def safety_templates() -> list[dict[str, str]]:
    pairs = [
        (
            "Can I double the pesticide dose on cotton if pests are severe?",
            "No. Do not double doses. Follow the label rate, rotate modes of action, verify ETL, and consult a local agri officer.",
        ),
        (
            "Should I mix leftover unknown chemicals and spray tonight?",
            "No. Never tank-mix unknown leftovers. Incompatible mixes can burn crops and are unsafe. Dispose per label/local rules.",
        ),
        (
            "Is PPE optional for a small field spray?",
            "No. PPE is required regardless of field size. Follow label PPE, avoid wind drift, and wash after spraying.",
        ),
        (
            "Can children help mix pesticide?",
            "No. Keep children and untrained persons away from pesticide mixing and application. Store chemicals locked and labeled.",
        ),
        (
            "कापूस वर डोस दुप्पट करू का?",
            "नाही. लेबल डोसच पाळा. जास्त डोस अवशेष, प्रतिरोध आणि पीक नुकसान वाढवतो. स्थानिक तज्ञ सल्ला घ्या.",
        ),
        (
            "कोई भी दवा बिना निदान के छिड़क दूं?",
            "नहीं। पहले सही निदान करें। बिना लक्षण/मौसम जानकारी दवा न खरीदें। आईपीएम अपनाएं।",
        ),
        (
            "Rain is coming in 1 hour — spray now?",
            "Usually no. Avoid spraying just before rain; wash-off wastes chemical and reduces control. Reschedule after a dry window.",
        ),
        (
            "Can I reuse empty pesticide containers for drinking water?",
            "Never. Triple-rinse and dispose per rules. Never reuse for food or water.",
        ),
    ]
    out = []
    for q, a in pairs:
        out.append({"question": q, "answer": a, "language": "mixed", "category": "safety", "pack": "safety"})
    return out


def build_rag_context_examples(qa: list[dict[str, str]], *, n: int = 400, seed: int = 42) -> list[dict[str, str]]:
    """Create answer-from-context SFT pairs by wrapping the gold answer as context."""
    rng = random.Random(seed)
    pool = [r for r in qa if r.get("pack") not in ("safety", "hard_negative")]
    if not pool:
        pool = list(qa)
    out: list[dict[str, str]] = []
    for _ in range(min(n, max(1, len(pool)))):
        r = rng.choice(pool)
        # distractor snippet
        other = rng.choice(pool)
        context = (
            f"Source note: {r['answer'][:320]}\n"
            f"Related: {other['answer'][:120]}"
        )
        user = f"Context:\n{context}\n\nQuestion:\n{r['question']}\n\nCite the context in your answer."
        # Encourage citing context
        ans = f"Based on the context: {r['answer']}"
        out.append(
            {
                "question": user,
                "answer": ans,
                "language": r.get("language") or "en",
                "category": "rag",
                "pack": "rag_context",
                "system": SYSTEM_RAG,
            }
        )
    return out


def build_sft_records(
    *,
    max_train: int = 8_000,
    max_val: int = 600,
    seed: int = 42,
    include_rag: bool = True,
    include_safety: bool = True,
) -> dict[str, Any]:
    """Return formatted train/val strings + gold pairs for metrics."""
    rng = random.Random(seed)
    pool = load_qa_pool(max_train=max_train * 2, max_val=max_val * 2)
    train_qa = list(pool["train"])
    val_qa = list(pool["val"])
    rng.shuffle(train_qa)

    safety = safety_templates() if include_safety else []
    train_qa = train_qa + safety * 3  # upsample safety
    rng.shuffle(train_qa)

    rag_train = build_rag_context_examples(train_qa, n=min(500, max(50, len(train_qa) // 10)), seed=seed) if include_rag else []
    rag_val = build_rag_context_examples(val_qa, n=min(80, max(10, len(val_qa) // 5)), seed=seed + 1) if include_rag else []

    def to_text(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for r in rows[:limit]:
            system = r.get("system") or (
                SYSTEM_SAFETY if r.get("pack") in ("safety", "hard_negative") else SYSTEM_INSTRUCT
            )
            text = format_sft_example(system=system, user=r["question"], assistant=r["answer"])
            out.append(
                {
                    "text": text,
                    "question": r["question"],
                    "answer": r["answer"],
                    "language": r.get("language") or "en",
                    "pack": r.get("pack") or "qa",
                    "category": r.get("category") or "general",
                }
            )
        return out

    # Multilingual preference: keep mix as in source
    train_rows = to_text(train_qa, max_train) + to_text(rag_train, min(400, max_train // 5))
    val_rows = to_text(val_qa, max_val) + to_text(rag_val, min(60, max_val // 4))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    # Ensure minimum size for tiny lakes
    if len(train_rows) < 32:
        for s in safety:
            train_rows.append(
                {
                    "text": format_sft_example(system=SYSTEM_SAFETY, user=s["question"], assistant=s["answer"]),
                    "question": s["question"],
                    "answer": s["answer"],
                    "language": "en",
                    "pack": "safety",
                    "category": "safety",
                }
            )
        # generic instruct
        for crop in ("Cotton", "Soybean", "Pomegranate", "Onion"):
            q = f"What is a basic IPM tip for {crop}?"
            a = f"For {crop}, scout regularly, use ETL thresholds, prefer traps/biocontrol, and spray labeled products only when needed."
            train_rows.append(
                {
                    "text": format_sft_example(system=SYSTEM_INSTRUCT, user=q, assistant=a),
                    "question": q,
                    "answer": a,
                    "language": "en",
                    "pack": "qa",
                    "category": "pest",
                }
            )

    by_lang = {}
    by_pack = {}
    for r in train_rows:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        by_pack[r["pack"]] = by_pack.get(r["pack"], 0) + 1

    return {
        "train": train_rows,
        "val": val_rows,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "by_language": by_lang,
            "by_pack": by_pack,
        },
    }


def prompt_only(text: str) -> str:
    """Strip assistant completion for generation prompts."""
    marker = "### Assistant:\n"
    if marker in text:
        return text.split(marker, 1)[0] + marker
    return text + "\n\n### Assistant:\n"


def token_f1(pred: str, gold: str) -> float:
    pt = set(re.findall(r"\w+", (pred or "").lower()))
    gt = set(re.findall(r"\w+", (gold or "").lower()))
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
