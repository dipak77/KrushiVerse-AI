"""Lightweight language detection for agri text (en / mr / hi / mixed) — Sprint 4."""

from __future__ import annotations

import re
from collections import Counter

from mini.contracts import LanguageCode

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")

# High-signal function words (not exhaustive; good enough for factory tagging)
_HI_WORDS = {
    "है", "हैं", "क्या", "की", "के", "का", "में", "और", "नहीं", "यह", "वह",
    "लिए", "से", "पर", "भी", "एक", "को", "था", "थे", "हो", "कर", "गया",
}
_MR_WORDS = {
    "आहे", "आहेत", "काय", "ची", "चे", "चा", "मध्ये", "आणि", "नाही", "हे", "ते",
    "साठी", "पासून", "वर", "एक", "ला", "होते", "कर", "गेला", "पाहिजे", "शेतकरी",
}


def detect_language(text: str) -> LanguageCode:
    """Detect primary language of a string."""
    if not text or not str(text).strip():
        return LanguageCode.UNKNOWN

    t = str(text)
    dev = len(_DEVANAGARI.findall(t))
    lat = len(_LATIN.findall(t))
    total = max(dev + lat, 1)

    # Tokenize loosely for Devanagari/Latin words
    tokens = re.findall(r"[\w\u0900-\u097F]+", t.lower())
    hi_hits = sum(1 for w in tokens if w in _HI_WORDS)
    mr_hits = sum(1 for w in tokens if w in _MR_WORDS)

    if dev == 0 and lat > 0:
        return LanguageCode.EN
    if dev > 0 and lat > 0 and lat / total > 0.25 and dev / total > 0.25:
        return LanguageCode.MIXED
    if dev > 0:
        if mr_hits > hi_hits:
            return LanguageCode.MR
        if hi_hits > mr_hits:
            return LanguageCode.HI
        # Default Devanagari agri content in this platform is Marathi-first
        return LanguageCode.MR
    if lat > 0:
        return LanguageCode.EN
    return LanguageCode.UNKNOWN


def detect_language_pair(question: str, answer: str) -> LanguageCode:
    """Combine Q/A language signals."""
    lq = detect_language(question)
    la = detect_language(answer)
    if lq == la:
        return lq
    if LanguageCode.MIXED in (lq, la):
        return LanguageCode.MIXED
    if {lq, la} <= {LanguageCode.MR, LanguageCode.HI, LanguageCode.MIXED}:
        # both Indic but disagree
        return LanguageCode.MIXED if lq != la else lq
    if LanguageCode.EN in (lq, la) and LanguageCode.UNKNOWN not in (lq, la):
        if lq != LanguageCode.EN and la != LanguageCode.EN:
            return LanguageCode.MIXED
        # one en one indic
        return LanguageCode.MIXED
    return lq if lq != LanguageCode.UNKNOWN else la


def language_distribution(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        lang = r.get("language") or detect_language(
            f"{r.get('question', '')} {r.get('answer', '')}"
        )
        if hasattr(lang, "value"):
            lang = lang.value
        counts[str(lang)] += 1
    return dict(counts)
