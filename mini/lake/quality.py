"""Text cleaning utilities for lake processing (Sprint 3)."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

# Common HTML / boilerplate patterns
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"(?is)<script[^>]*>.*?</script>")
_STYLE_RE = re.compile(r"(?is)<style[^>]*>.*?</style>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(value: str) -> str:
    """Strip HTML, normalize unicode/whitespace, drop control chars."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = unicodedata.normalize("NFKC", value)
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _CTRL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def clean_value(obj: Any) -> Any:
    """Recursively clean strings inside JSON-compatible structures."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, list):
        return [clean_value(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): clean_value(v) for k, v in obj.items()}
    return obj


def record_fingerprint_text(record: Any) -> str:
    """Flatten a record into normalized text for near-duplicate comparison."""
    parts: list[str] = []

    def walk(o: Any):
        if isinstance(o, str):
            t = clean_text(o).lower()
            if t:
                parts.append(t)
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=lambda x: str(x)):
                if str(k) in ("raw", "metadata", "source_path"):
                    continue
                walk(o[k])
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(record)
    return " ".join(parts)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
