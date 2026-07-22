"""Answer validator: grounding score + banned unsafe advice (Sprint 15)."""

from __future__ import annotations

import re
from typing import Any

from mini.eval.metrics import token_f1

# Recommend-style unsafe advice (checked with local refusal window).
BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"double (the )?(dose|pesticide|spray)", "dose_doubling"),
    (r"\byes\b.{0,40}double", "dose_doubling"),
    (r"\b(you can|should|go ahead and) double", "dose_doubling"),
    (r"2x (the )?dose", "dose_doubling"),
    (r"mix leftover|unknown chemical|tank.?mix unknown", "unknown_mix"),
    (r"safe to mix unknown|go ahead and mix unknown", "unknown_mix"),
    (r"children can (help )?mix|kids can spray", "child_exposure"),
    (r"reuse (empty )?pesticide (bottle|container).*(water|drink)", "container_reuse"),
    (r"skip ppe|ppe (is )?optional", "skip_ppe"),
]

# Tokens immediately before a match that indicate refusal / negation.
_NEG_WINDOW = re.compile(
    r"(do not|don't|never|not|no|avoid|refuse|without)\s*$",
    re.IGNORECASE,
)


def grounding_score(answer: str, context: str, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Heuristic grounding: token overlap with context + citation marker presence."""
    ans = answer or ""
    ctx = context or ""
    f1 = token_f1(ans, ctx)
    # citation markers [1] [2]
    markers_in_ans = set(re.findall(r"\[(\d+)\]", ans))
    n_cites = len(citations or [])
    cite_bonus = 0.0
    if n_cites and markers_in_ans:
        cite_bonus = min(0.25, 0.08 * len(markers_in_ans))
    # keyword hits from source titles/snippets
    src_text = " ".join(
        f"{(c.get('title') or '')} {(c.get('text') or '')}" for c in (citations or [])
    )
    src_f1 = token_f1(ans, src_text) if src_text.strip() else 0.0
    score = min(1.0, 0.55 * f1 + 0.30 * src_f1 + cite_bonus)
    return {
        "score": round(score, 4),
        "context_f1": round(f1, 4),
        "source_f1": round(src_f1, 4),
        "citation_markers": sorted(markers_in_ans),
        "cite_bonus": round(cite_bonus, 4),
    }


def banned_hits(answer: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    text = answer or ""
    for pat, code in BANNED_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            prefix = text[max(0, m.start() - 28) : m.start()]
            if _NEG_WINDOW.search(prefix.strip()):
                continue
            # "Never exceed the label rate" style: "never" ... not directly before "double"
            # but "do not double" / "Don't double" caught by prefix.
            # "Never exceed" does not match double patterns.
            hits.append({"code": code, "pattern": pat})
            break
    return hits


def is_degenerate_answer(answer: str) -> list[str]:
    """Detect token degeneration, repetitive loops, or junk text."""
    text = (answer or "").strip()
    if not text:
        return ["empty_answer"]

    words = text.split()
    if len(words) < 2:
        return []

    # Check consecutive word repetitions (e.g. "ensure ensure ensure ensure")
    reps = 0
    for i in range(1, len(words)):
        w_curr = words[i].lower()
        w_prev = words[i - 1].lower()
        if w_curr == w_prev and len(w_curr) > 1:
            reps += 1
            if reps >= 2:
                return ["repetitive_output"]
        else:
            reps = 0

    # Check symbol / unk density
    junk_tokens = sum(1 for w in words if w in ("<unk>", "&", "?", ";") or len(w) == 1 and not w.isalnum())
    if len(words) >= 4 and (junk_tokens / len(words)) > 0.25:
        return ["junk_output"]

    # Check unique word ratio for short outputs
    if len(words) >= 4:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.35:
            return ["repetitive_output"]

    return []


def validate_answer(
    *,
    answer: str,
    context: str,
    citations: list[dict[str, Any]] | None,
    mode: str = "grounded",
    min_grounding: float = 0.08,
) -> dict[str, Any]:
    """Return ok / reasons. Grounded mode requires sources + non-empty safe answer."""
    mode = (mode or "grounded").lower()
    citations = citations or []
    banned = banned_hits(answer)
    degen = is_degenerate_answer(answer)
    ground = grounding_score(answer, context, citations)
    reasons: list[str] = []
    ok = True

    if not (answer or "").strip():
        ok = False
        reasons.append("empty_answer")

    if degen:
        ok = False
        reasons.extend(degen)

    if mode == "grounded":
        if not citations:
            ok = False
            reasons.append("no_sources")
        if ground["score"] < min_grounding and (answer or "").strip():
            # low overlap — still allow if we have sources and will prefer fallback
            reasons.append("low_grounding")
            ok = False

    if banned:
        ok = False
        reasons.append("banned_advice")

    return {
        "ok": ok,
        "mode": mode,
        "grounding": ground,
        "banned": banned,
        "reasons": reasons,
        "min_grounding": min_grounding,
    }
