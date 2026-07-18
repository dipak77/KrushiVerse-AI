"""Template synthesizer fallback when Mini confidence is low (Sprint 15)."""

from __future__ import annotations

from typing import Any


def template_synthesize(
    *,
    query: str,
    intent: str,
    crops: list[str],
    context_text: str,
    citations: list[dict[str, Any]],
    language: str = "en",
    reason: str = "low_confidence",
) -> dict[str, Any]:
    """Build a grounded template answer from retrieved context snippets."""
    crop = crops[0] if crops else "the crop"
    snippets: list[str] = []
    for c in citations[:4]:
        snip = (c.get("text") or "").strip()
        if snip:
            snippets.append(f"{c.get('marker', '')} {snip[:280]}")
    ctx_preview = "\n".join(snippets) if snippets else (context_text[:500] if context_text else "")

    if language in {"mr", "hi"} and language == "mr":
        body = (
            f"संदर्भ स्रोत वापरून सल्ला ({crop} / {intent}):\n"
            f"{ctx_preview or 'अधिक संदर्भ आवश्यक आहे.'}\n\n"
            "कृपया लेबल डोस, PPE आणि स्थानिक कृषी अधिकारी मार्गदर्शन पाळा. "
            "डोस दुप्पट करू नका किंवा अज्ञात रसायने मिसळू नका."
        )
    else:
        body = (
            f"Based on retrieved sources for {crop} ({intent}):\n"
            f"{ctx_preview or 'More context is needed to answer safely.'}\n\n"
            "Follow labeled rates, use PPE, and consult a local agri officer. "
            "Never exceed the label rate; never tank-mix unidentified products."
        )

    if not citations:
        body = (
            "I cannot provide a grounded answer without sources. "
            "Please rephrase or enable retrieval. For chemical use, always follow the product label."
        )

    # ensure citation markers appear for grounding when sources exist
    if citations and not any(m in body for m in ("[1]", "[2]", "[3]")):
        markers = " ".join(str(c.get("marker") or "") for c in citations[:3]).strip()
        if markers:
            body = body + f"\n\nCitations: {markers}"

    return {
        "answer": body.strip(),
        "engine": "template_fallback",
        "reason": reason,
        "citations": citations,
    }
