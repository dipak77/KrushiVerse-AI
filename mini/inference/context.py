"""Context builder with citation markers for Mini+RAG (Sprint 15)."""

from __future__ import annotations

from typing import Any


def normalize_sources(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize heterogeneous RAG citation / doc dicts into a stable shape."""
    out: list[dict[str, Any]] = []
    for i, c in enumerate(raw or []):
        if not isinstance(c, dict):
            continue
        sid = str(c.get("id") or c.get("doc_id") or f"src-{i + 1}")
        title = str(c.get("title") or c.get("name") or c.get("source") or sid)
        text = str(
            c.get("text")
            or c.get("snippet")
            or c.get("content")
            or c.get("chunk")
            or ""
        ).strip()
        origin = str(c.get("origin") or c.get("source") or "local")
        url = c.get("url") or c.get("source_url")
        score = c.get("score")
        out.append(
            {
                "id": sid,
                "marker": f"[{i + 1}]",
                "title": title[:160],
                "text": text[:600],
                "origin": origin,
                "url": url,
                "score": score,
            }
        )
    return out


def build_context_pack(
    *,
    query: str,
    sources: list[dict[str, Any]],
    agent_notes: list[str] | None = None,
    max_sources: int = 6,
    max_chars: int = 1800,
) -> dict[str, Any]:
    """Build a cited context block Mini should ground on.

    Format:
      [1] Title — origin
      snippet...
    """
    sources = sources[:max_sources]
    blocks: list[str] = []
    used = 0
    cited: list[dict[str, Any]] = []
    for s in sources:
        marker = s.get("marker") or f"[{len(cited) + 1}]"
        head = f"{marker} {s.get('title') or s.get('id')} — {s.get('origin') or 'local'}"
        body = (s.get("text") or "").strip()
        if not body:
            body = "(no snippet; title-only source)"
        chunk = f"{head}\n{body}"
        if used + len(chunk) > max_chars and cited:
            break
        blocks.append(chunk)
        used += len(chunk) + 2
        cited.append({**s, "marker": marker})

    if agent_notes:
        note = "Agent notes:\n" + "\n".join(f"- {n}" for n in agent_notes[:4])
        if used + len(note) <= max_chars + 200:
            blocks.append(note)

    context_text = "\n\n".join(blocks).strip()
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question:\n{query.strip()}\n\n"
        "Answer using ONLY the context. Cite sources like [1], [2]. "
        "If context is insufficient, say you need more information."
    )
    return {
        "context_text": context_text,
        "user_prompt": user_prompt,
        "citations": cited,
        "n_sources": len(cited),
        "has_sources": len(cited) > 0,
    }


def format_answer_with_citations(answer: str, citations: list[dict[str, Any]]) -> str:
    ans = (answer or "").strip()
    if not citations:
        return ans
    lines = [ans, "", "**Sources:**"]
    for c in citations[:8]:
        marker = c.get("marker") or ""
        title = c.get("title") or c.get("id")
        origin = c.get("origin") or ""
        url = c.get("url") or ""
        if url:
            lines.append(f"{marker} {title} — {origin} ({url})")
        else:
            lines.append(f"{marker} {title} — {origin}")
    return "\n".join(lines)
