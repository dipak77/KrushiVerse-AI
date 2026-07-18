"""Web & open-web retrieval tools (DuckDuckGo, Wikipedia) with offline fallbacks."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote_plus

import httpx

from app.config import settings


class WebSearchProvider:
    """
    Multi-backend web search for advanced RAG.
    Uses free/open endpoints — no API key required for core paths.
    """

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self.cache_ttl = getattr(settings, "WEB_CACHE_TTL_SEC", 300)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        cache_key = f"{query}::{max_results}"
        now = time.time()
        if cache_key in self._cache:
            ts, hits = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return hits

        results: list[dict] = []
        results.extend(self._duckduckgo_instant(query, max_results=max_results))
        if len(results) < max_results:
            results.extend(self._wikipedia_search(query, max_results=max_results - len(results)))
        if len(results) < 2:
            results.extend(self._offline_web_stubs(query, max_results=max_results))

        # Deduplicate by URL/title
        seen = set()
        unique = []
        for r in results:
            key = (r.get("url") or r.get("title") or "").lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        unique = unique[:max_results]
        self._cache[cache_key] = (now, unique)
        return unique

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": "KrushiVerse-AI-RAG/10 (+https://github.com/local; educational)"},
            follow_redirects=True,
        )

    def _duckduckgo_instant(self, query: str, max_results: int = 5) -> list[dict]:
        out: list[dict] = []
        try:
            with self._client() as client:
                r = client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                )
                if r.status_code != 200:
                    return out
                data = r.json()
                if data.get("AbstractText"):
                    out.append({
                        "title": data.get("Heading") or query,
                        "snippet": data.get("AbstractText"),
                        "url": data.get("AbstractURL") or "https://duckduckgo.com/",
                        "source": "DuckDuckGo Instant Answer",
                        "provider": "duckduckgo",
                    })
                for topic in data.get("RelatedTopics", [])[: max_results + 2]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        out.append({
                            "title": (topic.get("Text") or "")[:80],
                            "snippet": topic.get("Text"),
                            "url": topic.get("FirstURL") or "https://duckduckgo.com/",
                            "source": "DuckDuckGo Related",
                            "provider": "duckduckgo",
                        })
                    elif isinstance(topic, dict) and "Topics" in topic:
                        for t in topic["Topics"][:2]:
                            if t.get("Text"):
                                out.append({
                                    "title": t.get("Text", "")[:80],
                                    "snippet": t.get("Text"),
                                    "url": t.get("FirstURL") or "https://duckduckgo.com/",
                                    "source": "DuckDuckGo Related",
                                    "provider": "duckduckgo",
                                })
        except Exception as e:
            out.append({
                "title": "DuckDuckGo unavailable",
                "snippet": f"Live web search failed ({type(e).__name__}); using local+fallback knowledge.",
                "url": "",
                "source": "web_error",
                "provider": "duckduckgo",
            })
        return out[:max_results]

    def _wikipedia_search(self, query: str, max_results: int = 3) -> list[dict]:
        out: list[dict] = []
        agri_query = f"{query} agriculture India"
        try:
            with self._client() as client:
                # Prefer English Wikipedia; also try Marathi lightly for bilingual queries
                for lang in ("en", "mr"):
                    r = client.get(
                        f"https://{lang}.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": agri_query if lang == "en" else query,
                            "srlimit": max_results,
                            "format": "json",
                            "utf8": 1,
                        },
                    )
                    if r.status_code != 200:
                        continue
                    hits = r.json().get("query", {}).get("search", [])
                    for h in hits:
                        title = h.get("title", "")
                        snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                        out.append({
                            "title": title,
                            "snippet": snippet,
                            "url": f"https://{lang}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                            "source": f"Wikipedia ({lang})",
                            "provider": "wikipedia",
                        })
                    if out:
                        break
        except Exception:
            pass
        return out[:max_results]

    def _offline_web_stubs(self, query: str, max_results: int = 3) -> list[dict]:
        """Curated open-source pointers when network is unavailable."""
        stubs = [
            {
                "title": "ICAR open agricultural knowledge hub",
                "snippet": f"For query '{query}': consult ICAR institute package-of-practices, IPM guides, and SAU advisories (public educational sources).",
                "url": "https://icar.org.in",
                "source": "Open Source Catalog (offline)",
                "provider": "offline_catalog",
            },
            {
                "title": "Agmarknet mandi prices",
                "snippet": "Daily Indian mandi arrivals and modal prices — open government market intelligence.",
                "url": "https://agmarknet.gov.in",
                "source": "Open Source Catalog (offline)",
                "provider": "offline_catalog",
            },
            {
                "title": "IMD weather & agromet advisories",
                "snippet": "India Meteorological Department district forecasts and agro-met advisories for irrigation/pest timing.",
                "url": "https://mausam.imd.gov.in",
                "source": "Open Source Catalog (offline)",
                "provider": "offline_catalog",
            },
            {
                "title": "data.gov.in agriculture datasets",
                "snippet": "Government open datasets for crops, production, and schemes usable for RAG grounding.",
                "url": "https://data.gov.in",
                "source": "Open Source Catalog (offline)",
                "provider": "offline_catalog",
            },
        ]
        q = query.lower()
        ranked = stubs
        if any(k in q for k in ("price", "mandi", "market", "भाव")):
            ranked = [stubs[1], stubs[0], stubs[3]]
        elif any(k in q for k in ("weather", "rain", "हवामान", "पाऊस")):
            ranked = [stubs[2], stubs[0], stubs[1]]
        elif any(k in q for k in ("scheme", "subsidy", "योजना")):
            ranked = [stubs[3], stubs[0], stubs[1]]
        return ranked[:max_results]

    def to_rag_docs(self, hits: list[dict]) -> list[dict]:
        docs = []
        for i, h in enumerate(hits):
            docs.append({
                "id": f"web_{i}_{abs(hash(h.get('url') or h.get('title') or i)) % 10_000_000}",
                "category": "Web",
                "title": h.get("title") or "Web result",
                "content": f"{h.get('title', '')}. {h.get('snippet', '')} Source: {h.get('source')} URL: {h.get('url', '')}",
                "metadata": h,
                "source": h.get("source", "web"),
                "url": h.get("url", ""),
                "provider": h.get("provider", "web"),
            })
        return docs


web_search_provider = WebSearchProvider()
