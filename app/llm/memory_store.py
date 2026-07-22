"""Working Memory & Conversational Context Store.

Tracks user farm profile (location, acreage, crop history) across multi-turn sessions
for hyper-personalized Meta AI / Gemini / ChatGPT level responses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserSessionMemory:
    session_id: str
    location: str | None = "Pune"
    acres: float | None = 1.0
    primary_crops: list[str] = field(default_factory=list)
    last_query: str | None = None
    query_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class WorkingMemoryStore:
    """In-memory session context store for personalizing micro-LLM advisory."""

    def __init__(self, max_history_per_session: int = 10):
        self._sessions: dict[str, UserSessionMemory] = {}
        self.max_history = max_history_per_session

    def get_or_create_session(self, session_id: str = "default_user") -> UserSessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = UserSessionMemory(session_id=session_id)
        return self._sessions[session_id]

    def update_session(
        self,
        session_id: str,
        *,
        query: str,
        crop: str | None = None,
        location: str | None = None,
        acres: float | None = None,
        response_summary: str | None = None,
    ) -> UserSessionMemory:
        mem = self.get_or_create_session(session_id)
        mem.last_query = query
        mem.updated_at = time.time()

        if location:
            mem.location = location
        if acres and acres > 0:
            mem.acres = acres
        if crop and crop not in mem.primary_crops:
            mem.primary_crops.append(crop)

        mem.query_history.append({
            "timestamp": mem.updated_at,
            "query": query,
            "crop": crop,
            "location": location,
            "acres": acres,
            "summary": response_summary,
        })

        if len(mem.query_history) > self.max_history:
            mem.query_history = mem.query_history[-self.max_history:]

        return mem

    def build_memory_context_prompt(self, session_id: str = "default_user") -> str:
        mem = self.get_or_create_session(session_id)
        facts = []
        if mem.location:
            facts.append(f"Location: {mem.location}")
        if mem.acres:
            facts.append(f"Land Size: {mem.acres} acre(s)")
        if mem.primary_crops:
            facts.append(f"Primary Crops: {', '.join(mem.primary_crops)}")

        if not facts:
            return ""

        return "👤 **वापरकर्ता संदर्भ (User Profile Context):** " + " | ".join(facts)


user_memory_store = WorkingMemoryStore()
