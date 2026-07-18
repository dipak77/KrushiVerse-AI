"""Source registry loader for Mini data lake ingest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini.paths import MINI_ROOT, REPO_ROOT

DEFAULT_REGISTRY_PATH = MINI_ROOT / "lake" / "sources.json"


class SourceSpec(BaseModel):
    id: str
    name: str
    type: str  # local_file | http_api | web
    domain: str
    also_domains: list[str] = Field(default_factory=list)
    path: str | None = None
    url: str | None = None
    format: str = "json"
    license: str = "unknown"
    origin: str | None = None
    enabled: bool = True
    requires_env: list[str] = Field(default_factory=list)
    notes: str | None = None

    def all_domains(self) -> list[str]:
        domains = [self.domain]
        for d in self.also_domains:
            if d not in domains:
                domains.append(d)
        return domains

    def env_satisfied(self) -> bool:
        for key in self.requires_env:
            val = os.getenv(key)
            if not val or val in ("YOUR_API_KEY", "demo", "changeme"):
                return False
        return True

    def resolve_path(self) -> Path | None:
        if not self.path:
            return None
        p = Path(self.path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p


class SourceRegistry(BaseModel):
    version: str = "1.0.0"
    sprint: str = "S2"
    description: str = ""
    sources: list[SourceSpec] = Field(default_factory=list)

    def enabled_sources(self, *, types: set[str] | None = None) -> list[SourceSpec]:
        out = []
        for s in self.sources:
            if not s.enabled:
                continue
            if types and s.type not in types:
                continue
            out.append(s)
        return out

    def get(self, source_id: str) -> SourceSpec | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        for s in self.sources:
            by_type[s.type] = by_type.get(s.type, 0) + 1
            for d in s.all_domains():
                by_domain[d] = by_domain.get(d, 0) + 1
        return {
            "version": self.version,
            "sprint": self.sprint,
            "total_sources": len(self.sources),
            "enabled": sum(1 for s in self.sources if s.enabled),
            "by_type": by_type,
            "by_domain": by_domain,
        }


def load_source_registry(path: Path | None = None) -> SourceRegistry:
    reg_path = path or DEFAULT_REGISTRY_PATH
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(data)
