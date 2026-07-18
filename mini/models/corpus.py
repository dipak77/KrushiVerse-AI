"""Agri text corpus + packing for Mini pretrain (Sprint 11).

Builds lines from lake QA, platform KB, and tokenizer corpus when present.
Provides a vocab-sized domain tokenizer (default 4096) so MiniLM stays ~1M params.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from mini.paths import (
    DATASETS_DIR,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    REPO_ROOT,
    TOKENIZER_DIR,
)

_WORD = re.compile(r"[\w\u0900-\u097F]+|[^\s\w]", re.UNICODE)


def iter_agri_text_lines(
    *,
    max_qa: int = 40_000,
    max_kb: int = 20_000,
) -> list[str]:
    """Collect plain-text lines for pretraining."""
    lines: list[str] = []

    # Existing tokenizer factory corpus if present
    for p in (
        TOKENIZER_DIR / "v0.1" / "corpus.txt",
        TOKENIZER_DIR / "corpus.txt",
    ):
        if p.exists():
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        t = line.strip()
                        if len(t) >= 12:
                            lines.append(t)
                        if i >= 80_000:
                            break
            except Exception:
                pass
            if lines:
                break

    # QA / standard records
    n_qa = 0
    for base in (LAKE_TRAINING, LAKE_VALIDATION, LAKE_TEST):
        for name in ("synth_records.jsonl", "standard_records.jsonl"):
            path = base / name
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if n_qa >= max_qa:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        for k in ("question", "answer"):
                            t = str(row.get(k) or "").strip()
                            if len(t) >= 12:
                                lines.append(t)
                                n_qa += 1
            except Exception:
                continue

    # Platform KB JSON
    data_dir = REPO_ROOT / "data"
    n_kb = 0
    for name in (
        "crops_and_diseases.json",
        "soil_and_fertilizers.json",
        "government_schemes.json",
        "agri_advisories.json",
        "irrigation_practices.json",
        "seed_varieties.json",
        "market_prices.json",
    ):
        p = data_dir / name
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in _walk_strings(obj):
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) >= 20:
                lines.append(s)
                n_kb += 1
                if n_kb >= max_kb:
                    break
        if n_kb >= max_kb:
            break

    # KG triples
    triples = DATASETS_DIR / "kg" / "graph_triples.jsonl"
    if triples.exists():
        try:
            with open(triples, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 3000:
                        break
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    t = str(row.get("text") or row.get("answer") or "").strip()
                    if len(t) >= 12:
                        lines.append(t)
        except Exception:
            pass

    # Dedup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out


def _walk_strings(obj: Any, out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_strings(v, out)
    return out


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "") if m.group(0).strip()]


class DomainTokenizer:
    """Frequency-based word tokenizer that fits MiniConfig.vocab_size (e.g. 4096)."""

    def __init__(
        self,
        *,
        vocab_size: int = 4096,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {
            pad_id: "<pad>",
            unk_id: "<unk>",
            bos_id: "<s>",
            eos_id: "</s>",
        }

    def build(self, lines: list[str], min_freq: int = 2) -> "DomainTokenizer":
        ctr: Counter[str] = Counter()
        for ln in lines:
            ctr.update(tokenize_words(ln))
        # reserve specials
        next_id = 4
        for tok, freq in ctr.most_common(self.vocab_size * 2):
            if freq < min_freq:
                break
            if next_id >= self.vocab_size:
                break
            if tok in self.token_to_id:
                continue
            self.token_to_id[tok] = next_id
            self.id_to_token[next_id] = tok
            next_id += 1
        return self

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for t in tokenize_words(text):
            ids.append(self.token_to_id.get(t, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            parts.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainTokenizer":
        tok = cls(
            vocab_size=int(d.get("vocab_size") or 4096),
            pad_id=int(d.get("pad_id") or 0),
            unk_id=int(d.get("unk_id") or 1),
            bos_id=int(d.get("bos_id") or 2),
            eos_id=int(d.get("eos_id") or 3),
        )
        tok.token_to_id = {str(k): int(v) for k, v in (d.get("token_to_id") or {}).items()}
        tok.id_to_token = {
            tok.pad_id: "<pad>",
            tok.unk_id: "<unk>",
            tok.bos_id: "<s>",
            tok.eos_id: "</s>",
        }
        for t, i in tok.token_to_id.items():
            tok.id_to_token[int(i)] = t
        return tok

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DomainTokenizer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def pack_token_ids(
    docs_ids: list[list[int]],
    *,
    block_size: int,
    pad_id: int = 0,
) -> list[list[int]]:
    """Pack variable-length docs into fixed-length blocks (context packing)."""
    stream: list[int] = []
    for ids in docs_ids:
        stream.extend(ids)
    if not stream:
        return []
    blocks: list[list[int]] = []
    for i in range(0, len(stream) - 1, block_size):
        chunk = stream[i : i + block_size]
        if len(chunk) < block_size:
            chunk = chunk + [pad_id] * (block_size - len(chunk))
        # need at least 2 non-pad for a training signal
        if sum(1 for x in chunk if x != pad_id) >= 8:
            blocks.append(chunk)
    return blocks


def split_blocks(
    blocks: list[list[int]],
    *,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[int]]]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(blocks)))
    rng.shuffle(idx)
    n_val = max(1, int(len(blocks) * val_frac)) if len(blocks) > 10 else max(1, len(blocks) // 10 or 1)
    n_val = min(n_val, max(1, len(blocks) - 1)) if len(blocks) > 1 else 0
    val_idx = set(idx[:n_val])
    train = [blocks[i] for i in range(len(blocks)) if i not in val_idx]
    val = [blocks[i] for i in range(len(blocks)) if i in val_idx]
    if not train and blocks:
        train = blocks[:-1] or blocks
        val = blocks[-1:]
    return train, val


def prepare_pretrain_data(
    *,
    vocab_size: int = 4096,
    block_size: int = 128,
    seed: int = 42,
    max_qa: int = 30_000,
) -> dict[str, Any]:
    """Build lines → tokenizer → packed train/val blocks."""
    lines = iter_agri_text_lines(max_qa=max_qa)
    if len(lines) < 50:
        # synthetic agri fallback so training always has mass
        crops = ["Cotton", "Soybean", "Pomegranate", "Onion", "Wheat", "Rice"]
        tips = [
            "apply basal NPK after soil test",
            "scout pests at ETL before spraying",
            "use drip irrigation in dry spell",
            "compare mandi modal price before sale",
            "enroll PMFBY in notified window",
        ]
        for c in crops:
            for t in tips:
                lines.append(f"{c} advisory: {t}. Maharashtra package of practice.")
                lines.append(f"{c} शेती सल्ला: {t}")

    tok = DomainTokenizer(vocab_size=vocab_size).build(lines, min_freq=1 if len(lines) < 500 else 2)
    docs_ids = [tok.encode(ln) for ln in lines]
    # drop empty-ish
    docs_ids = [d for d in docs_ids if len(d) >= 6]
    blocks = pack_token_ids(docs_ids, block_size=block_size, pad_id=tok.pad_id)
    train, val = split_blocks(blocks, val_frac=0.1, seed=seed)
    return {
        "lines": len(lines),
        "docs": len(docs_ids),
        "blocks": len(blocks),
        "train_blocks": len(train),
        "val_blocks": len(val),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "tokenizer": tok,
        "train": train,
        "val": val,
    }
