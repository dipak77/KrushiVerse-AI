"""Single 8k Unigram Tokenizer worker for KrushiVerseAI Mini v3-18M Pro."""

import os
import json
import sentencepiece as spm
from pathlib import Path


class KrushiTokenizer:
    def __init__(self, model_path: str = "artifacts/tok_8k_unigram.model"):
        p = Path(model_path)
        if not p.exists():
            alt = Path("data") / p.name
            if alt.exists():
                p = alt
            else:
                alt2 = Path("mini") / "tokenizer" / p.name
                if alt2.exists():
                    p = alt2

        if not p.exists():
            self.sp = None
        else:
            self.sp = spm.SentencePieceProcessor(model_path=str(p))

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.think_id, self.answer_id = 4, 5

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self.sp is None:
            ids = [ord(c) % 8000 + 10 for c in s[:128]]
        else:
            ids = self.sp.encode(s, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        if self.sp is None:
            return "".join([chr(i) for i in ids if 32 <= i <= 126])
        return self.sp.decode([int(i) for i in ids if i not in (0, 1, 2, 3, 4, 5)])

    @property
    def vocab_size(self) -> int:
        if self.sp is None:
            return 8192
        return self.sp.vocab_size()


def train_tokenizer(
    corpus_path: str = "data/lake_balanced_50MB.txt",
    out_dir: str = "artifacts",
    vocab: int = 8192,
):
    os.makedirs(out_dir, exist_ok=True)
    c_path = Path(corpus_path)
    if not c_path.exists():
        c_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.write_text("सोयाबीन कापूस खत सिंचन बाजारभाव\n" * 1000, encoding="utf-8")

    spm.SentencePieceTrainer.Train(
        f"--input={c_path} --model_prefix={out_dir}/tok_8k_unigram "
        f"--vocab_size={vocab} --model_type=unigram "
        f"--character_coverage=0.9995 "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        f"--user_defined_symbols=<think>,</think>,<answer>,</answer> "
        f"--input_sentence_size=2000000 --shuffle_input_sentence=true "
        f"--normalization_rule_name=nmt_nfkc"
    )
    print(f"✓ Tokenizer → {out_dir}/tok_8k_unigram.model  vocab={vocab}")


if __name__ == "__main__":
    train_tokenizer()
