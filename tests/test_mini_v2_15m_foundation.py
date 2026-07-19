"""Mini v2 15M foundation — config, param count, assistant-only collate, cleanup dry-run."""

from __future__ import annotations

from mini.models.config import MiniConfig, load_config_json
from mini.models.model import MiniLM, count_parameters
from mini.models.sft import _assistant_start_index, _collate
from mini.models.corpus import DomainTokenizer
from mini.tools.cleanup_local import plan_versions_cleanup, run_cleanup
import torch


def test_v2_config_loads():
    cfg = MiniConfig.v2_15m()
    assert cfg.vocab_size == 8192
    assert cfg.n_embd == 320
    assert cfg.n_layer == 10
    assert cfg.n_head == 8
    assert cfg.n_hidden == 864
    assert cfg.block_size == 1024
    assert cfg.tie_weights is True
    assert cfg.n_embd % cfg.n_head == 0


def test_v2_param_count_near_15m():
    cfg = MiniConfig.v2_15m()
    model = MiniLM(cfg)
    stats = count_parameters(model)
    unique = int(stats.get("unique_params") or stats.get("trainable_tensors_sum") or 0)
    # ~15.0M ± 15% (tying / implementation detail)
    assert 12_000_000 <= unique <= 18_000_000, unique
    assert unique / 1e6 > 10


def test_gradient_checkpoint_forward():
    cfg = MiniConfig.v2_15m()
    cfg.gradient_checkpointing = True
    # tiny forward with smaller block for speed
    cfg.block_size = 64
    model = MiniLM(cfg)
    model.train()
    x = torch.randint(4, cfg.vocab_size, (2, 32))
    y = x.clone()
    y[:, :8] = -100  # ignore prefix
    logits, loss = model(x, y)
    assert logits.shape[-1] == cfg.vocab_size
    assert loss is not None
    loss.backward()


def test_assistant_only_collate_masks_prompt():
    tok = DomainTokenizer(vocab_size=256)
    tok.build(
        [
            "### System: You are Krushi Mitra. ### User: cotton bollworm? ### Assistant: Use traps and ETL.",
            "### Assistant:",
            "System User Assistant cotton traps",
        ],
        min_freq=1,
    )
    text = (
        "### System:\nYou are Krushi Mitra.\n\n"
        "### User:\ncotton bollworm?\n\n"
        "### Assistant:\nUse traps and ETL."
    )
    ids = tok.encode(text, add_special=True)
    assert len(ids) >= 8
    start = _assistant_start_index(ids, tok)
    assert start > 0
    t = torch.tensor(ids, dtype=torch.long)
    meta = {"text": text, "_tokenizer": tok}
    inp, lab, _ = _collate([(t, meta)], pad_id=tok.pad_id, assistant_only=True, ignore_index=-100)
    # Many labels should be ignored (prompt)
    ignored = (lab == -100).sum().item()
    assert ignored >= 1
    # At least one non-ignored token (assistant)
    assert (lab != -100).sum().item() >= 1


def test_cleanup_dry_run():
    plan = plan_versions_cleanup(keep_synth=1, keep_other=1)
    assert "mb_freeable" in plan
    report = run_cleanup(dry_run=True, keep_synth=1, keep_other=1, smoke_ckpts=True, runs=False)
    assert report["dry_run"] is True
    assert report.get("ok") is True
