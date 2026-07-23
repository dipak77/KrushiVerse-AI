import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser("KrushiVerse v3-18M Pro Orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    tok_p = sub.add_parser("tokenizer")
    tok_p.add_argument("--vocab", type=int, default=8192)
    tok_p.add_argument("--model-type", default="unigram")
    
    pp = sub.add_parser("pretrain")
    pp.add_argument("--config", default="configs/config_v3_18M_pro.json")
    pp.add_argument("--steps", type=int, default=15000)
    pp.add_argument("--batch", type=int, default=8)
    pp.add_argument("--dtype", default="fp16")
    pp.add_argument("--seed", type=int, default=42)
    
    ps = sub.add_parser("sft")
    ps.add_argument("--config", default="configs/config_v3_18M_pro.json")
    ps.add_argument("--steps-v03", type=int, default=3000)
    ps.add_argument("--steps-v04", type=int, default=3000)
    ps.add_argument("--steps-reasoning", type=int, default=3000)
    
    pe = sub.add_parser("eval")
    pe.add_argument("--version", default="v3-18M-pro")
    pe.add_argument("--profile", default="strict")
    
    pq = sub.add_parser("quant")
    pq.add_argument("--format", default="int8-onnx")
    
    a = p.parse_args()
    if a.cmd == "tokenizer":
        from mini.workers.tokenizer import train_tokenizer
        train_tokenizer(vocab=a.vocab)
    elif a.cmd == "pretrain":
        from mini.workers.training import Trainer
        Trainer(a.config).pretrain("data/lake_balanced_50MB.bin", steps=a.steps, batch=a.batch)
    elif a.cmd == "sft":
        from mini.workers.training import Trainer
        t = Trainer(a.config)
        t.sft("data/sft_stage1_instruct.json", a.steps_v03, "s1", "artifacts/sft_s1.pt")
        t.sft("data/sft_stage2_rag.json", a.steps_v04, "s2", "artifacts/sft_s2.pt")
        t.sft("data/sft_stage3_reason.json", a.steps_reasoning, "s3", "artifacts/sft_s3_final.pt")
    elif a.cmd == "eval":
        from mini.workers.tester import KrushiBulkTester
        t = KrushiBulkTester()
        q_path = Path("bulk_queries_30.json")
        if not q_path.exists():
            q_path = Path("data/bulk_30.json")
        if q_path.exists():
            with open(q_path, encoding="utf-8") as f:
                Q = json.load(f)
            rep = t.run_bulk(Q)
            Path("artifacts/bulk_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(rep["summary"], indent=2))
        else:
            print("No bulk queries dataset found to evaluate.")
    elif a.cmd == "quant":
        from mini.workers.quant import quantize
        quantize(fmt=a.format)

if __name__ == "__main__":
    main()
