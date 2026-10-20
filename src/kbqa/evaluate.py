"""Phase 3/6 - evaluate one system (model x answering mode) on the FIXED evaluation set.

Run examples:
  uv run python -m kbqa.evaluate --name retrieval-only --mode retrieval
  uv run python -m kbqa.evaluate --name qwen3-0.6b/closed --model Qwen/Qwen3-0.6B --mode closed
  uv run python -m kbqa.evaluate --name qwen3-0.6b/rag    --model Qwen/Qwen3-0.6B --mode rag
  uv run python -m kbqa.evaluate --name sft-lora/closed   --model Qwen/Qwen3-0.6B --adapter outputs/sft-lora --mode closed

Outputs:
  results/raw/<name>.jsonl    every question, prediction and per-item score (contains KBA text -> git-ignored)
  results/eval_results.csv    one summary row per system (numbers only -> safe to commit)

Evaluation set (fixed, seed 42): 120 answerable + all unanswerable test_unseen questions, all of test_gold,
80 sampled test_seen questions.
Generation is greedy (deterministic), max 150 new tokens, identical prompts for every model.
"""
import argparse
import hashlib
import json
import random
import time

import pandas as pd
import torch

from kbqa.config import QA_DIR, RESULTS_DIR, SEED
from kbqa.metrics import score_items

N_SEEN = 80
N_UNSEEN_ANSWERABLE = 120   # + all 32 unanswerable; caps evaluation time on the slow laptop


def read(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


N_VAL = 120


def eval_set(which="test"):
    """test: the final report set. val: a 120-question sample of the validation split, used ONLY to
    choose models / retriever, so the test set never influences a decision."""
    if which == "val":
        val = read(QA_DIR / "val.jsonl")
        una = [r for r in val if r["kind"] == "unanswerable"]          # all of them: refusals matter
        ans = [r for r in val if r["kind"] != "unanswerable"]
        random.Random(SEED).shuffle(ans)
        return sorted(una + ans[:N_VAL - len(una)], key=lambda r: r["qa_id"])
    rng = random.Random(SEED)
    seen = read(QA_DIR / "test_seen.jsonl")
    rng.shuffle(seen)
    unseen = read(QA_DIR / "test_unseen.jsonl")
    una = [r for r in unseen if r["kind"] == "unanswerable"]          # keep every unanswerable question
    ans = [r for r in unseen if r["kind"] != "unanswerable"]
    rng.shuffle(ans)
    unseen = sorted(una + ans[:N_UNSEEN_ANSWERABLE], key=lambda r: r["qa_id"])
    return unseen + read(QA_DIR / "test_gold.jsonl") + sorted(seen[:N_SEEN], key=lambda r: r["qa_id"])


def eval_set_hash(items):
    return hashlib.sha256("|".join(i["qa_id"] + i["question"] for i in items).encode()).hexdigest()[:12]


def summarise(rows, items):
    """Aggregate per test set. 'gold' = SAP's FAQ questions, 'seen'/'unseen' = trained / never-trained facts."""
    df = pd.DataFrame(rows)
    out = {}
    groups = (("unseen", df["split"] == "test_unseen"), ("seen", df["split"] == "test_seen"),
              ("gold", df["split"] == "test_gold"), ("val", df["split"] == "val"))
    for name, mask in groups:
        part = df[mask]
        ans = part[part["answerable"]]
        out[f"{name}_n"] = len(ans)
        for m in ("token_f1", "rouge_l", "semantic_sim", "key_fact_recall"):
            out[f"{name}_{m}"] = round(float(ans[m].dropna().mean()), 4) if len(ans) else None
        out[f"{name}_false_refusal"] = round(float(ans["abstained"].mean()), 4) if len(ans) else None
    una = df[~df["answerable"]]
    out["unanswerable_n"] = len(una)
    out["unanswerable_correct_refusal"] = round(float(una["abstained"].mean()), 4) if len(una) else None
    return out


def retrieval_recall(items, retrieved):
    """Share of answerable questions whose source chunk is among the retrieved chunks (recall@3)."""
    hits = [bool(set(it["source_chunk_ids"]) & set(r)) for it, r in zip(items, retrieved)
            if it["kind"] != "unanswerable" and r]
    return round(sum(hits) / len(hits), 4) if hits else None


def run(name, mode, model_id=None, adapter=None, quant_4bit=False, which="test", limit=None):
    from kbqa import qa_system
    items = eval_set(which)[:limit]
    questions = [it["question"] for it in items]
    load_s, peak_gb, model = None, None, None
    retriever = None
    if mode in ("rag", "retrieval"):
        from kbqa.retrieval import Retriever, load_chunks
        retriever = Retriever(load_chunks())
    if mode in ("closed", "rag"):
        from kbqa.llm import load_model
        torch.cuda.reset_peak_memory_stats()
        model, tok, load_s = load_model(model_id, quant_4bit=quant_4bit, adapter_dir=adapter)
    t0 = time.perf_counter()
    if mode == "closed":
        outs, retrieved, gen_s = qa_system.answer_closed_book(model, tok, questions)
    elif mode == "rag":
        outs, retrieved, gen_s = qa_system.answer_rag(model, tok, retriever, questions)
    else:
        outs, retrieved, gen_s = qa_system.answer_retrieval_only(retriever, questions)
    if model is not None:
        peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    preds = [o["text"] for o in outs]
    rows = score_items(items, preds)

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / f"{which}__{name.replace('/', '__')}.jsonl").open("w", encoding="utf-8") as f:
        for it, o, r, row in zip(items, outs, retrieved, rows):
            f.write(json.dumps({**row, "question": it["question"], "reference": it["answer"], "prediction": o["text"],
                                "retrieved": r, "output_tokens": o["output_tokens"]}, ensure_ascii=False) + "\n")

    out_tokens = sum(o["output_tokens"] for o in outs)
    summary = {"system": name, "eval": which, "mode": mode, "model": model_id or "-", "adapter": adapter or "-",
               "quant_4bit": quant_4bit, "eval_set": eval_set_hash(items), "n_items": len(items),
               **summarise(rows, items), "retrieval_recall@3": retrieval_recall(items, retrieved),
               "load_s": round(load_s, 1) if load_s else None, "batch_eval_s": round(gen_s, 1),
               "batch_output_tok_per_s": round(out_tokens / gen_s, 1) if out_tokens else None,
               "avg_output_tokens": round(out_tokens / len(items), 1), "peak_vram_gb": peak_gb,
               "date": time.strftime("%Y-%m-%d")}
    path = RESULTS_DIR / ("eval_results.csv" if which == "test" else "eval_results_val.csv")
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[table["system"] != name]
    pd.concat([table, pd.DataFrame([summary])]).to_csv(path, index=False)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["closed", "rag", "retrieval"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--quant4", action="store_true")
    ap.add_argument("--set", choices=["test", "val"], default="test")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N questions")
    a = ap.parse_args()
    run(a.name, a.mode, a.model, a.adapter, a.quant4, a.set, a.limit)
