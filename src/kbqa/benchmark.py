"""Phase 3/6 - inference latency & memory benchmark for one system (single-user, batch size 1).

Run:  uv run python -m kbqa.benchmark --name qwen3-0.6b/rag --model Qwen/Qwen3-0.6B --mode rag [--adapter DIR]
Out:  results/latency_results.csv (one row per system), results/latency_raw/<name>.json (every request)

Procedure (same for every system, see docs/README.md §10):
1. Model load time is measured separately (from_pretrained -> GPU, synchronised).
2. 2 warm-up requests are run and discarded (CUDA kernels, caches).
3. N_REQUESTS fixed questions from test_unseen, one at a time, greedy, max 150 new tokens.
   For each request we measure, with torch.cuda.synchronize() around the GPU work:
   - retrieval time (RAG only)
   - TTFT  = time from submitting the prompt until the first generated token exists
             (measured with a streamer callback that records the first token's arrival)
   - total = end-to-end time inside the backend (retrieval + tokenisation + generation)
   - output tokens -> decode speed = (output_tokens - 1) / (total - TTFT - retrieval)
4. Mean, median and p95 are reported. Peak VRAM = torch.cuda.max_memory_allocated during the run.
Measurements are made on the owner's laptop with the conditions recorded in the row (free RAM).
"""
import argparse
import json
import statistics
import time

import pandas as pd
import psutil
import torch
from transformers import TextStreamer

from kbqa.config import QA_DIR, RESULTS_DIR

N_WARMUP = 2
N_REQUESTS = 10


class FirstTokenTimer(TextStreamer):
    """A streamer is called by generate() as tokens appear; we only record when the first one arrives."""

    def __init__(self, tok):
        super().__init__(tok, skip_prompt=True)
        self.first = None
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:        # the first call carries the prompt tokens
            self.prompt_seen = True
            return
        if self.first is None:
            torch.cuda.synchronize()
            self.first = time.perf_counter()

    def end(self):
        pass


def percentile(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def run(name, mode, model_id, adapter=None, quant_4bit=False, gate=False):
    from kbqa import qa_system
    from kbqa.llm import load_model
    questions = [json.loads(l)["question"] for l in (QA_DIR / "test_unseen.jsonl").open(encoding="utf-8")]
    questions = questions[: N_WARMUP + N_REQUESTS]
    retriever = None
    if mode == "rag":
        from kbqa.retrieval import Retriever, load_chunks
        retriever = Retriever(load_chunks())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tok, load_s = load_model(model_id, quant_4bit=quant_4bit, adapter_dir=adapter)
    threshold = qa_system.load_gate_threshold() if gate else None

    records = []
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        r_s = 0.0
        if gate and qa_system.gate_refuses(retriever, q, threshold):
            torch.cuda.synchronize()
            total = time.perf_counter() - t0          # refused without calling the LLM
            rec = {"input_tokens": 0, "output_tokens": 0, "retrieval_s": total, "ttft_s": total,
                   "total_s": total, "decode_tok_per_s": None}
            if i >= N_WARMUP:
                records.append(rec)
            continue
        if mode == "rag":
            hits = retriever.search(q, k=qa_system.RAG_K)
            r_s = time.perf_counter() - t0
            prompt = qa_system.rag_prompt(tok, q, hits)
        else:
            prompt = qa_system.closed_book_prompt(tok, q)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        timer = FirstTokenTimer(tok)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=qa_system.MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id, streamer=timer)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        n_out = out.shape[1] - inputs["input_ids"].shape[1]
        ttft = (timer.first or time.perf_counter()) - t0
        decode_s = max(total - ttft, 1e-6)
        rec = {"input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(n_out),
               "retrieval_s": r_s, "ttft_s": ttft, "total_s": total,
               "decode_tok_per_s": (n_out - 1) / decode_s if n_out > 1 else None}
        if i >= N_WARMUP:
            records.append(rec)

    def stats(key):
        vals = [r[key] for r in records if r[key] is not None]
        return (round(statistics.mean(vals), 3), round(statistics.median(vals), 3), round(percentile(vals, 95), 3))

    ttft, total, speed = stats("ttft_s"), stats("total_s"), stats("decode_tok_per_s")
    row = {"system": name, "mode": mode, "model": model_id, "adapter": adapter or "-", "quant_4bit": quant_4bit,
           "n_requests": len(records), "load_s": round(load_s, 2),
           "ttft_mean_s": ttft[0], "ttft_median_s": ttft[1], "ttft_p95_s": ttft[2],
           "e2e_mean_s": total[0], "e2e_median_s": total[1], "e2e_p95_s": total[2],
           "decode_tok_s_mean": speed[0], "decode_tok_s_median": speed[1],
           "retrieval_mean_s": round(statistics.mean(r["retrieval_s"] for r in records), 4),
           "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in records), 1),
           "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in records), 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
           "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "free_ram_gb_at_end": round(psutil.virtual_memory().available / 2**30, 2),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%d")}
    (RESULTS_DIR / "latency_raw").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latency_raw" / f"{name.replace('/', '__')}.json").write_text(json.dumps(records, indent=1))
    path = RESULTS_DIR / "latency_results.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[table["system"] != name]
    pd.concat([table, pd.DataFrame([row])]).to_csv(path, index=False)
    print(json.dumps(row, indent=1))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["closed", "rag"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--quant4", action="store_true")
    ap.add_argument("--gate", action="store_true", help="out-of-scope gate in front of the LLM (rag only)")
    a = ap.parse_args()
    run(a.name, a.mode, a.model, a.adapter, a.quant4, a.gate)
