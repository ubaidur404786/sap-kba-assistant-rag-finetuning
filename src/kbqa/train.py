"""Phase 5 - fine-tuning experiments (one function, many methods).

Run:  uv run python -m kbqa.train --exp sft-lora --model Qwen/Qwen3-0.6B [--smoke]
Out:  outputs/<model>/<exp>/            adapter (or full model) weights   (git-ignored)
      results/train_results.csv         one row per run: time, throughput, memory, params (safe to commit)

Three kinds of training data ("task"):
  sft      closed-book supervised fine-tuning: prompt = system + question, completion = answer
  rag_sft  "hybrid": prompt = system + 3 retrieved KBA excerpts + question, completion = answer.
           The source chunk is always among the excerpts for answerable questions, so the model
           learns to *copy from context*; unanswerable questions get irrelevant excerpts -> ABSTAIN.
  dapt     domain-adaptive continued pretraining: plain KBA text, next-token prediction on every token.

Loss: for sft / rag_sft only the completion (answer) tokens are learned (TRL prompt-completion format),
the prompt tokens are masked. Prompts come from kbqa.qa_system, i.e. exactly what is used at inference.
"""
import argparse
import json
import random
import time

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from kbqa.config import OUTPUTS_DIR, QA_DIR, RESULTS_DIR, SEED

# One entry per experiment. Everything not listed uses the defaults in build_args().
EXPERIMENTS = {
    "sft-lora":      {"task": "sft", "method": "lora"},
    "sft-qlora":     {"task": "sft", "method": "lora", "quant_4bit": True},
    "sft-ia3":       {"task": "sft", "method": "ia3", "lr": 3e-3},
    "sft-prompt":    {"task": "sft", "method": "prompt", "lr": 3e-2},
    "sft-full":      {"task": "sft", "method": "full", "lr": 2e-5},
    "dapt-lora":     {"task": "dapt", "method": "lora"},
    "dapt-sft-lora": {"task": "sft", "method": "lora", "init_adapter": "dapt-lora"},
    "ragsft-lora":   {"task": "rag_sft", "method": "lora"},
}
# Micro-batch 8 x accumulation 4 = effective batch 32. Batch 32 in one go ran out of the 6 GB VRAM
# (the 152k-token vocabulary makes the output logits large); gradient checkpointing is on for every
# run so all methods have the same memory/time trade-off.
DEFAULTS = {"lr": 2e-4, "epochs": 2, "batch_size": 8, "grad_accum": 4, "max_length": 512,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "quant_4bit": False}


def read(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def short_name(model_id):
    return model_id.split("/")[-1].lower()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset(task, tok, split):
    from kbqa import qa_system
    if task == "dapt":
        rows = read(QA_DIR / f"dapt_{'train' if split == 'train' else 'val'}.jsonl")
        return Dataset.from_list([{"text": r["text"]} for r in rows])
    rows = read(QA_DIR / f"{split}.jsonl")
    if task == "sft":
        data = [{"prompt": qa_system.closed_book_prompt(tok, r["question"]), "completion": r["answer"]} for r in rows]
    else:  # rag_sft
        from kbqa.retrieval import Retriever, load_chunks
        chunks = load_chunks()
        by_id = {c["chunk_id"]: c for c in chunks}
        retriever = Retriever(chunks)
        rng = random.Random(SEED)
        data = []
        for r in rows:
            hits = retriever.search(r["question"], k=qa_system.RAG_K)
            ids = [c["chunk_id"] for c, _ in hits]
            gold = [i for i in r["source_chunk_ids"] if i in by_id]
            if r["kind"] != "unanswerable" and gold and not set(gold) & set(ids):
                hits[rng.randrange(len(hits))] = (by_id[gold[0]], 0.0)     # make sure the answer is in context
            data.append({"prompt": qa_system.rag_prompt(tok, r["question"], hits), "completion": r["answer"]})
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# Model + PEFT method
# ---------------------------------------------------------------------------
def peft_config(cfg, model_id):
    from peft import IA3Config, LoraConfig, PromptTuningConfig, PromptTuningInit
    m = cfg["method"]
    if m in ("lora", "dora"):
        return LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                          target_modules="all-linear", use_dora=(m == "dora"), task_type="CAUSAL_LM")
    if m == "ia3":
        return IA3Config(target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"],
                         task_type="CAUSAL_LM")
    if m == "prompt":
        return PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20,
                                  prompt_tuning_init=PromptTuningInit.TEXT, tokenizer_name_or_path=model_id,
                                  prompt_tuning_init_text="Answer SAP HANA SQL performance questions from the knowledge base:")
    return None   # full fine-tuning


def load_for_training(model_id, cfg):
    quant = None
    if cfg["quant_4bit"]:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0},
                                                 quantization_config=quant)
    if cfg["quant_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def dir_size_mb(path):
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20, 2)


def train(exp, model_id, smoke=False):
    cfg = {**DEFAULTS, **EXPERIMENTS[exp]}
    if cfg["task"] == "rag_sft":
        cfg.update(batch_size=2, grad_accum=16, max_length=1536)
    torch.manual_seed(SEED)
    out_dir = OUTPUTS_DIR / short_name(model_id) / (exp + ("-smoke" if smoke else ""))
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds, val_ds = build_dataset(cfg["task"], tok, "train"), build_dataset(cfg["task"], tok, "val")
    if smoke:
        train_ds, val_ds = train_ds.select(range(32)), val_ds.select(range(16))

    model = load_for_training(model_id, cfg)
    peft_cfg = peft_config(cfg, model_id)
    if cfg.get("init_adapter"):                       # continue training the DAPT adapter on SFT data
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, OUTPUTS_DIR / short_name(model_id) / (cfg["init_adapter"] + ("-smoke" if smoke else "")),
                                          is_trainable=True)
        peft_cfg = None

    args = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=1 if smoke else cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", warmup_steps=0.05,   # a float = share of total steps (transformers 5)
        max_length=cfg["max_length"],
        bf16=True, gradient_checkpointing=True,
        optim="adamw_bnb_8bit" if cfg["method"] == "full" else "adamw_torch",
        eval_strategy="epoch", save_strategy="no", logging_steps=10, report_to="none", seed=SEED,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg)
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    train_tokens = sum(len(x) for x in trainer.train_dataset["input_ids"]) * args.num_train_epochs

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    train_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    eval_loss = trainer.evaluate()["eval_loss"]
    trainer.save_model(str(out_dir))

    row = {"run": f"{short_name(model_id)}/{exp}" + ("-smoke" if smoke else ""), "model": model_id, "exp": exp,
           "task": cfg["task"], "method": cfg["method"], "quant_4bit": cfg["quant_4bit"],
           "init_adapter": cfg.get("init_adapter", "-"), "lr": cfg["lr"], "epochs": args.num_train_epochs,
           "effective_batch": cfg["batch_size"] * cfg["grad_accum"], "max_length": cfg["max_length"],
           "train_examples": len(train_ds), "train_tokens_total": train_tokens,
           "trainable_params": n_trainable, "total_params": n_total,
           "trainable_pct": round(100 * n_trainable / n_total, 4),
           "train_time_s": round(train_s, 1), "train_tokens_per_s": round(train_tokens / train_s, 1),
           "train_samples_per_s": round(result.metrics.get("train_samples_per_second", 0), 2),
           "final_train_loss": round(result.metrics.get("train_loss", float("nan")), 4),
           "eval_loss": round(eval_loss, 4), "peak_vram_gb": round(peak, 3),
           "saved_size_mb": dir_size_mb(out_dir), "seed": SEED, "date": time.strftime("%Y-%m-%d")}
    path = RESULTS_DIR / "train_results.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[table["run"] != row["run"]]
    pd.concat([table, pd.DataFrame([row])]).to_csv(path, index=False)
    print(json.dumps(row, indent=1))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=sorted(EXPERIMENTS), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--smoke", action="store_true", help="32 examples, 1 epoch - checks the pipeline only")
    a = ap.parse_args()
    train(a.exp, a.model, a.smoke)
