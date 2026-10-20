"""The whole experiment pipeline, one process per step, strictly one after another.

Run:  uv run python -m kbqa.run_experiments            (resumable: finished steps are skipped)

Order and decision points (every decision uses the VALIDATION set, never the test set):
  1. data split + retriever choice (bm25 / dense / hybrid by recall@3 on val)
  2. RAG on val for every tiny model           -> choose the fine-tuning models (model_selection.json)
  3. baselines on test: retrieval-only, every tiny model closed-book and with RAG
  4. latency of every tiny model with RAG (+ main model closed-book)
  5. fine-tuning on the main model: technique comparison (closed-book SFT with LoRA, QLoRA, DoRA,
     IA3, prompt tuning, full FT), DAPT, DAPT->SFT, RAG-SFT; RAG-SFT on the second model
  6. every fine-tuned system on val (for selection) and test (for the report)
  7. latency of the fine-tuned finalists
Separate processes: each model is loaded, measured and fully released before the next one.
"""
import json
import subprocess
import sys
import time

import pandas as pd

from kbqa.config import RESULTS_DIR, ROOT
from kbqa.llm import TINY_MODELS

PY = [sys.executable, "-m"]
# Most decision-relevant runs first; sft-qlora last because it hung once (it only saves memory,
# and LoRA already fits a 0.6B model in 5 GB).
MAIN_EXPERIMENTS = ["sft-lora", "ragsft-lora", "dapt-lora", "dapt-sft-lora", "sft-ia3", "sft-prompt",
                    "sft-full", "sft-qlora"]
SECOND_EXPERIMENTS = ["ragsft-lora"]
PARAMS_M = {"smollm2-135m": 135, "smollm2-360m": 362, "qwen2.5-0.5b": 494, "qwen3-0.6b": 596, "tinyllama-1.1b": 1100}


def val_score(row):
    """Selection score on val (fixed before results): faithfulness first.
    0.5 x key-fact recall + 0.25 x ROUGE-L + 0.25 x correct refusals on unanswerable questions."""
    get = lambda k: 0.0 if pd.isna(row.get(k)) else float(row[k])
    return 0.5 * get("val_key_fact_recall") + 0.25 * get("val_rouge_l") + 0.25 * get("unanswerable_correct_refusal")


# A step that runs longer than this is considered hung (the sft-qlora run on 2026-10-19 stopped at
# step 31/108 with 0 % GPU use for 8 hours). It is killed, recorded and skipped on later runs.
TIMEOUT_S = {"train": 3 * 3600, "val": 3600, "test": 2 * 3600, "latency": 3600, "data": 3600}
FAILED_FILE = RESULTS_DIR / "failed_steps.json"


def failed_steps():
    return json.loads(FAILED_FILE.read_text()) if FAILED_FILE.exists() else {}


def log_step(name, cmd, stage):
    print(f"{time.strftime('%H:%M:%S')} start: {name}", flush=True)
    t0 = time.perf_counter()
    with (ROOT / "logs" / f"pipeline_{stage}.log").open("a", encoding="utf-8") as log:
        log.write(f"\n===== {name} =====\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=TIMEOUT_S.get(stage, 3600))
        except subprocess.TimeoutExpired:
            subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # kill the whole process tree
            rc = "timeout"
    if rc != 0:
        failed = failed_steps()
        failed[name] = {"result": str(rc), "seconds": round(time.perf_counter() - t0), "date": time.strftime("%Y-%m-%d %H:%M")}
        FAILED_FILE.write_text(json.dumps(failed, indent=1))
    print(f"{time.strftime('%H:%M:%S')} {'ok' if rc == 0 else f'FAILED ({rc})'}: {name} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return rc


def done(csv, col, value):
    path = RESULTS_DIR / csv
    return path.exists() and value in set(pd.read_csv(path)[col])


def step(stage, csv, col, key, cmd):
    if done(csv, col, key):
        print(f"skip (done): {key}", flush=True)
        return 0
    if key in failed_steps():
        print(f"skip (failed before, see results/failed_steps.json): {key}", flush=True)
        return 1
    return log_step(key, cmd, stage)


def choose_models():
    """Main fine-tuning model = best tiny model with RAG on val; second = best of another family."""
    path = RESULTS_DIR / "model_selection.json"
    if path.exists():
        return json.loads(path.read_text())
    val = pd.read_csv(RESULTS_DIR / "eval_results_val.csv")
    rows = []
    for short, mid in TINY_MODELS.items():
        r = val[val["system"] == f"base/{short}/rag"]
        if len(r):
            rows.append({"short": short, "model": mid, "params_m": PARAMS_M[short], "val_score": round(val_score(r.iloc[0]), 4)})
    ranked = sorted(rows, key=lambda r: -r["val_score"])
    main = ranked[0]
    family = lambda s: s.split("-")[0].rstrip("0123456789.")
    second = next((r for r in ranked[1:] if family(r["short"]) != family(main["short"])), ranked[1])
    sel = {"rule": "val_score = 0.5*key_fact_recall + 0.25*rouge_l + 0.25*correct_refusal (RAG, val set)",
           "ranking": ranked, "main": main, "second": second}
    path.write_text(json.dumps(sel, indent=1))
    return sel


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    # 1. data + retriever
    if not (ROOT / "data" / "qa" / "train.jsonl").exists() or not (RESULTS_DIR / "retrieval_selection.json").exists():
        log_step("split", PY + ["kbqa.split"], "data")
        log_step("retriever selection", PY + ["kbqa.retrieval"], "data")

    # 2. model selection on val
    step("val", "eval_results_val.csv", "system", "retrieval-only",
         PY + ["kbqa.evaluate", "--set", "val", "--name", "retrieval-only", "--mode", "retrieval"])
    for short, mid in TINY_MODELS.items():
        name = f"base/{short}/rag"
        step("val", "eval_results_val.csv", "system", name,
             PY + ["kbqa.evaluate", "--set", "val", "--name", name, "--model", mid, "--mode", "rag"])
    sel = choose_models()
    main_model, second_model = sel["main"]["model"], sel["second"]["model"]
    print("selected:", main_model, "+", second_model, flush=True)

    # 3. baselines on test
    step("test", "eval_results.csv", "system", "retrieval-only",
         PY + ["kbqa.evaluate", "--name", "retrieval-only", "--mode", "retrieval"])
    for short, mid in TINY_MODELS.items():
        for mode in ("closed", "rag"):
            name = f"base/{short}/{mode}"
            step("test", "eval_results.csv", "system", name, PY + ["kbqa.evaluate", "--name", name, "--model", mid, "--mode", mode])

    # 4. latency of baselines
    main_short = sel["main"]["short"]
    for short, mid in TINY_MODELS.items():
        modes = ("rag", "closed") if short == main_short else ("rag",)
        for mode in modes:
            name = f"base/{short}/{mode}"
            step("latency", "latency_results.csv", "system", name,
                 PY + ["kbqa.benchmark", "--name", name, "--model", mid, "--mode", mode])

    # 5. training
    plan = [(main_model, e) for e in MAIN_EXPERIMENTS] + [(second_model, e) for e in SECOND_EXPERIMENTS]
    for model, exp in plan:
        run = f"{model.split('/')[-1].lower()}/{exp}"
        step("train", "train_results.csv", "run", run, PY + ["kbqa.train", "--exp", exp, "--model", model])

    # 6 + 7. evaluate fine-tuned systems (val + test), then latency of finalists
    trained = set(pd.read_csv(RESULTS_DIR / "train_results.csv")["run"]) if (RESULTS_DIR / "train_results.csv").exists() else set()
    for model, exp in plan:
        short = model.split("/")[-1].lower()
        if f"{short}/{exp}" not in trained:
            continue
        out = str(ROOT / "outputs" / short / exp)
        mode = "rag" if exp.startswith("ragsft") else "closed"
        target = ["--model", out] if exp == "sft-full" else ["--model", model, "--adapter", out]
        target += ["--quant4"] if "qlora" in exp else []
        name = f"ft/{short}/{exp}/{mode}"
        for which, csv in (("val", "eval_results_val.csv"), ("test", "eval_results.csv")):
            step(which, csv, "system", name, PY + ["kbqa.evaluate", "--set", which, "--name", name, "--mode", mode] + target)
        if exp in ("sft-lora", "sft-qlora", "ragsft-lora"):
            step("latency", "latency_results.csv", "system", name,
                 PY + ["kbqa.benchmark", "--name", name, "--mode", mode] + target)

    # 8. out-of-scope gate (threshold chosen on val, applied to stored predictions; CPU only, always
    #    re-run so new systems get their "+gate" rows), then latency of the gated RAG finalists.
    log_step("gate", PY + ["kbqa.gate"], "data")
    for model, exp in plan:
        short = model.split("/")[-1].lower()
        if exp == "ragsft-lora" and f"{short}/{exp}" in trained:
            name = f"ft/{short}/{exp}/rag+gate"
            step("latency", "latency_results.csv", "system", name,
                 PY + ["kbqa.benchmark", "--name", name, "--mode", "rag", "--gate", "--model", model,
                       "--adapter", str(ROOT / "outputs" / short / exp)])
    log_step("final selection", PY + ["kbqa.select_final"], "data")
    print("pipeline finished", flush=True)


if __name__ == "__main__":
    main()
