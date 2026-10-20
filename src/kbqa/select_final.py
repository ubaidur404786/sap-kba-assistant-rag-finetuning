"""Phase 7 - apply the pre-registered selection rule and build the comparison table.

Run:  uv run python -m kbqa.select_final
In:   results/eval_results_val.csv (decides), results/eval_results.csv (reported),
      results/latency_results.csv, results/train_results.csv
Out:  results/final_selection.json, results/comparison.csv, results/comparison.md

Rule (docs/PLANNING.md, fixed before results):
  eligible  = correct refusals on unanswerable val questions >= 0.80
              AND key-fact recall on val >= retrieval-only baseline on val
              AND median end-to-end latency <= 10 s (benchmark.py)  AND peak VRAM <= 6 GB
  rank      = val_score (0.5 key-fact recall + 0.25 ROUGE-L + 0.25 correct refusals), then latency
If nothing is eligible, the system that fails the fewest criteria is reported, with the failures listed.
"""
import json

import pandas as pd

from kbqa.config import RESULTS_DIR
from kbqa.run_experiments import val_score

MIN_REFUSAL, MAX_LATENCY_S, MAX_VRAM_GB = 0.80, 10.0, 6.0


def load(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["system"])


def main():
    val, test, lat, train = (load(n) for n in ("eval_results_val.csv", "eval_results.csv",
                                               "latency_results.csv", "train_results.csv"))
    retrieval_kfr = val.loc[val["system"] == "retrieval-only", "val_key_fact_recall"]
    retrieval_kfr = float(retrieval_kfr.iloc[0]) if len(retrieval_kfr) else 0.0

    candidates = []
    for _, r in val.iterrows():
        if r["system"] == "retrieval-only":
            continue
        l = lat[lat["system"] == r["system"]]
        median = float(l["e2e_median_s"].iloc[0]) if len(l) else None
        if median is None and r["mode"] == "retrieval":       # no LLM: per-question time from the eval run
            median = round(float(r["batch_eval_s"]) / float(r["n_items"]), 4)
        vram = float(l["peak_vram_gb"].iloc[0]) if len(l) else (0.0 if pd.isna(r["peak_vram_gb"]) else float(r["peak_vram_gb"]))
        fails = []
        if pd.isna(r["unanswerable_correct_refusal"]) or r["unanswerable_correct_refusal"] < MIN_REFUSAL:
            fails.append(f"correct refusals {r['unanswerable_correct_refusal']} < {MIN_REFUSAL}")
        if pd.isna(r["val_key_fact_recall"]) or r["val_key_fact_recall"] < retrieval_kfr:
            fails.append(f"key-fact recall {r['val_key_fact_recall']} < retrieval-only {retrieval_kfr}")
        if median is None:
            fails.append("latency not measured")
        elif median > MAX_LATENCY_S:
            fails.append(f"median latency {median} s > {MAX_LATENCY_S} s")
        if vram > MAX_VRAM_GB:
            fails.append(f"peak VRAM {vram} GB > {MAX_VRAM_GB}")
        candidates.append({"system": r["system"], "val_score": round(val_score(r), 4), "e2e_median_s": median,
                           "peak_vram_gb": vram, "fails": fails})

    eligible = [c for c in candidates if not c["fails"]]
    pool = eligible or sorted(candidates, key=lambda c: len(c["fails"]))[:max(1, sum(
        len(c["fails"]) == min(len(x["fails"]) for x in candidates) for c in candidates))]
    best = sorted(pool, key=lambda c: (-c["val_score"], c["e2e_median_s"] or 1e9))[0]
    selection = {"rule": __doc__.split("Rule")[1].strip(), "retrieval_only_val_key_fact_recall": retrieval_kfr,
                 "eligible": [c["system"] for c in eligible], "selected": best,
                 "candidates": sorted(candidates, key=lambda c: -c["val_score"])}
    (RESULTS_DIR / "final_selection.json").write_text(json.dumps(selection, indent=1))

    # Comparison table: test-set quality + latency + training cost, one row per system.
    cols = ["system", "unseen_key_fact_recall", "unseen_rouge_l", "unseen_semantic_sim", "seen_key_fact_recall",
            "seen_rouge_l", "gold_semantic_sim", "unanswerable_correct_refusal", "unseen_false_refusal",
            "retrieval_recall@3"]
    table = test[[c for c in cols if c in test.columns]].copy()
    table = table.merge(lat[["system", "load_s", "ttft_median_s", "e2e_median_s", "e2e_p95_s", "decode_tok_s_median",
                             "peak_vram_gb"]], on="system", how="left") if len(lat) else table
    train = train[~train["run"].astype(str).str.endswith("-smoke")] if len(train) else train
    if len(train):
        tr = train.assign(key=train["run"]).set_index("key")
        table["train_time_s"] = [tr["train_time_s"].get("/".join(s.split("/")[1:3])) if s.startswith("ft/") else None
                                 for s in table["system"]]
        table["saved_size_mb"] = [tr["saved_size_mb"].get("/".join(s.split("/")[1:3])) if s.startswith("ft/") else None
                                  for s in table["system"]]
    table = table.sort_values("unseen_key_fact_recall", ascending=False)
    table.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    (RESULTS_DIR / "comparison.md").write_text(table.round(3).to_markdown(index=False), encoding="utf-8")
    print(json.dumps(selection["selected"], indent=1))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
