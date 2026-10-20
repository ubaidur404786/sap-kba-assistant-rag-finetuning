"""Phase 7 - a retrieval-confidence gate for "I don't know" (no GPU needed).

Run:  uv run python -m kbqa.gate
Out:  results/gate_selection.json, extra "<system>+gate" rows in eval_results_val.csv / eval_results.csv

Why: the fine-tuned RAG model copies facts well but rarely refuses (unanswerable questions are only
~2 % of its training data). A cheap, explainable fix: if the question is not similar to any chunk,
the knowledge base probably has nothing on it, so the system answers ABSTAIN without calling the LLM.

Signal: the highest cosine similarity between the question and any chunk (bge-small embeddings).
Chosen on the VALIDATION set by ROC-AUC for separating answerable from unanswerable questions:
dense max-cosine 0.988, BM25 top score / query token 0.812, BM25 top score 0.740.
How: the threshold is chosen on the VALIDATION set only (maximising correct refusals + answered
answerable questions), then applied unchanged to the test set. The gate re-uses the stored
predictions in results/raw/, so it is exact and costs no GPU time.
"""
import json

import numpy as np
import pandas as pd

from kbqa.config import ABSTAIN, RESULTS_DIR
from kbqa.evaluate import eval_set, summarise
from kbqa.metrics import score_items
from kbqa.qa_system import mentions_known_id
from kbqa.retrieval import QUERY_PREFIX, HybridRetriever, load_chunks


def top_scores(retriever, items):
    """Highest cosine similarity between each question and any KBA chunk."""
    q = retriever.embedder.encode([QUERY_PREFIX + it["question"] for it in items], normalize_embeddings=True)
    return (q @ retriever.emb.T).max(axis=1)


def choose_threshold(scores, answerable):
    """Balanced accuracy of the refuse/answer decision: mean of
    (share of unanswerable questions refused) and (share of answerable questions answered)."""
    best = (0.0, -1.0)
    for t in np.unique(np.round(scores, 3)):
        refuse = scores < t
        acc = 0.5 * refuse[~answerable].mean() + 0.5 * (~refuse[answerable]).mean()
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


def gated_rows(system, which, items, scores, known, threshold):
    raw = RESULTS_DIR / "raw" / f"{which}__{system.replace('/', '__')}.jsonl"
    preds = {r["qa_id"]: r["prediction"] for r in map(json.loads, raw.open(encoding="utf-8"))}
    new_preds = [ABSTAIN if (s < threshold and not k) else preds[it["qa_id"]]
                 for it, s, k in zip(items, scores, known)]
    return summarise(score_items(items, new_preds), items)


def main():
    retriever = HybridRetriever(load_chunks())
    val_items, test_items = eval_set("val"), eval_set("test")
    val_scores, test_scores = top_scores(retriever, val_items), top_scores(retriever, test_items)
    answerable = np.array([it["kind"] != "unanswerable" for it in val_items])
    threshold, bal_acc = choose_threshold(val_scores, answerable)
    # Identifier override (see qa_system.gate_refuses): never refuse a question naming a KBA identifier.
    val_known = np.array([mentions_known_id(retriever, it["question"]) for it in val_items])
    test_known = np.array([mentions_known_id(retriever, it["question"]) for it in test_items])
    val_refuse = (val_scores < threshold) & ~val_known
    sel = {"signal": "max cosine similarity question-chunk (bge-small-en-v1.5)", "threshold": threshold, "val_balanced_accuracy": round(bal_acc, 4),
           "val_refused_unanswerable": round(float(val_refuse[~answerable].mean()), 4),
           "val_refused_answerable": round(float(val_refuse[answerable].mean()), 4),
           "val_refused_answerable_without_id_override": round(float((val_scores < threshold)[answerable].mean()), 4),
           "id_override": "never refuse when the question names an identifier found verbatim in the KBA"}
    (RESULTS_DIR / "gate_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)

    # Apply the gate to every RAG / retrieval system that has stored predictions.
    for which, items, scores, known, csv in (("val", val_items, val_scores, val_known, "eval_results_val.csv"),
                                             ("test", test_items, test_scores, test_known, "eval_results.csv")):
        table = pd.read_csv(RESULTS_DIR / csv)
        table = table[~table["system"].str.endswith("+gate")]
        new = []
        for _, row in table.iterrows():
            if row["mode"] not in ("rag", "retrieval"):
                continue
            summary = gated_rows(row["system"], which, items, scores, known, threshold)
            new.append({**row.to_dict(), **summary, "system": row["system"] + "+gate"})
        pd.concat([table, pd.DataFrame(new)]).to_csv(RESULTS_DIR / csv, index=False)
        print(which, "gated systems:", [n["system"] for n in new])


if __name__ == "__main__":
    main()
