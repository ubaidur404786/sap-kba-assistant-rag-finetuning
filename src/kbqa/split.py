"""Phase 2, step 5 - leakage-safe train / val / test split + DAPT corpus.

Run:  uv run python -m kbqa.split
In:   data/qa/canonical.jsonl, data/qa/generated.jsonl (optional), data/processed/chunks.jsonl
Out:  data/qa/{train,val,test_seen,test_unseen,test_gold}.jsonl, data/qa/dapt_{train,val}.jsonl,
      results/data_report.json (counts + test-set hashes only, no KBA text -> safe to commit)

Key idea - we split *knowledge units*, not questions:
  a unit = one table row (or merged rows) / one text chunk / one unanswerable question.
  All questions about the same unit stay together, so a test question can never be a
  re-worded copy of a training question about the same fact - unless we *want* that:
- train units (80 %): all questions go to train EXCEPT the last wording, which is held out
      -> test_seen  ("did fine-tuning store a fact it was trained on?")
      (15 % of these held-out wordings go to val instead, for checkpoint selection)
- val units (10 %):  all questions -> val            (unseen facts, for monitoring)
- test units (10 %): all questions -> test_unseen    ("does it generalise / abstain?")
- gold FAQ questions (35): never trained on -> test_gold, labelled with how much of their
      source text was in the training units.
"""
import hashlib
import json
import random
import re
from collections import defaultdict

from kbqa.config import PROCESSED_DIR, QA_DIR, RESULTS_DIR, SEED

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
UNANSWERABLE_FRACTIONS = {"train": 0.5, "val": 0.15, "test": 0.35}
VAL_SHARE_OF_HELDOUT = 0.15


def read_jsonl(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def assign_units(unit_ids, fractions, rng):
    """Shuffle units and cut them into train/val/test by the given fractions."""
    ids = sorted(unit_ids)
    rng.shuffle(ids)
    n_train = round(len(ids) * fractions["train"])
    n_val = round(len(ids) * fractions["val"])
    return {u: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            for i, u in enumerate(ids)}


def normalise_q(text):
    return re.findall(r"[a-z0-9_/]+", text.lower())


def shingles(text, n=8):
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def main():
    rng = random.Random(SEED)
    canonical = read_jsonl(QA_DIR / "canonical.jsonl")
    generated = read_jsonl(QA_DIR / "generated.jsonl")
    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")

    # ---- 1. collect all questions per knowledge unit (canonical first, generated after) ----
    units = defaultdict(list)          # unit_id -> list of QA dicts
    for q in canonical:
        if q["kind"] != "gold":
            origin = "template" if q["kind"] == "table_row" else "handwritten"
            units[q["unit_id"]].append({**q, "origin": origin})
    # Generated questions are dropped when they (almost) repeat another question: a gold FAQ question
    # (gold must stay test-only) or any question already used for another unit (word Jaccard >= 0.8).
    gold_words = [set(normalise_q(q["question"])) for q in canonical if q["kind"] == "gold"]
    used = {" ".join(normalise_q(q["question"])) for q in canonical}
    dropped = 0
    for g in generated:
        origin = "paraphrase" if g["task"] == "paraphrase" else "llm_qgen"
        for text in g["questions"]:
            words = set(normalise_q(text))
            key = " ".join(normalise_q(text))
            if key in used or any(len(words & gw) / len(words | gw) >= 0.8 for gw in gold_words):
                dropped += 1
                continue
            used.add(key)
            units[g["unit_id"]].append({k: g[k] for k in ("kind", "unit_id", "source_chunk_ids", "faq_no",
                                                          "answer", "truncated")} | {"question": text,
                                                                                     "origin": origin})
    print(f"generated questions dropped as duplicates / near-copies of gold questions: {dropped}")

    # ---- 2. assign units to splits, stratified by FAQ section so every topic is represented ----
    by_faq = defaultdict(list)
    for u, qs in units.items():
        by_faq["unanswerable" if qs[0]["kind"] == "unanswerable" else qs[0]["faq_no"]].append(u)
    unit_split = {}
    for faq, ids in sorted(by_faq.items(), key=lambda x: str(x[0])):
        frac = UNANSWERABLE_FRACTIONS if faq == "unanswerable" else FRACTIONS
        unit_split.update(assign_units(ids, frac, rng))

    # ---- 3. distribute questions ----
    splits = defaultdict(list)
    for u in sorted(units):
        qs, s = units[u], unit_split[u]
        if s == "train":
            if len(qs) >= 2:                         # hold out the last wording
                *train_qs, held = qs
                target = "val" if rng.random() < VAL_SHARE_OF_HELDOUT else "test_seen"
                splits[target].append({**held, "fact_seen_in_training": True})
            else:
                train_qs = qs
            splits["train"].extend(train_qs)
        elif s == "val":
            splits["val"].extend({**q, "fact_seen_in_training": False} for q in qs)
        else:
            splits["test_unseen"].extend({**q, "fact_seen_in_training": False} for q in qs)

    train_chunk_ids = {c for q in splits["train"] for c in q["source_chunk_ids"]}
    for q in canonical:
        if q["kind"] == "gold":
            src = q["source_chunk_ids"]
            coverage = sum(c in train_chunk_ids for c in src) / len(src)
            splits["test_gold"].append({**q, "origin": "gold", "train_coverage": round(coverage, 2)})

    # ---- 4. leakage checks ----
    seen_q = {}
    for name, rows in splits.items():
        for r in rows:
            key = " ".join(r["question"].lower().split())
            assert seen_q.setdefault(key, name) == name, f"question in two splits: {r['question']}"
    train_units = {q["unit_id"] for q in splits["train"]}
    assert not train_units & {q["unit_id"] for q in splits["test_unseen"]}, "unit leakage"
    # Near-duplicate answers: an "unseen" fact whose answer text mostly appears in a train answer.
    train_sh = set().union(*(shingles(q["answer"]) for q in splits["train"] if q["kind"] != "unanswerable"))
    for r in splits["test_unseen"] + splits["val"]:
        if r["kind"] == "unanswerable" or r.get("fact_seen_in_training"):
            continue
        sh = shingles(r["answer"])
        r["answer_overlap_with_train"] = round(len(sh & train_sh) / len(sh), 2)

    # ---- 5. write QA files with stable ids ----
    hashes, counts = {}, {}
    for name, rows in splits.items():
        for i, r in enumerate(rows):
            r["qa_id"] = f"{name}-{i:04d}"
            r["split"] = name
        path = QA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        counts[name] = {"total": len(rows)} | {k: sum(r["kind"] == k for r in rows)
                                                for k in ("table_row", "text_chunk", "unanswerable", "gold")}

    # ---- 6. DAPT corpus: raw chunk text of train units only (val units -> dapt_val) ----
    unit_of_chunk = {c: q["unit_id"] for qs in units.values() for q in qs for c in q["source_chunk_ids"]}
    dapt = defaultdict(list)
    for c in chunks:
        s = unit_split.get(unit_of_chunk.get(c["chunk_id"]), "train")   # chunks without QA -> train
        if s in ("train", "val"):
            dapt[s].append({"chunk_id": c["chunk_id"], "text": f"{c['section_question']}\n{c['text']}"})
    write_jsonl(QA_DIR / "dapt_train.jsonl", dapt["train"])
    write_jsonl(QA_DIR / "dapt_val.jsonl", dapt["val"])

    near_dup = sum(r.get("answer_overlap_with_train", 0) >= 0.8 for r in splits["test_unseen"])
    report = {"seed": SEED, "fractions": FRACTIONS, "unanswerable_fractions": UNANSWERABLE_FRACTIONS,
              "n_units": len(units), "units_per_split": {s: sum(v == s for v in unit_split.values())
                                                         for s in FRACTIONS},
              "questions": counts, "file_sha256_16": hashes,
              "test_unseen_near_duplicate_answers(>=0.8)": near_dup,
              "dapt_chunks": {k: len(v) for k, v in dapt.items()},
              "generated_questions_used": sum(len(g["questions"]) for g in generated) - dropped,
              "generated_questions_dropped_as_duplicates": dropped}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
