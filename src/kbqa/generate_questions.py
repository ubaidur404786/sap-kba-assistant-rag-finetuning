"""Phase 2, step 4 - generate extra *questions* (never answers) with a tiny LOCAL Hugging Face model.

Run:  uv run python -m kbqa.generate_questions
In:   data/qa/canonical.jsonl, data/processed/chunks.jsonl
Out:  data/qa/generated.jsonl   (appended; re-running skips finished items, so it is resumable)

Two jobs:
A. paraphrase  - rewrite every canonical table-row / unanswerable question in new words.
                 Why: a model that only ever sees one wording memorises the wording, not the fact.
                 Test-Seen uses held-out paraphrases to check exactly this.
B. qgen        - write questions for plain-text chunks (they have no table structure to build
                 a template from). The answer stays the chunk text.

Quality filters (a tiny LLM makes mistakes, so nothing is trusted blindly):
- paraphrases must keep every technical term of the original (IDs, numbers, table names) and
  at least half of the content words of quoted phrases;
- generated questions must be real questions and must retrieve their own chunk in the BM25
  top-5 ("round-trip consistency", Alberti et al. 2019, https://arxiv.org/abs/1906.05416).
Confidential text never leaves the machine.
"""
import json
import re
import time

from kbqa.config import QA_DIR, SEED
from kbqa.llm import chat_prompt, generate, load_model
from kbqa.retrieval import BM25Retriever, load_chunks

GEN_MODEL = "Qwen/Qwen3-0.6B"     # tiny (0.6B), Apache-2.0, already cached; thinking mode off
TEMPERATURE = 0.8                 # sampling -> varied wordings
BATCH_SIZE = 32                   # big batches amortise per-step Python/CUDA overhead
N_PARAPHRASES = {"table_row": 3, "unanswerable": 2}
N_QGEN = 4
MIN_CHUNK_TOKENS = 30             # skip tiny chunks like "The following tools exist:"
COMMON_CAPS = {"SAP", "HANA", "SQL", "ABAP", "KBA", "I", "A"}


def protected_terms(question):
    """Technical terms a paraphrase must keep exactly: words with digits/underscores/slashes
    (IDs, hashes, table names like /SAPAPO/MARM) and ALL-CAPS names (ACDOCA, SELECT)."""
    terms = []
    for word in re.findall(r"[\w/<>-]+", question):
        if re.search(r"[\d_/]", word) or (word.isupper() and len(word) >= 3 and word not in COMMON_CAPS):
            terms.append(word.strip("-"))
    return [t for t in terms if t]


def quoted_phrase_kept(p, original, min_share=0.5):
    """Quoted natural-language phrases (e.g. a symptom) may be reworded, but at least half of their
    content words (>= 4 letters) must still appear - otherwise the paraphrase changed the topic."""
    for phrase in re.findall(r'"([^"]+)"', original):
        words = re.findall(r"[a-z]{4,}", phrase.lower())
        if words and sum(w in p.lower() for w in words) / len(words) < min_share:
            return False
    return True


def valid_paraphrase(p, original):
    if not p.endswith("?") or p.lower() == original.lower() or len(p.split()) < 4:
        return False
    return (all(t.lower() in p.lower() for t in protected_terms(original))
            and quoted_phrase_kept(p, original))


def parse_questions(text):
    """Tiny models are unreliable with JSON, so we ask for a numbered list and parse lines."""
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip().strip('"*') for l in text.split("\n")]
    return [l for l in lines if l.endswith("?") and len(l) > 10]


def paraphrase_prompt(tok, question, n):
    return chat_prompt(tok, f"Rewrite the question below in {n + 1} different ways, as different users "
                            f"might ask it. Keep every technical term, ID, table name and number exactly "
                            f"as written. Do not answer it. Output a numbered list of questions only.\n\n"
                            f"Question: {question}")


def qgen_prompt(tok, chunk):
    return chat_prompt(tok, f"Text from the SAP HANA knowledge base section \"{chunk['section_question']}\":\n"
                            f"<<<\n{chunk['text']}\n>>>\n\nWrite {N_QGEN + 1} different questions a user could "
                            f"ask that this text answers. Each question must name the concrete term it is "
                            f"about. Never mention 'the text'. Do not answer. Output a numbered list only.")


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QA_DIR / "generated.jsonl"
    done = {json.loads(l)["parent_key"] for l in out_path.open(encoding="utf-8")} if out_path.exists() else set()

    canonical = [json.loads(l) for l in (QA_DIR / "canonical.jsonl").open(encoding="utf-8")]
    chunks = load_chunks()
    retriever = BM25Retriever(chunks)
    jobs = [("paraphrase", q["question"], q) for q in canonical if q["kind"] in N_PARAPHRASES]
    jobs += [("qgen", c["chunk_id"], c) for c in chunks
             if c["type"] == "text" and c["n_tokens"] >= MIN_CHUNK_TOKENS]
    todo = [j for j in jobs if j[1] not in done]
    print(f"{len(jobs)} jobs, {len(todo)} still to do, model {GEN_MODEL}", flush=True)
    if not todo:
        return

    model, tok, load_s = load_model(GEN_MODEL)
    print(f"model loaded in {load_s:.1f} s", flush=True)
    stats, t0 = {"kept": 0, "rejected": 0}, time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), BATCH_SIZE):
            batch = todo[b:b + BATCH_SIZE]
            prompts = [paraphrase_prompt(tok, x["question"], N_PARAPHRASES[x["kind"]]) if task == "paraphrase"
                       else qgen_prompt(tok, x) for task, _, x in batch]
            outs = generate(model, tok, prompts, max_new_tokens=220, batch_size=BATCH_SIZE,
                            temperature=TEMPERATURE, seed=SEED + b)
            for (task, key, x), out in zip(batch, outs):
                cands = list(dict.fromkeys(parse_questions(out["text"])))
                if task == "paraphrase":
                    good = [p for p in cands if valid_paraphrase(p, x["question"])][:N_PARAPHRASES[x["kind"]]]
                    record = {"kind": x["kind"], "unit_id": x["unit_id"], "source_chunk_ids": x["source_chunk_ids"],
                              "faq_no": x["faq_no"], "answer": x["answer"], "truncated": x["truncated"]}
                else:
                    good = []
                    for q in cands:
                        top = [c["chunk_id"] for c, _ in retriever.search(q, k=5)]
                        refers_to_text = re.search(r"\b(passage|the text|this text|article)\b", q, re.I)
                        if len(q.split()) >= 5 and not refers_to_text and x["chunk_id"] in top:
                            good.append(q)
                    good = good[:N_QGEN]
                    record = {"kind": "text_chunk", "unit_id": x["chunk_id"], "source_chunk_ids": [x["chunk_id"]],
                              "faq_no": x["faq_no"], "answer": x["text"], "truncated": False}
                stats["kept"] += len(good)
                stats["rejected"] += len(cands) - len(good)
                # Raw candidates are stored too, so the filters can be changed without re-generating.
                record.update({"parent_key": key, "task": task, "questions": good, "candidates": cands,
                               "n_candidates": len(cands), "model": GEN_MODEL,
                               "options": {"temperature": TEMPERATURE, "top_p": 0.9, "seed": SEED}})
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {b + len(batch)}/{len(todo)}  kept {stats['kept']} rejected {stats['rejected']}  "
                  f"({(b + len(batch)) / (time.perf_counter() - t0):.2f} jobs/s)", flush=True)
    print(f"Done: kept {stats['kept']}, rejected {stats['rejected']} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
