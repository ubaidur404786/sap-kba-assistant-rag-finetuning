"""Answer-quality metrics. Each one answers a different question:

token_f1       - how many answer words overlap with the reference? (SQuAD-style, word level)
rouge_l        - longest common word sequence with the reference (rewards copying the right text in order)
semantic_sim   - cosine similarity of sentence embeddings (rewards the same meaning in other words)
key_fact_recall- share of "hard facts" in the reference (SAP Note numbers, hashes, parameter and
                 table names) that appear in the answer. The metric closest to "is it correct?" here.
abstained      - did the model say it cannot answer? Correct for unanswerable questions, a
                 "false refusal" for answerable ones.
None of these is perfect for open-ended answers; they are reported together, never alone.
"""
import re
import string
from collections import Counter

from rouge_score import rouge_scorer

from kbqa.config import ABSTAIN

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_embedder = None
ABSTAIN_PATTERNS = re.compile(
    r"does not contain (the )?information|not (contained|covered|available) in the knowledge base|"
    r"(i|we) (do not|don't|cannot|can't) (know|answer|find)|no information (is )?available|"
    r"cannot be answered|not able to answer", re.I)


def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) - {"_", "/"})
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


def token_f1(pred, ref):
    p, r = normalise(pred).split(), normalise(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def semantic_sim(preds, refs):
    """Batch cosine similarity with bge-small-en-v1.5 (33M params, runs on CPU in seconds)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    a = _embedder.encode(preds, normalize_embeddings=True, batch_size=32)
    b = _embedder.encode(refs, normalize_embeddings=True, batch_size=32)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def key_facts(text):
    """Hard facts worth checking literally: 4-7 digit numbers (SAP Notes, check IDs), 32-char hashes,
    identifiers with '_' / '/' / '.' (M_SQL_PLAN_CACHE, /SAPAPO/MARM, indexserver.ini), ALL-CAPS names."""
    facts = set(re.findall(r"\b[0-9a-f]{32}\b", text))
    facts |= set(re.findall(r"\b\d{4,7}\b", text))
    facts |= {w.strip(".") for w in re.findall(r"[A-Za-z0-9]*[_/.][A-Za-z0-9_/.]+", text) if len(w) > 4}
    facts |= set(re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", text)) - {"HANA", "SELECT", "NULL", "WHERE", "FROM", "SQL", "ABAP"}
    return facts


def key_fact_recall(pred, ref, question=""):
    """Only facts the question does not already contain count - echoing the question is not knowledge."""
    facts = {f for f in key_facts(ref) if f.lower() not in question.lower()}
    if not facts:
        return None
    return sum(f.lower() in pred.lower() for f in facts) / len(facts)


def abstained(pred):
    return bool(ABSTAIN_PATTERNS.search(pred)) or pred.strip() == ABSTAIN


def score_items(items, preds):
    """items: QA dicts (question, answer, kind); preds: list of strings. Returns per-item metric dicts."""
    sims = semantic_sim(preds, [it["answer"] for it in items])
    rows = []
    for it, pred, sim in zip(items, preds, sims):
        answerable = it["kind"] != "unanswerable"
        row = {"qa_id": it["qa_id"], "split": it["split"], "kind": it["kind"], "answerable": answerable,
               "abstained": abstained(pred)}
        if answerable:
            row.update({"token_f1": token_f1(pred, it["answer"]), "rouge_l": rouge_l(pred, it["answer"]),
                        "semantic_sim": sim, "key_fact_recall": key_fact_recall(pred, it["answer"], it["question"])})
        rows.append(row)
    return rows
