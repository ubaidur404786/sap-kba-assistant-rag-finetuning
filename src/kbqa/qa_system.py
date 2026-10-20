"""The three ways of answering a question that we compare.

1. closed-book : the model answers from its weights only (base model, or fine-tuned on the KBA).
2. rag         : a retriever finds the 3 best chunks; the model must answer from them only.
3. retrieval   : no LLM at all - return the best chunk's text (a strong, honest baseline).

The prompts are the SAME for base and fine-tuned models, so differences come from the weights.
Fine-tuning data (train.py) is formatted with exactly these prompt functions.
"""
import time

from kbqa.config import ABSTAIN
from kbqa.llm import chat_prompt, generate

SYSTEM = ("You are an SAP HANA support assistant. Answer the user's question concisely and precisely. "
          f"If you do not know the answer, reply exactly: \"{ABSTAIN}\"")
SYSTEM_RAG = ("You are an SAP HANA support assistant. Answer the question using ONLY the knowledge base "
              "excerpts provided. Copy SAP Note numbers, parameters and table names exactly. If the excerpts "
              f"do not contain the answer, reply exactly: \"{ABSTAIN}\"")
RAG_K = 3
MAX_NEW_TOKENS = 150   # answers in the KBA are usually shorter; caps evaluation time


def closed_book_prompt(tok, question):
    return chat_prompt(tok, question, SYSTEM)


def format_context(hits):
    return "\n\n".join(f"[{i + 1}] {c['section_question']}\n{c['text']}" for i, (c, _) in enumerate(hits))


def rag_prompt(tok, question, hits):
    return chat_prompt(tok, f"Knowledge base excerpts:\n{format_context(hits)}\n\nQuestion: {question}", SYSTEM_RAG)


def answer_closed_book(model, tok, questions, batch_size=32):
    t0 = time.perf_counter()
    outs = generate(model, tok, [closed_book_prompt(tok, q) for q in questions],
                    max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size)
    return outs, [[] for _ in questions], time.perf_counter() - t0


def answer_rag(model, tok, retriever, questions, batch_size=16):
    t0 = time.perf_counter()
    hits = [retriever.search(q, k=RAG_K) for q in questions]
    outs = generate(model, tok, [rag_prompt(tok, q, h) for q, h in zip(questions, hits)],
                    max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size)
    return outs, [[c["chunk_id"] for c, _ in h] for h in hits], time.perf_counter() - t0


def answer_retrieval_only(retriever, questions):
    t0 = time.perf_counter()
    hits = [retriever.search(q, k=RAG_K) for q in questions]
    outs = [{"text": h[0][0]["text"], "input_tokens": 0, "output_tokens": 0} for h in hits]
    return outs, [[c["chunk_id"] for c, _ in h] for h in hits], time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Out-of-scope gate (kbqa.gate chose the signal and threshold on the validation set)
# ---------------------------------------------------------------------------
def load_gate_threshold():
    import json

    from kbqa.config import RESULTS_DIR
    path = RESULTS_DIR / "gate_selection.json"
    return json.loads(path.read_text())["threshold"] if path.exists() else None


def technical_ids(question):
    """Identifiers such as M1125, M_SQL_PLAN_CACHE or /SAPAPO/MARM: letters+digits, '_' or a path."""
    import re
    return [w for w in re.findall(r"[\w/.-]+", question)
            if len(w) >= 4 and ((re.search(r"[a-z]", w, re.I) and re.search(r"\d", w)) or "_" in w or w.count("/") >= 2)]


def mentions_known_id(retriever, question):
    """True if the question names an identifier that appears verbatim in the KBA."""
    if not hasattr(retriever, "_corpus_lower"):
        retriever._corpus_lower = " ".join(c["text"] for c in retriever.chunks).lower()
    return any(t.lower() in retriever._corpus_lower for t in technical_ids(question))


def gate_refuses(retriever, question, threshold):
    """True when the question is not similar enough to ANY knowledge-base chunk AND names no identifier
    that exists in the KBA -> answer ABSTAIN without calling the LLM. The identifier override was added
    after a check-ID question ("mini check M1125") was wrongly refused; on the validation set it cut
    wrong refusals from 3.9 % to 1.0 % with correct refusals unchanged (100 %)."""
    from kbqa.retrieval import QUERY_PREFIX
    q = retriever.embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]
    return float((retriever.emb @ q).max()) < threshold and not mentions_known_id(retriever, question)
