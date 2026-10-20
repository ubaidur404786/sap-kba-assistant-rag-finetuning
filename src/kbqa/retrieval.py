"""Retrieval over KBA chunks.

- BM25: classic keyword ranking (term frequency x rarity). Strong on exact technical terms such as
  table names (ACDOCA), check IDs (M1125) or statement hashes, which are very common in this KBA.
- Embeddings: bge-small-en-v1.5 (33M params, CPU) finds chunks with the same *meaning* in other words.
- Hybrid: Reciprocal Rank Fusion (RRF, Cormack et al. 2009) - every chunk gets 1/(60 + rank) from each
  ranking and the scores are added. Simple, no score normalisation, robust.
"""
import hashlib
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from kbqa.config import PROCESSED_DIR, QA_DIR, RESULTS_DIR

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # from the bge model card
RRF_K = 60


def tokenize(text):
    """Lower-case word tokens; keeps underscores and slashes so M_SQL_PLAN_CACHE or /SAPAPO/MARM stay whole."""
    return re.findall(r"[a-z0-9_/]+", text.lower())


def load_chunks():
    return [json.loads(l) for l in (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8")]


def chunk_document(chunk):
    """What the retriever indexes: the FAQ question and, for table rows, the sentence that introduces
    the table (e.g. "...MiniChecks returns a critical issue for one of the following checks:") give
    context to short rows such as "Check ID: M1110"."""
    intro = chunk.get("table_intro", "") if chunk["type"] == "table_row" else ""
    return "\n".join(p for p in (chunk["section_question"], intro, chunk["text"]) if p)


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk_document(c)) for c in chunks])

    def ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores))

    def search(self, query, k=5):
        """Return the k best chunks as (chunk, score), best first."""
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


class HybridRetriever:
    def __init__(self, chunks):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        docs = [chunk_document(c) for c in chunks]
        digest = hashlib.sha256("\n".join(docs).encode()).hexdigest()[:16]
        cache = PROCESSED_DIR / f"chunk_embeddings_{digest}.npy"      # recomputed when documents change
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.embedder.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            np.save(cache, self.emb)

    def dense_ranking(self, query):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ q)))

    def search(self, query, k=3, method="hybrid"):
        """Return the k best chunks as (chunk, score), best first. method: bm25 | dense | hybrid."""
        if method == "bm25":
            return self.bm25.search(query, k)
        if method == "dense":
            return [(self.chunks[i], 0.0) for i in self.dense_ranking(query)[:k]]
        scores = np.zeros(len(self.chunks))
        for ranking in (self.bm25.ranking(query), self.dense_ranking(query)):
            for rank, i in enumerate(ranking[:50]):
                scores[i] += 1 / (RRF_K + rank + 1)
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]


SELECTION_FILE = RESULTS_DIR / "retrieval_selection.json"


def selected_method():
    """The method chosen on the VALIDATION set (never the test set) by recall@3, see select_method()."""
    if SELECTION_FILE.exists():
        return json.loads(SELECTION_FILE.read_text())["selected"]
    return "bm25"


class Retriever(HybridRetriever):
    """The retriever used by RAG everywhere (evaluation, training data, backend)."""

    def search(self, query, k=3, method=None):
        return super().search(query, k, method or selected_method())


def select_method(k=3):
    """Compare bm25 / dense / hybrid on the answerable validation questions and store the winner."""
    retriever = HybridRetriever(load_chunks())
    val = [json.loads(l) for l in (QA_DIR / "val.jsonl").open(encoding="utf-8")]
    val = [v for v in val if v["kind"] != "unanswerable"]
    recall = {}
    for method in ("bm25", "dense", "hybrid"):
        hits = [bool(set(v["source_chunk_ids"]) & {c["chunk_id"] for c, _ in retriever.search(v["question"], k, method)})
                for v in val]
        recall[method] = round(sum(hits) / len(hits), 4)
    best = max(recall, key=recall.get)
    SELECTION_FILE.write_text(json.dumps({"metric": f"recall@{k} on val (answerable, n={len(val)})",
                                          "recall": recall, "selected": best}, indent=1))
    print(recall, "->", best)
    return best


if __name__ == "__main__":
    select_method()
