"""FastAPI backend for the KBA question-answering chatbot (local-only demo).

Run:  uv run uvicorn backend.app:app --port 8000
      (settings via environment variables, see SETTINGS below)

Endpoints
  GET  /api/health  -> model / mode information
  POST /api/ask     {"question": "..."} -> answer, sources, abstained flag, timings, token counts

The model is loaded ONCE at start-up. A lock makes requests run one at a time: there is a single
small GPU, and parallel generation would only make every request slower.
"""
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kbqa import qa_system
from kbqa.llm import load_model
from kbqa.metrics import abstained
from kbqa.retrieval import Retriever, load_chunks

from kbqa.config import ABSTAIN, ROOT

# Defaults = the system chosen by kbqa.select_final (results/final_selection.json):
# Qwen3-0.6B + LoRA adapter fine-tuned on retrieved excerpts (RAG-SFT) + out-of-scope gate.
SETTINGS = {
    "model": os.environ.get("KBQA_MODEL", "Qwen/Qwen3-0.6B"),
    # models/ holds only the selected adapter (copied from outputs/qwen3-0.6b/ragsft-lora, git-ignored)
    "adapter": os.environ.get("KBQA_ADAPTER", str(ROOT / "models" / "qwen3-0.6b-ragsft-lora")) or None,
    "mode": os.environ.get("KBQA_MODE", "rag"),             # "rag" or "closed"
    "gate": os.environ.get("KBQA_GATE", "1") == "1",        # refuse out-of-scope questions before the LLM
}
STATE = {}
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    chunks = load_chunks()
    STATE["chunks"] = {c["chunk_id"]: c for c in chunks}
    STATE["retriever"] = Retriever(chunks)
    STATE["model"], STATE["tok"], STATE["load_s"] = load_model(SETTINGS["model"], adapter_dir=SETTINGS["adapter"])
    STATE["threshold"] = qa_system.load_gate_threshold() if SETTINGS["gate"] else None
    print(f"model ready in {STATE['load_s']:.1f} s: {SETTINGS}")
    yield
    STATE.clear()


app = FastAPI(title="KBA QA", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/api/health")
def health():
    return {"status": "ok" if "model" in STATE else "loading", **SETTINGS,
            "load_seconds": round(STATE.get("load_s", 0), 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.post("/api/ask")
def ask(req: AskRequest):
    if "model" not in STATE:
        raise HTTPException(503, "model is still loading")
    model, tok = STATE["model"], STATE["tok"]
    with LOCK:
        t0 = time.perf_counter()
        hits = []
        if STATE["threshold"] is not None and qa_system.gate_refuses(STATE["retriever"], req.question, STATE["threshold"]):
            ms = round(1000 * (time.perf_counter() - t0), 1)
            return {"answer": ABSTAIN, "abstained": True, "gated": True, "sources": [],
                    "timings_ms": {"retrieval": ms, "generation": 0.0, "backend_total": ms},
                    "tokens": {"input": 0, "output": 0}, "model": SETTINGS["model"],
                    "adapter": SETTINGS["adapter"], "mode": SETTINGS["mode"]}
        if SETTINGS["mode"] == "rag":
            hits = STATE["retriever"].search(req.question, k=qa_system.RAG_K)
            prompt = qa_system.rag_prompt(tok, req.question, hits)
        else:
            prompt = qa_system.closed_book_prompt(tok, req.question)
        t_retrieval = time.perf_counter() - t0
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=qa_system.MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[0, inputs["input_ids"].shape[1]:]
        answer = tok.decode(new, skip_special_tokens=True).strip()
        total = time.perf_counter() - t0
    return {
        "answer": answer,
        "abstained": abstained(answer),
        "gated": False,
        "sources": [{"chunk_id": c["chunk_id"], "faq_no": c["faq_no"], "section": c["section_question"],
                     "text": c["text"], "score": round(s, 4)} for c, s in hits],
        "timings_ms": {"retrieval": round(1000 * t_retrieval, 1),
                       "generation": round(1000 * (total - t_retrieval), 1), "backend_total": round(1000 * total, 1)},
        "tokens": {"input": int(inputs["input_ids"].shape[1]), "output": int(len(new))},
        "model": SETTINGS["model"], "adapter": SETTINGS["adapter"], "mode": SETTINGS["mode"],
    }
