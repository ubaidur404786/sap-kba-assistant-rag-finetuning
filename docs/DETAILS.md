# Project details: data, dataset, models and evaluation

The [README](../README.md) is kept short. This page has everything else: how the data looks, how the
dataset was built, which models and techniques were tested, how they were compared, and the limits.

## The data (type only, the content is not published)

The source is one SAP HANA Knowledge Base Article (KBA) about SQL performance, available to SAP customers
with an S-user. It is proprietary, so neither the document nor anything derived from it is in this repository.

- **Format:** semi-structured JSON. Metadata fields (id, title, version, language, product) plus one big HTML
  field with the article body.
- **Inside the HTML:** about 35 FAQ sections (`<h3>` question headings), each with paragraphs, bullet lists,
  parameter **tables** (`<tr>` rows: parameter → value → meaning) and SQL/code blocks (`<pre>`).
- **Size:** about 93 k tokens, parsed into 698 source-tagged chunks (one chunk = a text piece of ≤ 200 words
  or one table row).

## The dataset

Built without inventing answers: every answer is copied from a chunk, and every question is linked to it.

| Question source | How it is made | Used in |
|---|---|---|
| Original FAQ questions | the article's own `<h3>` headings (written by humans) | test only |
| Table-row questions | one template question per table row | train / val / test |
| Paraphrases | written by a local Qwen3-0.6B, kept only if every ID and table name survives | train / val / test |
| Unanswerable | hand-written questions the article does not cover | train / val / test |

The split is **by knowledge unit** (80 / 10 / 10, seed 42), so a fact in the test set was never seen in training.
That is what the "unseen" columns in the results measure. Counts are in [`results/data_report.json`](../results/data_report.json).

## How models and techniques were chosen

- **Every decision used the validation set**: retriever (BM25 recall@3 0.86 vs dense 0.57), the model to fine-tune
  (Qwen3-0.6B, best validation score among 5 tiny models), the gate signal and threshold, and the final system.
  The test set was only used to report.
- **Selection rule, fixed in advance**: eligible only if correct refusals ≥ 80 %, key facts ≥ retrieval-only baseline,
  median latency ≤ 10 s and VRAM ≤ 6 GB; then ranked by 0.5·key facts + 0.25·ROUGE-L + 0.25·refusals.
  Three systems were eligible; Qwen3-0.6B + RAG-SFT + gate ranked first (0.842 vs 0.794 and 0.764).
- **Fair comparison**: same data, seed (42), prompts, effective batch (32), epochs (2) and evaluation for every
  fine-tuning run; only the learning rate differs per method, as each method requires.

## Approach

1. **Data** — parse semi-structured HTML by role (`<h3>` FAQ question, `<tr>` fact, `<pre>` code) into 698
   source-tagged chunks; build 2,809 questions without inventing answers: original FAQ questions (test only),
   one templated question per table row, paraphrases from a local tiny LLM filtered to keep every ID and table name,
   and unanswerable questions; split by knowledge unit to prevent leakage.
2. **Baselines** — 5 tiny models closed-book and with RAG, plus retrieval-only.
3. **Fine-tuning** — closed-book SFT with 5 techniques, DAPT, DAPT→SFT, RAG-SFT on two model families.
4. **Evaluation** — key-fact recall, token F1, ROUGE-L, semantic similarity (bge-small), correct/wrong refusals,
   retrieval recall@3; latency (load, TTFT, end-to-end, tokens/s), peak VRAM, training time, adapter size.
5. **App** — FastAPI `/api/ask` (gate → BM25 → model) and a React UI that shows the answer, its sources and a
   response-time breakdown.

## Run it

Requirements: Windows or Linux, NVIDIA GPU with ≥ 4 GB VRAM, Python 3.12, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
uv sync                                               # Python environment (PyTorch CUDA build)
uv run python -m kbqa.parse_kba                       # needs your own KB JSON files in "raw data/"
uv run python -m kbqa.chunk
uv run python -m kbqa.build_qa
uv run python -m kbqa.generate_questions
uv run python -m kbqa.split
uv run python -m kbqa.run_experiments                 # all baselines, training, evaluation (hours)
uv run uvicorn backend.app:app --port 8000            # API
npm --prefix frontend install && npm --prefix frontend run dev   # UI on http://localhost:5173
```

## Limitations and constraints

- **Data**: one knowledge-base article (≈ 93 k tokens). The pipeline handles more files, but results are measured on
  one document; with more articles retrieval gets harder and closed-book fine-tuning even less realistic.
- **Synthetic questions**: most questions are templates or paraphrases from a 0.6 B model; some paraphrases drift in
  meaning. Only the 35 original FAQ questions were written by humans.
- **Metrics**: key-fact recall and ROUGE-L reward copying source wording; they do not judge reasoning or completeness
  of long answers. No LLM judge was used (judges small enough for this GPU are unreliable) and there was no
  systematic human rating — representative answers were inspected by hand only.
- **Unanswerable questions** are few (34 test, 18 validation) and hand-written, so the 94 % refusal rate has a wide
  confidence interval; the gate threshold could over-fit to their style.
- **Hardware**: latency was measured on a laptop with only 0.5–2 GB of free RAM; Windows memory paging made generation
  slow (2–8 tokens/s) and noisy, so absolute latencies are pessimistic and runs are not perfectly comparable.
  A QLoRA run once hung on the GPU (killed and re-run); full fine-tuning exceeded the 6 GB VRAM.
- **Training budget**: 2 epochs per run, one seed, no hyper-parameter search; DoRA, prefix tuning, BitFit and adapter
  layers were studied but not run (time; Qwen3 has no bias terms for BitFit).
- **Deployment**: local only. The source content is proprietary, so there is no public demo and no data in this repo.

## Tech stack and resources

**Models (Hugging Face, all Apache-2.0)**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (selected; also the
question generator), [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct),
[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (embeddings, MIT).

**Libraries**: PyTorch 2.14 (CUDA 13.0), Transformers 5.20, PEFT 0.21, TRL 1.16, bitsandbytes 0.50, Accelerate 1.15,
Datasets 5.1, sentence-transformers 6.1, rank-bm25, rouge-score, BeautifulSoup + lxml, pandas, matplotlib, Jupyter,
FastAPI + Uvicorn, React 19 + Vite 8 + TypeScript, uv.

**Papers the design relies on**: LoRA (Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) ·
QLoRA (Dettmers et al., 2023, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)) ·
IA3 (Liu et al., 2022, [arXiv:2205.05638](https://arxiv.org/abs/2205.05638)) ·
Prompt tuning (Lester et al., 2021, [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)) ·
DAPT, "Don't Stop Pretraining" (Gururangan et al., 2020, [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)) ·
RAG (Lewis et al., 2020, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) ·
Fine-tuning vs retrieval for knowledge injection (Ovadia et al., 2023, [arXiv:2312.05934](https://arxiv.org/abs/2312.05934)) ·
New knowledge in fine-tuning and hallucination (Gekhman et al., 2024, [arXiv:2405.05904](https://arxiv.org/abs/2405.05904)) ·
Round-trip consistency for synthetic QA (Alberti et al., 2019, [arXiv:1906.05416](https://arxiv.org/abs/1906.05416)) ·
Reciprocal Rank Fusion (Cormack et al., 2009).

**Hardware**: NVIDIA GeForce RTX 3050 6 GB Laptop GPU, Intel Core i5-13420H, 7.6 GB RAM, Windows 11.

## Data note

The knowledge-base content is proprietary and is not included. No raw data, generated questions, model outputs or
adapters are published; `results/` contains numbers only.
