# SAP HANA Knowledge-Base Assistant: a Fine-Tuned 0.6B Model That Answers Only From the Docs

## The problem

Support engineers lose time searching long SAP Knowledge Base Articles (KBAs) for one parameter, one SQL view,
or one setting. A general chatbot does not help: it does not know the article, and it **makes up** confident
answers. A support assistant has to answer only from the article, and say *"not covered"* when the article
has no answer.

The goal was to do that with **small open models (≤ 1.1 B parameters) on a 6 GB laptop GPU**: no paid API, and
no data leaving the machine.

**Data type:** one SAP HANA KBA about SQL performance, as **semi-structured JSON with embedded HTML**: FAQ
headings, paragraphs, parameter tables and SQL code blocks. The real document is SAP-owned and **not included
here**. Data, dataset and model details are in [docs/DETAILS.md](docs/DETAILS.md).

## The solution

**Retrieval + a fine-tuned small model + a scope gate:**

1. A **scope gate** checks whether the question is close to anything in the article. If not, the assistant
   refuses without calling the model.
2. **BM25 retrieval** picks the 3 most relevant chunks of the article.
3. **Qwen3-0.6B with a LoRA adapter** writes a short answer from those 3 chunks. It was fine-tuned on
   *question + retrieved chunks → answer*, which teaches it to use the excerpts instead of guessing (RAG-SFT).

Example of the format (illustrative, not taken from the KBA):

> **Q:** Which parameter controls X for long-running statements?<br>
> **A:** `indexserver.ini → [section] → parameter_x`. true: enable X (can produce large data volumes); false: disable X (default).<br>
> *Sources: 3 excerpts shown next to the answer.*
>
> **Q:** What is the capital of France?<br>
> **A:** The knowledge base does not contain information to answer this question.

## The app

The app runs **locally** (FastAPI backend + React frontend): the model and the article stay on the machine.
The source document is not in this repository, so you cannot test with the real data. The screenshots
below are from the real local app.

![Dashboard](docs/screenshots/dashboard.jpg)

<table>
  <tr>
    <td width="50%"><b>1 · In-domain question</b><br>Input: <i>How can I capture bind values of prepared SQL statements?</i><br>Output: the exact parameter and its settings, with 3 sources<br><img src="docs/screenshots/case1_in_domain.jpg"></td>
    <td width="50%"><b>2 · In-domain question</b><br>Input: <i>How can I identify the client coding responsible for a database request?</i><br>Output: the SQL views that hold the information<br><img src="docs/screenshots/case2_in_domain.jpg"></td>
  </tr>
  <tr>
    <td><b>3 · Near the domain, not in the article</b><br>Input: <i>How do I renew an expired SAP HANA SSL certificate?</i><br>Output: refused in 0.1 s, the model is not called<br><img src="docs/screenshots/case3_near_domain_refused.jpg"></td>
    <td><b>4 · Completely off-topic</b><br>Input: <i>What is the capital of France?</i><br>Output: refused, no guessing<br><img src="docs/screenshots/case4_off_topic_refused.jpg"></td>
  </tr>
</table>

## Results of the selected system

Measured on a fixed test set (seed 42, greedy decoding). Questions are *unseen*: the facts behind them were never
in the training data.

| Metric | Qwen3-0.6B + RAG-SFT + gate |
|---|---|
| Key-fact recall (unseen questions) | **0.80** |
| ROUGE-L (unseen questions) | **0.86** |
| Semantic similarity to the reference | 0.95 |
| Off-topic questions correctly refused | **0.94** |
| In-domain questions wrongly refused | 0.03 |
| Median answer time / peak GPU memory | 6.6 s / 1.5 GB |
| Training time / adapter size | 66 min / 49 MB |

### How one answer is made

![Model flow](docs/diagrams/model_flow.svg)

### System architecture

Train once offline, save the adapter, load it in the backend, and the user asks through the frontend.

![System architecture](docs/diagrams/system_architecture.svg)

## Comparison with the other approaches

**Same model, four setups:** retrieval alone or fine-tuning alone is not enough. Only the two together work.

![Four setups](docs/figures/four_setups.png)

**All 20 systems** (gate variants left out, since the gate does not change recall). Fine-tuning *without* retrieval
(LoRA, QLoRA, IA3, prompt tuning, full fine-tuning, domain pretraining) stays at or below 0.03 recall.

![All systems](docs/figures/all_systems_recall.png)

**Quality and speed:**

![Quality vs latency](docs/figures/quality_vs_latency.png)

### Top 5 systems

| Rank | System | Key-fact recall | ROUGE-L | Off-topic refused | Wrong refusals | Median latency |
|---|---|---|---|---|---|---|
| 1 | **Qwen3-0.6B + RAG-SFT (LoRA) + gate** | **0.800** | **0.856** | **0.94** | 0.03 | 6.6 s |
| 2 | Qwen3-0.6B + RAG-SFT (LoRA) | 0.800 | 0.856 | 0.53 | 0.03 | 3.1 s |
| 3 | SmolLM2-360M + RAG-SFT (LoRA) + gate | 0.712 | 0.786 | 0.94 | 0.02 | 9.8 s |
| 4 | Retrieval only + gate (no LLM, top excerpt returned) | 0.710 | 0.670 | 0.94 | 0.00 | ≈ 0.02 s |
| 5 | TinyLlama-1.1B base + RAG (no fine-tuning) | 0.135 | 0.145 | 0.26 | 0.54 | 9.1 s |

The winner was picked by a selection rule written **before** the results (in short: refuse ≥ 80 % of off-topic
questions, beat retrieval-only, answer in ≤ 10 s on 6 GB; then rank by score).
**All 28 systems with every metric:** [results/comparison.md](results/comparison.md).

## How to run

Requirements: NVIDIA GPU with ≥ 4 GB VRAM, Python 3.12, [uv](https://docs.astral.sh/uv/), Node 20+.
You need your own knowledge-base JSON file in `raw data/` (the original is not shared).

```bash
uv sync
```

```bash
uv run python -m kbqa.run_experiments
```

```bash
uv run uvicorn backend.app:app --port 8000
```

```bash
npm --prefix frontend install
```

```bash
npm --prefix frontend run dev
```

1. `uv sync` installs the Python environment.
2. `run_experiments` runs the whole pipeline: parse, chunk, build QA, train, evaluate. This takes hours. The step-by-step
   modules are in [`src/kbqa/`](src/kbqa/).
3. Copy the selected adapter into `models/qwen3-0.6b-ragsft-lora/` (or set `KBQA_ADAPTER`), then start the API with `uvicorn`.
4. Install and start the frontend, then open http://localhost:5173.

To redraw the result figures from `results/comparison.csv`: `uv run python -m kbqa.make_figures`.

## Use it, ask me

Anyone is welcome to use or adapt this code for their own knowledge base. For questions or feedback, contact me
through my portfolio: **[ubaidur404786.github.io](https://ubaidur404786.github.io/)**.
