"""Draw the README result figures from results/comparison.csv (measured numbers only).

Run:  uv run python -m kbqa.make_figures
Writes PNG files to docs/figures/.
"""
import matplotlib

matplotlib.use("Agg")  # no window: we only save files
import matplotlib.pyplot as plt
import pandas as pd

from kbqa.config import RESULTS_DIR, ROOT

OUT = ROOT / "docs" / "figures"
GREEN, BLUE, ORANGE, GREY = "#2e7d32", "#1e88e5", "#fb8c00", "#9e9e9e"


def short(system):
    # "ft/qwen3-0.6b/ragsft-lora/rag+gate" -> "Qwen3-0.6B RAG-SFT + gate"
    parts = system.split("/")
    if system.startswith("retrieval-only"):
        return "Retrieval only" + (" + gate" if "gate" in system else "")
    name = parts[1].replace("qwen", "Qwen").replace("smollm2", "SmolLM2").replace("tinyllama", "TinyLlama")
    name = name.replace("-instruct", "")
    family, size = name.rsplit("-", 1)
    name = f"{family}-{size.upper()}"          # 360m -> 360M, 0.6b -> 0.6B
    if parts[0] == "base":
        return f"{name} base" + (" + RAG" if "rag" in parts[2] else " closed") + (" + gate" if "gate" in parts[2] else "")
    method = parts[2].upper().replace("RAGSFT", "RAG-SFT").replace("-LORA", "-LoRA").replace("-QLORA", "-QLoRA")
    return f"{name} {method}" + (" + gate" if "gate" in parts[3] else "")


def color(system):
    if system.startswith("ft/") and "/rag" in system:
        return GREEN      # fine-tuned + retrieval
    if system.startswith("retrieval-only"):
        return ORANGE     # no language model
    if "/rag" in system:
        return BLUE       # base model + retrieval
    return GREY           # closed-book


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "comparison.csv")

    # 1) Every system, one row each (gate variants dropped: the gate does not change recall)
    one = df[~df.system.str.endswith("+gate")].sort_values("unseen_key_fact_recall")
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh([short(s) for s in one.system], one.unseen_key_fact_recall, color=[color(s) for s in one.system])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Key-fact recall on unseen test questions (higher is better)")
    ax.set_title("All systems: only fine-tuning + retrieval gets the facts right", fontweight="bold")
    for c, label in [(GREEN, "fine-tuned + RAG"), (ORANGE, "retrieval only (no LLM)"),
                     (BLUE, "base model + RAG"), (GREY, "closed-book (no RAG)")]:
        ax.bar(0, 0, color=c, label=label)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "all_systems_recall.png", dpi=130)

    # 2) Same model, four setups: what does each ingredient add?
    rows = {"Base\nno RAG": "base/qwen3-0.6b/closed", "Fine-tuned\nno RAG": "ft/qwen3-0.6b/sft-lora/closed",
            "Base\n+ RAG": "base/qwen3-0.6b/rag", "Fine-tuned\n+ RAG (ours)": "ft/qwen3-0.6b/ragsft-lora/rag+gate"}
    d = df.set_index("system").loc[list(rows.values())]
    panels = [("unseen_key_fact_recall", "Key-fact recall ↑"), ("unseen_rouge_l", "ROUGE-L ↑"),
              ("unanswerable_correct_refusal", "Refuses off-topic ↑"), ("unseen_false_refusal", "Wrong refusals ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (col, title) in zip(axes, panels):
        b = ax.bar(list(rows), d[col], color=[GREY, BLUE, ORANGE, GREEN])
        ax.bar_label(b, fmt="%.2f", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Qwen3-0.6B, same test set: what retrieval and fine-tuning each add", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "four_setups.png", dpi=130)

    # 3) Quality next to speed, for the systems whose latency was benchmarked
    lat = df.dropna(subset=["e2e_median_s"]).sort_values("unseen_key_fact_recall")
    names = [short(s) for s in lat.system]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    b = left.barh(names, lat.unseen_key_fact_recall, color=[color(s) for s in lat.system])
    left.bar_label(b, fmt="%.2f", padding=3, fontsize=8)
    left.set_xlim(0, 1)
    left.set_title("Key-fact recall (unseen) ↑")
    b = right.barh(names, lat.e2e_median_s, color=[color(s) for s in lat.system])
    right.bar_label(b, fmt="%.1f s", padding=3, fontsize=8)
    right.set_title("Median answer time, batch 1 ↓")
    right.set_xlim(0, lat.e2e_median_s.max() * 1.15)
    fig.suptitle("Quality and latency on the same RTX 3050 6 GB laptop GPU", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "quality_vs_latency.png", dpi=130)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
