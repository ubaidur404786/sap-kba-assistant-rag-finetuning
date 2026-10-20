"""Phase 2, step 2 - cut the clean sections into source-tagged chunks.

Run:  uv run python -m kbqa.chunk
In:   data/processed/sections.jsonl
Out:  data/processed/chunks.jsonl

A chunk is the smallest unit we retrieve, train on, and cite. Rules:
- text blocks of one FAQ answer are merged until MAX_TOKENS is reached;
- every table row is its own chunk (a row = one fact). If a row is too long, its longest
  cell is split, and every part repeats the row's short cells so it still makes sense alone;
- every chunk gets a stable id:  <kba>-q<faq>-<t|r><n>[-p<part>]
Lengths are measured with the Qwen3 tokenizer, i.e. in the same tokens the model sees.
"""
import json
import re

from transformers import AutoTokenizer

from kbqa.config import PROCESSED_DIR

MAX_TOKENS = 384          # chunk size limit (tokens); keeps SFT sequences short on a 6 GB GPU
TOKENIZER_ID = "Qwen/Qwen3-0.6B"   # all Qwen3 sizes share this tokenizer

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)


def n_tokens(text):
    return len(tok(text, add_special_tokens=False).input_ids)


def split_long_text(text, limit):
    """Split text into pieces <= limit tokens: first by lines, then by sentences, then hard cut."""
    pieces, current = [], ""
    units = re.split(r"(?<=\n)", text)                      # keep line breaks
    if len(units) == 1:
        units = re.split(r"(?<=[.!?])\s+", text)            # one long line -> sentences
    for unit in units:
        if n_tokens(unit) > limit:                           # a single huge line/sentence
            ids = tok(unit, add_special_tokens=False).input_ids
            for i in range(0, len(ids), limit):
                pieces.append(tok.decode(ids[i:i + limit]))
            continue
        if current and n_tokens(current + unit) > limit:
            pieces.append(current)
            current = ""
        current += unit if current else unit.lstrip()
    if current.strip():
        pieces.append(current)
    # Drop pieces that contain nothing but code-fence markers (left over when a code block is split).
    return [p.strip() for p in pieces if re.sub(r"[`\s]", "", p)]


def row_text(cells):
    """'Header: value' per line, skipping empty cells."""
    return "\n".join(f"{h}: {v}" for h, v in cells.items() if v)


def short_key(value, max_chars=80):
    """First line of a cell, shortened - used to name a row in its continuation parts."""
    first = value.split("\n")[0].strip()
    return first if len(first) <= max_chars else first[:max_chars].rsplit(" ", 1)[0] + " ..."


def chunk_section(sec):
    base = {"kba_id": sec["kba_id"], "faq_no": sec["faq_no"], "section_question": sec["question"]}
    prefix = f"{sec['kba_id']}-q{sec['faq_no']:02d}"
    chunks, buffer, t_no, r_no = [], [], 0, 0

    def flush_text():
        nonlocal buffer, t_no
        if not buffer:
            return
        for piece in split_long_text("\n".join(buffer), MAX_TOKENS):
            chunks.append({**base, "chunk_id": f"{prefix}-t{t_no:03d}", "type": "text", "text": piece})
            t_no += 1
        buffer = []

    for block in sec["blocks"]:
        if block["type"] == "text":
            if buffer and n_tokens("\n".join(buffer + [block["text"]])) > MAX_TOKENS:
                flush_text()
            buffer.append(block["text"])
            continue

        flush_text()
        cells = block["cells"]
        headers = list(cells)
        row = {**base, "type": "table_row", "table_no": block["table_no"], "row_no": block["row_no"],
               "table_intro": block["table_intro"], "key_header": headers[0], "key_value": cells[headers[0]],
               "cells": cells}
        full = row_text(cells)
        if n_tokens(full) <= MAX_TOKENS:
            chunks.append({**row, "chunk_id": f"{prefix}-r{r_no:03d}", "text": full})
        else:
            # Too long for one chunk: split the row text and start every part with a short
            # context line naming the row, so each part still makes sense on its own.
            context = f"[{headers[0]}: {short_key(cells[headers[0]])} (continued)]"
            budget = MAX_TOKENS - n_tokens(context) - 2
            for part, piece in enumerate(split_long_text(full, budget)):
                text = piece if part == 0 else f"{context}\n{piece}"
                chunks.append({**row, "chunk_id": f"{prefix}-r{r_no:03d}-p{part}", "part": part,
                               "text": text})
        r_no += 1
    flush_text()
    return chunks


def main():
    sections = [json.loads(l) for l in (PROCESSED_DIR / "sections.jsonl").open(encoding="utf-8")]
    chunks = [c for s in sections for c in chunk_section(s)]
    for c in chunks:
        c["n_tokens"] = n_tokens(c["text"])
    with (PROCESSED_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lengths = sorted(c["n_tokens"] for c in chunks)
    n_text = sum(c["type"] == "text" for c in chunks)
    n_parts = sum("part" in c for c in chunks)
    print(f"{len(chunks)} chunks ({n_text} text, {len(chunks) - n_text} table-row, "
          f"{n_parts} of them parts of split rows)")
    print(f"tokens/chunk: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}, total {sum(lengths)}")


if __name__ == "__main__":
    main()
