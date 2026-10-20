"""Phase 2, step 1 - turn the raw KBA JSON + HTML into clean, structured sections.

Run:  uv run python -m kbqa.parse_kba
Out:  data/processed/sections.jsonl   (one line per FAQ question)

Design (see docs/ARCHITECTURE.md §6):
- The Resolution section is an FAQ: every <h3> starts a new question.
- Inside a question we keep two kinds of "blocks":
    * "text"      - paragraphs, lists, code blocks (in ``` fences)
    * "table_row" - one row of a table, stored as {header: value} cells, because every row
                    of these tables is a self-contained fact (e.g. one mini-check, one statement).
- The "Cause" section only repeats the 35 questions as a table of contents -> skipped.
- Symptom + Environment are short; they become section 0 ("overview").
The raw file is only read, never modified.
"""
import json
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from kbqa.config import PROCESSED_DIR, RAW_KBA_FILES


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def normalise(text):
    """Remove invisible characters and squeeze whitespace, but keep code blocks intact."""
    text = text.replace("﻿", "").replace("\xa0", " ").replace("\r", "")
    parts = text.split("```")
    for i in range(0, len(parts), 2):          # even parts = outside code fences
        p = re.sub(r"[ \t]+", " ", parts[i])
        p = re.sub(r" *\n *", "\n", p)
        parts[i] = re.sub(r"\n{3,}", "\n\n", p)
    return "```".join(parts).strip()


# ---------------------------------------------------------------------------
# HTML -> text (recursive)
# ---------------------------------------------------------------------------
def node_to_text(node):
    """Convert any HTML node to readable plain text, preserving lists, code and nested tables."""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = node.name
    if name == "br":
        return "\n"
    if name == "img":                       # screenshots are not available -> keep alt text only
        return f"[Image: {node.get('alt', '').strip()}]"
    if name == "pre":                       # SQL / parameter snippets: keep verbatim
        code = node.get_text().strip("\n")
        return "\n```\n" + code + "\n```\n" if code.strip() else ""
    if name == "table":                     # a table *inside* a cell: one line per row
        header, rows = table_to_rows(node)
        lines = [" | ".join(f"{h}: {v}" for h, v in zip(header, r) if v) for r in rows]
        return "\n" + "\n".join(lines) + "\n"
    inner = "".join(node_to_text(c) for c in node.children)
    if name == "li":
        return "\n- " + inner.strip() + "\n"
    if name in ("p", "div", "ul", "ol", "h3", "h4"):
        return "\n" + inner + "\n"
    return inner                            # inline tags: a, strong, em, span, ...


def table_to_rows(table):
    """Return (header, rows) for a table. Only direct rows, so nested tables stay inside their cell."""
    body = table.find("tbody", recursive=False) or table
    trs = body.find_all("tr", recursive=False)
    cells = [[normalise(node_to_text(td)) for td in tr.find_all(["td", "th"], recursive=False)]
             for tr in trs]
    header, rows = cells[0], cells[1:]
    # Pad / trim so every row has exactly one value per header column.
    rows = [(r + [""] * len(header))[: len(header)] for r in rows]
    return header, rows


# ---------------------------------------------------------------------------
# KBA -> sections
# ---------------------------------------------------------------------------
def split_faq(resolution_html):
    """Yield (question_text, [nodes]) for every <h3> question in the Resolution HTML."""
    body = BeautifulSoup(resolution_html, "lxml").body
    current_q, nodes = None, []
    for child in body.children:
        if isinstance(child, Tag) and child.name == "h3":
            if current_q is not None:
                yield current_q, nodes
            current_q, nodes = normalise(child.get_text(" ")), []
        elif current_q is not None:
            nodes.append(child)
    if current_q is not None:
        yield current_q, nodes


def nodes_to_blocks(nodes):
    """Turn the HTML nodes of one FAQ answer into text blocks and table-row blocks."""
    blocks, table_no, last_text = [], 0, ""
    for node in nodes:
        if isinstance(node, Tag) and node.name == "table":
            header, rows = table_to_rows(node)
            for row_no, row in enumerate(rows):
                blocks.append({
                    "type": "table_row",
                    "table_no": table_no,
                    "row_no": row_no,
                    "table_intro": last_text[-300:],   # the sentence introducing the table
                    "cells": dict(zip(header, row)),
                })
            table_no += 1
        else:
            text = normalise(node_to_text(node))
            if text:
                blocks.append({"type": "text", "text": text})
                last_text = text
    return blocks


def parse_kba(path):
    kba = json.loads(path.read_text(encoding="utf-8"))
    texts = {s["title"]: s["text"] for s in kba["texts"]}
    base = {"kba_id": kba["_id"], "kba_title": kba["shortText"],
            "component": kba["applicationComponent"], "kba_version": kba["version"]}

    sections = []
    overview = "\n".join(normalise(node_to_text(BeautifulSoup(texts.get(t, ""), "lxml")))
                         for t in ("Symptom", "Environment"))
    sections.append({**base, "faq_no": 0, "question": "Overview (Symptom / Environment)",
                     "blocks": [{"type": "text", "text": normalise(overview)}]})

    for question, nodes in split_faq(texts["Resolution"]):
        m = re.match(r"(\d+)\.\s*(.*)", question)
        faq_no, q = (int(m.group(1)), m.group(2)) if m else (len(sections), question)
        sections.append({**base, "faq_no": faq_no, "question": q, "blocks": nodes_to_blocks(nodes)})
    return sections


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "sections.jsonl"
    n_sections = n_rows = n_text = 0
    with out.open("w", encoding="utf-8") as f:
        for path in RAW_KBA_FILES:
            for s in parse_kba(path):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_sections += 1
                n_rows += sum(b["type"] == "table_row" for b in s["blocks"])
                n_text += sum(b["type"] == "text" for b in s["blocks"])
    print(f"Wrote {out.relative_to(PROCESSED_DIR.parents[1])}: "
          f"{n_sections} sections, {n_text} text blocks, {n_rows} table rows")


if __name__ == "__main__":
    main()
