"""Phase 1 - print a structural summary of the raw KBA JSON file(s).

Run:  uv run python -m kbqa.inspect_raw

It only *reads* the raw data and prints counts, never modifies anything.
The numbers it prints are the ones recorded in docs/MEMORY.md.
"""
import collections
import json
import re

from bs4 import BeautifulSoup

from kbqa.config import RAW_KBA_FILES


def summarise_kba(path):
    kba = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n=== {path.name}: KBA {kba['_id']} - {kba['shortText']}")
    print(f"component={kba['applicationComponent']}  category={kba['categoryText']}  "
          f"version={kba['version']}  lastChanged={kba['lastChangedOn']}")

    # Which top-level fields actually carry a value?
    filled = [k for k, v in kba.items() if v not in (None, [], "")]
    empty = [k for k, v in kba.items() if v in (None, [], "")]
    print(f"{len(filled)} filled fields, {len(empty)} empty fields")

    # The 'texts' list holds the article sections as HTML strings.
    for section in kba["texts"]:
        soup = BeautifulSoup(section["text"], "lxml")
        words = len(soup.get_text(" ", strip=True).split())
        print(f"  section {section['type']}: {section['title']!r:28} "
              f"html_chars={len(section['text']):>7} words={words:>6}")

    # The Resolution section is an FAQ: one <h3> per question.
    resolution = next(s["text"] for s in kba["texts"] if s["title"] == "Resolution")
    soup = BeautifulSoup(resolution, "lxml")
    questions = [h.get_text(" ", strip=True).lstrip("﻿") for h in soup.find_all("h3")]
    print(f"  FAQ questions (<h3>): {len(questions)}")
    tags = collections.Counter(t.name for t in soup.find_all(True))
    print(f"  HTML tags: {dict(tags.most_common(12))}")
    print(f"  tables={len(soup.find_all('table'))}  rows={len(soup.find_all('tr'))}  "
          f"code blocks (<pre>)={len(soup.find_all('pre'))}  images={len(soup.find_all('img'))}")
    note_links = re.findall(r"/sap/support/notes/(\d+)", resolution)
    print(f"  links to other SAP Notes: {len(note_links)} ({len(set(note_links))} unique)")

    refs = collections.Counter(r["grDefTitle"] for r in kba["references"] or [])
    print(f"  references: {sum(refs.values())} -> {dict(refs)} (titles only, no content)")


if __name__ == "__main__":
    for p in RAW_KBA_FILES:
        summarise_kba(p)
