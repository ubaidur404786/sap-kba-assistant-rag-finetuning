"""Phase 2, step 3 - build the *canonical* (deterministic) QA pairs.

Run:  uv run python -m kbqa.build_qa
In:   data/processed/chunks.jsonl, data/processed/sections.jsonl
Out:  data/qa/canonical.jsonl

Golden rule: we never write an answer ourselves. Every answer is text copied from the KBA
(possibly shortened at a sentence/line boundary), and every QA pair records the chunk(s) it
comes from. Only the *questions* are constructed.

Three kinds of canonical QA pairs:
1. "gold"         - the 35 FAQ questions written by SAP. Answer = the opening text of the answer
                    (the full answers are up to ~24k words, far too long to be a target).
2. "table_row"    - one templated question per table row (template chosen by the table's first
                    column header). Answer = the row's other cells.
3. "unanswerable" - plausible questions the KBA does not answer (checked automatically, see
                    UNANSWERABLE). Answer = the fixed refusal sentence ABSTAIN.
Text chunks get their questions later from a local LLM (generate_questions.py).
"""
import json
import re

from transformers import AutoTokenizer

from kbqa.config import ABSTAIN, PROCESSED_DIR, QA_DIR


ANSWER_MAX_TOKENS = 300            # longer answers are cut at a line/sentence boundary

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def n_tokens(text):
    return len(tok(text, add_special_tokens=False).input_ids)


def shorten(text, limit=ANSWER_MAX_TOKENS):
    """Keep whole lines/sentences until the token limit. Returns (text, was_truncated)."""
    if n_tokens(text) <= limit:
        return text, False
    out = ""
    for unit in re.split(r"(?<=\n)|(?<=[.!?]) ", text):
        if unit is None:
            continue
        if n_tokens(out + unit) > limit:
            break
        out += unit if out.endswith(("\n", " ")) or not out else " " + unit
    out = out.strip()
    if out.count("```") % 2:        # do not leave an unclosed code block
        out += "\n```"
    return out, True


def clean_key(value, max_chars=90):
    """Make a cell value usable inside a question: drop code fences, keep first line, shorten."""
    v = value.replace("```", "").strip().split("\n")[0].strip()
    return v if len(v) <= max_chars else v[:max_chars].rsplit(" ", 1)[0] + " ..."


# ---------------------------------------------------------------------------
# Question templates for table rows. Key = first column header; some headers occur in
# several FAQ sections with different meaning, so (header, faq_no) can override.
# {key} = cleaned first-column value, other {placeholders} = other columns.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "Alert": "What does SAP HANA alert {key} ({Name}) indicate?",
    "Check ID": "What does the SAP HANA mini check {key} report?",
    "Column name": "What does the column {key} identify in SAP HANA monitoring views?",
    "Time column type": "How should the {key} time columns in the SAP HANA SQL plan cache be interpreted?",
    "Trace": "What is the {key} trace used for when capturing expensive SQL statements?",
    "SQL command": "What information does {key} provide when analysing a specific SQL statement?",
    "Tool": "What does the {key} tool show about the execution of a SQL statement in SAP HANA?",
    "Possible symptoms": "How can I tune an expensive SQL statement when I see the symptom \"{key}\"?",
    "Feature": "What is the SAP HANA feature \"{key}\" and when is it useful for SQL tuning?",
    "Parameter": "What does the SAP HANA parameter {key} control?",
    "Constellation": "What INSERT throughput is typical for \"{key}\"?",
    "Command": "What does the EXPLAIN variant {key} do in SAP HANA?",
    "Environment": "How can \"{Option}\" help to identify the client coding responsible for a database request?",
    "Problem": "Which problem can particularly large SQL statement texts cause regarding \"{key}\"?",
    "Variant": "What does a SQL statement look like when it uses {key}?",
    "Tuning approach": "How can \"{key}\" be used to tune compatibility view accesses?",
    "Reason": "Why can TUDF unfolding be blocked with the reason \"{key}\"?",
    ("Scenario", 15): "Is a database operation with \"{key}\" recorded in the SQL plan cache (M_SQL_PLAN_CACHE)?",
    ("Scenario", 26): "How can the SLT performance topic \"{key}\" be tuned?",
    ("Scenario", 27): "How can SAP ABAP client copies be tuned regarding \"{key}\"?",
    ("Scenario", 34): "Why can \"{key}\" lead to internal statement executions in SAP HANA?",
    ("Area", 18): "How can the performance of data modifications be tuned in the area \"{key}\"?",
    ("Area", 19): "How does the SAP ABAP database interface change SQL statements regarding \"{key}\"?",
    ("Area", 22): "How can the maximum SQL statement size be limited in {key}?",
}
GENERIC = "Regarding \"{section}\": what does the knowledge base say about {header} \"{key}\"?"


def row_questions(row):
    """Return 1-2 canonical questions for a table row."""
    cells, header = row["cells"], row["key_header"]
    if header == "Statement hash":           # the 326-row table of known expensive statements
        typ, objs = clean_key(cells.get("Type", "")), clean_key(cells.get("Accessed objects", ""), 60)
        qs = []
        if objs:
            qs.append(f"Is there a standard recommendation for expensive {typ} statements on {objs} in SAP HANA?")
        else:
            qs.append(f"Is there a standard recommendation for expensive {typ} statements in SAP HANA?")
        hashes = cells["Statement hash"].split()
        if len(hashes) == 1 and re.fullmatch(r"[0-9a-f]{32}", hashes[0]):
            qs.append(f"What is known about the SAP HANA statement hash {hashes[0]}?")
        return qs
    template = TEMPLATES.get((header, row["faq_no"])) or TEMPLATES.get(header) or GENERIC
    values = {h: clean_key(v) for h, v in cells.items()}
    return [template.format(key=clean_key(cells[header]), section=row["section_question"],
                            header=header.lower(), **values)]


def row_answer(cells, key_header):
    """The other cells of the row; the main 'Details' cell last and without its label."""
    others = {h: v for h, v in cells.items() if h != key_header and v}
    detail_h = next((h for h in ("Details", "Detail", "Description") if h in others), None)
    lines = [f"{h}: {v}" for h, v in others.items() if h != detail_h]
    if detail_h:
        lines.append(others[detail_h])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unanswerable questions: realistic SAP / HANA questions this KBA does not cover.
# Each has "absent" keywords; build_qa verifies none of them occurs anywhere in the KBA text,
# otherwise the question is dropped (and printed) - it might be answerable after all.
# ---------------------------------------------------------------------------
UNANSWERABLE = [
    ("How do I install SAP HANA on a new Linux server with hdblcm?", ["hdblcm"]),
    ("What is the price of an SAP HANA Enterprise Edition license?", ["price", "pricing"]),
    ("How do I perform a full data backup of SAP HANA with backint?", ["backint"]),
    ("How can I restore an SAP HANA tenant database to a point in time?", ["point in time", "point-in-time", "recovery"]),
    ("How do I renew an expired SAP HANA SSL certificate?", ["certificate"]),
    ("Which Linux kernel parameters does SAP recommend for HANA on SUSE?", ["SUSE", "kernel parameter"]),
    ("How do I reset the password of the SYSTEM user in SAP HANA?", ["password"]),
    ("How do I set up SAP HANA dynamic tiering?", ["dynamic tiering"]),
    ("How do I connect SAP Analytics Cloud to SAP HANA via a live connection?", ["Analytics Cloud"]),
    ("Which SAP HANA revision is required to upgrade to S/4HANA 2023?", ["S/4HANA 2023"]),
    ("How do I configure LDAP authentication for SAP HANA users?", ["LDAP"]),
    ("How do I migrate an Oracle database to SAP HANA with the DMO option?", ["DMO", "Oracle"]),
    ("What is the maximum number of tenant databases per SAP HANA system?", ["tenant databases"]),
    ("How can I encrypt SAP HANA data volumes at rest?", ["encrypt"]),
    ("How do I configure the SAP HANA cockpit for a new system?", ["cockpit"]),
    ("How do I activate the SAP HANA audit log for DDL statements?", ["audit"]),
    ("How do I install the SAP HANA client on Windows?", ["client installation", "hdbclient"]),
    ("How do I create a calculation view in SAP Web IDE?", ["Web IDE"]),
    ("How do I schedule a database backup in DBACOCKPIT for SAP MaxDB?", ["MaxDB"]),
    ("How can I size the memory of a new SAP HANA appliance?", ["sizing", "appliance"]),
    ("Which ports does SAP HANA use for HTTPS communication with XS Advanced?", ["XS Advanced", "XSA"]),
    ("How do I upgrade SAP HANA from SPS 05 to SPS 07 with zero downtime?", ["zero downtime"]),
    ("How do I add a new host to a scale-out SAP HANA system?", ["add host", "new host"]),
    ("How do I configure SAP HANA smart data access to a Teradata source?", ["Teradata"]),
    ("Which SAP HANA version supports the JSON document store?", ["document store"]),
    ("How do I enable the SAP HANA graph engine?", ["graph engine", "graph workspace"]),
    ("How do I configure SAP HANA spatial reference systems?", ["spatial"]),
    ("How do I uninstall an SAP HANA system with hdbuninst?", ["hdbuninst", "uninstall"]),
    ("How do I change the SID of an SAP HANA system?", ["hdbrename", "change the SID"]),
    ("How can I monitor SAP HANA with SAP Solution Manager technical monitoring?", ["technical monitoring"]),
    ("How do I configure a firewall for SAP HANA internal network communication?", ["firewall"]),
    ("How do I export an SAP HANA table to a CSV file?", ["CSV"]),
    ("What is the default port of the SAP HANA indexserver SQL interface for instance 00?", ["30015"]),
    ("How do I create a Fiori launchpad tile?", ["launchpad tile"]),
    ("How do I activate SAP HANA native storage extension warm data tiering for a table?", ["warm data", "NSE", "native storage extension"]),
    ("How do I configure email alerts in the SAP HANA alerting framework?", ["e-mail", "email"]),
    ("How do I write an ABAP report that reads data with OData?", ["write an ABAP report"]),
    ("What is the capital of France?", ["France"]),
    ("Who won the football world cup in 2014?", ["football", "world cup"]),
    ("How do I bake a chocolate cake?", ["cake"]),
    ("What is the weather forecast for Paris tomorrow?", ["weather"]),
]


def kba_full_text():
    sections = [json.loads(l) for l in (PROCESSED_DIR / "sections.jsonl").open(encoding="utf-8")]
    parts = []
    for s in sections:
        parts.append(s["question"])
        for b in s["blocks"]:
            parts.append(b.get("text", "") + " " + " ".join(b.get("cells", {}).values()))
    return " ".join(parts).lower()


def main():
    chunks = [json.loads(l) for l in (PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8")]
    sections = [json.loads(l) for l in (PROCESSED_DIR / "sections.jsonl").open(encoding="utf-8")]
    qa = []

    # 1. Gold FAQ questions -> the opening ~300 tokens of their answer (text and table rows in
    #    document order). Gold questions are TEST-ONLY: they are never trained on (see split.py).
    for sec in sections:
        if sec["faq_no"] == 0:
            continue
        used, text = [], ""
        for c in (c for c in chunks if c["faq_no"] == sec["faq_no"]):
            if text and n_tokens(text + "\n" + c["text"]) > ANSWER_MAX_TOKENS:
                break
            used.append(c["chunk_id"])
            text = f"{text}\n{c['text']}" if text else c["text"]
        answer, cut = shorten(text)
        qa.append({"kind": "gold", "unit_id": f"gold-q{sec['faq_no']:02d}", "source_chunk_ids": used,
                   "faq_no": sec["faq_no"], "question": sec["question"], "answer": answer, "truncated": cut})

    # 2. Table rows (rows split into parts are re-assembled: the unit is the whole row).
    rows = {}
    for c in chunks:
        if c["type"] == "table_row":
            unit = c["chunk_id"].split("-p")[0]
            rows.setdefault(unit, {**c, "parts": []})["parts"].append(c["chunk_id"])
    # Several rows can produce the same question (e.g. two different rows about SELECTs on
    # ACDOCA). Both answers are correct, so they are merged into ONE QA pair, and the rows are
    # merged into one split unit - otherwise the same answer text could leak across splits.
    by_question = {}
    for unit, row in rows.items():
        answer = row_answer(row["cells"], row["key_header"])
        if not answer:
            continue
        for q in row_questions(row):
            by_question.setdefault(q, []).append((unit, row, answer))
    unit_group = {u: u for u in rows}                 # row unit -> merged group unit

    def root(u):
        while unit_group[u] != u:
            u = unit_group[u]
        return u

    for entries in by_question.values():
        for unit, _, _ in entries[1:]:
            unit_group[root(unit)] = root(entries[0][0])
    for q, entries in by_question.items():
        answer, cut = shorten("\n\n".join(a for _, _, a in entries))
        qa.append({"kind": "table_row", "unit_id": root(entries[0][0]),
                   "source_chunk_ids": [p for _, row, _ in entries for p in row["parts"]],
                   "faq_no": entries[0][1]["faq_no"], "question": q, "answer": answer, "truncated": cut,
                   "merged_rows": len(entries)})
    for item in qa:                                   # second pass: all QAs use the group unit
        if item["kind"] == "table_row":
            item["unit_id"] = root(item["unit_id"])

    # 3. Unanswerable questions, verified against the whole KBA text.
    full = kba_full_text()
    for i, (q, absent) in enumerate(UNANSWERABLE):
        hits = [k for k in absent if k.lower() in full]
        if hits:
            print(f"  dropped unanswerable candidate (keyword found: {hits}): {q}")
            continue
        qa.append({"kind": "unanswerable", "unit_id": f"unanswerable-{i:02d}", "source_chunk_ids": [],
                   "faq_no": None, "question": q, "answer": ABSTAIN, "truncated": False})

    unique = qa
    assert len({x["question"].lower() for x in unique}) == len(unique), "duplicate questions"

    QA_DIR.mkdir(parents=True, exist_ok=True)
    with (QA_DIR / "canonical.jsonl").open("w", encoding="utf-8") as f:
        for i, item in enumerate(unique):
            item["qa_id"] = f"c{i:04d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    kinds = {k: sum(x["kind"] == k for x in unique) for k in ("gold", "table_row", "unanswerable")}
    print(f"Wrote data/qa/canonical.jsonl: {len(unique)} QA pairs {kinds}; "
          f"truncated answers: {sum(x['truncated'] for x in unique)}")


if __name__ == "__main__":
    main()
