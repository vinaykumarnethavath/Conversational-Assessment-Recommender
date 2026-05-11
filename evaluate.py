from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from rag import AssessmentRAG


def _extract_user_messages(md_text: str) -> List[str]:
    # Matches blocks like:
    # **User**
    #
    # > message
    messages: List[str] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "**User**":
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(">"):
                i += 1
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                msg = lines[i].lstrip()[1:].strip()
                if msg:
                    messages.append(msg)
                i += 1
            continue
        i += 1
    return messages


def _extract_final_shortlist_names(md_text: str) -> List[str]:
    # Extract the last markdown table in the doc, then parse the Name column.
    # Table format:
    # | # | Name | Test Type | ... |
    # |---|------|-----------|-----|
    # | 1 | Occupational Personality Questionnaire OPQ32r | P | ... |
    tables: List[List[str]] = []
    current: List[str] = []
    for line in md_text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    if not tables:
        return []

    table = tables[-1]
    header = table[0]
    cols = [c.strip() for c in header.strip("|").split("|")]
    try:
        name_idx = cols.index("Name")
    except ValueError:
        return []

    names: List[str] = []
    for row in table[2:]:
        parts = [c.strip() for c in row.strip("|").split("|")]
        if len(parts) <= name_idx:
            continue
        name = parts[name_idx]
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            names.append(name)
    return names


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = {r.lower().strip() for r in relevant}
    ret = {r.lower().strip() for r in retrieved[:k]}
    hits = len(rel.intersection(ret))
    return hits / len(rel)


def evaluate(sample_dir: Path, k: int = 10) -> Dict[str, float]:
    rag = AssessmentRAG()

    recalls: List[float] = []
    for md_path in sorted(sample_dir.glob("*.md")):
        md_text = md_path.read_text(encoding="utf-8")
        user_messages = _extract_user_messages(md_text)
        relevant = _extract_final_shortlist_names(md_text)
        if not user_messages or not relevant:
            continue

        messages = [{"role": "user", "content": m} for m in user_messages]
        state = rag.extract_state(messages)
        retrieved_items = rag.search(state, top_k=k)
        retrieved_names = [item.get("name", "") for item in retrieved_items]

        recalls.append(recall_at_k(relevant, retrieved_names, k))

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return {"mean_recall@k": mean_recall, "num_conversations": float(len(recalls))}


if __name__ == "__main__":
    base = Path("data") / "sample_conversations" / "GenAI_SampleConversations"
    metrics = evaluate(base, k=10)
    print(metrics)
