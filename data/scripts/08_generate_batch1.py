#!/usr/bin/env python3
"""
Generate condensed source summaries for batch 1 shapes (A-C).
Extracts the most relevant passages from the source texts and writes
them as smaller, readable files.
"""

import json
from pathlib import Path

SOURCES_DIR = Path(__file__).parent.parent / "assembled" / "source_texts"
OUT_DIR = Path(__file__).parent.parent / "assembled" / "batch1_summaries"

SHAPES = [
    "agnolotti", "agnolotti-alessandrini", "anelli", "anelletti", "anolini",
    "bavette", "bigoli", "bringoli", "bucatini", "busiate", "busiata",
    "candele", "cannelloni", "capellini", "cappellacci", "cappellacci-dei-briganti",
    "cappellacci-di-zucca", "cappelletti", "capunti", "casarecce", "casoncelli",
    "casonsei", "cavatelli", "cencioni", "cicatelli", "ciriole", "cjalsons",
    "cjarsons", "corzetti", "croxetti", "culurgiones", "culingionis",
]


def extract_best_passages(data: dict, max_chars: int = 4000) -> str:
    """Extract the most relevant passages from the source text, limited to max_chars."""
    by_book = data.get("sources_by_book", {})
    name = data.get("name", "")

    passages = []
    total = 0

    for book, texts in by_book.items():
        for text in texts:
            # Prioritize text that mentions the shape name
            name_lower = name.lower()
            if name_lower in text.lower():
                # Get the relevant paragraph around the mention
                text_lower = text.lower()
                idx = text_lower.find(name_lower)
                start = max(0, idx - 200)
                end = min(len(text), idx + 800)
                passage = text[start:end].strip()
                if passage and total + len(passage) < max_chars:
                    passages.append(f"[{book}]: {passage}")
                    total += len(passage)

    # If we didn't find enough, add the closest chunks
    if total < 2000:
        all_text = data.get("all_text", "")
        if all_text:
            # Get first 2000 chars
            remaining = max_chars - total
            passages.append(f"[Additional context]: {all_text[:remaining]}")

    return "\n\n".join(passages)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for slug in SHAPES:
        src_path = SOURCES_DIR / f"{slug}.json"
        if not src_path.exists():
            print(f"  Missing source: {slug}")
            continue

        with open(src_path) as f:
            data = json.load(f)

        summary = extract_best_passages(data)
        name = data.get("name", slug.replace("-", " ").title())
        category = data.get("category", "")

        out = {
            "slug": slug,
            "name": name,
            "category": category,
            "source_count": data.get("source_count", 0),
            "summary": summary,
        }

        out_path = OUT_DIR / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Done! Summaries in {OUT_DIR}")


if __name__ == "__main__":
    main()
