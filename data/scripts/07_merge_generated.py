#!/usr/bin/env python3
"""
Phase 7: Merge Generated Content
==================================
Takes the generated content from the agents (per-shape and per-dough JSON)
and merges it into a clean final database for the website.
"""

import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
GENERATED_DIR = BASE_DIR / "assembled" / "generated"
OUTPUT_PATH = BASE_DIR / "assembled" / "pasta_database_v2.json"
SITE_DATA = BASE_DIR.parent / "site" / "src" / "data" / "pasta_database.json"


def load_generated_shapes() -> dict:
    """Load all generated shape JSON files."""
    shapes = {}
    for f in sorted(GENERATED_DIR.glob("*.json")):
        if f.name.startswith("dough-"):
            continue
        with open(f) as fh:
            data = json.load(fh)
            slug = data.get("slug", f.stem)
            shapes[slug] = data
    return shapes


def load_generated_doughs() -> dict:
    """Load all generated dough JSON files."""
    doughs = {}
    for f in sorted(GENERATED_DIR.glob("dough-*.json")):
        with open(f) as fh:
            data = json.load(fh)
            slug = data.get("slug", f.stem.replace("dough-", ""))
            doughs[slug] = data
    return doughs


def main():
    print("Merging generated content...\n")

    shapes = load_generated_shapes()
    doughs = load_generated_doughs()

    print(f"  Generated shapes: {len(shapes)}")
    print(f"  Generated doughs: {len(doughs)}")

    # Quality stats
    with_instructions = sum(1 for s in shapes.values() if s.get("instructions"))
    with_history = sum(1 for s in shapes.values() if s.get("history"))
    with_description = sum(1 for s in shapes.values() if s.get("description"))

    print(f"\n  Shape quality:")
    print(f"    With description:  {with_description}")
    print(f"    With instructions: {with_instructions}")
    print(f"    With history:      {with_history}")

    dough_with_recipe = sum(1 for d in doughs.values() if d.get("ingredients"))
    print(f"\n  Dough quality:")
    print(f"    With ingredients:  {dough_with_recipe}")

    # Build final database
    database = {
        "total_shapes": len(shapes),
        "total_doughs": len(doughs),
        "shapes": shapes,
        "doughs": doughs,
    }

    # Save v2 database
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUTPUT_PATH}")

    # Copy to site
    with open(SITE_DATA, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)
    print(f"  Copied to: {SITE_DATA}")

    print("\nDone!")


if __name__ == "__main__":
    main()
