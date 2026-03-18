#!/usr/bin/env python3
"""
Phase 4: Dough Recipe Entity Extraction
=========================================
Extracts and normalizes base dough recipes from all sources.
Uses semantic search to find dough-related content, then
pattern matching to extract ingredients and methods.

Target dough categories:
- Egg dough (standard, rich/extra yolk, whole egg only)
- Semolina & water dough (no egg — Pugliese style)
- Flour & water dough
- Buckwheat dough
- Chestnut flour dough
- Filled pasta dough (thinner, more pliable)
- Colored doughs (spinach, beet, squid ink, saffron, etc.)
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings


# ── Configuration ────────────────────────────────────────────────────────────

VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"
EXTRACTED_DIR = Path(__file__).parent.parent / "extracted"
OUTPUT_DIR = Path(__file__).parent.parent / "entities" / "doughs"


# ── Semantic Search for Dough Content ─────────────────────────────────────────


def find_dough_content(collection) -> list[dict]:
    """Use semantic search to find all dough-related content."""
    queries = [
        "basic egg pasta dough recipe flour eggs",
        "semolina water dough recipe no eggs",
        "pasta dough ingredients flour semolina",
        "how to make fresh pasta dough from scratch",
        "egg yolk pasta dough rich",
        "buckwheat pasta dough recipe",
        "spinach pasta dough green colored",
        "filled pasta dough recipe ravioli tortellini",
        "durum wheat semolina pasta dough",
        "pasta dough kneading resting instructions",
        "squid ink black pasta dough",
        "saffron pasta dough yellow",
        "chestnut flour pasta dough",
        "whole wheat pasta dough recipe",
        "rye flour pasta dough",
        "cocoa chocolate pasta dough",
        "beet pasta dough red",
    ]

    all_results = []
    seen_ids = set()

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=20,
        )

        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            if chunk_id not in seen_ids and dist < 1.3:
                seen_ids.add(chunk_id)
                all_results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "query": query,
                })

    return all_results


# ── Recipe Pattern Extraction ─────────────────────────────────────────────────


def extract_ingredients(text: str) -> list[dict]:
    """Extract ingredient lines from text."""
    ingredients = []

    # Common patterns:
    # "300g flour" or "300 g flour" or "2 cups flour"
    # "3 large eggs" or "3 eggs"
    # "1 tablespoon olive oil"
    patterns = [
        # Metric: "300g flour" or "300 g flour"
        r"(\d+(?:\.\d+)?)\s*(?:g|grams?)\s+(?:of\s+)?(.+?)(?:\n|$|,)",
        # Metric: "150ml water" or "150 ml water"
        r"(\d+(?:\.\d+)?)\s*(?:ml|milliliters?)\s+(?:of\s+)?(.+?)(?:\n|$|,)",
        # Imperial: "2 cups flour"
        r"(\d+(?:/\d+)?(?:\s*-\s*\d+(?:/\d+)?)?)\s+(cups?|tablespoons?|tbsp|teaspoons?|tsp|pounds?|lbs?|ounces?|oz)\s+(?:of\s+)?(.+?)(?:\n|$|,)",
        # Count: "3 large eggs" or "4 egg yolks"
        r"(\d+)\s+(large\s+|medium\s+|small\s+)?(eggs?|egg\s+yolks?|egg\s+whites?)(?:\n|$|,)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            ingredients.append({
                "raw": match.group(0).strip().rstrip(","),
                "amount": match.group(1),
            })

    return ingredients


def classify_dough_type(text: str) -> str | None:
    """Classify the type of dough based on text content."""
    text_lower = text.lower()

    classifications = [
        ("egg-yolk-rich", ["egg yolk", "yolks only", "extra yolk", "rich dough"]),
        ("whole-egg", ["whole egg", "eggs", "egg dough"]),
        ("semolina-water", ["semolina", "water", "no egg", "without egg", "eggless"]),
        ("buckwheat", ["buckwheat", "grano saraceno"]),
        ("chestnut", ["chestnut", "castagna", "farina di castagne"]),
        ("spinach", ["spinach", "spinaci", "green pasta", "green dough"]),
        ("beet", ["beet", "beetroot", "barbabietola", "red dough"]),
        ("squid-ink", ["squid ink", "nero di seppia", "black pasta", "cuttlefish"]),
        ("saffron", ["saffron", "zafferano", "yellow dough"]),
        ("cocoa", ["cocoa", "cacao", "chocolate pasta"]),
        ("whole-wheat", ["whole wheat", "wholemeal", "integrale"]),
        ("rye", ["rye", "segale"]),
        ("flour-water", ["flour and water", "flour water", "simple dough"]),
    ]

    scores = {}
    for dough_type, keywords in classifications:
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dough_type] = score

    if scores:
        return max(scores, key=scores.get)
    return None


# ── Main Pipeline ────────────────────────────────────────────────────────────


CANONICAL_DOUGHS = {
    "egg-dough": {
        "name": "Standard Egg Dough",
        "slug": "egg-dough",
        "description": "The classic Italian pasta dough — soft wheat flour and eggs. The foundation for most northern Italian pasta shapes.",
        "keywords": ["egg", "eggs", "flour and egg", "pasta all'uovo", "uovo"],
    },
    "egg-yolk-rich": {
        "name": "Rich Egg Yolk Dough",
        "slug": "egg-yolk-rich",
        "description": "An extra-rich dough using mostly or all egg yolks. Creates a deeply golden, silky pasta perfect for delicate shapes like tajarin and maltagliati.",
        "keywords": ["egg yolk", "yolks", "rich", "golden"],
    },
    "semolina-water": {
        "name": "Semolina & Water Dough",
        "slug": "semolina-water",
        "description": "The quintessential southern Italian dough — semolina flour and water, no eggs. Creates a firm, rustic texture ideal for hand-shaped pasta.",
        "keywords": ["semolina", "water", "no egg", "southern", "puglia", "orecchiette"],
    },
    "flour-water": {
        "name": "Flour & Water Dough",
        "slug": "flour-water",
        "description": "Simple dough of soft wheat flour and water. Used across Italy for hand-shaped and hand-cut pastas where a tender, pliable texture is desired.",
        "keywords": ["flour", "water", "simple", "pici", "pinci"],
    },
    "buckwheat": {
        "name": "Buckwheat Dough",
        "slug": "buckwheat",
        "description": "Earthy, nutty dough using buckwheat flour, traditional in Lombardy and Valtellina. Used for pizzoccheri and other Alpine pasta.",
        "keywords": ["buckwheat", "grano saraceno", "pizzoccheri", "valtellina"],
    },
    "chestnut": {
        "name": "Chestnut Flour Dough",
        "slug": "chestnut",
        "description": "Sweet, delicate dough using chestnut flour. Traditional in Liguria and other chestnut-growing regions.",
        "keywords": ["chestnut", "castagna", "farina di castagne"],
    },
    "spinach": {
        "name": "Spinach Dough",
        "slug": "spinach",
        "description": "Vibrant green dough enriched with pureed spinach. Used for colorful lasagne, tagliatelle verde, and festive preparations.",
        "keywords": ["spinach", "verde", "green"],
    },
    "squid-ink": {
        "name": "Squid Ink Dough",
        "slug": "squid-ink",
        "description": "Dramatic black dough colored with squid (or cuttlefish) ink. Adds a subtle briny flavor. Popular in Venice and coastal regions.",
        "keywords": ["squid ink", "nero", "cuttlefish", "seppia"],
    },
    "saffron": {
        "name": "Saffron Dough",
        "slug": "saffron",
        "description": "Golden dough infused with saffron. Traditional in Sardinia for malloreddus and other regional shapes.",
        "keywords": ["saffron", "zafferano", "malloreddus", "sardinia"],
    },
    "whole-wheat": {
        "name": "Whole Wheat Dough",
        "slug": "whole-wheat",
        "description": "Hearty dough using whole wheat or a blend of whole wheat and white flour. Nuttier flavor and more rustic texture.",
        "keywords": ["whole wheat", "integrale", "wholemeal"],
    },
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 4: Dough Recipe Entity Extraction")
    print("=" * 60)

    # Connect to vector DB
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("pasta_knowledge")

    # Find dough-related content
    print("\n  Searching for dough content across all sources...")
    dough_chunks = find_dough_content(collection)
    print(f"  → Found {len(dough_chunks)} relevant chunks")

    # Group chunks by dough type
    dough_evidence = {slug: [] for slug in CANONICAL_DOUGHS}
    unclassified = []

    for chunk in dough_chunks:
        dough_type = classify_dough_type(chunk["text"])

        # Map to canonical dough
        type_to_canonical = {
            "whole-egg": "egg-dough",
            "egg-yolk-rich": "egg-yolk-rich",
            "semolina-water": "semolina-water",
            "flour-water": "flour-water",
            "buckwheat": "buckwheat",
            "chestnut": "chestnut",
            "spinach": "spinach",
            "squid-ink": "squid-ink",
            "saffron": "saffron",
            "whole-wheat": "whole-wheat",
        }

        canonical = type_to_canonical.get(dough_type)
        if canonical and canonical in dough_evidence:
            dough_evidence[canonical].append(chunk)
        else:
            unclassified.append(chunk)

    # Build dough profiles
    dough_profiles = {}

    for slug, canonical in CANONICAL_DOUGHS.items():
        evidence = dough_evidence.get(slug, [])
        if not evidence:
            # Still include canonical doughs even without direct evidence
            dough_profiles[slug] = {
                **canonical,
                "evidence_count": 0,
                "source_books": [],
                "sample_recipes": [],
                "ingredients_found": [],
            }
            continue

        # Collect source books
        source_books = list(set(
            chunk["metadata"]["book_title"] for chunk in evidence
        ))

        # Extract ingredients from evidence
        all_ingredients = []
        for chunk in evidence[:10]:  # Check top 10 chunks
            ingredients = extract_ingredients(chunk["text"])
            all_ingredients.extend(ingredients)

        # Collect recipe text samples
        sample_recipes = []
        for chunk in evidence[:5]:
            sample_recipes.append({
                "book": chunk["metadata"]["book_title"],
                "author": chunk["metadata"]["book_author"],
                "section": chunk["metadata"]["section_title"],
                "text_preview": chunk["text"][:500],
                "distance": chunk["distance"],
            })

        dough_profiles[slug] = {
            **canonical,
            "evidence_count": len(evidence),
            "source_books": source_books,
            "sample_recipes": sample_recipes,
            "ingredients_found": all_ingredients[:20],
        }

    # Save individual dough files
    for slug, profile in dough_profiles.items():
        dough_path = OUTPUT_DIR / f"{slug}.json"
        with open(dough_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    # Save full dough registry
    registry_path = OUTPUT_DIR.parent / "dough_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_doughs": len(dough_profiles),
            "doughs": dough_profiles,
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase 4 complete!")
    print(f"  Canonical doughs: {len(dough_profiles)}")
    print(f"  Unclassified chunks: {len(unclassified)}")
    print(f"\n  Dough types and evidence:")
    for slug, profile in sorted(
        dough_profiles.items(),
        key=lambda x: x[1]["evidence_count"],
        reverse=True,
    ):
        evidence = profile["evidence_count"]
        books = len(profile["source_books"])
        print(f"    {profile['name']:<30s} {evidence:3d} chunks from {books} books")


if __name__ == "__main__":
    main()
