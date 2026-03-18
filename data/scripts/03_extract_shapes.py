#!/usr/bin/env python3
"""
Phase 3: Pasta Shape Entity Extraction
========================================
Uses a hybrid approach to extract pasta shape entities:
1. Pattern-based extraction from structured sources (encyclopedias, A-Z books)
2. Semantic search to find mentions across all sources
3. Deduplication and alias resolution to build canonical registry

Key sources for shape identification:
- zanini_encyclopedia: THE definitive reference (300+ shapes)
- roddy_atoz: Alphabetical entries with narrative
- coastal_encyclopedia: 350+ recipes organized by shape
- louis_byhand: Hand-shaped pasta focus
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import chromadb
from chromadb.config import Settings


# ── Configuration ────────────────────────────────────────────────────────────

EXTRACTED_DIR = Path(__file__).parent.parent / "extracted"
VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"
ENTITIES_DIR = Path(__file__).parent.parent / "entities" / "shapes"
OUTPUT_DIR = Path(__file__).parent.parent / "entities"

# Shape categories
CATEGORIES = {
    "hand-cut": "Pasta cut by hand (knife, wheel, or scissors) from rolled sheets",
    "hand-shaped": "Pasta shaped by hand from pieces of dough (no rolling pin needed)",
    "filled": "Pasta shapes that contain a filling (ravioli, tortellini, etc.)",
    "extruded": "Pasta pushed through a die or press (industrial or hand-cranked)",
    "rolled": "Pasta rolled into sheets and then cut or shaped",
}


# ── Pattern-based Extraction from Zanini Encyclopedia ─────────────────────────


def extract_from_zanini(filepath: Path) -> list[dict]:
    """
    Extract pasta shape entries from Zanini De Vita's Encyclopedia.
    This is the gold standard — each entry follows a consistent pattern:
    shape name (in caps or bold), followed by regional info and description.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []
    # The encyclopedia text is in the raw_pages
    full_text = "\n\n".join(p["text"] for p in data.get("raw_pages", []))

    # Zanini entries typically follow pattern:
    # SHAPE NAME (or shape name in italic)
    # ALSO KNOWN AS: aliases
    # Region/area info
    # Description text

    # Find entries by looking for capitalized pasta names followed by descriptions
    # The encyclopedia has entries like "Agnolini", "Agnolotti", etc.

    # Strategy: scan for lines that look like pasta entry headings
    lines = full_text.split("\n")
    current_entry = None
    entry_text_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Detect entry headings: short lines that look like pasta names
        # Zanini entries are typically single Italian words or short phrases
        is_heading = (
            len(stripped) < 60
            and not stripped.startswith("(")
            and not stripped.startswith("[")
            and not stripped.endswith(".")
            and not stripped[0].isdigit()
            and stripped[0].isalpha()
            # Check if it looks like an Italian pasta name
            and re.match(r"^[A-Za-zàèéìòù\s\'-]+$", stripped)
            and len(stripped.split()) <= 5
        )

        # Also detect ALSO KNOWN AS patterns
        also_known = re.match(r"also known as:?\s*(.*)", stripped, re.IGNORECASE)

        if is_heading and len(stripped) > 2:
            # Save previous entry
            if current_entry and entry_text_lines:
                current_entry["raw_text"] = "\n".join(entry_text_lines)
                if len(current_entry["raw_text"]) > 100:  # Filter out noise
                    shapes.append(current_entry)

            current_entry = {
                "name": stripped.strip(),
                "source": "zanini_encyclopedia",
                "aliases": [],
                "raw_text": "",
            }
            entry_text_lines = []
        elif current_entry:
            if also_known:
                aliases = [a.strip() for a in also_known.group(1).split(",")]
                current_entry["aliases"].extend(aliases)
            entry_text_lines.append(stripped)

    # Save last entry
    if current_entry and entry_text_lines:
        current_entry["raw_text"] = "\n".join(entry_text_lines)
        if len(current_entry["raw_text"]) > 100:
            shapes.append(current_entry)

    return shapes


def extract_from_roddy(filepath: Path) -> list[dict]:
    """
    Extract pasta shapes from Roddy's A-Z of Pasta.
    The book is organized alphabetically with narrative entries per shape.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []

    for section in data["sections"]:
        text = section.get("text", "")
        title = section.get("title", "")

        # Look for section headings that are pasta names
        # Roddy uses markdown-style headings from our extraction
        heading_matches = re.finditer(
            r"^(?:#{1,3}\s+)?([A-Z][a-zàèéìòù]+(?:\s+[a-zàèéìòù]+)*)\s*$",
            text,
            re.MULTILINE,
        )

        for match in heading_matches:
            name = match.group(1).strip()
            # Get text after this heading until next heading
            start = match.end()
            next_heading = re.search(
                r"^(?:#{1,3}\s+)?[A-Z][a-zàèéìòù]+",
                text[start:],
                re.MULTILINE,
            )
            end = start + next_heading.start() if next_heading else len(text)
            entry_text = text[start:end].strip()

            if len(entry_text) > 50 and len(name) > 3:
                shapes.append({
                    "name": name,
                    "source": "roddy_atoz",
                    "aliases": [],
                    "raw_text": entry_text[:2000],
                })

    return shapes


def extract_from_coastal(filepath: Path) -> list[dict]:
    """
    Extract pasta shapes from Coastal Kitchen's Encyclopedia of Pasta.
    Has 350+ recipes organized by pasta type.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []
    seen_names = set()

    for section in data["sections"]:
        title = section.get("title", "")
        text = section.get("text", "")

        if not title:
            continue

        # Many sections are named after pasta shapes
        # Look for titles that are pasta names (not "Introduction", "Index", etc.)
        skip_words = {
            "introduction", "index", "contents", "copyright", "acknowledgments",
            "about", "foreword", "preface", "glossary", "bibliography", "notes",
            "cover", "title", "dedication", "section", "chapter", "part",
        }

        title_lower = title.lower().strip()
        if title_lower in skip_words or len(title) < 3:
            continue

        # Extract the pasta name from recipe titles like "Pappardelle with Duck Ragu"
        pasta_match = re.match(
            r"^([\w\s'àèéìòù]+?)(?:\s+(?:with|in|al|alla|alle|ai|con|e)\s+.+)?$",
            title,
            re.IGNORECASE,
        )

        if pasta_match:
            name = pasta_match.group(1).strip()
            name_lower = name.lower()

            if name_lower not in seen_names and len(name) > 3:
                seen_names.add(name_lower)
                shapes.append({
                    "name": name,
                    "source": "coastal_encyclopedia",
                    "aliases": [],
                    "raw_text": text[:1500],
                })

    return shapes


def extract_from_louis(filepath: Path) -> list[dict]:
    """
    Extract pasta shapes from Jenn Louis's Pasta By Hand.
    Focused specifically on hand-shaped regional pasta.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []

    for section in data["sections"]:
        title = section.get("title", "")
        text = section.get("text", "")

        if not title or len(text) < 100:
            continue

        # Jenn Louis organizes by shape name with regional context
        skip_words = {
            "introduction", "index", "contents", "copyright", "acknowledgments",
            "about", "foreword", "preface", "glossary", "bibliography",
            "equipment", "ingredients", "techniques", "basics", "sauces",
            "sources", "resources", "metric",
        }

        if title.lower().strip() in skip_words:
            continue

        # Check if the section text mentions pasta-making techniques
        pasta_indicators = [
            "dough", "roll", "shape", "cut", "flour", "semolina",
            "egg", "knead", "pasta", "hand", "press", "pinch",
        ]

        has_pasta_content = any(
            indicator in text.lower() for indicator in pasta_indicators
        )

        if has_pasta_content and len(title) > 2:
            shapes.append({
                "name": title.strip(),
                "source": "louis_byhand",
                "category_hint": "hand-shaped",
                "aliases": [],
                "raw_text": text[:2000],
            })

    return shapes


# ── Deduplication & Canonical Registry ────────────────────────────────────────


# Well-known pasta shapes for validation and dedup
KNOWN_PASTA_SHAPES = {
    # Long pasta
    "spaghetti", "linguine", "fettuccine", "tagliatelle", "pappardelle",
    "bucatini", "vermicelli", "capellini", "angel hair", "bigoli",
    "pici", "tonnarelli", "spaghetti alla chitarra", "chitarra",
    "tagliolini", "trenette", "bavette",
    # Short pasta
    "penne", "rigatoni", "fusilli", "farfalle", "conchiglie",
    "orecchiette", "cavatelli", "gnocchetti", "maccheroni", "macaroni",
    "mezze maniche", "paccheri", "calamarata", "ziti", "tortiglioni",
    "casarecce", "gemelli", "strozzapreti", "trofie", "busiate",
    "malloreddus", "lorighittas", "garganelli",
    # Filled pasta
    "ravioli", "tortellini", "tortelloni", "agnolotti", "cappelletti",
    "pansotti", "mezzelune", "culurgiones", "anolini", "cjarsons",
    "schlutzkrapfen", "casoncelli",
    # Sheet/ribbon pasta
    "lasagna", "lasagne", "cannelloni", "manicotti", "crespelle",
    # Soup/small pasta
    "orzo", "stelline", "ditalini", "pastina", "acini di pepe",
    "alfabeto", "anellini", "tubettini", "fregula", "fregola",
    # Dumpling-like
    "gnocchi", "gnudi", "canederli", "passatelli", "pisarei",
    # Hand-shaped
    "cavatelli", "foglie d'ulivo", "orecchiette", "strascinati",
    "capunti", "cicatelli", "sagne", "lagane",
    "maccheroni al ferretto", "fusilli al ferretto",
}


def normalize_name(name: str) -> str:
    """Normalize a pasta shape name for deduplication."""
    name = name.lower().strip()
    name = re.sub(r"[''`]", "'", name)
    name = re.sub(r"\s+", " ", name)
    # Remove common suffixes that create duplicates
    name = re.sub(r"\s*\(.*?\)\s*$", "", name)  # Remove parenthetical
    return name


def is_likely_pasta_shape(name: str, text: str = "") -> bool:
    """Heuristic to determine if a name is likely a pasta shape."""
    normalized = normalize_name(name)

    # Direct match with known shapes
    if normalized in KNOWN_PASTA_SHAPES:
        return True

    # Check if any known shape is a substring
    for known in KNOWN_PASTA_SHAPES:
        if known in normalized or normalized in known:
            return True

    # Check text context for pasta-related terms
    if text:
        pasta_terms = [
            "pasta", "dough", "flour", "semolina", "shape", "noodle",
            "boil", "sauce", "rolled", "cut", "hand-shaped", "extruded",
            "durum", "wheat", "region", "italian", "traditional",
        ]
        term_count = sum(1 for term in pasta_terms if term in text.lower())
        if term_count >= 3:
            return True

    # Italian-sounding name heuristics
    italian_endings = [
        "etti", "ette", "ini", "ine", "oni", "one", "acci", "acci",
        "elle", "elli", "otti", "otte", "ucci", "ucce", "ali", "ale",
        "ari", "ata", "ate", "ato", "ola", "ole", "oli",
    ]
    if any(normalized.endswith(ending) for ending in italian_endings):
        return True

    return False


def build_canonical_registry(all_shapes: list[dict]) -> dict:
    """
    Deduplicate shapes and build a canonical registry.
    Returns a dict of canonical_name -> shape_data.
    """
    # Group by normalized name
    groups = defaultdict(list)
    for shape in all_shapes:
        norm = normalize_name(shape["name"])
        groups[norm].append(shape)

    registry = {}

    for norm_name, entries in groups.items():
        # Skip if doesn't look like a pasta shape
        all_text = " ".join(e.get("raw_text", "") for e in entries)
        if not is_likely_pasta_shape(norm_name, all_text):
            continue

        # Pick the best display name (prefer proper capitalization)
        display_names = [e["name"] for e in entries]
        # Prefer names that start with uppercase and aren't ALL CAPS
        good_names = [n for n in display_names if n[0].isupper() and not n.isupper()]
        display_name = good_names[0] if good_names else display_names[0]

        # Collect all aliases
        all_aliases = set()
        for entry in entries:
            for alias in entry.get("aliases", []):
                if alias and alias.lower() != norm_name:
                    all_aliases.add(alias.strip())
            # Also add variant names from different sources
            entry_norm = normalize_name(entry["name"])
            if entry_norm != norm_name and entry["name"] not in all_aliases:
                all_aliases.add(entry["name"])

        # Collect sources
        sources = list(set(e["source"] for e in entries))

        # Determine category hint
        category = None
        for entry in entries:
            if "category_hint" in entry:
                category = entry["category_hint"]
                break

        # Merge raw text (keep from each source)
        source_texts = {}
        for entry in entries:
            src = entry["source"]
            if src not in source_texts:
                source_texts[src] = entry.get("raw_text", "")

        registry[norm_name] = {
            "name": display_name,
            "normalized": norm_name,
            "aliases": sorted(all_aliases),
            "category": category,
            "sources": sources,
            "source_count": len(sources),
            "source_texts": source_texts,
        }

    return registry


# ── Semantic Search Enhancement ───────────────────────────────────────────────


def enhance_with_semantic_search(registry: dict, vectordb_dir: Path) -> dict:
    """
    Use the vector DB to find additional mentions and context
    for each shape in the registry.
    """
    client = chromadb.PersistentClient(
        path=str(vectordb_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("pasta_knowledge")

    print(f"\n  Enhancing {len(registry)} shapes with semantic search...")

    for i, (norm_name, shape) in enumerate(registry.items()):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(registry)}] shapes enhanced")

        # Search for this shape across all sources
        query = f"{shape['name']} pasta shape"
        results = collection.query(
            query_texts=[query],
            n_results=10,
        )

        # Track which books mention this shape
        mentioning_books = set(shape["sources"])
        relevant_chunks = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Only keep reasonably relevant results
            if dist < 1.2:
                book_id = meta["book_id"]
                mentioning_books.add(book_id)
                relevant_chunks.append({
                    "book_id": book_id,
                    "book_title": meta["book_title"],
                    "section": meta["section_title"],
                    "distance": round(dist, 3),
                    "preview": doc[:200],
                })

        shape["mentioning_books"] = sorted(mentioning_books)
        shape["mention_count"] = len(mentioning_books)
        shape["semantic_matches"] = relevant_chunks[:5]  # Keep top 5

    return registry


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3: Pasta Shape Entity Extraction")
    print("=" * 60)

    all_shapes = []

    # Extract from each structured source
    extractors = {
        "zanini_encyclopedia": extract_from_zanini,
        "roddy_atoz": extract_from_roddy,
        "coastal_encyclopedia": extract_from_coastal,
        "louis_byhand": extract_from_louis,
    }

    for book_id, extractor in extractors.items():
        filepath = EXTRACTED_DIR / f"{book_id}.json"
        if filepath.exists():
            print(f"\n  Extracting from: {book_id}")
            shapes = extractor(filepath)
            all_shapes.extend(shapes)
            print(f"    → Found {len(shapes)} candidate shapes")
        else:
            print(f"  ⚠ Missing: {filepath}")

    print(f"\n  Total candidates: {len(all_shapes)}")

    # Build canonical registry
    print("\n  Building canonical registry (deduplicating)...")
    registry = build_canonical_registry(all_shapes)
    print(f"  → {len(registry)} unique pasta shapes")

    # Enhance with semantic search
    registry = enhance_with_semantic_search(registry, VECTORDB_DIR)

    # Sort by mention count (most referenced first)
    sorted_shapes = dict(
        sorted(registry.items(), key=lambda x: x[1]["mention_count"], reverse=True)
    )

    # Save individual shape files
    for norm_name, shape in sorted_shapes.items():
        safe_name = re.sub(r"[^a-z0-9_]", "_", norm_name.replace(" ", "_"))
        shape_path = ENTITIES_DIR / f"{safe_name}.json"
        with open(shape_path, "w", encoding="utf-8") as f:
            json.dump(shape, f, indent=2, ensure_ascii=False)

    # Save full registry
    registry_path = OUTPUT_DIR / "shape_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_shapes": len(sorted_shapes),
            "shapes": sorted_shapes,
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase 3 complete!")
    print(f"  Unique shapes:  {len(sorted_shapes)}")
    print(f"  Registry:       {registry_path}")
    print(f"  Shape files:    {ENTITIES_DIR}/")

    # Show top shapes by coverage
    print(f"\n  Top 20 shapes by source coverage:")
    for i, (name, shape) in enumerate(sorted_shapes.items()):
        if i >= 20:
            break
        books = shape["mention_count"]
        aliases = ", ".join(shape["aliases"][:3]) if shape["aliases"] else "—"
        print(f"    {i+1:2d}. {shape['name']:<25s} ({books} sources) aliases: {aliases}")


if __name__ == "__main__":
    main()
