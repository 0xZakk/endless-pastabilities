#!/usr/bin/env python3
"""
Phase 5: Knowledge Assembly — Per-Shape Profiles
==================================================
Assembles comprehensive profiles for each pasta shape using:
1. Semantic search (ChromaDB) to gather relevant passages
2. Pattern matching to extract structured fields (region, instructions, etc.)
3. Heuristic ranking to select the best content from multiple sources
4. Cross-referencing with dough registry

All local — no API keys, no external calls.

Output: One JSON file per pasta shape in assembled/shapes/
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import chromadb
from chromadb.config import Settings


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
ENTITIES_DIR = BASE_DIR / "entities"
VECTORDB_DIR = BASE_DIR / "vectordb"
ASSEMBLED_DIR = BASE_DIR / "assembled"
SHAPES_OUT = ASSEMBLED_DIR / "shapes"
DOUGHS_OUT = ASSEMBLED_DIR / "doughs"

# Shapes that are actually recipes or non-shape entries to filter out
BLOCKLIST_PATTERNS = [
    r"^(lobster|shrimp|crab|chicken|duck|lamb|pork|beef|veal|sausage|mushroom|pumpkin|sweet potato|potato|ricotta|butternut)\s",
    r"^(weeknight|quick|easy|simple|classic|traditional|homemade|basic)\s",
    r"\s(soup|salad|bake|gratin|pie|cake|broth|stock|stew|curry|laksa)$",
    r"^(how to|the art|introduction|equipment|ingredients|techniques|basics)",
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"\s(with|and|in a|for)\s",  # Recipe titles like "Penne with Sausage"
]

# Core shapes we definitely want (even if evidence is thin)
PRIORITY_SHAPES = {
    "spaghetti", "linguine", "fettuccine", "tagliatelle", "pappardelle",
    "bucatini", "vermicelli", "capellini", "bigoli", "pici",
    "tonnarelli", "tagliolini", "trenette", "bavette",
    "penne", "rigatoni", "fusilli", "farfalle", "conchiglie",
    "orecchiette", "cavatelli", "gnocchetti", "maccheroni",
    "paccheri", "calamarata", "ziti", "casarecce", "gemelli",
    "strozzapreti", "trofie", "busiate", "malloreddus",
    "lorighittas", "garganelli",
    "ravioli", "tortellini", "tortelloni", "agnolotti",
    "cappelletti", "pansotti", "mezzelune", "culurgiones",
    "anolini", "casoncelli", "cjarsons", "schlutzkrapfen",
    "lasagne", "lasagna", "cannelloni", "manicotti",
    "orzo", "stelline", "ditalini", "pastina",
    "fregula", "fregola",
    "gnocchi", "gnudi", "canederli", "passatelli", "pisarei",
    "foglie d'ulivo", "strascinati", "capunti", "cicatelli",
    "sagne", "lagane", "mafalde", "mafaldine",
    "corzetti", "croxetti", "pizzoccheri",
    "maltagliati", "quadrucci", "stringozzi", "strangozzi",
    "umbricelli", "cencioni", "testaroli",
    "chitarra", "spaghetti alla chitarra",
    "tajarin", "tacconi",
}

# Region patterns
ITALIAN_REGIONS = [
    "Puglia", "Apulia", "Sicily", "Sicilia", "Sardinia", "Sardegna",
    "Campania", "Calabria", "Basilicata", "Lazio", "Latium",
    "Tuscany", "Toscana", "Emilia-Romagna", "Emilia Romagna",
    "Lombardy", "Lombardia", "Veneto", "Piedmont", "Piemonte",
    "Liguria", "Umbria", "Marche", "Abruzzo", "Molise",
    "Trentino", "Alto Adige", "Friuli", "Venezia Giulia",
    "Valle d'Aosta", "Naples", "Napoli", "Bologna", "Rome", "Roma",
    "Genoa", "Genova", "Palermo", "Bari", "Lecce", "Otranto",
]

# Dough-to-shape mapping heuristics
DOUGH_HINTS = {
    "semolina-water": [
        "orecchiette", "cavatelli", "busiate", "malloreddus",
        "lorighittas", "foglie d'ulivo", "strascinati", "capunti",
        "cicatelli", "trofie", "fregula", "fregola", "pisarei",
        "sagne", "lagane", "fusilli", "maccheroni al ferretto",
    ],
    "egg-dough": [
        "tagliatelle", "fettuccine", "pappardelle", "tagliolini",
        "garganelli", "tortellini", "tortelloni", "agnolotti",
        "cappelletti", "ravioli", "lasagne", "lasagna", "cannelloni",
        "maltagliati", "quadrucci", "tonnarelli", "tajarin",
        "corzetti", "farfalle", "anolini", "casoncelli",
    ],
    "egg-yolk-rich": [
        "tajarin", "tagliolini",
    ],
    "flour-water": [
        "pici", "umbricelli", "stringozzi", "strangozzi",
        "testaroli", "cencioni",
    ],
    "semolina-water": [
        "orecchiette", "cavatelli", "busiate", "strozzapreti",
    ],
    "buckwheat": [
        "pizzoccheri",
    ],
    "saffron": [
        "malloreddus", "lorighittas", "fregula",
    ],
}

# Category classification heuristics
CATEGORY_HINTS = {
    "filled": [
        "ravioli", "tortellini", "tortelloni", "agnolotti", "cappelletti",
        "pansotti", "mezzelune", "culurgiones", "anolini", "cjarsons",
        "schlutzkrapfen", "casoncelli", "cannelloni", "manicotti",
    ],
    "hand-shaped": [
        "orecchiette", "cavatelli", "trofie", "busiate", "gnocchetti",
        "malloreddus", "lorighittas", "foglie d'ulivo", "strascinati",
        "capunti", "cicatelli", "pisarei", "corzetti", "croxetti",
        "garganelli", "fusilli",
    ],
    "hand-cut": [
        "tagliatelle", "fettuccine", "pappardelle", "tagliolini",
        "maltagliati", "quadrucci", "tajarin", "pici", "stringozzi",
        "strangozzi", "umbricelli", "lagane", "sagne", "mafalde",
        "pizzoccheri", "bavette", "trenette", "chitarra",
        "spaghetti alla chitarra", "tonnarelli",
    ],
    "extruded": [
        "spaghetti", "linguine", "bucatini", "penne", "rigatoni",
        "paccheri", "ziti", "vermicelli", "capellini", "ditalini",
        "fusilli", "casarecce", "gemelli", "calamarata", "bigoli",
    ],
    "sheet": [
        "lasagne", "lasagna", "cannelloni", "manicotti", "testaroli",
    ],
    "dumpling": [
        "gnocchi", "gnudi", "canederli", "passatelli",
    ],
    "small/soup": [
        "orzo", "stelline", "ditalini", "pastina", "fregula", "fregola",
        "acini di pepe", "anellini",
    ],
}


def is_blocked(name: str) -> bool:
    """Check if a name matches the blocklist (recipes, not shapes)."""
    for pattern in BLOCKLIST_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


def classify_category(name: str) -> str | None:
    """Classify a shape into a category."""
    norm = name.lower().strip()
    for category, shapes in CATEGORY_HINTS.items():
        if norm in shapes:
            return category
    return None


def get_dough_for_shape(name: str) -> list[str]:
    """Get likely dough types for a shape."""
    norm = name.lower().strip()
    doughs = []
    for dough_slug, shapes in DOUGH_HINTS.items():
        if norm in shapes:
            doughs.append(dough_slug)
    if not doughs:
        # Default heuristic: if it's filled or hand-cut, likely egg dough
        cat = classify_category(name)
        if cat in ("filled", "hand-cut", "sheet"):
            doughs = ["egg-dough"]
        elif cat in ("hand-shaped",):
            doughs = ["semolina-water"]
        elif cat == "dumpling":
            doughs = ["flour-water"]
    return doughs


# ── Text Analysis Extractors ─────────────────────────────────────────────────


def extract_regions(text: str) -> list[str]:
    """Find Italian region mentions in text."""
    found = []
    for region in ITALIAN_REGIONS:
        if re.search(r"\b" + re.escape(region) + r"\b", text, re.IGNORECASE):
            found.append(region)
    return list(set(found))


def extract_instructions(text: str, shape_name: str) -> list[str]:
    """
    Extract step-by-step pasta shaping instructions from text.
    Looks for imperative verb patterns and numbered/sequential steps.
    """
    instructions = []

    # Pattern 1: Numbered steps ("1. Roll the dough...")
    numbered = re.findall(
        r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\n\n|$)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if numbered and len(numbered) >= 2:
        return [step.strip().replace("\n", " ") for step in numbered if len(step.strip()) > 20]

    # Pattern 2: Imperative sentences about shaping
    shaping_verbs = [
        "roll", "cut", "shape", "press", "pinch", "fold", "twist",
        "flatten", "stretch", "pull", "drag", "wrap", "curl", "crimp",
        "seal", "place", "lay", "dust", "flour", "knead", "divide",
        "form", "make", "take", "using", "with your", "use a",
    ]

    sentences = re.split(r"(?<=[.!])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        sent_lower = sent.lower()
        # Check if sentence is instructional
        if any(sent_lower.startswith(verb) or f" {verb} " in sent_lower
               for verb in shaping_verbs):
            # Extra check: mentions the shape or generic pasta terms
            if (shape_name.lower() in sent_lower
                    or "dough" in sent_lower
                    or "pasta" in sent_lower
                    or "piece" in sent_lower
                    or "sheet" in sent_lower):
                instructions.append(sent.replace("\n", " "))

    return instructions[:10]  # Cap at 10 steps


def extract_narrative(text: str, shape_name: str) -> list[str]:
    """
    Extract historical/narrative passages about a pasta shape.
    Looks for sentences containing historical markers, region references,
    or storytelling language.
    """
    narrative_markers = [
        "tradition", "traditional", "historically", "history",
        "origin", "originated", "century", "centuries",
        "ancient", "medieval", "dating", "dates back",
        "legend", "story", "said to", "believed",
        "region", "village", "town", "city",
        "festival", "celebration", "feast", "holiday",
        "grandmother", "nonna", "family", "generation",
        "local", "typical", "characteristic", "famous",
        "name comes from", "named after", "derives from",
        "dialect", "meaning", "means", "called",
        "served", "eaten", "prepared", "made for",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    narrative = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 500:
            continue
        sent_lower = sent.lower()
        # Score by narrative markers
        score = sum(1 for marker in narrative_markers if marker in sent_lower)
        if score >= 2:
            narrative.append(sent.replace("\n", " "))

    return narrative[:8]


def extract_description(text: str, shape_name: str) -> str | None:
    """
    Extract a short description of the pasta shape.
    Looks for definitional sentences.
    """
    desc_patterns = [
        rf"{re.escape(shape_name)}\s+(?:is|are)\s+(.+?\.)",
        rf"{re.escape(shape_name)},?\s+(?:a|an)\s+(.+?\.)",
        r"(?:small|large|long|short|flat|round|thin|thick|tube|ribbon|sheet)\s+(?:pasta|noodle).+?\.",
    ]

    for pattern in desc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            desc = match.group(0).strip()
            if 20 < len(desc) < 300:
                return desc

    # Fallback: first substantial sentence mentioning the shape
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if shape_name.lower() in sent.lower() and 30 < len(sent) < 250:
            return sent.strip()

    return None


# ── Main Assembly Pipeline ────────────────────────────────────────────────────


def main():
    SHAPES_OUT.mkdir(parents=True, exist_ok=True)
    DOUGHS_OUT.mkdir(parents=True, exist_ok=True)

    print("Phase 5: Knowledge Assembly")
    print("=" * 60)

    # Load shape registry
    registry_path = ENTITIES_DIR / "shape_registry.json"
    with open(registry_path) as f:
        shape_registry = json.load(f)["shapes"]

    # Load dough registry
    dough_registry_path = ENTITIES_DIR / "dough_registry.json"
    with open(dough_registry_path) as f:
        dough_registry = json.load(f)["doughs"]

    # Connect to vector DB
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("pasta_knowledge")

    # Filter shapes: remove recipes, keep real pasta shapes
    print("\n  Filtering shape registry...")
    valid_shapes = {}
    blocked_count = 0
    for norm_name, shape in shape_registry.items():
        if is_blocked(shape["name"]):
            blocked_count += 1
            continue
        # Keep if it's a priority shape OR has multiple sources OR passes heuristics
        if (norm_name in PRIORITY_SHAPES
                or shape.get("mention_count", 0) >= 3
                or is_likely_shape(norm_name, shape)):
            valid_shapes[norm_name] = shape

    print(f"  → {len(valid_shapes)} shapes (filtered {blocked_count} non-shapes)")

    # Assemble profiles
    print(f"\n  Assembling {len(valid_shapes)} shape profiles...")
    assembled = {}

    for i, (norm_name, shape) in enumerate(valid_shapes.items()):
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(valid_shapes)}] assembled")

        # Semantic search for this shape
        queries = [
            f"{shape['name']} pasta shape description",
            f"how to make {shape['name']} pasta",
            f"{shape['name']} history origin tradition",
        ]

        all_chunks_text = []
        source_books = set()

        for query in queries:
            results = collection.query(query_texts=[query], n_results=8)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if dist < 1.3:
                    all_chunks_text.append(doc)
                    source_books.add(meta["book_title"])

        # Also include source texts from entity extraction
        for src, text in shape.get("source_texts", {}).items():
            all_chunks_text.append(text)

        # Combine all evidence text
        combined_text = "\n\n".join(all_chunks_text)

        # Extract structured fields
        description = extract_description(combined_text, shape["name"])
        regions = extract_regions(combined_text)
        instructions = extract_instructions(combined_text, shape["name"])
        narrative = extract_narrative(combined_text, shape["name"])
        category = shape.get("category") or classify_category(norm_name)
        doughs = get_dough_for_shape(norm_name)

        # Build the assembled profile
        slug = re.sub(r"[^a-z0-9]", "-", norm_name.replace("'", "")).strip("-")
        slug = re.sub(r"-+", "-", slug)

        profile = {
            "name": shape["name"],
            "slug": slug,
            "normalized": norm_name,
            "aliases": shape.get("aliases", []),
            "category": category,
            "regions": regions,
            "dough_recipes": doughs,
            "description": description,
            "instructions": instructions,
            "narrative": narrative,
            "source_books": sorted(source_books),
            "source_count": len(source_books),
        }

        assembled[norm_name] = profile

        # Save individual file
        shape_path = SHAPES_OUT / f"{slug}.json"
        with open(shape_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    # Copy dough profiles to assembled
    for slug, dough in dough_registry.items():
        dough_path = DOUGHS_OUT / f"{slug}.json"
        with open(dough_path, "w", encoding="utf-8") as f:
            json.dump(dough, f, indent=2, ensure_ascii=False)

    # Save master assembled file
    master_path = ASSEMBLED_DIR / "pasta_database.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_shapes": len(assembled),
            "total_doughs": len(dough_registry),
            "shapes": assembled,
            "doughs": dough_registry,
        }, f, indent=2, ensure_ascii=False)

    # Stats
    shapes_with_instructions = sum(
        1 for s in assembled.values() if s["instructions"]
    )
    shapes_with_narrative = sum(
        1 for s in assembled.values() if s["narrative"]
    )
    shapes_with_description = sum(
        1 for s in assembled.values() if s["description"]
    )
    shapes_with_region = sum(
        1 for s in assembled.values() if s["regions"]
    )

    print(f"\n{'='*60}")
    print(f"Phase 5 complete!")
    print(f"  Total shapes assembled:  {len(assembled)}")
    print(f"  With description:        {shapes_with_description}")
    print(f"  With instructions:       {shapes_with_instructions}")
    print(f"  With narrative:          {shapes_with_narrative}")
    print(f"  With region:             {shapes_with_region}")
    print(f"  Dough recipes:           {len(dough_registry)}")
    print(f"\n  Output: {master_path}")

    # Show a sample
    print(f"\n  Sample profiles:")
    samples = ["orecchiette", "tagliatelle", "tortellini", "pici", "malloreddus"]
    for sample in samples:
        if sample in assembled:
            s = assembled[sample]
            print(f"\n  ── {s['name']} ──")
            print(f"     Category: {s['category']}")
            print(f"     Regions: {', '.join(s['regions'][:3]) if s['regions'] else '—'}")
            print(f"     Dough: {', '.join(s['dough_recipes']) if s['dough_recipes'] else '—'}")
            print(f"     Description: {(s['description'] or '—')[:100]}...")
            print(f"     Instructions: {len(s['instructions'])} steps")
            print(f"     Narrative: {len(s['narrative'])} passages")
            print(f"     Sources: {s['source_count']} books")


def is_likely_shape(norm_name: str, shape: dict) -> bool:
    """Additional heuristic to determine if something is a real pasta shape."""
    # Check source texts for pasta-related content
    for text in shape.get("source_texts", {}).values():
        text_lower = text.lower()
        indicators = ["pasta", "dough", "flour", "shape", "boil", "sauce", "semolina"]
        if sum(1 for ind in indicators if ind in text_lower) >= 3:
            return True
    # Italian-sounding endings
    italian_endings = [
        "etti", "ette", "ini", "ine", "oni", "one", "elle", "elli",
        "otti", "otte", "ali", "ale", "oli", "ola",
    ]
    if any(norm_name.endswith(e) for e in italian_endings):
        return True
    return False


if __name__ == "__main__":
    main()
