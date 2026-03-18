#!/usr/bin/env python3
"""
Phase 6: Gather Source Material Per Shape
==========================================
For each pasta shape, queries the vector DB to collect ALL relevant
source text. Outputs one file per shape with the raw source passages,
organized by book. This becomes the input for content generation.
"""

import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings


BASE_DIR = Path(__file__).parent.parent
VECTORDB_DIR = BASE_DIR / "vectordb"
ASSEMBLED_DIR = BASE_DIR / "assembled"
SOURCES_OUT = BASE_DIR / "assembled" / "source_texts"


# Curated list of real pasta shapes with proper categories
CURATED_SHAPES = {
    # Hand-shaped
    "orecchiette": "hand-shaped",
    "cavatelli": "hand-shaped",
    "trofie": "hand-shaped",
    "busiate": "hand-shaped",
    "busiata": "hand-shaped",
    "malloreddus": "hand-shaped",
    "lorighittas": "hand-shaped",
    "lorighitta": "hand-shaped",
    "foglie d'ulivo": "hand-shaped",
    "strascinati": "hand-shaped",
    "capunti": "hand-shaped",
    "cicatelli": "hand-shaped",
    "pisarei": "hand-shaped",
    "corzetti": "hand-shaped",
    "croxetti": "hand-shaped",
    "garganelli": "hand-shaped",
    "gnocchetti": "hand-shaped",
    "fusilli": "hand-shaped",
    "strozzapreti": "hand-shaped",
    "casarecce": "hand-shaped",
    "gemelli": "hand-shaped",
    "cencioni": "hand-shaped",
    "passatelli": "hand-shaped",
    "fregula": "hand-shaped",
    "fregola": "hand-shaped",
    "filindeu": "hand-shaped",

    # Hand-cut
    "tagliatelle": "hand-cut",
    "fettuccine": "hand-cut",
    "pappardelle": "hand-cut",
    "tagliolini": "hand-cut",
    "tajarin": "hand-cut",
    "maltagliati": "hand-cut",
    "quadrucci": "hand-cut",
    "quadretti": "hand-cut",
    "pici": "hand-cut",
    "stringozzi": "hand-cut",
    "strangozzi": "hand-cut",
    "umbricelli": "hand-cut",
    "lagane": "hand-cut",
    "sagne": "hand-cut",
    "sagne ncannulate": "hand-cut",
    "mafalde": "hand-cut",
    "mafaldine": "hand-cut",
    "pizzoccheri": "hand-cut",
    "bavette": "hand-cut",
    "trenette": "hand-cut",
    "tonnarelli": "hand-cut",
    "spaghetti alla chitarra": "hand-cut",
    "chitarra": "hand-cut",
    "fazzoletti": "hand-cut",
    "farfalle": "hand-cut",
    "ciriole": "hand-cut",
    "bringoli": "hand-cut",
    "lombrichelli": "hand-cut",
    "scialatielli": "hand-cut",
    "troccoli": "hand-cut",
    "manfricoli": "hand-cut",
    "testaroli": "hand-cut",
    "reginette toscane": "hand-cut",

    # Filled
    "ravioli": "filled",
    "tortellini": "filled",
    "tortelloni": "filled",
    "tortelli": "filled",
    "tortelli di zucca": "filled",
    "tortelli maremmani": "filled",
    "tortelli romagnoli": "filled",
    "agnolotti": "filled",
    "agnolotti alessandrini": "filled",
    "cappelletti": "filled",
    "cappellacci": "filled",
    "cappellacci di zucca": "filled",
    "cappellacci dei briganti": "filled",
    "pansoti": "filled",
    "pansotti": "filled",
    "mezzelune": "filled",
    "culurgiones": "filled",
    "culingionis": "filled",
    "anolini": "filled",
    "cjalsons": "filled",
    "cjarsons": "filled",
    "casoncelli": "filled",
    "casonsei": "filled",
    "schlutzkrapfen": "filled",
    "ravioli alla genovese": "filled",
    "ravioli alla napoletana": "filled",
    "ravioli di ricotta": "filled",
    "marubini": "filled",
    "krafi": "filled",
    "tortello sulla lastra": "filled",
    "impanadas": "filled",

    # Extruded
    "spaghetti": "extruded",
    "linguine": "extruded",
    "bucatini": "extruded",
    "penne": "extruded",
    "rigatoni": "extruded",
    "paccheri": "extruded",
    "ziti": "extruded",
    "vermicelli": "extruded",
    "capellini": "extruded",
    "ditalini": "extruded",
    "calamarata": "extruded",
    "bigoli": "extruded",
    "candele": "extruded",
    "tubetti": "extruded",
    "tufoli": "extruded",
    "gramigna": "extruded",
    "maccheroni": "extruded",

    # Sheet/Baked
    "lasagne": "sheet",
    "lasagna": "sheet",
    "cannelloni": "sheet",
    "vincisgrassi": "sheet",

    # Dumpling
    "gnocchi": "dumpling",
    "gnocchi di patate": "dumpling",
    "gnocchi di semolino": "dumpling",
    "gnocchi ricci": "dumpling",
    "gnudi": "dumpling",
    "canederli": "dumpling",
    "malfatti": "dumpling",
    "spätzle": "dumpling",

    # Small/Soup
    "pastina": "small",
    "anelli": "small",
    "anelletti": "small",
    "stelline": "small",
    "orzo": "small",
}


def gather_source_text(collection, shape_name: str) -> list[dict]:
    """Query vector DB for all relevant text about a shape."""
    queries = [
        f"{shape_name} pasta",
        f"{shape_name} how to make shape dough",
        f"{shape_name} history origin region tradition",
        f"{shape_name} ingredients recipe",
    ]

    seen_ids = set()
    results = []

    for query in queries:
        res = collection.query(query_texts=[query], n_results=15)
        for doc, meta, dist, chunk_id in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["ids"][0],
        ):
            if chunk_id not in seen_ids and dist < 1.1:
                seen_ids.add(chunk_id)
                results.append({
                    "text": doc,
                    "book": meta["book_title"],
                    "author": meta["book_author"],
                    "section": meta["section_title"],
                    "distance": round(dist, 3),
                })

    # Sort by relevance
    results.sort(key=lambda x: x["distance"])
    return results


def main():
    SOURCES_OUT.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection("pasta_knowledge")

    # Load existing assembled data
    with open(ASSEMBLED_DIR / "pasta_database.json") as f:
        db = json.load(f)

    print(f"Gathering source texts for {len(CURATED_SHAPES)} curated shapes...\n")

    all_sources = {}

    for i, (name, category) in enumerate(CURATED_SHAPES.items()):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(CURATED_SHAPES)}]")

        sources = gather_source_text(collection, name)

        # Group by book
        by_book = {}
        for s in sources:
            book = s["book"]
            if book not in by_book:
                by_book[book] = []
            by_book[book].append(s["text"])

        slug = re.sub(r"[^a-z0-9]", "-", name.replace("'", "")).strip("-")
        slug = re.sub(r"-+", "-", slug)

        shape_data = {
            "name": name.title(),
            "slug": slug,
            "category": category,
            "source_count": len(by_book),
            "chunk_count": len(sources),
            "sources_by_book": by_book,
            "all_text": "\n\n---\n\n".join(s["text"] for s in sources[:20]),
        }

        all_sources[slug] = shape_data

        # Save individual source file
        out_path = SOURCES_OUT / f"{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(shape_data, f, indent=2, ensure_ascii=False)

    # Save manifest
    manifest = {
        "total_shapes": len(all_sources),
        "shapes": {k: {
            "name": v["name"],
            "slug": v["slug"],
            "category": v["category"],
            "source_count": v["source_count"],
            "chunk_count": v["chunk_count"],
        } for k, v in all_sources.items()},
    }
    with open(SOURCES_OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(all_sources)} shapes with source texts in {SOURCES_OUT}")

    # Stats
    total_chunks = sum(v["chunk_count"] for v in all_sources.values())
    avg_chunks = total_chunks / len(all_sources)
    print(f"Total chunks gathered: {total_chunks}")
    print(f"Average chunks per shape: {avg_chunks:.1f}")


if __name__ == "__main__":
    main()
