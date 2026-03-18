#!/usr/bin/env python3
"""
Phase 2: Text Chunking & Embedding
====================================
Takes extracted text from Phase 1 and:
1. Chunks it into semantically meaningful segments
2. Generates embeddings using sentence-transformers
3. Stores everything in ChromaDB for semantic search

Chunking strategy:
- Split on paragraph boundaries
- Target chunk size: ~500-1000 chars (good for embedding models)
- Overlap: 100 chars between chunks for context continuity
- Preserve metadata: source book, section, page numbers
"""

import json
import re
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings


# ── Configuration ────────────────────────────────────────────────────────────

EXTRACTED_DIR = Path(__file__).parent.parent / "extracted"
CHUNKS_DIR = Path(__file__).parent.parent / "chunks"
VECTORDB_DIR = Path(__file__).parent.parent / "vectordb"

CHUNK_SIZE = 800       # Target characters per chunk
CHUNK_OVERLAP = 150    # Overlap between chunks
MIN_CHUNK_SIZE = 100   # Minimum chunk size to keep

COLLECTION_NAME = "pasta_knowledge"


# ── Chunking ─────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Clean extracted text of common artifacts."""
    # Remove excessive whitespace
    text = re.sub(r" {3,}", "  ", text)
    # Remove page number artifacts
    text = re.sub(r"\n\d{1,3}\n", "\n", text)
    # Normalize line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks at paragraph boundaries.
    Tries to break at natural paragraph boundaries (\n\n) first,
    then falls back to sentence boundaries, then hard breaks.
    """
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    # Split into paragraphs first
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from end of previous
                if overlap > 0 and len(current_chunk) > overlap:
                    # Find a sentence boundary near the overlap point
                    overlap_text = current_chunk[-overlap:]
                    sent_break = overlap_text.find(". ")
                    if sent_break > 0:
                        overlap_text = overlap_text[sent_break + 2:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Single paragraph exceeds chunk size — split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 > chunk_size:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                    else:
                        sub_chunk = (sub_chunk + " " + sent).strip()
                current_chunk = sub_chunk
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()

    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())

    return chunks


def process_book(filepath: Path) -> list[dict]:
    """Process a single extracted book JSON into chunks with metadata."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    book_id = metadata["short_id"]
    book_title = metadata["title"]
    book_author = metadata["author"]

    all_chunks = []
    sections = data.get("sections", [])

    for section_idx, section in enumerate(sections):
        section_title = section.get("title") or f"Section {section_idx + 1}"
        section_text = section.get("text", "")

        if not section_text or len(section_text.strip()) < MIN_CHUNK_SIZE:
            continue

        chunks = chunk_text(section_text)

        for chunk_idx, chunk_text_content in enumerate(chunks):
            chunk_id = f"{book_id}_s{section_idx:03d}_c{chunk_idx:03d}"

            chunk_meta = {
                "book_id": book_id,
                "book_title": book_title,
                "book_author": book_author,
                "section_title": section_title,
                "section_index": section_idx,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_text_content),
            }

            # Add page numbers if available (PDFs)
            if "pages" in section:
                chunk_meta["pages"] = str(section["pages"])
            if "chapter_id" in section:
                chunk_meta["chapter_id"] = section["chapter_id"]

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text_content,
                "metadata": chunk_meta,
            })

    return all_chunks


# ── Embedding & Storage ──────────────────────────────────────────────────────


def build_vector_store(all_chunks: list[dict]):
    """
    Store chunks in ChromaDB with embeddings.
    ChromaDB will auto-generate embeddings using its default model
    (all-MiniLM-L6-v2 via sentence-transformers).
    """
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if it exists (fresh build)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Pasta knowledge base from 8 source books"},
    )

    # ChromaDB has a batch limit, so we insert in batches
    BATCH_SIZE = 100
    total = len(all_chunks)

    print(f"\n  Embedding and storing {total:,} chunks in ChromaDB...")

    for i in range(0, total, BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        progress = min(i + BATCH_SIZE, total)
        print(f"    [{progress:,}/{total:,}] chunks embedded", end="\r")

    print(f"\n  ✓ Vector store built: {collection.count():,} chunks indexed")
    return collection


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Find all extracted book JSONs
    extracted_files = sorted(EXTRACTED_DIR.glob("*.json"))
    extracted_files = [f for f in extracted_files if f.name != "manifest.json"]

    print(f"Found {len(extracted_files)} extracted books\n")

    all_chunks = []
    book_stats = []

    for filepath in extracted_files:
        print(f"  Chunking: {filepath.stem}...")
        chunks = process_book(filepath)
        all_chunks.extend(chunks)

        stats = {
            "book_id": filepath.stem,
            "chunk_count": len(chunks),
            "total_chars": sum(c["metadata"]["char_count"] for c in chunks),
            "avg_chunk_size": (
                sum(c["metadata"]["char_count"] for c in chunks) // len(chunks)
                if chunks else 0
            ),
        }
        book_stats.append(stats)
        print(f"    → {len(chunks)} chunks (avg {stats['avg_chunk_size']} chars)")

    # Save chunks as JSON for inspection
    chunks_output = CHUNKS_DIR / "all_chunks.json"
    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "book_stats": book_stats,
            "chunks": all_chunks,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Chunks saved to: {chunks_output}")

    # Build vector store
    collection = build_vector_store(all_chunks)

    # Test a sample query
    print("\n  Testing semantic search...")
    test_queries = [
        "orecchiette from Puglia",
        "egg pasta dough recipe",
        "history of pappardelle",
    ]
    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"\n  Query: \"{query}\"")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            print(f"    [{dist:.3f}] {meta['book_title']} / {meta['section_title']}")
            print(f"           {doc[:100]}...")

    print(f"\n{'='*60}")
    print(f"Phase 2 complete!")
    print(f"  Total chunks: {len(all_chunks):,}")
    print(f"  Vector DB:    {VECTORDB_DIR}")
    print(f"  Collection:   {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
