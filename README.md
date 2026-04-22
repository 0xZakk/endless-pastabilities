# Endless Pastabilities

I bought 8 pasta books, extracted 4.5 million characters of text from them, and built a data pipeline to turn all of that into a browsable encyclopedia of 123 pasta shapes.

It's wildly overengineered on purpose — NLP entity extraction, vector embeddings, semantic search, multi-source knowledge assembly — because the whole point was to see how far you could push a RAG pipeline on a weird, constrained domain. Turns out pretty far.

Every description, shaping instruction, and historical note on the site comes directly from the source books. Nothing fabricated.

## How it works

Six phases, each one feeding the next:

```
8 Books (3 PDFs, 5 EPUBs)
  → Extract raw text (PyMuPDF + pdfplumber + ebooklib)
    → Chunk and embed (4,602 vectors in ChromaDB)
      → Pull out pasta shapes (983 candidates → 123 curated)
        → Pull out dough recipes (10 canonical types)
          → Assemble knowledge per shape (semantic search across all sources)
            → Generate static site (Astro → 136 pages, 512ms build)
```

The fun part is phase 5 — for each of the 123 shapes, the pipeline runs semantic search across all 8 books, gathers every relevant passage, and synthesizes a description, shaping instructions, and history from the combined evidence. Orecchiette pulls from 7 different books to build one coherent entry.

## The numbers

- **4,498,875** characters extracted from 8 books
- **4,602** semantic chunks embedded in ChromaDB
- **983** candidate pasta shapes identified → **123** curated with full content
- **10** canonical dough types
- **136** static pages, **512ms** build time, **0** client-side JavaScript

## Tech stack

**Pipeline (Python):** PyMuPDF, pdfplumber, ebooklib, BeautifulSoup, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, custom entity extractors

**Site:** Astro, scoped CSS, system serif typography, zero JS

The pipeline is 7 scripts that run in sequence — `01_extract.py` through `07_merge_generated.py`. Each one does exactly what it sounds like.

## Project structure

```
site/           Astro frontend
data/
├── sources/    Raw books (gitignored)
├── extracted/  Text extraction output
├── chunks/     Chunked + embedded text
├── vectordb/   ChromaDB store
├── entities/   Extracted shapes and doughs
├── assembled/  Final assembled knowledge
├── images/     Generated illustrations
└── scripts/    The 7-phase pipeline
```

There's a full technical deep-dive into each phase — extraction strategies, chunking decisions, the entity extraction funnel, how knowledge assembly works — in the [How We Built This](data/ARCHITECTURE.md) doc.

## License

MIT
