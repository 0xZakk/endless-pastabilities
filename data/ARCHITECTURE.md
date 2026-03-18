# Endless Pastabilities — Data Pipeline Architecture

## Vision

A comprehensive, beautifully designed encyclopedia of every pasta shape at endlesspastabilities.com. Each shape gets its own page with a watercolor illustration, dough recipe, step-by-step shaping instructions, and historical narrative — all sourced from 10 authoritative pasta books.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  SOURCES: 10 books (3 PDFs, 7 EPUBs)                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 1: TEXT EXTRACTION                                │
│  PyMuPDF/pdfplumber (PDFs) + ebooklib (EPUBs)           │
│  → Structured JSON with metadata per source             │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 2: CHUNKING & EMBEDDING                           │
│  Semantic chunking → sentence-transformers embeddings    │
│  → ChromaDB vector store with full-text search           │
└────────┬─────────────────────────────┬──────────────────┘
         ▼                             ▼
┌────────────────────┐   ┌─────────────────────────┐
│ Phase 3: EXTRACT   │   │ Phase 4: EXTRACT        │
│ PASTA SHAPES       │   │ DOUGH RECIPES           │
│ NER + LLM-based    │   │ Recipe parsing + norm.  │
│ → Shape registry   │   │ → Canonical dough set   │
└────────┬───────┬───┘   └────────────┬────────────┘
         │       │                     │
         │       ▼                     │
         │  ┌────────────────────┐     │
         │  │ Phase 6: IMAGE GEN │     │
         │  │ Watercolor/botanic │     │
         │  │ style per shape    │     │
         │  └────────┬───────────┘     │
         ▼           │                 ▼
┌────────────────────────────────────────────────────────┐
│  Phase 5: KNOWLEDGE ASSEMBLY (RAG)                      │
│  Semantic search + LLM synthesis per shape              │
│  → Instructions, history, narrative, citations          │
└──────────────────────┬─────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 7: WEBSITE BUILD                                  │
│  Static site (Astro/Next.js) — minimal, elevated design │
│  Typography-forward, white space, watercolor images      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 8: HACKATHON SLIDES                               │
│  Architecture walkthrough + demo                         │
└──────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| PDF extraction | PyMuPDF + pdfplumber | Best combo for text + table extraction |
| EPUB extraction | ebooklib + BeautifulSoup | Standard EPUB parsing with HTML cleanup |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, high-quality local embeddings |
| Vector DB | ChromaDB | Lightweight, file-based, Python-native |
| NLP/NER | spaCy + custom patterns + LLM extraction | Hybrid approach for best coverage |
| Knowledge assembly | RAG with Claude API | Synthesize multi-source info per shape |
| Image generation | DALL-E 3 / Stable Diffusion | Watercolor botanical-style illustrations |
| Website | Astro or Next.js (static) | Fast static site generation |
| Styling | Tailwind CSS + custom typography | Minimal, elevated design system |

## Sources

### PDFs (3)
1. **Encyclopedia of Pasta** — Oretta Zanini De Vita, Maureen B. Fant
2. **Pasta By Hand** — Jenn Louis, Ed Anderson, Mario Batali
3. **Pasta and Noodles: A Global History** — Kantha Shelke

### EPUBs (7)
4. **Pasta** — Missy Robbins, Talia Baiocchi
5. **Pasta Masterclass** — Mateo Zielonka
6. **Mastering Pasta** — Marc Vetri, David Joachim
7. **Pasta** — Theo Randall
8. **Pasta Grannies** — Vicky Bennison
9. **The Encyclopedia of Pasta** (Over 350 Recipes) — The Coastal Kitchen
10. **An A-Z of Pasta** — Rachel Roddy

## Data Model (Target)

### Pasta Shape
```yaml
name: "Orecchiette"
slug: "orecchiette"
aliases: ["recchietelle", "strascinati"]
category: "hand-shaped"  # hand-cut | hand-shaped | filled | extruded
region: "Puglia"
dough_recipes: ["semolina-water"]
description: "Small ear-shaped pasta..."
history: "Originating in Puglia..."
instructions:
  - step: 1
    text: "Roll dough into a rope..."
sources:
  - book: "Encyclopedia of Pasta"
    author: "Zanini De Vita"
    pages: [xxx]
```

### Dough Recipe
```yaml
name: "Semolina & Water Dough"
slug: "semolina-water"
ingredients:
  - name: "semolina flour"
    amount: "300g"
  - name: "warm water"
    amount: "150ml"
method:
  - "Mound the flour on a work surface..."
used_for: ["orecchiette", "cavatelli", "foglie d'ulivo"]
sources:
  - book: "Pasta By Hand"
    author: "Jenn Louis"
```

## Directory Structure

```
data/
├── ARCHITECTURE.md          # This file
├── README.md
├── pasta.yaml               # Shape categories
├── sources/                 # Raw PDFs and EPUBs
├── extracted/               # Phase 1 output: raw text JSON per book
├── chunks/                  # Phase 2 output: chunked text
├── vectordb/                # Phase 2 output: ChromaDB store
├── entities/                # Phase 3-4 output: extracted entities
│   ├── shapes/              # Per-shape JSON files
│   └── doughs/              # Per-dough JSON files
├── assembled/               # Phase 5 output: final assembled data
│   ├── shapes/              # Complete shape profiles
│   └── doughs/              # Complete dough recipes
├── images/                  # Phase 6 output: generated illustrations
└── scripts/                 # Python pipeline scripts
    ├── requirements.txt
    ├── 01_extract.py        # Text extraction
    ├── 02_chunk_embed.py    # Chunking & embedding
    ├── 03_extract_shapes.py # Shape entity extraction
    ├── 04_extract_doughs.py # Dough recipe extraction
    └── 05_assemble.py       # Knowledge assembly
```
