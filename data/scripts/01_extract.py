#!/usr/bin/env python3
"""
Phase 1: Text Extraction Pipeline
==================================
Extracts raw text from PDF and EPUB source books about pasta.
Outputs structured JSON per book with metadata preservation.

Each output file contains:
- Book metadata (title, author, format)
- Sections/chapters with text content
- Page numbers (PDFs) or chapter IDs (EPUBs)
"""

import json
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

import pymupdf  # PyMuPDF
import pdfplumber
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


# ── Configuration ────────────────────────────────────────────────────────────

SOURCES_DIR = Path(__file__).parent.parent / "sources"
OUTPUT_DIR = Path(__file__).parent.parent / "extracted"

# Book metadata registry — maps filename patterns to clean metadata
BOOK_REGISTRY = {
    "Encyclopedia of Pasta (Oretta Zanini De Vita": {
        "title": "Encyclopedia of Pasta",
        "author": "Oretta Zanini De Vita",
        "translator": "Maureen B. Fant",
        "year": 2009,
        "short_id": "zanini_encyclopedia",
    },
    "Pasta (Missy Robbins": {
        "title": "Pasta",
        "author": "Missy Robbins & Talia Baiocchi",
        "year": 2021,
        "short_id": "robbins_pasta",
    },
    "Pasta Masterclass": {
        "title": "Pasta Masterclass",
        "author": "Mateo Zielonka",
        "year": 2023,
        "short_id": "zielonka_masterclass",
    },
    "Mastering Pasta": {
        "title": "Mastering Pasta",
        "author": "Marc Vetri & David Joachim",
        "year": 2015,
        "short_id": "vetri_mastering",
    },
    "Pasta (Theo Randall": {
        "title": "Pasta",
        "author": "Theo Randall",
        "year": 2013,
        "short_id": "randall_pasta",
    },
    "Pasta By Hand": {
        "title": "Pasta By Hand",
        "author": "Jenn Louis",
        "year": 2015,
        "short_id": "louis_byhand",
    },
    "Pasta and Noodles": {
        "title": "Pasta and Noodles: A Global History",
        "author": "Kantha Shelke",
        "year": 2016,
        "short_id": "shelke_history",
    },
    "Pasta Grannies": {
        "title": "Pasta Grannies",
        "author": "Vicky Bennison",
        "year": 2019,
        "short_id": "bennison_grannies",
    },
    "The Encyclopedia of Pasta": {
        "title": "The Encyclopedia of Pasta: Over 350 Recipes",
        "author": "The Coastal Kitchen",
        "year": 2023,
        "short_id": "coastal_encyclopedia",
    },
    "A-Z of Pasta": {
        "title": "An A-Z of Pasta",
        "author": "Rachel Roddy",
        "year": 2021,
        "short_id": "roddy_atoz",
    },
}


def match_book_metadata(filename: str) -> dict:
    """Match a filename to its book metadata from the registry."""
    for pattern, metadata in BOOK_REGISTRY.items():
        if pattern in filename:
            return metadata
    return {
        "title": filename,
        "author": "Unknown",
        "short_id": hashlib.md5(filename.encode()).hexdigest()[:12],
    }


# ── PDF Extraction ───────────────────────────────────────────────────────────


def extract_pdf_pymupdf(filepath: Path) -> list[dict]:
    """Extract text from PDF using PyMuPDF. Returns list of page dicts."""
    doc = pymupdf.open(str(filepath))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip(),
                "char_count": len(text.strip()),
            })
    doc.close()
    return pages


def extract_pdf_pdfplumber(filepath: Path) -> list[dict]:
    """Extract text from PDF using pdfplumber (better for tables). Returns list of page dicts."""
    pages = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            if text.strip() or tables:
                page_data = {
                    "page_number": page_num + 1,
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                }
                if tables:
                    page_data["tables"] = [
                        [row for row in table if any(cell for cell in row)]
                        for table in tables
                    ]
                pages.append(page_data)
    return pages


def extract_pdf(filepath: Path) -> dict:
    """
    Extract text from a PDF using both PyMuPDF and pdfplumber.
    Uses PyMuPDF as primary (better general text), pdfplumber for tables.
    """
    metadata = match_book_metadata(filepath.name)
    print(f"  Extracting PDF: {metadata['title']}")

    # Primary extraction with PyMuPDF
    pymupdf_pages = extract_pdf_pymupdf(filepath)

    # Secondary extraction with pdfplumber for tables
    plumber_pages = extract_pdf_pdfplumber(filepath)

    # Build table index from pdfplumber
    tables_by_page = {}
    for page in plumber_pages:
        if "tables" in page:
            tables_by_page[page["page_number"]] = page["tables"]

    # Merge: use PyMuPDF text + pdfplumber tables
    sections = []
    current_section = {"title": None, "pages": [], "text_parts": []}

    for page in pymupdf_pages:
        page_data = {
            "page_number": page["page_number"],
            "text": page["text"],
            "char_count": page["char_count"],
        }
        if page["page_number"] in tables_by_page:
            page_data["tables"] = tables_by_page[page["page_number"]]

        # Simple heuristic: detect section breaks by short lines in ALL CAPS
        lines = page["text"].split("\n")
        for line in lines[:5]:  # Check first few lines
            clean = line.strip()
            if (clean and len(clean) < 80 and clean.isupper()
                    and len(clean.split()) <= 8):
                # Likely a chapter/section heading
                if current_section["text_parts"]:
                    current_section["text"] = "\n\n".join(current_section["text_parts"])
                    del current_section["text_parts"]
                    current_section["pages"] = list(set(current_section["pages"]))
                    sections.append(current_section)
                current_section = {
                    "title": clean.title(),
                    "pages": [],
                    "text_parts": [],
                }
                break

        current_section["pages"].append(page["page_number"])
        current_section["text_parts"].append(page["text"])

    # Don't forget the last section
    if current_section["text_parts"]:
        current_section["text"] = "\n\n".join(current_section["text_parts"])
        del current_section["text_parts"]
        current_section["pages"] = list(set(current_section["pages"]))
        sections.append(current_section)

    total_chars = sum(p["char_count"] for p in pymupdf_pages)
    total_pages = len(pymupdf_pages)

    return {
        "metadata": {
            **metadata,
            "format": "pdf",
            "source_file": filepath.name,
            "total_pages": total_pages,
            "total_characters": total_chars,
            "extraction_date": datetime.now(timezone.utc).isoformat(),
            "extractors": ["pymupdf", "pdfplumber"],
        },
        "sections": sections,
        "raw_pages": pymupdf_pages,
    }


# ── EPUB Extraction ──────────────────────────────────────────────────────────


def clean_html(html_content: str) -> str:
    """Clean HTML content from EPUB chapters into readable text."""
    soup = BeautifulSoup(html_content, "lxml")

    # Remove scripts, styles, and other non-content elements
    for tag in soup.find_all(["script", "style", "meta", "link"]):
        tag.decompose()

    # Extract text with some structure preservation
    lines = []
    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "div"]
    ):
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue

        # Mark headings
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(element.name[1])
            prefix = "#" * level
            lines.append(f"\n{prefix} {text}\n")
        elif element.name == "li":
            lines.append(f"  - {text}")
        elif element.name == "blockquote":
            lines.append(f"> {text}")
        else:
            lines.append(text)

    result = "\n".join(lines)
    # Clean up excessive whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_heading_from_html(html_content: str) -> str | None:
    """Try to extract the first heading from HTML content."""
    soup = BeautifulSoup(html_content, "lxml")
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            return heading.get_text(strip=True)
    return None


def extract_epub(filepath: Path) -> dict:
    """Extract text from an EPUB file."""
    metadata = match_book_metadata(filepath.name)
    print(f"  Extracting EPUB: {metadata['title']}")

    book = epub.read_epub(str(filepath), options={"ignore_ncx": True})

    # Get book-level metadata
    epub_title = book.get_metadata("DC", "title")
    epub_creator = book.get_metadata("DC", "creator")

    sections = []
    total_chars = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode("utf-8", errors="replace")
            text = clean_html(content)

            if not text or len(text.strip()) < 50:
                continue

            heading = extract_heading_from_html(content)
            char_count = len(text)
            total_chars += char_count

            sections.append({
                "title": heading,
                "chapter_id": item.get_name(),
                "text": text,
                "char_count": char_count,
            })

    return {
        "metadata": {
            **metadata,
            "format": "epub",
            "source_file": filepath.name,
            "epub_title": epub_title[0][0] if epub_title else None,
            "epub_creator": epub_creator[0][0] if epub_creator else None,
            "total_sections": len(sections),
            "total_characters": total_chars,
            "extraction_date": datetime.now(timezone.utc).isoformat(),
            "extractors": ["ebooklib", "beautifulsoup4"],
        },
        "sections": sections,
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────


def get_source_files() -> list[Path]:
    """Get all valid source files, excluding partial downloads."""
    files = []
    for ext in ["*.pdf", "*.epub"]:
        for f in sorted(SOURCES_DIR.glob(ext)):
            # Skip partial downloads
            if ".part" in f.name:
                continue
            # Skip duplicate Missy Robbins copies (keep the z-library.sk one)
            if "Missy Robbins" in f.name and "Z-Library" in f.name:
                continue
            files.append(f)
    return files


def extract_all():
    """Run the full extraction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_files = get_source_files()
    print(f"Found {len(source_files)} source files to extract:\n")
    for f in source_files:
        print(f"  {'[PDF] ' if f.suffix == '.pdf' else '[EPUB]'} {f.name[:70]}...")

    print(f"\n{'='*60}")
    print("Starting extraction...\n")

    results = []
    for filepath in source_files:
        try:
            if filepath.suffix == ".pdf":
                result = extract_pdf(filepath)
            elif filepath.suffix == ".epub":
                result = extract_epub(filepath)
            else:
                print(f"  Skipping unsupported format: {filepath.name}")
                continue

            # Write output
            short_id = result["metadata"]["short_id"]
            output_path = OUTPUT_DIR / f"{short_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            section_count = len(result.get("sections", []))
            total_chars = result["metadata"]["total_characters"]
            print(f"    ✓ {section_count} sections, {total_chars:,} chars → {output_path.name}")
            results.append(result["metadata"])

        except Exception as e:
            print(f"    ✗ ERROR extracting {filepath.name}: {e}")
            import traceback
            traceback.print_exc()

    # Write extraction manifest
    manifest = {
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(results),
        "total_characters": sum(r["total_characters"] for r in results),
        "sources": results,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Sources processed: {len(results)}")
    print(f"  Total characters:  {manifest['total_characters']:,}")
    print(f"  Output directory:  {OUTPUT_DIR}")
    print(f"  Manifest:          {manifest_path}")


if __name__ == "__main__":
    extract_all()
