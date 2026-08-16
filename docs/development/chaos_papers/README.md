# LLM-ready Josephson / JTWPA paper corpus

This directory is an extraction-and-reproduction corpus built from the eight PDFs supplied in the conversation. It is designed for another LLM agent to use without relying on a prose summary alone.

## What “full content” means here

For every PDF, the corpus includes **the entire mechanically extracted page text**, both as layout-preserving text and as one-record-per-page JSONL. That full-source layer is separate from the synthesized `reproduction_dossier.md`, so an agent can always retrieve the actual source wording/equations instead of trusting a summary. The numbered-equation and figure/table indices are navigation aids over that full layer.

PDF mathematical text extraction is not perfectly lossless: stacked fractions, superscripts and some older-font Greek glyphs can be garbled. For that reason every equation entry records its PDF page and label. The original PDF is authoritative for exact typography. No OCR was used because these PDFs contain native text.

## Directory structure

Each folder under `documents/` contains:

- `reproduction_dossier.md` — source-grounded model, parameters, method, figure interpretation, and a concrete reproduction sequence;
- `equations_index.md` — detected numbered equations with page-local source context;
- `figures_tables_index.md` — detected captions and nearby author discussion;
- `source_pages.jsonl` — complete page-by-page text, ideal for retrieval agents;
- `source_fulltext_layout.txt` — complete layout-preserving extraction;
- `source_fulltext_raw.txt` — independent full raw text extraction;
- `README.md` — source identity and hash.

At the root, `manifest.json` is machine-readable metadata, `corpus_pages.jsonl` concatenates every PDF page for one-shot indexing, and `00_cross_paper_map.md` explains how the papers relate. The `sources/` directory contains copies of all eight original PDFs, so the corpus is self-contained and exact figures/equation typography remain available to a multimodal agent.

## Recommended agent usage

1. Start with `manifest.json` and `00_cross_paper_map.md`.
2. Read the relevant paper's `reproduction_dossier.md`.
3. Retrieve exact equations from `equations_index.md`, then verify ambiguous notation in `source_pages.jsonl` / original PDF page.
4. Use `figures_tables_index.md` to identify every validation plot and what the authors say it demonstrates.
5. Before implementing, search the complete page text for every parameter and boundary condition used in the target figure.

## Source fidelity

The synthesized dossiers intentionally do not add outside literature or silently “fix” the papers. Where a source is ambiguous or an old PDF extracts poorly, the corpus preserves that ambiguity and points back to the source page.
