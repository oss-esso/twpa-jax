# Agent guide

Treat this corpus as a two-layer source:

1. **Source layer** (`source_pages.jsonl`, `source_fulltext_layout.txt`, `source_fulltext_raw.txt`) is the complete mechanically extracted content of each supplied PDF.
2. **Interpretation layer** (`reproduction_dossier.md`, `equations_index.md`, `figures_tables_index.md`) is navigation and source-grounded synthesis.

When answering a technical question, cite internally by `document slug / PDF page / equation or figure label`. If a mathematical symbol in an extracted equation looks suspicious, do not silently repair it from general knowledge: retrieve the page from `source_pages.jsonl` and, if exact typography is still ambiguous, inspect the original PDF page.

For reproducing a paper, use this order:

- read `reproduction_dossier.md`;
- collect all parameter values and boundary conditions from the complete page text;
- implement the numbered equations in `equations_index.md`;
- reproduce every validation object in `figures_tables_index.md` in the same logical order as the paper;
- only then compare the final gain/chaos/dispersion result.

The 1982 and 1986 PDFs have old text encodings and their mathematical glyph extraction is noisier than the modern papers. Their equation labels and page numbers are reliable navigation anchors, but exact formula typography should be checked visually in the PDF.
