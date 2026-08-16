# Extraction quality and coverage report

The corpus covers all eight supplied PDFs and all 255 PDF pages. No OCR was used; all PDFs had native text layers. Two independent full-text extractions are retained per source (layout-preserving `pdftotext` and PyMuPDF raw text), plus one JSONL record per page.

Numbered-equation indices are based on the source's printed numbering. Modern papers and the 2022 thesis extract cleanly. The older 1982/1986 typesetting sometimes converts parentheses to braces or damages Greek/math glyphs; the index normalizes equation *labels* but deliberately does not rewrite uncertain symbols. The original PDF page is authoritative.

Figure/table indices combine extracted captions with nearby paragraphs where the authors discuss those figures. The thesis index contains the 86 main-chapter figures plus Appendix Figure B.1. Short-paper captions that were embedded in multi-column/image blocks were recovered from the layout text and added as supplemental entries.

The reproduction dossiers are not substitutes for the source layer. They are deliberately separated so an LLM can audit every claim against the complete extracted PDF content.
