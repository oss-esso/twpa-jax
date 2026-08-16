"""Phase 9 documentation acceptance gates."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    """Read one tracked Phase 9 document as UTF-8 text."""

    return (ROOT / relative).read_text(encoding="utf-8")


def test_phase9_deliverables_exist() -> None:
    """All required Phase 9 documents are present."""

    for relative in (
        "docs/design_format.md",
        "docs/development/GDS_API_MAPPING.md",
        "docs/development/circuit_api.md",
        "docs/development/circuit_oop_migration_report.md",
        "CLAUDE.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_design_format_declares_python_authority() -> None:
    """The YAML document describes an adapter, not a second implementation."""

    text = _read("docs/design_format.md")
    assert "authoritative design authoring interface" in text
    assert "YAML adapter" in text
    assert "designs/python/" in text


def test_api_guide_covers_required_distinctions() -> None:
    """The guide records profiles, numbering, edits, and explicit geometry."""

    text = _read("docs/development/circuit_api.md")
    for required in (
        "HalfSine",
        "sin(pi*t/2)",
        "Hann",
        "half_cosine",
        "node_numbering=\"creation\"",
        "node_numbering=\"legacy\"",
        "ExplicitCouplerGeometry",
        "set_value",
        "YAML adapter",
    ):
        assert required in text, required


def test_mapping_and_report_record_scope_and_acceptance() -> None:
    """The mapping and report state retained code and non-GDS claims."""

    mapping = _read("docs/development/GDS_API_MAPPING.md")
    report = _read("docs/development/circuit_oop_migration_report.md")
    assert "Phase 9 finalized mapping" in mapping
    assert "add_jj_array" in mapping
    assert "not validated against" in mapping
    assert "fabrication GDS" in mapping
    assert "Acceptance checklist" in report
    assert "builders/blocks.py" in report
    assert "-24.99999994" in report
