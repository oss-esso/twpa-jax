"""
Inventory both Harmonia Julia repos.

Repos inspected:

    D:/Projects/Thesis/Harmonia.jl
    D:/Projects/Thesis/Harmonia

Purpose:

    - classify stable package code vs exploratory scripts;
    - find Harmonia builders;
    - find JosephsonCircuits usage;
    - find hbsolve / linearized.S / QE / CM workflows;
    - find IPM / rf-SQUID / JTWPA / coupler scripts;
    - produce docs for migration into stable simulation templates.

This is read-only. It does not modify Julia repos.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


FUNCTION_RE = re.compile(
    r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_!?.]*)\s*\((.*?)\)",
    re.MULTILINE,
)
INCLUDE_RE = re.compile(r'include\(["\']([^"\']+)["\']\)')
USING_RE = re.compile(r"^\s*(using|import)\s+(.+)$", re.MULTILINE)


KEYWORDS = {
    "josephson": ["JosephsonCircuits", "hbsolve", "linearized", ".S(", ".QE(", ".CM("],
    "hb": ["hbsolve", "Npumpharmonics", "Nmodulationharmonics", "sources", "pump"],
    "sparams": ["linearized.S", ".S(", "S_matrix", "S21", "S11", "scattering"],
    "noise": ["QE", ".QE(", "CM", ".CM(", "quantum efficiency", "commutation"],
    "ipm": ["IPM", "make_IPM", "directional", "coupler", "generate_and_append_coupler"],
    "rfsquid": ["RF_JTL", "RF_squid", "rf_SQUID", "Lrf", "SQUID"],
    "jtwpa": ["JTWPA", "JTL", "add_JTL", "RPM", "phase matching"],
    "fitting": ["fit", "loss", "cost", "objective", "Optim", "BlackBoxOptim", "NelderMead"],
    "io": ["HDF5", "save_hdf5", "jld", "jls", "serialize", "CSV", "DataFrame"],
    "plotting": ["Plots", "plot(", "heatmap", "savefig"],
    "measurement": ["measurement", "experimental", "VNA", "CSV.read", "data"],
}


PROMOTION_HINTS = {
    "STABLE_PACKAGE_CODE": [
        r"[/\\]src[/\\]",
    ],
    "REUSABLE_PROTOTYPE": [
        "hbsolve",
        "linearized.S",
        "fit",
        "cost",
        "IPM",
        "rf",
        "SQUID",
        "JTWPA",
        "save_hdf5",
    ],
    "ONE_OFF_EXPERIMENT": [
        "plot",
        "test",
        "bench",
        "old",
        "tutorial",
        "dev",
        "try",
    ],
}


@dataclass(frozen=True)
class JuliaFileRecord:
    repo_name: str
    relative_path: str
    absolute_path: str
    classification: str
    n_lines: int
    functions: list[str]
    using_imports: list[str]
    includes: list[str]
    keyword_hits: dict[str, int]
    calls_josephsoncircuits: bool
    calls_hbsolve: bool
    calls_linearized_s: bool
    calls_qe_or_cm: bool
    saves_data: bool
    has_fitting_logic: bool
    has_plotting: bool
    likely_device_tags: list[str]
    recommended_action: str


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except Exception as exc:
        return f"<<READ_FAILED: {type(exc).__name__}: {exc}>>"


def count_keyword_hits(text: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    text_lower = text.lower()

    for group, terms in KEYWORDS.items():
        count = 0
        for term in terms:
            count += text_lower.count(term.lower())
        if count:
            hits[group] = count

    return hits


def extract_functions(text: str) -> list[str]:
    names: list[str] = []
    for match in FUNCTION_RE.finditer(text):
        name = match.group(1)
        if name in {"if", "for", "while", "println", "plot", "Dict"}:
            continue
        names.append(name)
    return sorted(set(names))


def extract_imports(text: str) -> list[str]:
    out: list[str] = []
    for match in USING_RE.finditer(text):
        out.append(match.group(0).strip())
    return sorted(set(out))


def extract_includes(text: str) -> list[str]:
    return sorted(set(INCLUDE_RE.findall(text)))


def classify_file(repo_name: str, rel: str, text: str, hits: dict[str, int]) -> str:
    rel_norm = rel.replace("\\", "/").lower()

    if repo_name == "Harmonia.jl" and "/src/" in f"/{rel_norm}":
        return "STABLE_PACKAGE_CODE"

    if repo_name == "Harmonia.jl" and "/test/" in f"/{rel_norm}":
        return "TEST_OR_VALIDATION"

    if "old" in rel_norm or "archive" in rel_norm or "deprecated" in rel_norm:
        return "OBSOLETE_OR_REFERENCE"

    if hits.get("josephson", 0) or hits.get("hb", 0) or hits.get("ipm", 0) or hits.get("rfsquid", 0):
        return "REUSABLE_PROTOTYPE"

    if hits.get("fitting", 0) or hits.get("measurement", 0):
        return "REUSABLE_PROTOTYPE"

    if hits.get("plotting", 0) and not (hits.get("josephson", 0) or hits.get("hb", 0)):
        return "ONE_OFF_EXPERIMENT"

    if repo_name == "Harmonia":
        return "UNKNOWN_ARCHAEOLOGY"

    return "UNKNOWN"


def device_tags(text: str, rel: str) -> list[str]:
    hay = f"{rel}\n{text}".lower()
    tags = []

    checks = {
        "JTWPA": ["jtwpa", "jtl"],
        "IPM": ["ipm", "interferometric"],
        "RF_SQUID": ["rf_squid", "rf-squid", "rfsquid", "lrf"],
        "DIRECTIONAL_COUPLER": ["directional", "coupler", "cpw"],
        "JPA": ["jpa"],
        "FLOQUET": ["floquet"],
        "RPM": ["rpm", "resonant"],
        "FITTING": ["fit", "cost", "loss", "objective"],
        "MEASUREMENT": ["measurement", "experimental", "vna"],
        "NOISE_QE_CM": ["qe", "cm", "quantum efficiency", "commutation"],
    }

    for tag, words in checks.items():
        if any(word in hay for word in words):
            tags.append(tag)

    return tags


def recommended_action(classification: str, hits: dict[str, int], tags: list[str]) -> str:
    if classification == "STABLE_PACKAGE_CODE":
        return "Keep in Harmonia.jl; wrap with typed config/tests/status schema."

    if "IPM" in tags or "RF_SQUID" in tags or "DIRECTIONAL_COUPLER" in tags:
        return "Mine as design recipe; promote only after reproducing with tiny smoke tests."

    if hits.get("fitting", 0) or "FITTING" in tags:
        return "Use as calibration-reference logic; port concepts into twpa_jax objectives/campaigns."

    if hits.get("hb", 0) or hits.get("josephson", 0):
        return "Extract minimal solver call pattern and add a tiny schema-preserving smoke."

    if classification == "ONE_OFF_EXPERIMENT":
        return "Reference only; do not promote until reusable parameters and tests are defined."

    if classification == "OBSOLETE_OR_REFERENCE":
        return "Keep as historical reference only."

    return "Inspect manually before promotion."


def inspect_file(repo_name: str, repo_root: Path, path: Path) -> JuliaFileRecord:
    rel = str(path.relative_to(repo_root))
    text = read_text_safe(path)
    hits = count_keyword_hits(text)
    funcs = extract_functions(text)
    imports = extract_imports(text)
    includes = extract_includes(text)
    tags = device_tags(text, rel)
    classification = classify_file(repo_name, rel, text, hits)

    return JuliaFileRecord(
        repo_name=repo_name,
        relative_path=rel,
        absolute_path=str(path),
        classification=classification,
        n_lines=text.count("\n") + 1,
        functions=funcs,
        using_imports=imports,
        includes=includes,
        keyword_hits=hits,
        calls_josephsoncircuits=hits.get("josephson", 0) > 0,
        calls_hbsolve="hbsolve" in text,
        calls_linearized_s=("linearized.S" in text or ".S(" in text),
        calls_qe_or_cm=(".QE(" in text or ".CM(" in text or "linearized.QE" in text or "linearized.CM" in text),
        saves_data=hits.get("io", 0) > 0,
        has_fitting_logic=hits.get("fitting", 0) > 0,
        has_plotting=hits.get("plotting", 0) > 0,
        likely_device_tags=tags,
        recommended_action=recommended_action(classification, hits, tags),
    )


def find_julia_files(repo_root: Path) -> list[Path]:
    ignored = {
        ".git",
        ".julia",
        "Manifest.toml",
        "outputs",
        "__pycache__",
    }

    files: list[Path] = []
    for path in repo_root.rglob("*.jl"):
        parts = set(path.parts)
        if parts.intersection(ignored):
            continue
        files.append(path)

    return sorted(files)


def write_markdown_inventory(path: Path, records: list[JuliaFileRecord]) -> None:
    by_repo = defaultdict(list)
    for r in records:
        by_repo[r.repo_name].append(r)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Harmonia Workspace Inventory\n\n")
        f.write("This file is generated by `scripts/inventory_harmonia_workspace.py`.\n\n")
        f.write("It inventories both `Harmonia.jl` and exploratory `Harmonia`.\n\n")

        f.write("## Summary\n\n")
        f.write(f"- total Julia files: {len(records)}\n")
        for repo, rows in sorted(by_repo.items()):
            f.write(f"- {repo}: {len(rows)} files\n")

        f.write("\n## Classification counts\n\n")
        counts = Counter(r.classification for r in records)
        for key, val in counts.most_common():
            f.write(f"- {key}: {val}\n")

        f.write("\n## Device/design tag counts\n\n")
        tag_counter = Counter(tag for r in records for tag in r.likely_device_tags)
        for key, val in tag_counter.most_common():
            f.write(f"- {key}: {val}\n")

        f.write("\n## High-priority reusable files\n\n")
        high_priority = [
            r for r in records
            if (
                r.calls_hbsolve
                or r.calls_linearized_s
                or r.calls_qe_or_cm
                or "IPM" in r.likely_device_tags
                or "RF_SQUID" in r.likely_device_tags
                or "DIRECTIONAL_COUPLER" in r.likely_device_tags
                or r.has_fitting_logic
            )
        ]

        for r in sorted(high_priority, key=lambda x: (x.repo_name, x.relative_path)):
            f.write(f"\n### `{r.repo_name}/{r.relative_path}`\n\n")
            f.write(f"- classification: `{r.classification}`\n")
            f.write(f"- lines: {r.n_lines}\n")
            f.write(f"- tags: {', '.join(r.likely_device_tags) if r.likely_device_tags else 'none'}\n")
            f.write(f"- calls JosephsonCircuits: {r.calls_josephsoncircuits}\n")
            f.write(f"- calls hbsolve: {r.calls_hbsolve}\n")
            f.write(f"- calls linearized.S: {r.calls_linearized_s}\n")
            f.write(f"- calls QE/CM: {r.calls_qe_or_cm}\n")
            f.write(f"- saves data: {r.saves_data}\n")
            f.write(f"- fitting logic: {r.has_fitting_logic}\n")
            f.write(f"- recommendation: {r.recommended_action}\n")
            if r.functions:
                f.write(f"- functions: `{', '.join(r.functions[:20])}`")
                if len(r.functions) > 20:
                    f.write(f" ... +{len(r.functions) - 20} more")
                f.write("\n")
            if r.using_imports:
                f.write("- imports:\n")
                for imp in r.using_imports[:10]:
                    f.write(f"  - `{imp}`\n")

        f.write("\n## Full file table\n\n")
        f.write("| repo | file | class | tags | JC | HB | S | QE/CM | IO | fit | action |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")

        for r in sorted(records, key=lambda x: (x.repo_name, x.relative_path)):
            tags = ", ".join(r.likely_device_tags)
            f.write(
                f"| {r.repo_name} | `{r.relative_path}` | {r.classification} | "
                f"{tags} | {int(r.calls_josephsoncircuits)} | {int(r.calls_hbsolve)} | "
                f"{int(r.calls_linearized_s)} | {int(r.calls_qe_or_cm)} | "
                f"{int(r.saves_data)} | {int(r.has_fitting_logic)} | "
                f"{r.recommended_action} |\n"
            )


def write_candidate_map(path: Path, records: list[JuliaFileRecord]) -> None:
    candidates = [
        r for r in records
        if (
            r.classification in {"STABLE_PACKAGE_CODE", "REUSABLE_PROTOTYPE"}
            or r.calls_hbsolve
            or r.has_fitting_logic
        )
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Harmonia Design Candidate Map\n\n")
        f.write("Purpose: decide what to promote first into stable Harmonia/JosephsonCircuits templates.\n\n")

        f.write("## Recommended implementation order\n\n")
        f.write("1. `harmonia_jtl_topology_smoke`: use `add_JTL!` to generate a tiny JTL netlist; validate element counts and metadata only.\n")
        f.write("2. `harmonia_jtl_linear_jc_smoke`: same tiny JTL netlist, passed to JosephsonCircuits linearized S if compatible.\n")
        f.write("3. `harmonia_rf_jtl_topology_smoke`: use `add_RF_JTL!` to generate a tiny RF-SQUID/JTL netlist; validate topology.\n")
        f.write("4. `harmonia_coupler_topology_smoke`: use `generate_and_append_coupler!` with tiny/cheap settings; validate coupled mesh metadata.\n")
        f.write("5. `harmonia_ipm_topology_smoke`: only after coupler and JTL smoke tests pass, because `make_IPM` has a large positional signature.\n")
        f.write("6. Mine exploratory `Harmonia` scripts for realistic IPM/rf-SQUID/JTWPA parameters and fitting logic.\n\n")

        f.write("## Candidate files\n\n")
        for r in sorted(candidates, key=lambda x: (x.classification, x.repo_name, x.relative_path)):
            f.write(f"### `{r.repo_name}/{r.relative_path}`\n\n")
            f.write(f"- class: `{r.classification}`\n")
            f.write(f"- tags: {', '.join(r.likely_device_tags) if r.likely_device_tags else 'none'}\n")
            f.write(f"- action: {r.recommended_action}\n")
            f.write(f"- functions: {', '.join(r.functions[:12]) if r.functions else 'none'}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=_WORKSPACE_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace_root
    repos = {
        "Harmonia.jl": workspace / "Harmonia.jl",
        "Harmonia": workspace / "Harmonia",
    }

    records: list[JuliaFileRecord] = []

    for repo_name, repo_root in repos.items():
        if not repo_root.exists():
            raise FileNotFoundError(f"Missing repo: {repo_root}")

        for path in find_julia_files(repo_root):
            records.append(inspect_file(repo_name, repo_root, path))

    out_dir = workspace / "outputs" / "harmonia_archaeology"
    docs_dir = _REPO_ROOT / "docs" / "harmonia_reuse"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "harmonia_workspace_inventory.json"
    md_path = docs_dir / "full_harmonia_workspace_inventory.md"
    candidate_path = docs_dir / "harmonia_design_candidate_map.md"

    json_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_inventory(md_path, records)
    write_candidate_map(candidate_path, records)

    summary = {
        "n_files": len(records),
        "json_path": str(json_path),
        "markdown_inventory": str(md_path),
        "candidate_map": str(candidate_path),
        "classification_counts": dict(Counter(r.classification for r in records)),
        "device_tag_counts": dict(Counter(tag for r in records for tag in r.likely_device_tags)),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Harmonia workspace inventory")
        print("============================")
        print(f"n_files:             {summary['n_files']}")
        print(f"json_path:           {json_path}")
        print(f"markdown_inventory:  {md_path}")
        print(f"candidate_map:       {candidate_path}")
        print(f"classifications:     {summary['classification_counts']}")
        print(f"device tags:         {summary['device_tag_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())