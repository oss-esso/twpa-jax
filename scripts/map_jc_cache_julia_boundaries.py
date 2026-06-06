from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".")
OUTDIR = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_i2_julia_boundary_map")

PATTERNS = [
    ("subprocess_run", r"\bsubprocess\.(run|Popen|check_call|check_output)\b"),
    ("julia_token", r"\bjulia\b|Julia"),
    ("harmonia_token", r"\bHarmonia\b|harmonia"),
    ("run_simulation_jl", r"run_simulation\.jl"),
    ("project_arg", r"--project"),
    ("benchmark_suite", r"run_harmonia_benchmark_suite"),
    ("config_loop", r"for .*config|for .*case|for .*run|for .*scenario"),
    ("single_run_function", r"def .*run.*harmonia|def .*run.*julia|def .*simulation"),
]

SEARCH_DIRS = [
    ROOT / "scripts",
    ROOT / "twpa",
    ROOT / "tests",
]


def iter_files() -> list[Path]:
    out: list[Path] = []
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            out.append(path)
    return sorted(out)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for path in iter_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        for idx, line in enumerate(lines, start=1):
            for label, pattern in PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    rows.append({
                        "file": str(path),
                        "line": str(idx),
                        "label": label,
                        "code": line.strip(),
                    })

    source_csv = OUTDIR / "julia_boundary_source_map.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "label", "code"])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["file"], row["label"])
        summary[key] = summary.get(key, 0) + 1

    summary_csv = OUTDIR / "julia_boundary_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "count"])
        writer.writeheader()
        for (file, label), count in sorted(summary.items()):
            writer.writerow({"file": file, "label": label, "count": count})

    report = OUTDIR / "julia_boundary_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# JC-3M-I2 Julia/Harmonia boundary map\n\n")
        f.write(f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "Find where twpa_jax launches Julia/Harmonia. The JC setup cache only helps "
            "when repeated requests run inside the same Julia process, so subprocess boundaries "
            "matter more than generic Python sweep functions.\n\n"
        )
        f.write("## Summary\n\n")
        f.write("| file | label | count |\n")
        f.write("|---|---|---:|\n")
        for (file, label), count in sorted(summary.items()):
            f.write(f"| `{file}` | `{label}` | {count} |\n")

    print("PASS")
    print(f"source_csv  = {source_csv}")
    print(f"summary_csv = {summary_csv}")
    print(f"report      = {report}")


if __name__ == "__main__":
    main()
