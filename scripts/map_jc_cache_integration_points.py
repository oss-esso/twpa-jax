from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".")
OUTDIR = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_i_workflow_cache_integration_map")

TARGET_PATTERNS = [
    ("harmonia_runner", r"run_harmonia|Harmonia|julia.*run_simulation|run_simulation\.jl"),
    ("campaign_loop", r"for .* in .*campaign|campaign|sweep|dataset|grid|parameter"),
    ("subprocess_julia", r"subprocess\.run|subprocess\.Popen|julia"),
    ("benchmark_suite", r"run_harmonia_benchmark_suite|benchmark"),
    ("hdf5_output", r"\.h5|hdf5|HDF5"),
    ("status_reader", r"status|summary|registry"),
    ("objective_eval", r"objective|cost|loss|fit|calibration"),
    ("cache_candidate", r"cache|memo|reuse|registry"),
]

SEARCH_DIRS = [
    ROOT / "scripts",
    ROOT / "twpa",
    ROOT / "tests",
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".json", ".toml", ".md"}:
                files.append(p)
    return sorted(files)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for path in iter_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        for line_no, line in enumerate(lines, start=1):
            for label, pattern in TARGET_PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    rows.append(
                        {
                            "file": str(path),
                            "line": str(line_no),
                            "label": label,
                            "code": line.strip(),
                        }
                    )

    csv_path = OUTDIR / "twpa_workflow_cache_integration_source_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "label", "code"])
        writer.writeheader()
        writer.writerows(rows)

    by_file_label: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["file"], r["label"])
        by_file_label[key] = by_file_label.get(key, 0) + 1

    summary_path = OUTDIR / "twpa_workflow_cache_integration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "count"])
        writer.writeheader()
        for (file, label), count in sorted(by_file_label.items()):
            writer.writerow({"file": file, "label": label, "count": count})

    report_path = OUTDIR / "twpa_workflow_cache_integration_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# JC-3M-I twpa_jax cached setup integration map\n\n")
        f.write(f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`\n")
        f.write(f"- repo: `{ROOT.resolve()}`\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "Map where repeated Harmonia/JosephsonCircuits simulations are launched so the "
            "internal HB setup cache can be integrated at the repeated-workflow layer, not as a "
            "one-shot public solver change.\n\n"
        )
        f.write("## Summary by file and label\n\n")
        f.write("| file | label | count |\n")
        f.write("|---|---|---:|\n")
        for (file, label), count in sorted(by_file_label.items()):
            f.write(f"| `{file}` | `{label}` | {count} |\n")
        f.write("\n## Decision rule\n\n")
        f.write(
            "The first real integration point should be the smallest repeated-simulation loop "
            "that already owns a batch/campaign context and can safely keep a cache object alive "
            "across repeated requests.\n"
        )

    print("PASS")
    print(f"source_map = {csv_path}")
    print(f"summary    = {summary_path}")
    print(f"report     = {report_path}")


if __name__ == "__main__":
    main()
