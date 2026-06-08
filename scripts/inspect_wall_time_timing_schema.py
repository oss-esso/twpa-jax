from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

WORKSPACE = Path(r"D:\Projects\Thesis")
OUT = WORKSPACE / "outputs"
REPORT_DIR = OUT / "jc_profiles" / "jc3m_wall_time_budget_inspection"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def flatten_keys(obj, prefix=""):
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            keys.append(name)
            keys.extend(flatten_keys(v, name))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            name = f"{prefix}[{i}]"
            keys.extend(flatten_keys(v, name))
    return keys

json_files = []
for root in [
    OUT / "benchmarks",
    OUT / "jc_profiles",
]:
    if root.exists():
        json_files.extend(root.glob("jc3*/**/*.json"))

timingish = [
    "time",
    "timing",
    "runtime",
    "wall",
    "hbsolve",
    "hblin",
    "block",
    "stage",
    "cache_telemetry",
]

rows = []
key_counter = Counter()
files_with_timing = []

for path in sorted(json_files):
    obj = load_json(path)
    if obj is None:
        continue

    keys = flatten_keys(obj)
    interesting = [k for k in keys if any(t in k.lower() for t in timingish)]

    if interesting:
        files_with_timing.append(path)
        for k in interesting:
            key_counter[k] += 1

        rows.append({
            "path": str(path),
            "n_interesting_keys": len(interesting),
            "sample_keys": "; ".join(interesting[:40]),
        })

with (REPORT_DIR / "json_timing_files.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "n_interesting_keys", "sample_keys"])
    writer.writeheader()
    writer.writerows(rows)

with (REPORT_DIR / "json_timing_key_counts.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["key", "count"])
    writer.writeheader()
    for key, count in key_counter.most_common():
        writer.writerow({"key": key, "count": count})

# CSV schema scan
csv_rows = []
csv_files = []
for root in [
    OUT / "benchmarks",
    OUT / "jc_profiles",
]:
    if root.exists():
        csv_files.extend(root.glob("jc3*/**/*.csv"))
        csv_files.extend(root.glob("jc3*/**/*.tsv"))

for path in sorted(set(csv_files)):
    try:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            header = next(reader, [])
    except Exception:
        continue

    interesting_cols = [c for c in header if any(t in c.lower() for t in timingish)]
    if interesting_cols:
        csv_rows.append({
            "path": str(path),
            "columns": "; ".join(header),
            "interesting_columns": "; ".join(interesting_cols),
        })

with (REPORT_DIR / "csv_timing_files.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "columns", "interesting_columns"])
    writer.writeheader()
    writer.writerows(csv_rows)

print("WROTE", REPORT_DIR)
print()
print("Top JSON timing keys:")
for key, count in key_counter.most_common(30):
    print(f"{count:4d}  {key}")

print()
print("CSV files with timing-ish columns:")
for row in csv_rows[:40]:
    print(row["path"])
    print("  ", row["interesting_columns"])
