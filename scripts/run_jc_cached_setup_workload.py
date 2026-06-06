from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JosephsonCircuits cached setup workload scenarios from twpa_jax."
    )
    parser.add_argument(
        "--jc-repo",
        default=r"D:\Projects\Thesis\JosephsonCircuits.jl",
        help="Path to the local JosephsonCircuits.jl checkout.",
    )
    parser.add_argument(
        "--julia",
        default="julia",
        help="Julia executable.",
    )
    parser.add_argument(
        "--outdir",
        default=r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_h_twpa_orchestrated_cache",
        help="Output directory for aggregate and per-scenario outputs.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["fixed", "varying", "mixed", "all"],
        help="Workload scenario to run.",
    )
    parser.add_argument("--cells", default="1,16,64,128")
    parser.add_argument("--nmodes", default="1,9,17")
    parser.add_argument("--requests-per-case", default="100")
    parser.add_argument("--repetitions", default="10")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv_one(path: Path) -> Dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def read_equivalence_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = str(row.get("signatures_match", "")).strip().lower()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def run_scenario(args: argparse.Namespace, scenario: str, root_outdir: Path) -> Dict[str, object]:
    jc_repo = Path(args.jc_repo)
    script = jc_repo / "experiments" / "thesis_gpu_parallel" / "run_cached_setup_workload.jl"

    if not script.exists():
        raise FileNotFoundError(f"Missing JC workload script: {script}")

    scenario_outdir = root_outdir / scenario
    scenario_outdir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        args.julia,
        f"--project={jc_repo}",
        str(script),
        "--outdir",
        str(scenario_outdir),
        "--cells",
        args.cells,
        "--nmodes",
        args.nmodes,
        "--requests-per-case",
        str(args.requests_per_case),
        "--scenario",
        scenario,
        "--repetitions",
        str(args.repetitions),
    ]

    print("\n=== Running scenario:", scenario, "===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    summary_path = scenario_outdir / "cached_setup_workload_summary.csv"
    equivalence_path = scenario_outdir / "cached_setup_workload_equivalence.tsv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not equivalence_path.exists():
        raise FileNotFoundError(equivalence_path)

    summary = read_csv_one(summary_path)
    eq_counts = read_equivalence_counts(equivalence_path)

    false_count = sum(count for key, count in eq_counts.items() if key != "true")
    true_count = eq_counts.get("true", 0)

    result: Dict[str, object] = {
        "scenario": scenario,
        "requests": int(summary["requests"]),
        "reference_median_s": float(summary["reference_median_s"]),
        "cached_median_s": float(summary["cached_median_s"]),
        "speedup": float(summary["speedup"]),
        "reference_mean_alloc_mib": float(summary["reference_mean_alloc_mib"]),
        "cached_mean_alloc_mib": float(summary["cached_mean_alloc_mib"]),
        "alloc_ratio_ref_over_cached": float(summary["alloc_ratio_ref_over_cached"]),
        "equivalence_true": true_count,
        "equivalence_false": false_count,
        "scenario_outdir": str(scenario_outdir),
    }

    if true_count == 0 or false_count != 0:
        raise RuntimeError(
            f"Equivalence failed for {scenario}: true={true_count}, false={false_count}, counts={eq_counts}"
        )

    return result


def write_aggregate(root_outdir: Path, results: Iterable[Dict[str, object]]) -> None:
    rows = list(results)
    csv_path = root_outdir / "jc_cached_setup_workload_results.csv"
    json_path = root_outdir / "jc_cached_setup_workload_results.json"

    fieldnames = [
        "scenario",
        "requests",
        "reference_median_s",
        "cached_median_s",
        "speedup",
        "reference_mean_alloc_mib",
        "cached_mean_alloc_mib",
        "alloc_ratio_ref_over_cached",
        "equivalence_true",
        "equivalence_false",
        "scenario_outdir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "status": "PASS",
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nPASS")
    print(f"aggregate_csv  = {csv_path}")
    print(f"aggregate_json = {json_path}")


def main() -> int:
    args = parse_args()
    root_outdir = Path(args.outdir)

    if args.force and root_outdir.exists():
        shutil.rmtree(root_outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    scenarios = ["fixed", "varying", "mixed"] if args.scenario == "all" else [args.scenario]

    results = []
    for scenario in scenarios:
        results.append(run_scenario(args, scenario, root_outdir))

    write_aggregate(root_outdir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
