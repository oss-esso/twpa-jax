from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(r"D:\Projects\Thesis")
JC_ROOT = WORKSPACE / "JosephsonCircuits.jl"
OUT = WORKSPACE / "outputs" / "jc_profiles" / "jc3m_wall_time_five_block_ws_matrix"
OUT.mkdir(parents=True, exist_ok=True)

WS_COUNTS = [9, 101, 501, 1001]
REPETITIONS = 5
NBATCHES = 1
ITERATIONS = 1000


def run_cmd(cmd: list[str], cwd: Path):
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, time.perf_counter() - t0, p.stdout, p.stderr


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_matrix_summary_from_stage_timings(stage_csv: Path, matrix_csv: Path, case_id: str) -> None:
    rows = read_csv(stage_csv)
    groups: dict[str, list[float]] = {}

    for r in rows:
        groups.setdefault(r["stage"], []).append(float(r["time_s"]))

    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "stage", "time_s_mean"])
        writer.writeheader()
        for stage, vals in sorted(groups.items()):
            writer.writerow({
                "case_id": case_id,
                "stage": stage,
                "time_s_mean": sum(vals) / len(vals),
            })


def fnum(row: dict[str, str], key: str) -> float:
    x = row.get(key, "")
    return 0.0 if x == "" else float(x)


def pct(num: float, den: float):
    return None if den <= 0 else 100.0 * num / den


block_names = [
    "block1_frequency_bookkeeping",
    "block2_circuit_matrix_construction",
    "block3_nonlinear_pump_hb_solve",
    "block4_linearized_signal_idler_solve",
    "block5_port_output_conversion_outer",
]

all_rows = []

for ws_count in WS_COUNTS:
    case_id = f"ws{ws_count}_nb{NBATCHES}"
    case_root = OUT / case_id
    profile_out = case_root / "raw_profile"
    analysis_out = case_root / "five_block_analysis"
    profile_out.mkdir(parents=True, exist_ok=True)
    analysis_out.mkdir(parents=True, exist_ok=True)

    profile_cmd = [
        "julia",
        "--project=.",
        "experiments/thesis_gpu_parallel/run_hbsolve_profile.jl",
        "--outdir",
        str(profile_out),
        "--repetitions",
        str(REPETITIONS),
        "--profile-repetitions",
        "0",
        "--ws-count",
        str(ws_count),
        "--nbatches",
        str(NBATCHES),
        "--iterations",
        str(ITERATIONS),
    ]

    rc, python_wall_s, stdout, stderr = run_cmd(profile_cmd, JC_ROOT)
    (profile_out / "wrapper_stdout.log").write_text(stdout, encoding="utf-8")
    (profile_out / "wrapper_stderr.log").write_text(stderr, encoding="utf-8")

    if rc != 0:
        all_rows.append({
            "case_id": case_id,
            "ws_count": ws_count,
            "nbatches": NBATCHES,
            "status": "PROFILE_FAIL",
            "failure_reason": stderr[-1000:],
        })
        continue

    stage_csv = profile_out / "stage_timings.csv"
    matrix_csv = profile_out / "profile_matrix_summary.csv"
    if not matrix_csv.exists():
        write_matrix_summary_from_stage_timings(stage_csv, matrix_csv, case_id)

    analysis_cmd = [
        "julia",
        "--project=.",
        "experiments/thesis_gpu_parallel/analyze_five_block_timing.jl",
        "--matrix-csv",
        str(matrix_csv),
        "--outdir",
        str(analysis_out),
    ]

    rc, analysis_wall_s, stdout, stderr = run_cmd(analysis_cmd, JC_ROOT)
    (analysis_out / "wrapper_stdout.log").write_text(stdout, encoding="utf-8")
    (analysis_out / "wrapper_stderr.log").write_text(stderr, encoding="utf-8")

    if rc != 0:
        all_rows.append({
            "case_id": case_id,
            "ws_count": ws_count,
            "nbatches": NBATCHES,
            "status": "ANALYSIS_FAIL",
            "failure_reason": stderr[-1000:],
        })
        continue

    five_rows = read_csv(analysis_out / "five_block_summary.csv")

    for r in five_rows:
        staged_total = fnum(r, "staged_total")
        public_hbsolve = fnum(r, "public_hbsolve_reference")

        out = {
            "case_id": r["case_id"],
            "ws_count": ws_count,
            "nbatches": NBATCHES,
            "status": "PASS",
            "python_subprocess_wall_s": python_wall_s,
            "analysis_wall_s": analysis_wall_s,
            "staged_total_s": staged_total,
            "public_hbsolve_reference_s": public_hbsolve,
            "staged_total_pct_python_subprocess_wall": pct(staged_total, python_wall_s),
            "public_hbsolve_pct_python_subprocess_wall": pct(public_hbsolve, python_wall_s),
            "failure_reason": "",
        }

        for b in block_names:
            val = fnum(r, b)
            out[b + "_s"] = val
            out[b + "_pct_staged"] = pct(val, staged_total)
            out[b + "_pct_public_hbsolve"] = pct(val, public_hbsolve)
            out[b + "_pct_python_subprocess_wall"] = pct(val, python_wall_s)

        all_rows.append(out)

csv_path = OUT / "wall_time_five_block_ws_matrix.csv"
fieldnames = sorted(set().union(*(row.keys() for row in all_rows)))
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

md = [
    "# Five-block wall-time matrix",
    "",
    "This uses the existing JosephsonCircuits staged profiler with larger `ws_count` values.",
    "",
    "The most useful denominator here is `% public hbsolve`, because Python/subprocess wall still includes Julia startup/script overhead.",
    "",
    "| ws_count | status | staged total s | public hbsolve s | B1 % hbsolve | B2 % hbsolve | B3 % hbsolve | B4 % hbsolve | B5 % hbsolve |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]

for row in all_rows:
    if row.get("status") != "PASS":
        md.append(f"| {row.get('ws_count')} | {row.get('status')} |  |  |  |  |  |  |  |")
        continue

    md.append(
        f"| {row['ws_count']} | PASS | "
        f"{row['staged_total_s']} | {row['public_hbsolve_reference_s']} | "
        f"{row['block1_frequency_bookkeeping_pct_public_hbsolve']} | "
        f"{row['block2_circuit_matrix_construction_pct_public_hbsolve']} | "
        f"{row['block3_nonlinear_pump_hb_solve_pct_public_hbsolve']} | "
        f"{row['block4_linearized_signal_idler_solve_pct_public_hbsolve']} | "
        f"{row['block5_port_output_conversion_outer_pct_public_hbsolve']} |"
    )

report_path = OUT / "wall_time_five_block_ws_matrix.md"
report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

manifest = {
    "ws_counts": WS_COUNTS,
    "repetitions": REPETITIONS,
    "nbatches": NBATCHES,
    "iterations": ITERATIONS,
    "csv": str(csv_path),
    "report": str(report_path),
}
(OUT / "wall_time_five_block_ws_matrix_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("WROTE", OUT)
print("REPORT", report_path)
print("CSV", csv_path)
