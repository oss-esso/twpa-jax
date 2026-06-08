from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(r"D:\Projects\Thesis")
JC_ROOT = WORKSPACE / "JosephsonCircuits.jl"
OUT = WORKSPACE / "outputs" / "jc_profiles" / "jc3m_wall_time_five_block_probe"
PROFILE_OUT = OUT / "raw_profile"
ANALYSIS_OUT = OUT / "five_block_analysis"

for p in (OUT, PROFILE_OUT, ANALYSIS_OUT):
    p.mkdir(parents=True, exist_ok=True)


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


def write_matrix_summary_from_stage_timings(stage_csv: Path, matrix_csv: Path) -> None:
    rows = read_csv(stage_csv)
    groups: dict[str, list[float]] = {}

    for r in rows:
        stage = r["stage"]
        groups.setdefault(stage, []).append(float(r["time_s"]))

    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "stage", "time_s_mean"])
        writer.writeheader()
        for stage, vals in sorted(groups.items()):
            writer.writerow({
                "case_id": "same_run_probe",
                "stage": stage,
                "time_s_mean": sum(vals) / len(vals),
            })


def fnum(row: dict[str, str], key: str) -> float:
    x = row.get(key, "")
    return 0.0 if x == "" else float(x)


def pct(num: float, den: float):
    return None if den <= 0 else 100.0 * num / den


profile_cmd = [
    "julia",
    "--project=.",
    "experiments/thesis_gpu_parallel/run_hbsolve_profile.jl",
    "--outdir",
    str(PROFILE_OUT),
    "--repetitions",
    "3",
    "--profile-repetitions",
    "0",
]

rc, python_wall_s, stdout, stderr = run_cmd(profile_cmd, JC_ROOT)
(PROFILE_OUT / "wrapper_stdout.log").write_text(stdout, encoding="utf-8")
(PROFILE_OUT / "wrapper_stderr.log").write_text(stderr, encoding="utf-8")

if rc != 0:
    raise SystemExit(
        "run_hbsolve_profile.jl failed\n"
        f"rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

stage_csv = PROFILE_OUT / "stage_timings.csv"
matrix_csv = PROFILE_OUT / "profile_matrix_summary.csv"

if not matrix_csv.exists():
    if not stage_csv.exists():
        raise SystemExit(f"Missing both {matrix_csv} and {stage_csv}")
    write_matrix_summary_from_stage_timings(stage_csv, matrix_csv)

analysis_cmd = [
    "julia",
    "--project=.",
    "experiments/thesis_gpu_parallel/analyze_five_block_timing.jl",
    "--matrix-csv",
    str(matrix_csv),
    "--outdir",
    str(ANALYSIS_OUT),
]

rc, analysis_wall_s, stdout, stderr = run_cmd(analysis_cmd, JC_ROOT)
(ANALYSIS_OUT / "wrapper_stdout.log").write_text(stdout, encoding="utf-8")
(ANALYSIS_OUT / "wrapper_stderr.log").write_text(stderr, encoding="utf-8")

if rc != 0:
    raise SystemExit(
        "analyze_five_block_timing.jl failed\n"
        f"rc={rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )

five_csv = ANALYSIS_OUT / "five_block_summary.csv"
five_rows = read_csv(five_csv)

block_names = [
    "block1_frequency_bookkeeping",
    "block2_circuit_matrix_construction",
    "block3_nonlinear_pump_hb_solve",
    "block4_linearized_signal_idler_solve",
    "block5_port_output_conversion_outer",
]

labels = {
    "block1_frequency_bookkeeping": "1 frequency bookkeeping",
    "block2_circuit_matrix_construction": "2 circuit/matrix setup",
    "block3_nonlinear_pump_hb_solve": "3 nonlinear pump solve",
    "block4_linearized_signal_idler_solve": "4 linearized solve",
    "block5_port_output_conversion_outer": "5 output conversion",
}

budget_rows = []

for r in five_rows:
    staged_total = fnum(r, "staged_total")
    public_hbsolve = fnum(r, "public_hbsolve_reference")

    row = {
        "case_id": r["case_id"],
        "python_wall_time_s": python_wall_s,
        "analysis_wall_time_s": analysis_wall_s,
        "staged_total_s": staged_total,
        "public_hbsolve_reference_s": public_hbsolve,
        "staged_total_pct_python_wall": pct(staged_total, python_wall_s),
        "public_hbsolve_pct_python_wall": pct(public_hbsolve, python_wall_s),
        "unattributed_python_wall_s": python_wall_s - staged_total,
        "unattributed_python_wall_pct": pct(python_wall_s - staged_total, python_wall_s),
    }

    for b in block_names:
        val = fnum(r, b)
        row[b + "_s"] = val
        row[b + "_pct_staged"] = pct(val, staged_total)
        row[b + "_pct_public_hbsolve"] = pct(val, public_hbsolve)
        row[b + "_pct_python_wall"] = pct(val, python_wall_s)

    budget_rows.append(row)

budget_csv = OUT / "wall_time_five_block_budget.csv"
with budget_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
    writer.writeheader()
    writer.writerows(budget_rows)

lines = [
    "# Same-run five-block wall-time budget",
    "",
    f"- Python/subprocess wall time: `{python_wall_s}` s",
    f"- Analysis wall time: `{analysis_wall_s}` s",
    "",
    "Python wall time includes Julia startup and script overhead. The staged HB percentages are the cleaner solver-internal denominator.",
    "",
    "| case | block | seconds | % staged HB | % public hbsolve | % Python wall |",
    "|---|---|---:|---:|---:|---:|",
]

for row in budget_rows:
    for b in block_names:
        lines.append(
            f"| {row['case_id']} | {labels[b]} | "
            f"{row[b + '_s']} | "
            f"{row[b + '_pct_staged']} | "
            f"{row[b + '_pct_public_hbsolve']} | "
            f"{row[b + '_pct_python_wall']} |"
        )

lines += [
    "",
    "## Accounting",
    "",
    "| case | staged total s | public hbsolve ref s | staged % Python wall | public hbsolve % Python wall | unattributed Python wall % |",
    "|---|---:|---:|---:|---:|---:|",
]

for row in budget_rows:
    lines.append(
        f"| {row['case_id']} | {row['staged_total_s']} | "
        f"{row['public_hbsolve_reference_s']} | "
        f"{row['staged_total_pct_python_wall']} | "
        f"{row['public_hbsolve_pct_python_wall']} | "
        f"{row['unattributed_python_wall_pct']} |"
    )

report_md = OUT / "wall_time_five_block_budget.md"
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "profile_command": profile_cmd,
    "analysis_command": analysis_cmd,
    "python_wall_time_s": python_wall_s,
    "analysis_wall_time_s": analysis_wall_s,
    "profile_out": str(PROFILE_OUT),
    "analysis_out": str(ANALYSIS_OUT),
    "budget_csv": str(budget_csv),
    "report_md": str(report_md),
}

(OUT / "wall_time_five_block_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("WROTE", OUT)
print("REPORT", report_md)
print("CSV", budget_csv)
