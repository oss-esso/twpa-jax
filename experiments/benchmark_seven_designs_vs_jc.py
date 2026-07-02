"""Native-Julia vs Python runtime table for the 7 JC doc designs.

Pairs the Python harmonic-balance pump+gain runtime (from the
``exp14_seven_design_summary`` outputs) against a native ``JosephsonCircuits.jl``
``hbsolve`` timing for the same design, and emits a single comparison table.

Python side (always runs here):
    Reads ``pump_runtime_s`` + ``gain_runtime_s`` per case from
    ``outputs/exp14_seven_design_summary/summary.csv``. Run
    ``python experiments/exp14_seven_design_summary.py`` first to refresh it.

Julia side (needs the user's local Harmonia.jl case builders):
    For each case, invokes ``exp14_jc_doc_curve_dump.jl`` at a SINGLE frequency
    point (so the JC time is one pump solve + one linearized gain point, matching
    the Python single-point timing) and parses ``runtime_s=``. A warmup call is
    made first so JIT compilation is excluded. Requires ``julia`` on PATH and
    ``JC_DOCS_CASES_DIR`` pointing at the ``jc_docs`` case builders (defaults to
    the ``D:\\Projects\\Thesis\\Harmonia.jl`` checkout baked into the .jl). Pass
    ``--julia-project`` for the JC.jl environment. If Julia or the cases dir is
    unavailable the JC column is left ``n/a`` and only the Python table is
    written -- nothing is fabricated.

Example (full comparison on the machine that has Harmonia.jl):
    JC_DOCS_CASES_DIR=D:/Projects/Thesis/Harmonia.jl/experiments/solver_benchmark/cases/jc_docs \\
    python experiments/benchmark_seven_designs_vs_jc.py \\
        --julia-project D:/Projects/Thesis/JosephsonCircuits.jl
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JC_DUMP = ROOT / "experiments" / "exp14_jc_doc_curve_dump.jl"
SUMMARY_CSV = ROOT / "outputs" / "exp14_seven_design_summary" / "summary.csv"

# Python case name (as in the exp14 summary) -> native JC builder short name.
# The JC name is the case with the leading ``jc_`` stripped.
CASE_ORDER = [
    "jc_jpa",
    "jc_jtwpa",
    "jc_fqjtwpa",
    "jc_fxjpa",
    "jc_fxjtwpa",
    "jc_dpjpa",
    "jc_fqjtwpa_diss",
]


@dataclass
class Row:
    case: str
    status: str
    py_pump_s: float | None
    py_gain_s: float | None
    py_total_s: float | None
    jc_total_s: float | None
    py_max_db: str
    jc_max_db: str

    @property
    def speedup_jc_over_py(self) -> float | None:
        if self.jc_total_s and self.py_total_s:
            return self.jc_total_s / self.py_total_s
        return None


def _f(v: object) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def read_python_timings() -> dict[str, dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise SystemExit(
            f"missing {SUMMARY_CSV}; run "
            "`python experiments/exp14_seven_design_summary.py` first."
        )
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def peak_frequency_ghz(case: str) -> float | None:
    """Single-point sweep frequency for the JC timing: the Python gain peak.

    Read from the case's gain report if present, else None (skip JC timing)."""
    # The summary stores py_peak (frequency of the Python gain max).
    return None  # filled from summary py_peak in build_rows


def run_julia_single_point(
    jc_name: str,
    freq_ghz: float,
    julia: str,
    julia_project: str | None,
    cases_dir: str | None,
    tmp_csv: Path,
    timeout_s: float,
) -> float | None:
    """Time one native hbsolve (1 freq point) after a JIT warmup; parse runtime_s."""
    env = dict(os.environ)
    if cases_dir:
        env["JC_DOCS_CASES_DIR"] = cases_dir
    cmd = [julia]
    if julia_project:
        cmd.append(f"--project={julia_project}")
    # Two solves in one process: warmup then measured (JIT excluded on the 2nd).
    # exp14_jc_doc_curve_dump.jl times a single invocation, so call it twice and
    # keep the second runtime_s.
    runtimes: list[float] = []
    for i in range(2):
        out = tmp_csv.parent / f"{jc_name}_jc_pt_{i}.csv"
        run_cmd = cmd + [
            str(JC_DUMP), jc_name,
            f"{freq_ghz:.6f}", f"{freq_ghz:.6f}", "1", str(out),
        ]
        try:
            p = subprocess.run(
                run_cmd, cwd=str(ROOT), env=env, text=True,
                capture_output=True, timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  [{jc_name}] julia TIMEOUT after {timeout_s}s", flush=True)
            return None
        if p.returncode != 0:
            print(f"  [{jc_name}] julia failed (rc={p.returncode}):\n"
                  f"{p.stderr.strip()[-800:]}", flush=True)
            return None
        for line in p.stdout.splitlines():
            if line.startswith("runtime_s="):
                runtimes.append(float(line.split("=", 1)[1]))
    return runtimes[-1] if runtimes else None


def build_rows(
    py: dict[str, dict[str, str]],
    *,
    do_julia: bool,
    julia: str,
    julia_project: str | None,
    cases_dir: str | None,
    tmp_dir: Path,
    timeout_s: float,
) -> list[Row]:
    rows: list[Row] = []
    for case in CASE_ORDER:
        r = py.get(case)
        if r is None:
            continue
        pump = _f(r.get("pump_runtime_s"))
        gain = _f(r.get("gain_runtime_s"))
        total = None if (pump is None and gain is None) else (pump or 0.0) + (gain or 0.0)
        jc_total = None
        if do_julia:
            freq = _f(r.get("py_peak"))
            if freq is not None:
                jc_name = case[3:] if case.startswith("jc_") else case
                print(f"[{case}] native JC hbsolve @ {freq:.4g} GHz ...", flush=True)
                jc_total = run_julia_single_point(
                    jc_name, freq, julia, julia_project, cases_dir,
                    tmp_dir / f"{case}.csv", timeout_s,
                )
        rows.append(Row(
            case=case,
            status=r.get("status", ""),
            py_pump_s=pump,
            py_gain_s=gain,
            py_total_s=total,
            jc_total_s=jc_total,
            py_max_db=r.get("py_max", ""),
            jc_max_db=r.get("jc_max", ""),
        ))
    return rows


def fmt(v: float | None, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, float) else "n/a"


def write_table(rows: list[Row], out_md: Path, out_csv: Path) -> None:
    hdr = ["case", "status", "py_pump_s", "py_gain_s", "py_total_s",
           "jc_total_s", "jc/py", "py_max_db", "jc_max_db"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow([
                r.case, r.status, fmt(r.py_pump_s), fmt(r.py_gain_s),
                fmt(r.py_total_s), fmt(r.jc_total_s),
                fmt(r.speedup_jc_over_py, 2), r.py_max_db, r.jc_max_db,
            ])

    lines = [
        "# Seven-design runtime: Python HB vs native JosephsonCircuits.jl",
        "",
        "Single-point pump+gain solve per design. `jc/py` > 1 means Python is "
        "faster. JC times exclude JIT (second of two warmup solves). Python "
        "times are from `outputs/exp14_seven_design_summary/summary.csv`.",
        "",
        "| design | status | py pump (s) | py gain (s) | py total (s) | "
        "JC total (s) | JC/py | py max (dB) | JC max (dB) |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.status} | {fmt(r.py_pump_s)} | {fmt(r.py_gain_s)} "
            f"| {fmt(r.py_total_s)} | {fmt(r.jc_total_s)} | "
            f"{fmt(r.speedup_jc_over_py, 2)} | {r.py_max_db} | {r.jc_max_db} |"
        )
    if all(r.jc_total_s is None for r in rows):
        lines += [
            "",
            "> JC column is `n/a`: `julia` or `JC_DOCS_CASES_DIR` was not "
            "available. Re-run on the machine with the Harmonia.jl case "
            "builders and pass `--julia-project`.",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path,
                   default=ROOT / "outputs" / "benchmark_seven_vs_jc")
    p.add_argument("--julia", type=str, default="julia",
                   help="julia executable (default: julia on PATH).")
    p.add_argument("--julia-project", type=str, default=None,
                   help="--project for the JosephsonCircuits.jl environment.")
    p.add_argument("--cases-dir", type=str,
                   default=os.environ.get("JC_DOCS_CASES_DIR"),
                   help="jc_docs case-builder dir (or set JC_DOCS_CASES_DIR).")
    p.add_argument("--no-julia", action="store_true",
                   help="Skip the native JC timing; emit the Python column only.")
    p.add_argument("--julia-timeout-s", type=float, default=1800.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.outdir / "jc_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    py = read_python_timings()

    do_julia = not args.no_julia
    if do_julia and shutil.which(args.julia) is None:
        print(f"julia executable '{args.julia}' not found on PATH; "
              "emitting Python-only table.", flush=True)
        do_julia = False
    if do_julia and not args.cases_dir:
        # Fall back to the .jl default path; warn if it does not exist.
        default = Path(r"D:\Projects\Thesis\Harmonia.jl"
                       r"\experiments\solver_benchmark\cases\jc_docs")
        if not default.exists():
            print("no --cases-dir / JC_DOCS_CASES_DIR and the baked-in default "
                  f"({default}) is absent; emitting Python-only table.",
                  flush=True)
            do_julia = False

    rows = build_rows(
        py, do_julia=do_julia, julia=args.julia,
        julia_project=args.julia_project, cases_dir=args.cases_dir,
        tmp_dir=tmp_dir, timeout_s=args.julia_timeout_s,
    )

    out_md = args.outdir / "seven_vs_jc.md"
    out_csv = args.outdir / "seven_vs_jc.csv"
    write_table(rows, out_md, out_csv)
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    for r in rows:
        print(f"  {r.case:16s} py_total={fmt(r.py_total_s):>8s}s "
              f"jc_total={fmt(r.jc_total_s):>8s}s "
              f"jc/py={fmt(r.speedup_jc_over_py, 2)}")


if __name__ == "__main__":
    main()
