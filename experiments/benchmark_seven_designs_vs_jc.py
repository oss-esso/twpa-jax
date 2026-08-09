"""Native-Julia vs Python runtime table for the 7 JC doc designs.

Pairs the Python harmonic-balance pump+gain runtime (from the
``exp14_seven_design_summary`` outputs) against native ``JosephsonCircuits.jl``
``hbsolve`` timing for the same design, and emits a combined table (CSV + md).

Python side (always runs here):
    Reads ``pump_runtime_s`` + ``gain_runtime_s`` per case from
    ``outputs/exp14_seven_design_summary/summary.csv`` (single signal point). Run
    ``python experiments/exp14_seven_design_summary.py`` first to refresh it.

JC side (needs julia + JosephsonCircuits):
    Runs each design's raw doc standalone (``.../jc_docs/raw/<DESIGN>.jl``) via
    ``experiments/jc_raw_timing.jl``, which strips plotting and reports the WARM
    (JIT-excluded) ``hbsolve`` time. The raw standalones sweep the full doc
    signal band, so the JC number is a full-sweep solve, not a single point --
    see the ``sweep`` caveat in the emitted table. ``jc_fqjtwpa_diss`` reuses the
    ``jc_fqjtwpa`` circuit, so its base file is prepended and only the
    dissipative solves are counted.

    Point ``--julia-project`` at an env with JosephsonCircuits (the local
    Harmonia.jl checkout has it + Plots) and ``--raw-dir`` at the ``raw`` folder.
    If julia is unavailable the JC column is left ``n/a`` -- nothing is fabricated.

Example:
    python experiments/benchmark_seven_designs_vs_jc.py \\
        --julia-project C:/Users/Edoardo/Documents/EPFL/Thesis/Harmonia.jl \\
        --raw-dir C:/Users/Edoardo/Documents/EPFL/Thesis/Harmonia.jl/experiments/solver_benchmark/cases/jc_docs/raw
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JC_TIMING = ROOT / "experiments" / "jc_raw_timing.jl"
SUMMARY_CSV = ROOT / "outputs" / "exp14_seven_design_summary" / "summary.csv"

# Python case -> raw standalone file(s) (order matters; a continuation variant is
# prepended with its base) and how many leading hbsolve calls belong to the base
# (dropped so only the variant's solves count).
CASES: list[tuple[str, list[str], int]] = [
    ("jc_jpa", ["JPA.jl"], 0),
    ("jc_jtwpa", ["JTWPA.jl"], 0),
    ("jc_fqjtwpa", ["FQJTWPA.jl"], 0),
    ("jc_fxjpa", ["FXJPA.jl"], 0),
    ("jc_fxjtwpa", ["FXJTWPA.jl"], 0),
    ("jc_dpjpa", ["DPJPA.jl"], 0),
    ("jc_fqjtwpa_diss", ["FQJTWPA.jl", "FQJTWPA_diss.jl"], 1),
]


@dataclass
class Row:
    case: str
    status: str
    py_pump_s: float | None
    py_gain_s: float | None
    py_total_s: float | None
    jc_runtime_s: float | None
    jc_solves: int
    py_max_db: str
    jc_note: str = ""

    @property
    def speedup_jc_over_py(self) -> float | None:
        if self.jc_runtime_s and self.py_total_s:
            return self.jc_runtime_s / self.py_total_s
        return None


def _f(v: object) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_python_timings() -> dict[str, dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise SystemExit(
            f"missing {SUMMARY_CSV}; run "
            "`python experiments/exp14_seven_design_summary.py` first."
        )
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def run_jc_timing(
    raw_files: list[Path],
    drop_leading: int,
    julia: str,
    julia_project: str | None,
    timeout_s: float,
) -> tuple[float | None, int]:
    """Return (warm hbsolve seconds counted, n_solves_counted)."""
    cmd = [julia]
    if julia_project:
        cmd.append(f"--project={julia_project}")
    cmd += [str(JC_TIMING), *[str(p) for p in raw_files]]
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), text=True, capture_output=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  julia TIMEOUT after {timeout_s}s", flush=True)
        return None, 0
    if p.returncode != 0:
        print(f"  julia failed (rc={p.returncode}):\n{p.stderr.strip()[-1200:]}",
              flush=True)
        return None, 0
    times: list[float] = []
    for line in p.stdout.splitlines():
        if line.startswith("hbsolve_times_s="):
            payload = line.split("=", 1)[1].strip()
            times = [float(x) for x in payload.split(",") if x]
    counted = times[drop_leading:] if drop_leading else times
    if not counted:
        return None, 0
    return sum(counted), len(counted)


def build_rows(
    py: dict[str, dict[str, str]], *, do_julia: bool, julia: str,
    julia_project: str | None, raw_dir: Path, timeout_s: float,
) -> list[Row]:
    rows: list[Row] = []
    for case, files, drop in CASES:
        r = py.get(case, {})
        pump = _f(r.get("pump_runtime_s"))
        gain = _f(r.get("gain_runtime_s"))
        total = None if (pump is None and gain is None) else (pump or 0.0) + (gain or 0.0)
        jc_s: float | None = None
        jc_n = 0
        note = ""
        if do_julia:
            raw_files = [raw_dir / f for f in files]
            missing = [f for f in raw_files if not f.exists()]
            if missing:
                note = f"missing {', '.join(m.name for m in missing)}"
            else:
                print(f"[{case}] native JC hbsolve ({'+'.join(files)}) ...",
                      flush=True)
                jc_s, jc_n = run_jc_timing(
                    raw_files, drop, julia, julia_project, timeout_s)
                if drop:
                    note = f"{jc_n} dissipative solves (base circuit excluded)"
                elif jc_n > 1:
                    note = f"{jc_n} hbsolve calls summed"
        rows.append(Row(
            case=case, status=r.get("status", ""),
            py_pump_s=pump, py_gain_s=gain, py_total_s=total,
            jc_runtime_s=jc_s, jc_solves=jc_n,
            py_max_db=r.get("py_max", ""), jc_note=note,
        ))
    return rows


def fmt(v: float | None, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, float) else "n/a"


def write_table(rows: list[Row], out_md: Path, out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "status", "py_pump_s", "py_gain_s", "py_total_s",
                    "jc_runtime_s", "jc_solves", "jc_over_py", "py_max_db", "note"])
        for r in rows:
            w.writerow([r.case, r.status, fmt(r.py_pump_s), fmt(r.py_gain_s),
                        fmt(r.py_total_s), fmt(r.jc_runtime_s), r.jc_solves,
                        fmt(r.speedup_jc_over_py, 2), r.py_max_db, r.jc_note])

    lines = [
        "# Seven-design runtime: Python HB vs native JosephsonCircuits.jl",
        "",
        "Python = single signal point (exp14 pump+gain). JC = raw doc standalone, "
        "WARM (JIT-excluded) `hbsolve`, plotting stripped. **Caveat:** the JC doc "
        "standalones sweep the full signal band (many points), so `jc_runtime_s` "
        "is a full-sweep solve, not a single point -- the pump solve is shared "
        "across the sweep, the per-frequency linear cost is small. `jc/py` < 1 "
        "means JC is faster.",
        "",
        "| design | status | py pump (s) | py gain (s) | py total (s) | "
        "JC hbsolve (s) | jc/py | note |",
        "|---|---|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.case} | {r.status} | {fmt(r.py_pump_s)} | {fmt(r.py_gain_s)} | "
            f"{fmt(r.py_total_s)} | {fmt(r.jc_runtime_s)} | "
            f"{fmt(r.speedup_jc_over_py, 2)} | {r.jc_note} |"
        )
    if all(r.jc_runtime_s is None for r in rows):
        lines += ["", "> JC column `n/a`: julia/JosephsonCircuits or --raw-dir "
                  "unavailable."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path,
                   default=ROOT / "outputs" / "benchmark_seven_vs_jc")
    p.add_argument("--julia", type=str, default="julia")
    p.add_argument("--julia-project", type=str, default=None,
                   help="--project env that has JosephsonCircuits (e.g. Harmonia.jl).")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="jc_docs/raw folder with the design standalones.")
    p.add_argument("--no-julia", action="store_true")
    p.add_argument("--julia-timeout-s", type=float, default=1800.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    py = read_python_timings()

    do_julia = not args.no_julia
    if do_julia and shutil.which(args.julia) is None:
        print(f"julia '{args.julia}' not on PATH; Python-only table.", flush=True)
        do_julia = False
    if do_julia and args.raw_dir is None:
        print("no --raw-dir given; Python-only table.", flush=True)
        do_julia = False

    rows = build_rows(
        py, do_julia=do_julia, julia=args.julia,
        julia_project=args.julia_project,
        raw_dir=args.raw_dir or Path("."), timeout_s=args.julia_timeout_s,
    )

    out_md = args.outdir / "seven_vs_jc.md"
    out_csv = args.outdir / "seven_vs_jc.csv"
    write_table(rows, out_md, out_csv)
    print(f"wrote {out_csv}\nwrote {out_md}")
    for r in rows:
        print(f"  {r.case:16s} py_total={fmt(r.py_total_s):>8s}s "
              f"jc={fmt(r.jc_runtime_s):>9s}s jc/py={fmt(r.speedup_jc_over_py, 2)}"
              + (f"  [{r.jc_note}]" if r.jc_note else ""))


if __name__ == "__main__":
    main()
