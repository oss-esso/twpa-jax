"""Run one gain-map frequency column and archive matrices created by the algorithms.

The solver contains matrix-free paths where important sparse matrices are short-lived
locals. This driver captures the sparse matrices that cross solver function
boundaries (a sys.setprofile call/return callback scoped to solver modules), while
copying the static design matrices. It is intentionally one-column-only.

Capture uses sys.setprofile (call/return events) rather than sys.settrace (per-line
events): line tracing the stiff assembly loops slowed one pump solve ~40x (2.2s ->
92s), whereas profiling scans locals only at function boundaries. The trade-off is
that a matrix built and discarded entirely within one function, without ever being a
local at a call/return boundary, is not archived.

The tracer is fail-safe: any error while saving a matrix (e.g. a Windows MAX_PATH
OSError on a deep --outdir) is logged and skipped so it can never abort the solve
(reported as matrix_save_errors / save_errors in matrix_index.json). Because the
setprofile callback is hot, a Ctrl-C also tends to surface inside it; that is a real
KeyboardInterrupt stopping the run, not a tracer bug. On Windows the archive paths
are deep -- prefer a short --outdir to stay under MAX_PATH and archive every matrix.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gain_map  # noqa: E402

# Production engine/solver config for the 2c gain map, mirroring the current
# production run_gain_map invocation. Directory-dependent flags, pump/frequency
# settings, grid-point counts (--n-power/--n-frequency) and --sidebands are
# intentionally excluded here: this one-column matrix-tracing driver controls
# those via its own CLI args (see parse_args / main). --frequency-chunk-size is
# likewise omitted because n_frequency == 1 keeps run_gain_map in-process (the
# sys.setprofile matrix tracer only sees the in-process solve, not chunk workers).
PRODUCTION_ENGINE_FLAGS: list[str] = [
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--inproc-fail-fast",
    "--fold-skip-patience", "2",
    "--inproc-schur-cache-size", "1",
    "--signal-detuning-mhz", "100",
    "--signal-backend", "direct",
    "--signal-solver", "superlu",
    "--pump-mode-count", "10",
    "--nt", "40",
    "--no-signal-spectrum",
]

def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)[:120]

class MatrixTracer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.seen: set[tuple[int, str, int]] = set()
        self.index: list[dict[str, Any]] = []
        # filename -> is-target-module, cached because sys.setprofile fires the
        # global dispatch once per Python call/return and filenames repeat heavily.
        self._target_cache: dict[str, bool] = {}
        # Count of matrices skipped due to a save error; capped logging below.
        self._save_errors = 0

    def _is_target(self, filename: str) -> bool:
        cached = self._target_cache.get(filename)
        if cached is None:
            normalized = filename.replace("\\", "/")
            cached = (
                "/experiments/" in normalized
                or "/src/twpa_solver/" in normalized
                or normalized.endswith("/scripts/run_gain_map.py")
            )
            self._target_cache[filename] = cached
        return cached

    def _context(self, frame) -> str:
        for name in ("pdir", "pump_dir", "gain_dir", "outdir", "pass_dir"):
            value = frame.f_locals.get(name)
            if value is not None:
                try:
                    path = Path(value)
                    if path.name:
                        return "_".join(_safe(part) for part in path.parts[-3:])
                except (TypeError, ValueError, OSError):
                    pass
        return "unscoped"

    def _save(self, value: Any, frame, line: int, name: str) -> None:
        if sp.issparse(value):
            lowered = name.lower()
            if not (name in {"A", "D", "K", "C", "G", "Bphi", "P", "M", "Mop", "Ktan"} or any(token in lowered for token in ("block", "khat", "jacobian", "schur", "matrix", "factor"))):
                return
            kind = "sparse"
        else:
            return
        key = (id(value), frame.f_code.co_filename, line)
        if key in self.seen:
            return
        self.seen.add(key)
        function = _safe(frame.f_code.co_name)
        directory = self.root / self._context(frame) / function / f"line_{line:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(name)}_{len(self.index):06d}.npz"
        if kind == "sparse":
            sp.save_npz(path, value.tocsr())
            shape, nnz = list(value.shape), int(value.nnz)
        else:
            np.savez_compressed(path, array=np.asarray(value))
            shape, nnz = list(value.shape), None
        self.index.append({
            "path": str(path.relative_to(self.root)),
            "kind": kind,
            "name": name,
            "function": frame.f_code.co_name,
            "source": str(Path(frame.f_code.co_filename).resolve()),
            "line": line,
            "shape": shape,
            "nnz": nnz,
            "context": self._context(frame),
        })

    def dispatch(self, frame, event, arg):
        # sys.setprofile global callback: fires on Python 'call'/'return' (and C
        # 'c_call'/'c_return'), NOT once per line. We scan a target frame's locals
        # only on 'return', when f_locals holds both the function's arguments and
        # its assembled results. This captures the sparse matrices flowing through
        # solver function boundaries at O(function calls) cost instead of the
        # O(lines executed) cost of sys.settrace line tracing, which slowed the
        # stiff assembly loops ~40x (one point's pump factor 2.2s -> 92s; setprofile
        # return-only -> ~6s). Trade-off vs line tracing: a matrix built and
        # discarded within one function, without ever being a local at that
        # function's return, is not archived; matrices that survive to a return
        # (results, and arguments still bound) still are.
        if event != "return":
            return
        if not self._is_target(frame.f_code.co_filename):
            return
        # This callback runs inside the solve; a raise here would abort the whole
        # run (the setprofile hook is hot, so it is also where a Ctrl-C or a
        # Windows MAX_PATH OSError on a deep --outdir tends to surface). Catch
        # Exception per matrix, log a capped warning, and keep going, so the
        # diagnostic archiver can never crash the computation it observes. Only
        # Exception is caught: KeyboardInterrupt/SystemExit still stop the run.
        for name, value in list(frame.f_locals.items()):
            try:
                self._save(value, frame, frame.f_lineno, name)
            except Exception as exc:
                self._save_errors += 1
                if self._save_errors <= 10:
                    logger.warning(
                        "matrix_tracer_save_skipped function=%s line=%s name=%s error=%r",
                        frame.f_code.co_name, frame.f_lineno, name, exc,
                    )

def copy_static_design(design_dir: Path, outdir: Path) -> None:
    target = outdir / "static_design"
    target.mkdir(parents=True, exist_ok=True)
    for path in design_dir.iterdir():
        if path.is_file() and path.suffix in {".npz", ".json", ".csv"}:
            shutil.copy2(path, target / path.name)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "gain_map_ipm_2c_fixed_one_column_matrices")
    parser.add_argument("--column-frequency-ghz", type=float, default=7.9)
    parser.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    parser.add_argument("--pump-power-max-dbm", type=float, default=-20.0)
    parser.add_argument("--n-power", type=int, default=50)
    parser.add_argument("--sidebands", type=int, default=6)
    parser.add_argument("--extra-map-args", nargs=argparse.REMAINDER)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    copy_static_design(args.design_dir.resolve(), outdir)
    trace_root = outdir / "matrix_trace"
    tracer = MatrixTracer(trace_root)
    map_args = [
        "--executor", "inprocess", "--mode", "warmstart", "--traversal", "column",
        "--circuit-dir", str(args.design_dir.resolve()), "--outdir", str(outdir / "map"),
        "--n-power", str(args.n_power), "--n-frequency", "1",
        "--pump-power-min-dbm", str(args.pump_power_min_dbm),
        "--pump-power-max-dbm", str(args.pump_power_max_dbm),
        "--pump-freq-min-ghz", str(args.column_frequency_ghz),
        "--pump-freq-max-ghz", str(args.column_frequency_ghz),
        "--sidebands", str(args.sidebands),
        *PRODUCTION_ENGINE_FLAGS,
        "--overwrite",
    ]
    if args.extra_map_args:
        map_args.extend(args.extra_map_args)
    sys.setprofile(tracer.dispatch)
    try:
        rc = run_gain_map.main(map_args)
    finally:
        sys.setprofile(None)
        (trace_root / "matrix_index.json").write_text(json.dumps({
            "description": "Matrices observed in solver frames during one frequency-column run.",
            "column_frequency_ghz": args.column_frequency_ghz,
            "map_args": map_args,
            "save_errors": tracer._save_errors,
            "matrices": tracer.index,
        }, indent=2), encoding="utf-8")
    print(f"matrix_archive={outdir}")
    print(f"matrix_count={len(tracer.index)}")
    print(f"matrix_save_errors={tracer._save_errors}")
    print(f"matrix_index={trace_root / 'matrix_index.json'}")
    return int(rc)

if __name__ == "__main__":
    raise SystemExit(main())

