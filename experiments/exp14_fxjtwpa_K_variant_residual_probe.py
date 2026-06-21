from __future__ import annotations

from pathlib import Path
import importlib.util
import inspect
import json
import shutil
import sys
import numpy as np

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
VARIANT_ROOT = ROOT / "outputs" / "exp14_fxjtwpa_K_variants"
SEED_ROOT = ROOT / "outputs" / "exp14_fxjtwpa_jcseed"
OUT = ROOT / "outputs" / "exp14_fxjtwpa_K_variant_residuals"
OUT.mkdir(parents=True, exist_ok=True)

PUMP_FILES = [
    SEED_ROOT / "pump" / "pump_solution.npz",
    SEED_ROOT / "pump_solved" / "pump_solution.npz",
]

DC_FILE = SEED_ROOT / "dc" / "dc_solution.npz"

VARIANTS = [
    "baseline_copy",
    "flip_big_delta4_offdiag",
    "scale_big_mutual_by_denom",
    "flip_and_scale_big_mutual",
]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def summarize_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    d = {"path": str(path), "keys": list(z.files)}
    for k in z.files:
        a = z[k]
        d[k] = {
            "shape": tuple(a.shape),
            "dtype": str(a.dtype),
            "absmax": float(np.max(np.abs(a))) if a.size and a.dtype.kind in "fcbiu" else None,
        }
    return d


def call_with_supported_kwargs(fn, kwargs: dict):
    sig = inspect.signature(fn)
    call = {}
    for name, p in sig.parameters.items():
        if name in kwargs:
            call[name] = kwargs[name]
    return fn(**call)


def norm_report(res) -> dict:
    if isinstance(res, tuple):
        # Common patterns: (F, info) or (status, F, info)
        arrays = [x for x in res if hasattr(x, "shape") and np.asarray(x).dtype.kind in "fcbiu"]
        if arrays:
            r = np.asarray(arrays[0]).ravel()
        else:
            r = np.asarray(res[0]).ravel()
    elif hasattr(res, "residual"):
        r = np.asarray(res.residual).ravel()
    elif isinstance(res, dict):
        for key in ["residual", "F", "r", "resid"]:
            if key in res:
                r = np.asarray(res[key]).ravel()
                break
        else:
            # Already a report-ish dict.
            return {"raw_keys": list(res.keys())}
    else:
        r = np.asarray(res).ravel()

    finite = r[np.isfinite(r)]
    return {
        "length": int(r.size),
        "finite": int(finite.size),
        "inf": float(np.max(np.abs(finite))) if finite.size else float("inf"),
        "l2": float(np.linalg.norm(finite)) if finite.size else float("inf"),
        "mean_abs": float(np.mean(np.abs(finite))) if finite.size else float("inf"),
    }


def main() -> None:
    m = import_module(EXP08, "exp08_runtime")

    print("=== exp08 callable inventory ===")
    candidates = []
    for name, obj in sorted(vars(m).items()):
        if callable(obj):
            low = name.lower()
            if any(k in low for k in ["load", "resid", "problem", "pump", "ipm", "assemble", "aft", "harmonic"]):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "<?>"
                print(f"{name}{sig}")
                candidates.append(name)

    print("\n=== seed summaries ===")
    for p in PUMP_FILES + [DC_FILE]:
        if p.exists():
            print(json.dumps(summarize_npz(p), indent=2))
        else:
            print("MISSING", p)

    # Preferred explicit residual function names. We try these in order.
    residual_names = [
        "pump_residual_from_solution",
        "evaluate_pump_residual",
        "compute_pump_residual",
        "residual_from_pump_solution",
        "residual_at_solution",
        "build_and_evaluate_pump_residual",
    ]

    load_ipm_names = [
        "load_ipm",
        "load_ipm_matrices",
        "load_problem",
        "load_ipm_problem",
        "load_design",
    ]

    load_pump_names = [
        "load_pump_solution",
        "load_pump",
        "load_solution",
    ]

    residual_fn = None
    for name in residual_names:
        if hasattr(m, name):
            residual_fn = getattr(m, name)
            print("\nUSING residual function", name, inspect.signature(residual_fn))
            break

    load_ipm_fn = None
    for name in load_ipm_names:
        if hasattr(m, name):
            load_ipm_fn = getattr(m, name)
            print("USING ipm loader", name, inspect.signature(load_ipm_fn))
            break

    load_pump_fn = None
    for name in load_pump_names:
        if hasattr(m, name):
            load_pump_fn = getattr(m, name)
            print("USING pump loader", name, inspect.signature(load_pump_fn))
            break

    if residual_fn is None:
        raise SystemExit(
            "No obvious residual function found in exp08. Paste the callable inventory above; "
            "we will bind to the real function name."
        )

    rows = []

    for variant in VARIANTS:
        ipm_dir = VARIANT_ROOT / variant
        if not ipm_dir.exists():
            print("MISSING variant", ipm_dir)
            continue

        for pump_file in PUMP_FILES:
            if not pump_file.exists():
                continue

            print("\n" + "=" * 100)
            print("VARIANT", variant)
            print("PUMP", pump_file)
            print("=" * 100)

            kwargs = {
                "ipm_dir": str(ipm_dir),
                "design_dir": str(ipm_dir),
                "pump_solution": str(pump_file),
                "pump_solution_path": str(pump_file),
                "pump_file": str(pump_file),
                "solution_path": str(pump_file),
                "dc_solution": str(DC_FILE) if DC_FILE.exists() else None,
                "dc_solution_path": str(DC_FILE) if DC_FILE.exists() else None,
                "harmonics": 4,
                "nt": 64,
                "rtol": 1e-8,
                "atol": 1e-10,
                "jvp_mode": "aft",
            }

            try:
                res = call_with_supported_kwargs(residual_fn, kwargs)
                rep = norm_report(res)
                print("REPORT", rep)
                rows.append({
                    "variant": variant,
                    "pump_file": str(pump_file),
                    "status": "OK",
                    **rep,
                })
            except Exception as e:
                print("FAILED", type(e).__name__, str(e))
                rows.append({
                    "variant": variant,
                    "pump_file": str(pump_file),
                    "status": f"FAIL:{type(e).__name__}",
                    "error": str(e),
                })

    out_csv = OUT / "summary.csv"
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else ["empty"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nWROTE", out_csv)
    print("\nSUMMARY")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
