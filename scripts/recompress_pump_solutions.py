"""Recompress existing ``pump_solution.npz`` files to float32 + DEFLATE.

The pump solver now writes solutions as compressed float32 (see
``twpa_solver.pump.io.write_results``), but maps produced before that change
stored float64 uncompressed (~1.5 MB/point, ~15 GB per 100x100 map). This walks
an output tree and rewrites each legacy solution in place, halving the store
without re-solving. float32's ~1e-7 relative precision is far below any
downstream tolerance (gain-map RMS targets ~1e-3 dB).

Idempotent: files already float32 + DEFLATE are skipped. Dry-run by default;
pass ``--apply`` to actually rewrite.

    python scripts/recompress_pump_solutions.py outputs            # preview
    python scripts/recompress_pump_solutions.py outputs --apply    # rewrite
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def is_already_optimized(path: Path) -> bool:
    """True if every array is float32/compressed or a small int index array."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if not infos:
                return False
            if any(i.compress_type != zipfile.ZIP_DEFLATED for i in infos):
                return False
        with np.load(path) as z:
            for key in ("X_real", "X_imag"):
                if key in z.files and z[key].dtype != np.float32:
                    return False
        return True
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def recompress(path: Path) -> None:
    """Rewrite one solution atomically as float32 + DEFLATE."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for key in ("X_real", "X_imag"):
        if key in arrays and arrays[key].dtype != np.float32:
            arrays[key] = arrays[key].astype(np.float32)

    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        # np.savez_compressed appends .npz to a path without one; mkstemp gave a
        # .npz suffix, so the written file matches tmp exactly.
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="directory to walk for pump_solution.npz")
    p.add_argument("--apply", action="store_true",
                   help="actually rewrite files (default: dry-run preview)")
    p.add_argument("--name", default="pump_solution.npz",
                   help="filename to match (default: pump_solution.npz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.root.rglob(args.name))
    if not files:
        print(f"no {args.name} under {args.root}")
        return

    before_total = 0
    after_total = 0
    skipped = 0
    changed = 0
    for i, path in enumerate(files, 1):
        before = path.stat().st_size
        before_total += before
        if is_already_optimized(path):
            skipped += 1
            after_total += before
            continue
        if args.apply:
            recompress(path)
            after = path.stat().st_size
        else:
            after = before // 2  # float32 estimate for the preview
        after_total += after
        changed += 1
        if i % 500 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] changed={changed} skipped={skipped}", flush=True)

    verb = "reclaimed" if args.apply else "would reclaim"
    print(f"\n{len(files)} files: {changed} to rewrite, {skipped} already optimized")
    print(f"before: {before_total / 1e9:.2f} GB")
    print(f"after:  {after_total / 1e9:.2f} GB")
    print(f"{verb}: {(before_total - after_total) / 1e9:.2f} GB")
    if not args.apply and changed:
        print("\ndry-run only. re-run with --apply to rewrite.")


if __name__ == "__main__":
    main()
