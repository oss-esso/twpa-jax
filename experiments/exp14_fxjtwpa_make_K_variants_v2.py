from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
BASE = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
OUTROOT = ROOT / "outputs" / "exp14_fxjtwpa_K_variants_v2"

if not BASE.exists():
    raise SystemExit(f"missing base design {BASE}")

OUTROOT.mkdir(parents=True, exist_ok=True)

k_coupling = 0.999
denom = 1.0 - k_coupling * k_coupling

variants = {
    "baseline_copy": "baseline",
    "flip_huge_offdiag": "flip_huge_offdiag",
    "scale_huge_all_by_denom": "scale_huge_all",
    "flip_offdiag_and_scale_huge_all": "flip_scale_huge_all",
}


def copy_design(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst)


def local_entries(K: sp.csr_matrix, rows_to_show=(191, 195)) -> list[dict]:
    out = []
    for row in rows_to_show:
        a, b = K.indptr[row], K.indptr[row + 1]
        cols = K.indices[a:b]
        vals = K.data[a:b]
        order = np.argsort(np.abs(vals))[::-1][:12]
        for idx in order:
            col = int(cols[idx])
            val = vals[idx]
            out.append({
                "row": row,
                "col": col,
                "delta": col - row,
                "value": float(np.real(val)),
                "abs": float(abs(val)),
            })
    return out


def patch_K(dst: Path, mode: str) -> dict:
    Kp = dst / "K.npz"
    K0 = sp.load_npz(Kp).tocoo()

    row = K0.row.copy()
    col = K0.col.copy()
    data = K0.data.astype(np.complex128).copy()

    huge = np.abs(data) > 1e15
    offdiag = row != col
    huge_offdiag = huge & offdiag

    if mode == "baseline":
        pass
    elif mode == "flip_huge_offdiag":
        data[huge_offdiag] *= -1.0
    elif mode == "scale_huge_all":
        data[huge] *= denom
    elif mode == "flip_scale_huge_all":
        data[huge_offdiag] *= -denom
        data[huge & ~offdiag] *= denom
    else:
        raise ValueError(mode)

    K2 = sp.coo_matrix((data, (row, col)), shape=K0.shape).tocsr()
    sp.save_npz(Kp, K2)

    meta = {
        "variant": mode,
        "base": str(BASE),
        "k_coupling_assumed": k_coupling,
        "denom": denom,
        "n_huge": int(huge.sum()),
        "n_huge_offdiag": int(huge_offdiag.sum()),
        "original_absmax": float(np.max(np.abs(K0.data))),
        "patched_absmax": float(np.max(np.abs(K2.data))),
        "local_entries": local_entries(K2),
    }
    (dst / "K_variant_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


for name, mode in variants.items():
    dst = OUTROOT / name
    copy_design(dst)
    meta = patch_K(dst, mode)

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print(json.dumps({
        k: v for k, v in meta.items()
        if k != "local_entries"
    }, indent=2))
    print("local_entries:")
    for r in meta["local_entries"]:
        print(r)

print(f"\nWROTE {OUTROOT}")
