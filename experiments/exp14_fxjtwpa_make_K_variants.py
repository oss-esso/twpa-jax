from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
BASE = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
OUTROOT = ROOT / "outputs" / "exp14_fxjtwpa_K_variants"

if not BASE.exists():
    raise SystemExit(f"missing base design {BASE}")

OUTROOT.mkdir(parents=True, exist_ok=True)

# FXJTWPA source uses K = 0.999 in the JC doc example.
k_coupling = 0.999
denom = 1.0 - k_coupling * k_coupling

variants = {
    "baseline_copy": None,
    "flip_big_delta4_offdiag": "flip",
    "scale_big_mutual_by_denom": "scale",
    "flip_and_scale_big_mutual": "flip_scale",
}

def copy_design(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst)

def patch_K(dst: Path, mode: str | None) -> dict:
    Kp = dst / "K.npz"
    K = sp.load_npz(Kp).tocoo()

    row = K.row.copy()
    col = K.col.copy()
    data = K.data.astype(np.complex128).copy()

    # The near-singular mutual inverse entries dominate at ~7.23e15 and are
    # exactly the +/-4 node-pair structure seen in the residual. Keep threshold
    # conservative so we don't touch ordinary Josephson/tangent/static terms.
    big = np.abs(data) > 1e15
    delta4 = np.abs(col - row) == 4
    diag = col == row

    mask_offdiag = big & delta4
    mask_diag = big & diag
    mask_all_big_mutual = mask_offdiag | mask_diag

    if mode == "flip":
        data[mask_offdiag] *= -1.0
    elif mode == "scale":
        data[mask_all_big_mutual] *= denom
    elif mode == "flip_scale":
        data[mask_offdiag] *= -denom
        data[mask_diag] *= denom
    elif mode is None:
        pass
    else:
        raise ValueError(mode)

    K2 = sp.coo_matrix((data, (row, col)), shape=K.shape).tocsr()
    sp.save_npz(Kp, K2)

    meta = {
        "variant": mode or "baseline_copy",
        "base": str(BASE),
        "k_coupling_assumed": k_coupling,
        "denom": denom,
        "n_big_diag": int(mask_diag.sum()),
        "n_big_delta4_offdiag": int(mask_offdiag.sum()),
        "n_big_mutual_total": int(mask_all_big_mutual.sum()),
        "original_absmax": float(np.max(np.abs(K.data))),
        "patched_absmax": float(np.max(np.abs(K2.data))),
    }
    (dst / "K_variant_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

rows = []
for name, mode in variants.items():
    dst = OUTROOT / name
    copy_design(dst)
    meta = patch_K(dst, mode)
    rows.append(meta | {"path": str(dst)})
    print(json.dumps(rows[-1], indent=2))

print(f"\nWROTE {OUTROOT}")
