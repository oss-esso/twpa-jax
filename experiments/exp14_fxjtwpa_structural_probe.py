from __future__ import annotations

from pathlib import Path
import ast
import json
import csv
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
OUT = ROOT / "outputs" / "exp14_fxjtwpa_structural_inventory"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ROOT / "experiments" / "exp10_jc_doc_python_design_builders.py",
    ROOT / "experiments" / "exp14_build_jc_warmstart.py",
    ROOT / "experiments" / "exp08_full_ipm_pump_solve.py",
]

KEYWORDS = [
    "fxjtwpa",
    "squid",
    "mutual",
    "coupling",
    "kmat",
    "kinv",
    "flux",
    "phi",
    "inductor",
    "lj",
    "branch",
    "warmstart",
    "nodeflux",
]


def dump_matching_functions(path: Path) -> str:
    txt = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = txt.splitlines()

    chunks = []
    try:
        tree = ast.parse(txt)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ClassDef := ast.ClassDef)):
                src = "\n".join(lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)])
                low = (node.name + "\n" + src).lower()
                if any(k in low for k in KEYWORDS):
                    a = max(1, node.lineno - 5)
                    b = min(len(lines), getattr(node, "end_lineno", node.lineno) + 5)
                    chunks.append((node.lineno, node.name, a, b))
    except SyntaxError as e:
        # Fallback for files with weird encoding or temporary syntax damage:
        # dump line windows around keywords instead of AST functions.
        seen = set()
        for i, line in enumerate(lines, start=1):
            low = line.lower()
            if any(k in low for k in KEYWORDS):
                a = max(1, i - 12)
                b = min(len(lines), i + 24)
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    chunks.append((i, f"keyword_window_after_SyntaxError_{e.lineno}", a, b))

    chunks.sort()
    parts = []
    for _, name, a, b in chunks:
        parts.append("\n" + "=" * 120)
        parts.append(f"{path.name} :: {name} :: lines {a}-{b}")
        parts.append("=" * 120)
        for i in range(a, b + 1):
            parts.append(f"{i:5d}: {lines[i-1]}")
    return "\n".join(parts)


def describe_npz(path: Path) -> dict:
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        return {"path": str(path), "error": repr(e)}

    info = {"path": str(path), "keys": list(z.files)}
    for k in z.files:
        arr = z[k]
        item = {"shape": arr.shape, "dtype": str(arr.dtype)}
        if arr.size and arr.dtype.kind in "biufc":
            flat = arr.ravel()
            finite = flat[np.isfinite(flat)] if arr.dtype.kind in "fcu" else flat
            if finite.size:
                item.update({
                    "min": float(np.min(finite.real)) if np.iscomplexobj(finite) else float(np.min(finite)),
                    "max": float(np.max(finite.real)) if np.iscomplexobj(finite) else float(np.max(finite)),
                    "absmax": float(np.max(np.abs(finite))),
                })
        info[k] = item
    return info


def sparse_stats(path: Path) -> dict:
    M = sp.load_npz(path).tocsr()
    data = M.data
    stats = {
        "path": str(path),
        "shape0": M.shape[0],
        "shape1": M.shape[1],
        "nnz": M.nnz,
    }
    if data.size:
        idx = int(np.argmax(np.abs(data)))
        stats.update({
            "absmax": float(np.max(np.abs(data))),
            "max_data": complex(data[idx]).real,
            "max_data_imag": complex(data[idx]).imag,
        })
    return stats


def top_sparse_entries(path: Path, n: int = 80) -> list[dict]:
    M = sp.load_npz(path).tocoo()
    if M.nnz == 0:
        return []
    order = np.argsort(np.abs(M.data))[::-1][:n]
    rows = []
    for j in order:
        rows.append({
            "row": int(M.row[j]),
            "col": int(M.col[j]),
            "value_real": float(np.real(M.data[j])),
            "value_imag": float(np.imag(M.data[j])),
            "abs": float(abs(M.data[j])),
            "delta": int(M.col[j] - M.row[j]),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# 1. Source snippets.
snippet_path = OUT / "source_snippets.txt"
with snippet_path.open("w", encoding="utf-8") as f:
    for path in FILES:
        f.write("\n\n" + "#" * 140 + "\n")
        f.write(str(path) + "\n")
        f.write("#" * 140 + "\n")
        if path.exists():
            f.write(dump_matching_functions(path))
        else:
            f.write("MISSING\n")

# 2. Output tree manifest.
manifest = []
for root in [
    ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa",
    ROOT / "outputs" / "exp14_fxjtwpa_jcseed",
    ROOT / "outputs" / "exp14_fxjtwpa_dense4_scale2",
    ROOT / "outputs" / "exp12_fxjtwpa_dc",
    ROOT / "outputs" / "exp12_fxjtwpa_pump",
]:
    if not root.exists():
        manifest.append({"root": str(root), "path": "MISSING", "size": ""})
        continue
    for p in sorted(root.rglob("*")):
        if p.is_file():
            manifest.append({
                "root": str(root),
                "path": str(p),
                "size": p.stat().st_size,
            })
write_csv(OUT / "artifact_manifest.csv", manifest)

# 3. Matrix stats for likely Python design export.
design = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
matrix_rows = []
top_rows = []

if design.exists():
    for name in ["C.npz", "G.npz", "K.npz", "Bphi.npz"]:
        p = design / name
        if p.exists():
            matrix_rows.append({"name": name, **sparse_stats(p)})
            for r in top_sparse_entries(p, 120):
                top_rows.append({"matrix": name, **r})

    arrays = design / "ipm_arrays.npz"
    if arrays.exists():
        (OUT / "ipm_arrays_summary.json").write_text(
            json.dumps(describe_npz(arrays), indent=2, default=str),
            encoding="utf-8",
        )

write_csv(OUT / "matrix_stats.csv", matrix_rows)
write_csv(OUT / "top_sparse_entries.csv", top_rows)

# 4. Specific suspicious local rows from prior diagnosis: internal/mutual node pairs spaced by +4.
# Dump K rows around first few known bad locations if matrix exists.
Kp = design / "K.npz"
local_rows = []
if Kp.exists():
    K = sp.load_npz(Kp).tocsr()
    suspect_centers = [191, 195, 941, 945, 1691, 1695, 2441, 2445]
    for center in suspect_centers:
        if center < 0 or center >= K.shape[0]:
            continue
        for row in range(max(0, center - 8), min(K.shape[0], center + 9)):
            a, b = K.indptr[row], K.indptr[row + 1]
            cols = K.indices[a:b]
            vals = K.data[a:b]
            order = np.argsort(np.abs(vals))[::-1][:20]
            for idx in order:
                col = int(cols[idx])
                val = vals[idx]
                if abs(col - center) <= 16 or abs(col - row) <= 16 or abs(val) > 1e14:
                    local_rows.append({
                        "center": center,
                        "row": row,
                        "col": col,
                        "delta": col - row,
                        "value_real": float(np.real(val)),
                        "value_imag": float(np.imag(val)),
                        "abs": float(abs(val)),
                    })
write_csv(OUT / "local_K_suspicious_rows.csv", local_rows)

print(f"WROTE {OUT}")
print(f"source_snippets={snippet_path}")
print(f"artifact_manifest={OUT / 'artifact_manifest.csv'}")
print(f"matrix_stats={OUT / 'matrix_stats.csv'}")
print(f"top_sparse_entries={OUT / 'top_sparse_entries.csv'}")
print(f"local_K_suspicious_rows={OUT / 'local_K_suspicious_rows.csv'}")

print("\n=== matrix_stats ===")
for r in matrix_rows:
    print(r)

print("\n=== first local suspicious K rows ===")
for r in local_rows[:80]:
    print(r)

print("\n=== source snippet preview ===")
snip = snippet_path.read_text(encoding="utf-8", errors="replace").splitlines()
for line in snip[:220]:
    print(line)
