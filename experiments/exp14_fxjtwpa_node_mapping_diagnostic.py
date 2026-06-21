from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import json
import csv
import math
import numpy as np
import scipy.sparse as sp

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")
EXP08 = ROOT / "experiments" / "exp08_full_ipm_pump_solve.py"
DESIGN = ROOT / "outputs" / "jc_doc_python_designs" / "jc_fxjtwpa"
SEED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump" / "pump_solution.npz"
SOLVED = ROOT / "outputs" / "exp14_fxjtwpa_jcseed" / "pump_solved" / "pump_solution.npz"
OUT = ROOT / "outputs" / "exp14_fxjtwpa_node_mapping_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

m = import_module(EXP08, "exp08_runtime_node_diag")
ipm = m.load_ipm(DESIGN)

def normalize_nodes(x):
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, dict):
            # Usually node label -> index; invert to index -> label.
            inv = {int(v): str(k) for k, v in item.items()}
            return [inv[i] for i in range(len(inv))]
        if isinstance(item, (list, tuple, np.ndarray)):
            return [str(v) for v in list(item)]
        return [str(item)]
    return [str(v) for v in arr.ravel().tolist()]

NODES = normalize_nodes(ipm.nodes)
NODE_TO_INDEX = {label: i for i, label in enumerate(NODES)}

print("node_container_type", type(ipm.nodes))
try:
    print("node_array_shape", np.asarray(ipm.nodes, dtype=object).shape)
except Exception:
    pass
print("normalized_node_count", len(NODES))
print("first_nodes", NODES[:12])
print("last_nodes", NODES[-12:])

C = ipm.C.tocsr()
G = ipm.G.tocsr()
K = ipm.K.tocsr()

def load_X(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X_real"] + 1j * z["X_imag"]
    modes = z["pump_modes"].astype(int) if "pump_modes" in z else z["harmonics"].astype(int)
    return X, modes

def omega_from_report(path: Path) -> float:
    report = path.parent / "pump_report.json"
    if report.exists():
        obj = json.loads(report.read_text(encoding="utf-8"))
        for d in [obj, obj.get("metadata", {}) if isinstance(obj, dict) else {}]:
            for k in ["omega_p", "pump_omega", "pump_omega_rad_s", "omega_rad_s"]:
                if k in d:
                    return float(d[k])
            for k in ["pump_freq_ghz", "freq_ghz", "pump_frequency_ghz"]:
                if k in d:
                    return 2.0 * math.pi * float(d[k]) * 1e9
    return 2.0 * math.pi * 20.0e9

def node_label(i: int) -> str:
    try:
        return str(NODES[i])
    except Exception:
        return f"<idx:{i}>"

def node_num(label: str):
    try:
        return int(label)
    except Exception:
        return None

def annotate_index(i: int) -> dict:
    lab = node_label(i)
    num = node_num(lab)
    out = {
        "index": int(i),
        "label": lab,
        "num": "" if num is None else num,
        "num_mod3": "" if num is None else num % 3,
        "num_mod4": "" if num is None else num % 4,
        "num_mod5": "" if num is None else num % 5,
        "approx_cell3": "" if num is None else (num - 1) // 3,
    }
    return out

def top_entries(v, n=80):
    order = np.argsort(np.abs(v))[::-1][:n]
    return order

def best_alpha(a, b):
    den = np.vdot(a, a)
    return 0.0j if abs(den) == 0 else np.vdot(a, b) / den

X_seed, modes = load_X(SEED)
X_solved, modes2 = load_X(SOLVED)
assert list(modes) == list(modes2)

omega_p = omega_from_report(SEED)
mode_idx = list(modes).index(1)

seed1 = X_seed[mode_idx]
solved1 = X_solved[mode_idx]
alpha1 = best_alpha(seed1, solved1)
seed1_scaled = alpha1 * seed1

D1 = K + (-(omega_p)**2) * C + (1j * omega_p) * G
r_seed = D1 @ seed1
r_solved = D1 @ solved1
diff_scaled = seed1_scaled - solved1

rows = []

for kind, vec in [
    ("seed_residual", r_seed),
    ("solved_residual", r_solved),
    ("scaled_seed_minus_solved", diff_scaled),
]:
    for rank, idx in enumerate(top_entries(vec, 120), start=1):
        row = {
            "kind": kind,
            "rank": rank,
            **annotate_index(int(idx)),
            "value_real": float(np.real(vec[idx])),
            "value_imag": float(np.imag(vec[idx])),
            "abs": float(abs(vec[idx])),
        }

        # Include local neighbors by node index, useful for ±3/±4 patterns.
        lab_num = node_num(row["label"])
        if lab_num is not None:
            for off in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                target = str(lab_num + off)
                try:
                    j = NODE_TO_INDEX[target]
                    row[f"idx_label_plus_{off:+d}"] = j
                except Exception:
                    row[f"idx_label_plus_{off:+d}"] = ""
        rows.append(row)

with (OUT / "top_index_annotations.csv").open("w", newline="", encoding="utf-8") as f:
    keys = sorted(set().union(*(r.keys() for r in rows)))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Dump node order windows around top seed residual indices.
windows = []
seen = set()
for idx in top_entries(r_seed, 20):
    idx = int(idx)
    a = max(0, idx - 8)
    b = min(len(NODES), idx + 9)
    key = (a, b)
    if key in seen:
        continue
    seen.add(key)
    for j in range(a, b):
        windows.append({
            "center_index": idx,
            **annotate_index(j),
        })

with (OUT / "node_order_windows.csv").open("w", newline="", encoding="utf-8") as f:
    keys = sorted(set().union(*(r.keys() for r in windows)))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in windows:
        w.writerow(r)

print("alpha1", alpha1, "abs", abs(alpha1), "angle", np.angle(alpha1))
print("seed residual l2", np.linalg.norm(r_seed), "inf", np.max(np.abs(r_seed)))
print("solved residual l2", np.linalg.norm(r_solved), "inf", np.max(np.abs(r_solved)))
print("scaled diff l2", np.linalg.norm(diff_scaled), "rel", np.linalg.norm(diff_scaled) / np.linalg.norm(solved1))
print()
print("TOP seed_residual annotated")
for r in [x for x in rows if x["kind"] == "seed_residual"][:50]:
    print(r)
print()
print("TOP scaled_seed_minus_solved annotated")
for r in [x for x in rows if x["kind"] == "scaled_seed_minus_solved"][:50]:
    print(r)
print()
print(f"WROTE {OUT}")
print(f"annotations={OUT / 'top_index_annotations.csv'}")
print(f"windows={OUT / 'node_order_windows.csv'}")
