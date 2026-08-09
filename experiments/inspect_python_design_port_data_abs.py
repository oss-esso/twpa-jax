import numpy as np
from pathlib import Path
import json

d = Path(r"D:\Projects\Thesis\twpa_jax\outputs\jc_doc_python_designs\jc_jpa")

print("NPZ keys:")
with np.load(d / "ipm_arrays.npz", allow_pickle=True) as z:
    for k in z.files:
        arr = z[k]
        val = arr if getattr(arr, "size", 999) <= 20 else "..."
        print(f"  {k}: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)} value={val}")

print("\nsummary.json:")
print((d / "summary.json").read_text())
