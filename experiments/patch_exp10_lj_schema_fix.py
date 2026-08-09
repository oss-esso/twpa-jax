from pathlib import Path
import re

path = Path("experiments/exp10_jc_doc_python_design_builders.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_lj_schema_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Add Lj vector near Ic construction, robustly.
old = """        Ic = np.zeros(len(self.josephson), dtype=np.float64)
        for col, jj in enumerate(self.josephson):
"""
new = """        Ic = np.zeros(len(self.josephson), dtype=np.float64)
        Lj = np.zeros(len(self.josephson), dtype=np.float64)
        for col, jj in enumerate(self.josephson):
"""
if old not in text:
    raise SystemExit("Could not find Ic allocation block.")
text = text.replace(old, new)

old = """            Ic[col] = jj.Ic
"""
new = """            Ic[col] = jj.Ic
            Lj[col] = jj.Lj
"""
if old not in text:
    raise SystemExit("Could not find Ic assignment block.")
text = text.replace(old, new)

# Add Lj to ipm_arrays.npz. Handle either already-patched or original block.
if "Lj=Lj," not in text:
    old = """            Ic=Ic,
            phi0_reduced=np.array(PHI0_REDUCED),
"""
    new = """            Ic=Ic,
            Lj=Lj,
            phi0_reduced=np.array(PHI0_REDUCED),
"""
    if old not in text:
        raise SystemExit("Could not find Ic=Ic save block.")
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("PATCH_OK exp10 now exports Lj")
print(f"backup={backup}")
