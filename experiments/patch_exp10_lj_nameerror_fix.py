from pathlib import Path

path = Path("experiments/exp10_jc_doc_python_design_builders.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_lj_nameerror_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

needle = """        port_numbers = np.array(sorted(assembled["ports"].keys()), dtype=np.int64)
        port_indices = np.array([assembled["ports"][k] for k in sorted(assembled["ports"].keys())], dtype=np.int64)
        ports_dict = {int(k): int(v) for k, v in assembled["ports"].items()}
"""

insert = """        # Export Josephson inductance vector in the schema expected by Exp08.
        # Some earlier patch paths only created Ic, so compute Lj directly here.
        Lj = np.array([float(jj.Lj) for jj in self.josephson], dtype=np.float64)

        port_numbers = np.array(sorted(assembled["ports"].keys()), dtype=np.int64)
        port_indices = np.array([assembled["ports"][k] for k in sorted(assembled["ports"].keys())], dtype=np.int64)
        ports_dict = {int(k): int(v) for k, v in assembled["ports"].items()}
"""

if "Lj = np.array([float(jj.Lj) for jj in self.josephson]" not in text:
    if needle not in text:
        raise SystemExit("Could not find port_numbers block. Paste the write() method if this fails.")
    text = text.replace(needle, insert)

# Ensure Lj is actually included in np.savez.
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
print("PATCH_OK Lj NameError fixed in exp10 export")
print(f"backup={backup}")
