from pathlib import Path

path = Path("experiments/exp10_jc_doc_python_design_builders.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_phi0_shape_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

text2 = text.replace(
    "phi0_reduced=np.array(PHI0_REDUCED),",
    "phi0_reduced=np.array([PHI0_REDUCED], dtype=np.float64),",
)

if text2 == text:
    raise SystemExit("No phi0_reduced=np.array(PHI0_REDUCED) occurrence found.")

path.write_text(text2, encoding="utf-8")
print("PATCH_OK exp10 exports phi0_reduced as shape-(1,) array")
print(f"backup={backup}")
