from pathlib import Path

path = Path("experiments/exp10_jc_doc_python_design_builders.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_schema_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

old = """        np.savez(
            outdir / "ipm_arrays.npz",
            Ic=Ic,
            phi0_reduced=np.array(PHI0_REDUCED),
            port_numbers=np.array(sorted(assembled["ports"].keys()), dtype=np.int64),
            port_indices=np.array([assembled["ports"][k] for k in sorted(assembled["ports"].keys())], dtype=np.int64),
        )
"""

new = """        port_numbers = np.array(sorted(assembled["ports"].keys()), dtype=np.int64)
        port_indices = np.array([assembled["ports"][k] for k in sorted(assembled["ports"].keys())], dtype=np.int64)
        ports_dict = {int(k): int(v) for k, v in assembled["ports"].items()}

        np.savez(
            outdir / "ipm_arrays.npz",
            # Compatibility with exp08_full_ipm_pump_solve.py / exp09_full_ipm_gain_from_pump.py
            nodes=np.array(assembled["node_count"], dtype=np.int64),
            node_count=np.array(assembled["node_count"], dtype=np.int64),
            ports=np.array(ports_dict, dtype=object),

            # Explicit portable representation too.
            port_numbers=port_numbers,
            port_indices=port_indices,

            Ic=Ic,
            phi0_reduced=np.array(PHI0_REDUCED),
        )
"""

if old not in text:
    raise SystemExit("Could not find old np.savez(ipm_arrays.npz) block. Show me the write() method if this fails.")

text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("PATCH_OK exp10 schema now exports nodes + ports")
print(f"backup={backup}")
