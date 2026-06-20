from pathlib import Path
import ast

p = Path(r"D:\Projects\Thesis\twpa_jax\experiments\exp09_full_ipm_gain_from_pump.py")
txt = p.read_text(encoding="utf-8")
lines = txt.splitlines()
tree = ast.parse(txt)

wanted = {
    "load_ipm",
    "build_rhs",
    "dynamic_block",
    "assemble_conversion_matrix",
    "solve_single_block_transfer",
    "solve_gain_one",
}

for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name in wanted:
        print(f"\n\n################ {n.name}: lines {n.lineno}-{n.end_lineno} ################")
        for i in range(n.lineno, n.end_lineno + 1):
            print(f"{i:5}: {lines[i-1]}")
