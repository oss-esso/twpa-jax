from pathlib import Path
import ast

path = Path(r"D:\Projects\Thesis\twpa_jax\experiments\exp09_full_ipm_gain_from_pump.py")
out = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp14_diss_manual_exp09_core_snippets.txt")
out.parent.mkdir(parents=True, exist_ok=True)

wanted = {
    "dynamic_block",
    "assemble_conversion_matrix",
    "solve_single_block_transfer",
    "solve_gain_one",
    "parse_args",
    "main",
}

txt = path.read_text(encoding="utf-8")
lines = txt.splitlines()
tree = ast.parse(txt)

chunks = []

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
        a = max(1, node.lineno - 3)
        b = min(len(lines), getattr(node, "end_lineno", node.lineno) + 3)
        chunks.append((node.lineno, node.name, a, b))

chunks.sort()

with out.open("w", encoding="utf-8") as f:
    for _, name, a, b in chunks:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"{name}: lines {a}-{b}\n")
        f.write("=" * 100 + "\n")
        for i in range(a, b + 1):
            f.write(f"{i:5d}: {lines[i-1]}\n")

print(f"WROTE {out}")
print()
print(out.read_text(encoding="utf-8"))
