from pathlib import Path

root = Path(r"D:\Projects\Thesis\JosephsonCircuits.jl\src")
out = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_jtwpa_gamma_hat_compare\jc_source_hits.txt")
out.parent.mkdir(parents=True, exist_ok=True)

terms = [
    "calcfj",
    "calcq",
    "phin",
    "phi",
    "pump",
    "linearized",
    "Npumpharmonics",
    "Nmodulationharmonics",
    "cos",
    "Ic",
    "Lj",
    "Kerr",
    "fourier",
    "harmonic",
    "Jacobian",
    "jacobian",
]

lines_out = []

for path in sorted(root.rglob("*.jl")):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        continue

    for i, line in enumerate(lines):
        low = line.lower()
        if any(t.lower() in low for t in terms):
            lo = max(0, i - 2)
            hi = min(len(lines), i + 3)
            lines_out.append("=" * 100)
            lines_out.append(f"{path}:{i+1}")
            for j in range(lo, hi):
                prefix = ">" if j == i else " "
                lines_out.append(f"{prefix} {j+1:5d}: {lines[j]}")

out.write_text("\n".join(lines_out), encoding="utf-8")
print("SOURCE_SEARCH_OK")
print("wrote =", out)
print("hit_blocks =", sum(1 for x in lines_out if x.startswith("=")))
