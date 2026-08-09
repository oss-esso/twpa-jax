from pathlib import Path

roots = [
    Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp14_fxjtwpa_jcseed"),
    Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp14_fxjtwpa_dense4_scale2"),
]

for root in roots:
    print("\n" + "=" * 100)
    print(root)
    print("=" * 100)
    if not root.exists():
        print("MISSING")
        continue

    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            print(f"{str(rel):70s} {p.stat().st_size}")
