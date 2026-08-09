from pathlib import Path

ROOT = Path(r"D:\Projects\Thesis\twpa_jax")

patterns = [
    "fxjtwpa",
    "warmstart",
    "nodeflux",
    "matrix",
    "khat",
    "mutual",
    "squid",
]

print("=== candidate experiment scripts ===")
for p in sorted((ROOT / "experiments").glob("*.py")):
    s = p.name.lower()
    if any(x in s for x in patterns):
        print(p)

print("\n=== candidate output folders ===")
for p in sorted((ROOT / "outputs").glob("*")):
    s = p.name.lower()
    if any(x in s for x in patterns):
        print(p)

print("\n=== files mentioning fxjtwpa / mutual / squid in experiments ===")
for p in sorted((ROOT / "experiments").glob("*.py")):
    try:
        txt = p.read_text(encoding="utf-8", errors="replace").lower()
    except Exception:
        continue
    hits = []
    for pat in ["fxjtwpa", "mutual", "squid", "nodeflux", "warmstart"]:
        if pat in txt:
            hits.append(pat)
    if hits:
        print(f"{p} :: {', '.join(hits)}")
