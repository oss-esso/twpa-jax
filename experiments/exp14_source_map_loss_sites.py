from pathlib import Path

root = Path(r"D:\Projects\Thesis\twpa_jax")
files = [
    root / "experiments" / "exp09_full_ipm_gain_from_pump.py",
    root / "experiments" / "exp08_full_ipm_pump_solve.py",
    root / "experiments" / "exp14_diss_loss_study.py",
]

patterns = [
    "real_capacitance",
    "complex",
    "loss",
    "tandelta",
    "dynamic_block",
    "C @",
    "@ C",
    "omega",
    "Omega",
    "sideband",
    "D =",
    "A =",
    "-omega",
    "1j",
    "Gamma",
    "gamma_hat",
    "build_khat",
]

for path in files:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    if not path.exists():
        print("MISSING")
        continue

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    hits = []
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            hits.append(i)

    # Merge nearby hits into context windows.
    windows = []
    for h in hits:
        a = max(1, h - 4)
        b = min(len(lines), h + 8)
        if windows and a <= windows[-1][1] + 3:
            windows[-1] = (windows[-1][0], max(windows[-1][1], b))
        else:
            windows.append((a, b))

    for a, b in windows[:30]:
        print(f"\n--- lines {a}-{b} ---")
        for j in range(a, b + 1):
            print(f"{j:5d}: {lines[j-1]}")
