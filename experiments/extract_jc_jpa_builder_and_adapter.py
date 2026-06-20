from pathlib import Path

files = [
    Path(r"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\cases\jc_docs\build_jc_jpa_case.jl"),
    Path(r"D:\Projects\Thesis\Harmonia.jl\experiments\solver_benchmark\backends\adapters\jc_adapter.jl"),
]

for p in files:
    print(f"\n\n################ {p} ################")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines, start=1):
        print(f"{i:5}: {line}")
