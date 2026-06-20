from pathlib import Path

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_gain_ipm_dir_clean_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Remove the bad stray line that got inserted near manifest_cases.
bad_lines = [
    '        "--ipm-dir", str(design_dir),\n',
    '        "--ipm-dir", str(design_dir),',
    '    "--ipm-dir", str(design_dir),\n',
    '    "--ipm-dir", str(design_dir),',
]
for bad in bad_lines:
    text = text.replace(bad, "")

# 2. Make Exp11 require Exp09 --ipm-dir support.
text = text.replace(
    '"--pump-dir", "--sweep", "--signal-start-ghz"',
    '"--pump-dir", "--ipm-dir", "--sweep", "--signal-start-ghz"',
)

# 3. Insert --ipm-dir into the gain command, using case_dir.
old = '''                "--pump-dir", str(pump_dir),
                "--sweep",'''
new = '''                "--pump-dir", str(pump_dir),
                "--ipm-dir", str(case_dir),
                "--sweep",'''

if old not in text:
    # Print local context and fail clearly.
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "gain_cmd = [" in line or '"--pump-dir", str(pump_dir)' in line:
            lo = max(0, idx - 5)
            hi = min(len(lines), idx + 20)
            print("=" * 80)
            print("\n".join(f"{j+1:04d}: {lines[j]}" for j in range(lo, hi)))
    raise SystemExit("Could not find expected gain_cmd pump-dir/sweep block.")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp11 cleaned and gain command now passes --ipm-dir case_dir")
print(f"backup={backup}")
