from pathlib import Path
import re

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_gain_ipm_dir_patch_v2")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# We only care about the gain command, identified by exp09_full_ipm_gain_from_pump.py.
# Find a block/list containing EXP09 or exp09_full_ipm_gain_from_pump.py and --pump-dir,
# then insert "--ipm-dir", str(design_dir), after the pump-dir value if absent.
lines = text.splitlines()
out = []
changed = False
inside_gain_cmd = False
recent_pump_dir_line = None

for i, line in enumerate(lines):
    stripped = line.strip()

    if "EXP09" in line or "exp09_full_ipm_gain_from_pump.py" in line:
        inside_gain_cmd = True

    if inside_gain_cmd and stripped in ("]", "),"):
        inside_gain_cmd = False
        recent_pump_dir_line = None

    out.append(line)

    if inside_gain_cmd and '"--pump-dir"' in line:
        recent_pump_dir_line = line
        continue

    # Typical layout:
    #   "--pump-dir", str(pump_dir),
    # or:
    #   "--pump-dir",
    #   str(pump_dir),
    #
    # After the value line, insert --ipm-dir unless the local command block already has it.
    if inside_gain_cmd and recent_pump_dir_line is not None:
        if "str(" in line or "pump" in line:
            # Look ahead a few lines to avoid duplicate insertion.
            lookahead = "\n".join(lines[i + 1:i + 8])
            if '"--ipm-dir"' not in lookahead:
                indent = re.match(r"^(\s*)", line).group(1)
                out.append(f'{indent}"--ipm-dir", str(design_dir),')
                changed = True
            recent_pump_dir_line = None

text2 = "\n".join(out) + ("\n" if text.endswith("\n") else "")

if not changed:
    # Fallback patch for compact same-line form in the gain command.
    pattern = r'("--pump-dir",\s*str\([^)]+\),)(?!\s*\n\s*"--ipm-dir")'
    def repl(m):
        return m.group(1) + '\n            "--ipm-dir", str(design_dir),'
    text3, n = re.subn(pattern, repl, text)
    if n > 0:
        text2 = text3
        changed = True

if not changed:
    print("NO_CHANGE. Printing nearby gain command lines for manual inspection:\n")
    for idx, line in enumerate(lines):
        if "EXP09" in line or "exp09_full_ipm_gain_from_pump.py" in line or '"--pump-dir"' in line:
            lo = max(0, idx - 8)
            hi = min(len(lines), idx + 20)
            print("=" * 80)
            print("\n".join(f"{j+1:04d}: {lines[j]}" for j in range(lo, hi)))
    raise SystemExit("Could not patch automatically.")

path.write_text(text2, encoding="utf-8")
print("PATCH_OK Exp11 gain command now passes --ipm-dir")
print(f"backup={backup}")
