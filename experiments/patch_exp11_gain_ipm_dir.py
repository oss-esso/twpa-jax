from pathlib import Path
import re

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_gain_ipm_dir_patch")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Patch gain command list: after "--pump-dir", str(pump_dir), add "--ipm-dir", str(design_dir).
if '"--ipm-dir"' not in text:
    old_variants = [
        '''"--pump-dir", str(pump_dir),''',
        '''"--pump-dir", str(pump_outdir),''',
        '''"--pump-dir", str(pump_dir)''',
        '''"--pump-dir", str(pump_outdir)''',
    ]

    done = False
    for old in old_variants:
        if old in text:
            new = old + '\n            "--ipm-dir", str(design_dir),'
            text = text.replace(old, new, 1)
            done = True
            break

    if not done:
        raise SystemExit("Could not find gain command --pump-dir block. Paste the gain_cmd list.")

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp11 passes --ipm-dir to Exp09")
print(f"backup={backup}")
