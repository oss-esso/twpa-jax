from pathlib import Path
import re

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_ipm_dir_patch")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Add CLI arg after --pump-dir.
if "--ipm-dir" not in text:
    m = re.search(r'(?m)^(\s*p\.add_argument\("--pump-dir".*\n)', text)
    if not m:
        raise SystemExit("Could not find --pump-dir argparse line.")
    insert = m.group(1) + '    p.add_argument("--ipm-dir", default=None, help="Design folder containing C/G/K/Bphi/ipm_arrays.npz. If omitted, use pump_report metadata or legacy default.")\n'
    text = text[:m.start()] + insert + text[m.end():]

# 2. Add helper to infer ipm_dir from pump_report.json.
if "def infer_ipm_dir_from_pump_report" not in text:
    marker = "def load_ipm"
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("Could not find def load_ipm insertion point.")

    helper = '''def infer_ipm_dir_from_pump_report(pump_dir: Path) -> str | None:
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return None
    try:
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return None

    # Common report layouts.
    for container in (report, report.get("metadata", {}), report.get("settings", {})):
        if isinstance(container, dict):
            value = container.get("ipm_dir")
            if value:
                return str(value)

    return None


'''
    text = text[:idx] + helper + text[idx:]

# 3. Replace legacy load_ipm call.
# We handle common variants robustly.
if "resolved_ipm_dir = args.ipm_dir or infer_ipm_dir_from_pump_report" not in text:
    patterns = [
        r'ipm\s*=\s*load_ipm\(args\.ipm_dir\)',
        r'ipm\s*=\s*load_ipm\(Path\(args\.ipm_dir\)\)',
        r'ipm\s*=\s*load_ipm\((["\']outputs/ipm_python_design["\']|Path\(["\']outputs/ipm_python_design["\']\))\)',
    ]

    replacement = '''resolved_ipm_dir = args.ipm_dir or infer_ipm_dir_from_pump_report(Path(args.pump_dir)) or "outputs/ipm_python_design"
    ipm = load_ipm(resolved_ipm_dir)'''

    done = False
    for pat in patterns:
        text2, n = re.subn(pat, replacement, text, count=1)
        if n == 1:
            text = text2
            done = True
            break

    if not done:
        # Fallback: look for the first load_ipm(...) after parse_args/main setup.
        m = re.search(r'(?m)^(\s*)ipm\s*=\s*load_ipm\(([^\n]+)\)\s*$', text)
        if not m:
            raise SystemExit("Could not find load_ipm(...) call. Paste Exp09 main() load section if this fails.")
        indent = m.group(1)
        repl = (
            indent + 'resolved_ipm_dir = args.ipm_dir or infer_ipm_dir_from_pump_report(Path(args.pump_dir)) or "outputs/ipm_python_design"\\n'
            + indent + 'ipm = load_ipm(resolved_ipm_dir)'
        )
        text = text[:m.start()] + repl + text[m.end():]

# 4. Print resolved ipm_dir in setup if possible.
if 'print(f"ipm_dir={resolved_ipm_dir}")' not in text:
    text = text.replace(
        'print("=== experiment 09 setup ===")',
        'print("=== experiment 09 setup ===")\n    print(f"ipm_dir={resolved_ipm_dir}")',
        1,
    )

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp09 now supports/infer --ipm-dir")
print(f"backup={backup}")
