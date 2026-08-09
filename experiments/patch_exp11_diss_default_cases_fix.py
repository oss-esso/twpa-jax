from pathlib import Path
import re

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_diss_default_cases_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Ensure SUPPORTED_NOW contains diss.
text = re.sub(
    r'SUPPORTED_NOW\s*=\s*\{([^}]*)\}',
    lambda m: (
        m.group(0)
        if "jc_fqjtwpa_diss" in m.group(0)
        else 'SUPPORTED_NOW = {"jc_jpa", "jc_jtwpa", "jc_fqjtwpa", "jc_fqjtwpa_diss"}'
    ),
    text,
    count=1,
)

# 2. Ensure CASE_SWEEPS contains diss.
if '"jc_fqjtwpa_diss"' not in text[text.find("CASE_SWEEPS"):text.find("def run")]:
    needle = '    "jc_fqjtwpa": {"source_port": 1, "out_port": 2, "start": 1.0, "stop": 14.0, "points": 27, "sidebands": 2},'
    insert = needle + '\n    "jc_fqjtwpa_diss": {"source_port": 1, "out_port": 2, "start": 1.0, "stop": 14.0, "points": 27, "sidebands": 2},'
    if needle not in text:
        raise SystemExit("Could not find jc_fqjtwpa CASE_SWEEPS line.")
    text = text.replace(needle, insert, 1)

# 3. Ensure argparse default --cases includes diss.
text = text.replace(
    'ap.add_argument("--cases", nargs="*", default=["jc_jpa", "jc_jtwpa", "jc_fqjtwpa"])',
    'ap.add_argument("--cases", nargs="*", default=["jc_jpa", "jc_jtwpa", "jc_fqjtwpa", "jc_fqjtwpa_diss"])',
)

# 4. Remove complex_valued_loss from unsupported reason for this case only is not needed if supported,
#    but keep unsupported_reason generic for genuinely unsupported future complex cases.

path.write_text(text, encoding="utf-8")
print("PATCH_OK EXP11 default cases now include jc_fqjtwpa_diss")
print(f"backup={backup}")
