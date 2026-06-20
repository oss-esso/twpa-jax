from pathlib import Path
import re

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_fqjtwpa_diss_manual_config")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Add to SUPPORTED_NOW set/list.
if "SUPPORTED_NOW" in text:
    m = re.search(r"SUPPORTED_NOW\s*=\s*([{\[])(.*?)([}\]])", text, flags=re.S)
    if m and '"jc_fqjtwpa_diss"' not in m.group(0) and "'jc_fqjtwpa_diss'" not in m.group(0):
        block = m.group(0)
        close = "}" if m.group(1) == "{" else "]"
        new_block = block.rsplit(close, 1)[0].rstrip()
        if not new_block.endswith(","):
            new_block += ","
        new_block += '\n    "jc_fqjtwpa_diss",\n' + close
        text = text[:m.start()] + new_block + text[m.end():]
else:
    raise SystemExit("Could not find SUPPORTED_NOW.")

# 2. Remove unsupported reason.
text = re.sub(
    r'\s*["\']jc_fqjtwpa_diss["\']\s*:\s*["\']complex_valued_loss["\']\s*,?',
    "",
    text,
)

# 3. Add/replace case config. We find the config dict containing jc_fqjtwpa.
#    Case config may be called CASE_CONFIG, CASES, SUPPORTED_NOW metadata, etc.,
#    so this patch targets the literal jc_fqjtwpa entry.
if '"jc_fqjtwpa_diss"' not in text[text.find('"jc_fqjtwpa"'):]:
    # Match a normal dict entry for jc_fqjtwpa.
    pat = r'(?P<entry>\s*["\']jc_fqjtwpa["\']\s*:\s*\{.*?\n\s*\},)'
    m = re.search(pat, text, flags=re.S)
    if not m:
        print("Could not find jc_fqjtwpa config entry automatically.")
        print("Run this and paste the output:")
        print('Select-String -Path .\\experiments\\exp11_jc_doc_solver_probe.py -Pattern "jc_fqjtwpa|CASE|SUPPORTED_NOW|UNSUPPORTED" -Context 5,12')
        raise SystemExit(1)

    diss_entry = '''
    "jc_fqjtwpa_diss": {
        "pump_port": 1,
        "source_port": 1,
        "out_port": 2,
        "pump_freq_ghz": 7.9,
        "pump_current_a": 1.2375e-6,
        "sweep": {"start": 1.0, "stop": 14.0, "points": 27},
    },'''

    text = text[:m.end()] + diss_entry + text[m.end():]

path.write_text(text, encoding="utf-8")
print("PATCH_OK jc_fqjtwpa_diss config added to EXP11")
print(f"backup={backup}")
