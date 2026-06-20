from pathlib import Path

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_enable_fqjtwpa_diss_clean")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Add to supported list/set if not already there.
if '"jc_fqjtwpa_diss"' not in text[text.find("SUPPORTED_NOW"):text.find("UNSUPPORTED") if "UNSUPPORTED" in text else len(text)]:
    text = text.replace(
        '"jc_fqjtwpa",',
        '"jc_fqjtwpa",\n    "jc_fqjtwpa_diss",',
        1,
    )

# Remove unsupported reason if present.
for bad in [
    '"jc_fqjtwpa_diss": "complex_valued_loss",',
    "'jc_fqjtwpa_diss': 'complex_valued_loss',",
    '"jc_fqjtwpa_diss": "complex_valued_loss"',
    "'jc_fqjtwpa_diss': 'complex_valued_loss'",
]:
    text = text.replace(bad, "")

# Add metadata/config by cloning jc_fqjtwpa if there is a case config dict.
if '"jc_fqjtwpa_diss"' not in text[text.find("CASE"):]:
    old = '''"jc_fqjtwpa": {
        "pump_port": 1,
        "source_port": 1,
        "out_port": 2,
        "pump_freq_ghz": 7.9,
        "pump_current_a": 1.1e-6,
        "sweep": {"start": 1.0, "stop": 14.0, "points": 27},
    },'''
    new = old + '''
    "jc_fqjtwpa_diss": {
        "pump_port": 1,
        "source_port": 1,
        "out_port": 2,
        "pump_freq_ghz": 7.9,
        "pump_current_a": 1.2375e-6,
        "sweep": {"start": 1.0, "stop": 14.0, "points": 27},
    },'''
    if old in text:
        text = text.replace(old, new, 1)
    else:
        print("WARNING: Could not auto-clone jc_fqjtwpa config. If EXP11 does not include diss, paste the case config block.")

path.write_text(text, encoding="utf-8")
print("PATCH_OK attempted to enable jc_fqjtwpa_diss in EXP11")
print(f"backup={backup}")
