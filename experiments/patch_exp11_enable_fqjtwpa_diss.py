from pathlib import Path
import re

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_fqjtwpa_diss_enable")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Add jc_fqjtwpa_diss to SUPPORTED_NOW if it is a list/set/dict-style declaration.
text = text.replace(
    '"jc_fqjtwpa",',
    '"jc_fqjtwpa",\n    "jc_fqjtwpa_diss",',
    1,
)

# 2. If unsupported reasons are hardcoded, remove complex_valued_loss as a blocker.
text = text.replace(
    '"jc_fqjtwpa_diss": "complex_valued_loss",',
    '',
)
text = text.replace(
    "'jc_fqjtwpa_diss': 'complex_valued_loss',",
    '',
)

# 3. Add/patch case settings for jc_fqjtwpa_diss.
# It should reuse the FQJTWPA ports/sweep, but with the lossy pump current exported by Exp10 metadata.
if '"jc_fqjtwpa_diss"' not in text[text.find("CASE_CONFIG") if "CASE_CONFIG" in text else 0:]:
    # Try to insert after jc_fqjtwpa config block by duplicating a compact common pattern.
    # If your file uses a dict called CASES or SUPPORTED_NOW with per-case metadata, this should still be inspectable after patch.
    marker = '"jc_fqjtwpa"'
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("Could not find jc_fqjtwpa marker. Paste top case config block.")
    print("WARNING: jc_fqjtwpa_diss not found near case config; you may need manual config insertion.")

path.write_text(text, encoding="utf-8")
print("PATCH_DONE partial jc_fqjtwpa_diss enable. Inspect SUPPORTED_NOW/CASE_CONFIG if needed.")
print(f"backup={backup}")
