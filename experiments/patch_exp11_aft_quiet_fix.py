from pathlib import Path

path = Path("experiments/exp11_jc_doc_solver_probe.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_aft_quiet_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

text2 = text

# Literal already-concatenated variants.
text2 = text2.replace('"aft--quiet"', '"aft", "--quiet"')
text2 = text2.replace("'aft--quiet'", "'aft', '--quiet'")

# Adjacent literal variants.
text2 = text2.replace('"aft" "--quiet"', '"aft", "--quiet"')
text2 = text2.replace("'aft' '--quiet'", "'aft', '--quiet'")

# More robust line-level patch: if a list contains "--jvp-mode", "aft" and then "--skip-time-residual"
# but no standalone "--quiet", insert it after "aft".
if '"--quiet"' not in text2 and "'--quiet'" not in text2:
    text2 = text2.replace(
        '"--jvp-mode", "aft",',
        '"--jvp-mode", "aft",\n        "--quiet",',
    )

if text2 == text:
    print("NO_CHANGE: no aft--quiet pattern found. If it still prints, paste the pump_cmd list.")
else:
    path.write_text(text2, encoding="utf-8")
    print("PATCH_OK exp11 jvp/quiet command fixed")
    print(f"backup={backup}")
