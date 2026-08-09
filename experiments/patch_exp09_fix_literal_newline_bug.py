from pathlib import Path
import py_compile

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

bad = '        f"{r.gain_db:.9g},"\\n        f"{r.s_param_abs:.9g},"\\n        f"{r.gain_vs_off_db:.9g},"'
good = '''        f"{r.gain_db:.9g},"
        f"{r.s_param_abs:.9g},"
        f"{r.gain_vs_off_db:.9g},"'''

if bad not in text:
    print("Exact bad literal not found. Trying broad repair...")
    text = text.replace(',"\\n        f"', ',"\n        f"')
else:
    text = text.replace(bad, good, 1)

path.write_text(text, encoding="utf-8")

py_compile.compile(str(path), doraise=True)
print("PATCH_OK syntax repaired and exp09 compiles")
