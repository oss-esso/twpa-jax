from pathlib import Path
import re
import py_compile

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_fix_args_scope")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Replace bad args.* usage inside solve_gain_one-created gain fields.
text = text.replace(
    "port_s_from_unit_current_response(vout_on, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)",
    "port_s_from_unit_current_response(vout_on, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm)",
)

# Add parameters to solve_gain_one signature.
m = re.search(r"def solve_gain_one\((.*?)\)\s*->\s*GainRow:", text, flags=re.S)
if not m:
    raise SystemExit("Could not find solve_gain_one signature.")

sig_body = m.group(1)
if "z0_ohm" not in sig_body:
    # Insert after source_current_a if present.
    sig_body_new = re.sub(
        r"(\n\s*source_current_a:\s*float,)",
        r"\1\n    source_port: int = 1,\n    out_port: int = 1,\n    z0_ohm: float = 50.0,",
        sig_body,
        count=1,
    )
    if sig_body_new == sig_body:
        raise SystemExit("Could not insert source_port/out_port/z0_ohm in solve_gain_one signature.")
    text = text[:m.start(1)] + sig_body_new + text[m.end(1):]

# Pass parameters from main call.
if "z0_ohm=args.z0_ohm" not in text:
    old = "        source_current_a=args.source_current_a,\n"
    new = old + "        source_port=args.source_port,\n        out_port=args.out_port,\n        z0_ohm=args.z0_ohm,\n"
    if old not in text:
        raise SystemExit("Could not find source_current_a=args.source_current_a call argument.")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
py_compile.compile(str(path), doraise=True)
print("PATCH_OK args scope fixed; exp09 compiles")
print(f"backup={backup}")
