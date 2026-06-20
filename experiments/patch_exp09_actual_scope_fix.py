from pathlib import Path
import py_compile

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_actual_jc_gain_scope_fix")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Extend the actual solve_gain_one signature.
old_sig = """    source_index: int,
    out_index: int,
    source_current_a: float,
) -> GainResult:
"""
new_sig = """    source_index: int,
    out_index: int,
    source_current_a: float,
    source_port: int,
    out_port: int,
    z0_ohm: float,
) -> GainResult:
"""
if old_sig not in text:
    raise SystemExit("Could not find exact solve_gain_one signature tail.")
text = text.replace(old_sig, new_sig, 1)

# 2. Fix GainResult construction.
# Use vout_on/source_current_a so the helper receives transimpedance V/I.
old_gain = """        s_param_abs=abs(port_s_from_unit_current_response(vout_on, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
        gain_db=gain_db_from_s(port_s_from_unit_current_response(vout_on, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
"""
new_gain = """        s_param_abs=abs(port_s_from_unit_current_response(vout_on / source_current_a, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm)),
        gain_db=gain_db_from_s(port_s_from_unit_current_response(vout_on / source_current_a, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm)),
"""
if old_gain not in text:
    raise SystemExit("Could not find bad args.* gain block.")
text = text.replace(old_gain, new_gain, 1)

# 3. Pass ports and z0 from main into solve_gain_one.
old_call = """            source_index=source_index,
            out_index=out_index,
            source_current_a=args.source_current_a,
        )
"""
new_call = """            source_index=source_index,
            out_index=out_index,
            source_current_a=args.source_current_a,
            source_port=args.source_port,
            out_port=args.out_port,
            z0_ohm=args.z0_ohm,
        )
"""
if old_call not in text:
    raise SystemExit("Could not find solve_gain_one call block in main.")
text = text.replace(old_call, new_call, 1)

# 4. Put z0 into metadata if not already present.
if '"z0_ohm": args.z0_ohm,' not in text:
    old_meta = '        "source_current_a": args.source_current_a,\n'
    new_meta = old_meta + '        "z0_ohm": args.z0_ohm,\n'
    if old_meta not in text:
        raise SystemExit("Could not find metadata source_current_a line.")
    text = text.replace(old_meta, new_meta, 1)

path.write_text(text, encoding="utf-8")
py_compile.compile(str(path), doraise=True)
print("PATCH_OK exp09 solve_gain_one receives source_port/out_port/z0_ohm and compiles")
print(f"backup={backup}")
