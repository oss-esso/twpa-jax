from pathlib import Path
import re

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_jc_style_gain_v3")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------
# 1. Add JC-style S conversion helpers before db10.
# ---------------------------------------------------------------------
if "def port_s_from_unit_current_response(" not in text:
    idx = text.find("def db10(")
    if idx < 0:
        raise SystemExit("Could not find def db10(")

    helper = r'''

def port_s_from_unit_current_response(
    response: complex,
    *,
    source_port: int,
    out_port: int,
    z0_ohm: float,
) -> complex:
    """Convert unit-current port voltage response into a JC-style S estimate.

    The current gain solver currently excites the source port with a unit current
    and reads the output port voltage-like response. Under the simple Z0
    Norton-port convention:

      transmission: S_ij = 2 V_i / (I_j Z0)
      reflection:   S_jj = 2 V_j / (I_j Z0) - 1

    This gives an absolute gain_db column comparable in form to JC.jl.
    """
    s = 2.0 * response / z0_ohm
    if int(source_port) == int(out_port):
        s -= 1.0
    return s


def gain_db_from_s(s: complex) -> float:
    return 20.0 * np.log10(max(abs(s), 1e-300))

'''
    text = text[:idx] + helper + "\n" + text[idx:]


# ---------------------------------------------------------------------
# 2. Add fields to GainRow dataclass.
# ---------------------------------------------------------------------
if "s_param_abs: float" not in text:
    old = "    gain_vs_off: float\n    gain_vs_off_db: float\n"
    new = "    gain_vs_off: float\n    s_param_abs: float\n    gain_db: float\n    gain_vs_off_db: float\n"
    if old not in text:
        raise SystemExit("Could not find GainRow gain_vs_off/gain_vs_off_db fields.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 3. Add --z0-ohm argument.
# ---------------------------------------------------------------------
if "--z0-ohm" not in text:
    m = re.search(r'(?m)^(\s*)p\.add_argument\("--outdir".*\)\s*$', text)
    if not m:
        m = re.search(r'(?m)^(\s*)p\.add_argument\("--gamma-nt".*\)\s*$', text)
    if not m:
        raise SystemExit("Could not find argparse insertion point.")
    indent = m.group(1)
    add = m.group(0) + f'\n{indent}p.add_argument("--z0-ohm", type=float, default=50.0, help="Port impedance for JC-style S-parameter gain extraction.")'
    text = text[:m.start()] + add + text[m.end():]


# ---------------------------------------------------------------------
# 4. Add s_param_abs and gain_db to GainRow constructor using vout_on.
# ---------------------------------------------------------------------
if "s_param_abs=abs(port_s_from_unit_current_response(vout_on" not in text:
    old = "        gain_vs_off=gain_vs_off,\n        gain_vs_off_db=db10(gain_vs_off),\n"
    new = """        gain_vs_off=gain_vs_off,
        s_param_abs=abs(port_s_from_unit_current_response(vout_on, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
        gain_db=gain_db_from_s(port_s_from_unit_current_response(vout_on, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
        gain_vs_off_db=db10(gain_vs_off),
"""
    if old not in text:
        raise SystemExit("Could not find GainRow constructor gain_vs_off block.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 5. Update one-point detailed prints if present.
# ---------------------------------------------------------------------
if 'print(f"gain_db={r.gain_db:.6f}")' not in text:
    old = '    print(f"gain_vs_off={r.gain_vs_off:.12e}")\n'
    new = '    print(f"s_param_abs={r.s_param_abs:.12e}")\n    print(f"gain_db={r.gain_db:.6f}")\n    print(f"gain_vs_off={r.gain_vs_off:.12e}")\n'
    if old in text:
        text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 6. Update sweep printed header.
# ---------------------------------------------------------------------
old = 'print("\\nsignal_ghz,status,gain_vs_off_db,gain_vs_pumpdiag_db,idler_rel_db,linear_rel_residual,factor_solve_runtime_s")'
new = 'print("\\nsignal_ghz,status,gain_db,s_param_abs,gain_vs_off_db,gain_vs_pumpdiag_db,idler_rel_db,linear_rel_residual,factor_solve_runtime_s")'
if old in text:
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 7. Update per-row printed sweep line.
# ---------------------------------------------------------------------
if 'f"{r.gain_db:.9g},"' not in text:
    old = '        f"{r.gain_vs_off_db:.9g},"'
    new = '        f"{r.gain_db:.9g},"\\n        f"{r.s_param_abs:.9g},"\\n        f"{r.gain_vs_off_db:.9g},"'
    if old in text:
        text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 8. Update CSV fieldnames and values.
# ---------------------------------------------------------------------
if '"gain_db",' not in text:
    old = '        "gain_vs_off_db",\n'
    new = '        "gain_db",\n        "s_param_abs",\n        "gain_vs_off_db",\n'
    if old in text:
        text = text.replace(old, new, 1)

if "r.gain_db," not in text:
    old = "        r.gain_vs_off_db,\n"
    new = "        r.gain_db,\n        r.s_param_abs,\n        r.gain_vs_off_db,\n"
    if old in text:
        text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 9. Add JC-style summary fields to report.
# ---------------------------------------------------------------------
if '"gain_db_max": max(r.gain_db for r in rows)' not in text:
    old = '        "max_gain_vs_off_db": max(r.gain_vs_off_db for r in rows),\n'
    new = '''        "gain_db_max": max(r.gain_db for r in rows),
        "gain_db_mean": float(np.mean([r.gain_db for r in rows])),
        "gain_db_min": min(r.gain_db for r in rows),
        "peak_frequency_ghz": max(rows, key=lambda r: r.gain_db).signal_ghz,
        "nfrequencies": len(rows),
        "z0_ohm": args.z0_ohm,
        "max_gain_vs_off_db": max(r.gain_vs_off_db for r in rows),
'''
    if old in text:
        text = text.replace(old, new, 1)
    else:
        print("Warning: could not find report max_gain_vs_off_db field; CSV/prints still patched.")


# ---------------------------------------------------------------------
# 10. Add final summary prints.
# ---------------------------------------------------------------------
if 'print(f"gain_db_max={max(r.gain_db for r in rows):.6f}")' not in text:
    old = '    print(f"max_gain_vs_off_db={max(r.gain_vs_off_db for r in rows):.6f}")\n'
    new = '''    print(f"gain_db_max={max(r.gain_db for r in rows):.6f}")
    print(f"gain_db_mean={float(np.mean([r.gain_db for r in rows])):.6f}")
    print(f"gain_db_min={min(r.gain_db for r in rows):.6f}")
    print(f"peak_frequency_ghz={max(rows, key=lambda r: r.gain_db).signal_ghz}")
    print(f"nfrequencies={len(rows)}")
    print(f"max_gain_vs_off_db={max(r.gain_vs_off_db for r in rows):.6f}")
'''
    if old in text:
        text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp09 emits JC-style gain_db using vout_on")
print(f"backup={backup}")
