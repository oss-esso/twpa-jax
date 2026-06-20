from pathlib import Path
import re

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_jc_style_gain_v2")
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
    """Convert unit-current transimpedance to JC-style port S.

    Transmission: S_ij = 2 Z_ij / Z0
    Reflection:   S_jj = 2 Z_jj / Z0 - 1
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
    old = "    gain_vs_off_db: float\n"
    new = "    s_param_abs: float\n    gain_db: float\n    gain_vs_off_db: float\n"
    if old not in text:
        raise SystemExit("Could not find GainRow gain_vs_off_db field.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 3. Add --z0-ohm argument.
# ---------------------------------------------------------------------
if "--z0-ohm" not in text:
    # Put it after --outdir if possible, otherwise after --gamma-nt.
    m = re.search(r'(?m)^(\s*)p\.add_argument\("--outdir".*\)\s*$', text)
    if not m:
        m = re.search(r'(?m)^(\s*)p\.add_argument\("--gamma-nt".*\)\s*$', text)
    if not m:
        raise SystemExit("Could not find argparse insertion point.")
    indent = m.group(1)
    add = m.group(0) + f'\n{indent}p.add_argument("--z0-ohm", type=float, default=50.0, help="Port impedance for JC-style S-parameter gain extraction.")'
    text = text[:m.start()] + add + text[m.end():]


# ---------------------------------------------------------------------
# 4. Infer the variable containing the full pumped central response.
#    Usually code has: gain_vs_off = abs(full) / ...
# ---------------------------------------------------------------------
full_var = None
patterns = [
    r"gain_vs_off\s*=\s*abs\((\w+)\)\s*/",
    r"gain_vs_off\s*=\s*np\.abs\((\w+)\)\s*/",
    r"gain_vs_off\s*=\s*safe_abs_ratio\((\w+),",
]
for pat in patterns:
    m = re.search(pat, text)
    if m:
        full_var = m.group(1)
        break

if full_var is None:
    print("Could not infer full response variable. Nearby lines:")
    for line in text.splitlines():
        if "gain_vs_off" in line or "GainRow(" in line:
            print(line)
    raise SystemExit("Patch failed before writing.")

print(f"Inferred full response variable: {full_var}")


# ---------------------------------------------------------------------
# 5. Add s_param_abs and gain_db to GainRow constructor.
# ---------------------------------------------------------------------
if "s_param_abs=abs(port_s_from_unit_current_response(" not in text:
    old = "        gain_vs_off_db=db10(gain_vs_off),\n"
    new = f"""        s_param_abs=abs(port_s_from_unit_current_response({full_var}, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
        gain_db=gain_db_from_s(port_s_from_unit_current_response({full_var}, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
        gain_vs_off_db=db10(gain_vs_off),
"""
    if old not in text:
        raise SystemExit("Could not find GainRow constructor gain_vs_off_db argument.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 6. Update printed sweep header.
# ---------------------------------------------------------------------
old = 'print("\\nsignal_ghz,status,gain_vs_off_db,gain_vs_pumpdiag_db,idler_rel_db,linear_rel_residual,factor_solve_runtime_s")'
new = 'print("\\nsignal_ghz,status,gain_db,s_param_abs,gain_vs_off_db,gain_vs_pumpdiag_db,idler_rel_db,linear_rel_residual,factor_solve_runtime_s")'
if old in text:
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 7. Update per-row CSV-ish print.
# ---------------------------------------------------------------------
if 'f"{r.gain_db:.9g},"' not in text:
    old = '        f"{r.gain_vs_off_db:.9g},"'
    new = '        f"{r.gain_db:.9g},"\\n        f"{r.s_param_abs:.9g},"\\n        f"{r.gain_vs_off_db:.9g},"'
    if old in text:
        text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 8. Update CSV fieldnames/list writer.
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
# 9. Add JC-style summary to report dict and final print.
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
        print("Warning: could not find report max_gain_vs_off_db field; CSV will still contain gain_db.")

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
print("PATCH_OK Exp09 dataclass version emits JC-style gain_db")
print(f"backup={backup}")
