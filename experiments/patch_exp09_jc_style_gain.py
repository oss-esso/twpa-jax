from pathlib import Path
import re

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_jc_style_gain")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------
# Add helper functions.
# ---------------------------------------------------------------------
if "def port_s_from_unit_current_response(" not in text:
    marker = "def db_ratio("
    idx = text.find(marker)
    if idx < 0:
        marker = "def main("
        idx = text.find(marker)
    if idx < 0:
        raise SystemExit("Could not find insertion point before db_ratio/main.")

    helper = r'''

def port_s_from_unit_current_response(
    response: complex,
    *,
    source_port: int,
    out_port: int,
    z0_ohm: float,
) -> complex:
    """Convert unit-current transimpedance response to a port S-parameter.

    The linear solve uses a unit Norton current source at the input port.
    With a Z0 port normalization, the corresponding scattering conversion is:

      S_ij = 2 Z_ij / Z0          for i != j
      S_jj = 2 Z_jj / Z0 - 1      for reflection

    This is the quantity to compare against JC.jl gain_db outputs.
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
# Add --z0-ohm CLI arg.
# ---------------------------------------------------------------------
if "--z0-ohm" not in text:
    m = re.search(r'(?m)^(\s*)p\.add_argument\("--outdir".*\)\s*$', text)
    if not m:
        raise SystemExit("Could not find --outdir argparse line.")
    indent = m.group(1)
    insert = m.group(0) + f'\n{indent}p.add_argument("--z0-ohm", type=float, default=50.0, help="Port impedance used for JC-style S-parameter gain extraction.")'
    text = text[:m.start()] + insert + text[m.end():]


# ---------------------------------------------------------------------
# Add JC-style gain columns to row dict.
# We expect existing row keys gain_vs_off_db and gain_vs_pumpdiag_db.
# ---------------------------------------------------------------------
needle = '            "gain_vs_off_db": db_ratio(full, off),\n'
if needle in text and '"gain_db": gain_db_from_s(' not in text:
    replacement = '''            "s_param_abs": abs(port_s_from_unit_current_response(full, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
            "gain_db": gain_db_from_s(port_s_from_unit_current_response(full, source_port=args.source_port, out_port=args.out_port, z0_ohm=args.z0_ohm)),
            "gain_vs_off_db": db_ratio(full, off),
'''
    text = text.replace(needle, replacement, 1)
elif '"gain_db": gain_db_from_s(' in text:
    pass
else:
    print("Could not find exact row insertion point. Nearby gain lines:")
    for line in text.splitlines():
        if "gain_vs_off_db" in line or "rows.append" in line:
            print(line)
    raise SystemExit("Patch failed before writing.")


# ---------------------------------------------------------------------
# Add JC-style summary fields.
# ---------------------------------------------------------------------
if 'gain_values = [float(r["gain_db"]) for r in rows if r.get("status") == "VALID_SOLVED"]' not in text:
    marker = '    max_gain_off = max(r["gain_vs_off_db"] for r in rows)\n'
    if marker not in text:
        print("Could not find max_gain_off marker. Nearby summary lines:")
        for line in text.splitlines():
            if "max_gain" in line or "gain_report" in line or "report =" in line:
                print(line)
        raise SystemExit("Patch failed before writing.")
    insert = '''    gain_values = [float(r["gain_db"]) for r in rows if r.get("status") == "VALID_SOLVED"]
    if gain_values:
        gain_db_max = max(gain_values)
        gain_db_min = min(gain_values)
        gain_db_mean = float(np.mean(gain_values))
        peak_frequency_ghz = float(max(rows, key=lambda r: r.get("gain_db", -1e300))["signal_ghz"])
    else:
        gain_db_max = None
        gain_db_min = None
        gain_db_mean = None
        peak_frequency_ghz = None

'''
    text = text.replace(marker, insert + marker, 1)

# Add fields to report dict near existing max_gain_vs_off_db.
if '"gain_db_max": gain_db_max' not in text:
    needle = '        "max_gain_vs_off_db": max_gain_off,\n'
    replacement = '''        "gain_db_max": gain_db_max,
        "gain_db_mean": gain_db_mean,
        "gain_db_min": gain_db_min,
        "peak_frequency_ghz": peak_frequency_ghz,
        "nfrequencies": len(rows),
        "z0_ohm": args.z0_ohm,
        "max_gain_vs_off_db": max_gain_off,
'''
    if needle not in text:
        raise SystemExit("Could not find report max_gain_vs_off_db field.")
    text = text.replace(needle, replacement, 1)

# Add final print lines.
if 'print(f"gain_db_max={gain_db_max}")' not in text:
    needle = '    print(f"max_gain_vs_off_db={max_gain_off:.6f}")\n'
    replacement = '''    print(f"gain_db_max={gain_db_max}")
    print(f"gain_db_mean={gain_db_mean}")
    print(f"gain_db_min={gain_db_min}")
    print(f"peak_frequency_ghz={peak_frequency_ghz}")
    print(f"nfrequencies={len(rows)}")
    print(f"max_gain_vs_off_db={max_gain_off:.6f}")
'''
    if needle in text:
        text = text.replace(needle, replacement, 1)
    else:
        # Non-blocking; report JSON/CSV are enough.
        pass

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp09 now emits JC-style S-parameter gain_db")
print(f"backup={backup}")
