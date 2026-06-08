from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

H5 = Path(r"D:\Projects\Thesis\Harmonia\IPM_JTWPA\2D_HeatingPlot\2026-04-22\18-32-02_2D-Pump-Frequency-and-Power-sweep-for-IPM-JTWPA\PumpSweep.h5")
OUT = Path(r"D:\Projects\Thesis\outputs\jc_profiles\jc3m_existing_ipm_pumpsweep_h5")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def walk_h5(name, obj):
    kind = "GROUP" if isinstance(obj, h5py.Group) else f"DATASET shape={obj.shape} dtype={obj.dtype}"
    print(f"{name}: {kind}")

def read_axis(file, candidates):
    for key in candidates:
        if key in file:
            return np.array(file[key]), key
    raise KeyError(f"Could not find any axis from {candidates}")

def scalarize_string(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)

def gain_from_smatrix(s):
    arr = np.array(s)
    # Most likely 2x2 complex S matrix. Use S21 = row 2, col 1 in 1-indexed notation.
    if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 1:
        s21 = arr[1, 0]
    elif arr.ndim == 3 and arr.shape[0] >= 2 and arr.shape[1] >= 1:
        s21 = arr[1, 0, 0]
    else:
        flat = arr.reshape(-1)
        s21 = flat[0]
    return 20.0 * np.log10(np.abs(s21))

with h5py.File(H5, "r") as f:
    print("=== HDF5 tree ===")
    f.visititems(walk_h5)

    pump_powers, power_key = read_axis(f, [
        "simulation_info/pump_powers_dBm_axis",
        "simulation_info/pump_powers_dBm",
        "pump_powers_dBm_axis",
        "pump_powers_dBm",
    ])
    pump_freqs, freq_key = read_axis(f, [
        "simulation_info/pump_freqs_GHz_axis",
        "simulation_info/pump_freqs_GHz",
        "pump_freqs_GHz_axis",
        "pump_freqs_GHz",
    ])

    pump_powers = pump_powers.astype(float)
    pump_freqs = pump_freqs.astype(float)

    print("\n=== axes ===")
    print("power_key =", power_key)
    print("freq_key  =", freq_key)
    print("pump_powers_dBm shape =", pump_powers.shape, "min =", pump_powers.min(), "max =", pump_powers.max())
    print("pump_freqs_GHz shape  =", pump_freqs.shape, "min =", pump_freqs.min(), "max =", pump_freqs.max())
    print("pump_powers_dBm =", pump_powers)
    print("pump_freqs_GHz  =", pump_freqs)

    result_group = f["results"] if "results" in f else f

    keys = []
    def collect_dataset_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    result_group.visititems(collect_dataset_keys)

    s_keys = [k for k in keys if "S_matrix" in k or k.lower().startswith("s")]
    print("\n=== result S keys ===")
    print("n_s_keys =", len(s_keys))
    for k in s_keys[:20]:
        print(k)
    if len(s_keys) > 20:
        print("...")

    gain = np.full((len(pump_powers), len(pump_freqs)), np.nan)

    # Try exact suffix convention first: S_matrix_P{P}dBm_F{F}GHz
    for i, p in enumerate(pump_powers):
        for j, fp in enumerate(pump_freqs):
            candidates = [
                f"S_matrix_P{p}dBm_F{fp}GHz",
                f"results/S_matrix_P{p}dBm_F{fp}GHz",
                f"S_matrix_P{float(p)}dBm_F{float(fp)}GHz",
                f"results/S_matrix_P{float(p)}dBm_F{float(fp)}GHz",
            ]
            found = None
            for c in candidates:
                rel = c.removeprefix("results/")
                if rel in result_group:
                    found = rel
                    break
            if found is not None:
                gain[i, j] = gain_from_smatrix(result_group[found][()])

    # Fallback: parse all S_matrix keys.
    if np.isnan(gain).all():
        pat = re.compile(r"S_matrix_P(?P<p>[-+0-9.]+)dBm_F(?P<f>[-+0-9.]+)GHz")
        for key in s_keys:
            m = pat.search(key)
            if not m:
                continue
            p = float(m.group("p"))
            fp = float(m.group("f"))
            i = int(np.argmin(np.abs(pump_powers - p)))
            j = int(np.argmin(np.abs(pump_freqs - fp)))
            if abs(pump_powers[i] - p) < 1e-6 and abs(pump_freqs[j] - fp) < 1e-6:
                gain[i, j] = gain_from_smatrix(result_group[key][()])

    print("\n=== gain map ===")
    print("shape =", gain.shape)
    print("finite =", np.isfinite(gain).sum(), "/", gain.size)
    if np.isfinite(gain).any():
        best_idx = np.unravel_index(np.nanargmax(gain), gain.shape)
        print("gain min =", np.nanmin(gain))
        print("gain max =", np.nanmax(gain))
        print("best power dBm =", pump_powers[best_idx[0]])
        print("best pump freq GHz =", pump_freqs[best_idx[1]])
        print("best gain dB =", gain[best_idx])

    np.savetxt(OUT / "existing_ipm_gain_map.csv", gain, delimiter=",")

    plt.figure()
    plt.imshow(
        gain,
        origin="lower",
        aspect="auto",
        extent=[pump_freqs.min(), pump_freqs.max(), pump_powers.min(), pump_powers.max()],
    )
    plt.colorbar(label="Gain from saved S21 (dB)")
    plt.xlabel("Pump frequency (GHz)")
    plt.ylabel("Pump power (dBm)")
    plt.title("Existing IPM/JTWPA PumpSweep.h5 gain map")
    plt.tight_layout()
    plt.savefig(PLOTS / "existing_ipm_pumpsweep_gain_map.png", dpi=220)
    plt.close()

    md = [
        "# Existing IPM/JTWPA PumpSweep.h5 inspection",
        "",
        f"- source: `{H5}`",
        f"- power axis key: `{power_key}`",
        f"- frequency axis key: `{freq_key}`",
        f"- pump power range: `{pump_powers.min()}` to `{pump_powers.max()}` dBm, n={len(pump_powers)}",
        f"- pump frequency range: `{pump_freqs.min()}` to `{pump_freqs.max()}` GHz, n={len(pump_freqs)}",
        f"- finite gain cells: `{int(np.isfinite(gain).sum())}/{gain.size}`",
    ]

    if np.isfinite(gain).any():
        best_idx = np.unravel_index(np.nanargmax(gain), gain.shape)
        md += [
            f"- gain min: `{float(np.nanmin(gain))}` dB",
            f"- gain max: `{float(np.nanmax(gain))}` dB",
            f"- best point: P=`{float(pump_powers[best_idx[0]])}` dBm, fp=`{float(pump_freqs[best_idx[1]])}` GHz",
        ]

    md += [
        "",
        "## Generated files",
        "",
        "- `existing_ipm_gain_map.csv`",
        "- `plots/existing_ipm_pumpsweep_gain_map.png`",
    ]

    (OUT / "existing_ipm_pumpsweep_h5_inspection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nWROTE", OUT / "existing_ipm_pumpsweep_h5_inspection.md")
    print("WROTE", PLOTS / "existing_ipm_pumpsweep_gain_map.png")
