from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_curve(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    fieldnames = set(rows[0].keys())

    freq_key = None
    for k in ["signal_ghz", "frequency_ghz", "freq_ghz", "f_ghz"]:
        if k in fieldnames:
            freq_key = k
            break
    if freq_key is None:
        raise RuntimeError(f"Could not find frequency column in {path}. Found: {sorted(fieldnames)}")

    gain_key = None
    for k in ["gain_db", "gain_db_max"]:
        if k in fieldnames:
            gain_key = k
            break
    if gain_key is None:
        raise RuntimeError(f"Could not find gain column in {path}. Found: {sorted(fieldnames)}")

    freq = np.array([float(r[freq_key]) for r in rows], dtype=float)
    gain = np.array([float(r[gain_key]) for r in rows], dtype=float)

    order = np.argsort(freq)
    freq = freq[order]
    gain = gain[order]

    return freq, gain


def interp_to_reference(x_ref, x, y):
    if np.array_equal(x_ref, x):
        return y
    return np.interp(x_ref, x, y)


def summarize(label, freq, gain):
    i = int(np.argmax(gain))
    print(f"{label}:")
    print(f"  points = {len(freq)}")
    print(f"  gain_db_max = {np.max(gain):.9f}")
    print(f"  gain_db_mean = {np.mean(gain):.9f}")
    print(f"  gain_db_min = {np.min(gain):.9f}")
    print(f"  peak_frequency_ghz = {freq[i]:.9f}")
    print()


def main():
    jc_csv = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_compare\jc_jpa_curve.csv")
    ours_csv = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_jpa_scale2_exp09_501\gain_sweep.csv")
    outdir = Path(r"D:\Projects\Thesis\twpa_jax\outputs\exp13_compare")
    outdir.mkdir(parents=True, exist_ok=True)

    jc_f, jc_g = load_curve(jc_csv)
    our_f, our_g = load_curve(ours_csv)

    summarize("JC", jc_f, jc_g)
    summarize("OURS", our_f, our_g)

    our_g_on_jc = interp_to_reference(jc_f, our_f, our_g)
    err = our_g_on_jc - jc_g

    peak_jc_idx = int(np.argmax(jc_g))
    peak_our_idx = int(np.argmax(our_g))

    print("Comparison:")
    print(f"  max abs error (dB) = {np.max(np.abs(err)):.9f}")
    print(f"  mean abs error (dB) = {np.mean(np.abs(err)):.9f}")
    print(f"  rms error (dB) = {np.sqrt(np.mean(err**2)):.9f}")
    print(f"  signed mean error (dB) = {np.mean(err):.9f}")
    print(f"  JC peak freq (GHz) = {jc_f[peak_jc_idx]:.9f}")
    print(f"  OUR peak freq (GHz) = {our_f[peak_our_idx]:.9f}")
    print(f"  peak freq error (GHz) = {our_f[peak_our_idx] - jc_f[peak_jc_idx]:.9f}")

    # Overlay plot
    plt.figure(figsize=(10, 6))
    plt.plot(jc_f, jc_g, label="JC")
    plt.plot(our_f, our_g, label="OURS")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Gain (dB)")
    plt.title("JPA gain vs frequency at fixed pump")
    plt.grid(True)
    plt.legend()
    overlay_path = outdir / "jpa_gain_overlay.png"
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160)
    plt.close()

    # Error plot
    plt.figure(figsize=(10, 4.8))
    plt.plot(jc_f, err)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("OURS - JC (dB)")
    plt.title("JPA gain error vs frequency")
    plt.grid(True)
    err_path = outdir / "jpa_gain_error.png"
    plt.tight_layout()
    plt.savefig(err_path, dpi=160)
    plt.close()

    # Comparison CSV
    cmp_csv = outdir / "jpa_gain_compare.csv"
    with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_ghz", "jc_gain_db", "our_gain_db", "error_db"])
        for fghz, gjc, gour, e in zip(jc_f, jc_g, our_g_on_jc, err):
            w.writerow([fghz, gjc, gour, e])

    print()
    print(f"wrote_overlay = {overlay_path}")
    print(f"wrote_error = {err_path}")
    print(f"wrote_compare_csv = {cmp_csv}")


if __name__ == "__main__":
    main()
