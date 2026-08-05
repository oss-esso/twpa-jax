"""Plot KIMPA 3WM gain, S11 amplitude/phase, and detuning maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args(argv)
    run_dir = args.run_dir
    outdir = args.outdir or run_dir / "plots_3wm"
    outdir.mkdir(parents=True, exist_ok=True)
    spectrum = list(csv.DictReader((run_dir / "best_point_spectrum.csv").open(encoding="utf-8")))
    freq = np.asarray([float(r["signal_ghz"]) for r in spectrum])
    s11_re = np.asarray([float(r["s11_real"]) for r in spectrum])
    s11_im = np.asarray([float(r["s11_imag"]) for r in spectrum])
    gain = np.asarray([float(r["gain_db"]) for r in spectrum])
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(freq, 20.0 * np.log10(np.maximum(np.hypot(s11_re, s11_im), 1e-300)))
    axes[0].set_ylabel("|S11| (dB)")
    axes[0].set_title("KIMPA 3WM S11 at best map pump")
    axes[1].plot(freq, np.unwrap(np.angle(s11_re + 1j * s11_im)) * 180.0 / np.pi)
    axes[1].set_xlabel("signal frequency (GHz)")
    axes[1].set_ylabel("phase S11 (deg)")
    fig.tight_layout()
    fig.savefig(outdir / "s11_amplitude_phase_8_to_10GHz.png", dpi=180)
    plt.close(fig)

    rows = list(csv.DictReader((run_dir / "kimpa_gain_map.csv").open(encoding="utf-8")))
    pumps = sorted({float(r["pump_dbm_internal"]) for r in rows})
    signals = sorted({float(r["signal_ghz"]) for r in rows})
    matrix = np.full((len(pumps), len(signals)), np.nan)
    ratio = np.full(len(pumps), np.nan)
    for r in rows:
        ip = pumps.index(float(r["pump_dbm_internal"]))
        jf = signals.index(float(r["signal_ghz"]))
        matrix[ip, jf] = float(r["gain_db"])
        ratio[ip] = float(r["max_current_over_ic"])
    detuning = np.asarray(signals) - float(json_from_summary(run_dir)["pump_ghz"]) / 2.0
    fig, ax = plt.subplots(figsize=(8, 5))
    for index in np.linspace(0, len(pumps) - 1, min(6, len(pumps))).astype(int):
        ax.plot(detuning, matrix[index], label=f"I/Ic={ratio[index]:.3f}")
    ax.set_xlabel("Δf_s = f_s − f_p/2 (GHz)")
    ax.set_ylabel("reflection gain (dB)")
    ax.set_title("KIMPA 3WM gain traces")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "gain_vs_detuning_traces.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", extent=[detuning[0], detuning[-1], ratio[0], ratio[-1]], cmap="magma")
    ax.axhline(1.0, color="cyan", linestyle="--", label="I/Ic=1")
    ax.set_xlabel("Δf_s = f_s − f_p/2 (GHz)")
    ax.set_ylabel("peak I/Ic")
    ax.set_title("KIMPA 3WM gain map")
    fig.colorbar(image, ax=ax, label="reflection gain (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "gain_vs_detuning_vs_I_over_Ic.png", dpi=180)
    plt.close(fig)
    print(f"wrote={outdir}")
    return 0


def json_from_summary(run_dir: Path) -> dict[str, object]:
    import json
    return json.loads((run_dir / "map_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
