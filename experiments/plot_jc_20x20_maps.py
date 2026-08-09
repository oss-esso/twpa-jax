"""Plot the two native-JosephsonCircuits 20x20 gain maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "outputs" / "jc_jtwpa_jc_20x20_halfcurrent",
    ROOT / "outputs" / "jc_ipm_2c_20x20_halfcurrent",
]
PLOT_VMIN = 0.0
PLOT_VMAX = 40.0


def load_grid(run: Path):
    frame = pd.read_csv(run / "gain_db_grid.csv")
    power = frame.iloc[:, 0].to_numpy(float)
    freq = np.array([float(c.removeprefix("fp_").removesuffix("_ghz")) for c in frame.columns[1:]])
    gain = frame.iloc[:, 1:].to_numpy(float)
    return power, freq, gain


def edges(values):
    step = np.diff(values).mean()
    return np.r_[values[0] - step / 2, values[:-1] + step / 2, values[-1] + step / 2]


def main():
    loaded = [(run.name, *load_grid(run)) for run in RUNS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    mesh = None
    for ax, (name, power, freq, gain) in zip(axes, loaded):
        mesh = ax.pcolormesh(edges(freq), edges(power), gain, shading="auto",
                             cmap="viridis", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Pump frequency $f_p$ (GHz)")
        ax.set_xticks(np.linspace(freq[0], freq[-1], 6))
        ax.grid(color="white", alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Pump power (dBm)")
    fig.colorbar(mesh, ax=axes, label="Gain (dB; values below 0 dB clipped)", shrink=0.92)
    fig.suptitle("Native JosephsonCircuits.jl maps — half-current convention")

    out = RUNS[0] / "jc_maps_20x20_halfcurrent.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    for name, power, freq, gain in loaded:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
        mesh = ax.pcolormesh(edges(freq), edges(power), gain, shading="auto",
                             cmap="viridis", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
        ax.set_xlabel("Pump frequency $f_p$ (GHz)")
        ax.set_ylabel("Pump power (dBm)")
        ax.set_title(name.replace("_", " "))
        fig.colorbar(mesh, ax=ax, label="Gain (dB)")
        fig.savefig(ROOT / "outputs" / name / "gain_map.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
