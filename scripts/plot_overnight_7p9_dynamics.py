"""Generate compact visual diagnostics for the overnight 7.9 GHz campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_records(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for phase in ("coarse", "boundary_refinement"):
        for item in campaign.get(phase, []):
            record = dict(item.get("final", {}))
            record["power_dbm"] = float(item["power_dbm"])
            record["phase"] = phase
            records.append(record)
    return sorted(records, key=lambda item: item["power_dbm"])


def label(record: dict[str, Any]) -> str:
    return str(record.get("regime", "UNRESOLVED"))


def target_plot(record: dict[str, Any], outdir: Path, ramp_periods: int) -> None:
    summary_path = Path(record["summary_path"])
    summary = read_json(summary_path)
    compact_path = summary_path.parent / "td_compact.npz"
    if not compact_path.exists():
        return
    compact = np.load(compact_path)
    period_offset = float(record.get("period_offset", 0.0))
    periods = np.asarray(compact["theta"], dtype=float) / (2.0 * np.pi) + period_offset
    peak = np.asarray(compact["max_abs_sin_phi"], dtype=float)
    min_cos = np.asarray(compact.get("min_cos_phi", []), dtype=float)
    winding_series = np.asarray(compact.get("phase_winding_cycles", []), dtype=float)
    strobe = summary.get("stroboscopic", {})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    axes = axes.ravel()
    axes[0].plot(periods, peak, color="tab:red", linewidth=1.0)
    if min_cos.size == periods.size:
        cos_axis = axes[0].twinx()
        cos_axis.plot(periods, min_cos, color="tab:blue", linewidth=1.0, alpha=0.9)
        cos_axis.set_ylabel("min cos(phi_J)", color="tab:blue")
        cos_axis.tick_params(axis="y", labelcolor="tab:blue")
        cos_axis.grid(False)
    ramp_end = float(record.get("ramp_end_periods", ramp_periods))
    axes[0].axvline(ramp_end, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("peak |I_J| / I_c")
    axes[0].set_title("junction utilization")
    for index, key in enumerate(("d1", "d2", "d3"), start=1):
        values = np.asarray(strobe.get(key, []), dtype=float)
        value_periods = np.asarray((strobe.get("periods_by_n") or {}).get(key, []), dtype=float) + period_offset
        if value_periods.size != values.size:
            value_periods = np.asarray(strobe.get("periods", []), dtype=float)[int(key[1:]):int(key[1:]) + values.size] + period_offset
        axes[index].semilogy(value_periods, np.maximum(values, 1e-16), linewidth=0.9)
        axes[index].axvline(ramp_end, color="k", linestyle="--", linewidth=0.8)
        axes[index].set_ylabel(key)
        axes[index].set_title(f"recurrence {key}")
    for key in ("d4", "d6", "d8"):
        values = np.asarray(strobe.get(key, []), dtype=float)
        if values.size:
            value_periods = np.asarray((strobe.get("periods_by_n") or {}).get(key, []), dtype=float) + period_offset
            axes[4].semilogy(value_periods, np.maximum(values, 1e-16), label=key, linewidth=0.8)
    axes[4].axvline(ramp_end, color="k", linestyle="--", linewidth=0.8)
    axes[4].legend()
    axes[4].set_ylabel("dN")
    axes[4].set_title("selected higher-order recurrence")
    if winding_series.size == periods.size:
        axes[5].plot(periods, winding_series, color="tab:purple", linewidth=0.9)
    else:
        axes[5].axhline(float(summary.get("mean_phase_winding_cycles") or 0.0), color="tab:purple")
    axes[5].axvline(ramp_end, color="k", linestyle="--", linewidth=0.8)
    axes[5].set_ylabel("mean winding (cycles)")
    axes[5].set_title("late mean phase winding")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("total pump periods")
        for checkpoint in (40, 90, 140, 250, 440):
            axis.axvline(checkpoint, color="0.6", linestyle=":", linewidth=0.45, alpha=0.5)
    fig.suptitle(
        f"7.9 GHz 2c | {record['power_dbm']:+.6f} dBm | zero-pump upward ramp | "
        f"{label(record)} | hold {record.get('hold_periods')} | Δθ={record.get('delta_theta')}"
    )
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    name = summary_path.parent.parent.name + "_" + summary_path.parent.name
    fig.savefig(outdir / f"{name}_summary.png", dpi=140)
    plt.close(fig)


def deep_plots(record: dict[str, Any], outdir: Path) -> None:
    summary = read_json(Path(record["summary_path"]))
    strobe = summary.get("stroboscopic", {})
    keys = [f"d{n}" for n in (1, 2, 3, 4, 5, 6, 8, 12, 16)]
    medians = []
    maxima = []
    present = []
    for key in keys:
        values = np.asarray(strobe.get(key, []), dtype=float)
        if values.size:
            values = values[max(0, values.size // 2):]
            present.append(int(key[1:]))
            medians.append(float(np.median(values)))
            maxima.append(float(np.max(values)))
    if not present:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(present, medians, yerr=np.maximum(0.0, np.asarray(maxima) - medians), fmt="o")
    ax.set_yscale("log")
    ax.set_xlabel("period offset N")
    ax.set_ylabel("late dN")
    ax.set_title(f"{record['power_dbm']:+.6f} dBm: late recurrence by N")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{record['power_dbm']:+.6f}_dN.png", dpi=140)
    plt.close(fig)
    pump = np.asarray(strobe.get("pump_flux", []), dtype=float)
    state_norm = np.asarray(strobe.get("state_norm", []), dtype=float)
    if pump.size and state_norm.size:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pump, state_norm, s=7, alpha=0.6)
        ax.set(xlabel="pump-node flux", ylabel="state norm", title="pump-stroboscopic projection")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"{record['power_dbm']:+.6f}_poincare.png", dpi=140)
        plt.close(fig)
    spectrum_path = Path(record["summary_path"]).parent / "late_time_spectrum.npz"
    if spectrum_path.exists():
        spectrum = np.load(spectrum_path)
        freq = np.asarray(spectrum["frequency_ghz"], dtype=float)
        amp = np.maximum(np.asarray(spectrum["amplitude"], dtype=float), 1e-18)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6))
        axes[0].plot(freq, 20.0 * np.log10(amp / np.max(amp)))
        axes[0].set_xlim(0.0, 16.0)
        axes[1].plot(freq, 20.0 * np.log10(amp / np.max(amp)))
        axes[1].set_xlim(0.0, 8.0)
        for ax in axes:
            for marker in (1.975, 2.633333, 3.95, 7.9, 15.8):
                ax.axvline(marker, color="k", alpha=0.25, linewidth=0.7)
            ax.set_ylabel("relative amplitude (dB)")
            ax.grid(True, alpha=0.2)
        axes[1].set_xlabel("frequency (GHz)")
        fig.suptitle(f"{record['power_dbm']:+.6f} dBm late spectrum")
        fig.tight_layout()
        fig.savefig(outdir / f"{record['power_dbm']:+.6f}_spectrum.png", dpi=140)
        plt.close(fig)


def overview(campaign: dict[str, Any], outdir: Path) -> None:
    records = final_records(campaign)
    if not records:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    powers = np.asarray([item["power_dbm"] for item in records])
    utilization = []
    min_cos_values = []
    d1 = []
    best_n = []
    winding = []
    regimes = []
    for item in records:
        summary = read_json(Path(item["summary_path"]))
        compact_path = Path(item["summary_path"]).parent / "td_compact.npz"
        compact = np.load(compact_path) if compact_path.exists() else None
        utilization.append(float(np.max(compact["max_abs_sin_phi"])) if compact is not None else np.nan)
        min_cos_values.append(float(np.min(compact["min_cos_phi"])) if compact is not None and "min_cos_phi" in compact else np.nan)
        strobe = summary.get("stroboscopic", {})
        tail = strobe.get("tail_median_by_n", {})
        if tail:
            finite = [(int(key[1:]), float(value)) for key, value in tail.items() if np.isfinite(value)]
            best_n.append(min(finite, key=lambda pair: pair[1])[0] if finite else np.nan)
        else:
            best_n.append(np.nan)
        d1.append(float(strobe.get("tail_median", np.nan)))
        winding.append(abs(float(summary.get("mean_phase_winding_cycles") or 0.0)))
        regimes.append(label(item))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(powers, utilization, "o-")
    axes[0, 0].set_ylabel("max |I_J| / I_c")
    axes[0, 1].semilogy(powers, np.maximum(d1, 1e-16), "o-")
    axes[0, 1].set_ylabel("late d1")
    axes[1, 0].plot(powers, best_n, "o-")
    axes[1, 0].set_ylabel("N with smallest late dN")
    axes[1, 1].semilogy(powers, np.maximum(winding, 1e-16), "o-")
    axes[1, 1].set_ylabel("|mean winding| (cycles)")
    for ax in axes.ravel():
        ax.set_xlabel("pump power (dBm)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("7.9 GHz 2c independent upward-turn-on overview")
    fig.tight_layout()
    fig.savefig(outdir / "overview_dynamics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cos_ax = ax.twinx()
    ax.plot(powers, utilization, "o-", color="tab:red", label="r_max = max |sin(phi_J)|")
    cos_ax.plot(powers, min_cos_values, "s-", color="tab:blue", label="min cos(phi_J)")
    ax.set_xlabel("pump power (dBm)")
    ax.set_ylabel("r_max = max |I_J| / I_c", color="tab:red")
    cos_ax.set_ylabel("min cos(phi_J)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:red")
    cos_ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.25)
    ax.set_title("7.9 GHz 2c overnight junction stress and tangent margin")
    handles = ax.get_legend_handles_labels()[0] + cos_ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + cos_ax.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "junction_headroom_vs_power.png", dpi=150)
    plt.close(fig)

    categories = {name: index for index, name in enumerate(sorted(set(regimes)))}
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(powers, [categories[name] for name in regimes], s=55)
    ax.set_yticks(list(categories.values()), list(categories.keys()))
    ax.set_xlabel("pump power (dBm)")
    ax.set_title("independently selected dynamical regime")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "regime_vs_power.png", dpi=150)
    plt.close(fig)

    first_nonperiodic = next((item for item in records if label(item) != "PERIOD1"), None)
    last_period1 = next((item for item in reversed(records) if label(item) == "PERIOD1"), None)
    representative = [item for item in (last_period1, first_nonperiodic) if item is not None]
    if representative:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for item in representative:
            summary = read_json(Path(item["summary_path"]))
            strobe = summary.get("stroboscopic", {})
            axes[0].scatter(item["power_dbm"], float(np.max(np.load(Path(item["summary_path"]).parent / "td_compact.npz")["max_abs_sin_phi"])), label=label(item))
            axes[1].scatter(item["power_dbm"], float(strobe.get("tail_median", np.nan)), label=label(item))
            axes[2].scatter(item["power_dbm"], abs(float(summary.get("mean_phase_winding_cycles") or 0.0)), label=label(item))
        axes[0].set_ylabel("peak |I_J| / I_c")
        axes[1].set_ylabel("late d1")
        axes[1].set_yscale("log")
        axes[2].set_ylabel("|mean winding|")
        for ax in axes:
            ax.set_xlabel("pump power (dBm)")
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.suptitle("first boundary representative points")
        fig.tight_layout()
        fig.savefig(outdir / "MASTER_7p9_DYNAMICS_SUMMARY.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--power-dbm", type=float, default=-24.4736842105)
    parser.add_argument("--ramp-periods", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_dir is not None:
        summary_path = args.target_dir / "summary.json"
        summary = read_json(summary_path)
        record = {
            "summary_path": str(summary_path),
            "power_dbm": args.power_dbm,
            "regime": "PERIOD1" if summary.get("classification") == "PERIOD_1" else "UNRESOLVED",
            "hold_periods": summary.get("hold_periods"),
            "delta_theta": 0.05,
        }
        target_plot(record, args.target_dir / "plots", args.ramp_periods)
        if record["regime"] != "PERIOD1":
            deep_plots(record, args.target_dir / "deep_classification")
        return 0
    if args.campaign_dir is None:
        raise ValueError("provide --campaign-dir or --target-dir")
    campaign_path = args.campaign_dir / "campaign_summary.json"
    if not campaign_path.exists():
        raise FileNotFoundError(campaign_path)
    campaign = read_json(campaign_path)
    records = final_records(campaign)
    by_power = args.campaign_dir / "plots" / "by_power"
    deep = args.campaign_dir / "deep_classification"
    for record in records:
        target_plot(record, by_power, int(campaign.get("ramp_periods", 40)))
        if label(record) != "PERIOD1":
            deep_plots(record, deep)
    overview(campaign, args.campaign_dir / "plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
