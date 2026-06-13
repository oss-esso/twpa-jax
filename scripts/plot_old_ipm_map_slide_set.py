from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"missing", "nan", "none"}:
        return np.nan
    return float(s.replace(",", "."))


def b(x):
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def is_nonconverged(row):
    status = str(row.get("status", "")).strip().upper()
    warning_seen = b(row.get("solver_warning_seen", "false"))
    solver_converged = str(row.get("solver_converged", "")).strip().lower()
    log_text = str(row.get("solver_log_text", "")).lower()

    if status in {"FINITE_NONCONVERGED", "FAIL", "NONFINITE", "BRANCH_UNSTABLE_DIAGNOSTIC"}:
        return True
    if warning_seen:
        return True
    if solver_converged == "false":
        return True
    if "solver did not converge" in log_text:
        return True
    return False


def is_valid(row):
    status = str(row.get("status", "")).strip().upper()
    return status in {"VALID_CONVERGED", "PASS"} and not is_nonconverged(row)


def build_grid(rows, y_key):
    fps = sorted(set(f(r["pump_frequency_ghz"]) for r in rows))
    ys = sorted(set(f(r[y_key]) for r in rows))

    gain = np.full((len(ys), len(fps)), np.nan)
    valid = np.zeros((len(ys), len(fps)), dtype=bool)
    nonconv = np.zeros((len(ys), len(fps)), dtype=bool)

    for r in rows:
        i = ys.index(f(r[y_key]))
        j = fps.index(f(r["pump_frequency_ghz"]))
        gain[i, j] = f(r.get("gain_db_max", "nan"))
        valid[i, j] = is_valid(r)
        nonconv[i, j] = is_nonconverged(r)

    return fps, ys, gain, valid, nonconv


def save_plot(path, fps, ys, gain, valid, nonconv, title, ylabel, clip_min, clip_max, marked=False, converged_only=False):
    plot_gain = np.array(gain, copy=True)

    if converged_only:
        plot_gain[~valid] = np.nan

    plot_gain = np.clip(plot_gain, clip_min, clip_max)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    im = ax.imshow(
        plot_gain,
        origin="lower",
        aspect="auto",
        extent=[min(fps), max(fps), min(ys), max(ys)],
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"Gain clipped to [{clip_min}, {clip_max}] dB")

    if marked:
        yy, xx = np.where(nonconv)
        if len(xx):
            ax.scatter(
                [fps[j] for j in xx],
                [ys[i] for i in yy],
                marker="x",
                s=18,
                linewidths=0.9,
                c="black",
                alpha=0.7,
                label="Nonconverged / warning",
            )
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Pump frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clip-min", type=float, default=-20.0)
    ap.add_argument("--clip-max", type=float, default=25.0)
    args = ap.parse_args()

    root = Path(args.root)
    csv_path = root / "report_old_ipm_power_map_rows.csv"
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # Current-axis plots
    fps, ys, gain, valid, nonconv = build_grid(rows, "pump_current_ua")
    save_plot(
        plots / "old_ipm_map_current_unmarked.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map",
        "Pump current (µA)",
        args.clip_min, args.clip_max,
        marked=False, converged_only=False,
    )
    save_plot(
        plots / "old_ipm_map_current_marked.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map with nonconverged cells marked",
        "Pump current (µA)",
        args.clip_min, args.clip_max,
        marked=True, converged_only=False,
    )
    save_plot(
        plots / "old_ipm_map_current_converged_only.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map (valid-converged cells only)",
        "Pump current (µA)",
        args.clip_min, args.clip_max,
        marked=False, converged_only=True,
    )

    # Power-axis plots
    fps, ys, gain, valid, nonconv = build_grid(rows, "external_power_dbm")
    save_plot(
        plots / "old_ipm_map_power_unmarked.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map",
        "External pump power (dBm)",
        args.clip_min, args.clip_max,
        marked=False, converged_only=False,
    )
    save_plot(
        plots / "old_ipm_map_power_marked.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map with nonconverged cells marked",
        "External pump power (dBm)",
        args.clip_min, args.clip_max,
        marked=True, converged_only=False,
    )
    save_plot(
        plots / "old_ipm_map_power_converged_only.png",
        fps, ys, gain, valid, nonconv,
        "Old-IPM gain map (valid-converged cells only)",
        "External pump power (dBm)",
        args.clip_min, args.clip_max,
        marked=False, converged_only=True,
    )

    print("WROTE plots to", plots)


if __name__ == "__main__":
    main()
