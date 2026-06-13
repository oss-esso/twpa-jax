from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"missing", "nan", "none"}:
        return np.nan
    return float(s.replace(",", "."))


def parse_bool(x):
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def row_solver_nonconverged(row):
    status = str(row.get("status", "")).strip().upper()
    warning_seen = parse_bool(row.get("solver_warning_seen", "false"))
    solver_converged_raw = str(row.get("solver_converged", "")).strip().lower()
    log_text = str(row.get("solver_log_text", "")).lower()

    log_has_warning = (
        "solver did not converge" in log_text
        or "did not converge after maximum iterations" in log_text
        or "maximum iterations" in log_text
        or "max iterations" in log_text
    )

    if status in {"FINITE_NONCONVERGED", "FINITE_NONCONVERGED_DIAGNOSTIC"}:
        return True

    if status in {"NONFINITE", "FAIL", "BRANCH_UNSTABLE_DIAGNOSTIC"}:
        return True

    if warning_seen or log_has_warning:
        return True

    if solver_converged_raw == "false":
        return True

    return False


def row_valid_converged(row):
    status = str(row.get("status", "")).strip().upper()

    # New status-aware scripts.
    if status == "VALID_CONVERGED":
        return not row_solver_nonconverged(row)

    # Older scripts may have written PASS even when warnings existed.
    if status == "PASS":
        return not row_solver_nonconverged(row)

    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Output folder containing report_old_ipm_power_map_rows.csv")
    ap.add_argument("--y-axis", choices=["current", "power"], default="current")
    ap.add_argument("--clip-min", type=float, default=-20.0)
    ap.add_argument("--clip-max", type=float, default=25.0)
    args = ap.parse_args()

    root = Path(args.root)
    csv_path = root / "report_old_ipm_power_map_rows.csv"
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    fps = sorted(set(parse_float(r["pump_frequency_ghz"]) for r in rows))

    if args.y_axis == "current":
        y_values = sorted(set(parse_float(r["pump_current_ua"]) for r in rows))
        y_key = "pump_current_ua"
        y_label = "Pump current (µA)"
        y_tag = "current"
    else:
        y_values = sorted(set(parse_float(r["external_power_dbm"]) for r in rows))
        y_key = "external_power_dbm"
        y_label = "External plot pump power (dBm)"
        y_tag = "power"

    gain = np.full((len(y_values), len(fps)), np.nan)
    valid = np.zeros((len(y_values), len(fps)), dtype=bool)
    nonconv = np.zeros((len(y_values), len(fps)), dtype=bool)
    suspicious = np.zeros((len(y_values), len(fps)), dtype=bool)
    status = np.empty((len(y_values), len(fps)), dtype=object)

    for r in rows:
        fp = parse_float(r["pump_frequency_ghz"])
        y = parse_float(r[y_key])
        i = y_values.index(y)
        j = fps.index(fp)

        g = parse_float(r.get("gain_db_max", "nan"))
        gain[i, j] = g

        is_valid = row_valid_converged(r)
        is_nonconv = row_solver_nonconverged(r)
        is_suspicious = np.isfinite(g) and (g < -100.0 or g > 80.0)

        valid[i, j] = is_valid
        nonconv[i, j] = is_nonconv
        suspicious[i, j] = is_suspicious
        status[i, j] = str(r.get("status", ""))

    clipped_gain = np.clip(gain, args.clip_min, args.clip_max)
    clean_gain = np.where(valid, clipped_gain, np.nan)

    extent = [min(fps), max(fps), min(y_values), max(y_values)]

    # Plot 1: raw/clipped gain with markers over nonconverged cells.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(
        clipped_gain,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"Gain clipped to [{args.clip_min}, {args.clip_max}] dB")

    yy, xx = np.where(nonconv)
    if len(xx):
        ax.scatter(
            [fps[j] for j in xx],
            [y_values[i] for i in yy],
            marker="x",
            s=70,
            linewidths=1.8,
            c="black",
            label="Solver nonconverged / warning",
        )

    yy, xx = np.where(suspicious & ~nonconv)
    if len(xx):
        ax.scatter(
            [fps[j] for j in xx],
            [y_values[i] for i in yy],
            marker="s",
            s=65,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Extreme finite gain",
        )

    ax.set_xlabel("Pump frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title("Old-IPM gain map with nonconverged cells marked")
    if len(ax.collections):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    raw_plot = plots / f"old_ipm_gain_map_{y_tag}_frequency_marked.png"
    fig.savefig(raw_plot, dpi=240)
    plt.close(fig)

    # Plot 2: clean/converged-only gain map.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(
        clean_gain,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"Converged-only gain clipped to [{args.clip_min}, {args.clip_max}] dB")

    yy, xx = np.where(~valid)
    if len(xx):
        ax.scatter(
            [fps[j] for j in xx],
            [y_values[i] for i in yy],
            marker="x",
            s=60,
            linewidths=1.5,
            c="black",
            label="Not valid converged",
        )

    ax.set_xlabel("Pump frequency (GHz)")
    ax.set_ylabel(y_label)
    ax.set_title("Old-IPM gain map: valid-converged cells only")
    if len(ax.collections):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    clean_plot = plots / f"old_ipm_gain_map_{y_tag}_frequency_converged_only.png"
    fig.savefig(clean_plot, dpi=240)
    plt.close(fig)

    # Write summary.
    total = gain.size
    finite_count = int(np.isfinite(gain).sum())
    valid_count = int(valid.sum())
    nonconv_count = int(nonconv.sum())
    suspicious_count = int(suspicious.sum())

    valid_gain = np.where(valid, gain, np.nan)
    best_valid_idx = None
    if np.isfinite(valid_gain).any():
        best = np.nanargmax(valid_gain)
        best_valid_idx = np.unravel_index(best, valid_gain.shape)

    unique_status, counts = np.unique(status.astype(str), return_counts=True)

    md = []
    md.append("# Old-IPM gain map with convergence markers")
    md.append("")
    md.append(f"- rows: `{len(rows)}`")
    md.append(f"- grid cells: `{total}`")
    md.append(f"- finite gain cells: `{finite_count}/{total}`")
    md.append(f"- valid converged cells: `{valid_count}/{total}`")
    md.append(f"- solver nonconverged/warning cells: `{nonconv_count}/{total}`")
    md.append(f"- extreme finite-gain cells: `{suspicious_count}/{total}`")
    md.append(f"- y-axis: `{args.y_axis}`")
    md.append(f"- gain clip: `[{args.clip_min}, {args.clip_max}] dB`")
    md.append("")

    if np.isfinite(gain).any():
        md.append(f"- raw gain min: `{float(np.nanmin(gain))}` dB")
        md.append(f"- raw gain max: `{float(np.nanmax(gain))}` dB")

    if best_valid_idx is not None:
        bi, bj = best_valid_idx
        md.append(f"- best valid-converged gain: `{float(gain[bi, bj])}` dB")
        md.append(f"- best valid-converged point: fp=`{fps[bj]}` GHz, {y_label}=`{y_values[bi]}`")

    md.append("")
    md.append("## Status counts")
    md.append("")
    for s, c in zip(unique_status, counts):
        md.append(f"- `{s}`: `{int(c)}`")

    md.append("")
    md.append("## Generated plots")
    md.append("")
    md.append(f"- `plots/{raw_plot.name}`")
    md.append(f"- `plots/{clean_plot.name}`")
    md.append("")
    md.append("Marker convention:")
    md.append("")
    md.append("- black `x`: solver warning, nonconverged, fail, or nonfinite cell")
    md.append("- hollow square: extreme finite gain without explicit warning")
    md.append("")

    summary_path = root / f"old_ipm_gain_map_{y_tag}_frequency_marked_summary.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("WROTE", raw_plot)
    print("WROTE", clean_plot)
    print("WROTE", summary_path)


if __name__ == "__main__":
    main()
