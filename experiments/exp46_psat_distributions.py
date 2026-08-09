"""Distribution of the saturation power for the 2c model and the Themis cube.

Saturation power is the output-referred 1 dB compression point,

    P_sat = P_1dB(input) + G0 - 1 dB

The -1 dB is not cosmetic: by definition the gain at the compression point has
already fallen to G0 - 1, so the output power there is the input plus the
*compressed* gain, not plus G0. The solver already uses this definition --
``scripts/run_compression.py:956`` computes ``p1db_output_dbm`` exactly this
way -- and the model values here are read from that field rather than
recomputed. The naive ``P_1dB + G0`` is a rigid 1 dB shift: it changes the
quoted number but not the shape of any distribution below.

Three panels are produced because P_sat is a composite. A shift in P_sat can
come from the compression point, from the small-signal gain, or from both, and
only the decomposition says which.

Measured-column extraction is offered in two modes, because the choice matters
by about 1.7 dB in the median:

``first``       the first threshold crossing. Identical to
                ``experiments/exp45_p1db_two_way_comparison.py``, so the numbers
                connect to the published two-way figure. Below about -128 dBm
                the VNA response swings by +-2 dB point to point, and a smoothed
                trace can dip across the threshold on noise alone; 13 of 1067
                columns land within 5 dB of the sweep floor this way.
``persistent``  the crossing after which the trace never recovers above the
                threshold. Compression that begins and does not undo itself.
                Removes 12 of those 13 and raises the median P1dB by 1.69 dB.

The cube is a single pump frequency already (7.256 GHz); its 2001 columns are
signal frequencies at 4 MHz spacing over 4-12 GHz. ``--match`` chooses how the
third, like-for-like group is drawn from those columns:

``gain``        columns whose G0 lies inside the model's own G0 span. Matches
                operating regime, ignores frequency.
``frequency``   the column nearest each of the model's 18 signal frequencies.
                Matches the sampling, ignores regime -- so the two groups are
                paired point by point and the difference is a per-frequency
                residual, not a distribution overlap.

    python experiments/exp46_psat_distributions.py --match frequency
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]

CUBE = ROOT / (
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
SIGNAL_LINE_LOSS_DB = 72.5
MEAS_PUMP_GHZ = 7.256
PUMP_EXCLUSION_GHZ = 0.15
MIN_MEASURED_G0_DB = 3.0
GAIN_MATCH_PAD_DB = 0.25
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 2

MODEL_COLOUR = "#1f6feb"
MEAS_COLOUR = "#d1495b"
MATCH_COLOUR = "#2a9d8f"


def measured_table(
    crossing: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-column measured (frequency GHz, G0 dB, P1dB dBm, usable mask).

    All 2001 columns are returned, not only the usable ones: frequency matching
    has to ask what sits nearest a requested frequency before it can ask whether
    that column passed the filters.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    response = np.asarray(data["Response"], dtype=float)
    # G0 = the single lowest-power row, unsmoothed. exp48 checked this against
    # the two hand-extracted reference points (Images/Saturation_*.png) and it
    # reproduces both to <0.2 dB; a median over the first 10 power rows reads
    # 0.19-0.33 dB low there, which the ~0.12-0.24 dB/dB knee slope amplifies
    # into a 1-3 dB P1dB error -- see g0_raw_vs_frequency.png.
    g0 = response[0, :]

    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        smooth = savgol_filter(response[:, index], SAVGOL_WINDOW, SAVGOL_ORDER)
        threshold = g0[index] - 1.0
        if crossing == "persistent":
            above = np.flatnonzero(smooth >= threshold)
            if above.size == 0 or above[-1] >= smooth.size - 1:
                continue
            knee = int(above[-1])
        else:
            crossings = np.flatnonzero(
                (smooth[:-1] >= threshold) & (smooth[1:] < threshold)
            )
            if crossings.size == 0:
                continue
            knee = int(crossings[0])
        p1db[index] = np.interp(
            threshold,
            [smooth[knee + 1], smooth[knee]],
            [power[knee + 1], power[knee]],
        )

    usable = (
        (np.abs(frequency - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ)
        & (g0 > MIN_MEASURED_G0_DB)
        & np.isfinite(p1db)
    )
    return frequency, g0, p1db, usable


def model_table(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frequency model (frequency GHz, G0 dB, P1dB dBm, P_sat dBm)."""
    frequency, g0, p1db, psat = [], [], [], []
    for directory in sorted(run_dir.glob("frequency_*")):
        summary_path = directory / "compression_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        gain = summary.get("small_signal_gain_vs_off_db")
        output = summary.get("p1db_output_dbm")
        if match is None or summary.get("p1db") is None or gain is None or output is None:
            continue
        frequency.append(float(match.group(1)))
        g0.append(float(gain))
        p1db.append(float(summary["p1db"]))
        psat.append(float(output))
    return np.asarray(frequency), np.asarray(g0), np.asarray(p1db), np.asarray(psat)


def match_columns(
    measured_ghz: np.ndarray,
    usable: np.ndarray,
    values: dict[str, np.ndarray],
    targets_ghz: np.ndarray,
    band_mhz: float,
    min_fill: float,
) -> list[dict[str, float | bool | int]]:
    """Aggregate the measured columns near each target frequency.

    ``band_mhz`` of 0 takes the single nearest column. That is the naive
    pairing, and it is dominated by extraction noise: columns 4 MHz apart --
    far below any physical ripple scale -- disagree by about 3.2 dB in P1dB,
    against a 4.1 dB physical spread. A band median over +-band/2 divides that
    noise by roughly sqrt(number of columns) while leaving the ~0.4 GHz
    physical structure intact, so the band must stay well inside that scale.

    ``min_fill`` rejects a target whose window is mostly outside the usable
    columns. Without it a frequency sitting on the edge of the pump-exclusion
    guard is admitted on a handful of columns, keeps the full noise of a single
    column, and is biased by whichever side of the window survived.
    """
    records: list[dict[str, float | bool | int]] = []
    half = 0.5 * band_mhz * 1e-3
    step = float(np.median(np.diff(measured_ghz)))
    expected = max(1, int(round(2.0 * half / step))) if half > 0.0 else 1
    for target in targets_ghz:
        if half <= 0.0:
            selected = np.array(
                [int(np.abs(measured_ghz - target).argmin())]
            )
        else:
            selected = np.flatnonzero(np.abs(measured_ghz - target) <= half)
        selected = selected[usable[selected]]
        record: dict[str, float | bool | int] = {
            "model_ghz": float(target),
            "n_columns": int(selected.size),
            "usable": bool(selected.size >= max(1, int(min_fill * expected))),
        }
        if not record["usable"]:
            selected = selected[:0]
        for name, array in values.items():
            if selected.size == 0:
                record[name] = float("nan")
                record[f"{name}_sem"] = float("nan")
                continue
            sample = array[selected]
            record[name] = float(np.median(sample))
            # Median SEM, the 1.253 factor being the normal-case penalty of the
            # median over the mean; the median is kept because the low-power
            # noise tail is one-sided.
            record[f"{name}_sem"] = (
                float(1.253 * np.std(sample, ddof=1) / np.sqrt(sample.size))
                if sample.size > 1
                else 0.0
            )
        records.append(record)
    return records


def describe(name: str, values: np.ndarray) -> dict[str, float | int | str]:
    quartiles = np.percentile(values, [25, 75])
    return {
        "group": name,
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "iqr": float(quartiles[1] - quartiles[0]),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def panel(axis, groups, title: str, xlabel: str) -> None:
    """Histogram with robust limits; tails are annotated, never silently cut."""
    # A median +- k*IQR window rather than a percentile clip: the measured
    # low-power noise tail is 1.2% of the population, so any percentile cut
    # loose enough to be honest is still wide enough to squash the bulk.
    pooled = np.concatenate([values for _, values, _ in groups])
    q25, median, q75 = np.percentile(pooled, [25, 50, 75])
    span = max(q75 - q25, 1e-9)
    lo = max(float(pooled.min()), median - 4.0 * span)
    hi = min(float(pooled.max()), median + 4.0 * span)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    bins = np.linspace(lo, hi, 46)

    outside = 0
    for label, values, colour in groups:
        outside += int(np.count_nonzero((values < lo) | (values > hi)))
        axis.hist(
            values, bins=bins, density=True, alpha=0.45, color=colour,
            label=f"{label}  (n={values.size})", edgecolor="none",
        )
        axis.axvline(
            float(np.median(values)), color=colour, linestyle="--", linewidth=1.7
        )

    axis.set_xlim(lo, hi)
    top = axis.get_ylim()[1]
    axis.set_ylim(-0.055 * top, top)

    # The model group is n=18; a density histogram of 18 points is easy to
    # over-read, so its individual values are also drawn as a rug.
    for _, values, colour in groups:
        if values.size <= 30:
            axis.plot(
                values, np.full(values.size, -0.030 * top), "|",
                color=colour, markersize=11, markeredgewidth=1.5,
            )

    if outside:
        axis.text(
            0.015, 0.02, f"{outside} point(s) beyond axis",
            transform=axis.transAxes, fontsize=7.5, color="0.35",
        )

    axis.set_title(title, fontsize=11)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("density")
    axis.legend(fontsize=8, framealpha=0.9, loc="upper left")
    axis.grid(alpha=0.25, linestyle=":")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "outputs" / "exp45_2c_p1db_vs_frequency",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "presentation")
    parser.add_argument(
        "--crossing", choices=["first", "persistent"], default="first",
        help="Measured-column threshold-crossing rule; see module docstring.",
    )
    parser.add_argument(
        "--match", choices=["gain", "frequency"], default="gain",
        help="How the third group is drawn from the cube; see module docstring.",
    )
    parser.add_argument(
        "--band-average-mhz", type=float, default=0.0,
        help=(
            "Full width of the median window used by --match frequency. "
            "0 takes the single nearest column and inherits its extraction "
            "noise; 200 keeps the ~0.4 GHz physical structure while cutting "
            "that noise by about sqrt(51)."
        ),
    )
    parser.add_argument(
        "--min-band-fill", type=float, default=0.5,
        help=(
            "Fraction of the band window that must contain usable columns. "
            "Rejects frequencies clipped by the pump-exclusion guard."
        ),
    )
    args = parser.parse_args()

    model_ghz, model_g0, model_p1db, model_psat = model_table(args.run_dir)
    if model_ghz.size == 0:
        raise SystemExit(f"no model curves under {args.run_dir}")

    all_ghz, all_g0, all_p1db, usable = measured_table(args.crossing)
    meas_ghz, meas_g0, meas_p1db = all_ghz[usable], all_g0[usable], all_p1db[usable]
    meas_psat = meas_p1db + meas_g0 - 1.0

    lo = float(model_g0.min()) - GAIN_MATCH_PAD_DB
    hi = float(model_g0.max()) + GAIN_MATCH_PAD_DB
    pairing: list[dict[str, float | bool]] = []
    if args.match == "frequency":
        # Frequency-matched subset: the cube aggregated at each model frequency.
        # The model and the cube then sample the same signal frequencies, so the
        # groups are paired and the comparison is a per-frequency residual.
        pairing = match_columns(
            all_ghz,
            usable,
            {"measured_g0_db": all_g0, "measured_p1db_dbm": all_p1db},
            model_ghz,
            args.band_average_mhz,
            args.min_band_fill,
        )
        band = (
            "nearest column" if args.band_average_mhz <= 0
            else f"$\\pm${args.band_average_mhz / 2:.0f} MHz median"
        )
        match_label = f"Themis at model $f_s$ ({band})"
        keep = np.array([bool(entry["usable"]) for entry in pairing])
        match_g0 = np.array([entry["measured_g0_db"] for entry in pairing])[keep]
        match_p1db = np.array([entry["measured_p1db_dbm"] for entry in pairing])[keep]
    else:
        # Gain-matched subset: measured columns whose small-signal gain lies
        # inside the model's own G0 span. The model reaches only ~5-10 dB here
        # while the cube spans 3-16 dB, and P1dB is strongly gain-dependent, so
        # the full measured population and the model are not sampling the same
        # operating regime.
        matched = (meas_g0 >= lo) & (meas_g0 <= hi)
        match_label = f"Themis, $G_0\\in[{lo:.1f},{hi:.1f}]$ dB"
        keep = matched
        match_g0, match_p1db = meas_g0[matched], meas_p1db[matched]
    match_psat = match_p1db + match_g0 - 1.0

    # Frequency matching makes the two groups paired, so the informative
    # statistic is the per-frequency residual, not the overlap of two spreads.
    paired_residuals: dict[str, dict[str, float]] = {}
    if pairing:
        deltas = {
            "g0_db": model_g0[keep] - match_g0,
            "p1db_db": model_p1db[keep] - match_p1db,
            "psat_db": model_psat[keep] - match_psat,
        }
        for key, delta in deltas.items():
            paired_residuals[key] = {
                "n": int(delta.size),
                "mean": float(np.mean(delta)),
                "std": float(np.std(delta, ddof=1)),
                "median": float(np.median(delta)),
                "max_abs": float(np.max(np.abs(delta))),
            }

    quantities = [
        ("Small-signal gain $G_0$", "$G_0$ (dB)",
         model_g0, meas_g0, match_g0),
        ("Input 1 dB compression $P_{1dB}$", "$P_{1dB}$ input (dBm)",
         model_p1db, meas_p1db, match_p1db),
        (r"Saturation power  $P_{sat}=P_{1dB}+G_0-1$", "$P_{sat}$ output (dBm)",
         model_psat, meas_psat, match_psat),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    records: list[dict[str, float | int | str]] = []
    for axis, (title, xlabel, model_v, meas_v, match_v) in zip(axes, quantities):
        groups = [
            ("model (18 curves)", model_v, MODEL_COLOUR),
            ("Themis, all columns", meas_v, MEAS_COLOUR),
            (match_label, match_v, MATCH_COLOUR),
        ]
        panel(axis, groups, title, xlabel)
        for label, values, _ in groups:
            record = describe(label, values)
            record["quantity"] = title
            records.append(record)

    figure.suptitle(
        "2c saturation power distributions -- model vs Themis   "
        f"(cube {CUBE.name}, pump {MEAS_PUMP_GHZ} GHz, "
        f"crossing rule: {args.crossing}, match: {args.match})",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.crossing == "first" else f"_{args.crossing}"
    if args.match != "gain":
        suffix += "_freqmatched"
        if args.band_average_mhz > 0:
            suffix += f"_band{args.band_average_mhz:.0f}mhz"
    png = args.outdir / f"2c_psat_distributions{suffix}.png"
    figure.savefig(png, dpi=180)
    plt.close(figure)

    # Sensitivity of the measured population to the extraction rule.
    sensitivity = {}
    for rule in ("first", "persistent"):
        _, alt_g0, alt_p1db, alt_usable = measured_table(rule)
        alt_g0, alt_p1db = alt_g0[alt_usable], alt_p1db[alt_usable]
        sensitivity[rule] = {
            "n": int(alt_p1db.size),
            "median_p1db_dbm": float(np.median(alt_p1db)),
            "median_psat_dbm": float(np.median(alt_p1db + alt_g0 - 1.0)),
            "n_below_minus_110_dbm": int(np.count_nonzero(alt_p1db < -110.0)),
        }

    payload = {
        "definition": "P_sat = P1dB_input + G0 - 1 dB (output-referred 1 dB compression)",
        "solver_reference": "scripts/run_compression.py:956 p1db_output_dbm",
        "naive_alternative": "P1dB + G0 omits the -1 dB; rigid shift, shape unchanged",
        "cube": str(CUBE),
        "signal_line_loss_db": SIGNAL_LINE_LOSS_DB,
        "measured_pump_ghz": MEAS_PUMP_GHZ,
        "crossing_rule": args.crossing,
        "match_mode": args.match,
        "gain_match_span_db": [lo, hi],
        "frequency_pairing": pairing,
        "paired_residuals": paired_residuals,
        "extraction_sensitivity": sensitivity,
        "statistics": records,
    }
    (args.outdir / f"2c_psat_distributions{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    header = (
        f"{'quantity':<34} {'group':<34} {'n':>5} {'median':>9} "
        f"{'mean':>9} {'IQR':>7} {'p05':>9} {'p95':>9}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{str(record['quantity'])[:34]:<34} {str(record['group'])[:34]:<34} "
            f"{record['n']:>5} {record['median']:>9.2f} {record['mean']:>9.2f} "
            f"{record['iqr']:>7.2f} {record['p05']:>9.2f} {record['p95']:>9.2f}"
        )
    if pairing:
        print(
            "\nfrequency pairing (model vs measurement at the same f_s):\n"
            f"  {'f_s GHz':>8} {'ncol':>5} {'G0 mdl':>8} {'G0 meas':>8} "
            f"{'P1dB mdl':>9} {'P1dB meas':>10} {'+-sem':>6} "
            f"{'Psat mdl':>9} {'Psat meas':>10}  note"
        )
        for entry, m_g0, m_p1db, m_psat in zip(
            pairing, model_g0, model_p1db, model_psat
        ):
            if entry["usable"]:
                here = entry["measured_p1db_dbm"] + entry["measured_g0_db"] - 1.0
                tail = (
                    f"{entry['measured_g0_db']:>8.2f} {m_p1db:>9.2f} "
                    f"{entry['measured_p1db_dbm']:>10.2f} "
                    f"{entry['measured_p1db_dbm_sem']:>6.2f} "
                    f"{m_psat:>9.2f} {here:>10.2f}  "
                )
            else:
                tail = (
                    f"{'--':>8} {m_p1db:>9.2f} {'--':>10} {'--':>6} "
                    f"{m_psat:>9.2f} {'--':>10}  excluded"
                )
            print(
                f"  {entry['model_ghz']:>8.2f} {entry['n_columns']:>5} "
                f"{m_g0:>8.2f} {tail}"
            )
    if paired_residuals:
        print("\npaired residual, model - measurement at the same signal frequency:")
        for key, stats in paired_residuals.items():
            print(
                f"  {key:<9} n={stats['n']:<3} mean={stats['mean']:+7.2f} dB  "
                f"std={stats['std']:6.2f}  median={stats['median']:+7.2f}  "
                f"max|.|={stats['max_abs']:6.2f}"
            )
    print("\nextraction sensitivity (measured columns):")
    for rule, stats in sensitivity.items():
        print(
            f"  {rule:<11} n={stats['n']:<5} median P1dB={stats['median_p1db_dbm']:8.2f}  "
            f"median Psat={stats['median_psat_dbm']:8.2f}  "
            f"columns below -110 dBm={stats['n_below_minus_110_dbm']}"
        )
    print(f"\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
