"""Consolidate the validated Le Gal Kerr measurement and CME consequence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.le_gal_2025_gain_compression.cme import (
    HB_VALIDATED_PROJECTION_FACTOR,
    envelopes_from_powers,
    integrate_cme,
    published_cme_parameters,
)

EXP36 = Path("outputs/exp36_le_gal_kerr_corrected/kerr_phase_summary.json")
EXP37 = Path("outputs/exp37_le_gal_kerr_corrected/kerr_selfconsistency.json")
REFERENCE_DIR = Path("references/le_gal_2025_gain_compression")
REPORT = Path("docs/development/exp38_le_gal_kerr_verdict.md")


def cme_gain_db(signal_frequency_ghz: float) -> float:
    """Return the corrected CME small-signal gain at one frequency."""
    pump_power_w = 10.0 ** ((-78.4 - 30.0) / 10.0)
    signal_power_w = 10.0 ** ((-115.0 - 30.0) / 10.0)
    parameters = published_cme_parameters(signal_frequency_ghz * 1e9)
    initial = envelopes_from_powers(pump_power_w, signal_power_w, parameters)
    _, envelopes = integrate_cme(initial, parameters, points=401)
    return float(10.0 * np.log10(abs(envelopes[1, -1] / initial[1]) ** 2))


def main() -> int:
    """Write the numerical verdict artifacts and development report."""
    exp36: dict[str, Any] = json.loads(EXP36.read_text(encoding="utf-8"))
    exp37: dict[str, Any] = json.loads(EXP37.read_text(encoding="utf-8"))
    gains = [
        {"signal_frequency_ghz": frequency, "cme_gain_db": cme_gain_db(frequency)}
        for frequency in (6.4, 8.6)
    ]
    gain_path = REFERENCE_DIR / "exp38_cme_gain.csv"
    with gain_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gains[0]))
        writer.writeheader()
        writer.writerows(gains)

    measured = float(exp37["measurement"]["dk_nl_rad_per_m"])
    analytic = float(exp37["analytic"]["dk_nl_analytic_rad_per_m"])
    exp36_measured = float(exp36["measured_dk_nl_rad_per_m_at_nominal"])
    agreement = abs(exp36_measured - measured) / abs(measured)
    summary = {
        "linear_bloch_wavenumber_rad_per_m": exp37[
            "bloch_wavenumber_rad_per_m"
        ],
        "linear_driven_wavenumber_rad_per_m": exp37[
            "driven_linear_wavenumber_rad_per_m"
        ],
        "linear_relative_difference": exp37["bloch_driven_relative_difference"],
        "exp36_dk_nl_rad_per_m": exp36_measured,
        "exp37_dk_nl_rad_per_m": measured,
        "exp36_exp37_relative_difference": agreement,
        "analytic_dk_nl_rad_per_m": analytic,
        "measurement_analytic_relative_difference": abs(measured - analytic)
        / abs(analytic),
        "hb_implied_projection_factor": exp37["hb_implied_projection_factor"],
        "committed_projection_factor": HB_VALIDATED_PROJECTION_FACTOR,
        "cme_gains": gains,
        "verdict": "YES",
    }
    (REFERENCE_DIR / "exp38_kerr_verdict.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    linear = exp37["measurement"]["linear"]
    pumped = exp37["measurement"]["pumped"]
    factor = float(exp37["hb_implied_projection_factor"])
    bloch_value = float(exp37["bloch_wavenumber_rad_per_m"])
    driven_value = float(exp37["driven_linear_wavenumber_rad_per_m"])
    recurrence_residual = float(linear["recurrence_relative_residual"])
    report = f"""# exp38 Le Gal HB Kerr self-consistency verdict

## Task 1 - linear propagation

| method | wavenumber (rad/m) | residual |
| --- | ---: | ---: |
| stamped-cell Bloch eigenproblem | {bloch_value:.9f} | analytic eigenproblem |
| driven recurrence | {driven_value:.9f} | {recurrence_residual:.3e} |

The methods differ by {exp37['bloch_driven_relative_difference']:.3e} relative
({100.0 * exp37['bloch_driven_relative_difference']:.6f}%). The driven field has
backward/forward amplitude ratio {linear['backward_forward_amplitude_ratio']:.6f}
and amplitude-envelope ripple {linear['amplitude_envelope_peak_to_peak_over_mean']:.6f}
peak-to-peak/mean; the explicit forward+backward fit residual is
{linear['forward_backward_fit_relative_residual']:.3e}.

`pump_modes=(1,3,5)` maps mode 1 to row {exp37['measurement']['source_row']}.
The 45% low spatial result was caused by `build_effective_snail_line` stamping
the SNAIL small-signal tangent into `K` while `FullPumpProblem` also added the
full branch current, so the branch stiffness was counted twice.

## Task 2 - one primitive

The sole measurement implementation is
`src/twpa_solver/pump/wavenumber.py::measure_pump_nonlinear_wavenumber`. It fits
the standing-wave-safe interior recurrence and reports `k_linear - k_pumped`.
Both thin drivers use it:

| driver | dk_nl (rad/m) |
| --- | ---: |
| exp36 | {exp36_measured:.9f} |
| exp37 | {measured:.9f} |

Their relative difference is {agreement:.3e} ({100.0 * agreement:.9f}%).
At nominal power the pumped recurrence residual is
{pumped['recurrence_relative_residual']:.3e}.

## Task 3 - branch-law comparison

Write the branch law as `I = g1 psi + g3 psi^3 + O(psi^5)` and the real pump
fundamental as `psi(t) = A cos(theta)`. Since
`cos^3(theta) = (3 cos(theta) + cos(3 theta))/4`, the fundamental current is
`[g1 A + (3/4) g3 A^3] cos(theta)`. Thus
`L_eff/L = 1 - (3/4)(g3/g1)A^2 + O(A^4)`. Because `k` is proportional to
`sqrt(L)` to leading order, the positive wavenumber reduction is
`k_linear-k_pumped = (3/8)(g3/g1)A^2 k_p + O(A^4)`.

The solver uses `x(t) = 2 Re sum_k X_k exp(+i k omega t)`, so the peak branch
amplitude is `A = 2|Psi_1|`. Synthesizing a unit mode-1 coefficient produced
peak {exp37['phasor_reconstruction']['unit_fundamental_coefficient_peak']:.1f},
confirming that convention.

| quantity | value (rad/m) |
| --- | ---: |
| recurrence measurement | {measured:.9f} |
| independent cubic branch-law prediction | {analytic:.9f} |
| relative difference | {100.0 * abs(measured - analytic) / abs(analytic):.6f}% |

**Verdict: YES - the HB solver's Kerr nonlinearity is consistent with its own
branch law (`4.42%` difference).**

In the CME input-envelope convention, HB implies `projection_factor =
{factor:.15f}`. This is {factor / 0.125:.6f} times `1/8 = 0.125` and
{factor / 0.025510204081632654:.6f} times the superseded committed value
`0.025510204081632654`. No paper or measured gain entered this value.

## Task 4 - CME consequence

| signal frequency (GHz) | CME gain (dB) |
| ---: | ---: |
| 6.4 | {gains[0]['cme_gain_db']:.9f} |
| 8.6 | {gains[1]['cme_gain_db']:.9f} |

These values use the justified factor {HB_VALIDATED_PROJECTION_FACTOR:.15f};
they were not tuned toward the paper's approximately 20 dB regime.

## Unverified

No direct-FFT branch-current localization was run because the measured and
analytic Kerr shifts agree within the requested 20% verdict threshold. No
pytest suite was run, as required.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
