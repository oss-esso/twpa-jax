# experiments/exp19_calcqe_validation.py
"""Validate the ported calc_qe/calc_qe_ideal (twpa_solver.signal.quantum_efficiency)
against real solved designs.

JosephsonCircuits.jl's `calcqe`/`calcqeideal` take a scattering matrix in the
field ladder operator basis: rows are output (port, sideband) modes, columns
are input (port, sideband) modes. calc_qe normalizes each output row by the
*total* power arriving there from every input mode, so it needs the FULL row
-- not just the signal/idler pair -- or the row-sum undercounts and qe comes
out above calc_qe_ideal's bound (unphysical; a first pass at this script with
only a 2x2 [signal,idler] truncation showed exactly that for all three
designs below).

So `build_full_signal_row` fixes the output at the physical signal sideband
and excites *every* solved sideband as the input (reusing one
`build_signal_schur_partition` via `solve_gain_one_schur`'s `schur_part=` so
each extra input only re-solves the small retained system, not the whole
circuit), matching the signal_m=0/idler_m=-2 sideband convention already used
by scripts/run_gain_map.py and the exp14 parity runs.

This codebase's solve_gain_one_schur returns classical voltage-ratio
(power-wave) S-parameters, not the frequency-normalized photon ladder
operators calc_qe expects. Since every sideband sits at a different
frequency, the row is reweighted Manley-Rowe style relative to the signal
frequency: S_ladder[n] = S_classical[n] * sqrt(freq[n]/freq_signal), before
calc_qe/calc_qe_ideal see it.

Reports, per design: qe_signal = calc_qe(S_row)[signal], qe_ideal_signal =
calc_qe_ideal(S_ss) (depends only on |S_ss|, so unaffected by the row width),
and efficiency = qe_signal / qe_ideal_signal -- how close the design's signal
output sits to its own quantum limit, bounded in [0, 1] once the full row is
used.

Designs (existing solved outputs, no new Newton solves):
    ipm_2c_fixed : designs/ipm_2c_fixed +
        outputs/gain_map_ipm_2c_fixed_one_column_matrices/map/warm/points/
        point_0004_p_m28p9388dbm_fp_7p9ghz (PASS, 18.54 dB @ 7.8 GHz)
    jc_jtwpa     : outputs/jc_doc_python_designs/jc_jtwpa +
        outputs/exp14_jtwpa_odd10_scale2/pump (near-peak 6.6 GHz)
    jc_fqjtwpa   : outputs/jc_doc_python_designs/jc_fqjtwpa +
        outputs/exp14_fqjtwpa_odd10_scale2/pump (near-peak 7.4 GHz, "floquet" TWPA)

Writes outputs/exp19_calcqe_validation/summary.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import port_s_from_unit_current_response
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat
from twpa_solver.signal.floquet import (
    build_signal_schur_partition,
    sideband_list,
    solve_gain_one_schur,
)
from twpa_solver.signal.io import load_pump
from twpa_solver.signal.quantum_efficiency import calc_qe, calc_qe_ideal

OUTDIR = Path("outputs/exp19_calcqe_validation")


@dataclass
class DesignCase:
    name: str
    circuit_dir: str
    pump_dir: str
    pump_freq_ghz: float
    signal_ghz: float
    sidebands: int
    gamma_nt: int = 96
    signal_m: int = 0
    idler_m: int = -2
    source_port: int = 1
    out_port: int = 2
    z0_ohm: float = 50.0


CASES: list[DesignCase] = [
    DesignCase(
        name="ipm_2c_fixed",
        circuit_dir="designs/ipm_2c_fixed",
        pump_dir=(
            "outputs/gain_map_ipm_2c_fixed_one_column_matrices/map/warm/points/"
            "point_0004_p_m28p9388dbm_fp_7p9ghz/pump"
        ),
        pump_freq_ghz=7.9,
        signal_ghz=7.8,
        sidebands=6,
    ),
    DesignCase(
        name="jc_jtwpa",
        circuit_dir="outputs/jc_doc_python_designs/jc_jtwpa",
        pump_dir="outputs/exp14_jtwpa_odd10_scale2/pump",
        pump_freq_ghz=7.12,
        signal_ghz=6.6,
        sidebands=10,
    ),
    DesignCase(
        name="jc_fqjtwpa",
        circuit_dir="outputs/jc_doc_python_designs/jc_fqjtwpa",
        pump_dir="outputs/exp14_fqjtwpa_odd10_scale2/pump",
        pump_freq_ghz=7.9,
        signal_ghz=7.4,
        sidebands=10,
    ),
]


def build_full_signal_row(case: DesignCase) -> tuple[list[int], np.ndarray]:
    """Build the full signal-output row of S, in the photon ladder-operator basis.

    calc_qe's row-sum denominator is over *every* input mode that can feed a
    given output mode. A 2-mode [signal, idler] truncation undercounts that
    denominator whenever the design routes non-negligible power to other
    sidebands, which inflates qe past calc_qe_ideal's bound (an unphysical
    "qe > qe_ideal" -- exactly what a first pass at this script showed for
    all three designs here). So this fixes the *output* at the physical
    signal sideband and excites *every* sideband `n` in the solved set as the
    input, reusing one `build_signal_schur_partition` (`schur_part=`) so the
    N solves only re-factor the small retained system, not the full circuit.

    S_classical[n] = response at signal output from excitation at sideband n.
    Converted to the ladder basis via Manley-Rowe reweighting relative to the
    signal frequency: S_ladder[n] = S_classical[n] * sqrt(freq[n]/freq_signal).
    """
    circuit = load_circuit(case.circuit_dir)
    pump = load_pump(case.pump_dir, fallback_pump_freq_ghz=case.pump_freq_ghz)

    ms = sideband_list(case.sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    gamma_hat = compute_gamma_hat(
        circuit=circuit, pump=pump, max_ell=max_ell, gamma_nt=case.gamma_nt,
    )
    khat = build_khat(Bphi=circuit.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
    gamma_off = circuit.Ic / circuit.phi0
    khat_off_0 = (
        circuit.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ circuit.Bphi.T
    ).astype(np.complex128).tocsr()

    source_index = circuit.port_to_index[case.source_port]
    out_index = circuit.port_to_index[case.out_port]

    schur_part = build_signal_schur_partition(
        circuit, pump.omega_p, case.signal_ghz, case.sidebands,
        source_index, out_index, loss_model="current_complex_c",
    )

    def s_entry(v_out: complex) -> complex:
        return port_s_from_unit_current_response(
            v_out, source_port=case.source_port, out_port=case.out_port,
            z0_ohm=case.z0_ohm,
        )

    other_m = next(m for m in ms if m != case.signal_m)
    s_classical = np.zeros(len(ms), dtype=np.complex128)
    for i, n in enumerate(ms):
        if n == case.signal_m:
            r = solve_gain_one_schur(
                circuit=circuit, khat=khat, khat_off_0=khat_off_0,
                omega_p=pump.omega_p, signal_ghz=case.signal_ghz,
                sidebands=case.sidebands, signal_m=n, idler_m=other_m,
                source_index=source_index, out_index=out_index,
                source_current_a=1.0, source_port=case.source_port,
                out_port=case.out_port, z0_ohm=case.z0_ohm,
                include_baselines=False, schur_part=schur_part,
            )
            s_classical[i] = s_entry(r.vout_on)
        else:
            r = solve_gain_one_schur(
                circuit=circuit, khat=khat, khat_off_0=khat_off_0,
                omega_p=pump.omega_p, signal_ghz=case.signal_ghz,
                sidebands=case.sidebands, signal_m=n, idler_m=case.signal_m,
                source_index=source_index, out_index=out_index,
                source_current_a=1.0, source_port=case.source_port,
                out_port=case.out_port, z0_ohm=case.z0_ohm,
                include_baselines=False, schur_part=schur_part,
            )
            s_classical[i] = s_entry(r.vout_idler)

    freq_signal = case.signal_ghz
    freqs_in = np.array(
        [abs(case.signal_ghz + n * case.pump_freq_ghz) for n in ms]
    )
    s_ladder = s_classical * np.sqrt(freqs_in / freq_signal)
    return ms, s_ladder


def validate_case(case: DesignCase) -> dict[str, object]:
    ms, s_row = build_full_signal_row(case)
    sig_idx = ms.index(case.signal_m)

    qe_row = calc_qe(s_row.reshape(1, -1))[0]
    qe_signal = float(qe_row[sig_idx])
    qe_ideal_signal = float(
        calc_qe_ideal(np.array([[s_row[sig_idx]]]))[0, 0]
    )
    efficiency = qe_signal / qe_ideal_signal if qe_ideal_signal > 0 else float("nan")

    return {
        "case": case.name,
        "signal_ghz": case.signal_ghz,
        "n_sidebands_summed": len(ms),
        "s_ss_abs": float(abs(s_row[sig_idx])),
        "gain_db_signal": float(20.0 * np.log10(max(abs(s_row[sig_idx]), 1e-300))),
        "qe_signal": qe_signal,
        "qe_ideal_signal": qe_ideal_signal,
        "efficiency_qe_over_ideal": efficiency,
        "qe_signal_leq_ideal": bool(qe_signal <= qe_ideal_signal + 1e-9),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        print(f"solving case={case.name} signal_ghz={case.signal_ghz} ...")
        row = validate_case(case)
        rows.append(row)
        print(
            f"  gain_db(signal)={row['gain_db_signal']:.3f} "
            f"qe_signal={row['qe_signal']:.4f} "
            f"qe_ideal_signal={row['qe_ideal_signal']:.4f} "
            f"efficiency={row['efficiency_qe_over_ideal']:.4f} "
            f"leq_ideal={row['qe_signal_leq_ideal']} "
            f"(n_sidebands_summed={row['n_sidebands_summed']})"
        )

    (OUTDIR / "summary.json").write_text(
        json.dumps({"cases": [asdict(c) for c in CASES], "results": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUTDIR / 'summary.json'}")


if __name__ == "__main__":
    main()
