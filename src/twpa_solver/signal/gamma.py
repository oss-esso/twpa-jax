from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp


from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.signal.io import PumpSolution

logger = logging.getLogger(__name__)

def synthesize_real_from_positive_harmonics(
    X: np.ndarray,
    omega: float,
    nt: int,
    modes: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    logger.debug("gamma_waveform_synthesis_start X_shape=%s omega=%s nt=%d modes=%r", X.shape, omega, nt, modes)
    """Reconstruct the real pump waveform x(t) = 2 Re sum_k X_k exp(+i k omega t).

    `modes` are the positive integer pump-harmonic indices (one per row of X).
    Defaults to dense [1, 2, ..., H] for legacy solutions.
    """
    H = X.shape[0]
    if modes is None:
        k = np.arange(1, H + 1, dtype=float)
    else:
        k = np.asarray(modes, dtype=float).reshape(-1)
        if k.size != H:
            raise ValueError(f"modes length {k.size} != pump rows {H}")
    t = np.arange(nt, dtype=float) * (2.0 * np.pi / omega) / nt
    E = np.exp(1j * omega * t[:, None] * k[None, :])
    x_t = 2.0 * np.real(E @ X)
    logger.debug("gamma_waveform_synthesis_complete t_shape=%s waveform_shape=%s max_abs=%s", t.shape, x_t.shape, np.max(np.abs(x_t)))
    return t, x_t


def load_dc_branch_flux(dc_solution: str | Path | None, circuit: CircuitMatrices) -> np.ndarray | None:
    if dc_solution is None:
        logger.debug("gamma_dc_flux_load_skipped reason=not_provided")
        return None

    p = Path(dc_solution)
    if p.is_dir():
        p = p / "dc_solution.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing dc solution: {p}")

    sol = np.load(p)
    if "psi_dc" in sol.files:
        psi_dc = np.asarray(sol["psi_dc"], dtype=np.float64).reshape(-1)
    elif "x_dc" in sol.files:
        x_dc = np.asarray(sol["x_dc"], dtype=np.float64).reshape(-1)
        if x_dc.size != circuit.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {circuit.C.shape[0]}")
        psi_dc = np.asarray(circuit.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"dc solution {p} must contain psi_dc or x_dc")

    if psi_dc.size != circuit.Bphi.shape[1]:
        raise ValueError(f"psi_dc length {psi_dc.size} != branch count {circuit.Bphi.shape[1]}")

    logger.debug("gamma_dc_flux_loaded path=%s size=%d", p, psi_dc.size)
    return psi_dc


def compute_gamma_hat(
    circuit: CircuitMatrices,
    pump: PumpSolution,
    max_ell: int,
    gamma_nt: int,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    logger.debug("gamma_hat_compute_start max_ell=%d gamma_nt=%d pump_modes=%r", max_ell, gamma_nt, pump.modes)
    t, x_t = synthesize_real_from_positive_harmonics(
        pump.X,
        pump.omega_p,
        gamma_nt,
        modes=pump.modes,
    )

    psi_t = (circuit.Bphi.T @ x_t.T).T
    if dc_branch_flux is not None:
        psi_t = psi_t + dc_branch_flux[None, :]
    gamma_t = (circuit.Ic[None, :] / circuit.phi0) * np.cos(psi_t / circuit.phi0)
    logger.debug("gamma_hat_time_domain_built psi_shape=%s gamma_shape=%s gamma_abs_range=(%s,%s)", psi_t.shape, gamma_t.shape, np.min(np.abs(gamma_t)), np.max(np.abs(gamma_t)))

    gamma_hat: dict[int, np.ndarray] = {}

    for ell in range(-max_ell, max_ell + 1):
        phase = np.exp(-1j * ell * pump.omega_p * t)
        gamma_hat[ell] = np.mean(gamma_t * phase[:, None], axis=0)

    logger.debug("gamma_hat_compute_complete n_coeffs=%d ell_range=(%d,%d)", len(gamma_hat), -max_ell, max_ell)
    return gamma_hat


def write_gamma_hat_summary(
    outdir: str | Path,
    gamma_hat: dict[int, np.ndarray],
) -> Path:
    """Diagnostic per-ell summary of the branch gamma_hat spectrum.

    Columns: ell, nbranches, l2_abs, l2_abs_over_zero_l2, max_abs, mean_abs,
    mean_real, mean_imag, conj_symmetry_rel_err.
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "gamma_hat_summary.csv"

    zero = gamma_hat.get(0)
    zero_l2 = float(np.linalg.norm(zero)) if zero is not None else 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ell",
                "nbranches",
                "l2_abs",
                "l2_abs_over_zero_l2",
                "max_abs",
                "mean_abs",
                "mean_real",
                "mean_imag",
                "conj_symmetry_rel_err",
            ]
        )
        for ell in sorted(gamma_hat):
            gh = np.asarray(gamma_hat[ell])
            l2 = float(np.linalg.norm(gh))
            absgh = np.abs(gh)

            conj_err = 0.0
            mirror = gamma_hat.get(-ell)
            if mirror is not None and l2 > 0.0:
                conj_err = float(
                    np.linalg.norm(gh - np.conj(mirror)) / l2
                )

            w.writerow(
                [
                    ell,
                    int(gh.size),
                    l2,
                    l2 / zero_l2 if zero_l2 > 0.0 else 0.0,
                    float(np.max(absgh)) if gh.size else 0.0,
                    float(np.mean(absgh)) if gh.size else 0.0,
                    float(np.mean(gh.real)) if gh.size else 0.0,
                    float(np.mean(gh.imag)) if gh.size else 0.0,
                    conj_err,
                ]
            )

    print(f"wrote_gamma_hat_summary={path}")
    return path


def build_khat(
    Bphi: sp.csr_matrix,
    gamma_hat: dict[int, np.ndarray],
    drop_tol: float,
) -> dict[int, sp.csr_matrix]:
    logger.debug("khat_build_start n_gamma=%d drop_tol=%s", len(gamma_hat), drop_tol)
    khat: dict[int, sp.csr_matrix] = {}

    for ell, gh in gamma_hat.items():
        if np.max(np.abs(gh)) < drop_tol:
            khat[ell] = sp.csr_matrix((Bphi.shape[0], Bphi.shape[0]), dtype=np.complex128)
            continue

        Kh = Bphi @ sp.diags(gh, offsets=0, format="csr") @ Bphi.T
        khat[ell] = Kh.astype(np.complex128).tocsr()

    logger.debug("khat_build_complete n_blocks=%d nnz_by_ell=%r", len(khat), {ell: block.nnz for ell, block in khat.items()})
    return khat
