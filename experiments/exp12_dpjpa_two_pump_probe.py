from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root


@dataclass
class IPM:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    Ic: np.ndarray
    phi0: float
    ports: dict[int, int]
    nodes: int


def load_ipm(ipm_dir: Path) -> IPM:
    arrays = np.load(ipm_dir / "ipm_arrays.npz", allow_pickle=True)
    C = sp.load_npz(ipm_dir / "C.npz").tocsr()
    G = sp.load_npz(ipm_dir / "G.npz").tocsr()
    K = sp.load_npz(ipm_dir / "K.npz").tocsr()
    Bphi = sp.load_npz(ipm_dir / "Bphi.npz").tocsr()
    ports = {int(k): int(v) for k, v in arrays["ports"].item().items()}
    return IPM(
        C=C, G=G, K=K, Bphi=Bphi,
        Ic=np.asarray(arrays["Ic"], dtype=float).reshape(-1),
        phi0=float(np.asarray(arrays["phi0_reduced"]).reshape(-1)[0]),
        ports=ports,
        nodes=int(np.asarray(arrays["nodes"]).reshape(-1)[0]),
    )


def synthesize_2pump(A: np.ndarray, modes: list[tuple[int, int]], t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    # x(t1,t2) = 2 Re sum A_mn exp(i(m t1 + n t2))
    nt = t1.size
    x = np.zeros((nt, A.shape[1]), dtype=float)
    for k, (m, n) in enumerate(modes):
        phase = np.exp(1j * (m * t1 + n * t2))
        x += 2.0 * np.real(phase[:, None] * A[k][None, :])
    return x


def project_2pump(y_t: np.ndarray, modes: list[tuple[int, int]], t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    out = np.zeros((len(modes), y_t.shape[1]), dtype=np.complex128)
    for k, (m, n) in enumerate(modes):
        phase = np.exp(-1j * (m * t1 + n * t2))
        out[k] = np.mean(y_t * phase[:, None], axis=0)
    return out


def pack(A: np.ndarray) -> np.ndarray:
    return np.concatenate([A.real.ravel(), A.imag.ravel()])


def unpack(z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    half = z.size // 2
    return z[:half].reshape(shape) + 1j * z[half:].reshape(shape)


def build_modes(order: int) -> list[tuple[int, int]]:
    # Positive pump lattice excluding DC.
    modes = []
    for m in range(0, order + 1):
        for n in range(0, order + 1):
            if m == 0 and n == 0:
                continue
            if m + n <= order:
                modes.append((m, n))
    return modes


def residual(
    z: np.ndarray,
    ipm: IPM,
    modes: list[tuple[int, int]],
    omega1: float,
    omega2: float,
    t1: np.ndarray,
    t2: np.ndarray,
    src_coeff: np.ndarray,
) -> np.ndarray:
    A = unpack(z, (len(modes), ipm.nodes))

    x_t = synthesize_2pump(A, modes, t1, t2)
    psi_t = (ipm.Bphi.T @ x_t.T).T
    ij_t = ipm.Ic[None, :] * np.sin(psi_t / ipm.phi0)
    nl_t = (ipm.Bphi @ ij_t.T).T
    nl_c = project_2pump(nl_t, modes, t1, t2)

    R = np.empty_like(A)
    for k, (m, n) in enumerate(modes):
        Omega = m * omega1 + n * omega2
        D = ipm.K - (Omega * Omega) * ipm.C + 1j * Omega * ipm.G
        R[k] = D @ A[k] + nl_c[k] - src_coeff[k]

    return pack(R)


def solve_two_pump(
    ipm: IPM,
    *,
    port: int,
    f1_ghz: float,
    f2_ghz: float,
    pump_current_a: float,
    order: int,
    nt: int,
) -> tuple[str, np.ndarray, dict]:
    omega1 = 2 * np.pi * f1_ghz * 1e9
    omega2 = 2 * np.pi * f2_ghz * 1e9
    modes = build_modes(order)

    # Quasi-periodic sampling on 2D torus flattened to nt*nt samples.
    grid = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    t1 = T1.ravel()
    t2 = T2.ravel()

    src_coeff = np.zeros((len(modes), ipm.nodes), dtype=np.complex128)
    pidx = ipm.ports[port]
    for k, mode in enumerate(modes):
        if mode in [(1, 0), (0, 1)]:
            # cos tone has positive complex coefficient I/2 in real synthesis convention.
            src_coeff[k, pidx] = 0.5 * pump_current_a

    # Linear seed.
    A0 = np.zeros_like(src_coeff)
    for k, (m, n) in enumerate(modes):
        Omega = m * omega1 + n * omega2
        D = ipm.K - (Omega * Omega) * ipm.C + 1j * Omega * ipm.G
        Kjj0 = ipm.Bphi @ sp.diags(ipm.Ic / ipm.phi0, format="csr") @ ipm.Bphi.T
        A0[k] = spla.spsolve((D + Kjj0).tocsc(), src_coeff[k])

    t0 = time.perf_counter()
    sol = root(
        lambda zz: residual(zz, ipm, modes, omega1, omega2, t1, t2, src_coeff),
        pack(A0),
        method="hybr",
        options={"maxfev": 2000, "xtol": 1e-10},
    )
    runtime = time.perf_counter() - t0

    A = unpack(sol.x, A0.shape)
    r = residual(sol.x, ipm, modes, omega1, omega2, t1, t2, src_coeff)
    rinf = float(np.max(np.abs(r)))
    rel = rinf / max(float(np.max(np.abs(pack(src_coeff)))), 1.0)

    status = "VALID_CONVERGED" if sol.success and rel < 1e-6 else "NONCONVERGED"

    report = {
        "status": status,
        "success": bool(sol.success),
        "message": str(sol.message),
        "runtime_s": runtime,
        "order": order,
        "nt": nt,
        "modes": modes,
        "nodes": ipm.nodes,
        "jj_branches": int(ipm.Ic.size),
        "pump_current_a": pump_current_a,
        "pump_freqs_ghz": [f1_ghz, f2_ghz],
        "residual_inf": rinf,
        "residual_rel": rel,
        "solution_norm": float(np.linalg.norm(sol.x)),
    }
    return status, A, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipm-dir", default="outputs/jc_doc_python_designs/jc_dpjpa")
    ap.add_argument("--outdir", default="outputs/exp12_dpjpa_two_pump")
    ap.add_argument("--port", type=int, default=1)
    ap.add_argument("--f1-ghz", type=float, default=4.65001)
    ap.add_argument("--f2-ghz", type=float, default=4.85001)
    ap.add_argument("--pump-current-a", type=float, default=0.00565e-6 * 1.7)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--nt", type=int, default=32)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ipm = load_ipm(Path(args.ipm_dir))
    status, A, report = solve_two_pump(
        ipm,
        port=args.port,
        f1_ghz=args.f1_ghz,
        f2_ghz=args.f2_ghz,
        pump_current_a=args.pump_current_a,
        order=args.order,
        nt=args.nt,
    )

    np.savez(
        outdir / "two_pump_solution.npz",
        A_real=A.real,
        A_imag=A.imag,
        modes=np.array(report["modes"], dtype=np.int64),
    )
    (outdir / "two_pump_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== EXP12 DPJPA two-pump probe ===")
    for k in ["status", "success", "runtime_s", "order", "nt", "nodes", "jj_branches", "pump_current_a", "pump_freqs_ghz", "residual_inf", "residual_rel", "solution_norm"]:
        print(f"{k}={report[k]}")
    print(f"wrote_solution={outdir / 'two_pump_solution.npz'}")
    print(f"wrote_report={outdir / 'two_pump_report.json'}")


if __name__ == "__main__":
    main()
