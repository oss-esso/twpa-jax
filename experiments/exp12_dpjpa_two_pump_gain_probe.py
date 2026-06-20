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


@dataclass
class TwoPumpState:
    A: np.ndarray
    modes: list[tuple[int, int]]
    f1_ghz: float
    f2_ghz: float
    pump_current_a: float
    order: int


def load_ipm(ipm_dir: Path) -> IPM:
    arrays = np.load(ipm_dir / "ipm_arrays.npz", allow_pickle=True)
    C = sp.load_npz(ipm_dir / "C.npz").tocsr()
    G = sp.load_npz(ipm_dir / "G.npz").tocsr()
    K = sp.load_npz(ipm_dir / "K.npz").tocsr()
    Bphi = sp.load_npz(ipm_dir / "Bphi.npz").tocsr()
    ports = {int(k): int(v) for k, v in arrays["ports"].item().items()}
    return IPM(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        Ic=np.asarray(arrays["Ic"], dtype=float).reshape(-1),
        phi0=float(np.asarray(arrays["phi0_reduced"]).reshape(-1)[0]),
        ports=ports,
        nodes=int(np.asarray(arrays["nodes"]).reshape(-1)[0]),
    )


def load_two_pump_state(pump_dir: Path) -> TwoPumpState:
    sol = np.load(pump_dir / "two_pump_solution.npz")
    report = json.loads((pump_dir / "two_pump_report.json").read_text(encoding="utf-8"))

    A = np.asarray(sol["A_real"]) + 1j * np.asarray(sol["A_imag"])
    modes = [tuple(map(int, m)) for m in report["modes"]]

    return TwoPumpState(
        A=A,
        modes=modes,
        f1_ghz=float(report["pump_freqs_ghz"][0]),
        f2_ghz=float(report["pump_freqs_ghz"][1]),
        pump_current_a=float(report["pump_current_a"]),
        order=int(report["order"]),
    )


def synthesize_2pump(A: np.ndarray, modes: list[tuple[int, int]], t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    x = np.zeros((t1.size, A.shape[1]), dtype=float)
    for k, (m, n) in enumerate(modes):
        phase = np.exp(1j * (m * t1 + n * t2))
        x += 2.0 * np.real(phase[:, None] * A[k][None, :])
    return x


def projection(y_t: np.ndarray, dm: int, dn: int, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    phase = np.exp(-1j * (dm * t1 + dn * t2))
    return np.mean(y_t * phase[:, None], axis=0)


def build_sidebands(order: int) -> list[tuple[int, int]]:
    modes: list[tuple[int, int]] = []
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            if abs(m) + abs(n) <= order:
                modes.append((m, n))
    modes.sort(key=lambda x: (abs(x[0]) + abs(x[1]), x[0], x[1]))
    return modes


def build_gamma_hat(
    ipm: IPM,
    pump: TwoPumpState,
    needed_diffs: set[tuple[int, int]],
    nt: int,
) -> dict[tuple[int, int], np.ndarray]:
    grid = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    t1 = T1.ravel()
    t2 = T2.ravel()

    x_t = synthesize_2pump(pump.A, pump.modes, t1, t2)
    psi_t = (ipm.Bphi.T @ x_t.T).T
    gamma_t = (ipm.Ic[None, :] / ipm.phi0) * np.cos(psi_t / ipm.phi0)

    return {
        diff: projection(gamma_t, diff[0], diff[1], t1, t2)
        for diff in sorted(needed_diffs)
    }


def khat_from_gamma(ipm: IPM, gamma: np.ndarray) -> sp.csr_matrix:
    return (ipm.Bphi @ sp.diags(gamma, format="csr") @ ipm.Bphi.T).astype(np.complex128).tocsr()


def solve_ltv_response(
    ipm: IPM,
    gamma_hat: dict[tuple[int, int], np.ndarray],
    *,
    signal_ghz: float,
    f1_ghz: float,
    f2_ghz: float,
    sidebands: list[tuple[int, int]],
    source_port: int,
    out_port: int,
    coupling: str,
) -> tuple[complex, float, float]:
    n = ipm.nodes
    ns = len(sidebands)
    omega_s = 2.0 * np.pi * signal_ghz * 1e9
    omega1 = 2.0 * np.pi * f1_ghz * 1e9
    omega2 = 2.0 * np.pi * f2_ghz * 1e9

    mode_to_i = {mn: i for i, mn in enumerate(sidebands)}
    src_idx = ipm.ports[source_port]
    out_idx = ipm.ports[out_port]

    rows = []
    cols = []
    data = []
    b = np.zeros(ns * n, dtype=np.complex128)

    # Unit diagnostic current source at central sideband.
    b[mode_to_i[(0, 0)] * n + src_idx] = 1.0

    khat_cache: dict[tuple[int, int], sp.csr_matrix] = {}

    def get_khat(diff: tuple[int, int]) -> sp.csr_matrix:
        if diff not in khat_cache:
            gh = gamma_hat.get(diff)
            if gh is None:
                khat_cache[diff] = sp.csr_matrix((n, n), dtype=np.complex128)
            else:
                khat_cache[diff] = khat_from_gamma(ipm, gh)
        return khat_cache[diff]

    for i, (m, q) in enumerate(sidebands):
        Omega = omega_s + m * omega1 + q * omega2
        D = (ipm.K - Omega * Omega * ipm.C + 1j * Omega * ipm.G).astype(np.complex128).tocsr()

        for j, (mp, qp) in enumerate(sidebands):
            diff = (m - mp, q - qp)

            if coupling == "off":
                if i != j:
                    continue
                block = D + khat_from_gamma(ipm, ipm.Ic / ipm.phi0)

            elif coupling == "pumpdiag":
                if i != j:
                    continue
                block = D + get_khat((0, 0))

            elif coupling == "full":
                block = get_khat(diff)
                if i == j:
                    block = D + block
            else:
                raise ValueError(coupling)

            coo = block.tocoo()
            rows.extend((i * n + coo.row).tolist())
            cols.extend((j * n + coo.col).tolist())
            data.extend(coo.data.tolist())

    A = sp.csr_matrix((data, (rows, cols)), shape=(ns * n, ns * n), dtype=np.complex128)
    y = spla.spsolve(A.tocsc(), b)

    rel = float(np.linalg.norm(A @ y - b) / max(np.linalg.norm(b), 1e-300))
    y0 = y[mode_to_i[(0, 0)] * n + out_idx]
    return y0, rel, float(A.nnz)


def db_ratio(a: complex, b: complex) -> float:
    return 20.0 * np.log10(max(abs(a), 1e-300) / max(abs(b), 1e-300))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipm-dir", default="outputs/jc_doc_python_designs/jc_dpjpa")
    ap.add_argument("--pump-dir", default="outputs/exp12_dpjpa_two_pump_order8")
    ap.add_argument("--outdir", default="outputs/exp12_dpjpa_two_pump_gain")
    ap.add_argument("--source-port", type=int, default=1)
    ap.add_argument("--out-port", type=int, default=1)
    ap.add_argument("--signal-start-ghz", type=float, default=4.5)
    ap.add_argument("--signal-stop-ghz", type=float, default=5.0)
    ap.add_argument("--points", type=int, default=41)
    ap.add_argument("--sideband-order", type=int, default=2)
    ap.add_argument("--gamma-nt", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ipm = load_ipm(Path(args.ipm_dir))
    pump = load_two_pump_state(Path(args.pump_dir))
    sidebands = build_sidebands(args.sideband_order)

    needed_diffs = {
        (m - mp, n - np_)
        for (m, n) in sidebands
        for (mp, np_) in sidebands
    }

    t0 = time.perf_counter()
    gamma_hat = build_gamma_hat(ipm, pump, needed_diffs, args.gamma_nt)
    gamma_runtime = time.perf_counter() - t0

    freqs = np.linspace(args.signal_start_ghz, args.signal_stop_ghz, args.points)

    rows = []
    for f in freqs:
        full, rel_full, nnz_full = solve_ltv_response(
            ipm, gamma_hat,
            signal_ghz=float(f),
            f1_ghz=pump.f1_ghz,
            f2_ghz=pump.f2_ghz,
            sidebands=sidebands,
            source_port=args.source_port,
            out_port=args.out_port,
            coupling="full",
        )
        off, rel_off, _ = solve_ltv_response(
            ipm, gamma_hat,
            signal_ghz=float(f),
            f1_ghz=pump.f1_ghz,
            f2_ghz=pump.f2_ghz,
            sidebands=sidebands,
            source_port=args.source_port,
            out_port=args.out_port,
            coupling="off",
        )
        diag, rel_diag, _ = solve_ltv_response(
            ipm, gamma_hat,
            signal_ghz=float(f),
            f1_ghz=pump.f1_ghz,
            f2_ghz=pump.f2_ghz,
            sidebands=sidebands,
            source_port=args.source_port,
            out_port=args.out_port,
            coupling="pumpdiag",
        )

        rows.append({
            "signal_ghz": float(f),
            "status": "VALID_SOLVED",
            "gain_vs_off_db": db_ratio(full, off),
            "gain_vs_pumpdiag_db": db_ratio(full, diag),
            "linear_rel_residual": max(rel_full, rel_off, rel_diag),
            "response_abs": abs(full),
            "matrix_nnz": int(nnz_full),
        })

    csv_path = outdir / "two_pump_gain_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    max_gain_off = max(r["gain_vs_off_db"] for r in rows)
    max_gain_diag = max(r["gain_vs_pumpdiag_db"] for r in rows)

    report = {
        "status": "VALID_SOLVED",
        "ipm_dir": args.ipm_dir,
        "pump_dir": args.pump_dir,
        "nodes": ipm.nodes,
        "jj_branches": int(ipm.Ic.size),
        "ports": ipm.ports,
        "pump_freqs_ghz": [pump.f1_ghz, pump.f2_ghz],
        "pump_current_a": pump.pump_current_a,
        "pump_order": pump.order,
        "sideband_order": args.sideband_order,
        "sideband_count": len(sidebands),
        "gamma_nt": args.gamma_nt,
        "gamma_hat_runtime_s": gamma_runtime,
        "points": args.points,
        "max_gain_vs_off_db": max_gain_off,
        "max_gain_vs_pumpdiag_db": max_gain_diag,
        "max_linear_rel_residual": max(r["linear_rel_residual"] for r in rows),
    }
    (outdir / "two_pump_gain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== EXP12 DPJPA two-pump gain probe ===")
    for k in [
        "status",
        "nodes",
        "jj_branches",
        "pump_freqs_ghz",
        "pump_current_a",
        "pump_order",
        "sideband_order",
        "sideband_count",
        "gamma_nt",
        "gamma_hat_runtime_s",
        "points",
        "max_gain_vs_off_db",
        "max_gain_vs_pumpdiag_db",
        "max_linear_rel_residual",
    ]:
        print(f"{k}={report[k]}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={outdir / 'two_pump_gain_report.json'}")


if __name__ == "__main__":
    main()
