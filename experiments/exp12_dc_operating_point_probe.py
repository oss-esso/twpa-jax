from __future__ import annotations

import argparse
import json
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
    Lj: np.ndarray
    phi0: float
    ports: dict[int, int]
    nodes: int


def load_ipm(ipm_dir: Path) -> IPM:
    arrays = np.load(ipm_dir / "ipm_arrays.npz", allow_pickle=True)

    C = sp.load_npz(ipm_dir / "C.npz").tocsr()
    G = sp.load_npz(ipm_dir / "G.npz").tocsr()
    K = sp.load_npz(ipm_dir / "K.npz").tocsr()
    Bphi = sp.load_npz(ipm_dir / "Bphi.npz").tocsr()

    nodes = int(np.asarray(arrays["nodes"]).reshape(-1)[0])
    Ic = np.asarray(arrays["Ic"], dtype=np.float64).reshape(-1)
    Lj = np.asarray(arrays["Lj"], dtype=np.float64).reshape(-1)
    phi0 = float(np.asarray(arrays["phi0_reduced"], dtype=np.float64).reshape(-1)[0])

    raw_ports = arrays["ports"].item()
    ports = {int(k): int(v) for k, v in raw_ports.items()}

    if C.shape != (nodes, nodes):
        raise ValueError(f"C shape {C.shape} inconsistent with nodes={nodes}")
    if G.shape != C.shape or K.shape != C.shape:
        raise ValueError("G and K must match C")
    if Bphi.shape[0] != nodes:
        raise ValueError("Bphi row count must match node count")
    if Bphi.shape[1] != Ic.size:
        raise ValueError("Bphi branch count must match Ic length")

    return IPM(C=C, G=G, K=K, Bphi=Bphi, Ic=Ic, Lj=Lj, phi0=phi0, ports=ports, nodes=nodes)


def dc_residual(ipm: IPM, x: np.ndarray, src: np.ndarray) -> np.ndarray:
    psi = ipm.Bphi.T @ x
    i_j = ipm.Ic * np.sin(psi / ipm.phi0)
    return np.asarray(ipm.K @ x + ipm.Bphi @ i_j - src, dtype=np.float64)


def dc_jacobian(ipm: IPM, x: np.ndarray) -> sp.csc_matrix:
    psi = ipm.Bphi.T @ x
    gamma = (ipm.Ic / ipm.phi0) * np.cos(psi / ipm.phi0)
    J = ipm.K + ipm.Bphi @ sp.diags(gamma, offsets=0, format="csr") @ ipm.Bphi.T
    return J.tocsc()


def damped_newton_dc(
    ipm: IPM,
    src: np.ndarray,
    *,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
) -> tuple[str, np.ndarray, list[dict]]:
    x = np.zeros(ipm.nodes, dtype=np.float64)
    src_norm = max(float(np.linalg.norm(src)), 1.0)
    hist: list[dict] = []

    for it in range(max_iter):
        r = dc_residual(ipm, x, src)
        rn = float(np.linalg.norm(r))
        rinf = float(np.max(np.abs(r))) if r.size else 0.0
        rel = rn / src_norm

        row = {
            "iter": it,
            "res_l2": rn,
            "res_inf": rinf,
            "res_rel": rel,
        }
        hist.append(row)

        if rn <= tol_abs or rel <= tol_rel:
            return "VALID_CONVERGED", x, hist

        J = dc_jacobian(ipm, x)

        try:
            dx = spla.spsolve(J, -r)
        except Exception as e:
            row["linear_error"] = repr(e)
            return "LINEAR_SOLVE_FAILED", x, hist

        if not np.all(np.isfinite(dx)):
            return "NONFINITE_NEWTON_STEP", x, hist

        base = rn
        alpha = 1.0
        accepted = False

        for _ in range(32):
            xt = x + alpha * dx
            rt = dc_residual(ipm, xt, src)
            rtn = float(np.linalg.norm(rt))

            if np.isfinite(rtn) and rtn < base:
                x = xt
                row["alpha"] = alpha
                row["trial_res_l2"] = rtn
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            return "LINE_SEARCH_FAILED", x, hist

    return "MAX_ITER_REACHED", x, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipm-dir", required=True)
    ap.add_argument("--dc-port", type=int, required=True)
    ap.add_argument("--dc-current-a", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--tol-rel", type=float, default=1e-10)
    ap.add_argument("--tol-abs", type=float, default=1e-12)
    args = ap.parse_args()

    ipm_dir = Path(args.ipm_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ipm = load_ipm(ipm_dir)

    if args.dc_port not in ipm.ports:
        raise SystemExit(f"dc port {args.dc_port} not in ports={ipm.ports}")

    src = np.zeros(ipm.nodes, dtype=np.float64)
    src[ipm.ports[args.dc_port]] = args.dc_current_a

    status, x_dc, hist = damped_newton_dc(
        ipm,
        src,
        max_iter=args.max_iter,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
    )

    psi_dc = np.asarray(ipm.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)
    ij_dc = ipm.Ic * np.sin(psi_dc / ipm.phi0)
    gamma_dc = (ipm.Ic / ipm.phi0) * np.cos(psi_dc / ipm.phi0)
    final_r = dc_residual(ipm, x_dc, src)

    report = {
        "status": status,
        "ipm_dir": str(ipm_dir),
        "nodes": ipm.nodes,
        "jj_branches": int(ipm.Ic.size),
        "ports": ipm.ports,
        "dc_port": args.dc_port,
        "dc_current_a": args.dc_current_a,
        "iterations": len(hist),
        "final_res_l2": float(np.linalg.norm(final_r)),
        "final_res_inf": float(np.max(np.abs(final_r))) if final_r.size else 0.0,
        "x_dc_max_abs": float(np.max(np.abs(x_dc))) if x_dc.size else 0.0,
        "psi_dc_max_abs": float(np.max(np.abs(psi_dc))) if psi_dc.size else 0.0,
        "psi_dc_over_phi0_max_abs": float(np.max(np.abs(psi_dc / ipm.phi0))) if psi_dc.size else 0.0,
        "ij_dc_max_abs": float(np.max(np.abs(ij_dc))) if ij_dc.size else 0.0,
        "gamma_dc_min": float(np.min(gamma_dc)) if gamma_dc.size else None,
        "gamma_dc_max": float(np.max(gamma_dc)) if gamma_dc.size else None,
        "history": hist,
    }

    np.savez(
        outdir / "dc_solution.npz",
        x_dc=x_dc,
        psi_dc=psi_dc,
        ij_dc=ij_dc,
        gamma_dc=gamma_dc,
        dc_port=np.array([args.dc_port], dtype=np.int64),
        dc_current_a=np.array([args.dc_current_a], dtype=np.float64),
    )

    (outdir / "dc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== EXP12 DC operating point ===")
    for key in [
        "status",
        "nodes",
        "jj_branches",
        "dc_port",
        "dc_current_a",
        "iterations",
        "final_res_l2",
        "final_res_inf",
        "x_dc_max_abs",
        "psi_dc_max_abs",
        "psi_dc_over_phi0_max_abs",
        "ij_dc_max_abs",
        "gamma_dc_min",
        "gamma_dc_max",
    ]:
        print(f"{key}={report[key]}")

    print(f"wrote_solution={outdir / 'dc_solution.npz'}")
    print(f"wrote_report={outdir / 'dc_report.json'}")


if __name__ == "__main__":
    main()
