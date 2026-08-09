# experiments/exp14_dpjpa_multitone.py
"""Two-tone (multi-pump) harmonic balance pump solve + reflection gain for DPJPA.

DPJPA is driven by two pump tones at wp1, wp2 (modes (1,0) and (0,1)). The scalar
positive-phasor mode policy cannot represent multi-index modes, so this is a
self-contained 2D-lattice HB built on integer mode tuples (k1, k2):

    omega(m) = k1*wp1 + k2*wp2
    psi(t)   = 2 Re sum_{m in modes+} X_m exp(+i omega(m) t)

evaluated on the 2-torus (theta1, theta2) so that the nonlinear current is
projected exactly by a 2D DFT. The circuit is tiny (2 nodes, 1 JJ), so the exact
Jacobian is assembled densely and solved directly.

Writes, compatible with exp14_seven_design_summary.py:
    <outdir>/pump/pump_report.json
    <outdir>/gain/gain_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# =============================================================================
# Design loading
# =============================================================================

def load_design(d: Path):
    C = sp.load_npz(d / "C.npz").toarray()
    G = sp.load_npz(d / "G.npz").toarray()
    K = sp.load_npz(d / "K.npz").toarray()
    Bphi = sp.load_npz(d / "Bphi.npz").toarray()
    arr = np.load(d / "ipm_arrays.npz")
    Ic = arr["Ic"].astype(float)
    phi0 = float(arr["phi0_reduced"][0])
    port_to_index = {
        int(p): int(i)
        for p, i in zip(arr["port_numbers"], arr["port_indices"])
    }
    return C, G, K, Bphi, Ic, phi0, port_to_index


# =============================================================================
# 2-tone mode lattice
# =============================================================================

def positive_half_lattice(order: int) -> list[tuple[int, int]]:
    """Modes (k1,k2) in the box [-order,order]^2, excluding (0,0), positive half.

    Positive half = first non-zero coordinate positive, so each +/- conjugate
    pair is represented once (real reconstruction uses 2 Re).
    """
    modes: list[tuple[int, int]] = []
    for k1 in range(-order, order + 1):
        for k2 in range(-order, order + 1):
            if k1 == 0 and k2 == 0:
                continue
            if k1 > 0 or (k1 == 0 and k2 > 0):
                modes.append((k1, k2))
    return sorted(modes)


def full_box(order: int) -> list[tuple[int, int]]:
    """All modes (k1,k2) in [-order,order]^2 including (0,0)."""
    return [
        (k1, k2)
        for k1 in range(-order, order + 1)
        for k2 in range(-order, order + 1)
    ]


def omega_of(mode: tuple[int, int], wp: tuple[float, float]) -> float:
    return mode[0] * wp[0] + mode[1] * wp[1]


def dynamic_block(K, C, G, omega):
    return K + (-omega * omega) * C + (1j * omega) * G


# =============================================================================
# 2-torus time grid + transforms
# =============================================================================

class Torus:
    def __init__(self, modes: list[tuple[int, int]], nt: int):
        self.modes = modes
        self.M = len(modes)
        th = 2.0 * math.pi * np.arange(nt) / nt
        TH1, TH2 = np.meshgrid(th, th, indexing="ij")
        self.phase1 = TH1.reshape(-1)
        self.phase2 = TH2.reshape(-1)
        self.T = self.phase1.size
        k1 = np.array([m[0] for m in modes], dtype=float)
        k2 = np.array([m[1] for m in modes], dtype=float)
        # E[t, m] = exp(i (k1 th1 + k2 th2))
        self.E = np.exp(
            1j * (self.phase1[:, None] * k1[None, :] + self.phase2[:, None] * k2[None, :])
        )
        self.Einv = self.E.conj().T / self.T  # orthogonal projector on the torus

    def synth_real(self, X: np.ndarray) -> np.ndarray:
        # X: (M, n) -> x(t): (T, n) real
        return 2.0 * np.real(self.E @ X)

    def project(self, y_t: np.ndarray) -> np.ndarray:
        return self.Einv @ y_t  # (M, n) complex


# =============================================================================
# Pump solve (exact dense Newton with current continuation)
# =============================================================================

def solve_pump(C, G, K, Bphi, Ic, phi0, wp, source_node, Ip,
               pump_order, nt, steps, tol, max_newton):
    modes = positive_half_lattice(pump_order)
    torus = Torus(modes, nt)
    n = C.shape[0]
    M = len(modes)
    BphiT = Bphi.T

    omegas = np.array([omega_of(m, wp) for m in modes])
    Dblocks = [dynamic_block(K, C, G, w) for w in omegas]

    # gamma_hat lattice keys we need for the Jacobian: m-q and m+q.
    mode_index = {m: i for i, m in enumerate(modes)}

    def source_vec(scale):
        S = np.zeros((M, n), dtype=np.complex128)
        for tone in ((1, 0), (0, 1)):
            S[mode_index[tone], source_node] += 0.5 * scale * Ip
        return S

    def residual(X, scale):
        x_t = torus.synth_real(X)            # (T, n)
        psi_t = x_t @ Bphi                   # (T, nb) = x_t @ (n,nb)? Bphi is (n,nb)
        i_t = Ic[None, :] * np.sin(psi_t / phi0)   # (T, nb)
        nl_t = i_t @ Bphi.T                  # (T, n)
        N = torus.project(nl_t)              # (M, n)
        S = source_vec(scale)
        R = np.empty_like(X)
        for j in range(M):
            R[j] = Dblocks[j] @ X[j] + N[j] - S[j]
        return R

    def gamma_hat_table(X, needed):
        x_t = torus.synth_real(X)
        psi_t = x_t @ Bphi
        gamma_t = (Ic[None, :] / phi0) * np.cos(psi_t / phi0)   # (T, nb)
        table = {}
        for l in needed:
            ph = np.exp(-1j * (l[0] * torus.phase1 + l[1] * torus.phase2))
            table[l] = np.mean(gamma_t * ph[:, None], axis=0)   # (nb,)
        return table

    # needed gamma lattice keys
    needed = set()
    for a in modes:
        for b in modes:
            needed.add((a[0] - b[0], a[1] - b[1]))
            needed.add((a[0] + b[0], a[1] + b[1]))

    def jacobian_dense(X):
        ght = gamma_hat_table(X, needed)
        khat = {}
        for l, gh in ght.items():
            khat[l] = Bphi @ np.diag(gh) @ Bphi.T   # (n,n) complex
        # assemble real (2*M*n) Jacobian: per (j,q) 2x2 super-block of L V + P conj(V)
        Mn = M * n
        JRR = np.zeros((Mn, Mn)); JRI = np.zeros((Mn, Mn))
        JIR = np.zeros((Mn, Mn)); JII = np.zeros((Mn, Mn))
        for j, mj in enumerate(modes):
            for q, mq in enumerate(modes):
                Lkey = (mj[0] - mq[0], mj[1] - mq[1])
                Pkey = (mj[0] + mq[0], mj[1] + mq[1])
                L = khat.get(Lkey, np.zeros((n, n), dtype=np.complex128)).copy()
                if j == q:
                    L = L + Dblocks[j]
                P = khat.get(Pkey, np.zeros((n, n), dtype=np.complex128))
                Lr, Li = L.real, L.imag
                Pr, Pi = P.real, P.imag
                rs, qs = j * n, q * n
                JRR[rs:rs + n, qs:qs + n] = Lr + Pr
                JRI[rs:rs + n, qs:qs + n] = Pi - Li
                JIR[rs:rs + n, qs:qs + n] = Li + Pi
                JII[rs:rs + n, qs:qs + n] = Lr - Pr
        return np.block([[JRR, JRI], [JIR, JII]])

    def pack(X):
        return np.concatenate([X.real.reshape(-1), X.imag.reshape(-1)])

    def unpack(v):
        Mn = M * n
        return (v[:Mn] + 1j * v[Mn:]).reshape(M, n)

    X = np.zeros((M, n), dtype=np.complex128)
    t0 = time.perf_counter()
    final_rel = math.inf
    for s_i in range(1, steps + 1):
        scale = s_i / steps
        for _ in range(max_newton):
            R = residual(X, scale)
            rhs = -pack(R)
            S = source_vec(scale)
            denom = max(np.linalg.norm(pack(S)), 1e-30)
            rel = np.linalg.norm(rhs) / denom
            if rel < tol:
                break
            J = jacobian_dense(X)
            dX = np.linalg.solve(J, rhs)
            X = X + unpack(dX)
        final_rel = rel
    runtime = time.perf_counter() - t0
    return X, modes, torus, final_rel, runtime


# =============================================================================
# Gain (2D conversion matrix), reflection S11
# =============================================================================

def solve_gain(C, G, K, Bphi, Ic, phi0, wp, X_pump, pump_modes, torus,
               source_node, out_node, source_port, out_port, z0,
               signal_ghz, sideband_order):
    n = C.shape[0]
    ms = full_box(sideband_order)
    Mi = {m: i for i, m in enumerate(ms)}
    Nm = len(ms)

    # gamma_hat over the lattice differences m-q
    x_t = torus.synth_real(X_pump)
    psi_t = x_t @ Bphi
    gamma_t = (Ic[None, :] / phi0) * np.cos(psi_t / phi0)

    def khat(l):
        ph = np.exp(-1j * (l[0] * torus.phase1 + l[1] * torus.phase2))
        gh = np.mean(gamma_t * ph[:, None], axis=0)
        return Bphi @ np.diag(gh) @ Bphi.T

    khat_cache = {}
    needed = {(a[0] - b[0], a[1] - b[1]) for a in ms for b in ms}
    for l in needed:
        khat_cache[l] = khat(l)

    omega_s = 2.0 * math.pi * signal_ghz * 1e9

    A = np.zeros((Nm * n, Nm * n), dtype=np.complex128)
    for a, ma in enumerate(ms):
        om = omega_s + omega_of(ma, wp)
        D = dynamic_block(K, C, G, om)
        for b, mb in enumerate(ms):
            l = (ma[0] - mb[0], ma[1] - mb[1])
            blk = khat_cache[l].copy()
            if a == b:
                blk = blk + D
            A[a * n:(a + 1) * n, b * n:(b + 1) * n] = blk

    rhs = np.zeros(Nm * n, dtype=np.complex128)
    rhs[Mi[(0, 0)] * n + source_node] = 1.0  # unit current at signal mode (0,0)
    y = np.linalg.solve(A, rhs)

    phi_out = y[Mi[(0, 0)] * n + out_node]
    vout_on = 1j * omega_s * phi_out

    # pump-off reference: D(omega_s) + linear JJ stiffness, same unit drive
    gamma_off = Ic / phi0
    Koff = Bphi @ np.diag(gamma_off) @ Bphi.T
    Aoff = dynamic_block(K, C, G, omega_s) + Koff
    boff = np.zeros(n, dtype=np.complex128)
    boff[source_node] = 1.0
    yoff = np.linalg.solve(Aoff, boff)
    vout_off = 1j * omega_s * yoff[out_node]

    def s_of(v):
        s = 2.0 * v / z0
        if source_port == out_port:
            s -= 1.0
        return s

    s_on = s_of(vout_on)
    gain_db = 20.0 * math.log10(max(abs(s_on), 1e-300))
    return gain_db, abs(s_on), complex(vout_on), complex(vout_off)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ipm-dir", default="outputs/jc_doc_python_designs/jc_dpjpa")
    p.add_argument("--outdir", default="outputs/exp14_dpjpa_multitone")
    p.add_argument("--pump1-ghz", type=float, default=4.65001)
    p.add_argument("--pump2-ghz", type=float, default=4.85001)
    p.add_argument("--pump-current-a", type=float, default=1.921e-8)  # 2 x 9.605e-9
    p.add_argument("--pump-port", type=int, default=1)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=1)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--pump-order", type=int, default=4)
    p.add_argument("--nt", type=int, default=16)
    p.add_argument("--sideband-order", type=int, default=3)
    p.add_argument("--continuation-steps", type=int, default=4)
    p.add_argument("--newton-tol", type=float, default=1e-10)
    p.add_argument("--max-newton", type=int, default=20)
    p.add_argument("--signal-start-ghz", type=float, default=4.5)
    p.add_argument("--signal-stop-ghz", type=float, default=5.0)
    p.add_argument("--points", type=int, default=21)
    args = p.parse_args()

    d = Path(args.ipm_dir)
    C, G, K, Bphi, Ic, phi0, ports = load_design(d)
    wp = (2.0 * math.pi * args.pump1_ghz * 1e9, 2.0 * math.pi * args.pump2_ghz * 1e9)
    source_node = ports[args.pump_port]

    print(f"=== DPJPA multi-tone HB (2 pumps {args.pump1_ghz}/{args.pump2_ghz} GHz) ===")
    print(f"nodes={C.shape[0]} jj={Bphi.shape[1]} Ic={Ic} phi0={phi0:.6e}")
    print(f"pump_current_a={args.pump_current_a:.6e} order={args.pump_order} nt={args.nt}^2")

    X, modes, torus, rel, pump_rt = solve_pump(
        C, G, K, Bphi, Ic, phi0, wp, source_node, args.pump_current_a,
        args.pump_order, args.nt, args.continuation_steps, args.newton_tol, args.max_newton,
    )
    rel = float(rel)
    pump_rt = float(pump_rt)
    converged = bool(rel < 1e-6)
    print(f"pump_final_coeff_rel={rel:.3e} converged={converged} runtime_s={pump_rt:.3f}")

    pump_dir = Path(args.outdir) / "pump"
    pump_dir.mkdir(parents=True, exist_ok=True)
    with open(pump_dir / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_status": "VALID_CONVERGED" if converged else "FAIL",
            "metadata": {
                "pump_mode_policy": "multi_tone_lattice",
                "pump_modes": [list(m) for m in modes],
                "pump_basis": "positive_phasor_2tone",
                "real_reconstruction_factor": 2,
                "pump_freqs_ghz": [args.pump1_ghz, args.pump2_ghz],
                "pump_current_a": args.pump_current_a,
                "pump_order": args.pump_order,
                "nt": args.nt,
            },
            "reports": [{"coeff_rel": rel, "converged": converged, "runtime_s": pump_rt}],
        }, f, indent=2)

    if not converged:
        print("PUMP NONCONVERGED - skipping gain")
        return

    src_node = ports[args.source_port]
    out_node = ports[args.out_port]
    freqs = np.linspace(args.signal_start_ghz, args.signal_stop_ghz, args.points)

    gain_dir = Path(args.outdir) / "gain"
    gain_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    print("signal_ghz,gain_db,s_param_abs")
    for fg in freqs:
        gdb, sabs, von, voff = solve_gain(
            C, G, K, Bphi, Ic, phi0, wp, X, modes, torus,
            src_node, out_node, args.source_port, args.out_port, args.z0_ohm,
            float(fg), args.sideband_order,
        )
        rows.append((float(fg), gdb, sabs))
        print(f"{fg:.4f},{gdb:.6f},{sabs:.6f}")
    gain_rt = time.perf_counter() - t0

    with open(gain_dir / "gain_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_ghz", "gain_db", "s_param_abs"])
        for r in rows:
            w.writerow(r)
    with open(gain_dir / "gain_report.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_runtime_s": gain_rt,
                                "sideband_order": args.sideband_order}}, f, indent=2)

    gmax = max(r[1] for r in rows)
    fpk = max(rows, key=lambda r: r[1])[0]
    print(f"gain_db_max={gmax:.6f} peak_ghz={fpk} gain_runtime_s={gain_rt:.3f}")


if __name__ == "__main__":
    main()
