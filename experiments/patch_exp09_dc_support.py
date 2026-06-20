from pathlib import Path
import re

path = Path("experiments/exp09_full_ipm_gain_from_pump.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_dc_support")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------
# 1. Add DC branch-flux loader before compute_gamma_hat.
# ---------------------------------------------------------------------
if "def load_dc_branch_flux(" not in text:
    marker = "def compute_gamma_hat("
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("Could not find def compute_gamma_hat(")

    helper = r'''

def load_dc_branch_flux(dc_solution: str | Path | None, ipm: IPMMatrices) -> np.ndarray | None:
    """Load a static DC operating point for linearized gain around DC+pump.

    Accepted inputs:
      --dc-solution path/to/dc_solution.npz
      --dc-solution path/to/folder_containing_dc_solution_npz

    The file may contain either:
      x_dc   node flux vector
      psi_dc branch flux vector Bphi.T @ x_dc

    Returns:
      psi_dc branch-flux vector, or None if no DC solution was requested.
    """
    if dc_solution is None:
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
        if x_dc.size != ipm.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {ipm.C.shape[0]}")
        psi_dc = np.asarray(ipm.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"dc solution {p} must contain psi_dc or x_dc")

    if psi_dc.size != ipm.Bphi.shape[1]:
        raise ValueError(f"psi_dc length {psi_dc.size} != branch count {ipm.Bphi.shape[1]}")

    return psi_dc

'''
    text = text[:idx] + helper + "\n" + text[idx:]


# ---------------------------------------------------------------------
# 2. Add dc_branch_flux kwarg to compute_gamma_hat signature.
# ---------------------------------------------------------------------
if "dc_branch_flux: np.ndarray | None = None" not in text:
    old = """def compute_gamma_hat(
    ipm: IPMMatrices,
    pump: PumpSolution,
    max_ell: int,
    gamma_nt: int,
) -> dict[int, np.ndarray]:
"""
    new = """def compute_gamma_hat(
    ipm: IPMMatrices,
    pump: PumpSolution,
    max_ell: int,
    gamma_nt: int,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
"""
    if old not in text:
        raise SystemExit("Could not find original compute_gamma_hat signature.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 3. Shift gamma(t) from pump-only psi to dc+pump psi.
# ---------------------------------------------------------------------
old = """    psi_t = (ipm.Bphi.T @ x_t.T).T
    gamma_t = (ipm.Ic[None, :] / ipm.phi0) * np.cos(psi_t / ipm.phi0)
"""
new = """    psi_t = (ipm.Bphi.T @ x_t.T).T
    if dc_branch_flux is not None:
        psi_t = psi_t + dc_branch_flux[None, :]
    gamma_t = (ipm.Ic[None, :] / ipm.phi0) * np.cos(psi_t / ipm.phi0)
"""
if old in text:
    text = text.replace(old, new, 1)
elif "psi_t = psi_t + dc_branch_flux[None, :]" in text:
    pass
else:
    raise SystemExit("Could not find compute_gamma_hat psi/gamma block.")


# ---------------------------------------------------------------------
# 4. Add --dc-solution CLI arg.
# ---------------------------------------------------------------------
if "--dc-solution" not in text:
    old = '    p.add_argument("--pump-dir", default=os.path.join("outputs", "exp08_full_ipm_pump"))\n'
    new = old + '    p.add_argument("--dc-solution", default=None, help="Optional dc_solution.npz or folder containing it. Uses gamma around DC+pump state.")\n'
    if old not in text:
        raise SystemExit("Could not find --pump-dir argparse line.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 5. Load DC branch flux after loading the pump.
# ---------------------------------------------------------------------
if "dc_branch_flux = load_dc_branch_flux(args.dc_solution, ipm)" not in text:
    old = "    pump = load_pump(args.pump_dir, args.fallback_pump_freq_ghz)\n"
    new = old + "    dc_branch_flux = load_dc_branch_flux(args.dc_solution, ipm)\n"
    if old not in text:
        raise SystemExit("Could not find pump load line.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 6. Print DC info in setup.
# ---------------------------------------------------------------------
if 'print(f"dc_solution={args.dc_solution}")' not in text:
    old = '    print(f"pump_freq_ghz={pump.pump_freq_ghz}")\n'
    new = old + """    if dc_branch_flux is not None:
        print(f"dc_solution={args.dc_solution}")
        print(f"dc_branch_flux_max_abs={float(np.max(np.abs(dc_branch_flux))):.12e}")
        print(f"dc_branch_flux_over_phi0_max_abs={float(np.max(np.abs(dc_branch_flux / ipm.phi0))):.12e}")
"""
    if old not in text:
        raise SystemExit("Could not find pump_freq_ghz print line.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 7. Pass DC branch flux into compute_gamma_hat call.
# ---------------------------------------------------------------------
if "dc_branch_flux=dc_branch_flux," not in text:
    old = """        gamma_nt=args.gamma_nt,
    )
"""
    new = """        gamma_nt=args.gamma_nt,
        dc_branch_flux=dc_branch_flux,
    )
"""
    if old not in text:
        raise SystemExit("Could not find compute_gamma_hat call argument block.")
    text = text.replace(old, new, 1)


# ---------------------------------------------------------------------
# 8. Use DC small-signal tangent for unpumped/off baseline.
# ---------------------------------------------------------------------
if "gamma_off = (ipm.Ic / ipm.phi0) * np.cos(dc_branch_flux / ipm.phi0)" not in text:
    old = """    khat_off_0 = (
        ipm.Bphi
        @ sp.diags(ipm.Ic / ipm.phi0, offsets=0, format="csr")
        @ ipm.Bphi.T
    ).astype(np.complex128).tocsr()
"""
    new = """    if dc_branch_flux is None:
        gamma_off = ipm.Ic / ipm.phi0
    else:
        gamma_off = (ipm.Ic / ipm.phi0) * np.cos(dc_branch_flux / ipm.phi0)

    khat_off_0 = (
        ipm.Bphi
        @ sp.diags(gamma_off, offsets=0, format="csr")
        @ ipm.Bphi.T
    ).astype(np.complex128).tocsr()
"""
    if old not in text:
        raise SystemExit("Could not find khat_off_0 baseline block.")
    text = text.replace(old, new, 1)


path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp09 supports --dc-solution")
print(f"backup={backup}")
