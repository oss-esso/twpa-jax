from pathlib import Path

path = Path("experiments/exp08_full_ipm_pump_solve.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_dc_support")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------
# Add DC loader helper before HarmonicPumpProblem.
# ---------------------------------------------------------------------
if "def load_dc_solution(" not in text:
    marker = "@dataclass\nclass HarmonicPumpProblem"
    if marker not in text:
        raise SystemExit("Could not find HarmonicPumpProblem marker.")

    helper = r'''

def load_dc_solution(dc_solution: str | Path | None, ipm: LoadedIPM) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a static DC operating point.

    Returns:
        x_dc: node-flux vector, or None
        psi_dc: branch-flux vector Bphi.T @ x_dc, or None

    The pump unknown X remains the AC periodic part. The nonlinear current used
    by the pump residual becomes:

        Ic sin((psi_dc + psi_ac) / phi0) - Ic sin(psi_dc / phi0)
    """
    if dc_solution is None:
        return None, None

    p = Path(dc_solution)
    if p.is_dir():
        p = p / "dc_solution.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing dc solution: {p}")

    sol = np.load(p)
    x_dc = None
    psi_dc = None

    if "x_dc" in sol.files:
        x_dc = np.asarray(sol["x_dc"], dtype=np.float64).reshape(-1)
        if x_dc.size != ipm.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {ipm.C.shape[0]}")
        psi_dc = np.asarray(ipm.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)

    if "psi_dc" in sol.files:
        psi_dc_file = np.asarray(sol["psi_dc"], dtype=np.float64).reshape(-1)
        if psi_dc_file.size != ipm.Bphi.shape[1]:
            raise ValueError(f"psi_dc length {psi_dc_file.size} != branch count {ipm.Bphi.shape[1]}")
        psi_dc = psi_dc_file

    if psi_dc is None:
        raise ValueError(f"dc solution {p} must contain x_dc or psi_dc")

    return x_dc, psi_dc

'''
    text = text.replace(marker, helper + "\n" + marker, 1)

# ---------------------------------------------------------------------
# Add field to HarmonicPumpProblem.
# ---------------------------------------------------------------------
if "dc_branch_flux: np.ndarray | None = None" not in text:
    if "    pump_current_a: float\n" not in text:
        raise SystemExit("Could not find pump_current_a field.")
    text = text.replace(
        "    pump_current_a: float\n",
        "    pump_current_a: float\n    dc_branch_flux: np.ndarray | None = None\n",
        1,
    )

# ---------------------------------------------------------------------
# Initialize DC branch flux in __post_init__.
# ---------------------------------------------------------------------
if "self.dc_branch_flux = np.zeros(self.nb, dtype=np.float64)" not in text:
    needle = "        self.nb = self.Bphi.shape[1]\n"
    insert = """        self.nb = self.Bphi.shape[1]

        if self.dc_branch_flux is None:
            self.dc_branch_flux = np.zeros(self.nb, dtype=np.float64)
        else:
            self.dc_branch_flux = np.asarray(self.dc_branch_flux, dtype=np.float64).reshape(-1)
            if self.dc_branch_flux.size != self.nb:
                raise ValueError(f"dc_branch_flux length {self.dc_branch_flux.size} != branch count {self.nb}")
"""
    if needle not in text:
        raise SystemExit("Could not find self.nb initialization.")
    text = text.replace(needle, insert, 1)

# ---------------------------------------------------------------------
# Shift nonlinear current by DC point and subtract static current.
# ---------------------------------------------------------------------
old = """    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi_t = self.branch_flux_time(X)
        i_t = self.branch.current(psi_t)
        return (self.Bphi @ i_t.T).T
"""
new = """    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi_t = self.branch_flux_time(X)
        psi_total_t = psi_t + self.dc_branch_flux[None, :]
        i_t = self.branch.current(psi_total_t) - self.branch.current(self.dc_branch_flux[None, :])
        return (self.Bphi @ i_t.T).T
"""
if old not in text:
    raise SystemExit("Could not find nonlinear_current_time block.")
text = text.replace(old, new, 1)

# ---------------------------------------------------------------------
# Shift tangent gamma by DC point.
# ---------------------------------------------------------------------
old = """        psi_t = (self.BphiT @ x_t.T).T
        gamma_t = self.branch.gamma(psi_t)
        gamma_mean = np.mean(gamma_t, axis=0)
"""
new = """        psi_t = (self.BphiT @ x_t.T).T
        psi_total_t = psi_t + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi_total_t)
        gamma_mean = np.mean(gamma_t, axis=0)
"""
if old not in text:
    raise SystemExit("Could not find tangent_state psi/gamma block.")
text = text.replace(old, new, 1)

# ---------------------------------------------------------------------
# Add CLI argument.
# ---------------------------------------------------------------------
if "--dc-solution" not in text:
    needle = '    p.add_argument("--ipm-dir", default=os.path.join("outputs", "ipm_python_design"))\n'
    repl = needle + '    p.add_argument("--dc-solution", default=None, help="Optional dc_solution.npz or folder containing it. Enables pump solve around a static DC operating point.")\n'
    if needle not in text:
        raise SystemExit("Could not find --ipm-dir argparse line.")
    text = text.replace(needle, repl, 1)

# ---------------------------------------------------------------------
# Load DC solution before constructing HarmonicPumpProblem.
# ---------------------------------------------------------------------
if "dc_x, dc_branch_flux = load_dc_solution(args.dc_solution, ipm)" not in text:
    needle = "    problem = HarmonicPumpProblem(\n"
    repl = """    dc_x, dc_branch_flux = load_dc_solution(args.dc_solution, ipm)
    if dc_branch_flux is not None:
        print(f"dc_solution={args.dc_solution}")
        print(f"dc_branch_flux_max_abs={float(np.max(np.abs(dc_branch_flux))):.12e}")
        print(f"dc_branch_flux_over_phi0_max_abs={float(np.max(np.abs(dc_branch_flux / ipm.phi0))):.12e}")

    problem = HarmonicPumpProblem(
"""
    if needle not in text:
        raise SystemExit("Could not find HarmonicPumpProblem construction.")
    text = text.replace(needle, repl, 1)

# ---------------------------------------------------------------------
# Pass dc_branch_flux to problem constructor.
# ---------------------------------------------------------------------
if "dc_branch_flux=dc_branch_flux," not in text:
    needle = "        pump_current_a=pump_current_a,\n"
    repl = "        pump_current_a=pump_current_a,\n        dc_branch_flux=dc_branch_flux,\n"
    if needle not in text:
        raise SystemExit("Could not find pump_current_a constructor arg.")
    text = text.replace(needle, repl, 1)

# ---------------------------------------------------------------------
# Put DC metadata into pump_report.
# ---------------------------------------------------------------------
if '"dc_solution": args.dc_solution' not in text:
    needle = "    write_results(args.outdir, X, reports, solution_summary, metadata)\n"
    repl = """    metadata["dc_solution"] = args.dc_solution
    metadata["dc_enabled"] = args.dc_solution is not None

    write_results(args.outdir, X, reports, solution_summary, metadata)
"""
    if needle not in text:
        raise SystemExit("Could not find write_results call.")
    text = text.replace(needle, repl, 1)

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp08 supports --dc-solution")
print(f"backup={backup}")
