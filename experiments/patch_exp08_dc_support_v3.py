from pathlib import Path
import re

path = Path("experiments/exp08_full_ipm_pump_solve.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_dc_support_v3")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# Insert helper before @dataclass/class FullIPMPumpProblem.
if "def load_dc_solution(" not in text:
    class_pos = text.find("class FullIPMPumpProblem")
    if class_pos < 0:
        raise SystemExit("Could not find class FullIPMPumpProblem.")

    insert_pos = class_pos
    before = text[:class_pos]
    # If immediately preceded by @dataclass, insert before the decorator.
    m = re.search(r"(?m)^@dataclass\s*$", before)
    if m:
        # Use the last @dataclass before the class.
        insert_pos = m.start()

    helper = r'''

def load_dc_solution(dc_solution: str | Path | None, ipm: LoadedIPM) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a static DC operating point for shifted pump solves."""
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
    text = text[:insert_pos] + helper + "\n" + text[insert_pos:]

# Add dc_branch_flux field after pump_current_a.
if "dc_branch_flux: np.ndarray | None = None" not in text:
    text, n = re.subn(
        r"(?m)^(\s*)pump_current_a:\s*float\s*$",
        r"\1pump_current_a: float\n\1dc_branch_flux: np.ndarray | None = None",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("Could not add dc_branch_flux field near pump_current_a.")

# Initialize after self.nb.
if "self.dc_branch_flux = np.zeros(self.nb, dtype=np.float64)" not in text:
    old = "        self.nb = self.Bphi.shape[1]\n"
    new = """        self.nb = self.Bphi.shape[1]

        if self.dc_branch_flux is None:
            self.dc_branch_flux = np.zeros(self.nb, dtype=np.float64)
        else:
            self.dc_branch_flux = np.asarray(self.dc_branch_flux, dtype=np.float64).reshape(-1)
            if self.dc_branch_flux.size != self.nb:
                raise ValueError(f"dc_branch_flux length {self.dc_branch_flux.size} != branch count {self.nb}")
"""
    if old not in text:
        raise SystemExit("Could not find self.nb initialization.")
    text = text.replace(old, new, 1)

# Shift nonlinear current by DC point.
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

# Shift tangent gamma by DC point.
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

# Add CLI argument.
if "--dc-solution" not in text:
    old = '    p.add_argument("--ipm-dir", default=os.path.join("outputs", "ipm_python_design"))\n'
    new = old + '    p.add_argument("--dc-solution", default=None, help="Optional dc_solution.npz or folder containing it. Enables pump solve around a static DC operating point.")\n'
    if old not in text:
        raise SystemExit("Could not find --ipm-dir argparse line.")
    text = text.replace(old, new, 1)

# Load DC solution before FullIPMPumpProblem construction.
if "dc_x, dc_branch_flux = load_dc_solution(args.dc_solution, ipm)" not in text:
    old = "    problem = FullIPMPumpProblem(\n"
    new = """    dc_x, dc_branch_flux = load_dc_solution(args.dc_solution, ipm)
    if dc_branch_flux is not None:
        print(f"dc_solution={args.dc_solution}")
        print(f"dc_branch_flux_max_abs={float(np.max(np.abs(dc_branch_flux))):.12e}")
        print(f"dc_branch_flux_over_phi0_max_abs={float(np.max(np.abs(dc_branch_flux / ipm.phi0))):.12e}")

    problem = FullIPMPumpProblem(
"""
    if old not in text:
        raise SystemExit("Could not find FullIPMPumpProblem construction.")
    text = text.replace(old, new, 1)

# Pass dc_branch_flux into constructor.
if "dc_branch_flux=dc_branch_flux," not in text:
    old = "        pump_current_a=pump_current_a,\n"
    new = "        pump_current_a=pump_current_a,\n        dc_branch_flux=dc_branch_flux,\n"
    if old not in text:
        raise SystemExit("Could not find pump_current_a constructor arg.")
    text = text.replace(old, new, 1)

# Add DC metadata.
if '"dc_solution": args.dc_solution' not in text:
    old = "    write_results(args.outdir, X, reports, solution_summary, metadata)\n"
    new = """    metadata["dc_solution"] = args.dc_solution
    metadata["dc_enabled"] = args.dc_solution is not None

    write_results(args.outdir, X, reports, solution_summary, metadata)
"""
    if old not in text:
        raise SystemExit("Could not find write_results call.")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("PATCH_OK Exp08 supports --dc-solution using FullIPMPumpProblem")
print(f"backup={backup}")
