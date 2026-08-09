from pathlib import Path
import re

path = Path("experiments/exp08_full_ipm_pump_solve.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_tangent_cache")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# 1. Add TangentState dataclass.
if "class TangentState" not in text:
    old = """# =============================================================================
# Pump problem
# =============================================================================

@dataclass
class FullIPMPumpProblem:"""

    new = """# =============================================================================
# Pump problem
# =============================================================================

@dataclass
class TangentState:
    gamma_t: np.ndarray
    gamma_mean: np.ndarray


@dataclass
class FullIPMPumpProblem:"""

    if old not in text:
        raise SystemExit("Could not find Pump problem marker for TangentState insertion.")
    text = text.replace(old, new)

# 2. Cache Bphi.T once.
old = """        self.Bphi = self.Bphi.tocsr()

        self.n = self.C.shape[0]"""

new = """        self.Bphi = self.Bphi.tocsr()
        self.BphiT = self.Bphi.T.tocsr()

        self.n = self.C.shape[0]"""

if "self.BphiT = self.Bphi.T.tocsr()" not in text:
    if old not in text:
        raise SystemExit("Could not find Bphi csr block.")
    text = text.replace(old, new)

# 3. Use cached transpose wherever possible.
text = text.replace("self.Bphi.T @", "self.BphiT @")

# 4. Replace jvp_coeffs block with tangent-cached version.
pattern = r"""    def jvp_coeffs\(self, X: np\.ndarray, V: np\.ndarray\) -> np\.ndarray:
.*?
    def time_residual"""

replacement = """    def tangent_state(self, X: np.ndarray) -> TangentState:
        x_t = self.grid.synthesize(X)
        psi_t = (self.BphiT @ x_t.T).T
        gamma_t = self.branch.gamma(psi_t)
        gamma_mean = np.mean(gamma_t, axis=0)
        return TangentState(gamma_t=gamma_t, gamma_mean=gamma_mean)

    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
        JV = np.empty_like(V)
        for h in range(self.H):
            JV[h] = self._linear_blocks[h] @ V[h]

        v_t = self.grid.synthesize(V)
        dpsi_t = (self.BphiT @ v_t.T).T

        di_t = tangent.gamma_t * dpsi_t
        dn_t = (self.Bphi @ di_t.T).T
        DN = self.grid.project_positive(dn_t)

        return JV + DN

    def jvp_coeffs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        tangent = self.tangent_state(X)
        return self.jvp_coeffs_with_tangent(V, tangent)

    def time_residual"""

new_text, n = re.subn(pattern, replacement, text, flags=re.S)
if n != 1:
    raise SystemExit(f"Could not replace jvp_coeffs block cleanly; replacements={n}")
text = new_text

# 5. Let the preconditioner reuse the tangent.
old = """    def build_preconditioner_factors(self, X: np.ndarray, mode: str) -> list[spla.SuperLU] | None:"""
new = """    def build_preconditioner_factors(self, X: np.ndarray, mode: str, tangent: TangentState | None = None) -> list[spla.SuperLU] | None:"""
text = text.replace(old, new)

old = """        psi_t = self.branch_flux_time(X)
        gamma_mean = np.mean(self.branch.gamma(psi_t), axis=0)"""

new = """        if tangent is None:
            tangent = self.tangent_state(X)
        gamma_mean = tangent.gamma_mean"""

if old in text:
    text = text.replace(old, new)

# 6. In each Newton step, build tangent once and use it for GMRES matvecs/preconditioner.
old = """            tf = time.perf_counter()
            factors = problem.build_preconditioner_factors(X, s.preconditioner)
            factor_s = time.perf_counter() - tf"""

new = """            tangent = problem.tangent_state(X)

            tf = time.perf_counter()
            factors = problem.build_preconditioner_factors(X, s.preconditioner, tangent=tangent)
            factor_s = time.perf_counter() - tf"""

if old not in text:
    raise SystemExit("Could not find preconditioner build block in solver.")
text = text.replace(old, new)

old = """                JV = problem.jvp_coeffs(X, V)"""
new = """                JV = problem.jvp_coeffs_with_tangent(V, tangent)"""
if old not in text:
    raise SystemExit("Could not find matvec JVP line.")
text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("PATCH_OK tangent-cache optimization applied")
print(f"backup={backup}")
