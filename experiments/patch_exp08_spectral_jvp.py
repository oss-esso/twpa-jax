from pathlib import Path
import re

path = Path("experiments/exp08_full_ipm_pump_solve.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.before_spectral_jvp")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")

# -------------------------------------------------------------------------
# 1. Add spectral tangent dataclass.
# -------------------------------------------------------------------------
if "class SpectralTangentState" not in text:
    old = """@dataclass
class TangentState:
    gamma_t: np.ndarray
    gamma_mean: np.ndarray


@dataclass
class FullIPMPumpProblem:"""

    new = """@dataclass
class TangentState:
    gamma_t: np.ndarray
    gamma_mean: np.ndarray


@dataclass
class SpectralTangentState:
    khat: dict[int, sp.csr_matrix]


@dataclass
class FullIPMPumpProblem:"""

    if old not in text:
        raise SystemExit("Could not find TangentState block.")
    text = text.replace(old, new)

# -------------------------------------------------------------------------
# 2. Add spectral methods after jvp_coeffs_with_tangent.
# -------------------------------------------------------------------------
if "def spectral_tangent_state" not in text:
    pattern = r"""    def jvp_coeffs_with_tangent\(self, V: np\.ndarray, tangent: TangentState\) -> np\.ndarray:
        JV = np\.empty_like\(V\)
        for h in range\(self\.H\):
            JV\[h\] = self\._linear_blocks\[h\] @ V\[h\]

        v_t = self\.grid\.synthesize\(V\)
        dpsi_t = \(self\.BphiT @ v_t\.T\)\.T

        di_t = tangent\.gamma_t \* dpsi_t
        dn_t = \(self\.Bphi @ di_t\.T\)\.T
        DN = self\.grid\.project_positive\(dn_t\)

        return JV \+ DN
"""

    replacement = """    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
        JV = np.empty_like(V)
        for h in range(self.H):
            JV[h] = self._linear_blocks[h] @ V[h]

        v_t = self.grid.synthesize(V)
        dpsi_t = (self.BphiT @ v_t.T).T

        di_t = tangent.gamma_t * dpsi_t
        dn_t = (self.Bphi @ di_t.T).T
        DN = self.grid.project_positive(dn_t)

        return JV + DN

    def spectral_tangent_state(self, tangent: TangentState) -> SpectralTangentState:
        # For positive unknown harmonics v_q, the real perturbation contains
        # both +q and -q components. Therefore J_k contains K_{k-q} v_q
        # and K_{k+q} conj(v_q). We need ell from 1-H through 2H.
        max_ell = 2 * self.H
        min_ell = 1 - self.H

        theta = self.grid.omega * self.grid.t
        khat: dict[int, sp.csr_matrix] = {}

        for ell in range(min_ell, max_ell + 1):
            phase = np.exp(-1j * ell * theta)
            gamma_hat_ell = np.mean(tangent.gamma_t * phase[:, None], axis=0)

            Kh = (
                self.Bphi
                @ sp.diags(gamma_hat_ell, offsets=0, format="csr")
                @ self.BphiT
            ).astype(np.complex128).tocsr()

            khat[ell] = Kh

        return SpectralTangentState(khat=khat)

    def jvp_coeffs_with_spectral_tangent(
        self,
        V: np.ndarray,
        spectral: SpectralTangentState,
    ) -> np.ndarray:
        JV = np.empty_like(V)

        for k_idx in range(self.H):
            k = k_idx + 1
            acc = self._linear_blocks[k_idx] @ V[k_idx]

            for q_idx in range(self.H):
                q = q_idx + 1

                K_k_minus_q = spectral.khat.get(k - q)
                if K_k_minus_q is not None:
                    acc = acc + K_k_minus_q @ V[q_idx]

                K_k_plus_q = spectral.khat.get(k + q)
                if K_k_plus_q is not None:
                    acc = acc + K_k_plus_q @ np.conj(V[q_idx])

            JV[k_idx] = acc

        return JV
"""

    new_text, n = re.subn(pattern, replacement, text, flags=re.S)
    if n != 1:
        raise SystemExit(f"Could not insert spectral JVP methods; replacements={n}")
    text = new_text

# -------------------------------------------------------------------------
# 3. Add setting field.
# -------------------------------------------------------------------------
old = """    continuation_predictor: str"""
new = """    continuation_predictor: str
    jvp_mode: str"""
if old in text and "    jvp_mode: str" not in text:
    text = text.replace(old, new)

# -------------------------------------------------------------------------
# 4. Add CLI flag.
# -------------------------------------------------------------------------
old = """    p.add_argument("--preconditioner", """
if "--jvp-mode" not in text:
    insert = """    p.add_argument("--jvp-mode", choices=["aft", "spectral"], default="aft")

"""
    idx = text.find(old)
    if idx < 0:
        raise SystemExit("Could not find preconditioner arg insertion point.")
    text = text[:idx] + insert + text[idx:]

# -------------------------------------------------------------------------
# 5. Pass jvp_mode into settings.
# -------------------------------------------------------------------------
old = """        continuation_predictor=args.continuation_predictor,
    )"""
new = """        continuation_predictor=args.continuation_predictor,
        jvp_mode=args.jvp_mode,
    )"""
if old in text and "jvp_mode=args.jvp_mode" not in text:
    text = text.replace(old, new)

# -------------------------------------------------------------------------
# 6. Print and metadata.
# -------------------------------------------------------------------------
old = """    print(f"preconditioner={args.preconditioner}")"""
new = """    print(f"preconditioner={args.preconditioner}")
    print(f"jvp_mode={args.jvp_mode}")"""
if old in text and "print(f\"jvp_mode={args.jvp_mode}\")" not in text:
    text = text.replace(old, new)

old = """        "preconditioner": args.preconditioner,"""
new = """        "preconditioner": args.preconditioner,
        "jvp_mode": args.jvp_mode,"""
if old in text and '"jvp_mode": args.jvp_mode' not in text:
    text = text.replace(old, new)

# -------------------------------------------------------------------------
# 7. Build spectral tangent once per Newton step and use it in matvec.
# -------------------------------------------------------------------------
old = """            tangent = problem.tangent_state(X)

            tf = time.perf_counter()
            factors = problem.build_preconditioner_factors(X, s.preconditioner, tangent=tangent)
            factor_s = time.perf_counter() - tf"""

new = """            tangent = problem.tangent_state(X)
            spectral_tangent = None
            if s.jvp_mode == "spectral":
                spectral_tangent = problem.spectral_tangent_state(tangent)

            tf = time.perf_counter()
            factors = problem.build_preconditioner_factors(X, s.preconditioner, tangent=tangent)
            factor_s = time.perf_counter() - tf"""

if old not in text:
    raise SystemExit("Could not find tangent/preconditioner block.")
text = text.replace(old, new)

old = """                JV = problem.jvp_coeffs_with_tangent(V, tangent)"""
new = """                if s.jvp_mode == "spectral":
                    JV = problem.jvp_coeffs_with_spectral_tangent(V, spectral_tangent)
                else:
                    JV = problem.jvp_coeffs_with_tangent(V, tangent)"""

if old not in text:
    raise SystemExit("Could not find JVP line in matvec.")
text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print("PATCH_OK spectral JVP mode added")
print(f"backup={backup}")
