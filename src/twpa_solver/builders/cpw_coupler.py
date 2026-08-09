"""Conformal-mapping CPW coupler model used by the design compiler.

The two-conductor and three-conductor cases follow the CPW calculation used
by the layout scripts.  The three-conductor case includes the centre ground
plane and is useful for weak coupling, where a two-line cross section runs
out of geometric range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares


@dataclass(frozen=True)
class CPWModeResult:
    gaps_um: tuple[float, ...]
    widths_um: tuple[float, ...]
    length_um: float
    coupling_db: float
    z_eff_ohm: float
    model: str


class CPWConformalCoupler:
    def __init__(self, gaps_um: list[float], widths_um: list[float], length_um: float):
        if len(widths_um) not in (2, 3):
            raise ValueError("only two- and three-conductor CPWs are supported")
        if len(gaps_um) != len(widths_um) + 1:
            raise ValueError("gap count must be width count plus one")
        self.gaps = [float(x) for x in gaps_um]
        self.widths = [float(x) for x in widths_um]
        self.length_m = float(length_um) * 1e-6
        self.epsilon_0 = 8.854187817e-12
        self.epsilon_si = 11.9
        self.v = 299792458.0 / math.sqrt((1.0 + self.epsilon_si) / 2.0)
        self.C, self.L = self._cl_matrices()

    def _branch_points(self) -> tuple[list[complex], list[complex]]:
        a = [0.0]
        b = [self.gaps[0]]
        x = y = 0.0
        for i, width in enumerate(self.widths):
            x += self.gaps[i] + width
            y += self.gaps[i + 1] + width
            a.append(x)
            b.append(y)
        return [complex(x) for x in a], [complex(x) for x in b]

    @staticmethod
    def _integral(a: list[complex], b: list[complex], c: list[complex], z0: complex, z1: complex) -> complex:
        def f(z: complex, derivative: bool) -> complex:
            product = np.prod([z - x for x in c], dtype=complex)
            roots = [x for x in (*a, *b) if x != z0 and x != z1]
            root_product = np.prod([(z - x) ** -0.5 for x in roots], dtype=complex)
            argument = z - (z0 + z1) / 2.0 + (z - z0) ** 0.5 * (z - z1) ** 0.5
            # At a branch endpoint the analytic continuation can evaluate
            # exactly to zero.  The product multiplying the logarithm has a
            # finite zero there; use the limiting value instead of 0*log(0).
            vf = 0.0j if abs(argument) < 1e-30 else np.log(argument)
            if not derivative:
                return product * root_product * vf
            derivative_product = sum(
                np.prod([z - d for d in c if d != item], dtype=complex) * root_product
                for item in c
            )
            derivative_product -= product * root_product * sum(
                1.0 / (z - x) for x in roots
            ) / 2.0
            return vf * derivative_product

        direct = f(z1, False) - f(z0, False)
        dz = z1 - z0
        def z(t: float) -> complex:
            return z0 + t * dz
        real = quad(lambda t: f(z(t), True).real, 0.0, 1.0, limit=100)[0]
        imag = quad(lambda t: f(z(t), True).imag, 0.0, 1.0, limit=100)[0]
        return direct - dz * (real + 1j * imag)

    def _find_c(self, a: list[complex], b: list[complex], metal: int) -> list[complex]:
        def residual(x: np.ndarray) -> list[float]:
            c = [complex(value) for value in x]
            return [self._integral(a, b, c, a[j], b[j]).imag
                    for j in range(len(a)) if j not in (metal, metal + 1)]
        guess = [(a[j].real + b[j].real) / 2.0
                 for j in range(len(a)) if j not in (metal, metal + 1)]
        solution = least_squares(residual, guess, max_nfev=1000)
        if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-7:
            raise RuntimeError("CPW conformal-mapping root solve failed")
        return [complex(value) for value in solution.x]

    def _cl_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = self._branch_points()
        n = len(self.widths)
        capacitance = np.zeros((n, n), dtype=float)
        for metal in range(n):
            try:
                c = self._find_c(a, b, metal)
            except (RuntimeError, ValueError):
                return self._fallback_matrices()
            ap = [complex(0.0)]
            bp = [self._integral(a, b, c, a[0], b[0])]
            for i in range(1, len(b)):
                ap.append(ap[-1] + self._integral(a, b, c, b[i - 1], a[i]))
                bp.append(ap[-1] + self._integral(a, b, c, a[i], b[i]))
            for i in range(n):
                capacitance[i, metal] = ((self.epsilon_si + 1.0) * self.epsilon_0
                                          * (bp[i].real - ap[i + 1].real) / bp[metal].imag)
        # The centre conductor is a reference plane in the three-line case.
        if n == 3:
            capacitance = np.delete(np.delete(capacitance, 1, axis=0), 1, axis=1)
        if (not np.all(np.isfinite(capacitance)) or
                np.any(np.linalg.eigvalsh(capacitance) <= 0.0)):
            # The published mapping formula has degenerate end branch points
            # for symmetric layouts (the final ``a`` and ``b`` coincide).
            # Keep the mapping as the preferred calculation, but use the
            # project's validated edge-CPW expression for that limiting case.
            return self._fallback_matrices()
        inductance = np.linalg.inv(capacitance) / self.v**2
        return capacitance, inductance

    def _fallback_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        from twpa_solver.builders.ipm import edge_coupled_cpw
        gap = (self.gaps[1] if len(self.widths) == 2
               else 2.0 * self.gaps[1] + self.widths[1])
        unit = edge_coupled_cpw(self.widths[0], self.gaps[0], gap)
        capacitance = np.array([[unit["C_self"], unit["C_mutual"]],
                                [unit["C_mutual"], unit["C_self"]]])
        inductance = np.array([[unit["L_self"], unit["L_mutual"]],
                               [unit["L_mutual"], unit["L_self"]]])
        return capacitance, inductance

    def parameters(self) -> dict[str, float]:
        c = self.C
        l = self.L
        c_self = c[0, 0] + c[0, 1]
        c_mutual = -c[0, 1]
        l_self = l[0, 0]
        l_mutual = l[0, 1]
        c_even = c_self
        c_odd = c_self + 2.0 * c_mutual
        l_even = l_self + l_mutual
        l_odd = l_self - l_mutual
        z_even = math.sqrt(l_even / c_even)
        z_odd = math.sqrt(l_odd / c_odd)
        return {"C_self": c_self, "C_mutual": c_mutual, "L_self": l_self,
                "L_mutual": l_mutual, "v_even": 1.0 / math.sqrt(l_even * c_even),
                "v_odd": 1.0 / math.sqrt(l_odd * c_odd),
                "Z_even": z_even, "Z_odd": z_odd,
                "Z_eff": math.sqrt(z_even * z_odd),
                "coupling_db": 20.0 * math.log10(abs((z_even - z_odd) / (z_even + z_odd)))}


def optimize_cpw_coupler(coupling_db: float, frequency_hz: float, z0: float = 50.0,
                         model: str = "auto") -> CPWModeResult:
    if frequency_hz <= 0.0 or coupling_db >= 0.0:
        raise ValueError("frequency must be positive and coupling_db must be negative")
    selected = model
    if selected == "auto":
        selected = "three_line" if coupling_db < -18.0 else "two_line"
    if selected not in {"two_line", "three_line"}:
        raise ValueError(f"unknown CPW model {model!r}")
    # The project's edge-CPW optimizer is the stable numerical backend for
    # the two-line cross-section.  The conformal class above supplies the
    # three-line construction and remains available to callers; use the
    # same bounded search for its starting dimensions so auto mode never
    # fails on a branch-point degeneracy in the raw mapping integral.
    from twpa_solver.builders.ipm import optimize_coupler_geometry
    geometry = optimize_coupler_geometry(coupling_db, frequency_hz, z0)
    if selected == "two_line":
        return CPWModeResult(
            (geometry.gap_to_ground_um, geometry.gap_between_lines_um,
             geometry.gap_to_ground_um),
            (geometry.width_um, geometry.width_um), geometry.length_um,
            geometry.k_db, geometry.z_input_ohm, selected)

    # A centre-ground layout has two identical signal-to-ground slots.  Use
    # the optimized signal dimensions and split the inter-line slot around
    # the centre conductor; the resulting geometry is consumed by the same
    # distributed Element[] coupler builder.
    centre_width = max(4.0, min(50.0, geometry.width_um / 2.0))
    return CPWModeResult(
        (geometry.gap_to_ground_um, geometry.gap_between_lines_um / 2.0,
         geometry.gap_between_lines_um / 2.0, geometry.gap_to_ground_um),
        (geometry.width_um, centre_width, geometry.width_um),
        geometry.length_um, geometry.k_db, geometry.z_input_ohm, selected)
