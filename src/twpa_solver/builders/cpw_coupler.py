"""Conformal-mapping CPW coupler model used by the design compiler.

The two-conductor and three-conductor cases follow the CPW calculation used
by the layout scripts.  The three-conductor case includes the centre ground
plane and is useful for weak coupling, where a two-line cross section runs
out of geometric range.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    def __init__(
        self,
        gaps_um: list[float],
        widths_um: list[float],
        length_um: float,
    ) -> None:
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
        x = 0.0
        y = self.gaps[0]
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
            return self._fallback_matrices()
        inductance = np.linalg.inv(capacitance) / self.v**2
        return capacitance, inductance

    def _fallback_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        from twpa_solver.builders.ipm import edge_coupled_cpw
        gap = (self.gaps[1] if len(self.widths) == 2
               else 2.0 * self.gaps[1] + self.widths[1])
        unit = edge_coupled_cpw(self.widths[0], self.gaps[0], gap)
        # edge_coupled_cpw returns direct modal self/mutual values. Convert
        # them to the Maxwell/nodal convention consumed by parameters().
        capacitance = np.array([
            [unit["C_self"] + unit["C_mutual"], -unit["C_mutual"]],
            [-unit["C_mutual"], unit["C_self"] + unit["C_mutual"]],
        ])
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


@lru_cache(maxsize=None)
def optimize_cpw_coupler(
    coupling_db: float,
    frequency_hz: float,
    z0: float = 50.0,
    model: str = "auto",
    initial_gaps_um: tuple[float, ...] | None = None,
    initial_widths_um: tuple[float, ...] | None = None,
) -> CPWModeResult:
    if frequency_hz <= 0.0 or coupling_db >= 0.0:
        raise ValueError("frequency must be positive and coupling_db must be negative")
    selected = model
    if selected == "auto":
        # The specification boundary is -20 dB; v1 (-14 dB) and v3 (-25 dB)
        # remain on their existing sides of this threshold.
        selected = "three_line" if coupling_db < -20.0 else "two_line"
    if selected not in {"two_line", "three_line"}:
        raise ValueError(f"unknown CPW model {model!r}")

    if selected == "two_line":
        gaps = initial_gaps_um or (10.5973385055, 44.762, 10.5973385055)
        widths = initial_widths_um or (39.897, 39.897)
        if len(gaps) != 3 or len(widths) != 2:
            raise ValueError("two_line initial geometry has invalid dimensions")
        initial = np.array([gaps[0], widths[0], gaps[1]], dtype=float)
        lower = np.array([4.0, 4.0, 4.0], dtype=float)
        upper = np.array([50.0, 50.0, 100.0], dtype=float)

        def evaluate(values: np.ndarray) -> dict[str, float]:
            gap_to_ground, width, gap_between = values
            return CPWConformalCoupler(
                [gap_to_ground, gap_between, gap_to_ground],
                [width, width],
                1000.0,
            ).parameters()

        def residual(values: np.ndarray) -> list[float]:
            parameters = evaluate(values)
            return [parameters["coupling_db"] - coupling_db,
                    parameters["Z_eff"] - z0]

        solution = least_squares(residual, initial, bounds=(lower, upper),
                                 diff_step=0.1, max_nfev=30)
        values = solution.x
        parameters = evaluate(values)
        gaps = (float(values[0]), float(values[2]), float(values[0]))
        widths = (float(values[1]), float(values[1]))
    else:
        gaps = initial_gaps_um or (5.5, 5.0, 5.0, 5.5)
        widths = initial_widths_um or (9.186, 15.0, 9.186)
        if len(gaps) != 4 or len(widths) != 3:
            raise ValueError("three_line initial geometry has invalid dimensions")
        initial = np.array([gaps[0], widths[0], gaps[1], widths[1]], dtype=float)
        lower = np.array([4.0, 4.0, 4.0, 4.0], dtype=float)
        upper = np.array([50.0, 50.0, 100.0, 50.0], dtype=float)

        def evaluate(values: np.ndarray) -> dict[str, float]:
            gap_to_ground, signal_width, gap_to_centre, centre_width = values
            return CPWConformalCoupler(
                [gap_to_ground, gap_to_centre, gap_to_centre, gap_to_ground],
                [signal_width, centre_width, signal_width],
                1000.0,
            ).parameters()

        def residual(values: np.ndarray) -> list[float]:
            parameters = evaluate(values)
            return [parameters["coupling_db"] - coupling_db,
                    parameters["Z_eff"] - z0]

        solution = least_squares(residual, initial, bounds=(lower, upper),
                                 diff_step=0.1, max_nfev=30)
        values = solution.x
        parameters = evaluate(values)
        gaps = (float(values[0]), float(values[2]), float(values[2]),
                float(values[0]))
        widths = (float(values[1]), float(values[3]), float(values[1]))

    residual_error = max(abs(value) for value in residual(values))
    if not solution.success and residual_error > 1.0e-2:
        raise RuntimeError(f"CPW geometry optimization failed: {solution.message}")

    beta_even = 2.0 * math.pi * frequency_hz / parameters["v_even"]
    # Deliberately reproduce the Prometheus getCouplerDimentions calculation:
    # its beta_odd expression uses v_even, although theory would use v_odd.
    beta_odd = 2.0 * math.pi * frequency_hz / parameters["v_even"]
    length_um = math.pi / (beta_even + beta_odd) * 1.0e6
    return CPWModeResult(
        gaps,
        widths,
        length_um,
        float(parameters["coupling_db"]),
        float(parameters["Z_eff"]),
        selected,
    )
