"""Compare one explicit three-large/one-small SNAIL cell with the reduction.

For total reduced flux ``phi`` and external half flux ``pi``, the static
four-junction cell potential is
``U/EJ = -3 cos(phi/3) - r cos(pi - phi)``.  Its current law is therefore
``I/Ic = sin(phi/3) - r sin(phi)``.  We solve its equilibrium and differentiate
the law analytically at that operating point, retaining the physical ratio
``g3/g1`` rather than relying on either coefficient's mirror-root sign.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def four_junction_coefficients(ratio: float = 0.062, flux_over_flux0: float = 0.5) -> dict[str, float]:
    phi_ext = 2.0 * np.pi * flux_over_flux0
    current = lambda phi: np.sin(phi / 3.0) - ratio * np.sin(phi_ext - phi)
    equilibrium = brentq(current, -np.pi, np.pi)
    g1 = np.cos(equilibrium / 3.0) / 3.0 + ratio * np.cos(phi_ext - equilibrium)
    g3 = -np.cos(equilibrium / 3.0) / 162.0 - ratio * np.cos(phi_ext - equilibrium) / 6.0
    return {
        "ratio": ratio, "flux_over_flux0": flux_over_flux0,
        "equilibrium_flux_rad": equilibrium, "g1_over_Ic": float(g1),
        "g3_over_Ic": float(g3), "g3_over_g1": float(g3 / g1),
    }


def main() -> None:
    result = four_junction_coefficients()
    result["reduction_g1_over_Ic"] = 0.271333333
    result["reduction_g3_over_Ic"] = 0.004160494
    result["reduction_g3_over_g1"] = 0.015333515
    result["ratio_to_reduction"] = result["g3_over_g1"] / result["reduction_g3_over_g1"]
    Path("references/le_gal_four_junction_snail_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
