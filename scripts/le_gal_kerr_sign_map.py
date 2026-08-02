"""Write the analytic SNAIL Kerr sign map without circuit solves."""

import csv
from pathlib import Path

from twpa_solver.core.nonlinear import snail_taylor_coefficients


def main() -> None:
    output = Path("references/le_gal_2025_gain_compression/kerr_sign_map.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ratio", "flux_over_flux0", "equilibrium_flux_rad", "g1_over_Ic", "g2_over_Ic", "g3_over_Ic", "g3_over_g1", "status"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ratio_index in range(61):
            ratio = ratio_index * 0.005
            for flux_index in range(31):
                flux = 0.20 + flux_index * 0.01
                try:
                    result = snail_taylor_coefficients(ratio, flux)
                    row = {
                        "ratio": ratio, "flux_over_flux0": flux,
                        "equilibrium_flux_rad": result["equilibrium_flux_rad"],
                        "g1_over_Ic": result["g1"], "g2_over_Ic": "",
                        "g3_over_Ic": result["g3"],
                        "g3_over_g1": result["g3_over_g1"], "status": "STABLE",
                    }
                except ValueError:
                    row = {"ratio": ratio, "flux_over_flux0": flux, "status": "NO_STABLE_EQUILIBRIUM"}
                writer.writerow(row)


if __name__ == "__main__":
    main()
