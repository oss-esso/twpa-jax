"""Consolidate exp35's assembled-dispersion result."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    """Write the dispersion-only report after exp38 superseded its Kerr budget."""
    source = Path(
        "references/le_gal_2025_gain_compression/exp35_dispersion.json"
    )
    dispersion = json.loads(source.read_text(encoding="utf-8"))
    builder_max = float(dispersion["builder_max_relative_error"])
    builder_rms = float(dispersion["builder_rms_relative_error"])
    cme_max = float(dispersion["cme_max_relative_error"])
    cme_rms = float(dispersion["cme_rms_relative_error"])
    report_text = f"""# exp35 Le Gal dispersion validation

The Bloch eigenvalue is extracted from the assembled residual linearization:
`K` plus the effective-SNAIL branch tangent. The builder ladder relation agrees
with it; the older CME ground-capacitance relation does not.

| candidate | max relative deviation | RMS relative deviation |
| --- | ---: | ---: |
| builder ladder | {builder_max:.3e} | {builder_rms:.3e} |
| old CME ground-capacitance form | {cme_max:.6f} | {cme_rms:.6f} |

The phase-budget and HB/CME sections formerly in this report are superseded by
`docs/development/exp38_le_gal_kerr_verdict.md`. exp38 found that the Le Gal
builder had also stamped the SNAIL tangent into `K`, double-counting the branch
stiffness in HB; therefore the old exp35 HB gain comparison is historical.
"""
    report = Path("docs/development/exp35_le_gal_dispersion_report.md")
    report.write_text(report_text, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
