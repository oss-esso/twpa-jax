# Saturation solver power-balance report

Phase 4 emits conservation diagnostics for every newly solved compression point.
The CSV fields are `power_balance_rel_err`, `manley_rowe_photon_flux`, and
`manley_rowe_rel_err`; the summary records their maxima without making either
quantity a run-failing gate.

The lossless and lossy fixtures both pass their conservation gates: the lossless
fixture conserves Manley--Rowe photon flux below `1e-12`, and the lossy fixture
closes the supplied-minus-dissipated balance below `1e-12`.

The checked-in exp20 compression artifacts predate this instrumentation: all
six exp20 operating-point CSVs contain neither conservation column, so their
measured Phase 4 maximum is **not available**, rather than zero. A fresh
Phase-4-enabled exp20 campaign must populate this table before interpreting
deep-saturation residuals. This is deliberately reported as missing data; the
driver does not silently backfill or gate on it.

| artifact set | operating points | points with balance | max balance residual |
| --- | ---: | ---: | ---: |
| exp20 multitone compression (jpa/jtwpa/fqjtwpa, S=2/4) | 6 | 0 | not available (pre-Phase 4) |

