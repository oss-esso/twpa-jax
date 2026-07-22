# Roadmap

1. RF-JTL S-parameter finite/non-finite boundary map.
2. Frequency-window diagnosis.
3. Parameter sensitivity for `Lrf_H`, `Lp_H`, and port impedance.
4. Frequency and pump-frequency operating maps.
5. Nonlinear HB/gain-map workflow.
6. GPU/accelerator reassessment only after numerical validity and CPU bottlenecks are understood.

## Immediate Deliverables

- A cell-count sweep for RF-JTL direct linear response.
- A frequency-window sweep around the failing 10000-cell RF-JTL case.
- A report that preserves all non-finite failures with resolved configs.

## Deferred Work

- `lumped_jpa_linear` direct backend.
- Nonlinear pumped direct-backend claims.
- GPU porting.
- Silent setup-cache enablement inside production solver paths.
