# Long conventional JTWPA versus true IPM ablation

This milestone keeps two questions separate:

1. `UNIFORM_JTWPA_2508` is a conventional long, uniform Josephson transmission-line
   benchmark. It is not an IPM-with-couplers-removed circuit.
2. `IPM_SINGLE_NONLINEAR_SECTION` is one actual 418-cell nonlinear section from the
   production IPM, with the couplers and auxiliary paths removed and 50-ohm source/load
   embedding retained.

All HB states used here are generated and validated by `twpa_solver.pump` through the
production validation contract. Outputs are written beneath the gitignored `outputs/`
tree; legacy untracked experiment directories were moved to
`outputs/legacy_experiments/`.

## Production and topology audit

The production IPM builder uses `IPMParams(array_length=418, num_rows=6)`. The top
nonlinear path contains six consecutive 418-cell sections, for 2508 JJs total. A
real directional coupler is inserted after every third section; the lower path is an
auxiliary/coupler embedding path, not a second nonlinear 418-cell line in the current
`make_ipm` implementation. The full H3 result is the verified endpoint reference.

The repository's separate conventional fixture is `build_jtwpa()` in
`src/twpa_solver/builders/jc_doc.py`: 2048 JJs, `Lj=ic_to_lj(3.4 uA)`, `Cj=55 fF`,
`Cg=45 fF`, and periodic capacitive/resonator loading every four cells. It is a
`CLOSE_CONVENTIONAL_JTWPA_PROXY`, not an exact 2502-JJ production match. The synthetic
2508 uniform line uses production IPM `Lj=123.9 pH`, `Cj=145 fF`, `Cg=66 fF`, no
periodic loading, 7.9 GHz drive, and 50-ohm terminations, so it is also only a
generic/conventional long-line benchmark.

## Initial controls

| topology / test | drive | result | local `r_J` | production HB residual |
|---|---:|---|---:|---:|
| `UNIFORM_JTWPA_2508`, constant drive | 1.0 Ic | `PERIOD_1` | 0.592 | 2.64e-10 |
| `IPM_SINGLE_NONLINEAR_SECTION`, constant drive | 1.0 Ic | `PERIOD_1` | 0.580 | 5.95e-11 |
| `IPM_SINGLE_NONLINEAR_SECTION`, 0.5→1.25 ramp | 1.25 Ic | broadband candidate | 0.700 | 9.43e-13 (start) |
| `IPM_SINGLE_NONLINEAR_SECTION`, 0.5→1.50 ramp | 1.50 Ic | broadband candidate | 0.829 | 9.43e-13 (start) |

The 2508 constant-drive control has a successful transient and HB round trip, with
stroboscopic tail distance about `7.2e-5` and negligible winding. Thus the earlier
1.0-Ic ramp broadband observation is not, by itself, a fixed-drive long-line
instability; it is consistent with ramp/basin selection.

The 418-cell section likewise has a stable fixed-drive PERIOD_1 control at 1.0 Ic.
Its higher-drive ramps select a broadband state, but a fresh production HB solve at
1.5 Ic does not converge (the continuation route stalls at full source scale).
Therefore those ramps do not yet establish a topology threshold or a physical
instability of a valid high-drive PERIOD_1 root.

## Current decision

The evidence currently supports:

- Branch A: `RAMP_BASIN_EFFECT` for the tested 2508 1.0-Ic control, not a proven
  `LONG_LINE_PHYSICAL_INSTABILITY`.
- Branch B: the single-section result is not sufficient to attribute the full-IPM
  limit because its high-drive fixed-drive root has not been established.

Overall status: `INCONCLUSIVE`. No coupler ablation or RCSJ experiment should be
started until the high-drive section control is obtained or the production solver's
explicit no-root boundary is separately reviewed.
