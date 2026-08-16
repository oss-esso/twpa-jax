# TD-to-HB residual homotopy

The production 2c map remains an HB map.  A transient bridge is used only when
fixed-drive Newton and bounded continuation cannot reach the next power point.

For a Fourier projection `X_td` of a settled transient orbit, the recovery
solver continues the translated residual

```text
F(X) - (1 - eta) F(X_td) = 0,   eta: 0 -> 1.
```

At `eta = 0`, the supplied TD projection is an exact root of the translated
problem.  At `eta = 1`, the residual is exactly the unchanged production HB
residual.  The implementation delegates all JVP, Schur reduction, and sparse
preconditioning operations to the existing pump problem.  No circuit damping,
source modification, Gmin, or physical parameter change is introduced.

The method is exposed as
`HarmonicNewtonKrylovSolver.solve_residual_homotopy()` and is used by
`scripts/run_hybrid_column.py` for TD PERIOD_1 handoff.  Each trial is bounded
by an adaptive eta step and an optional wall-time budget.  A state is accepted
only after the final eta reaches one and the normal production solve-point
validation succeeds.

An unpolished TD projection is deliberately not evaluated as an ordinary gain
point.  Its checkpoint metadata records the projection and remains classified
as unresolved until HB validation passes.

The transient projection now averages several final periods using an
independent Fourier sampling grid.  This avoids coupling the HB projection
quality to the transient output sampling while keeping storage bounded.
