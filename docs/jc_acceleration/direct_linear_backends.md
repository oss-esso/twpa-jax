# Direct Linear Backends

## Summary

The direct linear backend phase replaced unnecessary nonlinear-wrapper calls in
selected linear Harmonia/JosephsonCircuits workflows.

Old path:

```julia
hbsolve(...).linearized.S(...)

New opt-in path:

hblinsolve(...)

The new path is only enabled after an equivalence probe proves that the resulting
S-parameters match the existing path.

Integrated backends
Circuit family    Config key    Direct backend value
JTL linear    solver.jtl_linear_backend    hblinsolve_direct
RF-JTL linear    solver.rf_jtl_linear_backend    hblinsolve_direct
ETHZ-JTL linear    solver.ethz_jtl_linear_backend    hblinsolve_direct
Safety guards
Default remains hbsolve.
Direct backend is opt-in.
Direct backend requires zero pump current.
Status telemetry records the backend.
Numeric matrix cache is not silently enabled.
Nonlinear gain workflows still require full HB.
Current decision

Stop rollout after JTL, RF-JTL, and ETHZ-JTL.

Do not patch lumped_jpa_linear until a separate equivalence probe proves it is
the same linear-response pattern.
