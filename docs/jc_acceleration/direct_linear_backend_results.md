# Direct Linear Backend Results

## Summary

We added opt-in direct `hblinsolve` backends for linear Harmonia/JosephsonCircuits workflows:

- JTL linear
- RF-JTL linear
- ETHZ-JTL linear

The previous path was:

```julia
hbsolve(...).linearized.S(...)

The new opt-in path is:

hblinsolve(...)

This is a solver-semantics cleanup and linear-response acceleration path. It is not a nonlinear pumped-gain HB acceleration claim.

Correctness

For all old-vs-direct comparisons, the S-parameters matched exactly:

max_abs_diff_vs_hbsolve = 0.0

Largest exact comparisons:

Family    Cells    Elements    Nodes    JC tuples
JTL    3000    6004    3002    9004
RF-JTL    2393    9576    4788    11969
ETHZ-JTL    2048    6653    3326    8700
Extreme direct-only runs
Family    Cells    Elements    Nodes    JC tuples    Time
JTL    10000    20004    10002    30004    ~2.67 s
JTL    20000    40004    20002    60004    ~9.01 s
JTL    30000    60004    30002    90004    ~40.23 s
RF-JTL    5000    20004    10002    25004    ~12.00 s
ETHZ-JTL    5000    16247    8123    21246    ~1.37 s
ETHZ-JTL    10000    32497    16248    42496    ~6.48 s
Boundary case

RF-JTL at 10000 cells failed with:

RF-JTL linear S-parameters contain non-finite values

This is recorded as a numerical/conditioning boundary, not hidden.

Interpretation

The direct backend does not give a universal speedup over warm hbsolve at large scale, because the old zero-pump path is already close to the linearized solve internally. The real improvement is:

exact-equivalent linear solver semantics;
explicit backend telemetry;
faster and cleaner cold/smoke workflows;
scalable large linear-response checks;
honest failure boundaries.
Decision

The direct-linear rollout is complete for JTL, RF-JTL, and ETHZ-JTL.

Do not patch lumped_jpa_linear without a separate equivalence probe.
