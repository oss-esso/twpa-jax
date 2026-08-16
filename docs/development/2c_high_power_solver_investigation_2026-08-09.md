# 2c high-power solver investigation — 2026-08-09

## Scope and reproducibility

This investigation used the production 2c model, not a reduced fixture. The
authoritative netlist source is `designs/ipm_2c.yaml`. The generated directory
`designs/ipm_2c_fixed` was rebuilt through `workflows/build_design_and_passive.py`
and then checked against `outputs/2c_netlist_baseline.csv`.

The check is byte-identical:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `outputs/2c_netlist_baseline.csv` | 1,014,288 | `7F1A4746001A15B6DE527902A23281F0C41D42804CE4A6D4C79FA8171C997BD6` |
| `designs/ipm_2c_fixed/elements.csv` | 1,014,288 | same |

The build parity required two corrections. The 2c YAML now selects the cached
coupler geometry used by the baseline, and the generic CSV writer uses the
legacy one-based `idx` convention. Before these corrections, the default
`auto` coupler path produced a different physical netlist; that result was
discarded and was not used in solver conclusions.

All pump runs below omitted `--attenuation-db`. The summary therefore records
the configured frequency-dependent `loss_A10` model. At 7.9 GHz its value is
35.2751289969 dB. The circuit has 6,136 non-ground nodes, 2,508 JJ branches,
and four ports.

## Failure diagnosis

The first direct obstruction is a nonlinear globalization/accessibility failure,
not a linear algebra crash. A representative failed production report has:

- Newton termination by `stalled` or `line search failed`;
- nonzero residuals after the failed iterate, commonly `10^-2` to `10^-1` in
  the normalized coefficient residual;
- successful GMRES/PARDISO actions before the Newton stall;
- no evidence in these runs of a missing matrix, NaN, or transient DAE failure.

The failure mechanism is column- and history-dependent. Starting a 7.9 GHz
column at -23 dBm reaches a different continuation history than starting at
-26 dBm. The full low-power history is therefore part of the benchmark.

The recovery telemetry distinguishes the observed classes:

1. A coarse power target can be missed even though adaptive fixed-frequency
   continuation reaches it (`POWER_SUBSTEP`).
2. At 7.7 GHz, local pseudo-arclength reaches targets that power continuation
   cannot (`ARCLENGTH_RECOVERY`). This is consistent with the existing
   singularity evidence for snaking/multiple folds.
3. When all bounded recovery tiers fail at the first unresolved target, the
   row is `FAILED_NUMERICAL`; higher rows are only labelled
   `PAST_CONNECTED_BRANCH_BOUNDARY` after that candidate boundary. No row is
   called physical failure from HB failure alone.

There is no evidence in this 2c benchmark for a period-doubled representation
problem. The existing positive-odd 4WM basis and production residual gate are
retained. Period doubling remains relevant to the separate rf-SQUID 3WM work,
not to this 2c implementation.

## Measured recovery results

The common benchmark used 20 powers from -26 to -16 dBm, one frequency column,
the exact baseline-identical circuit, Schur HB, `real_coupled_fast`,
`--inproc-fail-fast`, no flat attenuation override, and no signal spectrum.
Every `PASS` below passed the normal pump and gain validation path.

| Column | Valid gain points | Routes | Runtime | Peak RSS |
| --- | ---: | --- | ---: | ---: |
| 7.9 GHz | 13/20 | 7 direct, 6 power-substep, 1 failed numerical, 6 past-boundary | 326.1 s | 1.42 GB |
| 7.7 GHz | 15/20 | 10 direct, 3 power-substep, 2 arclength, 1 failed numerical, 4 past-boundary | 377.6 s | 1.79 GB |

The 7.9 GHz first unresolved point is -19.1579 dBm; the last validated point
is -19.6842 dBm. The 7.7 GHz first unresolved point is -18.1053 dBm; the
recovery ladder reaches -18.6316 dBm. These results reproduce the archived G1
coverage counts while moving the ladder into the production column runner and
preserving route telemetry in `map_points.csv` and `map_summary.json`.

The archived full-map reference remains:

- baseline: 187/400 gain-valid points;
- existing enhanced run: 223/400 gain-valid points.

No new full 20x20 map is claimed by this document. The new evidence is two
real production columns. A full-map run should use the same explicit ladder and
classify its rows by the recorded route before comparing raw coverage.

## Alternatives inventory

| Method | Hypothesis | Scale / effort | Result | Decision |
| --- | --- | --- | --- | --- |
| Adaptive power substeps | A coarse target skips a reachable branch state. | O(number of microsteps) HB solves; scales with existing Schur backend. | Recovered all six 7.9 GHz G1 recovery points and three 7.7 GHz points. | Adopt as Tier 2. |
| Local pseudo-arclength | A bordered continuation path can cross a simple turning geometry. | One local bordered solve; higher setup cost but same sparse circuit actions. | Recovered two 7.7 GHz points; no 7.9 GHz recovery in this benchmark. | Adopt as bounded Tier 3. |
| Nearby-frequency detour | A neighbouring frequency provides an anchor around a narrow column obstruction. | Several same-size HB solves plus frequency substeps. | No recovery in these two columns; retained as a bounded fallback because archived G1 found a 7.7 GHz frequency recovery in a related run. | Retain as Tier 4, not default. |
| Reseed from zero | Newton may enter a different basin. | Cheap to state, expensive in repeated sparse factorisations. | Existing map history shows it improves some points but causes large runtime/memory overhead. | Keep legacy fallback; skip when ladder is enabled. |
| Deflation / branch discovery | Multiple roots may coexist at the same drive. | Potentially scalable only with a deflation-aware preconditioner; implementation is substantial. | Not added blindly: no independently validated second root was available for the benchmark, and deflation does not select the physical ramp state by itself. | Defer to the 7.0/snaking campaign. |
| Full pseudo-arclength branch tracing | Trace every branch and select a root. | Too expensive for a full map and can enter irrelevant multifold branches. | Existing project evidence shows this can follow complicated structures; local use is more effective. | Keep local only. |
| Shooting / transient orbit solve | Directly solve the physical periodic orbit when HB is inaccessible. | Time integration scales with circuit size and periods; unsuitable as the map engine. | Existing TD bridge is a physical oracle and branch-transfer tool, not a default 2c map solver. | Retain fallback-only. |
| Period-doubled or multifrequency HB | The missing state may not be PERIOD1. | Doubles or expands harmonic blocks; requires a matching Floquet gain formulation. | No 2c evidence in the tested columns; positive-odd 4WM residuals are already validated. | Do not implement for 2c without spectral/stability evidence. |
| Generic trust-region / dense least-squares | Stronger globalization could enlarge Newton basins. | Dense or generic sparse Jacobian costs are not credible for 6k–75k-node production maps. | Not prototyped on the real circuit; would replace working exact-JVP/preconditioner infrastructure without evidence. | Reject for production. |
| Krylov/preconditioner recycling | Consecutive HB states share tangent structure. | Favourable for large maps; requires invalidation rules near folds. | Existing solver already has reuse controls; the recovery ladder preserves that infrastructure. | Retain and measure separately. |

## Architecture recommendation

Use HB/AFT with exact JVP, Schur reduction, sparse preconditioning, and the
existing gain linearisation as the production engine. Add the explicit bounded
column ladder around it:

```text
direct HB
  -> adaptive power continuation
  -> local pseudo-arclength
  -> nearby-frequency anchor and return
  -> explicit numerical-hole or connected-boundary status
```

Use TD only for physical-state classification or branch transfer after this
ladder. Do not make TD or a dense global nonlinear solver the map engine.

At 2,400 JJs the measured 7.7 GHz peak RSS is approximately 1.8 GB for a
single in-process column. The circuit-size scaling is dominated by sparse
preconditioner factors and the number of recovery solves, not by the small
route telemetry. For larger devices, process-isolated frequency columns,
Schur reduction, compact output, factor reuse, and strict recovery deadlines
remain necessary.

## Decision tree

- If the failed target is recovered by Tier 2, accept only after the normal
  pump/gain validation and label `POWER_SUBSTEP`.
- If Tier 2 reaches a step floor and Tier 3 reaches the target, accept and label
  `ARCLENGTH_RECOVERY`.
- If a nearby-frequency anchor is validated and the return walk reaches the
  target, accept and label `FREQUENCY_RECOVERY`.
- If all bounded tiers fail at the first target, label `FAILED_NUMERICAL` and
  retain the last validated state. Do not call this a physical boundary.
- For higher powers after that target, label
  `PAST_CONNECTED_BRANCH_BOUNDARY` only as a connected-branch bookkeeping
  status; retain the possibility of another branch and do not count it as a
  physical exclusion without TD or stability evidence.
- If future stability/Floquet diagnostics show a subharmonic or multifrequency
  state, add the matching orbit representation before changing the gain solver.

## Remaining limitations

The integrated ladder does not yet recover all missing map points, and no new
full 20x20 map has been run. The 7.0 GHz snaking regime still needs a deliberate
branch-discovery method, likely deflation or a controlled global branch trace.
The current gain solver remains a PERIOD1 linearisation and must not be used
for a future PERIOD2 state without a Floquet/subharmonic extension. Physical
boundary status remains unresolved for rows labelled `FAILED_NUMERICAL` or
`PAST_CONNECTED_BRANCH_BOUNDARY` until the TD/stability evidence is collected.

## Decision

`CURRENT_ARCHITECTURE_NEEDS_TARGETED_CHANGES`.

The real-circuit experiments show that the HB/AFT, exact-JVP, Schur, and sparse
linear-algebra architecture is sound and scalable enough to retain. The missing
coverage is materially reduced by a targeted, bounded recovery ladder, while
the remaining 7.0/snaking and physical-boundary questions require separate
branch/state evidence rather than indiscriminate iteration increases.
