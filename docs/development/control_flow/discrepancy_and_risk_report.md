# Discrepancy and Risk Report

Revision `8ac9016f475e38a1d65911dd5d515b73e586b3d8`. Companion to
`pump_and_signal_solver_control_flow.md` (§13–§14). Each item is verified against the
called code, not inferred from flag names.

## A. Parser accepts a value that is not implemented / fails late

| # | Item | Detail | File / line |
| --- | --- | --- | --- |
| A1 | `--pump-mode-policy` has **no `argparse choices=`** | Any string is accepted; validation is deferred to `resolve_pump_basis`, which raises `ValueError` for unknown policies. | run_gain_map.py:2394; basis.py:207 |
| A2 | `positive_phasor_explicit` unreachable from `run_gain_map` | `_build_problem` hardcodes `explicit_modes=None`; selecting this policy raises `ValueError` ("requires --pump-modes"). Only reachable via the subprocess exp08 CLI. | run_gain_map.py:551; basis.py:217 |
| A3 | `real_coupled_fast` + `full` backend | `assemble_real_coupled_fast` exists only on `SchurReducedProblem`; the combination raises `NotImplementedError`. | solver.py:242 |

## B. Documented value not accepted / naming mismatch

| # | Item | Detail |
| --- | --- | --- |
| B1 | Status enum names in the task spec are not literal | The code uses `VALID_CONVERGED`/`FAIL` (pump), `VALID_SOLVED`/`CHECK` (gain), `PASS`/`ERROR`/`SKIP_PAST_FOLD` (cell). `GMRES_FAILED`, `LINE_SEARCH_FAILED`, `MAX_NEWTON`, `DEADLINE`, `STALL`, `NONFINITE`, `UNSUPPORTED_BACKEND`, `INVALID_CONFIGURATION`, `MISSING_SEED`, `PARTIAL` are **not** present — those conditions surface as free-text `StepReport.failure_reason` phrases or raised exceptions. |
| B2 | `--loss-model` is not a `run_gain_map` flag | The gain solve hardcodes `current_complex_c`; the 7-model `dynamic_block` switch is reachable only through library calls / other scripts. |

## C. Flag silently ignored in some modes

| # | Item | Detail | File / line |
| --- | --- | --- | --- |
| C1 | `--linear-seed-maxiter` parsed but unused in-process | The in-process seed is adaptive continuation from zeros; `build_linear_phasor_seed` (which would use it) is only on the subprocess exp08 path. | seeds.py:46 |
| C2 | `--skip-baselines` ignored under `--signal-backend direct` | Consulted only in the `schur` branch; `solve_gain_one` always computes baselines. | run_gain_map.py:1092 |
| C3 | subprocess-only flags ignored in-process | `--pump-timeout-s`, `--gain-timeout-s`, `--python-executable` only affect `run_point`; the default in-process executor ignores them. Conversely most `--inproc-*` flags do nothing under `--executor subprocess`. | run_gain_map.py:298 |
| C4 | A user-set `--frequency-chunk-size` is discarded by non-`column` traversals | `main` forces it to 0 (single process) with a printed notice, not an error. | run_gain_map.py:2761 |

## D. Two flags / paths that behave inconsistently

| # | Item | Detail | File / line |
| --- | --- | --- | --- |
| D1 | Two fold short-circuits with different gates | Legacy column pass requires `verified_fold AND consec_fail>=patience`; the traversal orchestrator trips on `col_fail[j]>=patience` alone (no `verified_fold`). Traversal skip is more aggressive → can cull a solvable region. | run_gain_map.py:1410 vs 1785 |
| D2 | `CHECK` gain collapses to `ERROR` | A high-residual-but-finite gain writes `gain_db` yet the cell is `ERROR`; there is no distinct visible "suspect gain" status in the map. | run_gain_map.py:781/798 |

## E. A fallback changes the requested backend without a clear record

| # | Item | Detail | File / line |
| --- | --- | --- | --- |
| E1 | `--allow-superlu-fallback` silent downgrade | Debug flag lets `real_coupled_fast` fall back PARDISO→SuperLU; the chosen backend is not written to `map_points.csv`, only surfaced if `--log-factor-backend` is also on. Under strict default a PARDISO failure raises instead. | fast_coupled.py:354; run_gain_map.py:2749 |
| E2 | `GmresDeadlineExceeded` absorbed by `tangent_predictor` | Subclasses `RuntimeError` so the broad `except` in `tangent_predictor` swallows a mid-tangent deadline and silently degrades to the copy predictor. | solver.py:31/830 |

## F. Approximation vs exact preconditioner (not a bug, a caveat)

| # | Item | Detail | File / line |
| --- | --- | --- | --- |
| F1 | Schur `real_coupled`/`spectral_coupled` use `D_nn` as the linear part | They drop the dense Schur correction (`D_ne D_ee^-1 D_en`); exactness is recovered only through the matrix-free operator GMRES corrects against. The conjugate `k+q` term is kept in all real-coupled variants. | schur_operators.py:16-20,260 |

## G. Defaults differing between related paths

| # | Item | Detail |
| --- | --- | --- |
| G1 | Executor default skew | `--executor inprocess` (default) uses `--inproc-*` numerics; the subprocess path uses `experiments/exp08_*`/`exp09_*` defaults which are independent. Switching executor without re-passing settings changes numerics. |
| G2 | `basis.py` module default vs CLI default | `load_pump_basis_from_solution` assumes `pump_mode_policy` default `dense_real` (basis.py:289) while the `run_gain_map` CLI default is `positive_odd_jc`. Consistent in practice because the policy is read back from saved metadata, but the raw defaults differ. |

## H. Out-of-scope confusable

| # | Item | Detail |
| --- | --- | --- |
| H1 | `scripts/run_pump_hb.py` is a **different** pump solver | Imports the `twpa.*` JAX distributed-HB package, not `twpa_solver`. Shares no solver code with the gain map. A reader must not conflate it with the traced pump HB. |
