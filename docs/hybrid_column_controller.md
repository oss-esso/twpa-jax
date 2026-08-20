# Hybrid HB -> TD -> HB column controller

The controller is implemented in `src/twpa_solver/hybrid_column.py` and the
production adapter/CLI is `scripts/run_hybrid_column.py`.

## State machine

`HB_FAST` calls the production `InProcessEngine.solve_point`. Failed targets
enter bounded `HB_RECOVERY` using the existing power-substep, PALC, and
frequency-substep helpers. Only a failed target with a previous working HB
state may enter `TD_BRIDGE`.

The validated H1 transient classifies the held state. A PERIOD_1 result must
produce a Fourier-projected seed and then pass back through production
`solve_point` before the column continues. Persistent non-PERIOD_1 or running
phase stops the column as `PHYSICAL_BOUNDARY_FOUND`; unresolved and numerical
TD results remain distinct.

## Storage and budgets

The controller stores compact records and writes one JSON summary. H1 writes
periodic restart checkpoints and only the projected TD seed for a successful
PERIOD_1 transfer. `HybridColumnBudget` defaults are two TD bridges
(`max_td_bridges`), three future boundary refinements
(`max_boundary_refinements`), and **200 TD periods for the whole column**
(`max_td_periods`) — that budget is accumulated across bridges, not granted
per bridge. The validation CLI is intentionally separate from the overnight
map and does not launch it automatically.

## Verification

The controller unit tests cover direct HB progression, TD-to-HB restart,
persistent physical-boundary stopping, and unresolved-budget handling. The
production smoke command is:

```powershell
python scripts/run_hybrid_column.py --freq-ghz 7.7 `
  --circuit-dir outputs/ipm_2c_passive --outdir outputs/hybrid_77
```

`--circuit-dir` still defaults to `designs/ipm_2c_fixed`, which is a gitignored
local artifact and is absent from a fresh checkout. Pass a directory you built
yourself — see [`workflows.md`](workflows.md) §2 — rather than relying on that
default.

The full 7.0/7.7 validation campaign remains a separate run; no overnight map
is started by this implementation.
