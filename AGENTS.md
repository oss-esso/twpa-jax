# Project working notes

## Language standard

Use ATS technical English in all conversations and documentation. Write in
plain, precise, and unambiguous technical language. Avoid idioms, marketing
language, unnecessary theory, and unexplained abbreviations. Use terminology
consistently in user-facing documentation, implementation notes, and status
updates.

- Dependency freedom: free Python packages may be installed when they materially
  improve the implementation or validation. The preinstalled environment is not
  a hard dependency limit; verify compatibility and record the package used.

## Branch and artifact policy

- `main` is the clean, installable package branch.
- `dev` is the development and cross-machine synchronization branch; it may
  contain experiments, generated artifacts, and incomplete work.
- Keep ignore rules compatible, but do not rely on `.gitignore` to separate
  tracked files. Ignored files are not synchronized; tracked files merge.
- Do not merge `dev` wholesale into `main`. Promote production work with
  focused commits and `git cherry-pick`, or use a manually curated merge.
- Keep generated outputs under ignored `outputs/` or another disposable path;
  source design YAML belongs under `designs/`.

## Stability, transient, and high-power branch work

Full evidence and plan:
`docs/development/high_power_79ghz_period1_floquet_plan.md`. Condensed
knowledge is in `CLAUDE.md`, section "7.9 GHz PERIOD1 branch and Floquet
stability (measured 2026-08-11)". Read both before touching this area.

Rules, all from measurements taken 2026-08-11:

- A harmonic-balance non-convergence is not a physical boundary. Before
  reporting one, rerun with a power step at or below `0.25` dB from the last
  converged checkpoint. At 7.9 GHz on `designs/ipm_2c_fixed` a `1.05` dB step
  failed at `-23.421053` dBm where `0.18`-`0.25` dB steps converge by plain
  Newton.
- Never quote diagnostics from a non-converged Newton iterate as physical
  state. Failure rows in `hb_up_to_failure.csv` carry last-iterate values, and
  they overstated junction utilization by `0.86` against a true `0.60`.
- Do not use the recurrence metric as the primary transient stability label.
  Its `4.25e-4` floor is constant over 11 dB of pump power and scales with both
  timestep and ramp length. Use the post-ramp `max_abs_phi` envelope slope;
  record `d1` and its trend only as secondary diagnostics.
- Do not use `implicit_trapezoid` for relaxation or stability questions. It is
  A-stable but not L-stable, so it applies no numerical damping. Use `Radau` or
  `BDF`, both already available via
  `scripts/h1_transient_branch_transfer.py --method`.
- Do not report a Floquet multiplier crossing on a circuit with
  `has_loss = False`. `designs/ipm_2c_fixed` has four port resistors and
  nothing else, so `1 - |lambda|` is between `1e-8` and `1e-11` at every pump
  power tested, including provably stable ones. Add physical dissipation before
  asking the question.
- Prefer the Hill route (`src/twpa_solver/signal/stability.py`) over the
  time-domain monodromy route (`src/twpa_solver/stability/`) on this device.
  ARPACK on the `12271`-state monodromy returned `0/2` converged eigenvectors
  after `651` matvecs; the clustered spectrum and the `3.9e-3` one-period
  closure error are both fatal to it at reachable timesteps.
- Resolve the measured mode comb. The 2c structure has a `~241.7 MHz` comb:
  approximately 700 points are recommended for a full Hill-zone scan, 200 is
  thin but not aliased, and the Phase 1 density guard rejects scans below
  approximately 175 points.
- Do not enable a PERIOD2, period-N, or two-frequency harmonic-balance ansatz.
  The scaffolding exists and is deliberately dormant. Enabling it requires a
  tracked multiplier crossing that is resolved, sideband-converged,
  timestep-converged, and corroborated by an L-stable transient run.
- The Hill CLI uses ASCII-only summary output and writes its JSON artifact
  before printing the summary, so a Windows console encoding failure cannot
  destroy a completed sweep.
- There is an external reference for the high-power boundary. The Themis
  `14.18.08` cube measures the device's collapse at 51 pump frequencies from
  `5.980` to `7.997` GHz, and its `Response` array is a pump-on/pump-off ratio,
  so it compares directly against `gain_vs_off_db` with no power-convention or
  line-loss assumption. Use it before asserting anything about where the model's
  boundary should be. The `17.03.10` cube covers only `7.043` to `7.373` GHz.
- Do not report a Themis collapse power at a frequency where the device did not
  collapse within the instrument's `-29.08` to `-19.03` dBm range. Five of the
  51 frequencies are censored that way and must be flagged, not fitted.
- Never compare a model pump frequency or pump power against a measured one at
  face value. The device carries unknown offsets in both axes, and the collapse
  boundary is a comb whose envelope slope is about `-20` dB/GHz, so `10` MHz of
  frequency offset is worth `0.2` dB of power and the measurement's own `40` MHz
  grid step is already `+/- 0.4` dB. A single-frequency "agreement" is a
  coincidence, not evidence.
- Any `(df, dP)` calibration fit must report its degrees of freedom and refuse
  to emit an estimate below `DOF = 8`. Two unknowns against two model points is
  vacuous no matter how small the residual. Fit the offset-invariant shape
  statistics first: comb period and envelope depth.
- Do not compare the model's single-tone `gain_vs_off_db` at `f_p - 500` MHz
  against the measurement's peak over the signal span. The measured bias between
  those two observables reaches `11.90` dB, grows with pump power, and is
  non-monotone: at `7.876` GHz the single-tone trace falls while the peak still
  rises. Run the model with a signal spectrum and reduce it with the
  measurement's own rule instead. No downstream statistic repairs this.
- To compare two gain trajectories, reference both to their own boundary
  (`u = P - P_boundary`) so the power offset never enters, then compare the
  threshold and slope from a `1/sqrt(G_linear)` versus pump-amplitude fit. Use
  `dG/dP` only as a secondary diagnostic. Do not use dynamic time warping,
  Procrustes, min-max normalization, or a bare correlation coefficient.

## FDTD timestep selection is per-device (measured 2026-08-14)

`dt_norm = 0.01` in `scripts/chaos/run_guarcello_jc_phase5.py` is the Guarcello
paper's prescription, 628 steps per **Josephson plasma** period. It is not a
universal budget. Verify it per device with the linear-limit check
(`_measure_linear_limit`), which sets `Ic = 0` and compares the kernel against
the continuous linear solve. That check costs about 60 s and has an exact
reference, so it depends on no other campaign result. Run it before spending any
campaign time on a new device.

Measured at each device's pump frequency:

| device | kernel \|S\| | exact \|S\| | rel. error | verdict |
| --- | ---: | ---: | ---: | --- |
| `ipm_2c_fixed` | 0.966364 | 0.973782 | 0.0076 | passes at `dt_norm = 0.01` |
| `rf_squid_2393_3wm` | 0.298453 | 0.478674 | 0.3765 | needs a finer step |
| `jc_jtwpa` | 0.0 | 0.0 | — | check is degenerate |
| `jc_fqjtwpa` | 0.0 | 0.0 | — | check is degenerate |

`rf_squid_2393_3wm` converges toward the exact value as the step shrinks, so the
kernel is correct and only the step is too coarse. Measured at 200 pump periods:

| `dt_norm` | steps/period | kernel \|S\| | rel. error | runtime |
| ---: | ---: | ---: | ---: | ---: |
| 0.0100 | 3112 | 0.298453 | 0.3765 | 117.1 s |
| 0.0050 | 6223 | 0.409857 | 0.1438 | 246.6 s |
| 0.0025 | 12447 | 0.480768 | **0.0044** | 599.1 s |

**Use `dt_norm = 0.0025` for `rf_squid_2393_3wm`**: 0.44 percent at 4x the
default cost. The error falls 2.6x then 32x, much faster than first order, so do
not extrapolate the required step from two points -- measure the third.
Duration is not the cause of the coarse-step error: at `dt_norm = 0.01` the
result is flat from 200 through 800 pump periods.

The check is **degenerate** on `jc_jtwpa` and `jc_fqjtwpa`. Setting `Ic = 0`
removes their only inductive path, so both sides return exactly 0.0 and
`relative_error` is `None`. That is neither a pass nor a failure, and those two
devices consequently have no independent linear validation. Covering them needs
a variant that retains a finite linear inductance instead of zeroing `Ic`.

### Optional: why the required step is device-dependent

Unresolved, and not needed for any current result. The obvious explanation does
not survive its own arithmetic. `rf_squid_2393_3wm` places an `Lm` in parallel
with each junction, giving an `Lm`-`Cj` mode at 147.013 GHz against the
59.824 GHz Josephson plasma frequency the timestep derives from, a ratio of
2.46. That would motivate deriving the step from the fastest linear mode rather
than from the plasma frequency. But the Guarcello device itself has
`Lg = 120 pH` and `Cj = 200 fF`, an `Lg`-`Cj` mode near 1027 GHz against a
plasma frequency near 28 GHz, a ratio of about 37, and its own integrator is
accurate at the same `dt_norm`. The mode ratio alone therefore does not predict
the required step.

Candidate mechanisms, none tested: the paper's formulation solves junction
phases through a tridiagonal system while this kernel solves node fluxes through
a banded one, so the two discretize different operators; the accuracy limit may
be set by the port terminations rather than by any internal mode; or the
governing scale may be the fastest mode measured against the **pump** frequency
rather than against the plasma frequency. Resolving this would replace a
per-device measurement with a predictive rule. Until then, measure.

## Canonical IPM coupler-count designs

- The canonical presentation sources are `designs/ipm_2c.yaml`,
  `designs/ipm_3c.yaml`, `designs/ipm_7c.yaml`, and `designs/ipm_20c.yaml`.
- `ipm_2c.yaml` is the renamed line-scoped 2c source. All four use the same
  IPM-line topology family and none sets `coupler_freq_hz` locally, so every
  coupler resolves the technology preset's value (`ipm_default` is 8 GHz).
  Only the historical `ipm_2c_coupler_edit.yaml` still pins 10 GHz.
- Couplers name the two lines they join by endpoint port number (`port in
  signal`, `port in pump`, `port out signal`, `port out pump`).
- For coupler-count comparisons and presentation runs, use only these four
  canonical files. Do not substitute `*_coupler_edit.yaml`,
  `*_no_coupler.yaml`, or unrelated legacy examples.
- Build and map artifacts belong below `outputs/`. For batched workflows,
  pass multiple design paths and one output root; each design receives a
  subdirectory named after its design directory or YAML stem.
- Repeated IPM sections use the declarative `repeat` group: `ipm_7c` and
  `ipm_20c` do. `ipm_2c` and `ipm_3c` write their sections out flat, because a
  `repeat` of count 1 or 2 costs more lines than it saves. Omitted `rows`,
  `cells`, `between`, and trailing inter-coupler fields resolve from the
  selected technology preset.
- `input_ports` and `output_ports` carry their own launch CPW from the preset.
  Do not add a separate `cpw` block beside a port block; that emits two CPWs
  in series.
- Dielectric loss and fabrication scatter are technology-backed design
  parameters. Global values belong under `parameters`; IPM/JTL blocks may also
  set local scatter fields and `tan_delta` for their shunt capacitors.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
