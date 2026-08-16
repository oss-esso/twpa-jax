# Bifurcation classification across five devices — gated plan

Status: proposed, 2026-08-13. Not started.

## Goal

Determine, for each of five devices, which bifurcation family terminates the
period-1 pumped branch, using time-domain evidence that is decidable without a
Floquet multiplier, and use that result — not a guess — to decide whether any
new harmonic-balance ansatz is warranted.

## Devices in scope

| id | source | nodes | junctions | f_plasma | f_pump | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `guarcello` | `docs/development/chaos_papers/guarcello_jtwpa_fdtd.py` | 990 cells | 990 | — | 7.0 GHz | Fig 2(a) reproduced |
| `jc_jtwpa` | `outputs/jc_doc_python_designs/jc_jtwpa` | 2560 | 2047 | 69.1 GHz | 7.12 GHz | kernel ported, unlaunched |
| `jc_fqjtwpa` | `outputs/jc_doc_python_designs/jc_fqjtwpa` | 2250 | 1999 | — | 7.90 GHz | kernel ported, unlaunched |
| `ipm_2c_fixed` | `designs/ipm_2c_fixed` | 6136 | 2508 | 37.6 GHz | 7.90 GHz | not ported |
| `rf_squid_2393_3wm` | `designs/rf_squid_2393_3wm.yaml` | ~9.6k | 2393 | ~59.8 GHz | ~7 GHz | not ported, not built |

`f_plasma` for `ipm_2c_fixed` is derived from `design_resolved.json`
(`Lj = 1.239e-10 H`, `Cj = 1.45e-13 F`). For `rf_squid_2393_3wm` it is derived
from `Ic = 0.93e-6 A` and `Cj = 20.0e-15 F` in the YAML; both are stated as
derived quantities, not measurements, and Phase C re-derives them from the
built matrices.

## Current state analysis

**The classification workflow is four steps; only the first three are
currently decidable on these circuits.**

1. Output spectrum — new components at `n f_p`, `n f_p / 2`, or incommensurate.
2. Waveform periodicity — `x(t + T_p)` against `x(t + 2 T_p)`.
3. Stroboscopic/Poincare sampling at `T_p`.
4. Floquet multiplier — `mu = -1`, `+1`, or `exp(+-i theta)`.

Step 4 is blocked on `ipm_2c_fixed` for a recorded physical reason, not a
solver defect: the circuit resolves `has_loss = False`
(`core/circuit.py:106` tests only `Im(C)`), its only dissipation being four
50-ohm port resistors in `G`. The 700-point Hill scan recorded in `CLAUDE.md`
returns `max |lambda| = 0.99999994` at **-35 dBm**, a power at which nothing is
marginal, and labels it `PERIOD_DOUBLING_CANDIDATE`. Exactly-real roots with
`|lambda| = 1` also appear, from port-decoupled internal modes. A new solver
inherits that spectrum unchanged.

**Existing assets.**

- `scripts/chaos/attractor_classify.py` — Poincare branches
  (`poincare_crossing_branches:29`), `sigma(V'_PS)` (`sigma_vprime_ps:57`),
  period clustering (`_period_clusters:97`), spectrum reduction
  (`fourier_map:321`), largest-Lyapunov map (`largest_lyapunov_map:292`),
  envelope decay (`envelope_decay:422`), and a four-verdict classifier
  (`classify_details:152`).
- `scripts/chaos/run_guarcello_jc_phase5.py` — the compiled Guarcello
  known-time-level banded kernel (`integrate_jc_banded_numba:199`), constant
  banded LU outside the loop (`_factor_banded_lu:139`), CSR products into
  reusable buffers (`_csr_matvec_into:162`), device derivation
  (`derive_device_spec:362`), and CLI knobs `--signal-current-a`,
  `--trace-out`, `--pump-power-values`.
- `src/twpa_solver/core/rcsj.py::stamp_rcsj_shunt:133` — RCSJ shunt stamping,
  `Bphi @ diag(1/Rj) @ Bphi.T` into `G`, with `inf` an exact bit-preserving
  control. Never run on `ipm_2c_fixed`.
- `src/twpa_solver/signal/stability.py`, `src/twpa_solver/stability/` — Hill
  and monodromy routes. `CLAUDE.md` records the TD monodromy route failing on
  2c (dimension 12271, ARPACK `0/2` eigenvectors converged, one-period closure
  error `3.922e-03` against a `1e-8` target) and recommends Hill.
- Dormant scaffolding, gated shut: `pump/floquet.py`,
  `pump/periodic_branch.py`, `signal/period_doubled.py`,
  `scripts/run_period_doubled_branch.py`.

**Three gaps.**

*Gap 1 — no pump-only run exists.* Every trace on disk injects a signal:
the Guarcello reproduction, the planned phase-5 scope, and the HB columns. An
injected tone forges the frequencies the classification tests for. A signal at
`f_s = f_p/2` forges `f_p/2` outright; a signal anywhere forges the whole
intermodulation comb, so a new line cannot be attributed to the bifurcation.
Pump-only is a separate sweep, not a re-read of existing traces.

*Gap 2 — the classifier cannot see pitchfork or fold.* `classify_details:152`
emits only `PERIOD_DOUBLING`, `NEIMARK_SACKER`, `CHAOS_NO_CLEAN_BIFURCATION`,
`NO_BIFURCATION_FOUND`. A pitchfork does not change the period, and its two
branches are related by `X_k -> -X_k`, which a power spectrum cannot see.
`fourier_map:321` returns `np.abs(...)` of `v - np.mean(v)`, discarding both
phase and DC — the two natural pitchfork order parameters. Separately,
`_period_clusters:97` uses a fixed relative tolerance `0.03`, which saturates
past period-4 because successive Feigenbaum splittings shrink by
`alpha ~ 2.503`.

There is a discriminant specific to these devices that needs neither phase nor
a multiplier. The unbiased 4WM pump basis is odd-only, `[1,3,...,19]`
(`pump/basis.py`), because the branch law is odd. A symmetry-broken branch
acquires **even pump harmonics and a DC offset**. So

    q_even = |X_2| / |X_1|,    q_dc = |X_0| / |X_1|

are scalar, magnitude-only order parameters that separate pitchfork from fold,
which `mu = +1` alone cannot. This is only meaningful for the odd-symmetric
devices; `rf_squid_2393_3wm` is a 3WM biased device whose symmetry is already
broken, so its baseline `q_even` is nonzero and the gate must be a *change*
from its own pump-off baseline, not an absolute threshold.

*Gap 3 — scheme accuracy.* Guarcello evaluates `sin phi` at the known time
level, making the nonlinear term explicit; this is what makes the matrix
constant and the run fast, and it is adequate for detecting a spectral line but
not for `|lambda|` at `1e-8`. Independently, `implicit_trapezoid` in the sparse
engine is A-stable but **not L-stable** (`R(z) -> -1`), applying no damping at
any frequency; `CLAUDE.md` already attributes the phantom
`UNRESOLVED_LONG_TRANSIENT` labels to this. `BDF`/`Radau` are available at
`scripts/h1_transient_branch_transfer.py:1557`.

## What we are NOT doing

- Not enabling any new HB ansatz. The `CLAUDE.md` gate stays shut through
  Phase F. `pump/floquet.py`, `pump/periodic_branch.py`,
  `signal/period_doubled.py` and `scripts/run_period_doubled_branch.py` remain
  dormant.
- Not building a torus/quasiperiodic basis from scratch under any Phase F
  outcome. If a complex pair crosses, the route is the auxiliary-generator
  closure on the existing `multitone` two-frequency lattice.
- Not modifying `docs/development/chaos_papers/guarcello_jtwpa_fdtd.py`. It is
  the paper reproduction and must stay bit-identical; Phase B re-checks Fig 2(a)
  against it.
- Not changing device parameters, the power convention, or the port update to
  improve agreement with anything.
- Not reporting a `|lambda|` crossing on a circuit with no physical
  dissipation.
- Not replacing the phase-5 gain/`r_j`-versus-HB comparison. That campaign is
  still owed and runs on the same kernel; this plan adds an axis to it.

## Prerequisites

- [ ] Phase-5 compiled kernel launched and producing rows. `outputs/chaos/phase5/`
      currently holds only `jc_jtwpa/pump_off/result.json` with
      `"status": "TIMEOUT"` and a stale `2912 steps/s` rate file, both from
      before the numba port. Phase B merges into that launch rather than
      queueing behind it.
- [ ] `numba` available (already a dependency, commit `cb5c3f5`).

---

## Phase A — extend the classifier

Pure post-processing. No solver work, no long runs. Testable offline on
synthetic traces and on the existing `outputs/chaos/phase2` traces.

### Changes required

**1. Preserve DC and phase**
**File**: `scripts/chaos/attractor_classify.py`
**Changes**: `fourier_map:321` gains a `keep_dc: bool = False` and returns a
`complex_amplitude` array alongside `amplitude`. The existing `amplitude` key
and its `v - np.mean(v)` behaviour stay as the default so every current caller
is unaffected.

**2. Symmetry order parameters**
**File**: `scripts/chaos/attractor_classify.py`
**Changes**: new `symmetry_order_parameters(t, v, pump_hz) -> dict` returning
`q_even = |X_2|/|X_1|`, `q_dc = |X_0|/|X_1|`, and the first six harmonic
magnitudes. Uses the exact-tone least-squares projection, not a raw FFT bin, so
it does not depend on the record length landing on a harmonic.

**3. Period-`n` test in the time domain**
**File**: `scripts/chaos/attractor_classify.py`
**Changes**: new `period_multiple(t, v, pump_hz, max_n=8) -> int` comparing
`||x(t + n T_p) - x(t)||` against `||x(t) - mean||` over the steady window,
returning the smallest `n` below tolerance. This is the direct test and does
not depend on Poincare clustering.

**4. Geometric cluster tolerance**
**File**: `scripts/chaos/attractor_classify.py`
**Changes**: `_period_clusters:97` takes `tolerance_decay: float = 2.503` so
the admitted gap shrinks with cluster depth. Default `1.0` reproduces present
behaviour exactly.

**5. Two new verdicts**
**File**: `scripts/chaos/attractor_classify.py`
**Changes**: add `PITCHFORK_CANDIDATE` and `FOLD_CANDIDATE`. Decision order,
evaluated against the point's own pump-off baseline:

| observation | verdict |
| --- | --- |
| `period_multiple >= 2` and half-integer lines present | `PERIOD_DOUBLING` |
| period unchanged, `q_even` or `q_dc` rises above baseline by the gate | `PITCHFORK_CANDIDATE` |
| period unchanged, symmetry unchanged, branch terminates | `FOLD_CANDIDATE` |
| lines at `n f_p +- m f_q`, `f_q/f_p` irrational | `NEIMARK_SACKER` |
| no finite lattice fits | `CHAOS_NO_CLEAN_BIFURCATION` |

Every verdict keeps the `_CANDIDATE` suffix where the evidence is spectral
only. Suffix is dropped only by Phase E.

### Success criteria

**Automated**
- `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\chaos_phaseA tests/test_attractor_classify.py`
- New tests, each verified by mutation:
  - **A-G1** synthetic period-2 signal is returned as `period_multiple == 2`;
    period-1 as `1`; period-4 as `4`.
  - **A-G2** synthetic odd-only waveform gives `q_even < 1e-12`; the same
    waveform plus a 3 percent second harmonic gives `q_even` within 1 percent
    of `0.03`.
  - **A-G3** `fourier_map` with default arguments returns arrays bit-identical
    to the pre-change implementation on a fixed trace.
  - **A-G4** `_period_clusters` with `tolerance_decay=1.0` returns bit-identical
    output to the pre-change implementation.
  - **A-G5** a synthetic period-8 cascade with Feigenbaum-scaled splittings is
    resolved to 8 clusters with decay enabled and saturates below 8 without it.

**Manual**
- Re-run the classifier over the existing `outputs/chaos/phase2` traces. The
  published Fig 2(a) verdicts must not change.

### Phase A execution record — 2026-08-13

Phase A is complete. `attractor_classify.py` now provides exact-tone
least-squares symmetry order parameters, direct time-domain period-multiple
detection, complex Fourier amplitudes with an opt-in DC component, and
Feigenbaum-scaled cluster tolerance. The legacy Fourier amplitude path and
the default cluster path remain bit-identical.

Verification:

- 16 focused classifier tests passed.
- Targeted mutants for the period test, legacy Fourier centring, and cluster
  tolerance each failed their corresponding gate before restoration.
- The validation figure is
  `outputs/chaos/phaseA/classifier_validation.png`.
- The offline saved-trace reduction is
  `outputs/chaos/phaseA/phase2_trace_classification.json` and covers 71
  `timeseries.npz` files: 5 `NO_BIFURCATION_FOUND`, 22 `PERIOD_DOUBLING`, and
  44 `CHAOS_NO_CLEAN_BIFURCATION`.
- The existing Fig. 2(a) directional reduction stores no verdict field, so
  there is no published verdict value to compare. Its source rows were not
  modified.

Phase B has not been started.

The committed-vs-working regression is recorded in
`outputs/chaos/phaseA/classifier_regression.json`. It separates the fixtures:

- Fig. 2(a) `fig2a_50ohm_mtls`: 51 points, 0 verdict changes and 0 cluster
  changes, reduced through `classify_details` from saved Poincare branches.
- Fig. 4 bias sweep: 69 traces, 5 verdict changes from the old classifier to
  `PERIOD_DOUBLING`; these are the intended finer-clustering changes and are
  explicitly retained as behaviour changes.
- Two single -54 dBm traces: 2 points, 0 changes.

The baseline source was constructed from
`git show HEAD:scripts/chaos/attractor_classify.py`; the comparison was not
treated as unavailable.

---

## Phase B — pump-only sweeps, Guarcello and JC devices

### Overview

Add a pump-only axis to the phase-5 runner and sweep three devices through
their transition. Signal-on runs are unaffected and still produce the gain and
`r_j` comparison the phase-5 campaign owes.

### Changes required

**1. Pump-only mode**
**File**: `scripts/chaos/run_guarcello_jc_phase5.py`
**Changes**: `--signal-current-a 0.0` must fully bypass
`_install_two_tone_source:421` rather than install a zero-amplitude tone, and
must record `signal_installed: false` in the result JSON. Gain fields are
written as `null`, not `-inf` or `NaN`, when no signal is present.

**2. Trace persistence**
**File**: `scripts/chaos/run_guarcello_jc_phase5.py`
**Changes**: each point writes `trace.npz` with `t` (s) and `v_out` (V) at the
existing `record_stride = 20`, roughly 2.9 MB per point. Classification is
post-processing and must never require a re-run.

**3. Sweep definition**
**File**: `scripts/chaos/run_guarcello_jc_phase5.py`
**Changes**: coarse pass of 20 points spanning the HB column plus six points
beyond the HB wall, then a fine pass of 10 points bracketing whichever coarse
interval first shows a verdict change. Fine-pass spacing is set by bisecting
that interval, not by a fixed dB step.

### Cost

600 pump periods per point. Measured post-JIT rates: `jc_jtwpa` 11,300 steps/s,
`jc_fqjtwpa` 12,700 steps/s. At `dt_norm = 0.01` that is 6087 steps per pump
period for `jc_jtwpa` (3.65e6 steps, 5.4 min/point) and comparable for
`jc_fqjtwpa`. 30 points x 3 devices ~ 8 h serial, ~2.5 h at three `nogil`
workers.

### Success criteria

**Automated**
- **B-G1** With `--signal-current-a 0.0`, the recorded spectrum contains no
  line within one FFT bin of `f_s` above the numerical floor. Direct proof the
  signal is absent, not merely small.
- **B-G2** `guarcello` pump-only reproduces the paper transition power to
  within the fine-pass bracket width, and the pump-on Fig 2(a) re-check against
  the untouched `guarcello_jtwpa_fdtd.py` is bit-identical to
  `outputs/chaos/phase2`.
- **B-G3** Every point writes `trace.npz`; the classifier runs end to end from
  disk with no solver import.
- **B-G4** Each device reports a verdict with its evidence: `period_multiple`,
  `q_even`, `q_dc`, half-integer line amplitudes **referred to the pump**, and
  the Poincare cluster count.

**Manual**
- Overlay spectra immediately below and immediately above the transition per
  device. Confirm by inspection whether new peaks land on `n f_p / 2`.

### Known risk

Pump-referred half-harmonic amplitude is the correct statistic, not
floor-referred. The earlier B2 gate failed on `guarcello` because the noise
floor itself rose 52 dB across the transition; pump-referred, `f_p/2` rises
66 dB and `3 f_p/2` rises 75 dB across the same step. Any new gate on line
amplitude must use the pump as the denominator.

---

## Phase C — port the banded kernel to `ipm_2c_fixed` and `rf_squid_2393_3wm`

### Overview

Guarcello's known-time-level scheme keeps the system matrix constant because
only `sin phi` is nonlinear and it is evaluated at the known level. Mutual
inductors, coupling capacitors and port resistors are all linear and stay in
the constant matrix, so the scheme ports unchanged. Only the ordering differs.

### Changes required

**1. Bandwidth-minimising reorder**
**File**: `scripts/chaos/run_guarcello_jc_phase5.py`
**Changes**: `derive_device_spec:362` measures the assembled pattern's
bandwidth in natural order and under reverse Cuthill-McKee, and selects the
smaller. Measured on the assembled pattern, never assumed:

| device | natural order | RCM |
| --- | ---: | ---: |
| `jc_jtwpa` | **2** | 4 |
| `ipm_2c_fixed` | 4578 | **5** |

`jc_jtwpa` must keep natural order and stay bit-identical to Phase B. 2c is not
a bare ladder — 758 mutual inductors, 760 coupling capacitors, two rails — and
its natural order is useless, but RCM=5 is still trivially banded. The
permutation is computed once outside the time loop.

**2. Device registration**
**File**: `scripts/chaos/run_guarcello_jc_phase5.py`
**Changes**: `--device` accepts `ipm_2c_fixed` and `rf_squid_2393_3wm`.
`derive_device_spec` must read a `designs/` directory, not only
`outputs/jc_doc_python_designs/`, and must handle a YAML source requiring a
build step.

**3. Per-cell arrays**
**Changes**: 2c carries per-cell `Ic` and the `rf_squid` line carries a
four-value `Cg_pattern`. Both must reach the kernel as arrays, matching the
existing FQJTWPA per-cell handling.

**4. Pump port**
**Changes**: `ipm_2c_fixed` drives the pump at **port 4** and scatters the
signal 1 -> 2. Using port 1 for both gives a `sqrt(2)` residual signature. This
has bitten the repo before and must be asserted, not assumed.

### Cost

2c is cheaper per pump period than `jc_jtwpa`, not worse: `f_plasma = 37.6 GHz`
against a 7.9 GHz pump gives 2986 steps per pump period at `dt_norm = 0.01`,
half `jc_jtwpa`'s 6087. That offsets the 2.4x node count. Estimated 5-7k
steps/s, roughly 5 min/point — an estimate, replaced by measurement at C-G3.

### Success criteria

**Automated**
- **C-G1** Assembled-pattern bandwidth is asserted at runtime under the
  selected ordering, and the selection is recorded in the result JSON.
- **C-G2** `jc_jtwpa` and `jc_fqjtwpa` results are bit-identical to Phase B
  after the reorder machinery lands. Regression against the ordering change.
- **C-G3** Measured steps/s per device recorded before any long run. No cost
  claim derived by scaling another device's rate.
- **C-G4** Linear check: with `Ic` scaled to zero the kernel reproduces
  `solve_linear_scattering` transmission at the pump frequency to 1e-9
  relative. Catches a mis-stamped mutual or coupler.
- **C-G5** Zero-drive check: from the static equilibrium with no pump, the
  state stays put to 1e-12 over 100 pump periods.

**Manual**
- Confirm the 2c pump split against the recorded live-circuit measurement
  (95.0 percent port 3, 4.9 percent port 2) before trusting any observable.

### Known risk

`designs/rf_squid_2393_3wm.yaml` has never been built or run here. If it does
not compile, or if its bandwidth is large under both orderings, drop it to
follow-up and finish the other four. Do not block the campaign on it.

### Phase C preflight execution record — 2026-08-14

The Phase C preflight was run without starting a pump sweep. All artifacts are
under `outputs/chaos/phaseC/`; no Phase B process or Phase B output was
modified.

The RF-SQUID YAML compiled under
`outputs/chaos/phaseC/build/rf_squid_2393_3wm/`. The built circuit has 7,180
non-ground nodes and 2,393 Josephson branches. Its plasma frequency derived
from the built per-cell matrices is 59.824 GHz. A 33-point linear `S21` scan
from 4 to 12 GHz selected 4.75 GHz as the pump frequency, where the measured
linear transmission is -0.0015 dB. The YAML has no DC-flux term. Its `Cg`
profile was expanded from `[C1, C2, C1, C3]` with counts `[6, 6, 6, 6]` and
persisted as a 2,393-element array.

The 2c plasma frequency derived from its persisted `Ic` and `Cj` arrays is
37.549 GHz. Its measured ordering is natural bandwidth 4,578 and RCM
bandwidth 5; RCM was selected. The pump port assertion is port 4, the signal
path is port 1 to port 2, and the pump-through linear check uses port 4 to
port 3. The 2c `Ic`, `Cj`, and `Cg` arrays each have 2,508 entries.

| Gate | Measurement | Verdict |
| --- | --- | --- |
| C-G1 | `ipm_2c_fixed`: selected RCM bandwidth 5; `rf_squid_2393_3wm`: selected natural bandwidth 2. Both selected bandwidths were re-measured after permutation and asserted. | PASS |
| C-G2 | `jc_jtwpa`: natural 2, RCM 4, identity permutation, unchanged matrix digests. `jc_fqjtwpa`: natural 2, RCM 2, identity permutation, unchanged matrix digests. | PASS |
| C-G3 | Measured rate: 2c 641.97 steps/s; RF-SQUID 5,211.81 steps/s, both at `dt_norm=0.01` and 200 steps. | FAIL for the 2c cost target; measured, not extrapolated |
| C-G4 | 2c reference `|S|=0.973782`; the time-domain kernel became non-finite over 100 pump periods. RF-SQUID kernel `|S|=0.948660` versus linear solve `|S|=0.999827`, relative error 5.12%. Required relative error is `1e-9`. | FAIL for both devices |
| C-G5 | Zero-drive integration for 100 pump periods: final-state maximum is exactly 0 for both devices. | PASS |

C-G4 is a measured failure. It was not closed by changing a device
parameter, power convention, port update, or timestep. The 2c result indicates
that the explicit known-time-level path is unstable at this preflight setting;
the RF-SQUID result shows a finite but non-converged transfer. The C-G4
artifact records both values and the failure.

Every generated point row now records the applied current, on-chip power,
instrument power, `legacy_traveling_wave` convention, A10 pump-line model, and
its attenuation. The legacy convention is `P = I^2 Z0 / 2`; the unresolved
source/outgoing-wave convention discrepancy remains the known 6.0206 dB
offset. The pump label uses `pump_line_loss_model()` because the 2c pump enters
port 4. The fixed Phase C grids are registered in the shared overnight driver
but were not launched: 2c is 0.300–1.200 `I/I_bound` in 0.025 steps (37
points), and the RF-SQUID control grid is 0.100–1.000 in 0.025 utilization
labels. Phase C plots remain deferred until those traces exist.

---

## Phase D — timestep convergence

### Overview

The known-level `sin` is explicit and therefore conditionally stable, and the
plasma-to-pump ratio varies roughly 2x across the device set. `dt_norm = 0.01`
is inherited from the paper's device and is not automatically valid elsewhere.

### Success criteria

**Automated**
- **D-G1** At one point immediately below and one immediately above each
  device's transition, halving `dt_norm` to `0.005` leaves the verdict
  unchanged and moves `q_even`, `q_dc` and the pump-referred half-harmonic
  amplitudes by less than 5 percent.
- **D-G2** The transition power itself moves by less than the fine-pass bracket
  width under the halving.

**Manual**
- If D-G1 fails on any device, that device's verdict is reported as
  `NOT_TIMESTEP_CONVERGED` and is excluded from the Phase F decision. It does
  not invalidate the others.

---

## Phase E — gated: physical dissipation, then Hill multipliers

**Gate to enter: at least one device passes Phase D with a verdict other than
`NO_BIFURCATION_FOUND`.**

### Overview

Steps 1 to 3 of the workflow are complete at this point. Step 4 requires
physical dissipation first, because `|lambda|` is not decidable on a circuit
whose only loss is four port resistors.

### Changes required

**1. RCSJ damping sweep**
**File**: new `scripts/chaos/run_rcsj_damping_sweep.py`
**Changes**: apply `stamp_rcsj_shunt` at a ladder of `resistance_ratio` values
with `inf` as the exact untouched control, and re-run the Phase B/C sweep at
the transition for each. This is the damping sweep `CLAUDE.md` has recorded as
owed and never run on 2c.

**2. Hill multipliers**
**File**: existing `scripts/floquet_stability_sweep.py`
**Changes**: none expected. Route through Hill, not TD monodromy — the
monodromy route is recorded as failing on 2c.

### Success criteria

- **E-G1** A pumped point known to be stable (`-35 dBm` on 2c) returns
  `1 - |lambda|` that grows monotonically with damping and is clearly separated
  from zero at the working damping. If it does not, `|lambda|` remains
  undecidable and Phase F proceeds on Phase B/C evidence alone.
- **E-G2** The Hill scan uses at least 175 points per zone; the mode comb on 2c
  near 7.9 GHz is ~241.7 MHz, so a full zone needs about 700.
- **E-G3** The multiplier is branch-tracked. `src/twpa_solver/stability/tracking.py`
  exists but is wired only to the monodromy scan; a secant that falls onto a
  power-independent neutral root is a tracking failure, not a result.
- **E-G4** `mu` at the transition agrees with the Phase B/C verdict:
  `-1` for period doubling, `+1` for pitchfork or fold,
  `exp(+-i theta)` for Neimark-Sacker. Disagreement is reported, not resolved
  by preferring one side.

---

## Phase F — gated: HB ansatz decision

**Gate to enter: E-G4 answered, or E-G1 failed with Phase D verdicts
consistent across at least two devices.**

No implementation in this phase. It produces a decision and its justification.

| Phase B-E outcome | indicated route |
| --- | --- |
| period doubling, `mu -> -1` | period-2 basis. Fundamental-halving of the existing single-tone basis; `signal/period_doubled.py` and `pump/periodic_branch.py` are already scaffolded. Not a new solver. |
| pitchfork, `mu -> +1`, symmetry broken | no new ansatz. Existing basis with even harmonics admitted. |
| fold, `mu -> +1`, symmetry intact | no new ansatz. This is the PALC fold already characterised at `I_bound ~= 1.163e-05 A`. |
| Neimark-Sacker, `mu -> exp(+-i theta)` | auxiliary-generator closure on the existing `multitone` two-frequency lattice: two extra real unknowns `(A_a, omega_a)`, two extra real equations `Y_AG = 0` in an outer loop. **Not** a torus basis from scratch. |
| chaos, no clean lattice | no ansatz can help. Report and stop. |

Three of five outcomes require no new solver. The `CLAUDE.md` gate is opened,
if at all, by naming which outcome was measured and which of its four
preconditions — multiplier crossing resolved, timestep-converged,
sideband-converged, corroborated by an L-stable TD run — each phase discharged.

---

## Testing strategy

### Project maturity level

Established production. The solver is under test; `scripts/chaos/` is campaign
code with partial coverage.

### Unit tests

- Phase A gates A-G1 to A-G5, each verified by mutation. Coverage target 90
  percent on the new classifier functions, since a silent classifier defect
  produces a plausible wrong verdict rather than a failure.
- Bit-identity regressions (A-G3, A-G4, C-G2) are the load-bearing tests. Every
  change in this plan is additive, and a behaviour change in a default path is
  a defect.
- Edge cases: zero-length trace, constant trace, trace shorter than one pump
  period, `pump_hz = 0`, non-uniform `t` spacing.

### Integration tests

- C-G4 and C-G5 are the device-port integration tests: linear limit against
  `solve_linear_scattering`, and zero-drive equilibrium persistence.
- B-G2 is the end-to-end reproduction regression against the untouched paper
  module.

### Manual verification

- Spectra overlaid below and above each transition, inspected for
  half-integer placement.
- Poincare scatter per device across the sweep, inspected for one point, two
  alternating points, a closed curve, or a cloud.

---

## Rollback

Every phase is additive and defaults to present behaviour.

- Phase A: new functions and two new verdict strings. Reverting is deleting
  them; A-G3 and A-G4 guarantee existing callers are untouched meanwhile.
- Phase B: new CLI flag and a new output file. Signal-on runs are unchanged.
- Phase C: reorder selection defaults to natural order, which C-G2 pins
  bit-identical for the JC devices. New devices are new `--device` values.
- Phase D and E: read-only campaigns writing under `outputs/chaos/`.
- Phase F: a document.

No production solver path is modified by Phases A to D. `stamp_rcsj_shunt` in
Phase E returns the original `CircuitMatrices` object without arithmetic at
`inf`, so the control arm is exactly the present circuit.

## Phase B classifier correction record — 2026-08-13

The first pump-only reduction exposed three post-processing defects. The
period test now uses fractional-delay resampling at the exact pump-period
shift, asserts a minimum of 4 samples per pump period, and returns 0 when no
tested integer period matches. A cap value is never interpreted as period
doubling. The pitchfork gate uses a measured low-power symmetry floor and a
factor of 20 above that floor. The persisted Phase B traces have no unstrided
record, so the smaller-stride alias check is recorded as unavailable without
re-integration.

The Guarcello disk re-reduction is sorted by pump power. Compact one-cluster
points below the transition are `NO_BIFURCATION_FOUND`, while broad clouds are
`CHAOS_NO_CLEAN_BIFURCATION`. The pump-off floor was not persisted
by the initial Phase B run; the selected floor therefore comes from the first
five low-power traces and is recorded with that limitation in
`outputs/chaos/phaseB/guarcello/reduction_metadata.json`.

### Phase B spectral correction record — 2026-08-13

The recorded Guarcello spectra exposed a disagreement between the direct
period test and the bifurcation spectrum. Residuals for the saved transition
traces are in
`outputs/chaos/phaseB/guarcello/transition_residual_diagnostics.json`.
The period test is not used as the sole decision variable: a pump-referred
half-integer line exceeding the stable low-power spectral floor by 18 dB is
recorded explicitly, with a disagreement flag when `period_multiple` does not
match. `PERIOD_DOUBLING` requires both spectral evidence and a matching
time-domain period. Spectral evidence without closure is labelled
`PERIOD_DOUBLING_ONSET`. A Poincare cloud with more than the maximum tested
period multiple (`max_n=8`) overrides a clean period-doubling label and remains
`CHAOS_NO_CLEAN_BIFURCATION`; this replaces the arbitrary threshold of 16.

The half-integer line is measured as the maximum in a fixed +/-80 MHz window
around `f_p/2` and `3 f_p/2`. This fixed window is not fitted to the result.
The corrected eight-point Guarcello gap scan from -53.95 to -53.40 dBm found
spectral-only onset rows at -53.636 and -53.557 dBm. Both have
`period_multiple=0`, so neither is a clean period-doubled orbit. The
neighbouring -53.714 dBm row is already a 201-cluster cloud, and -53.479 dBm
is a 295-cluster cloud. The current evidence therefore supports direct
transition to a non-periodic response with a half-harmonic precursor, not a
resolved clean period-2 window. Each row now persists the pump-referred
half-harmonic level, residual at n=1, winning residual, and winning n.
The smaller-record-stride alias test remains pending; it is now implemented as
a short measurement rather than being treated as unavailable.

### Phase C structural diagnosis and corrected preflight â 2026-08-14

The overnight Phase B campaign was not touched or relaunched by this work.
No Phase C campaign was started. The following measurements supersede the
initial C-G3--C-G5 values above.

The explicit stability table was measured from the generalized pencil

`K + Bphi diag(Ic/phi0) Bphi.T, C + G`

with `eigsh` at `dt_norm = 0.01`:

| Device | `lambda_max` (s^-2) | Explicit limit (s) | `dt / limit` | Result |
| --- | ---: | ---: | ---: | --- |
| `jc_jtwpa` | `1.559875e23` | `5.063900e-12` | `0.004556` | finite bound; stable |
| `jc_fqjtwpa` | `1.311243e23` | `5.523171e-12` | `0.003964` | finite bound; stable |
| `ipm_2c_fixed` | `8.003625e26` | `7.069466e-14` | `0.599561` | finite bound; stable but close |
| `rf_squid_2393_3wm` | `1.933005e25` | `4.548974e-13` | `0.058483` | finite dynamic-subspace bound; stable |

The RF-SQUID `C + G` matrix is singular because it has 2,393 uncoupled
zero-mass algebraic rows. Its finite bound was therefore computed on the
remaining 4,787 dynamic rows; the full generalized pencil is explicitly
recorded as singular rather than being regularized.

The constant matrix now includes `K` by default for every device. The old
explicit-RHS treatment remains available only through an explicit regression
override. The RF-SQUID geometric-inductor flag is derived from the built
element records (`rf_squid_lpar`), not from a device-name literal. Its branch
law is represented as `K` containing `Lpar^-1` plus the Josephson
`Bphi Ic sin(phi/phi0)` branch law. The built YAML and element list contain no
DC-flux or DC-bias term, so the 3WM name does not imply an implemented bias.

The corrected preflight was then run without a campaign:

| Gate | Corrected measurement | Verdict |
| --- | --- | --- |
| C-G3 | Warmed 200-step rate: 2c `5,174.50` steps/s; RF-SQUID `5,751.39` steps/s. A separate primitive profile measured the 2c band solve at `116.18` microseconds/step and the RF solve at `89.92` microseconds/step. | Measured; no campaign launched |
| C-G4 | 2c is finite, `|S|=0.966364` versus `0.973782`, relative error `0.7618%`. RF-SQUID remains `|S|=0.948660` versus `0.999827`, relative error `5.1177%`. | FAIL for both |
| C-G5 | Zero-drive final-state maximum after 100 pump periods is exactly `0` for both devices. | PASS |

The 2c non-finite failure is removed by putting `K` in the constant implicit
matrix. The remaining C-G4 errors are not closed by tolerance changes. A
discrete phasor check gives `|S|=0.966326` for 2c and `0.948798` for the
RF-SQUID, matching the transient values and identifying the residual as the
known-time-level discretization response relative to the continuous
`solve_linear_scattering` reference. The RF value did not move under the
geometric-inductor metadata correction because the algebraic-row condition
had already placed `K` implicitly in the earlier RF preflight; the built
`Lpar` branch is now verified and recorded rather than silently omitted.

The explicit-versus-implicit short regressions are in
`outputs/chaos/phaseC/C-G2_stiffness_paths.json`. At 2,000 steps, the retained
explicit path and the new implicit path differ in final-state relative norm
by `6.39e-6` for `jc_jtwpa` and `3.02e-6` for `jc_fqjtwpa`; the short output
voltage is below the propagation floor at that duration, so its pointwise
relative difference is not a meaningful physical comparison. The ordering
and matrix-identity C-G2 regression remains PASS. No full Phase B trace was
rewritten.

### Phase C E-validation — 2026-08-14

The preceding statement that the C-G4 residual was identified by a matching
discrete phasor response is withdrawn. The required timestep measurements are
recorded in `outputs/chaos/phaseC/E_validation.json`.

E1 tested the retained explicit-K override for `ipm_2c_fixed`. The eigenvalue
table gives `dt/limit = 0.599561` at `dt_norm=0.01`; the corresponding ratios
are `0.299780` and `0.149890` at `dt_norm=0.005` and `0.0025`.

| `dt_norm` | `dt/limit` | explicit-K result |
| ---: | ---: | --- |
| 0.0100 | 0.599561 | non-finite |
| 0.0050 | 0.299780 | non-finite |
| 0.0025 | 0.149890 | non-finite |

Therefore the non-finite result is not a marginal crossing of the measured
second-order generalized-pencil bound. The bound is not the correct stability
instrument for this explicit known-level implementation, or the explicit path
has another defect. The implicit-K path is required for Phase C diagnostics;
the explicit path remains a regression-only path and is not used for campaigns.

E2 tested the corrected implicit-K C-G4 against the continuous linear solve.
The relative errors were:

| Device | `dt_norm=0.01` | `dt_norm=0.005` | `dt_norm=0.0025` |
| --- | ---: | ---: | ---: |
| `ipm_2c_fixed` | 0.761825% | 0.381696% | 0.191526% |
| `rf_squid_2393_3wm` | 5.117655% | 2.600084% | 1.316287% |

Both errors approximately halve per timestep halving, not quarter. They are
not flat, but they also do not show the expected second-order
`(omega dt)^2` scaling. The RF-SQUID discrepancy therefore must not be called
discrete-time dispersion on the basis of the earlier phasor comparison. The
1e-9 C-G4 gate remains FAIL for both devices at all three resolutions.

E3 applied an in-memory tenfold mutation to all 2,393 built `rf_squid_lpar`
elements. The C-G4 relative error changed from `5.117654953%` to
`5.117661127%`, an absolute change of `6.17e-8`; this is numerically unchanged
at the scale of the 5.12% discrepancy. The mutation artifact is
`outputs/chaos/phaseC/E3_lpar_mutation.json`.

The built RF-SQUID topology contains, per cell, `Lw`, `Lm`, `Lpar`, a
Josephson inductive branch, and two grounding capacitors. The `Lpar` and
Josephson elements form a series branch from the wire node to the right node,
parallel to `Lm`; it is not a standalone JJ-to-ground shunt. The mass matrix is
`7180 x 7180` with 2,393 zero-mass rows and 4,787 dynamic rows. Thus `Lpar` is
present in the built linear stamp, but the mutation demonstrates that changing
it does not remove the C-G4 residual. The residual is therefore not explained
by the earlier claim that the Lpar flag was simply omitted; the remaining
structural discrepancy is unresolved.

E4: `rf_squid_2393_3wm` has no explicit DC flux or bias term in the YAML or
built element list. Its branch law is consequently odd and the design does
not implement three-wave mixing despite its filename. Its `q_even` baseline
must be interpreted against an unbiased RF-SQUID/4WM baseline; no pitchfork or
3WM conclusion is valid without a separately measured bias.

### Phase C RF-SQUID bias correction — 2026-08-14

The preceding E4 conclusion is withdrawn. The bias is a runtime DC current
source at port 1, applied together with the AC pump; it is not a topology
element and therefore is not represented in the YAML element list. The
comparison record specifies `dc=true`, `threewavemixing=true`, and
`fourwavemixing=true` for the two port-1 source modes.

The RF-SQUID build will use the repository's self-consistent convention,
`phi_dc = phi_ext - beta_L sin(phi_dc)`, with external flux fraction `0.33`.
The lightweight metadata check gives:

| Quantity | Value |
| --- | ---: |
| `beta_L` | `0.1655940748` |
| `phi_ext` | `2.0734511514 rad` |
| `phi_dc` | `1.9177228112 rad` (`0.305215 Phi0`) |
| `Idc` at port 1 | `11.64479812 uA` |

The pump frequency is fixed to `12.080 GHz`, matching the existing map. The
previous `4.75 GHz` unbiased-band selection is invalid for this operating
state. The RF-SQUID campaign grid is now direct on-chip current: 40
log-spaced points from `2.5e-6 A` to `1.5e-5 A`, plus exact points at
`6.3246e-6 A`, `7.0963e-6 A`, and `1.0024e-5 A` (43 total points).

The DC source, bias convention, `beta_L`, `phi_ext`, and `phi_dc` are emitted
in every RF-SQUID result row. C-G3, C-G4, C-G5, and the 2,400-period settling
check remain pending because the 2c settling process was active; no heavy
compute was started in this correction.
