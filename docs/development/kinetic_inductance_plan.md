# Implementation Plan: lumped kinetic inductance in the general solver

Status: implemented, with compression/saturation validation intentionally skipped
to avoid the repository's high-memory test paths. Written 2026-08-05; implementation
completed 2026-08-05.

Validation completed for the implemented non-saturation scope:

- branch-law math, routing, persistence, DC bias, KIMPA construction/passive
  readout, gain, half-pump basis, loss separation, fitting, and port environment;
- 398 repository tests passed with compression/saturation/P1dB/PSAT tests and
  slow-marked tests excluded;
- manual passive sweeps completed for all three KIMPA fixtures and a reduced
  DC-biased gain run completed with waveform/continuation artifacts.

## Goal

A lumped kinetic-inductance branch element (`Lk`, `Ic`, material preset) and a
KIMPA one-port reflection-amplifier netlist run through the **existing** HB /
AFT / Floquet / multitone solver, with no change to Newton, GMRES, JVP,
preconditioners, continuation, or the Schur backends.

## Current state analysis

The solver is already law-pluggable in the way this work needs.

| Fact | Location |
| --- | --- |
| `BranchLaw` protocol: `current(flux)`, `tangent(flux)`, `metadata`; arrays shaped `(nt, nbranch)` | `src/twpa_solver/core/nonlinear.py:57` |
| `make_branch_law(circuit)` returns `circuit.branch_law` or a Josephson default | `core/nonlinear.py:154` |
| `EffectiveSnailBranchLaw` — precedent for a law carrying an equilibrium-flux shift | `core/nonlinear.py:83` |
| Residual/JVP go through `branch.current` / `branch.gamma` only | `pump/problem.py:207,232`; `multitone/problem.py:150,165` |
| `SchurMultiToneProblem` inherits `full.branch` — no separate wiring | `multitone/schur.py:45` |
| Floquet tangent is `K_hat_l = Bphi diag(gamma_hat_l) Bphi.T` | `signal/gamma.py:163` |
| Conversion matrix `A[m][q] = K_hat_{m-q} + delta_mq D(omega_s + m omega_p)` | `signal/floquet.py:130` |
| One-port S11 already exists (`s -= 1` when `source_port == out_port`) | `core/linear.py:100` |
| `solve_gain_one` already emits **both** `gain_db = 20log|S11|` and `gain_vs_off_db` | `signal/floquet.py:386,388` |
| Nonlinear branches never stamp `K`; ordinary inductors stamp `1/L` | `builders/le_gal_2025.py:112`; `builders/ipm.py:1051` |
| Ports are Norton: shunt `1/Z0` into `G` plus a port marker | `builders/ipm.py:840` |
| `dc_branch_flux` is a first-class field; residual forms `i(psi + psi_dc) - i(psi_dc)` | `pump/problem.py:117,210`; `multitone/problem.py:41,152` |

### Blockers found

**Six call sites hardcode the Josephson law and bypass `circuit.branch_law`.**

| Site | Hardcodes |
| --- | --- |
| `pump/problem.py:74` `JosephsonBranchArray` | `Ic sin`, `Ic cos / phi0` |
| `scripts/run_compression.py:388` | builds `JosephsonBranchArray(circuit.Ic, circuit.phi0)` |
| `scripts/run_gain_map.py:619` | same |
| `scripts/multitone_convergence_study.py:202` | same |
| `signal/gamma.py:89` | `(Ic/phi0) cos(psi/phi0)` |
| `signal/passive.py:40` and `scripts/run_gain_map.py:1214` | `gamma_off = Ic / phi0` |

`scripts/run_le_gal_2025_hb.py:51` is the one driver that already passes
`circuit.branch_law`; it is the pattern to copy.

**Circuit persistence is hardcoded to one law.** `load_circuit` only
reconstructs `effective_snail` (`core/circuit.py:176`); `save_circuit` only
serializes that law's arrays (`core/circuit.py:232`).

**The tone lattice cannot express `omega_p = 2 omega_s`.** `ToneIndex(h,q)`
means `h*omega_p + q*delta` with integer `h`; `MultiToneBasis` *requires*
`(1,0),(1,-1),(1,1)` (`multitone/basis.py:107`) and `pump_tone` is the hardcoded
property `(1,0)` (`:137`), consumed by source injection
(`run_compression.py:519`), Manley-Rowe scope (`observables.py:344`), spatial
profiles (`:505`), and stability (`stability.py:91`).

**No driver exposes DC bias.** The plumbing exists on every problem class, but
`run_compression.py` has no flag and `run_gain_map.py:1205` passes
`dc_branch_flux=None`. There is no DC solver — `pump/seeds.py:13` only *loads* a
`dc_solution.npz`.

**`observables._port_current_coefficients` (`:29`) does not subtract the DC
current**, so port waves are wrong under bias.

### Constitutive-law correction

The source plan's §4 flux law and tangent are mutually consistent but do not
follow from `L_k(I) = L_k0 [1 + (I/I*2)^2]`. Three candidate conventions:

| Convention | `Phi(I)` | `dPhi/dI` |
| --- | --- | --- |
| `Phi = integral_0^I L(I') dI'` | `Lk[I + I^3/(3 I2^2)]` | `Lk[1 + (I/I2)^2]` |
| `Phi = L(I) I` | `Lk[I + I^3/I2^2]` | `Lk[1 + 3 (I/I2)^2]` |
| source plan (1, 3, 2) | as written | `Lk[1 + a^2 + 6aI/I2^2 + 6I^2/I2^2]` |

Only the first makes the **measured** quantity — the small-signal inductance at
bias, extracted from the resonance shift — equal `dPhi/dI` at `I_dc`. Adopted,
and extended through the quartic so one law covers both DC tuning and dynamic
mixing:

```
Phi_k(I)   = Lk [ I + I^3 / (3 I2^2) + I^5 / (5 I4^4) ]
dPhi_k/dI  = Lk [ 1 + (I/I2)^2 + (I/I4)^4 ]
```

The tangent is exactly the paper's DC-tuning law including the quartic. Appendix
D drops the quartic to reach a closed-form gain Hamiltonian; a numerical HB
solver has no such constraint, so nothing here is "static only".

`Phi_k` is strictly monotone (all coefficients positive), so the flux-to-current
inversion has a unique real root. Cubic-only case has an exact stable closed
form: with `u = I/I2` and `phi_tilde = phi/(Lk I2)`, `u^3 + 3u = 3 phi_tilde`,
and substituting `u = 2 sinh(theta)` gives `2 sinh(3 theta) = 3 phi_tilde`, so

```
I = I2 * 2 * sinh( arcsinh(1.5 * phi / (Lk * I2)) / 3 )
```

No branch cuts, no cancellation, fully vectorized. With the quintic term active
this is the seed for a damped Newton polish.

## What we are NOT doing

- No change to Newton, GMRES, JVP, preconditioners, continuation, arclength,
  Schur partitioning, or the factor backends.
- No new solver backend, no new preconditioner.
- Not re-running or re-deriving any published JJ result. Phase 2 is a pure
  refactor and must be bit-identical.
- **No circulator, bias tee, or diplexer as separate network elements — ever,
  not "deferred".** Their ideal behaviour is already a no-op on a one-port: the
  circulator separates incident from reflected (which
  `port_s_from_unit_current_response` does), the bias tee delivers DC (which
  `dc_branch_flux` does), and the diplexer merges pump and signal (both already
  inject at port 1). Any *non-ideal* behaviour belongs in the `Z_env` port
  environment of Phase 10, not in a component model.

Phases 9, 9b and 10 cover work that an earlier draft of this plan listed as
out of scope. The `hung_2025_alt` deferral was withdrawn on a factual
correction — see Phase 9b.

## Prerequisites

- [ ] Saturation work closed, so the JJ regression surface is stable.
- [ ] `python -c "import twpa_solver, sys; print(twpa_solver.__file__)"` resolves
      inside this repo, not `D:\tmp\finalclone` (editable install can shadow it).
- [ ] Full suite green first, to own the baseline:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_ki_base --run-slow`

---

## Phase 1: kinetic-inductance branch law

### Overview

Pure math and pure new code. No existing file changes, no wiring. Everything is
unit-testable in isolation.

### Changes required

#### 1. Model presets

**File**: `src/twpa_solver/core/kinetic.py` (new)
**Changes**: Named material presets mapping `Ic` to the nonlinear scale
currents, with the paper values kept visible as literal ratios.

```python
KI_MODEL_PRESETS: dict[str, dict[str, float]] = {
    "hung_2025": {
        # Hung 2025: Ic = 1.15 mA, I*2 = 3.25 mA, I*4 = 1.70 mA.
        "istar2_over_ic": 3.25e-3 / 1.15e-3,   # 2.8260869565217392
        "istar4_over_ic": 1.70e-3 / 1.15e-3,   # 1.4782608695652173
    },
}
```

`resolve_ki_model(name, critical_current_a) -> tuple[np.ndarray, np.ndarray]`
returns `(istar2, istar4)` per branch. Unknown name raises listing valid names.

#### 2. The law

**File**: `src/twpa_solver/core/kinetic.py`
**Changes**: `KineticInductorBranchLaw`, frozen dataclass, satisfying the
`BranchLaw` protocol plus `gamma` (which `pump/problem.py` and
`multitone/problem.py` both call).

```python
@dataclass(frozen=True)
class KineticInductorBranchLaw:
    kinetic_inductance_h: np.ndarray   # Lk, per branch
    critical_current_a: np.ndarray     # Ic, validity boundary only
    istar2_a: np.ndarray               # quadratic scale
    istar4_a: np.ndarray | None        # quartic scale; None disables the term
    model: str = "hung_2025"
    newton_max_iter: int = 20
    newton_rtol: float = 1e-14
```

Methods:

- `flux(current)` — `Lk[I + I^3/(3 I2^2) + I^5/(5 I4^4)]`. Forward direction,
  used by tests, by the DC helper, and by the inversion residual.
- `differential_inductance(current)` — `Lk[1 + (I/I2)^2 + (I/I4)^4]`.
- `current(flux)` — the inversion. `arcsinh` seed above; if `istar4_a is None`
  return it directly, otherwise damped Newton on `flux(I) - phi` (monotone,
  positive derivative) to `newton_rtol`, raising `RuntimeError` with the worst
  residual if it does not converge in `newton_max_iter`.
- `tangent(flux)` — `1.0 / differential_inductance(current(flux))`.
- `gamma = tangent` (alias, matching `JosephsonBranchLaw.gamma`).
- `metadata` — `{"type": "kinetic_inductor", "model": ..., "quartic": bool}`.

Note the inversion of roles versus the Josephson law: JJ has current explicit in
flux, KI has flux explicit in current. The protocol is unchanged because the
inversion is closed-form (or a guaranteed-convergent polish).

#### 3. Composite law for mixed JJ + KI circuits

**File**: `src/twpa_solver/core/nonlinear.py`
**Changes**: Append `CompositeBranchLaw`.

```python
@dataclass(frozen=True)
class CompositeBranchLaw:
    laws: tuple[Any, ...]
    columns: tuple[np.ndarray, ...]   # index arrays into the branch axis
```

`current`/`tangent`/`gamma` allocate the full `(nt, nbranch)` output and write
each sub-law's slice. Validation in `__post_init__`: columns must partition
`range(nbranch)` exactly, no overlap, no gap. `metadata` returns
`{"type": "composite", "parts": [...]}`.

#### 4. Validity metric

**File**: `src/twpa_solver/core/kinetic.py`
**Changes**: `kinetic_validity(law, branch_flux_time, dc_branch_flux)` returning
per-branch `max_abs_current_a`, `current_over_ic`, and a status string:

```
SUPERCONDUCTING       max|I_dc + i(t)| <  Ic
THRESHOLD_CROSSED     max|I_dc + i(t)| >= Ic
```

The law itself must **not** diverge at `Ic` — the polynomial stays finite and
the threshold is reported, never enforced. The paper observes an abrupt loss of
amplification at breakdown, not a smooth nonlinear fold.

### Success criteria

**Automated**: `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_ki tests/test_kinetic_branch.py`

New `tests/test_kinetic_branch.py`:

- `test_flux_current_round_trip` — `flux(current(phi)) == phi` to 1e-14 relative
  over `phi` spanning `+/- Phi(2 Ic)`, quartic on and off.
- `test_cubic_inversion_matches_closed_form` — with `istar4_a=None`, `current`
  agrees with a `numpy.roots` solve per sample to 1e-13.
- `test_tangent_matches_finite_difference` — `tangent` vs central differences of
  `current`, relative error < 1e-7.
- `test_differential_inductance_reproduces_paper_dc_tuning` — at
  `I_dc in {0, 300, 530, 600} uA` with the `hung_2025` preset,
  `differential_inductance(I_dc)/Lk` equals
  `1 + (I/3.25mA)^2 + (I/1.70mA)^4` exactly.
- `test_zero_bias_tangent_is_even_in_current` — at `I_dc = 0` the tangent has no
  term linear in `i`, so a symmetric drive produces no even harmonic in gamma.
- `test_dc_bias_breaks_inversion_symmetry` — at `I_dc = 550 uA` the FFT of
  `tangent` over a sinusoidal flux has a non-zero fundamental (`ell = +/-1`)
  component. This is the 3WM enabling condition.
- `test_threshold_status_transitions` — `SUPERCONDUCTING` below `Ic`,
  `THRESHOLD_CROSSED` at and above.
- `test_composite_law_dispatches_per_branch` — a 2-JJ + 2-KI composite equals
  each pure law on its own columns.
- `test_composite_rejects_overlapping_or_incomplete_columns`.
- `test_unknown_preset_lists_valid_names`.

**Manual**: none — this phase is self-contained math.

---

## Phase 2: route every call site through the branch law

### Overview

Delete the six hardcoded Josephson evaluations. **Pure refactor: every existing
JJ result must be bit-identical**, because `JosephsonBranchLaw.tangent` is
literally `Ic cos(psi/phi0)/phi0` and at zero flux is `Ic/phi0`.

### Changes required

#### 1. Floquet tangent

**File**: `src/twpa_solver/signal/gamma.py`
**Changes**: Replace line 89 with
`gamma_t = make_branch_law(circuit).tangent(psi_t)`. The function already
accepts and applies `dc_branch_flux`, so nothing else changes.

#### 2. Pump-off stiffness

**File**: `src/twpa_solver/signal/passive.py`
**Changes**: Replace `gamma_off = circuit.Ic / circuit.phi0` (line 40) with the
branch-law tangent at the DC operating point. Add an optional
`dc_branch_flux: np.ndarray | None = None` parameter to `passive_s_matrix` and
evaluate `make_branch_law(circuit).tangent(dc[None, :])[0]`. Default `None`
means zeros, preserving current behavior exactly.

#### 3. Gain-map driver

**File**: `scripts/run_gain_map.py`
**Changes**: line 619 — `self.branch = make_branch_law(self.ipm08)`; line 1214 —
branch-law tangent as above. Add `--dc-current-a` and `--dc-solution` so
line 1205's `dc_branch_flux=None` becomes the resolved value.

#### 4. Compression driver

**File**: `scripts/run_compression.py`
**Changes**: line 388 — `branch=make_branch_law(circuit)`. Add `--dc-current-a`
and thread the resulting `dc_branch_flux` into `FullPumpProblem` and
`FullMultiToneProblem`.

#### 5. Convergence study

**File**: `scripts/multitone_convergence_study.py`
**Changes**: line 202 — `branch=make_branch_law(circuit)`.

#### 6. Port currents under DC bias

**File**: `src/twpa_solver/multitone/observables.py`
**Changes**: `_port_current_coefficients` (line 29) takes an optional
`dc_branch_flux` and forms `law.current(phase + dc) - law.current(dc)`, matching
the residual convention in `multitone/problem.py:152`. Default zeros is a no-op
for every existing call.

#### 7. Deprecate the duplicate

**File**: `src/twpa_solver/pump/problem.py`
**Changes**: Leave `JosephsonBranchArray` (line 74) exported for the
`experiments/` provenance scripts, but add a docstring line stating production
drivers must use `make_branch_law`. Do not delete — `experiments/exp08` and
friends construct it directly and are frozen provenance.

### Success criteria

**Automated**:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_ki_p2 --run-slow
```

Plus a new `tests/test_branch_law_routing.py`:

- `test_gamma_hat_matches_legacy_josephson_formula` — `compute_gamma_hat` on a
  JJ circuit equals the old `(Ic/phi0) cos(psi/phi0)` expression bit-for-bit.
- `test_passive_gamma_off_matches_ic_over_phi0` — same for `passive_s_matrix`.
- `test_port_currents_unchanged_at_zero_dc_flux`.
- Each of the three assertions must be shown **failing** against a deliberately
  mutated tangent before the phase is called done.

**Manual**: rerun one small published gain map and diff `gain_db` against the
stored artifact — must agree to all printed digits, not "close".

---

## Phase 2b: separate pump-line and signal-line insertion loss

### Overview

Two measured lines now exist: `docs/development/loss_A10.csv` for the **pump**
line and `docs/development/loss_B1.csv` for the **signal** line. Both become
defaults. This phase is deliberately kept out of Phase 2 because it **changes
numbers**, and Phase 2's gate is bit-identity — the two must not share a commit.

`run_compression.py` currently carries **one** scalar `attenuation_db`,
resolved from the A10 model at the *pump* frequency (`:291`), and then uses that
same scalar to refer *signal* power to dBm (`:834`, `:1012`, `:1119`) and to
convert a signal dBm back to an on-chip current (`:948`, `:1006`). With one line
that was merely imprecise; with two measured lines it is wrong.

### Fitted coefficients

Both CSVs fit `att_dB(f) = c + a sqrt(f) + b f` over 0-20 GHz, 1001 points.

| Model | Line | `c` | `a` | `b` | fit RMS | fit max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `loss_A10` | pump | 27.3882157727 | 0.4579029666 | 0.8354288817 | 0.374 dB | 1.810 dB |
| `loss_B1` | signal | 50.0000036748 | 3.2999976712 | 0.1400003844 | 1.1e-5 dB | 1.4e-4 dB |

**B1 is an analytic curve, not raw measurement data.** The fit recovers exactly
`c = 50.0`, `a = 3.3`, `b = 0.14` to 1e-5 dB, so refitting it is a
self-consistency check, not a measurement fit. Store the exact round numbers as
the frozen constants and record the provenance distinction in the docstring —
A10's 0.374 dB scatter is real instrument noise, B1's is float round-off.

Evaluated where this work needs them:

| Frequency | Role | A10 (pump line) | B1 (signal line) |
| ---: | --- | ---: | ---: |
| 8.00 GHz | reference | 35.367 dB | 60.454 dB |
| 8.47 GHz | KIMPA signal | 35.797 dB | **60.790 dB** |
| 16.94 GHz | KIMPA pump | **43.425 dB** | 65.954 dB |

Both KIMPA frequencies sit inside the measured 0-20 GHz span; neither
extrapolates.

### Changes required

#### 1. Two named models

**File**: `src/twpa_solver/loss.py`
**Changes**: Add `LOSS_B1_C_DB = 50.0`, `LOSS_B1_A_DB = 3.3`,
`LOSS_B1_B_DB = 0.14`, plus:

```python
def pump_loss_model() -> InsertionLossModel:      # loss_A10
def signal_loss_model() -> InsertionLossModel:    # loss_B1
```

`default_loss_model()` stays as-is and stays an alias of `pump_loss_model()`.
Every existing caller uses it for the pump (`run_compression.py:291`,
`run_gain_map.py:64`), so the alias is correct and nothing breaks. Mark it in
the docstring as "pump line; prefer the explicit name".

The module docstring must be corrected while here: it says the model "replaces
the flat 35 dB attenuation" and cites only A10, which is now half the story.

#### 2. Split the scalar in the compression driver

**File**: `scripts/run_compression.py`
**Changes**: `_resolve_attenuation` returns a pair. `--attenuation-db` keeps
forcing **both** to one value for reproducing old runs; new
`--pump-attenuation-db` and `--signal-attenuation-db` force one each.

| Call site | Was | Becomes |
| --- | --- | --- |
| `:291` pump dBm -> current | A10 at `pump_freq_ghz` | `pump_loss_model()` at `pump_freq_ghz` |
| `:834`, `:1012`, `:1119` signal current -> dBm | same pump scalar | `signal_loss_model()` at `signal_ghz` |
| `:948`, `:1006` signal dBm -> current | same pump scalar | signal model at `signal_ghz` |

The CSV gains `pump_attenuation_db` and `signal_attenuation_db` columns, and the
single `attenuation_db` column is retained carrying the pump value so existing
readers do not silently pick up the signal number.

#### 3. Gain-map driver

**File**: `scripts/run_gain_map.py`
**Changes**: `attenuation_db_for` (used at `:79` and by the debug scripts) is a
pump-side conversion and keeps the A10 model. Add `--signal-attenuation-db` for
the signal spectrum sweep, defaulting to `signal_loss_model()` at the signal
frequency.

### The consequence, stated plainly

Adopting B1 for signal referral shifts **every input-referred P1dB** by
`B1(f_s) - A10(f_p)`. At the KIMPA operating point that is **+17.365 dB**; for a
JJ device with pump and signal both near 8 GHz it is **+25.087 dB**.

This is a correction, not a regression — signal power arrives down the signal
line, so it must be referred through the signal line's loss. But it means every
previously published input-referred P1dB in this repository was computed under a
single-line convention. Those numbers are not wrong *as measured*; they are
stated in a convention that no longer holds. They must be either restated or
explicitly labelled old-convention. Do not quietly overwrite them.

The device-referred (on-chip) quantities are untouched.

### Success criteria

**Automated**: extend `tests/test_loss_model.py`

- `test_b1_refit_recovers_exact_analytic_coefficients` — refitting
  `loss_B1.csv` returns `50.0 / 3.3 / 0.14` to 1e-4, documenting that B1 is
  synthetic.
- `test_a10_refit_matches_frozen_constants` — existing check, unchanged.
- `test_pump_and_signal_models_are_distinct` — `signal_loss_model()` exceeds
  `pump_loss_model()` by 25.087 dB at 8.0 GHz and 22.529 dB at 8.47 GHz.
- `test_explicit_attenuation_forces_both_lines` — `--attenuation-db` reproduces
  the old single-scalar behaviour exactly, so an old run can be replayed.
- `test_signal_power_uses_signal_line` — a compression run at
  `f_p = 16.94 GHz`, `f_s = 8.47 GHz` reports `signal_power_dbm` exactly
  17.365 dB above the same run under the old single-scalar path.

**Manual**: re-emit one stored compression summary with `--attenuation-db` set
to the old pump scalar and confirm the CSV is byte-identical to the artifact.

---

## Phase 3: persist the kinetic branch law

### Overview

`--circuit-dir` workflows need KI circuits to survive a save/load round trip.

### Changes required

#### 1. Serialization schema

**File**: `src/twpa_solver/core/circuit.py`
**Changes**: Flat per-branch arrays rather than a nested schema, so composite
circuits need no recursion.

Added to `ipm_arrays.npz`:

| Array | dtype | Meaning |
| --- | --- | --- |
| `branch_law_kind` | int8 | `0` josephson, `1` kinetic |
| `ki_lk` | float64 | `Lk`, NaN on non-KI branches |
| `ki_istar2` | float64 | NaN on non-KI |
| `ki_istar4` | float64 | NaN when quartic disabled or non-KI |

`save_circuit` writes these whenever `circuit.branch_law` is a
`KineticInductorBranchLaw` or a `CompositeBranchLaw` containing one, and records
`metadata["branch_law"] = law.metadata`.

`load_circuit` dispatches on `branch_law_kind`: absent -> current behavior
(unchanged, so every stored design still loads); all `0` -> `None`; all `1` ->
`KineticInductorBranchLaw`; mixed -> `CompositeBranchLaw`. The existing
`effective_snail` path (line 176) is left untouched.

### Success criteria

**Automated**: `tests/test_kinetic_persistence.py`

- `test_kinetic_circuit_round_trips` — save then load, compare `Lk`, `Ic`,
  `istar2`, `istar4`, and `law.current` on a random flux array to 0.0 exact.
- `test_mixed_jj_ki_circuit_round_trips` — composite columns preserved.
- `test_legacy_design_without_kind_array_still_loads` — load
  `designs/ipm_2c_fixed` and assert `branch_law is None`.
- `test_quartic_disabled_survives_round_trip` — `istar4_a is None` stays `None`,
  not NaN-valued arrays.

**Manual**: `load_circuit(designs/ipm_2c_fixed)` still reports 16312 elements.

---

## Phase 4: half-pump multitone lattice

### Overview

Make the pump tone configurable so `omega_p = 2 omega_s` fits the integer
lattice. With `omega_0 = omega_p / 2` as the lattice fundamental: signal
`(1,-1)`, idler `(1,1)`, physical pump `(2,0)`, pump harmonics `(4,0)`, `(6,0)`.
The existing required-tone triple `(1,0),(1,-1),(1,1)` is still satisfied, so
this is additive.

The **small-signal Floquet path needs no change at all** — `sideband_list` uses
true integer multiples of the physical `omega_p`, and the 3WM idler sits at
`m = -1` (`omega_s - omega_p ~= -omega_s`, conjugated).

### Changes required

#### 1. Configurable pump tone

**File**: `src/twpa_solver/multitone/basis.py`
**Changes**:

- New field `pump_tone_index: ToneIndex = ToneIndex(1, 0)` on `MultiToneBasis`.
- `required` (line 107) becomes
  `{ToneIndex(1,0), ToneIndex(1,-1), ToneIndex(1,1), self.pump_tone_index}` —
  identical to today when the default is used.
- `pump_tone` property (line 137) returns `self.pump_tone_index`.
- `signal_tone` / `idler_tone` unchanged.
- `to_metadata` gains `"pump_tone": {"h": ..., "q": ...}`.

#### 2. Half-pump basis constructor

**File**: `src/twpa_solver/multitone/basis.py`
**Changes**: `build_half_pump_basis(pump_harmonics, sidebands, omega_p_physical,
delta, omega_max=None)`. Sets `omega_0 = omega_p_physical / 2`, pump modes
`[2*h for h in pump_harmonics]`, `pump_tone_index = ToneIndex(2, 0)`, and
delegates to `build_sideband_matched_basis`.

Docstring must state plainly: **`sidebands` counts steps of `omega_0`, i.e.
half-pump quanta, not physical pump quanta.** The tone `(1,0) = omega_0` is
retained because the basis invariant requires it; it is weakly excited and
expected to solve near zero.

Pump harmonics are **even only** — the pump solve alone produces harmonics of
`omega_p`, which are `h = 2, 4, 6` in `omega_0` units. Odd `h` is populated only
once a signal is present.

#### 3. Pump promotion scale

**File**: `src/twpa_solver/multitone/seed.py`
**Changes**: `promote_pump_solution` (line 24) currently maps physical harmonic
`h` to `ToneIndex(h, 0)`. Scale by the pump tone:
`ToneIndex(int(mode) * multitone_basis.pump_tone.h, 0)`. Identity when
`pump_tone == (1,0)`.

Same scale applies in `seed_from_floquet` (line 30) wherever a Floquet sideband
index is converted to a tone; audit that function and apply the identical
factor.

#### 4. Pump-mode policy

No code change. The DC-biased KI nonlinearity is not odd-symmetric, so the pump
basis must be `dense_real` (`[1,2,3]`), never `positive_odd_jc`. This is an
existing supported policy (`pump/basis.py`); it becomes a documented requirement
of the KIMPA driver, and the driver raises if an odd-only policy is requested
together with a non-zero DC bias.

### Success criteria

**Automated**: extend `tests/test_multitone_basis.py`

- `test_default_pump_tone_preserves_required_set` — the existing
  `test_basis_rejects_conjugate_pair_and_missing_required_tone` still passes
  unmodified.
- `test_half_pump_basis_places_pump_at_h2` — pump tone frequency equals
  `omega_p_physical` to 0.0 exact; signal and idler frequencies equal
  `omega_p/2 -/+ delta`.
- `test_half_pump_basis_requires_even_pump_modes` — odd physical harmonics land
  on even `h`.
- `test_promote_pump_solution_scales_by_pump_tone` — a 3-mode pump solution
  lands on `(2,0),(4,0),(6,0)`, and the default lattice still lands on
  `(1,0),(2,0),(3,0)`.
- `test_torus_round_trip_on_half_pump_basis` — `project(synthesize(X)) == X` to
  1e-14.

**Manual**: none.

---

## Phase 5: KIMPA builder and passive validation

### Overview

Build the three-section reflection amplifier and validate it with the pump off.
This is source-plan Stage 2.

### Changes required

#### 1. Transmission-line ladder

**File**: `src/twpa_solver/builders/kimpa.py` (new)
**Changes**: `add_transmission_line_ladder(builder, prefix, node_from, node_to,
z0_ohm, electrical_length, reference_frequency_hz, cells)` on top of the
named-node `CircuitBuilder` (`builders/jc_doc.py:76`).

```
tau_quarter = 1 / (4 f0)      tau_half = 1 / (2 f0)
L_line = Z0 tau               C_line = tau / Z0
```

Symmetric Pi discretization: `cells` series inductors of `L_line/cells`, and
`cells+1` shunt capacitors of `C_line/cells` with the two end capacitors halved
to `C_line/(2 cells)`. Do **not** stamp one series L and one shunt C — those are
totals to divide.

`ipm.py:610 add_tl` is an asymmetric L-ladder (full shunt C at the start node,
none at the last). It is not reused here because the end asymmetry biases a
reflection measurement.

At `f0 = 8 GHz` the fabricated targets are:

| Section | `Z0` | Length | `L_line` | `C_line` |
| --- | ---: | --- | ---: | ---: |
| input transformer | 80 ohm | quarter | 2.500 nH | 390.6 fF |
| resonator line | 30 ohm | half | 1.875 nH | 2.083 pF |
| high-impedance line | 180 ohm | quarter | 5.625 nH | 173.6 fF |

#### 2. Kinetic branch in the builder

**File**: `src/twpa_solver/builders/jc_doc.py`
**Changes**: Add `kinetic_inductor(name, n1, n2, Lk, Ic, model)`, appending to a
new `self.kinetic: list[KineticBranch]`. In `assemble`:

- KI branches contribute `Bphi` columns after the Josephson ones, in list order.
- **No `K` stamp** — same rule as Josephson (`docs/circuit_builders.md`).
- `Ic` array is extended with the KI critical currents so `CircuitMatrices`
  shape validation passes; it is a validity boundary for KI branches, never the
  nonlinearity scale.
- Return a `branch_law` in the assembled dict: pure `KineticInductorBranchLaw`
  when there are no Josephson branches, `CompositeBranchLaw` when both exist.

#### 3. The netlist

**File**: `src/twpa_solver/builders/kimpa.py`
**Changes**: `build_kimpa(...) -> CircuitMatrices`.

```
port 1 (50 ohm Norton)
   |
   +-- 80 ohm quarter-wave ladder  --> n1
   +-- 30 ohm half-wave ladder     --> n2
   +-- 180 ohm quarter-wave ladder --> nr
   |
   nr --- C_NR = 330 fF --- ground
   nr --- Lg = 200 pH --- nki          (linear inductor, stamps 1/Lg into K)
   nki -- kinetic inductor -- ground   (Bphi column, no K stamp)
```

`Lg` on its own node `nki` keeps `K` meaningful and keeps the branch law purely
kinetic — the law never has to know about a series geometric term.

Internal loss: a real shunt conductance `G = omega_0 C_NR / Qi` at node `nr`,
with `Qi = 1e5` and `omega_0` the reference frequency. This is deliberately not
a complex-`C` loss tangent: complex `C` forces `conductance_abs_omega` and
disables Tier-2 complex-omega resonance refinement (`CLAUDE.md`, dielectric
dissipation section). The conductance is exact at `omega_0` and a mild
approximation elsewhere; record that in metadata.

The 180 ohm NbTiN section is modelled as **linear**. All intended nonlinearity
lives in the lumped 250 nm x 20 nm nanowire.

#### 4. Three fixtures

**File**: `src/twpa_solver/builders/kimpa.py`

| Fixture | `Lk` | `Lg` | Lines | `C_NR` | Provenance |
| --- | ---: | ---: | --- | ---: | --- |
| `kimpa_ideal_synthesis` | 0.999 nH | 200 pH | 67.6 / 33.9 / 180 ohm | 330 fF | Appendix B synthesis at exactly 8.0 GHz |
| `kimpa_fabricated_nominal` | 0.835 nH | 200 pH | 80 / 30 / 180 ohm | 330 fF | `L_tot = Z_NR^2 C_NR = 1.0349 nH` at `Z_NR = 56 ohm` |
| `kimpa_measured_seed` | 0.633 nH | 200 pH | 80 / 30 / 180 ohm | 330 fF | `f_NR(0) ~= 9.6 GHz` measured, `L_tot = 832.9 pH` |

All three use `Ic = 1.15 mA`, `model = "hung_2025"`. The paper reports no unique
lumped `Lk`; each fixture is a different defensible reduction and they must not
be conflated. Arithmetic verified: 8.612 GHz / 8.000 GHz / 9.600 GHz isolated LC
frequencies respectively.

Register the three names in `scripts/run_compression.py:262` `_fixture_circuit`
and in the `--fixture` choices at `:144`.

#### 5. Passive readout

**File**: `scripts/run_kimpa_passive.py` (new)
**Changes**: Sweeps `passive_s_matrix(..., ports=(1,), dc_branch_flux=...)` over
6-12 GHz for a chosen fixture; writes CSV (`freq_ghz`, `s11_re`, `s11_im`,
`s11_db`, `s11_phase_rad`) and a PNG. Locates resonance minima by parabolic
interpolation on `|S11|` and reports them.

### Success criteria

**Automated**: `tests/test_kimpa_builder.py`

- `test_ladder_reproduces_analytic_line_impedance` — a bare ladder terminated in
  `Z0` has `|S11| < -40 dB` at `f0`; asserts the discretization, not just that
  code runs.
- `test_ladder_quarter_wave_transforms_impedance` — a quarter-wave `Z0` section
  terminated in `R` presents `Z0^2/R` at `f0` to 1%.
- `test_ladder_cell_convergence` — resonance frequency at 10/20/10, 20/40/20 and
  40/80/40 cells agrees to better than 0.1%, so the reported answer does not
  depend on the subdivision.
- `test_geometric_inductance_is_in_K_not_the_branch_law` — `Lg` appears in `K`;
  the branch law's `differential_inductance(0)` equals `Lk` exactly, with no
  `Lg` contribution.
- `test_kinetic_branch_not_stamped_into_K` — removing the KI branch from the
  netlist leaves `K` unchanged.
- `test_three_fixtures_have_expected_isolated_lc_frequency` — 8.612 / 8.000 /
  9.600 GHz to 0.1%.
- `test_fabricated_nominal_lk_from_z_and_c` — `Lk = Z_NR^2 C_NR - Lg` to 0.1 pH.

**Manual**: run `run_kimpa_passive.py` on all three fixtures; confirm a two-pole
`S11` structure and that `kimpa_measured_seed` puts the nonlinear-resonator pole
near 9.6 GHz.

---

## Phase 6: DC bias and DC tuning

### Overview

Source-plan Stage 3: reproduce the measured resonance shift versus `I_dc`,
isolating the static KI model before any pump is applied.

### Changes required

#### 1. DC flux helper

**File**: `src/twpa_solver/core/kinetic.py`
**Changes**: `kinetic_dc_branch_flux(circuit, dc_current_a) -> np.ndarray`.
Returns `Phi_k(I_dc)` on KI branches and `0.0` on all others.

No DC solve is needed and none is added. The DC path is
port -> series ladder inductors -> `nr` -> `Lg` -> `nki` -> KI -> ground; the
shunt capacitors block DC and every series element on that path except the
nanowire is linear, so only the KI branch's DC flux matters. The linear network
is untouched because the HB basis carries no DC tone (`basis.py:100` rejects
`(0,0)`) and a linear element has no DC-to-AC coupling.

This makes the existing residual convention `i(psi + psi_dc) - i(psi_dc)`
exactly the source plan's "fixed branch-current offset" option, already
implemented on every problem class.

#### 2. Driver flag

**File**: `scripts/run_kimpa_passive.py`
**Changes**: `--dc-current-a` (repeatable) drives a sweep; writes one CSV row
per `(I_dc, frequency)` and a resonance-vs-bias summary.

### Success criteria

**Automated**: `tests/test_kinetic_dc.py`

- `test_dc_branch_flux_inverts_to_the_requested_current` —
  `law.current(kinetic_dc_branch_flux(circuit, I)) == I` to 1e-14 relative.
- `test_dc_flux_is_zero_on_non_kinetic_branches`.
- `test_resonance_shift_follows_paper_dc_law` — over
  `I_dc = 0 .. 600 uA`, the fitted resonance shift `f(I)/f(0)` matches
  `1/sqrt(1 + (I/3.25mA)^2 + (I/1.70mA)^4)` weighted by the kinetic
  participation `Lk/(Lk+Lg)`, to better than 0.5%. This is the phase's real
  gate: it ties the implemented law to the paper's measured curve.
- `test_residual_removes_the_static_dc_current` — the HB residual at `X = 0`
  under bias is zero to 1e-30.

**Manual**: plot resonance versus `I_dc` for all three fixtures; the shift over
0-600 uA should be a few tens of MHz and monotone downward.

---

## Phase 7: pump solve and one-port Floquet gain

### Overview

Source-plan Stages 4 and 5. Pump near 16.94 GHz, signal near 8.47 GHz, one-port
reflection gain.

### Changes required

#### 1. Driver

**File**: `scripts/run_kimpa_gain.py` (new)
**Changes**: Composes existing pieces; no new solver code.

0. Pump power: `pump_loss_model().dbm_to_peak_current_a(P_p, 16.94)` — **43.425
   dB** of A10 line loss, applied by default (Phase 2b). The paper's
   `P_p = -29.6 dBm` is an instrument-referred number; the matched-port current
   it implies at the *chip* is what drives the network. Report the on-chip pump
   power alongside, and note the 6.02 dB Norton caveat: ports are
   Norton-terminated, so `I^2 Z/2` overstates delivered power by 6.02 dB unless
   the outgoing wave is used.
1. `resolve_pump_basis(policy="dense_real", mode_count=3, omega_p=...)`,
   `nt = 32` (needs `>= 2*3+1`). Raise if an odd-only policy is combined with
   non-zero DC bias.
2. `FullPumpProblem(..., branch=make_branch_law(circuit),
   dc_branch_flux=kinetic_dc_branch_flux(circuit, I_dc),
   pump_node_index=port_to_index[1], source_mode=1)`, solved by
   `HarmonicNewtonKrylovSolver` with natural-parameter continuation in source
   scale, exactly as the JJ path does.
3. `compute_gamma_hat(circuit, pump, max_ell, gamma_nt, dc_branch_flux)` —
   already correct after Phase 2.
4. `solve_gain_one(..., source_index=out_index=port_to_index[1],
   source_port=1, out_port=1)`.

`GainResult` already carries both requested outputs: `gain_db = 20 log|S11|`
(source-plan `G_s`) and `gain_vs_off_db = 10 log|V_on/V_off|^2` (source-plan
`G_norm`). Both are written, so a normalization convention can never manufacture
an apparent discrepancy.

Saved per point: nonlinear-branch pump current waveform, total instantaneous
current, harmonic spectrum, `max|I|/Ic`, `KineticValidity` status, pump residual,
and continuation telemetry.

#### 2. Cost note

The KIMPA netlist is roughly 82 nodes (20 + 40 + 20 ladder nodes plus `nr` and
`nki`) against 2048 for the 2c device. The coupled Jacobian memory scaling
`(n_pump_modes + 2S + 1)^2` documented in `CLAUDE.md` therefore never bites
here; the `full` backend and the default `pardiso` factor backend are correct
choices, and neither `banded` nor `--signal-workers` tuning is needed.

### Success criteria

**Automated**: `tests/test_kimpa_gain.py` (marked slow)

- `test_pump_solve_converges_under_dc_bias` — coefficient residual < 1e-10 at
  `I_dc = 550 uA`, `P_p = -29.6 dBm`.
- `test_gamma_hat_has_odd_ell_under_dc_bias` — `|gamma_hat[1]|` is non-negligible
  relative to `|gamma_hat[0]|` under bias and vanishes at `I_dc = 0`. This is
  the direct 3WM test: a DC-biased KI element must produce odd pump-order
  coupling that an unbiased symmetric nonlinearity cannot.
- `test_gamma_hat_conjugate_symmetry` — `gamma_hat[-l] == conj(gamma_hat[l])`
  to 0.0, the existing correctness invariant for a real pump.
- `test_one_port_gain_matches_pump_off_normalization` — `gain_db` and
  `gain_vs_off_db` differ by exactly the pump-off `|S11|` in dB.
- `test_gain_is_unity_with_pump_off` — `|S11| = 1` to 1e-6 on the lossless
  variant (`Qi = inf`), a passive-reciprocity check on the whole one-port chain.

**Manual**: sweep signal frequency at `P_p = -29.6 dBm`, `I_dc` in
530-600 uA. Expect the ideal circuit's **two** gain maxima. The paper's ideal
negative-resistance calculation gives `|xi_3|/2pi = 1.56 GHz` with about 400 MHz
of bandwidth at 17 dB. The four-peak experimental structure is **not** expected
here — it comes from the standing-wave environment, which is out of scope.

Record whether the two-pole shape appears. A negative result is a result; do not
tune fixtures to manufacture it.

---

## Phase 8: finite-signal compression

### Overview

Source-plan Stage 7. Reuses `run_compression.py` wholesale.

### Changes required

#### 1. Half-pump wiring in the compression driver

**File**: `scripts/run_compression.py`
**Changes**:

- `--multitone-lattice {full_pump,half_pump}`, default `full_pump` so every
  existing invocation is untouched.
- Under `half_pump`, `_build_multitone_basis` (line 296) calls
  `build_half_pump_basis`; `delta` becomes `omega_p/2 - omega_s` rather than
  `omega_p - omega_s` (line 516).
- `MultiToneDrive(basis.pump_tone, ...)` at line 519 already reads the basis, so
  it follows the configurable pump tone with no edit.
- `--dc-current-a` threads `kinetic_dc_branch_flux` into both problems.
- `--source-port 1 --out-port 1` is already supported; the one-port case needs
  no new code.
- Signal power referral uses `signal_loss_model()` at `f_s = 8.47 GHz`
  (**60.790 dB**, Phase 2b), not the pump line. The paper's output saturation
  figure of `-51 +/- 3 dBm` is an output-referred number; state which line and
  which reference plane every reported dBm belongs to, in the CSV header, not
  only in prose.

#### 2. Threshold reporting

**File**: `scripts/run_compression.py`
**Changes**: Per power point emit `max_current_over_ic` and
`kinetic_status`. Three outcomes stay strictly distinct in the CSV, never
collapsed:

```
SMOOTH_COMPRESSION       1 dB compression reached, max|I| < Ic throughout
THRESHOLD_CROSSED        max|I| >= Ic before 1 dB compression
SOLVER_FAILED            Newton did not converge
```

The reported point is
`P1dB_effective = min(P1dB_smooth, P_SC)`, with which branch produced it named
in the output. A threshold crossing is a physical statement about the device; a
solver failure is a statement about the solver. Conflating them would repeat the
"fold versus convergence failure" error recorded in project memory.

#### 3. Observables scope

`conversion_manley_rowe_*` is restricted to pump/signal/idler
(`observables.py:344`) and picks up the configurable pump tone automatically.
Per project memory, all-tone Manley-Rowe is **not** a valid invariant and must
not be used as a gate here either.

### Success criteria

**Automated**: `tests/test_kimpa_compression.py` (marked slow)

- `test_half_pump_compression_runs_and_reports_status` — a coarse 3-point sweep
  produces `SMOOTH_COMPRESSION` or `THRESHOLD_CROSSED`, never a bare number
  without a status.
- `test_threshold_crossing_is_not_reported_as_p1db` — force a low `Ic` and
  assert the CSV reports `THRESHOLD_CROSSED` and no `p1db` value.
- `test_full_pump_lattice_default_is_unchanged` — an existing jpa compression
  fixture run reproduces its stored `p1db` to all printed digits.
- `test_conversion_manley_rowe_uses_configured_pump_tone`.

**Manual**: compare the output saturation power against the paper's
`-51 +/- 3 dBm` at `P_p ~= -29.7 dBm`. Report the comparison as-is; the paper is
the only external reference available for this device and it has not been
validated against this solver before, so an agreement claim needs the
uncertainty stated alongside.

---

## Phase 9: joint S11 + DC-tuning parameter fit

### Overview

The paper reports no unique lumped `Lk`. The three Phase 5 fixtures span
0.633-0.999 nH — a 58% spread. This phase replaces the guess with a fit, and it
comes before the remaining physics work because every quantitative claim depends
on it.

Worth knowing before starting: the nonlinearity scales with the **participation
ratio** `Lk/(Lk+Lg)`, which is 0.760 -> 0.833 across that fixture range, only a
**9%** spread. The 58% `Lk` uncertainty is therefore mostly a *resonance*
uncertainty (8.61 -> 9.60 GHz), not a gain uncertainty. Say so when reporting;
the fixture spread looks more alarming than it is for gain.

### Changes required

#### 1. Fit driver

**File**: `scripts/fit_kimpa_linear.py` (new)
**Changes**: Reuses the architecture of `scripts/align_map_to_measurement.py` —
nuisance parameters, coarse-then-fine grid search, ROI weighting,
`--loss {l2,huber}`, JSON output, and **a plotted loss surface**. That script
already learned the lesson this one needs: a point estimate hides a
weakly-identified basin (its full-map `df = -0.30 GHz` minimum was elongated and
only per-section fits were well identified).

Parameters, four of them:

| Parameter | Bound | Why |
| --- | --- | --- |
| `Lk` | 0.60-1.00 nH | the practical prior from the three fixtures |
| `C_NR` | 250-400 fF | design value 330 fF |
| line electrical-length scale | 0.9-1.1 | **one** scalar across all three sections |
| `Qi` | 1e4-1e6 | design value ~1e5 |

One length scale, not three. Three independent lengths are degenerate against
each other on a two-pole response and will produce a flat valley rather than a
minimum.

#### 2. Joint objective — the point of this phase

`Lk` and `C_NR` are near-degenerate on a single resonance, because only the
product `LC` sets it. Two poles help but do not fully separate them. The clean
degeneracy-breaker is the **DC tuning curve**: the fractional shift depends on
`Lk/(Lk+Lg)` and **not** on `LC`.

So fit Stage 2 and Stage 3 **jointly**, one objective over both datasets:

```
J = w_S11 * ||S11_model(f) - S11_meas(f)||_ROI
  + w_dc  * ||f_res_model(I_dc) - f_res_meas(I_dc)||
```

Sequential fitting — S11 first, DC tuning as a check afterwards — would let a
wrong `Lk`/`C_NR` split survive Stage 2 and then be misattributed to the
nonlinear model in Stage 3. That failure mode is exactly the "model-fidelity
gap" pattern already recorded for the 2c campaign.

Fit the pole frequencies and linewidths rather than raw `|S11|` where possible:
fewer degrees of freedom and insensitive to an overall calibration offset.

#### 3. Gate

Do not run this before Phase 7 shows the ideal two-pole gain shape. Fitting a
model that cannot produce the right qualitative behaviour only launders the
error into the parameters.

### Success criteria

**Automated**: `tests/test_fit_kimpa_linear.py`, mirroring
`tests/test_align_map.py`

- `test_fit_recovers_synthetic_parameters` — generate S11 and a DC-tuning curve
  from known `(Lk, C_NR, scale, Qi)`, add noise, recover to within 2%.
- `test_lk_and_cnr_are_degenerate_on_s11_alone` — S11-only fitting leaves a
  valley along constant `LC`; **assert the degeneracy exists**, so the joint
  objective is justified by a test rather than by argument.
- `test_joint_objective_breaks_the_degeneracy` — adding the DC-tuning term
  collapses that valley to a single minimum.
- `test_loss_surface_is_emitted`.
- `test_three_independent_line_lengths_are_rejected`.

**Manual**: inspect the loss surface. If the basin is elongated, report the
fitted values **with that shape shown**, not as a bare point estimate.

---

## Phase 9b: the alternative static model

### Overview

The paper fits its DC data two ways. Phase 1 implements the polynomial; this
implements the alternative, so the choice of DC law can be shown to be — or not
to be — a lever on the predicted gain.

### Correction to an earlier judgement

An earlier draft deferred this on the grounds that
`L(I) = Lk / [1 - (I/I**)^n]^{1/n}` has no closed-form flux integral. **That was
wrong.** Substituting `x = I** t` then `s = t^n` turns the integral into an
incomplete Beta function:

```
Phi(I) = (Lk I** / n) * B(w; 1/n, 1 - 1/n),          w = (I/I**)^n
       = (Lk I** / n) * (pi / sin(pi/n)) * betainc(1/n, 1 - 1/n, w)
```

using `B(1/n, 1-1/n) = pi / sin(pi/n)` by the reflection formula.
`scipy.special.betainc` is vectorized, and the **inversion is also closed form**
via `scipy.special.betaincinv` — no Newton iteration at all:

```
w = betaincinv(1/n, 1 - 1/n, Phi / Phi_max)
I = I** * w**(1/n)
```

At `n = 2.21`: `pi/sin(pi/2.21) = 3.1772`, so
`Phi_max = Phi(I**) = 1.4376 * Lk * I**` — **finite**, because the integrand's
`(1-t)^(-1/n)` singularity has exponent `-0.4525 > -1`. The inductance diverges
at `I**`; the flux does not.

### Changes required

#### 1. The law

**File**: `src/twpa_solver/core/kinetic.py`
**Changes**: `KineticInductorAltBranchLaw` with `istarstar_a` and `n`,
preset `hung_2025_alt` carrying `I**/Ic = 1.65/1.15 = 1.4347826087`,
`n = 2.21`. Odd extension for `I < 0`. Registered as a **separate named model**;
it must never be silently mixed with the polynomial.

#### 2. The domain wall — the only real engineering issue

`|Phi| <= Phi_max` is a hard boundary. Raising from inside a residual evaluation
is dangerous: a trial Newton step that overshoots would abort the solve instead
of being backtracked. Instead:

- extrapolate monotonically and smoothly past `Phi_max` (linear continuation at
  the boundary slope),
- increment an `out_of_domain_samples` counter on the law,
- let the existing line search (`min_alpha = 1/1024`) pull the step back,
- surface a non-zero counter in the report as a diagnostic, never as a
  hard failure.

At the operating point this rarely fires: `I_dc/I** = 550 uA / 1.65 mA = 0.333`,
so `w = 0.092`, far from the wall.

### Success criteria

**Automated**: extend `tests/test_kinetic_branch.py`

- `test_alt_flux_matches_numerical_quadrature` — the Beta form agrees with
  `scipy.integrate.quad` of `L(I)` to 1e-12 over `|I| < 0.99 I**`.
- `test_alt_inversion_is_exact` — `flux(current(phi)) == phi` to 1e-13 without
  any Newton step.
- `test_alt_flux_max_is_finite_and_matches_1_4376_lk_istarstar`.
- `test_alt_reproduces_paper_dc_tuning` — the fitted resonance shift matches the
  paper's alternative one-parameter curve.
- `test_out_of_domain_extrapolates_and_counts_instead_of_raising` — feeding
  `Phi > Phi_max` returns a finite monotone value and increments the counter.

**Manual**: rerun Phase 7's gain sweep with both DC laws. **If the gain agrees,
the DC-law choice is not a lever on the answer** — that is the deliverable of
this phase, and it is worth more than either individual number.

---

## Phase 10: standing-wave port environment

### Overview

Source-plan Stage 6. The paper's ideal circuit produces two gain maxima; the
measured four-peak structure comes from the environment. This is the only route
to the measured shape, so it is not optional forever — but it is meaningless
until Phase 7 shows the ideal two-pole result.

```
Z_env(w) = 50 + Z1 exp(i(w tau1 + phi1)) + Z2 exp(i(w tau2 + phi2))
Z1 = 14.2 ohm, phi1 = -0.7 pi
Z2 =  1.9 ohm, phi2 = 0
```

### Resolve before writing code

The source states `tau1/2pi = 10.5 ns` and `tau2/2pi = 121 ns`. Read literally
that gives `tau2 = 760 ns`, which is implausibly long for this setup; almost
certainly the delays are 10.5 ns and 121 ns written in a frequency-domain
convention. **Check the paper text before implementing.** A `2 pi` error here is
a factor-6 error in the ripple period, which would present as "the comb spacing
is wrong" and be misdiagnosed as a circuit-model problem.

### Changes required

#### 1. Port environment object

**File**: `src/twpa_solver/core/environment.py` (new)
**Changes**: `PortEnvironment` with `admittance(omega) -> complex`, returning
`1/Z_env(w) - 1/Z0` — the *correction* to the already-stamped Norton
conductance, so a null environment is exactly zero.

`__post_init__` asserts passivity: `Re Z_env(w) > 0` across the retained band.
Here `Z1 + Z2 = 16.1 < 50`, so `Re Z_env >= 33.9 ohm` and the assertion holds
with margin — but an environment that fails it would inject energy, and that
must be caught at construction, not diagnosed from a diverging solve.

#### 2. Three block-builder hooks

A complex, frequency-dependent termination cannot be stamped into `C`, `G` or
`K`, which are real and frequency-independent. It *can* be added per block,
because every block builder already evaluates at a known frequency:

| Path | Builder | Frequency |
| --- | --- | --- |
| pump HB | `pump/problem.py:169 _build_linear_blocks` | `k * omega_p` |
| Floquet | `floquet.py:121` and `:152 dynamic_block` | `omega_s + m * omega_p` |
| multitone | `multitone/problem.py:83` | `basis.omegas` |

Each adds `env.admittance(omega)` to the port-node diagonal of its own block.
Default `None` is a no-op everywhere, so this is additive.

### Success criteria

**Automated**: `tests/test_port_environment.py`

- `test_null_environment_is_exactly_zero` — `Z_env = 50` reproduces the ideal
  result bit-for-bit across all three paths. This is the regression gate.
- `test_passivity_assertion_rejects_active_environment` — `Z1 = 60` raises.
- `test_environment_appears_in_all_three_block_builders` — mutating the
  environment changes the pump, Floquet and multitone matrices; a hook silently
  missing from one path is the obvious failure mode.
- `test_ripple_period_matches_delay` — a single-term environment produces `|S11|`
  ripple of period `1/tau`, which pins the `2 pi` convention numerically instead
  of by reading.

**Manual**: rerun the Phase 7 gain sweep with the environment on. Expect the
two-peak ideal shape to split toward the measured four-peak structure. Record
the outcome either way; do not tune `Z1`, `Z2`, or the delays to manufacture
four peaks — they are quoted values, not free parameters.

---

## Testing strategy

### Project maturity level

**Established Production.** The solver has ~60 test modules, physics gates, and
published numbers that downstream documents cite. The refactor phase (2) touches
the shared JJ path, so it carries the highest regression risk in the whole plan
despite being the smallest diff.

### Unit tests

- Branch-law math: round trip, closed-form agreement, finite-difference tangent,
  DC-law reproduction, symmetry under and without bias, threshold status.
- Composite dispatch and column-partition validation.
- Persistence round trips including the legacy no-`branch_law_kind` path.
- Basis: default-preserving pump tone, half-pump placement, promotion scale,
  torus round trip.
- Ladder: analytic impedance, quarter-wave transformation, cell convergence.
- Loss: B1 analytic-coefficient recovery, A10 frozen constants, the two models
  being distinct, and old-convention replay via `--attenuation-db`.
- Alt law: Beta form against quadrature, exact inversion, finite `Phi_max`,
  out-of-domain extrapolation instead of raising.
- Environment: null case bit-identical, passivity assertion, presence in all
  three block builders, ripple period pinning the `2 pi` convention.
- Fit: synthetic recovery, the S11-only degeneracy asserted to exist, and the
  joint objective collapsing it.
- Boundaries: `I = 0`, `I = Ic`, `I > Ic`, `istar4_a = None`, `|Phi| > Phi_max`,
  zero-length KI list, KI-only circuit, mixed JJ+KI circuit.

Coverage target: 80% on the new modules (`core/kinetic.py`,
`builders/kimpa.py`), and **every new gate must be demonstrated failing against
a deliberate mutation before it is trusted** — per project memory, a gate that
has never been seen red is not a gate.

### Integration tests

- Phase 2 bit-identity: one published gain map and one published compression
  point reproduced exactly.
- Phase 2b: old-convention replay byte-identical under `--attenuation-db`.
- Phase 5: pump-off `S11` two-pole structure across three fixtures.
- Phase 6: resonance shift versus `I_dc` against the paper's polynomial.
- Phase 7: pump-on one-port gain, both normalizations, `gamma_hat` odd-`ell`
  content under bias.
- Phase 8: compression with the three-way status split.
- Phase 9: synthetic parameter recovery, then the real joint fit.
- Phase 9b: gain compared under both DC laws.
- Phase 10: null environment bit-identical to Phase 7.

Full suite, temporary directory outside the repository:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_ki_full --run-slow
```

### Manual verification

1. `twpa_solver.__file__` points inside this repo before trusting any run.
2. Phase 2 diffs against stored artifacts, digit for digit.
3. Passive `S11` plots for the three fixtures.
4. DC tuning curve versus the paper's polynomial.
5. Gain sweep: two-pole shape present or absent, recorded either way.

---

## Rollback plan

Each phase is an independent commit on a `kinetic-inductance` branch off `dev`.

| Phase | Rollback |
| --- | --- |
| 1, 3, 5, 9b | Pure additions (`core/kinetic.py`, `builders/kimpa.py`, new tests, new npz keys ignored by old readers). Revert the commit; nothing else references them. |
| 2 | The riskiest phase. It is a pure refactor with bit-identity gates, so revert restores the previous expressions verbatim. Keep it as one commit touching six files and nothing else, so `git revert` is clean. |
| 2b | The only phase that intentionally changes published numbers. Kept out of Phase 2 for exactly that reason. Revert restores the single-scalar path; `--attenuation-db` reproduces it without a revert. |
| 4 | Additive dataclass field with a behavior-preserving default. Revert removes `build_half_pump_basis` and the `pump_tone_index` field; every existing basis construction is unaffected either way. |
| 6, 7, 8, 9 | New scripts plus opt-in flags defaulting to current behavior. Revert removes the scripts and the flags. |
| 10 | Additive `PortEnvironment` defaulting to `None` in three block builders. The null-case bit-identity test is the guard; revert removes the parameter. |

If Phase 2 bit-identity fails on any published artifact, stop. Do not proceed to
later phases and do not adjust the expected value — a mismatch there means the
Josephson law and the hardcoded expression disagree somewhere, which is a real
finding about existing results, not a test to relax.

Phase 2b is the exception to that rule and the reason it is a separate phase: it
changes input-referred numbers **by design**. Its gate is that the old
convention remains exactly reproducible on demand, not that the new numbers
match the old ones.

No commits without explicit request. No `Co-Authored-By` trailer (repo
convention).
