# Physically correct model↔measurement saturation comparison — fix plan

Status: planned, not implemented. Written 2026-08-05.

## Goal

Every power on both sides carries one derivable convention; the model operating
point is fitted to the only calibration-free observable; `G0` and `P1dB` are
compared as **separate** observables before `P_sat` is formed; and the residual
saturation discrepancy is reported as a dimensionless pump-referred number that
no calibration constant can absorb.

---

## Verified inputs

Everything below was measured or derived during planning, not assumed.

### Norton port termination is a fact of the netlist

`designs/ipm_2c_fixed` has a `G` matrix with **exactly four nonzeros**, all
`0.02 S = 50 Ω`, one per port (nodes 0, 4576, 4577, 6135). Nothing else in the
circuit is conductive. So every drive is an ideal current source in parallel
with `G₀ = 1/Z₀`:

- the load sees `I/2` (peak), so `P = ½·(I/2)²·Z₀ = I²Z₀/8`;
- equivalently the incident wave is `a = (I/2)√Z₀` and `P_inc = |a|²/2`.

The traveling-wave form `I²Z₀/2` applies to a wave amplitude, not to a Norton
drive current, and **overstates by exactly `10log₁₀4 = 6.0206 dB`**.

Independent confirmation: at `I = 7.3803e-6 A` the model pump reads
**−64.68 dBm** under `I²Z₀/8` against the depletion-inferred on-chip pump of
**−65.98 dBm** — agreement **1.30 dB**. Under `I²Z₀/2` it reads −58.66 dBm, off
by 7.32 dB.

`CLAUDE.md` currently records `I²Z₀/2` as "validated 2026-07-18". That
validation was a JosephsonCircuits.jl parity check about a factor of two in
*current*; JC has since been retired as a reference and it never addressed the
Norton question.

### loss_B1 is a closed form, not a noisy measurement

`docs/development/loss_B1.csv` is reproduced by

    att(f) = 50.0 + 3.3·√f + 0.14·f      (f in GHz, dB)

to **RMS 2.80e-5 dB**, max 5.17e-5 dB. Same functional family as `loss_A10`, so
`InsertionLossModel.fit_csv` already handles it and the coefficients can be
frozen the way A10's are at `src/twpa_solver/loss.py:26-28`.

| f (GHz) | loss_B1 (dB) | fabricated flat | on-chip power error |
| ---: | ---: | ---: | ---: |
| 4.0 | 57.16 | 72.5 | **+15.34 dB** |
| 7.256 | 59.91 | 72.5 | +12.59 dB |
| 12.0 | 63.11 | 72.5 | **+9.39 dB** |

The flat constant also erased a **5.95 dB tilt** across the band. That tilt is
missing from every measured `P1dB(f)` and `P_sat(f)` curve published so far.

### The pump line is loss_A10, forced by energy conservation

Signal + idler output at the compression point cannot exceed the pump:

    P_sat + 3 dB  ≤  P_pump

With `loss_B1` on the signal line the measured `P_sat` is **−64.43 dBm**, so
signal+idler is ≈ **−61.4 dBm** and the on-chip pump must exceed ≈ −61 dBm, i.e.
the pump-line loss must be **below ~40 dB**.

| pump-line loss | on-chip pump | `P_sat − P_pump` | verdict |
| --- | ---: | ---: | --- |
| `loss_A10`, 34.54 dB @7.256 | **−55.54 dBm** | −8.89 dB | **consistent** |
| inferred 45.0 dB | −66.00 dBm | +1.57 dB | violates energy conservation |
| `loss_B1`, 59.91 dB | −80.91 dBm | +16.48 dB | impossible |

So: **signal line = `loss_B1`, pump line = `loss_A10`**, on-chip pump
**−55.54 dBm** (instrument `PumpPower = −21 dBm`).

### What the corrections do to the headline

| quantity | as published | corrected |
| --- | ---: | ---: |
| measured `P1dB` median | −85.92 | **−73.34** dBm |
| measured `P_sat` median | −77.76 | **−64.43** dBm |
| model `P1dB` median | −93.78 | **−99.80** dBm |
| model `P_sat` median | −81.33 | **−87.35** dBm |
| model on-chip pump | −58.66 | **−64.68** dBm |

The device needs **9.14 dB more pump than the model** for comparable gain. That
is the expected sign for real-device nonidealities and is precisely why the
operating points legitimately differ — the model's OP is *not* meant to be the
device's OP.

What survives that, because each side is now internally consistent:

| | measured | model | gap |
| --- | ---: | ---: | ---: |
| `P_sat − P_pump` | −8.89 dB | −22.67 dB | **13.78 dB** |
| pump depletion at compression | ≈26% | ≈1.3% | ≈20× |

No line-loss constant can move that, since each side's pump and signal now share
one convention. Note also that **26% depletion producing only 1 dB of gain
compression is itself questionable** and may point at the pump-line figure
rather than at the model. Phase 7 is built to expose that rather than absorb it.

---

## The four defects

None of them is a physics-model error.

| # | Defect | Site | Size |
| --- | --- | --- | --- |
| 1 | Port power uses `I²Z₀/2`; ports are Norton, available power is `I²Z₀/8` | `scripts/run_compression.py:338-339`, `:720-721`, `:947`, `:1006`; `src/twpa_solver/loss.py:47` | exactly **6.0206 dB** |
| 2 | Signal line loss is a fabricated flat 72.5 dB | `scripts/measured_psat_pipeline.py:72`, `:164` | **+9.39…+15.34 dB** plus a **5.95 dB** band tilt |
| 3 | `P_sat` mixes two gain estimates: threshold uses `g0_local`, sum uses `G0_smooth` | `measured_psat_pipeline.py:490-497` vs `:204` | 0.23 dB median, **−4.46…+2.53 dB** per column |
| 4 | Operating point matched one map cell's S21 at 500 MHz detuning against the measured **band peak** | no script; ad hoc | model band **+2.63 dB, +351 MHz** off |

**Defects 1-3 need no re-solve.** Gain is `20log₁₀|V_out/I_in|` pump-on over
pump-off (`run_compression.py:660-680`) — a ratio of ratios, invariant under any
source-scale convention. So:

- the Norton fix is a rigid **−6.0206 dB** relabel of model powers;
- the `loss_B1` fix is a **per-column rigid shift** of the measured power axis,
  since attenuation is constant across each column's power sweep:
  `P1dB_new(f) = P1dB_old(f) + 72.5 − att_B1(f)`;
- existing gain maps are relabeled, not invalidated, **provided the drive stays
  specified as a current**.

---

## What we're NOT doing

- No change to solver physics, circuit builders, or basis truncation.
- No re-solve for Phases 1-3 — pure relabeling, justified by the invariance
  argument above.
- No re-running of existing gain maps; their dBm axes get a documented
  −6.0206 dB relabel applied at read time.
- Not resolving *why* the device needs 9.14 dB more pump (loss vs disorder vs
  phase mismatch).
- Not touching the production-basis truncation caveat.
- Not deleting legacy numbers — they stay reproducible behind an explicit flag.

## Prerequisites

- [ ] Confirm `twpa_solver.__file__` resolves inside this repo, not
      `D:\tmp\finalclone` (the editable install has shadowed it before).
- [ ] Baseline green:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_psatfix tests/test_loss_model.py tests/test_run_compression_cli.py`

---

## Phase 1 — One source of truth for port power

### Overview

Norton-source power becomes a single derived function; every conversion site
calls it. Legacy stays reproducible behind a flag.

### Changes

#### 1. New port-power module

**File**: `src/twpa_solver/ports.py` (new)

```python
PORT_POWER_CONVENTIONS = ("norton", "legacy_traveling_wave")
LEGACY_TW_OFFSET_DB = 10.0 * math.log10(4.0)  # 6.020599913279624

def port_available_power_w(current_a, z0_ohm, convention="norton") -> float: ...
def port_current_from_power_a(power_w, z0_ohm, convention="norton") -> float: ...
```

Docstring carries the derivation and the `G`-diagonal evidence.

#### 2. Compression driver

**File**: `scripts/run_compression.py`

`_current_to_dbm` (`:338`), the `signal_power`/`pump_power` pair (`:720-721`),
and both inverse conversions inside P1dB refinement (`:947`, `:1006`) route
through `ports`. Add `--power-convention {norton,legacy_traveling_wave}`,
default `norton`; record it in `compression_summary.json` as
`power_convention`.

#### 3. Loss-model inverse

**File**: `src/twpa_solver/loss.py`

`dbm_to_peak_current_a` (`:47`) takes `convention` and routes through `ports`;
default `norton`. Docstring notes that for a fixed dBm the Norton drive current
is **2×** the legacy one, so any map regenerated with the same dBm bounds is a
different physical sweep.

#### 4. Gain-map driver

**File**: `scripts/run_gain_map.py`

Forward `--power-convention`; write it into map metadata. Maps lacking the key
were produced under `legacy_traveling_wave`; their dBm axis is relabeled by
−6.0206 dB at read time.

### Success criteria

**Automated**
- `tests/test_port_power.py`: `port_available_power_w(I, 50) == I²·50/8`;
  round-trip against `port_current_from_power_a`; legacy branch exactly
  `LEGACY_TW_OFFSET_DB` above; mutation check that swapping 8→2 fails.
- Netlist-consistency test: for `designs/ipm_2c_fixed`,
  `G[port, port] == 1/z0_ohm` at all four ports.
- `tests/test_run_compression_cli.py` green, plus a case asserting
  `--power-convention legacy_traveling_wave` reproduces the pre-change
  `p1db_input_dbm` bit-for-bit.

**Manual**
- Model pump at `I = 7.3803e-6 A` reads **−64.68 dBm** (was −58.66); residual
  against −65.98 dBm is **+1.30 dB**.
- Reprocessing the existing 20-point sweep shifts `p1db_input_dbm` and
  `p1db_output_dbm` by exactly **−6.0206 dB** and leaves
  `small_signal_gain_vs_off_db` bit-identical.

---

## Phase 2 — Real, frequency-resolved line calibration

### Overview

Delete the fabricated constant. Signal line = `loss_B1`, pump line =
`loss_A10`, both frequency-dependent, both named and traceable.

### Changes

#### 1. Freeze loss_B1

**File**: `src/twpa_solver/loss.py`

Add `LOSS_B1_C_DB = 50.0`, `LOSS_B1_A_DB = 3.3`, `LOSS_B1_B_DB = 0.14` beside
the A10 block, plus `signal_line_loss_model()` and `pump_line_loss_model()`
(= `default_loss_model()`, A10). Comment records the 2.80e-5 dB fit residual and
the energy-conservation argument that fixes A10 on the pump line.

#### 2. Measured pipeline power axis

**File**: `scripts/measured_psat_pipeline.py`

Remove `SIGNAL_LINE_LOSS_DB`. `MeasurementCube` stores `instrument_power_dbm`
(1-D, 121) and `signal_attenuation_db` (1-D, 2001), and exposes
`on_chip_power_dbm(column_index) -> np.ndarray`. Add
`MEAS_PUMP_INSTRUMENT_DBM = -21.0` and derive

```python
on_chip_pump_dbm = -21.0 - pump_line_loss_model().attenuation_db(MEAS_PUMP_GHZ)
```

Every `cube.power_dbm` consumer — the `p1db_cut_at_p2db` call site (`:299`) and
both diagnostic plots (`:587`, `:655`) — takes the column index.

### Success criteria

**Automated**
- `tests/test_loss_model.py` extended: `fit_csv("docs/development/loss_B1.csv")`
  reproduces the frozen coefficients to 1e-6, RMS residual < 1e-4 dB.
- Shift identity test: `p1db_new(f) − p1db_old(f) == 72.5 − att_B1(f)` per
  column, to 1e-9.

**Manual**
- Attenuation runs 57.16 dB @4 GHz → 63.11 dB @12 GHz.
- Measured median `P1dB` moves **−85.92 → −73.34 dBm**; on-chip pump prints
  **−55.54 dBm**.
- Every figure caption carries both loss models by name.

---

## Phase 3 — Internally consistent P_sat

### Overview

`P1dB` is *defined* by `G(P1dB) = g0_local − 1`, so the output-referred point is
`P1dB + g0_local − 1` identically. Using `G0_smooth` in the sum makes the added
gain not the gain at compression.

### Changes

#### 1. Stage 4

**File**: `scripts/measured_psat_pipeline.py`

`stage4_psat_vs_frequency_plot` takes `g0_local_usable` instead of `g0_usable`.
`G0_smooth` keeps exactly one role — the `MIN_SMOOTHED_G0_DB` usability gate,
which is justified: in the 9.6-12 GHz dead band 47 columns pass a raw
`row0 > 3 dB` on noise spikes and **zero** pass the smoothed gate. Print both
variants once, labelled, so the size of the old bug is on record.

#### 2. Model-side symmetry

**File**: `scripts/run_compression.py`

`p1db_output_dbm` (`:1048`) already uses the gain that defined the threshold.
Add an assertion that it equals
`p1db_input_dbm + small_signal_gain_vs_off_db - 1.0` to within 1e-9, so both
sides provably use the same estimator.

### Success criteria

**Automated**: `psat == p1db_fit + g0_local - 1.0` exactly
(`assert_allclose`, `rtol=0`).

**Manual**: measured `P_sat` median **−64.43 dBm** on the B1 axis; the mixed
form reported alongside (it differed by +0.23 dB on the old axis).

---

## Phase 4 — Operating point by band fit

### Overview

`G0(f)` is a transmission ratio — the only observable free of every line-loss
constant. Because real-device nonidealities make `I_p^device ≠ I_p^model`, both
pump frequency and pump current must be free. Two parameters fitted against a
whole curve.

### Changes

**File**: `scripts/fit_model_operating_point.py` (new)

- **Target**: measured `G0(f)` over the usable band, ripple-averaged (savgol
  envelope), pump gap excised.
- **Model**: linear Floquet `solve_gain_one` sweep — cheap, no multitone — at
  each candidate `(f_p, I_p)`.
- **Sampling constraint**: model grid ≤ **25 MHz**. Both curves ripple 3-4 dB
  peak-to-peak; the measured grid is 4 MHz and the current 172 MHz model grid
  aliases its own ripple. Sub-sample the measurement onto the model grid, never
  the reverse.
- **Objective**: weighted least squares on the ripple-averaged envelopes, weight
  ∝ measured `G0` so the amplified ridge drives the fit rather than the flat
  background. Coarse grid then refine, mirroring
  `scripts/align_map_to_measurement.py`.
- **Output**: JSON with `(f_p, I_p)`, on-chip pump dBm under Norton, peak gain,
  peak frequency, 3 dB bandwidth for both sides; PNG overlay; loss surface.

### Success criteria

**Automated**: `tests/test_fit_operating_point.py` — a synthetic target built
from the model itself at a known `(f_p, I_p)` is recovered to within the grid
step; the objective is finite and the weighting excludes the pump gap.

**Manual**: fitted model band peak within **0.3 dB and 50 MHz** of the measured
**15.87 dB @ 7.080 GHz**. The current point fails this by **2.63 dB / 351 MHz** —
that failure is the acceptance evidence.

---

## Phase 5 — Re-run the model sweep at the fitted point

### Overview

Sample densely enough to resolve ripple, bottom out the power grid, and recover
the band-edge failures that bias the model median high.

### Changes

#### 1. Sweep configuration

Driven by `scripts/run_compression.py`; no code change beyond Phase 1.

**≥ 60 signal frequencies at ≤ 50 MHz spacing over 4.5-9.6 GHz.** Current: 30
points at 172 MHz, of which 10 failed — **all at band edges**, i.e. the
lowest-gain and therefore lowest-`P_sat` points, so their loss biases the model
median high. Keep `--stop-after-p1db`.

#### 2. Small-signal floor guard

**File**: `scripts/run_compression.py`

After the sweep, assert `|G(P_min) − G(P_min + Δ)| < 0.05 dB`; on failure set
`status = "G0_GRID_NOT_FLAT"` rather than reporting a P1dB. Documented trap:
`G0` is read from the lowest solved point, so a grid that never reaches the flat
region reports P1dB too high. The current data passes (15.9736 vs 15.9311 at
7.086 GHz, Δ = 0.043 dB) but nothing enforced it.

### Success criteria

**Automated**: unit test for the flatness guard on a synthetic
already-compressed grid.

**Manual**: ≥ 90% of requested frequencies reach `VALID_SOLVED`; failures listed
with reason, never silently dropped.

---

## Phase 6 — G0 and P1dB as separate primary observables

### Overview

**`P_sat` is formed last, and never quoted alone.** Two things go wrong if the
sum is formed first.

**1. Cancellation.** Since `P_sat = P1dB + G0 − 1`, pointwise

    ΔP_sat(f) = ΔP1dB(f) + ΔG0(f)

and the two terms have had **opposite signs at every operating point examined**,
so the sum understates the underlying defect. Under the pre-correction
conventions the masking was more than 2×:

| | ΔP1dB | ΔG0 | ΔP_sat |
| --- | ---: | ---: | ---: |
| as published | −7.86 dB | +3.88 dB | **−3.57 dB** |
| after Phases 1-3 | −26.46 dB | +2.30 dB | **−22.92 dB** |

The headline `−3.57 dB` was a `−7.86 dB` P1dB defect with `+3.88 dB` of excess
model gain subtracted out of it.

**2. Medians do not add.** `median(A + B) ≠ median(A) + median(B)` unless the
pair is comonotone — visible above, where `−26.46 + 2.30 = −24.16 ≠ −22.92`.
So the decomposition must be computed **pointwise on a common frequency grid and
then summarized**, never by differencing three independently taken medians.
Every summary statistic quoted so far violates this.

### Changes

**File**: `scripts/measured_psat_pipeline.py`

#### 1. Common comparison grid

Interpolate the **measured** curves onto the **model** grid, never the reverse:
upsampling a 60-point model onto 2001 columns invents ripple structure the model
never resolved. Compare only where the model grid resolves ripple (≤ 50 MHz
after Phase 5).

Apply the `G0 > MIN_SMOOTHED_G0_DB` usability gate to **both** sides
identically. It is currently applied to the measured side only, which silently
keeps low-gain model points that have no measured counterpart.

#### 2. Three separate deliverables, in this order

| artifact | contents |
| --- | --- |
| `g0_model_vs_measured.png` | `G0(f)` both sides, raw **and** ripple-averaged envelope, plus `ΔG0(f)` |
| `p1db_model_vs_measured.png` | `P1dB(f)` both sides plus `ΔP1dB(f)` |
| `psat_decomposition.png` | stacked `ΔP1dB(f)`, `ΔG0(f)`, `ΔP_sat(f)` |
| `psat_decomposition.csv` | per-frequency columns for all six series and three deltas |

`ΔG0` is reported in **both** raw and envelope form, because raw `ΔG0` flips
sign between adjacent frequencies — 6.500 GHz has the model +0.735 dB above
measurement and 6.562 GHz has it −2.423 dB below, and a 22 MHz grid offset moves
model gain 0.5 dB.

#### 3. Cancellation metric

Report the median pointwise **masked fraction**

    1 − |ΔP_sat(f)| / |ΔP1dB(f)|

so the amount of the P1dB defect that `P_sat` hides is a printed number rather
than something a reader must reconstruct.

### Success criteria

**Automated**
- Pointwise identity `ΔP_sat == ΔP1dB + ΔG0` asserted to 1e-9 on the common
  grid (`assert_allclose`, `rtol=0`).
- Test that the gate is applied symmetrically: a model point below the gate is
  dropped from all three deltas, not just one.
- Test that interpolation runs measured→model, asserting the output grid equals
  the model grid.

**Manual**: no summary line, figure, or table anywhere in the pipeline quotes
`ΔP_sat` without `ΔP1dB` and `ΔG0` printed beside it.

---

## Phase 7 — Combined comparison, ordered by calibration dependence

### Overview

`P_sat` and absolute powers come last, after Phase 6's separate observables.
Panels ordered by how many calibration constants they rest on.

### Changes

**File**: `scripts/measured_psat_pipeline.py`

#### 1. Comparison stage

| panel | quantity | constants it depends on |
| --- | --- | --- |
| 1 | `G0(f)` overlay + envelope delta | **none** |
| 2 | gain vs `(P_in − P1dB)` — compression-curve shape | **none** |
| 3 | `P_sat − P_pump`, and depletion fraction `η = (P_sig,out + P_idl,out)/P_pump` | one per side, each self-consistent |
| 4 | absolute `P_sat(f)` | both line losses, printed in the caption |

#### 2. Energy-conservation gate

Assert `P_sat + 3 dB ≤ P_pump` on **both** sides; raise with the offending
numbers rather than plotting. This is the test that eliminated the 45.7 dB
pump-line figure.

#### 3. Model depletion cross-check

Compare the energy-accounted `η` against the solver's own
`p1db_pump_depletion_all_port_db` — **−0.0579 dB = 1.32%** at 7.086 GHz,
matching the ≈1.1% from signal+idler accounting.

Use the **all-port** field. `p1db_pump_depletion_db` (−0.538 dB) reads port 2,
which carries 9.1% of the pump; port 3 carries 90.2%. Disagreement beyond 2× is
a hard failure.

### Success criteria

**Automated**: energy gate raises on a synthetic `P_sat` above pump; depletion
cross-check raises on a 10× mismatch.

**Manual**: headline reads pump-referred — measured **−8.89 dB**, model
**−22.67 dB**, gap **13.78 dB** — with absolute dBm as supporting detail.
Depletion at compression ≈26% measured vs ≈1.3% model. That ratio, not the dBm
gap, is the physics result, and the write-up must flag that 26% depletion
producing only 1 dB of compression is itself a consistency question for the
measured side.

---

## Phase 8 — Statistics that match the data's independence

### Overview

The measured "distribution" is the histogram of a fitted curve. Stop presenting
it as a sample.

### Changes

**File**: `scripts/measured_psat_pipeline.py`

- Primary artifact is `P_sat(f)` with a residual band, not a histogram.
- Report `n_eff`, not `n`. `G0_smooth` uses a **699-column ≈ 2.80 GHz** window,
  so ~3 independent `G0` values span 8 GHz; the P1dB robust savgol smooths
  columns whose residual autocorrelation falls to 1/e in **1 column ≈ 4 MHz**,
  i.e. it is smoothing genuine spread, not correlated noise. Removing all
  smoothing moves the median only **+0.31 dB** but takes std **1.40 → 6.05** and
  skew **+0.52 → −5.61**.
- If a distribution is wanted, block bootstrap with block length ≥ the `G0`
  smoothing window; report the bootstrap CI and `n_eff`.
- Drop `kstest` against `norm`/`skewnorm` as a verdict. Both reject
  (p ≈ 1e-7 to 1e-9) on comb structure the smoothing created. Keep them only as
  a labelled artifact diagnostic.
- Each of the three Phase 6 delta series gets its own `n_eff`.
- Model side: ≈60 points after Phase 5, but each is one deterministic solve —
  quote curve and range, no parametric fit.

### Success criteria

**Automated**: `n_eff` tested against a synthetic AR(1) series with known
correlation length.

**Manual**: no figure or printed line asserts a distributional fit to `P_sat` as
a physical result.

---

## Phase 9 — Record the conventions

**Files**: `CLAUDE.md`, `docs/pump_current_conversions.tex`, memory
`pump-power-norton-6db`

- Replace `CLAUDE.md`'s "Pump-current conversion (validated 2026-07-18)" section
  with the Norton derivation, the `G`-diagonal evidence, and the depletion
  cross-check. Note explicitly that the old validation was a JC parity check
  about a factor of two in *current* and never addressed the Norton question.
- Record signal line = `loss_B1`, pump line = `loss_A10`, on-chip pump
  −55.54 dBm, and that the 72.5 dB constant was fabricated.
- Update memory `pump-power-norton-6db` from "NOT yet resolved" to resolved,
  citing the energy-conservation bound as the closing argument.
- New memory: measured saturates 8.89 dB below pump, model 22.67 dB — the
  calibration-robust residual — and that `P_sat` alone masks the P1dB defect.

---

## Testing strategy

**Project maturity**: Established Production — the solver is pinned by physics
gates and this pipeline feeds thesis figures.

### Unit tests

| file | covers |
| --- | --- |
| `tests/test_port_power.py` (new) | Norton/legacy conversions, round-trip, exact 6.0206 dB offset, netlist consistency against `G[port,port]` |
| `tests/test_loss_model.py` (extend) | `loss_B1` re-fit, frozen coefficients, RMS < 1e-4 dB |
| `tests/test_measured_psat_pipeline.py` (new) | per-column power-axis shift identity; `P_sat` identity; pointwise `ΔP_sat = ΔP1dB + ΔG0`; symmetric gating; `n_eff` on synthetic AR(1); energy gate raises |
| `tests/test_fit_operating_point.py` (new) | synthetic self-recovery of `(f_p, I_p)` |
| `tests/test_run_compression_cli.py` (extend) | legacy convention reproduces old numbers bit-for-bit; flatness guard |

**Every new gate must be shown failing under mutation before it counts.**

### Integration

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_psatfix --run-slow
```

End-to-end isolation run: reprocess the existing 20-point sweep through
Phases 1-3 **only** and confirm the model shift is exactly −6.0206 dB and the
measured medians land at −73.34 / −64.43 dBm. That separates the relabeling from
the Phase 4-5 re-run.

---

## Rollback

- Phases 1-3 are pure post-processing. `--power-convention
  legacy_traveling_wave` plus restoring `SIGNAL_LINE_LOSS_DB = 72.5` reproduces
  every published number bit-for-bit. No artifact is overwritten — new runs land
  in new directories.
- Phases 4-5 produce a new operating point and a new sweep directory;
  `outputs/exp45_2c_p1db_vs_frequency_op7p379` is untouched.
- Existing gain maps are never rewritten; the −6.0206 dB relabel is applied at
  read time via the recorded `power_convention` key, absent = legacy.
