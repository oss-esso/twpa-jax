# Implementation Plan: dielectric dissipation (tan δ) across `twpa_solver`

Status: proposed, not implemented. Written for an implement agent.

## Goal

A circuit built with a dielectric loss tangent produces a demonstrably
dissipative solve on every path — pump harmonic balance, small-signal Floquet
gain, and multitone saturation — with the attenuation verified against the
analytic `exp(-αL)` law rather than against another simulator, and with a
`tan δ = 0` build reproducing today's results bit-for-bit.

## Current state analysis

### What already works

| Component | State |
| --- | --- |
| `core/linear.py::dynamic_block` | Seven loss conventions built around `Im(C)`, including the one this plan needs |
| `signal/floquet.py` | Threads `loss_model` into every sideband block (lines 121, 152, 215, 426) |
| `multitone/problem.py` | Threads `loss_model` per tone (line 91), caches on `(loss_model, power)` |
| `builders/jc_doc.py::jc_fqjtwpa_diss` | Existing lossy design; stamps `C/(1 + i·tanδ)`, exposes `--tandelta` |
| `builders/le_gal_2025.py:143` | `conductance_from_loss_tangent(tan_delta, capacitance_f, omega_rad_s)` |
| `builders/{profiles,scatter}.py` | Lj/Cg profiles, independent Lj/Cj/Cg scatter, per-component RNG streams |
| `builders/ipm.py::build_variant_design` (line 1414) | Rebuilds `designs/ipm_2c_fixed` bit-exact under `coupler_mode="cached"` |
| `pump/backends/fast_coupled.py:214`, `multitone/schur.py:200` | Split `D` into real/imag parts — complex `D` already supported |

The storage convention is therefore already chosen by the codebase: **loss lives
in `Im(C)`**. This plan does not invent a new one.

### The sign convention, stated once

A lossy capacitor is `C_stamped = C·(1 − i·tanδ)`, so

    Im(C_stamped) = −C·tanδ

`dynamic_block`'s `conductance_abs_omega` branch (line 57) computes

    G_eff = G − |ω|·Im(C) = G + |ω|·C·tanδ   ≥ 0

which is dissipative for both signs of ω, and satisfies

    D(−ω) = conj(D(ω))

This last identity is the whole reason to prefer `conductance_abs_omega` over
the current default. It is what makes a real time-domain waveform stay real.

### Six gaps

**G-1. The pump harmonic balance ignores `loss_model` entirely.**
`pump/problem.py:159-171` hardcodes `Dk = Kc + (−wk·wk)·Cc + (1j·wk)·Gc`.
There is no `loss_model` field on `FullPumpProblem`. For pump modes alone this
is numerically equivalent to the correct model (all `k ≥ 1`, so `|ω| = ω`), but
it is equivalence by accident, not by construction, and the class carries
`use_real_capacitance` (line 117-121) which **deletes `Im(C)` outright** —
silently discarding all dissipation with no diagnostic.

**G-2. `time_residual` silently discards the loss.**
`pump/problem.py:325`: `return np.asarray(r, dtype=float)` after `self.C @ ddx_t`.
Verified on this environment (numpy 2.5.1): the cast emits `ComplexWarning` and
drops the imaginary part rather than raising. `scripts/run_gain_map.py` sets
`compute_time_residual=True`, so `pump_time_rel` becomes meaningless — and
misleadingly *good*, since the discarded term is exactly the loss.

**G-3. Same defect in the multitone power balance.**
`multitone/observables.py:154`: `(circuit.C @ acceleration.T).T` on a complex
`C`, feeding a real power accounting. The dissipated power vanishes from the
balance, which would make a lossy device look conservative.

**G-4. The signal solve's default loss model has the wrong sign at negative
sideband frequencies.** Defaults are `current_complex_c` throughout
`signal/floquet.py`, and `scripts/run_gain_map.py:1313,1326` hardcode it.
Floquet sidebands are `ω_m = ω_s + m·ω_p`, which go negative for the lower
idlers. With complex `C`, at `ω < 0` the loss term is `−ω²·(−iC·tanδ) =
+iω²C·tanδ`, i.e. **gain**. Under `current_complex_c` a lossy device would
amplify on its lower sidebands.

**G-5. No `tan δ` anywhere in `builders/ipm.py`.** No CLI flag, no per-role
stamping, nothing in `build_component_plan` or `build_matrices`.

**G-6. Cj scatter is independent and breaks the plasma frequency.**
`build_component_plan:742` makes the *nominal* Cj track the Lj profile
(`cj_nominal = params.Cj * (params.Lj / lj_nominal)`), holding `ω_J` constant
across a profile. But line 747 then draws Cj scatter on its own RNG stream, so
scatter breaks what the profile preserves. There is no plasma-locked mode.

### Element roles on 2c

A fresh `make_ipm` build tags capacitors as:

| Role | Count on 2c | Loss |
| --- | ---: | --- |
| `jj_cj` (junction capacitance) | 2508 | **excluded** |
| `jtl_cg` (JTL ground capacitance) | ~2508 | included |
| `capacitor` (TL + coupler ground) | ~3636 | included |
| `coupling_cap` (coupler `Cc`) | 760 | included |

6904 lossy capacitors, 2508 lossless. Note the *stored*
`designs/ipm_2c_fixed/ipm_elements.csv` has an empty `role` column — it predates
role tagging — but `build_variant_design` rebuilds through `make_ipm`, so role
selection operates on in-memory `Element.role`, never on the stored CSV.

## Design decision, and the alternative that was rejected

**Chosen: store loss in `Im(C)`, make it impossible to drop silently.**

Reuses `dynamic_block`'s seven validated conventions, matches the existing
`jc_fqjtwpa_diss` precedent, and keeps every backend that already splits
`D.real/D.imag` working unmodified. Its one weakness — a real-casting consumer
drops the loss without complaint — is exactly gaps G-2 and G-3, and is closed
by fixing those two sites and adding a guard that makes any future occurrence
raise.

**Rejected: a separate real `C_loss.npz` matrix and `G_eff = G + |ω|·C_loss`.**
Cleaner in isolation and immune to silent-drop by construction, but it orphans
`dynamic_block`'s loss-model machinery and its tests, adds a field to
`CircuitMatrices` that every builder and every artifact reader must learn, and
splits the loss representation across two files that can disagree. Not worth it
when the silent-drop surface is two known lines.

## What we are NOT doing

- **No series/resistive branch loss.** `tan δ` is a dielectric quantity. Adding
  a series resistance to the inductors is a separate model and a separate plan.
- **No loss on the junction capacitance `Cj`** (decision recorded above).
- **No frequency-dependent `tan δ`.** Constant with frequency; the `|ω|` factor
  in `G_eff` already gives the standard linear-in-ω dielectric conductance.
- **No change to the pump *line* attenuation model** (`loss.py`, the measured
  `loss_A10` fit). That is loss *before* the chip; this plan is loss *on* the
  chip. They compose and must not be confused.
- **No re-run of exp20–exp32.** Published compression numbers are untouched.
- **No Tier-2 complex-ω stability refinement under loss.** See "Accepted
  limitations".
- **No new external validation claim.** Per `CLAUDE.md`, JosephsonCircuits.jl is
  not a reference. The attenuation gate below is analytic and self-contained.

## Prerequisites

- [ ] `designs/ipm_2c_fixed` reproduces bit-exact:
      `build_variant_design(source, tmp, coupler_mode="cached")` gates at
      16312/16312 elements, `C/G/K/Bphi` maxdiff 0.0.
- [ ] Baseline map artifacts available for the identity gate (Phase 6).
- [ ] Full suite green before starting:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_diss_pre --run-slow`

---

## Phase 1: loss-model resolution and the silent-drop guard

### Overview

Establish one place that decides which loss convention a circuit uses, and make
every real-cast of a complex `C` either correct or loud. No behaviour change for
lossless circuits.

### Changes required

#### 1. `CircuitMatrices` learns whether it is lossy
**File**: `src/twpa_solver/core/circuit.py`

Add to the dataclass:

```python
@property
def has_loss(self) -> bool:
    """True when any stamped capacitance carries a dielectric loss tangent."""
    return bool(np.iscomplexobj(self.C.data) and np.any(self.C.data.imag != 0.0))
```

In `__post_init__`, do **not** coerce `C` to real. Log `has_loss` and, when
true, the implied `tan δ` range so it appears in every debug trace.

`load_circuit` (line 127) already round-trips complex through `sp.load_npz`;
verify with a test rather than assuming.

#### 2. Canonical loss model resolver
**File**: `src/twpa_solver/core/linear.py`

```python
LOSSY_DEFAULT_LOSS_MODEL = "conductance_abs_omega"

def default_loss_model_for(circuit: CircuitMatrices) -> str:
    """The loss convention a circuit must be solved under.

    ``conductance_abs_omega`` is the only stocked convention satisfying
    D(-w) == conj(D(w)), which a real time-domain waveform requires and which
    keeps the loss dissipative at negative sideband frequencies. Lossless
    circuits keep the historical default so no existing result moves.
    """
    return LOSSY_DEFAULT_LOSS_MODEL if circuit.has_loss else "current_complex_c"
```

#### 3. The guard
**File**: `src/twpa_solver/core/linear.py`

```python
def require_real(matrix_or_array, *, what: str):
    """Return a real view, refusing to silently discard a loss term."""
```

Raises `ValueError` naming `what` and the largest discarded imaginary magnitude
when the input is complex with any non-zero imaginary part; returns `.real`
otherwise. Every existing `astype(float)` / `dtype=float` cast of a
`C`-derived quantity routes through this.

### Success criteria

**Automated**
- `pytest tests/test_loss_conventions.py -q`
- New test asserts `D(-w) == conj(D(w))` **bit-exactly** for
  `conductance_abs_omega` on a lossy fixture, and asserts it **fails** for
  `current_complex_c` (the second half is the point — it pins gap G-4).
- New test asserts `require_real` raises on a complex input with non-zero imag,
  and is a no-op on a real one.
- New test round-trips a complex-`C` circuit through `save_circuit`/`load_circuit`
  and asserts `has_loss` survives and `C` is bitwise equal.
- Full existing suite unchanged: `--run-slow`, zero new failures.

**Manual**
- `default_loss_model_for` on `designs/ipm_2c_fixed` returns
  `"current_complex_c"` — i.e. today's behaviour is untouched.

---

## Phase 2: pump harmonic balance carries the loss

### Overview

Close G-1 and G-2. After this phase the pump solve is dissipative and its
reported residuals are honest.

### Changes required

#### 1. `FullPumpProblem` gains a loss model
**File**: `src/twpa_solver/pump/problem.py`

- Add field `loss_model: str = "current_complex_c"` (keeps every existing
  caller byte-identical).
- Rewrite `_build_linear_blocks` (line 159) to delegate:

```python
for k in self.grid.k:
    wk = float(k) * self.grid.omega
    blocks.append(dynamic_block_from_parts(
        self.C, self.G, self.K, wk, loss_model=self.loss_model).tocsc())
```

  Factor the convention switch out of `core.linear.dynamic_block` into a
  `dynamic_block_from_parts(C, G, K, omega, *, loss_model)` that
  `dynamic_block` then calls, so there is exactly one implementation. Assert in
  a test that both entry points produce bitwise-identical blocks.

- `use_real_capacitance` (line 117): when the incoming `C` is complex, log at
  `WARNING` and record `"loss_discarded_by_use_real_capacitance": True` in the
  problem metadata. It stays a legal diagnostic switch; it stops being a silent
  one.

#### 2. Fix `time_residual`
**File**: `src/twpa_solver/pump/problem.py:319-326`

The current formula assumes memoryless elements. A frequency-domain loss has no
time-domain stamp, so the linear part must be evaluated per harmonic and
synthesised back:

```python
def time_residual(self, X, source_scale):
    """Real time-domain residual.

    The linear part is evaluated per harmonic and synthesised back, rather than
    stamped as C·xddot + G·xdot + K·x, because a frequency-domain loss model
    has no memoryless time-domain equivalent. For a real C this is algebraically
    identical to the stamped form and agrees with it to machine precision.
    """
    lin = np.empty_like(X)
    for h in range(self.H):
        lin[h] = self._linear_blocks[h] @ X[h]
    r = self.grid.synthesize(lin)
    r = r + self.nonlinear_current_time(X)
    r = r - self.source_time(source_scale)
    return np.asarray(r, dtype=float)
```

`self.grid.synthesize` already applies `2·Re Σ_k (·) e^{+ikωt}`, which is the
correct real reconstruction.

#### 3. Thread it through the drivers
**Files**: `scripts/run_gain_map.py:687`, `scripts/run_compression.py:404`

Pass `loss_model=default_loss_model_for(circuit)` at both construction sites.

The Schur path needs no change: `backends/schur_partition.py:82` consumes
`linear_blocks` from the full problem, so it inherits the fix.

### Success criteria

**Automated**
- New test: on a **real-`C`** fixture, the new `time_residual` equals the old
  stamped formula to `< 1e-12` relative. This is the regression that proves the
  rewrite is faithful.
- New test: on a lossy fixture, the old formula and the new one **differ**, and
  the old one under-reports (mutation-style — it must be shown failing).
- New test: `FullPumpProblem` blocks equal `dynamic_block` blocks bitwise for
  every value in `LOSS_MODELS`.
- Existing pump/gate tests unchanged.

**Manual**
- One 2c pump point at `tan δ = 0` reproduces the stored `pump_coeff_rel` and
  `pump_time_rel` to all printed digits.

---

## Phase 3: signal, multitone, and observables

### Overview

Close G-3 and G-4.

### Changes required

#### 1. Driver defaults
**File**: `scripts/run_gain_map.py:1313,1326`

Replace the two hardcoded `loss_model="current_complex_c"` with
`default_loss_model_for(circuit)`. Add `--loss-model` (default `auto` →
resolver) so a convention can be forced for diagnostics.

#### 2. Multitone power balance
**File**: `src/twpa_solver/multitone/observables.py:154`

The acceleration term must use the same per-tone block the residual uses, not a
raw `circuit.C` product. Add an explicit dissipated-power term to the balance:

    P_diss = Σ_tones  ½·|V_tone|² · (|ω|·C·tanδ)   (summed over lossy caps)

and include it in the accounting so a lossy device balances. Route the real cast
through `require_real` so a future regression raises.

#### 3. Multitone problem default
**File**: `src/twpa_solver/multitone/problem.py:39`

`loss_model` default becomes the resolver result rather than the literal string,
resolved in `__post_init__` when the caller passes `None`. Keep the explicit
string path so the cache key (line 78) is unaffected.

### Accepted limitation, to be documented in-code

`conductance_abs_omega` is in `signal/stability.py::NON_ANALYTIC_LOSS_MODELS`
(pinned by `tests/test_floquet_stability.py:99`), because `|ω|` is not analytic
in ω. **Tier-2 complex-ω resonance refinement is therefore unavailable for lossy
circuits.** Tier-1 stability is unaffected. This is a real restriction of the
convention, not a defect introduced here, and must be stated in the docstring
and in `CLAUDE.md` rather than worked around.

### Success criteria

**Automated**
- New test: pump-off `power_balance` on a lossy fixture closes to `< 1e-9`
  relative *with* the dissipated term and **fails without it** (mutation).
- New test: `tan δ = 0` leaves every observable bitwise unchanged.
- New test: `refine_complex_resonance` on a lossy circuit raises the existing
  clear error rather than returning a wrong root.

**Manual**
- `gamma_hat_summary.csv` still reports `conj_symmetry_rel_err == 0` on a lossy
  pump solve — the pump waveform must stay real.

---

## Phase 4: the builder stamps `tan δ`

### Overview

Close G-5. Ground/substrate capacitors only; junction `Cj` stays lossless.

### Changes required

#### 1. Loss specification
**File**: `src/twpa_solver/builders/ipm.py`

```python
LOSSLESS_CAPACITOR_ROLES = frozenset({"jj_cj"})

@dataclass(frozen=True)
class LossSpec:
    """Dielectric loss tangent per capacitor role.

    ``default`` applies to every capacitor role not named in ``by_role``, except
    those in LOSSLESS_CAPACITOR_ROLES, which are never lossy: the junction
    capacitance is an AlOx barrier, not the substrate dielectric this model
    describes.
    """
    default: float = 0.0
    by_role: Mapping[str, float] = field(default_factory=dict)

    def tan_delta_for(self, role: str) -> float: ...
```

#### 2. Stamp it
**File**: `src/twpa_solver/builders/ipm.py:994`

```python
if e.kind in ("capacitor", "coupling_capacitor"):
    td = loss.tan_delta_for(e.role)
    value = float(e.value) * (1.0 - 1j * td) if td else float(e.value)
    add_stamp_2node(C_r, C_c, C_d, node_to_idx, int(e.n1), int(e.n2), value)
```

`C` is built complex only when some `td != 0`; otherwise the real dtype is
preserved exactly, so a `tan δ = 0` build is bit-identical to today's.

#### 3. Plumb through the builders
**Files**: `builders/ipm.py` — `build_matrices`, `write_outputs`,
`build_variant_design` (line 1414), `parse_args`

- `build_variant_design(..., loss: LossSpec = LossSpec())`.
- The topology gate keeps running on a **nominal** rebuild
  (`assert_source_topology`), per the rule already in `CLAUDE.md`; the loss is
  applied after gating, so a lossy variant still gates at 16312/16312.
- CLI: `--tan-delta FLOAT` and repeatable `--tan-delta-role ROLE=VALUE`.
- Summary gains a `loss` block: `default`, `by_role`, the resolved per-role
  values, `lossy_capacitor_count`, `lossless_capacitor_count`, and the total
  lossy capacitance, so an artifact is self-describing.

### Success criteria

**Automated**
- `tan δ = 0` build is bitwise identical to the current builder: `C.npz` same
  dtype, same `data`, same `indices`.
- `tan δ = 1e-3` on 2c: exactly 6904 lossy capacitors and 2508 lossless;
  `Im(C)` non-zero at ground/coupler nodes and **exactly zero** on every
  junction-capacitance entry.
- `Im(C).data == -tanδ · Re(C).data` elementwise on lossy entries.
- Per-role override test: `--tan-delta 1e-4 --tan-delta-role coupling_cap=1e-2`
  produces two distinct ratios.
- Variant topology gate passes on a lossy build.

**Manual**
- `ipm_summary.json` from a lossy build carries the `loss` block.

---

## Phase 5: plasma-locked Cj scatter

### Overview

Close G-6. Scatter `Lj`, and derive `Cj` so `ω_J = 1/√(Lj·Cj)` is constant per
cell.

### Changes required

#### 1. New scatter mode
**File**: `src/twpa_solver/builders/scatter.py`

Add `mode: str = "independent"` to `ScatterSpec`, accepting
`{"independent", "plasma_locked"}`.

#### 2. Apply it
**File**: `src/twpa_solver/builders/ipm.py::build_component_plan` (line 731)

```python
lj, lj_meta = apply_scatter(lj_nominal, lj_scatter, component_rng(seed, "Lj"))
if cj_scatter.mode == "plasma_locked":
    # Cj is the exact reciprocal of the Lj factor, so Lj*Cj -- and therefore the
    # plasma frequency -- is unchanged cell by cell. This consumes no Cj RNG
    # stream, which is why the Lj stream stays bit-identical to the
    # independent-mode draw at the same seed.
    lj_factors = lj / lj_nominal
    cj = cj_nominal / lj_factors
    cj_meta = {...}   # same key set, plus "mode" and the derived factor stats
else:
    cj, cj_meta = apply_scatter(cj_nominal, cj_scatter, component_rng(seed, "Cj"))
```

In `plasma_locked` mode the Cj *sigma* is not an independent knob; it is
determined by the Lj sigma. Passing a non-zero `cj_scatter.sigma` alongside
`mode="plasma_locked"` must **raise**, not be silently ignored.

#### 3. CLI
**File**: `builders/ipm.py::parse_args` (line 1502)

`--cj-scatter-mode {independent,plasma_locked}`, default `independent`.

### Success criteria

**Automated**
- `Lj·Cj` constant across all 2508 cells to `< 1e-15` relative, at every sigma
  in {0.01, 0.03, 0.05, 0.10}.
- The `Lj` array in `plasma_locked` mode is **bitwise identical** to the
  `independent`-mode array at the same seed and sigma — the modes differ only
  in `Cj`.
- The σ-series is nested: with a fixed seed,
  `(Lj(σ) − Lj_nom)/σ` is bitwise identical across all four sigmas, so the four
  runs are the same realisation at four amplitudes rather than four unrelated
  draws.
- `mode="plasma_locked"` with `cj_scatter.sigma > 0` raises.
- `mode="independent"` reproduces today's arrays bitwise.

---

## Phase 6: verification gates

These are the physics gates. Each must be **shown failing under mutation**
before being reported as passing.

**File**: `tests/test_dissipation_physics.py`

### Gate 1 — attenuation obeys `exp(−αL)`, over three decades

The strongest self-contained check available, and the reason this plan needs no
external reference. On a passive (pump-off) line, dielectric loss gives a field
attenuation

    α(ω) = ½ · ω · tanδ · √(L·C_lossy) ·(C_lossy/C_total)   [per unit cell]

so `ln|S21|` is **linear in tan δ** at fixed frequency and length.

Measure `|S21|` at one frequency for `tan δ ∈ {1e-5, 1e-4, 1e-3}` and assert:
- `ln|S21|` scales as `1 : 10 : 100` to better than 2% (the small-α regime;
  1e-3 on 2c should still be well inside it — verify, and drop to
  {1e-6,1e-5,1e-4} if it is not);
- `|S21| < 1` strictly, and monotone decreasing in `tan δ`;
- extrapolating to `tan δ → 0` recovers the lossless `|S21|` to `< 1e-9`.

**Mutation**: flip the sign in the `conductance_abs_omega` branch — the gate must
report gain and fail.

### Gate 2 — passivity, pump off

`Σ_i |S_i1|² ≤ 1` at every sampled frequency, strictly decreasing in `tan δ`.
This catches the negative-sideband sign error (G-4) directly.

**Mutation**: force `loss_model="current_complex_c"` — the lower sidebands
should push the sum above 1 and fail.

### Gate 3 — conjugate symmetry

`D(−ω)` bitwise equal to `conj(D(ω))` under the canonical model. Also assert it
is **violated** by `current_complex_c`, pinning why the default changed.

### Gate 4 — zero-loss identity

A `tan δ = 0` variant of `designs/ipm_2c_fixed`, run through the full gain-map
path, reproduces the baseline `gain_db` to `< 1e-10` dB at a fixed
(power, frequency) point. Guards the whole refactor against accidental drift.

### Gate 5 — loss is monotone in gain

At a fixed pump point with gain, `gain_db` decreases monotonically across
`tan δ ∈ {0, 1e-5, 1e-4, 1e-3}`. Weak, but it catches a wrong-magnitude stamp
that Gate 1 (a passive check) cannot see, because it exercises the pumped path.

### Gate 6 — power balance closes

Lossy multitone power balance closes to `< 1e-9` relative with the dissipated
term, and demonstrably fails without it.

### Success criteria

**Automated**: `pytest tests/test_dissipation_physics.py -q --run-slow`, all
gates green, every gate individually shown red under its stated mutation.

---

## Phase 7: the campaign

### Overview

Eight 20×20 gain maps on `designs/ipm_2c_fixed`. Dissipation and scatter are
**not** crossed.

### Fixed grid, all eight runs

| Parameter | Value |
| --- | --- |
| Pump frequency | 7.60 – 7.85 GHz, 20 points |
| Pump power | −26 – −16 dBm, 20 points |
| Line attenuation | **measured `loss_A10` model** (the default; do *not* pass `--attenuation-db`) |
| `--pump-current-jc-scale` | `1.0` (the validated Python/IPM conversion) |
| Scatter seed | `1` for every scattered run, so the σ-series is nested |

Note this window sits ~6 dB above the ceiling of the existing
`outputs/map_2c_scan_6p0_8p5_100x70` map (−32…−19 dBm, same attenuation model).
That is taken as intentional. The campaign summary must record the on-chip peak
current for the window corners alongside the nominal dBm, since the dBm figure
alone is meaningless without the attenuation convention.

### The eight designs

| # | Run id | `tan δ` | `Lj` σ | Cj mode |
| --- | --- | ---: | ---: | --- |
| 0 | `2c_base` | 0 | 0 | — |
| 1 | `2c_td1e5` | 1e-5 | 0 | — |
| 2 | `2c_td1e4` | 1e-4 | 0 | — |
| 3 | `2c_td1e3` | 1e-3 | 0 | — |
| 4 | `2c_sc1pct` | 0 | 0.01 | plasma_locked |
| 5 | `2c_sc3pct` | 0 | 0.03 | plasma_locked |
| 6 | `2c_sc5pct` | 0 | 0.05 | plasma_locked |
| 7 | `2c_sc10pct` | 0 | 0.10 | plasma_locked |

Run 0 is the identity check as well as the baseline: it must reproduce the
pre-change solver bitwise (Gate 4).

### Build step

```powershell
python -m twpa_solver.builders.ipm --variant-source designs/ipm_2c_fixed `
  --outdir designs/campaign_diss/2c_td1e4 --coupler-mode cached `
  --tan-delta 1e-4 --overwrite
```

```powershell
python -m twpa_solver.builders.ipm --variant-source designs/ipm_2c_fixed `
  --outdir designs/campaign_diss/2c_sc3pct --coupler-mode cached `
  --lj-scatter-sigma 0.03 --cj-scatter-mode plasma_locked `
  --scatter-seed 1 --overwrite
```

### Run step

Production engine flags, from `PRODUCTION_ENGINE_FLAGS` in
`scripts/run_gain_map_column_matrices.py`:

```powershell
python scripts/run_gain_map.py --circuit-dir designs/campaign_diss/<id> `
  --outdir outputs/campaign_diss/<id> `
  --pump-freq-min-ghz 7.60 --pump-freq-max-ghz 7.85 --n-frequency 20 `
  --pump-power-min-dbm -26 --pump-power-max-dbm -16 --n-power 20 `
  --pump-current-jc-scale 1.0 `
  --inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast `
  --pump-mode-policy positive_odd_jc --pump-mode-count 10 --nt 40 `
  --signal-detuning-mhz 100
```

### Mandatory calibration before launching

Run **one column** of `2c_td1e3` (the slowest case — loss makes the pump solve
stiffer) and record wall time and peak RSS. Only then set
`--frequency-chunk-size` and the worker count, and only then extrapolate the
eight-run budget. Do not launch 3200 points on an estimate. Per the standing
note, heavy runs launch from the user's terminal, not from an agent.

### Comparison artifact

`scripts/compare_campaign_diss.py` (new): reads the eight `map_points.csv`,
emits one CSV and one figure set:

- peak `gain_db` versus `tan δ` (log x) and versus σ;
- the shift in the peak-gain pump frequency and power — dissipation and disorder
  move the optimum, and *where* it moves is the physically interesting result;
- coverage: PASS fraction per run, since disorder is expected to cost
  convergence and that cost must be reported, not hidden;
- for the `tan δ` arm only, `gain_db` against the Gate-1 analytic `exp(−αL)`
  prediction, as a consistency check at the campaign's own operating point.

### Success criteria

**Automated**
- Eight `map_summary.json` written, each recording `tan δ`, σ, Cj mode, scatter
  seed, and attenuation model.
- Run 0 matches the pre-change baseline gain to `< 1e-10` dB.

**Manual**
- `gain_db` monotone non-increasing in `tan δ` at matched (power, frequency).
- Coverage reported per run; any run below ~80% PASS is investigated before its
  numbers are quoted.

---

## Testing strategy

**Project maturity**: Established production. The solver has published results
riding on it, so the bar is a bitwise-identity gate on the lossless path plus
independent physics gates on the lossy one.

### Unit tests

| File | Covers |
| --- | --- |
| `tests/test_loss_conventions.py` | `default_loss_model_for`, `require_real`, conjugate symmetry, complex round-trip |
| `tests/test_pump_loss.py` | Pump `loss_model` field, `time_residual` rewrite, block equality with `dynamic_block` |
| `tests/test_ipm_loss_stamping.py` | Role selection, per-role override, counts, `Im(C)` ratio, zero-loss bit-identity |
| `tests/test_component_scatter.py` (extend) | `plasma_locked` mode, `Lj·Cj` invariance, nested σ-series, the raise |
| `tests/test_dissipation_physics.py` | Gates 1–6 |

Coverage target: every new branch exercised; every physics gate mutation-verified.

### Edge cases

- `tan δ = 0` — must be bit-identical to today on every path (this is the single
  most important test in the plan).
- `tan δ` negative — must raise, not silently produce gain.
- `tan δ` very large (≥ 1) — outside the small-loss regime; warn, do not fail.
- σ = 0 with `plasma_locked` — must equal the nominal build exactly.
- Loading a legacy real-`C` artifact — `has_loss` False, everything unchanged.

### Integration

- One 2c pump point end-to-end at `tan δ = 0` reproducing stored residuals.
- One 2c gain point end-to-end at `tan δ = 1e-4` with the pump-off attenuation
  cross-checked against Gate 1.
- One `plasma_locked` variant built, gated, and solved for a single point.

---

## Rollback plan

Every phase is independently revertable, and Phases 1–5 are inert until a
`tan δ > 0` design exists:

1. **Phases 1–3** (solver) — behaviour-preserving for real `C` by construction,
   pinned by bit-identity tests. Revert = `git revert` the phase commit; no
   artifact is invalidated.
2. **Phases 4–5** (builders) — additive. Existing designs are untouched on disk;
   `tan δ = 0` and `mode="independent"` reproduce current output bitwise.
3. **Phase 7** (campaign) — outputs land under `outputs/campaign_diss/`, which is
   gitignored. Deleting the directory is a complete rollback.

The one change with reach beyond dissipation is the **loss-model default flip**
in Phase 3. It is gated on `circuit.has_loss`, so no existing lossless design
changes convention. If it must be reverted independently, revert only
`default_loss_model_for` and pass `--loss-model` explicitly.

---

## Documentation to update on completion

- `CLAUDE.md` — a dissipation section: the `Im(C)` convention and its sign, why
  `conductance_abs_omega` is canonical, the junction-`Cj` exclusion, the
  `plasma_locked` scatter mode, and the Tier-2 stability limitation.
- `docs/circuit_builders.md` — the `--tan-delta` flags and the role table.
- `thesis/chapters/03_circuit_model.tex` — the loss stamp, in the section that
  already covers loss conventions.
- `thesis/chapters/08_validation_and_limits.tex` — the attenuation gate, as a
  genuinely self-contained physical check, alongside the standing note that
  saturation still has no external reference.

## Open hypothesis, recorded but not claimed

`jc_fqjtwpa_diss` is the one design of seven that has never matched
(~0.89 dB), and it is the only lossy one. It currently solves under
`current_complex_c`, which gap G-4 shows has the wrong sign at negative
sideband frequencies. Re-running it under `conductance_abs_omega` after Phase 3
is a cheap diagnostic. **This is a hypothesis, not a prediction**, and per
`CLAUDE.md` a JosephsonCircuits.jl comparison is a drift check, not validation —
so the outcome must not be reported as validating the loss model either way.
