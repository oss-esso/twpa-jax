# Intermodulation distortion (IMD) as a compression-sweep output — implementation plan

Status: implemented; G3/G5 validation measured 2026-08-13. G3a was retracted as
vacuous and replaced by a selectivity test. G0 is basis-converged; P1dB is NOT
converged at order 15 and its apparent plateau is a refinement-resolution
artifact -- see the G5 section at the end before quoting any P1dB number.

Audience: an implementing agent with no prior context on this task. Read the
whole file before editing anything. Every number under "Verified inputs" was
measured during planning on this machine, not assumed; re-measuring is cheap and
encouraged, inventing replacements is not.

---

## Goal

`scripts/run_compression.py` emits, per signal-power point, the power of each
retained intermodulation product in dBc relative to the signal fundamental, so
that a compression sweep shows at which signal power each IM order rises out of
the floor.

---

## A previous attempt failed. Do not repeat it.

An FFT post-process was implemented and reverted on 2026-08-13. It added
`compute_imd_spectrum` to `multitone/observables.py`, which called
`np.fft.rfft` on `basis.synthesize(...)` output and read IM amplitudes out of
1-D frequency bins.

It was wrong at the level of the data structure, not the arithmetic:

- **`MultiToneBasis` is a 2-D torus, not a time series.** Tones are `(h, q)`
  with `ω = h·ω_p + q·δ`. `synthesize()` (`multitone/basis.py:195-202`) returns
  `ifft2` over an `(n_p, n_delta)` grid, raster-flattened to
  `(n_p·n_delta, n_nodes)`. There is no 1-D time axis to transform. A 1-D
  `rfft` over that raster has no relationship to frequency.
- **Measured consequence:** holding the physical state fixed and scaling
  `n_p`/`n_delta` by 1×/2×/3× moved the reported values by **283–329 dB**.
  Values were either the double-precision noise floor (≈ −290 dBc) or
  impossible positives (+25 dBc, an IM product above the signal).
- **The products were not in the state to begin with.** See "Verified inputs".

Two further defects in that attempt, listed so they are not reintroduced when
writing the replacement: it dropped the DC branch flux and the linear current
term that `_port_current_coefficients` applies, and it applied
`REAL_RECONSTRUCTION_FACTOR` to voltage but not to current. **The replacement
must not hand-roll any port-wave arithmetic.** Use `extract_port_waves`, which
already handles all three correctly.

---

## Verified inputs

### 1. Every IM product is exactly a tone index

For signal `(1,-1)` and pump `(1,0)`:

```
m·signal − n·pump  ≡  ToneIndex(h = m−n, q = −m)
```

Verified for all `m,n` up to order 9; direct frequency and tone-index frequency
agree to `< 1e-9` GHz. `order = m + n`, odd orders only (3, 5, 7, 9).

### 2. Four of the twenty products are already in the production basis

Raw tones at negative frequency fold to their conjugate via
`basis.canonicalize`. The `m=1` family folds onto tones the matched basis
already retains — these are signal Floquet sidebands, not new physics:

| product | raw `(h,q)` | canonical | note |
| --- | --- | --- | --- |
| IM3 m=1,n=2 | (−1,−1) | **(1, 1)** | this is the idler |
| IM5 m=1,n=4 | (−3,−1) | **(3, 1)** | already retained |
| IM7 m=1,n=6 | (−5,−1) | **(5, 1)** | already retained |
| IM9 m=1,n=8 | (−7,−1) | **(7, 1)** | already retained |

The genuinely new physics is the `m ≥ 2` family. Sixteen distinct new tones are
needed to cover orders 3–9:

```
(1,-5) (1,-4) (1,-3) (1,-2) (1,2) (1,3) (1,4)
(3,-6) (3,-5) (3,-4) (3,2) (3,3)
(5,-7) (5,-6) (5,2)
(7,-8)
```

Because the readout is a **power** (`|b|²`), conjugation is irrelevant to the
reported dBc. No sign or conjugate bookkeeping is required.

### 3. The production basis cannot represent them at all

`scripts/run_compression.py` defaults to `--multitone-basis matched`.
`build_sideband_matched_basis` retains **`q ∈ {−1, 0, +1}` at every sideband
count** — raising `S` adds pump harmonics (`h`), never signal orders (`q`):

| matched `S` | n_tones | q-range | genuine IM retained |
| ---: | ---: | --- | --- |
| 2 | 15 | −1..1 | 1/20 |
| 6 | 23 | −1..1 | 3/20 |
| 10 | 31 | −1..1 | 4/20 |

The 4 "retained" are exactly the `m=1` family from §2. **Zero genuine IM
products exist in a production state vector**, which is why no post-process can
recover them.

### 4. The full lattice is the wrong instrument — use a targeted tone set

`build_lattice_basis(pump_modes, signal_order_max=Q, ...)` gives `q ∈ [−Q, Q]`,
but as a full outer product with `pump_modes`. Measured on 2c
(`pump_modes=[1,3,…,19]`, 6136 nodes, `fast_coupled_footprint`):

| lattice | n_tones | IM retained | peak (pardiso) | peak (banded) |
| --- | ---: | ---: | ---: | ---: |
| Q=1 | 30 | 4/20 | 5.40 GB | 3.89 GB |
| Q=2 | 50 | 8/20 | 14.65 GB | 10.45 GB |
| Q=3 | 70 | 11/20 | 28.51 GB | 20.29 GB |

Adding **only the wanted tones** to the matched `S=10` basis is far cheaper for
strictly more products:

| tier | new tones | n_tones | n_delta | peak (pardiso) | peak (banded) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline matched S=10 | — | 31 | 6 | 5.76 GB | 4.14 GB |
| **IM3** | +1 | 32 | 10 | 6.12 GB | **4.40 GB** |
| **IM3+IM5** | +4 | 35 | 18 | 7.28 GB | **5.22 GB** |
| IM3+IM5+IM7 | +9 | 40 | 26 | 9.45 GB | 6.76 GB |
| IM3…IM9 | +16 | 47 | 34 | 12.97 GB | 9.26 GB |

Full lattice Q=2 costs 14.65 GB for 8 products; targeted IM3+IM5 costs 7.28 GB
for the same 8. Use the targeted set.

### 5. Consequence for tier selection on a 7 GB machine

Only **IM3** and **IM3+IM5** fit with `--factor-backend banded`. IM3+5+7 is
marginal at 6.76 GB (no room for a second worker). **IM9 at the production basis
does not fit** — it needs ≈ 9.3 GB banded.

The original request was "IM9, as many components as computationally decent".
The measurement above is what "decent" means quantitatively. Implement the tier
knob so all four are selectable; expect to *run* IM3 or IM3+IM5 on this machine,
and treat IM7/IM9 as available only on a larger host or a smaller device
(`jpa`/`jtwpa` fixtures are far cheaper than 2c).

`n_delta` grows from 6 to 34 across the tiers. That is torus-FFT cost inside
`synthesize`/`project`, not matrix cost; it does not enter
`fast_coupled_footprint`. Watch wall time, not just RAM.

### 6. The Phase 1 and Phase 2 code below was executed during planning

The function bodies in this document are not sketches. They were run verbatim
against `build_jpa` + a matched `S=10` basis before this plan was written:

| check | result |
| --- | --- |
| `enumerate_im_products(9, …)` | 20 products |
| `required_new_tones` at order 9 / order 3 | 16 / 1 tone |
| `raw.omega` vs `m·ω_signal − n·ω_pump` | max error `3.05e-05` rad/s (float32-level, on ~4.4e10 rad/s) |
| every product tone positive-frequency | true |
| `conjugated` flag vs sign of raw frequency | exact |
| extended `n_delta`, order 3 / order 9 | 10 / 34 (matches §4) |
| empty product list | returns the same object |
| zero state | NaN for every product |
| signal tone against itself | `0.0` dBc exactly |
| tone absent from basis | NaN |
| **G2 grid independence, `n_p`/`n_delta` at 1× vs 2×** | **`0.000e+00` dB** |

That last row is the whole point. The reverted implementation moved
283–329 dB under the same perturbation; this readout does not move at all,
because it reads a solved tone amplitude instead of resampling a waveform.

Two caveats on that run: it used a random `X_full`, not a converged state, so it
validates plumbing and grid-invariance only — not physics. Physics is Phase 4's
job, and G4 is the gate that decides it.

---

## Design

Three pieces. Nothing hand-rolls port arithmetic; nothing post-processes a
waveform.

1. **`multitone/imd.py`** (new) — pure tone bookkeeping: enumerate IM products,
   map to `ToneIndex`, canonicalize, dedupe.
2. **`multitone/observables.py`** — one readout function that calls the existing
   `extract_port_waves` and normalizes to the signal tone.
3. **`scripts/run_compression.py`** — a `--imd-max-order` flag that extends the
   basis and threads the columns into the CSV.

Default **off** (`--imd-max-order 0`), so every existing run stays
byte-identical. This matches the repo convention that new layers are opt-in and
defaults reproduce legacy behaviour.

---

## Phase 1 — tone bookkeeping (`src/twpa_solver/multitone/imd.py`)

New file. Pure functions, no circuit, no solver, fully unit-testable.

```python
"""Intermodulation product tone bookkeeping.

An IM product ``m*signal - n*pump`` of order ``m+n`` is exactly the tone
``ToneIndex(h=m-n, q=-m)``.  Products at negative physical frequency fold onto
their positive-frequency conjugate; because IMD is reported as a power ratio,
that fold carries no sign bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex, canonicalize


@dataclass(frozen=True)
class ImProduct:
    """One intermodulation product and the tone that carries it."""

    order: int          # m + n, odd
    m: int              # signal multiplicity
    n: int              # pump multiplicity
    raw: ToneIndex      # (m-n, -m), may be negative-frequency
    tone: ToneIndex     # positive-frequency representative
    conjugated: bool    # True if `tone` is the conjugate of `raw`

    @property
    def label(self) -> str:
        return f"imd_o{self.order}_m{self.m}n{self.n}"


def enumerate_im_products(
    max_order: int, omega_p: float, delta: float
) -> list[ImProduct]:
    """Return every IM product of odd order 3..max_order, positive-frequency.

    Products whose frequency is exactly zero are skipped: `canonicalize` rejects
    them and they are not observable as a distinct tone.
    """
    if max_order < 3 or max_order % 2 == 0:
        raise ValueError(f"max_order must be an odd integer >= 3, got {max_order}")
    products: list[ImProduct] = []
    for order in range(3, max_order + 1, 2):
        for m in range(1, order):
            n = order - m
            raw = ToneIndex(m - n, -m)
            try:
                tone, conjugated = canonicalize(raw, omega_p, delta)
            except ValueError:
                continue  # exactly-zero frequency; not an observable tone
            products.append(ImProduct(order, m, n, raw, tone, conjugated))
    return products


def required_new_tones(
    products: list[ImProduct], basis: MultiToneBasis
) -> list[ToneIndex]:
    """Return the tones a basis must gain to represent every product.

    Deduplicated and sorted, excluding tones already present.  A tone whose
    conjugate is already retained is NOT added: `MultiToneBasis` forbids both,
    and the conjugate already carries the same power.
    """
    have = set(basis.tones)
    conj = {t.conjugate() for t in basis.tones}
    wanted = {p.tone for p in products if p.tone not in have and p.tone not in conj}
    return sorted(wanted, key=lambda t: (t.h, t.q))


def extend_basis_with_im_tones(
    basis: MultiToneBasis, products: list[ImProduct]
) -> MultiToneBasis:
    """Return a copy of `basis` extended to carry every product's tone.

    Returns `basis` unchanged when nothing new is required.
    """
    new = required_new_tones(products, basis)
    if not new:
        return basis
    return MultiToneBasis(
        tones=list(basis.tones) + new,
        omega_p=basis.omega_p,
        delta=basis.delta,
        pump_tone_index=basis.pump_tone_index,
    )
```

Note `n_p`/`n_delta` are deliberately **not** carried over — `__post_init__`
re-derives the alias-safe minimum for the enlarged tone set. Copying the old
`n_delta` would under-size the torus and silently alias.

### Phase 1 tests — `tests/test_imd_tones.py`

Mandatory. The reverted attempt shipped zero tests; that is what let a 300 dB
artifact reach a CSV.

- `enumerate_im_products(9, ...)` returns 20 products; every `raw` equals
  `ToneIndex(m-n, -m)`.
- For each product, `raw.omega(ω_p, δ)` equals `m·ω_signal − n·ω_pump` to
  `< 1e-9` relative.
- Every returned `tone` has strictly positive frequency.
- `conjugated` is `True` exactly when `raw` had negative frequency.
- On a matched `S=10` basis with `pump_modes=[1,3,…,19]`: `required_new_tones`
  returns exactly 16 tones for `max_order=9`, and 1 tone for `max_order=3`.
- `extend_basis_with_im_tones` returns a basis where `index_of(p.tone)` succeeds
  for every product, and is a no-op (identity) when nothing is new.
- The extended basis has `n_delta` **greater than or equal to** the original;
  assert the specific measured values from §4 for `max_order=3` and `9`.
- `enumerate_im_products` rejects even and `< 3` `max_order`.

---

## Phase 2 — readout (`src/twpa_solver/multitone/observables.py`)

Append one function. It calls `extract_port_waves` and nothing else — that
helper already applies `dc_branch_flux`, the linear + nonlinear current, the
`REAL_RECONSTRUCTION_FACTOR`, and the repo's `port_waves()` convention.

```python
def imd_products_dbc(
    X_full: np.ndarray,
    basis: MultiToneBasis,
    circuit: CircuitMatrices,
    products: list["ImProduct"],
    *,
    out_port: int,
    z0_ohm: float = 50.0,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, float]:
    """Return each IM product's outgoing power in dBc against the signal tone.

    Products whose tone is absent from `basis` yield NaN rather than a fabricated
    value: the basis, not this function, decides what is observable.
    """
    waves = extract_port_waves(
        X_full, basis, circuit, ports=[int(out_port)],
        z0_ohm=z0_ohm, dc_branch_flux=dc_branch_flux,
    )
    b_power = waves["b_power"]
    signal_power = b_power.get((basis.signal_tone, int(out_port)))
    results: dict[str, float] = {}
    for product in products:
        key = f"{product.label}_dbc"
        power = b_power.get((product.tone, int(out_port)))
        if not signal_power or power is None or power <= 0.0:
            results[key] = float("nan")
            continue
        results[key] = float(10.0 * math.log10(power / signal_power))
    return results
```

`math` is already imported in this module. Import `ImProduct` under
`TYPE_CHECKING` to avoid a runtime import cycle
(`imd.py` imports from `basis.py`; `observables.py` imports from both).

### Phase 2 tests — `tests/test_imd_observables.py`

- **Signal tone against itself is 0 dBc.** Construct a product list containing a
  synthetic `ImProduct` whose `tone` is `basis.signal_tone`; assert the result is
  `0.0` to `< 1e-12`.
- **Absent tone yields NaN**, not a number: pass a product whose tone is not in
  the basis.
- **Zero state yields NaN** for every product (guards the divide).
- **Grid independence (the gate that v1 failed).** Same physical `X_full`, same
  tone list, two bases differing only in `n_p`/`n_delta` (2× the auto-derived
  minimum). Require agreement `< 0.01 dB` on every product. Mutation-verify this
  test by temporarily reintroducing an `rfft`-style readout and confirming it
  fails.
- **Conjugation invariance:** a product folded to its conjugate reports the same
  dBc as one that was not folded, for a state built to make them equal.

---

## Phase 3 — driver wiring (`scripts/run_compression.py`)

### 3a. CLI

```python
parser.add_argument(
    "--imd-max-order",
    type=int,
    default=0,
    help=(
        "Emit intermodulation products up to this odd order (3, 5, 7, 9, 11, 13, 15). "
        "0 disables IMD and leaves the basis untouched. Extends the multitone "
        "basis, which raises memory and wall time -- see "
        "docs/development/imd_intermodulation_output_plan.md for measured cost."
    ),
)
```

Validate in the same place the other mutually-exclusive argument checks live:
reject even values and values `> 9`; `0` and odd `3..9` are the legal set.

### 3b. Basis extension

In `_build_multitone_basis`, **after** the existing pump-mode representation
guard (which must keep passing — it only inspects `q == 0` tones, and IM tones
all have `q ≠ 0`, so extending cannot trip it):

```python
    if getattr(args, "imd_max_order", 0):
        products = enumerate_im_products(args.imd_max_order, omega_p, delta)
        basis = extend_basis_with_im_tones(basis, products)
    return basis
```

Log the result at INFO: requested order, tones added, resulting `n_tones`,
`n_p`, `n_delta`. A silent basis change is the failure mode this whole document
exists to prevent.

### 3c. Per-point readout

`measure_state` already receives `state_full`. Add, alongside the existing
`power_balance` call:

```python
        imd = (
            imd_products_dbc(
                state_full, basis, circuit, im_products,
                out_port=out_port,
                z0_ohm=args.z0_ohm,
                dc_branch_flux=dc_branch_flux,
            )
            if im_products
            else {}
        )
```

where `im_products` is computed **once** outside the sweep loop (it is pure tone
bookkeeping; recomputing per point is waste). Merge `**imd` into the returned
dict.

Use the local `out_port` and `dc_branch_flux` variables, not `args.out_port` —
`args.out_port` is `None` by default and the resolved value is computed at
`run_compression.py:738`.

### 3d. CSV columns — both paths must agree

`write_compression_outputs` takes the **union** of keys across points. The
converged and failed branches must therefore emit the **same** key set, or
converged rows end up blank in columns that failed rows populate.

Build one canonical NaN dict once:

```python
    imd_nan = {f"{p.label}_dbc": float("nan") for p in im_products}
```

Use it in the `else` (solver-failed) branch, and merge the IM keys into the
`points.append({...})` dict from `metrics` in the converged branch. Assert in a
test that both branches produce identical key sets.

### 3e. Summary metadata

Add to the summary JSON: `imd_max_order`, the list of product labels, each
product's `(h, q)` and frequency in Hz, and the tones added to the basis. A
consumer must be able to tell which physical frequency each column refers to
without re-deriving it.

### Phase 3 tests — extend `tests/test_run_compression_cli.py`

- `--imd-max-order 0` (default) leaves `_build_multitone_basis` output identical
  to the current basis — same tones, same `n_p`, same `n_delta`.
- `--imd-max-order 3` adds exactly one tone on a matched `S=10` 2c-like basis.
- Even / out-of-range values are rejected with a clear message.
- Converged and failed points emit identical IM key sets.
- The pump-mode truncation guard still raises when it should, with IMD enabled.

---

## Phase 4 — physical validation gates

Phases 1–3 make numbers appear. This phase decides whether to believe them.
**Do not report any IMD number, and do not add anything to `CLAUDE.md`, until
G1–G4 pass and their outputs are recorded.**

Run on the `jpa` fixture first (cheap, seconds per point), then 2c.

### G1 — basis round-trip
Every requested product's tone is in the basis and `index_of` resolves. Failure
means Phase 1 is wrong.

### G2 — grid independence
Re-run one operating point with `n_p`, `n_delta` at 1× and 2× the auto-derived
minimum. **Require every product to agree within 0.01 dB.**

This is the gate the reverted implementation failed by 283–329 dB. It is the
single most important check in this document. If it fails, stop; do not tune
around it.

### G3 — pump-referenced zero-signal null and per-product floor

> **Corrected 2026-08-13.** The original dBc null criterion was ill-posed:
> dBc is `P_IM / P_signal`, so the signal-off state evaluates `0/0` and
> `imd_products_dbc` correctly returns NaN. The null test is therefore
> pump-referenced, while the dBc floor is measured only at nonzero signal.

#### G3a — pump-only null

Use the converged `pump_only_state_full` (pump on, signal current zero). Call
`extract_port_waves` directly and read `b_power` at every requested IM product
tone and at the pump tone. Report

```
10*log10(P_IM / P_pump)
```

for every product. Every product must be at the numerical floor. Record the
actual worst value; an appreciable finite value indicates pump content is being
misidentified as intermodulation and is a readout defect.

#### G3b — per-product dBc floor

For each product, sweep signal power downward and find the first local
three-point slope window, starting from high signal power, where the slope of
`log10(P_IM)` against `log10(P_signal)` differs from `m` by more than `0.35`.
Report the departure window, its centre power, dBc, and absolute `b_power`.
This is the dynamic-range bound below which that product must not be quoted.
Products with no departure in the tested window must be reported as such,
including the tested lower bound.

### G4 — slope equals `m` (the physical signature)

> **Corrected 2026-08-13 after measurement. The original form of this gate was
> wrong and failed a correct implementation.** It required slope 3 for IM3,
> importing the two-tone convention where *both* tones are swept together. This
> is a **fixed-pump** sweep: only the signal moves.

An IM product has amplitude `∝ A_signal^m · A_pump^n`. The pump is held at the
operating point, so `A_pump^n` is constant and:

```
slope of absolute IM power vs signal power  =  m        (not the order)
slope in dBc (ratio to signal)              =  m − 1
```

The order `m+n` does not enter. Sweep signal power, fit `log10(P_IM)` against
`log10(P_signal)`, and require **slope within ±0.35 of `m`** with `R² ≥ 0.98`
over at least 3 points inside a valid window.

This is a far stronger gate than the original: it makes a *different* prediction
for every product, and they must all line up.

**Measured on `jpa`** (pump 4.75001 GHz, `Ic`-normalized pump 1.13e-08 A,
`--imd-max-order 7`, `outputs/imd_slope_order7{,_hi}`):

| m | products | measured slope | R² |
| ---: | --- | --- | ---: |
| 1 | o3_m1n2, o5_m1n4, o7_m1n6 | 1.0000, 1.0001, 1.0002 | 1.00000 |
| 2 | o3_m2n1, o5_m2n3, o7_m2n5 | 2.0005, 2.0006, 2.0007 | 1.00000 |
| 3 | o5_m3n2, o7_m3n4 | 3.0003, 3.0010 | 1.00000 |
| 4 | o5_m4n1, o7_m4n3 | 4.1346, 4.1307 | 0.99986 |
| 5 | o7_m5n2 | 5.1745 | 0.99985 |

Eleven products, five distinct predictions, all met.

**Free sanity check:** the `m=1` family is the idler. Its dBc slope is `0` — a
constant ratio to the signal, exactly what a phase-insensitive parametric
amplifier does. Measured `0.0007`. If this is not flat, the readout is wrong
before any higher product is worth reading.

#### The valid window is bounded on both sides

A product is only fittable where it is **above the numerical floor** and
**below compression**. High `m` needs high signal power to clear the floor,
because `P_IM ∝ P_signal^m` while the floor is flat.

Measured, same device: at signal `−216..−168 dBm`, `m=1,2,3` fit perfectly but
`m=4,5,6` return 3.74 / 3.21 / 1.12 with R² decaying 0.993 → 0.887 → 0.453 —
the signature of a fit running into the floor, *not* a defect. Moving to
`−168..−156 dBm` brings `m=4,5` onto their true slopes. `m=6` was still floored
at that window and is not validated here.

So: **do not report a slope without stating the window and the floor.** A
failing high-`m` slope is a dynamic-range statement about the sweep, not
evidence against the implementation. Diagnose by checking whether R² is also
collapsing and whether the low-power points are flat — both indicate the floor.

The slight overshoot at `m=4,5` (4.13, 5.17 rather than 4.00, 5.00) is gain
drift inside the window: gain moved 7.44 → 8.09 dB across those three points.
Tighten the window if a sharper number is needed.

### G5 — production quantities unchanged
Extending the basis must not move the physics that was already validated. Run
the same operating point with and without `--imd-max-order`, and compare `G0`
and `P1dB`. A shift larger than the 2c basis-convergence figure already on
record (`< 1e-6` dB between S=10/12/14, from `exp54`) means the added tones are
perturbing the solve, and the tier is not free.

Record the measured deltas. If they are non-trivial, that is a real finding
about basis truncation and belongs in `CLAUDE.md` — but it also means IMD costs
more than "an extra column".

---

## Phase 5 — plotting (only after Phase 4 passes)

`scripts/plot_imd_spectrum.py`: IM dBc against signal power, one line per
product, with horizontal reference lines at **−30 and −40 dBc**.

- Read the column set from the CSV header (`imd_o*_dbc`); do not hardcode.
- Draw the G3 numerical floor as a shaded band. A curve inside that band is not
  a measurement, and the plot must show that.
- Group by order (colour) and distinguish `(m,n)` (linestyle); 20 flat colours
  is unreadable.
- `pandas` and `matplotlib` are already declared dependencies.

Onset is read off the plot against the two threshold lines. Do not compute an
"onset power" scalar: with a floor-limited low end and a compression-limited
high end, a threshold crossing is only meaningful once you can see both.

---

## Traps

- **Do not reuse `--multitone-sidebands` as the IM knob.** It is passed as
  `signal_order_max` on the `lattice` path only, and as sideband count on
  `matched`. Overloading it a third time will silently change existing runs.
- **Do not copy `n_p`/`n_delta` when extending a basis.** Let `__post_init__`
  re-derive them or the enlarged tone set aliases.
- **Do not add a tone whose conjugate is retained.** `MultiToneBasis.__post_init__`
  rejects it. `required_new_tones` handles this; keep that behaviour.
- **Do not hand-roll `b = (V/Z0 − I)/2`.** `CLAUDE.md` documents an unresolved
  disagreement between the Norton KCL subtraction in `extract_port_waves` and
  the travelling-wave convention in `ports.py`. Adding a third convention makes
  that worse. Going through `extract_port_waves` means IMD inherits whatever is
  decided there. Note in the summary JSON that dBc is a **ratio at one port**
  and is therefore invariant under the source-power convention — that is a real
  advantage of reporting dBc rather than absolute dBm, and it should be stated.
- **Do not claim IM9 was measured on this machine** without showing the memory
  headroom it ran in. Per §5 it does not fit at the production basis.

---

## Definition of done

- [ ] `multitone/imd.py` added; `tests/test_imd_tones.py` passing.
- [ ] `imd_products_dbc` added; `tests/test_imd_observables.py` passing,
      including the grid-independence test, mutation-verified.
- [ ] `--imd-max-order` wired; CLI tests passing; default `0` leaves existing
      behaviour byte-identical.
- [ ] G1–G5 run and their outputs recorded under `outputs/` with the commands
      that produced them.
- [ ] G4 slope figure produced; measured slope, R², **and the fit window** written
      down per product, against the prediction `slope = m`.
- [ ] Plot script added.
- [ ] `CLAUDE.md` updated with: the tier cost table, the G3 floor, the G4
      measured slopes, and the G5 deltas — stated as measurements with their
      operating point, not as capabilities.

Running without `--run-slow` is not complete validation.

---

## Validation results — G3 and G5 (2026-08-13)

All measurements below use the JPA fixture at
`pump_freq=4.75001 GHz`, `signal_freq=4.75 GHz`, `pump_current=1.13e-08 A`,
`out_port=1`, zero attenuation, and the `legacy_traveling_wave` convention.
The 2c fixture was not run: `designs/ipm_2c_fixed` is absent from this
working tree (`C.npz` and `ipm_summary.json` missing), so 2c validation is
deferred pending those data.

### G3a result

Command: `outputs/imd_g3a_order7/commands.txt`.

The converged pump-only state has `P_pump = 1.5960743672155426e-15 W` at the
output port. Every order-3/5/7 product has extracted `P_IM = 0.0 W`, hence
`10*log10(P_IM/P_pump) = -inf dBc` for every product. This is an exact zero in
the saved double-precision state, not a substituted finite floor. The worst
finite value is therefore not defined.

This is structural-zero evidence, not a valid selectivity or dynamic-range
gate: the pump-only solution has no signal-sector content, so the result is
guaranteed by the block structure of the solved state. A pump-scaling gate was
tried and rejected for the same reason; scaling the pump cannot distinguish a
correct IMD readout from a readout that returns zero in every signal sector.
G3a is therefore **retracted as a physical validation gate**. The selectivity
unit test is the applicable regression guard; G3b remains the measured
nonzero-signal floor bound.

### G3b result

Command: `outputs/imd_g3b_order7/commands.txt`. The sweep was 21 points from
`-216` to `-156 dBm` in 3 dB steps. Local slopes use three adjacent points,
6 dB windows, and are scanned from high signal power downward. `b_power` is
the outgoing power at port 1.

| product | m | departure window (dBm) | centre (dBm) | local slope | dBc at centre | b_power at centre (W) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| o3_m1n2 | 1 | none in [-216,-156] | — | — | -0.2078 at -216 | 2.5611e-24 at -216 |
| o3_m2n1 | 2 | none in [-216,-156] | — | — | -78.2068 at -216 | 4.0599e-32 at -216 |
| o5_m1n4 | 1 | none in [-216,-156] | — | — | -39.1534 at -216 | 3.2648e-28 at -216 |
| o5_m2n3 | 2 | none in [-216,-156] | — | — | -78.8344 at -216 | 3.5137e-32 at -216 |
| o5_m3n2 | 3 | none in [-216,-156] | — | — | -153.9689 at -216 | 1.0772e-39 at -216 |
| o5_m4n1 | 4 | [-210,-204] | -207 | 3.6401 | -239.4562 | 2.4187e-47 |
| o7_m1n6 | 1 | none in [-216,-156] | — | — | -88.6905 at -216 | 3.6321e-33 at -216 |
| o7_m2n5 | 2 | none in [-216,-156] | — | — | -116.2901 at -216 | 6.3124e-36 at -216 |
| o7_m3n4 | 3 | none in [-216,-156] | — | — | -154.4680 at -216 | 9.6028e-40 at -216 |
| o7_m4n3 | 4 | [-213,-207] | -210 | 1.8583 | -211.9698 | 6.7955e-45 |
| o7_m5n2 | 5 | [-162,-156] | -159 | 5.3532 | -119.4565 | 1.6175e-30 |
| o7_m6n1 | 6 | [-162,-156] | -159 | 6.4364 | -185.1863 | 4.3237e-37 |

The `m=6` result is recorded but is not treated as a validated physical slope;
the requested operating-point notes identify that order as still floor-limited
in the prior window.

### G5 result

Command: `outputs/imd_g5_order{0,3,5,7}/commands.txt`; each run used 9 points,
`signal_current=1e-12..3e-08 A`, and `p1db_power_tol_db=0.1`.

| max order | G0 (dB) | delta G0 (dB) | p1db (dBm) | delta p1db (dB) | p1db_interpolated (dBm) | delta interpolated (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.402509879 | 0 | -132.405254 | 0 | -132.754650 | 0 |
| 3 | 7.402505081 | -4.79798e-06 | -135.378342 | -2.97309 | -135.915485 | -3.16083 |
| 5 | 7.400366947 | -2.14293e-03 | -133.279692 | -0.874438 | -133.337797 | -0.583146 |
| 7 | 7.400366922 | -2.14296e-03 | -133.104804 | -0.699550 | -133.513552 | -0.758901 |

The P1dB deltas shrink after order 3, but the G0 delta grows from order 3 to
orders 5/7 and remains about `2.14e-3 dB`; the P1dB shifts are also much larger
than the previously cited `1e-6 dB` convergence figure. Therefore G5 is not a
clean basis-invariance pass. This is a real basis-truncation finding, not noise
and not tuned away. `CLAUDE.md` was intentionally not modified.

### Extended G5 result: orders 9–15

The CLI validation tiers were extended to orders 11, 13, and 15 for this
measurement. The same JPA operating point and sweep were used at every tier.

| max order | tones | n_delta | G0 (dB) | delta G0 (dB) | P1dB (dBm) | delta P1dB (dB) | interpolated P1dB (dBm) | delta interpolated (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 34 | 34 | 7.400366921517 | -2.14295755e-3 | -132.667585284 | -0.262331 | -132.845983876 | -0.091333 |
| 11 | 44 | 42 | 7.400366921760 | -2.14295731e-3 | -132.667585284 | -0.262331 | -133.035289741 | -0.280639 |
| 13 | 56 | 50 | 7.400366921428 | -2.14295764e-3 | -132.492697734 | -0.087444 | -132.739075936 | +0.015575 |
| 15 | 70 | 58 | 7.400366921698 | -2.14295737e-3 | -132.492697734 | -0.087444 | -132.868438699 | -0.113788 |

**The `p1db` "plateau" is a refinement-resolution artifact. Do not quote
`-132.492697734 dBm` as a converged value.**

The sweep grid is 9 points spanning `-196.0206 .. -106.4782 dBm`, so the
bracket is `11.1928 dB` wide. The bracketed refinement bisects it six times:
`11.1928 / 2^6 = 0.1749 dB`. **Every reported `p1db` is quantized to that
rung.** The signature is unmistakable in the series above — orders 9 and 11 are
bit-identical, orders 13 and 15 are bit-identical, and the only nonzero steps
are exactly `+0.1749` (which also appears at 5→7). Different bases cannot
produce values agreeing to twelve significant figures; the number is a bisection
endpoint, not physics.

`p1db_interpolated_dbm` is log-linear and therefore **not** quantized, which
makes it the honest convergence probe. Its steps:

| step | 0→3 | 3→5 | 5→7 | 7→9 | 9→11 | 11→13 | 13→15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Δ (dB) | −3.161 | +2.578 | −0.176 | +0.668 | −0.189 | +0.296 | −0.129 |

That is a **decaying oscillation, not a plateau** — sign alternates every step
and the envelope (0.668, 0.189, 0.296, 0.129) is not monotone. At order 15 it is
still moving `±0.13 dB`. **P1dB is not converged at order 15.**

What can honestly be said:

- **G0 is converged.** Stable to `~3.4e-10 dB` from order 7 onward. Clean pass.
- **The total P1dB effect is small but unresolved.** Order 0 → order 15 is
  `0.087 dB` quantized / `0.114 dB` interpolated — both *below* the `0.1749 dB`
  refinement resolution, so this setup cannot resolve the effect it is trying to
  measure. The dramatic `−2.97 dB` at order 3 was real (far above the rung) and
  is explained by adding the single unbalanced tone `(1,-2)`; completing the set
  removes it.
- **Resolving the residual requires a finer instrument**, not more orders: a
  denser power grid and an explicit `--p1db-power-tol-db` (default is `None`
  here) well below 0.05 dB. Until then, "P1dB converged" is not supportable.

These are measurements at one JPA operating point on the least production-like
fixture, not general P1dB claims. No `CLAUDE.md` entry was added.
