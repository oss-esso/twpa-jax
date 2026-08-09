# Ripple Pump Placement (twpa_jax)

A pump-frequency selection and gain-characterisation workflow for the IPM JTWPA
(2- or 3-coupler) built on the fast twpa_jax pump/gain stack
(`exp07`/`exp08`/`exp09`). It chooses the pump frequency from the device's
**passive coupler ripple** — cheap, deterministic, and needing no expensive
pump-power/frequency map — then finds the strongest pump current that genuinely
converges, so every reported gain curve is a physical solution. A companion step
**cross-checks a pump map's top candidates** against the same ripple.

This is the port of the Harmonia.jl `ripple_pump_placement` workflow; the maps
solve much faster here and reach the current fold, so the whole loop runs in
minutes.

## Files

| File | Role |
| ---- | ---- |
| `experiments/ripple_common.py` | shared: design build, passive S-matrix, peak/period, snap-to-120, pump-ladder + gain-sweep subprocess helpers |
| `experiments/exp17_ripple_pump_placement.py` | driver: passive ripple → +120° placement → Ic ladder → gain sweep |
| `experiments/exp17_ripple_map_crosscheck.py` | driver: verify a pump map's top cells against the ripple |
| `experiments/plot_ripple_pump_placement.py` | one PNG per operating point from a run's `manifest.json` |

## Ports and S-parameters

The IPM JTWPA is a 4-port device: `1 = signal in`, `2 = signal out`,
`3 = pump rail`, `4 = pump source`.

| Symbol | Element (`out ← in`) | Meaning |
| ------ | -------------------- | ------- |
| `S21`  | `2 ← 1`              | forward signal gain |
| `S12`  | `1 ← 2`              | reverse transmission |
| `S24`  | `2 ← 4`              | pump → signal-out leakage |
| `S42`  | `4 ← 2`              | signal → pump-out (passive coupler transmission) |

The **passive** (pump-off) 4-port S-matrix is computed directly from the exp07
design matrices: the exp09 linear block `D(ω) + K̂₀` already carries the
Josephson inductance through `γ̂₀ = Ic/φ₀`, so one sparse LU per frequency
(solving every source column) yields a genuine passive S-parameter under the
same Norton-port normalisation exp09 uses for gain, `S_ij = 2 V_i /(I_j Z0)`
(`−1` on the diagonal). No pump solve is needed for the ripple — it is a handful
of seconds for a 1401-point fine grid.

## The workflow

For each design (`--design 2c` or `3c`):

1. **Build** the design (`ripple_common.build_design`). `2c` reuses exp07's
   cached coupler geometry; `3c` re-optimises the coupler for −20 dB at 10 GHz
   and doubles the JTL array (`array_length=648`, `arrays_per_dc=2`).
2. **Passive ripple** — solve `|S42|` (coupler transmission) and `|S21|`
   (through path) on a fine grid. `|S42|` shows a periodic coupler ripple (deep
   notches between shallow peaks, ≈180–190 MHz period for `2c`); `|S21|` is flat
   near 0 dB.
3. **+120° placement** — place `fp` about one third of a local ripple period
   (**≈120°**) above a strong `S42` **peak**. Peaks are auto-selected so the
   placed `fp = peak + period/3` lands inside the pump-map band
   (`--map-pump-band-ghz`, default 6.0–8.5 GHz).
4. **Ic ladder** — at each `fp`, sweep the pump current (`--ic-ladder`, ×Ic) and
   keep the **strongest accepted** solve (see below). Then sweep the signal
   (exp09) at that `fp` to get `S21(fs)` (and, with `--extra-sparams`, `S12` and
   `S24`).

`Ic = φ₀/(2π Lj) = 2.656 µA` (median junction).

### Convergence: accept the fold edge, not just the strict flag

The pump HB is stiff near the current fold. exp08's strict
`final_status == VALID_CONVERGED` requires the Newton residual to reach `1e-9`
at full drive; at the fold edge a **physically fine** solve (bounded node flux,
20 dB gain) plateaus slightly above that and is flagged `FAIL`. Judging by that
flag alone throws away the best operating point (e.g. `2c` at `fp = 7.86 GHz`
gives 5 dB at the strict 3 Ic but **20.6 dB** at the fold-edge 4 Ic).

The ladder therefore keeps the strongest **accepted** rung:

> **accepted** = the continuation reached the full requested current
> (`source_scale == 1`) **and** the node flux is bounded (`ψ/φ₀ < 1e3`).

The full-scale test rejects rungs that stalled *below* the requested current
(genuinely past the fold — those never reach it), while the flux bound rejects
runaways. Rungs that reach full scale with bounded flux are physical even when
their residual sits at `~1e-2`; their gain sweep is `VALID_SOLVED`. Both flags
(`strictly_converged`, `reached_full_scale`) and `coeff_rel` are recorded per
point.

The pump solve uses the JC positive-odd phasor basis (`positive_odd_jc`, K=10
modes, Nt=40) with **adaptive continuation + secant predictor** from a
`linear_phasor` seed — the same settings as the trusted pump-map reference pass,
and what reaches the fold robustly (fixed-step continuation stalls at
fp-specific fold tongues).

## Running it

```console
$ python experiments/exp17_ripple_pump_placement.py --design 2c
$ python experiments/exp17_ripple_pump_placement.py --design 3c --extra-sparams
$ python experiments/plot_ripple_pump_placement.py --rundir outputs/ripple_pump_placement_2c
```

Outputs land in `outputs/ripple_pump_placement_<design>/`:

- `passive_ripple.npz` — `freq_ghz`, `s21_db`, `s42_db`, full `S`, port order.
- `manifest.json` — one record per operating point (`fp`, reference peak, degrees
  offset, chosen `Ic`, flux, flags, peak `S21`, sweep CSV paths).
- `point_<k>_fp<MHz>/` — per-rung pump solves + gain sweep CSVs.

The plotter emits one PNG per point: the pump-off `S42`/`S21` ripple (reference
peak, `fp`, measured degrees), the pump-on `S21` gain (with a `ws = fp − 100 MHz`
marker), and — when swept — `S12` and `S24`.

## Cross-checking against a pump map

Once a pump map exists (e.g. the 50×50 trailing-signal map,
`outputs/exp10_pump_map_trailing_50x50_m30_m20/`), `exp17_ripple_map_crosscheck.py`
closes the loop:

```console
$ python experiments/exp17_ripple_map_crosscheck.py \
      --design 2c --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20 --top-n 4
$ python experiments/plot_ripple_pump_placement.py \
      --rundir outputs/ripple_map_crosscheck_2c
```

It reads the map's points CSV (`gain_map_points.csv` **or** `map_points.csv`),
nominates a **pool** of the `--pool-n` (12) highest-gain cells and, walking the
pool in gain order, for each cell:

1. **snaps** the map `fp` onto the nearest passive `S42` `peak + period/3` target
   (the +120° design point), recording the original map offset and the shift;
2. **re-solves** the pump + gain at that snapped `fp` with the cell's own pump
   current (adaptive continuation);
3. keeps the cell only if the re-solve is **physical** — accepted (full scale,
   bounded flux) **and** a finite gain in `[−40, 60]` dB.

It stops once `--top-n` cells verify.

### What the cross-check reveals

The current fold is **fp-specific**: a map cell that converges to high gain at
its own `fp` and drive can be pushed *past* the fold when snapped to a nearby
+120° target at the same current. So the survivors of the pool are the cells
whose snapped `fp` + current still sit below the local fold — the ones with a
genuinely coherent +120° operating point. This is the same "snapping can move a
candidate off the fold" phenomenon seen in Harmonia, and it is why the check
re-solves rather than trusting the raw map gain.

### Results

**Placement (2c, no map).** The +120° point at `fp = 7.859 GHz` (peak 7.798,
+61 MHz) reaches the fold at **4.0 Ic** (10.63 µA, flux ψ/φ₀ = 0.68,
`coeff_rel ≈ 1e-2`, `reached_full_scale`) and gives a physical **20.6 dB** peak
at 7.95 GHz — versus only 5 dB at the strict 3 Ic, illustrating why the fold-edge
acceptance matters.

**Cross-check (2c) vs the 50×50 trailing map**
(`exp10_pump_map_trailing_50x50_m30_m20`, peak cell 27.1 dB @ 7.27 GHz,
−20.4 dBm). Walking the top-gain pool, every headline cell at **4.0–4.2 Ic** is
*past the cold fold* once snapped to its +120° target and fails to re-solve
(`reached_full_scale = False`, even with adaptive+secant — the map reached them
by **warm-starting**, which a cold solve cannot cross). The verified survivor is
lower in the ranking but coherent:

| Cell | Map gain | Map `fp` / offset | Snapped `fp` | `Ic` | Re-solved `S21` | flux | +120° |
| ---- | -------- | ----------------- | ------------ | ---- | --------------- | ---- | ----- |
| pool 7 | 22.3 dB | 7.286 GHz (+57°) | 7.319 GHz | 3.68 | **21.1 dB** @ 7.40 | 0.77 | ✅ |
| pool 1 | 27.1 dB | 7.265 GHz (+19°) | 7.319 GHz | 4.04 | — (past cold fold) | 0.68 | ✗ |
| pool 2 | 26.1 dB | 7.490 GHz (+117°) | 7.491 GHz | 4.23 | — (past cold fold) | 0.72 | ✗ |

The takeaway matches Harmonia: a map cell's high gain lives on a warm-started
branch past the cold current fold at its *specific* `fp`; the cold-reproducible,
+120°-coherent operating point at that peak is **~21 dB at 3.68 Ic**. Judge map
candidates by the re-solved fold-edge gain, never by the raw warm-started map
cell.

## Knobs

| Constant / flag | Purpose |
| --------------- | ------- |
| `--design` | `2c` (cached coupler) or `3c` (−20 dB @ 10 GHz, re-optimised) |
| `--ic-ladder` | pump-current ratios (×Ic) tried per `fp`; strongest accepted wins |
| `--ripple-grid-ghz` / `--ripple-band-ghz` | passive fine grid and peak-search band |
| `--map-pump-band-ghz` / `--n-points` | band the placed `fp` must land in, and how many points |
| `--signal-grid-ghz` / `--sidebands` / `--gamma-nt` | pump-on gain sweep grid and HB detail |
| `--extra-sparams` | also sweep `S12` (2→1) and `S24` (4→2) |
| `--pool-n` / `--top-n` (cross-check) | pool size and how many verified cells to keep |
