# Runbook — pump maps, ripple placement & cross-check

Quick reference for the runs we do routinely, across **both** repos:

- **`twpa_jax`** — the fast Python/scipy pump+gain stack (exp07/08/09/10/17).
- **`Harmonia.jl`** — the JosephsonCircuits.jl reference stack (config-driven).

Paths below are relative to each repo root:
`C:\Users\Edoardo\Documents\EPFL\Thesis\twpa_jax` and
`...\Thesis\Harmonia.jl`.

---

## Conventions (both stacks)

| Quantity | Value / rule |
| --- | --- |
| Signal readout | **trailing**: `ws = wp − 100 MHz` per cell |
| Pump attenuation | 35 dB (external dBm → on-chip) |
| JC current scale | injected pump current = **2 ×** physical port current (JC positive-phasor convention) |
| Median `Ic` | **2.656 µA** (both 2c and 3c) |
| Pump basis | `positive_odd_jc`, K=10 modes, Nt=40, `linear_phasor` seed |
| Fold acceptance | reached `source_scale == 1` **and** `ψ/φ₀ < 1e3` (not the strict Newton `1e-9` flag) |

**External dBm → injected `Ic`** (35 dB att, 50 Ω, ×2 JC scale) — use this to pick a
power window around a design's fold:

| dBm | inj. Ic | dBm | inj. Ic |
| --- | --- | --- | --- |
| −36 | 1.34 | −26 | 4.24 |
| −34 | 1.69 | −24 | 5.34 |
| −32 | 2.13 | −22 | 6.73 |
| −30 | 2.68 | −20 | 8.47 |
| −28 | 3.37 | −18 | 10.7 |

- **2c** folds around **~7–8 injected Ic** → good window **−30 → −20 dBm**.
- **3c** folds **~5–6 dB lower** (longer JTL array = stronger gain/pass, folds sooner),
  around **~2.5–3.5 injected Ic** → good window **−36 → −28 dBm**. Using the 2c window
  on 3c gives an almost-empty (all past-fold) map.

---

## twpa_jax

### Designs

| Design | Dir | Build |
| --- | --- | --- |
| 2c (standard IPM, 2508 JJ) | `outputs/ipm_python_design` | prebuilt (exp07 default) |
| 3c (−18.45 dB coupler, 3888 JJ) | `outputs/ipm_python_design_3c` | build once (below) |

Build the 3c design (fast, ~seconds):

```bash
python -c "import sys; sys.path.insert(0,'experiments'); import ripple_common as rc; rc.build_design('3c','outputs/ipm_python_design_3c')"
```

`ripple_common.build_design('2c'|'3c', dir)` writes the exp07 matrices. `Ic` via
`rc.ic_reference_a(dir)` (= 2.656 µA).

### 1. Pump/gain map (exp10) — the main run

The trusted, fastest solver path (`schur_cpu_mt` + `real_coupled_fast` + secant
predictor), warm-started up each pump-power column, through the fold:

```bash
# 2c reference map (−30→−20 dBm × 7.0–8.0 GHz)
python experiments/exp10_full_ipm_pump_map_warmstart.py --executor inprocess \
    --mode warmstart --inproc-pump-backend schur_cpu_mt \
    --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant \
    --inproc-fail-fast --fold-skip-patience 2 \
    --ipm-dir outputs/ipm_python_design \
    --n-power 50 --n-frequency 50 \
    --pump-power-min-dbm -30 --pump-power-max-dbm -20 \
    --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 \
    --signal-detuning-mhz 100 --overwrite \
    --outdir outputs/exp10_pump_map_trailing_50x50_m30_m20

# 3c fold-centered map (−36→−28 dBm × 7.0–8.0 GHz) — note the shifted window
python experiments/exp10_full_ipm_pump_map_warmstart.py --executor inprocess \
    --mode warmstart --inproc-pump-backend schur_cpu_mt \
    --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant \
    --inproc-fail-fast --fold-skip-patience 2 \
    --ipm-dir outputs/ipm_python_design_3c \
    --n-power 50 --n-frequency 50 \
    --pump-power-min-dbm -36 --pump-power-max-dbm -28 \
    --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 \
    --signal-detuning-mhz 100 --overwrite \
    --outdir outputs/exp10_pump_map_trailing_50x50_3c
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--executor inprocess` | run pump+gain in one process (no per-point import tax) |
| `--inproc-pump-backend schur_cpu_mt` | Schur-reduced sparse backend (2.5–4.5× faster at the fold) |
| `--inproc-preconditioner real_coupled_fast` | exact coupled Jacobian, GMRES in ~1 iter |
| `--inproc-fold-predictor secant` | extrapolate the guess along the power axis (fewer Newton steps near the fold) |
| `--inproc-fail-fast` | over-fold points fail in ~one stalled solve (skip reseed/fallback recovery); keep warm-starting from the last converged neighbour |
| `--fold-skip-patience 2` | **speed**: after 2 consecutive over-fold failures going up a column, skip the rest of the column unsolved (`SKIP_PAST_FOLD`, NaN gain). See note below. |
| `--inproc-schur-cache-size N` | **memory cap**: max per-frequency Schur partitions kept (default 2). See note below. |
| `--n-power / --n-frequency` | grid size |
| `--pump-power-min/max-dbm` | external power window (dBm) |
| `--pump-freq-min/max-ghz` | pump frequency window (GHz) |
| `--signal-detuning-mhz 100` | trailing signal `ws = wp − 100 MHz` |
| `--overwrite` | wipe the outdir first |

Outputs land in `--outdir`: `map_points.csv`, `map_arrays.npz`
(`gain_db_warm` 50×50), `map_summary.json`.

**Memory note (fixed 2026-07-04):** the in-process engine used to cache one
Schur partition **per frequency, unbounded** — each holds a big factorized block,
so a 50-frequency map accumulated ~16 GB and OOM'd (`malloc fails for dworkptr[]`)
around frequency ~35, worse for the larger 3c. The cache is now **LRU-bounded**
(`--inproc-schur-cache-size`, default 2), so a single-process 50×50 (or larger)
stays flat. **No chunking needed anymore.** Eviction+rebuild is numerically
identical (partitions are deterministic from the design matrices).

**Speed note (fold short-circuit, added 2026-07-05):** on a hot/over-fold map
the non-converging cells dominate runtime (~65–71 Newton iters and 5–8 s each vs
2–4 iters / ~0.9 s for a PASS). Measured wasted share: zoom map **79%**, 2c **51%**.
Within a frequency column the HB fold is a turning point (no re-convergence
above it), so `--fold-skip-patience 2` marks every cell above the fold
`SKIP_PAST_FOLD` **without solving** — validated to leave every accepted cell
bit-identical (0 lost PASS cells). Pair it with `--inproc-fail-fast`. Together
they take a hot 50×50 from ~130 min toward the PASS-only floor (~30 min).
`SKIP_PAST_FOLD` cells read as NaN gain (map holes), same as a real over-fold
failure.

Plot:

```bash
# NOTE: do NOT pass --signal-ghz on a trailing map — it stamps a single wrong
# "signal X GHz" in the title. Omit it and the title reads the trailing
# convention ("trailing signal ws = fp − 100 MHz") from map_summary.json.
python experiments/plot_exp10_gain_map.py outputs/exp10_pump_map_trailing_50x50_3c
# -> <outdir>/gain_map_warm.png  (grey = non-converged/skipped cells, red star = peak)
```

**Solver-effort diagnostics** (per-point wall time / Newton / GMRES colormaps,
one PNG per metric per map folder) — the scratchpad `plot_solvetime_maps.py`
reads `elapsed_s`, `pump_newton_total`, `pump_gmres_total` from `map_points.csv`.
Handy for confirming where a map spends its time (the fold triangle).

### 2. Passive ripple → +120° placement → gain (exp17)

Chooses `fp` from the passive S42 coupler ripple (no pump map needed), runs an
Ic ladder, sweeps gain:

```bash
python experiments/exp17_ripple_pump_placement.py --design 2c
python experiments/exp17_ripple_pump_placement.py --design 3c --extra-sparams
python experiments/plot_ripple_pump_placement.py --rundir outputs/ripple_pump_placement_2c
```

Outputs: `outputs/ripple_pump_placement_<design>/` (`passive_ripple.npz`,
`manifest.json`, per-point sweeps + one PNG/point). See
[`ripple_pump_placement.md`](ripple_pump_placement.md) for all knobs.

### 3. Cross-check a pump map against the ripple (exp17)

Snaps a map's top-gain cells onto the nearest +120° target and cold-re-solves
them (exposes warm-start-only cells past the cold fold):

```bash
python experiments/exp17_ripple_map_crosscheck.py \
    --design 2c --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20 --top-n 4
python experiments/plot_ripple_pump_placement.py --rundir outputs/ripple_map_crosscheck_2c
```

### 4. Quick pump-convergence probe (find a design's fold)

One-off: does a design converge at `fp` / current? (used to locate the 3c fold):

```bash
python -c "
import sys; sys.path.insert(0,'experiments'); import ripple_common as rc
from pathlib import Path
ipm='outputs/ipm_python_design_3c'; ic=rc.ic_reference_a(ipm)
d=Path('outputs/_diag'); d.mkdir(parents=True,exist_ok=True)
for fp,r in [(7.25,1.5),(7.25,2.5),(7.25,3.5)]:
    o=rc.solve_pump(ipm, d/f'p{fp}_{r}', fp_ghz=fp, ratio_ic=r, ic_a=ic, timeout_s=180)
    print(fp, r, 'accepted', o.accepted, 'flux', round(o.flux_over_phi0,3))
"
```

### 5. Passive S42 peaks / +120° placements for a design

```bash
python -c "
import sys, numpy as np; sys.path.insert(0,'experiments'); import ripple_common as rc
ipm='outputs/ipm_python_design_3c'
f=np.linspace(6.0e9,8.6e9,1301); S=rc.passive_s_matrix(ipm,f)
s42=rc.db20(S[:,3,1]); fg=f/1e9
peaks=rc.find_s42_peaks(fg,s42,(6.2,8.5))
for p in peaks:
    pl=rc.place_120(float(p),peaks); print(round(p,3),'-> fp',round(pl.fp_ghz,3))
"
```

### Long detached runs (only relevant when *I* launch them)

When you run these yourself in a terminal, ignore this. When Claude launches a
long map as a tracked background task, an idle conversation can get it reaped, so
launch fully detached and poll the on-disk `map_summary.json`:

```bash
nohup python experiments/exp10_full_ipm_pump_map_warmstart.py ... > logs/map.log 2>&1 &
```

---

## Harmonia.jl (JosephsonCircuits reference)

Everything is **config-driven**. From the repo root, activate the project once
(`julia --project=.`). Runs write to `../runs/<name>/` (sibling of the repo).

### Pump maps (dBm gain maps)

```bash
# 2c 50×50
julia --project=. scripts/run_simulation.jl \
    --config examples/configs/harmonia_ipm_jtwpa_pump_map_dbm_50x50.json \
    --output ../runs/ipm_dbm_50x50

# 3c 50×50
julia --project=. scripts/run_simulation.jl \
    --config examples/configs/harmonia_ipm_jtwpa_3couplers_pump_map_dbm_50x50.json \
    --output ../runs/ipm_3c_dbm_50x50
```

Also `_dbm_10x10` / `_dbm_25x25` config variants for quick sweeps. The map window
(power/frequency, grid) lives **inside the config JSON** (`examples/configs/`), not
on the CLI — edit the config to change ranges. Plot both devices:

```bash
python scripts/plot_pump_maps.py
# reads ../runs/ipm_dbm_50x50 and ../runs/ipm_3c_dbm_50x50 -> scripts/outputs/*.png
```

(`plot_pump_maps.py` looks for run dirs named `ipm_dbm_50x50` / `ipm_3c_dbm_50x50`
— match those `--output` names, or edit the `DEVICES` list in the script.)

### Ripple placement + gain (the Julia port's source of truth)

```bash
julia --project=. scripts/ripple_pump_placement.jl 2c      # or 3c
python scripts/plot_ripple_pump_placement.py --design 2c \
    --rundir ../runs/ripple_pump_placement_2c
```

Writes `../runs/ripple_pump_placement_<design>/` (`passive_ripple.h5`,
`manifest.json`, `point_*/simulation.h5`). Reference S42 peaks are hard-set for 2c
(`[7.99, 8.16, 8.355, 8.525]`) and auto-selected for 3c.

### Ripple map cross-check

Requires the passive ripple file above to exist first.

```bash
julia --project=. scripts/ripple_map_crosscheck.jl 2c ipm_dbm_25x25 4
#                                                  <design> <map_run> <top_n>
python scripts/plot_ripple_pump_placement.py --design 2c \
    --rundir ../runs/ripple_map_crosscheck_2c --label 2c-map
```

Reads the top-N converged cells (both `converged` and `jc_converged` flags) from
`../runs/<map_run>`, snaps each to the S42 ripple, re-runs the pumped signal sweep
at the map's own `(fp, pump current)`.

### Other plotters

| Script | Purpose |
| --- | --- |
| `scripts/plot_gain_curves.py` | gain S21(fs) for a single run |
| `scripts/plot_top_pump_signal_sparams.py` | S-params of the top pump/signal cells |
| `scripts/auto_operating_point.py` | pick an operating point automatically |

---

## Where results live

| | twpa_jax | Harmonia.jl |
| --- | --- | --- |
| Maps | `outputs/exp10_pump_map_*` | `../runs/ipm[_3c]_dbm_*` |
| Ripple | `outputs/ripple_pump_placement_*` | `../runs/ripple_pump_placement_*` |
| Cross-check | `outputs/ripple_map_crosscheck_*` | `../runs/ripple_map_crosscheck_*` |
| Figures | `outputs/*.png` (next to each run) | `scripts/outputs/*.png` |

`outputs/` (twpa_jax) and `../runs/` (Harmonia) are gitignored / regenerable.


 Commands for the three spectrum maps

  Using the validated fast path (schur_cpu_mt + real_coupled_fast + secant + fail-fast + fold short-circuit) with the spectrum on, sidebands 6 + 6 workers (your near-fold accuracy/speed pick):

  cd C:\Users\Edoardo\Documents\EPFL\Thesis\twpa_jax

  # common flags
  $C = "--executor inprocess --mode warmstart --inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant --inproc-fail-fast --fold-skip-patience 2
  --signal-detuning-mhz 100 --signal-spectrum --sidebands 6 --signal-workers 6 --n-power 50 --n-frequency 50 --overwrite"

  # 2c  (-30 -> -20 dBm x 7.0-8.0 GHz)
  python experiments/exp10_full_ipm_pump_map_warmstart.py $C.Split(' ') --ipm-dir outputs/ipm_python_design    --pump-power-min-dbm -30 --pump-power-max-dbm -20 --pump-freq-min-ghz 7.0
  --pump-freq-max-ghz 8.0 --outdir outputs/exp10_spectrum_2c_m30_m20 ;
  # 2c peak  (-24 -> -18 dBm x 7.1-7.4 GHz)
  python experiments/exp10_full_ipm_pump_map_warmstart.py $C.Split(' ') --ipm-dir outputs/ipm_python_design    --pump-power-min-dbm -24 --pump-power-max-dbm -18 --pump-freq-min-ghz 7.1
  --pump-freq-max-ghz 7.4 --outdir outputs/exp10_spectrum_2c_m24_m18 ;
  # 3c  (-36 -> -20 dBm x 7.0-8.0 GHz, full fold coverage)
  python experiments/exp10_full_ipm_pump_map_warmstart.py $C.Split(' ') --ipm-dir outputs/ipm_python_design_3c --pump-power-min-dbm -36 --pump-power-max-dbm -20 --pump-freq-min-ghz 7.0
  --pump-freq-max-ghz 8.0 --outdir outputs/exp10_spectrum_3c





-----


cd D:\Projects\Thesis\twpa_jax

$C = "--executor inprocess --mode warmstart --inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant --inproc-fail-fast --fold-skip-patience 2 --inproc-schur-cache-size 2 --signal-detuning-mhz 100 --signal-backend direct --signal-solver superlu --sidebands 6 --signal-workers 6 --pump-mode-count 10 --nt 40 --inproc-max-newton 16 --inproc-solve-deadline-s 14 --n-power 100 --n-frequency 100 --frequency-chunk-size 10 --no-signal-spectrum --overwrite"

python scripts/run_gain_map.py $C.Split(' ') --ipm-dir outputs/ipm_python_design --pump-power-min-dbm -35 --pump-power-max-dbm -22.5 --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.8
 --outdir outputs/solver_nospectrum_2c_noscatter_m35_m22p5_7p5_8p5_100x100_sb10_failfast

c

@'
import csv, subprocess, sys
from pathlib import Path

runs = [
    ("2c", Path("outputs/solver_nospectrum_2c_noscatter_m35_m22p5_7p5_8p5_100x100_sb10_failfast"), Path("outputs/ipm_python_design")),
    ("3c", Path("outputs/solver_nospectrum_3c_noscatter_m38_m20_7p25_7p75_100x100_sb10_failfast"), Path("outputs/ipm_python_design_3c")),
]

half_span_ghz = 2.5
points = 500
sidebands = 6

for design, run_dir, ipm_dir in runs:
    rows = list(csv.DictReader((run_dir / "map_points.csv").open(encoding="utf-8")))
    rows = [r for r in rows if r.get("status") == "PASS" and r.get("gain_db") not in ("", "None", None)]
    rows.sort(key=lambda r: float(r["gain_db"]), reverse=True)

    for rank, r in enumerate(rows[:3], start=1):
        fp = float(r["pump_freq_ghz"])
        # pump_dir in map_points.csv is already repo-root-relative (it contains
        # run_dir/chunks/.../pump), so use it as-is -- do NOT prepend run_dir.
        pump_dir = Path(r["pump_dir"])

        out = run_dir / "candidate_sweeps" / f"{design}_rank_{rank:03d}_point_{int(r['point_index']):04d}"
        s21 = out / "s21"
        s24 = out / "s24"

        base = [
            sys.executable, "experiments/exp09_full_ipm_gain_from_pump.py",
            "--ipm-dir", str(ipm_dir),
            "--pump-dir", str(pump_dir),
            "--fallback-pump-freq-ghz", str(fp),
            "--sweep",
            "--signal-start-ghz", str(fp - half_span_ghz),
            "--signal-stop-ghz", str(fp + half_span_ghz),
            "--points", str(points),
            "--sidebands", str(sidebands),
            "--gamma-nt", "96",
        ]

        subprocess.check_call(base + ["--source-port", "1", "--out-port", "2", "--outdir", str(s21)])
        subprocess.check_call(base + ["--source-port", "4", "--out-port", "2", "--outdir", str(s24)])
'@ | python -


@'
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from twpa_solver.plotting.metrics import dense_fit_curve


def load(path):
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r["status"] == "VALID_SOLVED"]
    return ([float(r["signal_ghz"]) for r in rows],
            [float(r["gain_db"]) for r in rows])


for d in Path("outputs").glob(
    "solver_nospectrum_*_100x100_sb10_failfast/candidate_sweeps/*"
):
    s21_csv = d / "s21" / "gain_sweep.csv"
    s24_csv = d / "s24" / "gain_sweep.csv"
    if not s21_csv.exists() or not s24_csv.exists():
        continue

    f21, g21 = load(s21_csv)
    f24, g24 = load(s24_csv)
    if len(f21) < 2 or len(f24) < 2:
        continue

    # S21 (gain) and S24 in two stacked subplots, shared signal-frequency axis.
    fig, (ax21, ax24) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax21.plot(f21, g21, color="#1f77b4", lw=1.8, label="S21 gain")
    try:  # savgol + cubic-spline envelope, same fit as plot_gain_map candidates
        _, _, f_dense, g_dense, _ = dense_fit_curve(
            f21, g21, n_dense=2000, window_frac=0.05, polyorder=3
        )
        ax21.plot(f_dense, g_dense, color="#ff7f0e", ls="--", lw=2.0, label="S21 fit")
    except ValueError:
        pass
    ax21.set_ylabel("Gain / dB")
    ax21.set_title(d.name)
    ax21.legend(loc="best")

    ax24.plot(f24, g24, color="#2ca02c", lw=1.8, label="S24")
    ax24.set_xlabel("Signal frequency / GHz")
    ax24.set_ylabel("Gain / dB")
    ax24.legend(loc="best")

    for ax in (ax21, ax24):
        ax.minorticks_on()
        ax.grid(which="major", alpha=0.5, linewidth=1.3)
        ax.grid(which="minor", alpha=0.25, linewidth=0.7)

    fig.tight_layout()
    out = d / "gain_s21_s24_spectrum"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
'@ | python -




## fast runs

cd D:\Projects\Thesis\twpa_jax
>> 
>> $C = "--executor inprocess --mode warmstart --inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant --inproc-fail-fast --fold-skip-patience 2 --inproc-schur-cache-size 2 --signal-detuning-mhz 500 --signal-backend direct --signal-solver superlu --skip-baselines --sidebands 10 --signal-workers 6 --pump-mode-count 10 --nt 40 --inproc-max-newton 16 --inproc-solve-deadline 14 --n-power 50 --n-frequency 20 --frequency-chunk-size 10 --no-signal-spectrum --overwrite"  
>> 
>> python scripts/run_gain_map.py $C.Split(' ') --ipm-dir outputs/ipm_python_design --pump-power-min-dbm -32 --pump-power-max-dbm -20 --pump-freq-min-ghz 8.1 --pump-freq-max-ghz 8.3

# column solve

python scripts/run_gain_map_column_matrices.py --design-dir designs/ipm_2c_fixed --column-frequency-ghz 7.9 --n-power 30 --pump-power-min-dbm -35 --pump-power-max-dbm -22.5 --extra-map-args --log-level INFO