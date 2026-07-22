# Agent handoff - spectrum peaks and junction scatter

This repo root is:

```powershell
cd C:\Users\Edoardo\Documents\EPFL\Thesis\twpa_jax
```

Use this with:

- `experiments/CLAUDE.md` for pump-basis/parity notes.
- `docs/runbook.md` for stable map commands and result locations.
- `docs/ripple_pump_placement.md` for ripple-placement workflow.

Current unrelated dirty file seen before this handoff was written:

```text
docs/reports/pump_solver_catalog.md
```

Do not revert it unless Edoardo asks.

## Current state

Main Python stack is in `experiments/`.

Important files:

| Path | Purpose |
| --- | --- |
| `experiments/exp10_full_ipm_pump_map_warmstart.py` | Main pump/gain map runner. Supports warmstart, Schur backend, fold skip, and per-cell signal spectra. |
| `experiments/plot_exp10_gain_map.py` | Current heatmap plotter. Reads `map_arrays.npz` and plots `gain_db_warm` / `gain_db_cold`. Needs extension for SG-filtered spectrum maxima. |
| `experiments/ripple_common.py` | Shared 2c/3c design helpers, passive ripple, pump solve, gain sweep. `build_design('2c'|'3c', outdir)` writes IPM matrices. |
| `experiments/exp10_jc_doc_python_design_builders.py` | Matrix/topology builder. `CircuitBuilder.josephson_inductor()` stores both `Lj` and `Ic = phi0_reduced / Lj`; `CircuitBuilder.write()` writes `ipm_arrays.npz`, `Bphi.npz`, `summary.json`. Best place to add deterministic junction scatter support. |
| `outputs/ipm_python_design` | 2c design matrices. |
| `outputs/ipm_python_design_3c` | 3c design matrices. |

Use the validated map path:

```text
--executor inprocess
--mode warmstart
--inproc-pump-backend schur_cpu_mt
--inproc-preconditioner real_coupled_fast
--inproc-fold-predictor secant
--inproc-fail-fast
--fold-skip-patience 2
--signal-detuning-mhz 100
--sidebands 6
```

Reason: this is the fast path already validated against the older runs. `fold-skip-patience 2` marks cells past fold as `SKIP_PAST_FOLD` / NaN instead of wasting time solving them.

## Existing spectrum runs

The three useful spectrum maps already exist:

| Run | Design | Window |
| --- | --- | --- |
| `outputs/exp10_spectrum_2c_m30_m20` | 2c | -30 to -20 dBm, 7.0 to 8.0 GHz |
| `outputs/exp10_spectrum_2c_m24_m18` | 2c peak zoom | -24 to -18 dBm, 7.1 to 7.4 GHz |
| `outputs/exp10_spectrum_3c` | 3c | -36 to -20 dBm, 7.0 to 8.0 GHz |

Each run contains:

| File | Contents |
| --- | --- |
| `map_points.csv` | One row per pump-power/frequency cell. Includes base trailing gain columns and raw spectrum peak columns: `spectrum_peak_gain_db`, `spectrum_peak_signal_ghz`. |
| `map_arrays.npz` | Base map arrays. Current plotter reads `gain_db_warm`. |
| `map_spectrum.npz` | Per-cell signal spectrum cube. This is the input for SG-filtered max gain. |
| `map_summary.json` | Runtime and convention metadata. |

`map_spectrum.npz` keys and shapes:

```text
pump_power_dbm      (50,)
pump_frequency_ghz  (50,)
signal_offset_mhz   (10,)
gain_spectrum_db    (50, 50, 10)   # i_power, j_freq, k_offset
signal_ghz          (10, 50)       # k_offset, j_freq
```

For the existing runs:

```text
signal_offset_mhz = [-1100, -850, -600, -350, -100, 100, 350, 600, 850, 1100]
```

`signal_ghz` depends on pump frequency because the convention is trailing `ws = wp - 100 MHz` plus offsets around that ladder. Do not assume one global signal axis.

## Re-run spectrum maps if needed

PowerShell form:

```powershell
$C = @(
  "--executor", "inprocess",
  "--mode", "warmstart",
  "--inproc-pump-backend", "schur_cpu_mt",
  "--inproc-preconditioner", "real_coupled_fast",
  "--inproc-fold-predictor", "secant",
  "--inproc-fail-fast",
  "--fold-skip-patience", "2",
  "--signal-detuning-mhz", "100",
  "--signal-spectrum",
  "--sidebands", "6",
  "--signal-workers", "6",
  "--n-power", "50",
  "--n-frequency", "50",
  "--overwrite"
)

python experiments/exp10_full_ipm_pump_map_warmstart.py @C `
  --ipm-dir outputs/ipm_python_design `
  --pump-power-min-dbm -30 --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 `
  --outdir outputs/exp10_spectrum_2c_m30_m20

python experiments/exp10_full_ipm_pump_map_warmstart.py @C `
  --ipm-dir outputs/ipm_python_design `
  --pump-power-min-dbm -24 --pump-power-max-dbm -18 `
  --pump-freq-min-ghz 7.1 --pump-freq-max-ghz 7.4 `
  --outdir outputs/exp10_spectrum_2c_m24_m18

python experiments/exp10_full_ipm_pump_map_warmstart.py @C `
  --ipm-dir outputs/ipm_python_design_3c `
  --pump-power-min-dbm -36 --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 `
  --outdir outputs/exp10_spectrum_3c
```

Plot current base maps:

```powershell
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m30_m20
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m24_m18
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_3c
```

Do not pass `--signal-ghz` for trailing maps; plot title should use `map_summary.json`.

## Next step 1: SG-filter spectrum max for map plots

Goal: for each pump map cell, use the full signal spectrum in `map_spectrum.npz`, smooth/interpolate along the signal-offset axis, extract the maximum gain, and use that as the plotted map value instead of the single trailing `gain_db_warm`.

Recommended behavior:

1. Load `map_spectrum.npz`.
2. For each `(i_power, j_freq)` spectrum vector, ignore cells where all values are NaN.
3. Smooth valid values along `signal_offset_mhz` with `scipy.signal.savgol_filter`.
4. Interpolate the smoothed curve on a denser offset grid.
5. Store/use:
   - `spectrum_sg_peak_gain_db` as `(n_power, n_freq)`.
   - `spectrum_sg_peak_signal_ghz` as `(n_power, n_freq)`.
   - optionally `spectrum_sg_peak_offset_mhz` as `(n_power, n_freq)`.
6. Plot a heatmap with the same axes/style as `plot_exp10_gain_map.py`, star on SG-filtered peak.

Implementation target:

- Best first patch: extend `experiments/plot_exp10_gain_map.py` with an option such as:

```text
--metric warm|spectrum_raw_peak|spectrum_sg_peak
--sg-window 5
--sg-polyorder 2
--interp-factor 25
```

Keep default behavior unchanged. When `--metric spectrum_sg_peak`, read `map_spectrum.npz`, compute peak grid, then call the existing plot helper.

SG details:

- SciPy is available in the current environment (`scipy.signal.savgol_filter`).
- With only 10 spectrum offsets, use a small odd window. Start with `window_length=5`, `polyorder=2`.
- If fewer than `window_length` finite points exist in a cell, either reduce to the largest valid odd window or fall back to raw finite max.
- Interpolate in offset MHz, not absolute GHz, because offset axis is common across pump frequencies.
- Convert winning offset to signal GHz per cell as:

```python
signal_peak_ghz = np.interp(offset_peak_mhz, signal_offset_mhz, signal_ghz[:, j_freq])
```

Sketch:

```python
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

def sg_peak_grid(spec, window=5, polyorder=2, interp_factor=25):
    offsets = spec["signal_offset_mhz"].astype(float)
    cube = spec["gain_spectrum_db"].astype(float)
    sig = spec["signal_ghz"].astype(float)
    dense_offsets = np.linspace(offsets.min(), offsets.max(),
                               (len(offsets) - 1) * interp_factor + 1)

    peak_gain = np.full(cube.shape[:2], np.nan)
    peak_sig = np.full(cube.shape[:2], np.nan)
    peak_off = np.full(cube.shape[:2], np.nan)

    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            y = cube[i, j, :]
            m = np.isfinite(y)
            if not np.any(m):
                continue
            x = offsets[m]
            yy = y[m]
            if len(yy) >= 5:
                w = min(window, len(yy) if len(yy) % 2 else len(yy) - 1)
                yy = savgol_filter(yy, window_length=w, polyorder=min(polyorder, w - 1), mode="interp")
            if len(yy) >= 2:
                f = PchipInterpolator(x, yy, extrapolate=False)
                yd = f(dense_offsets)
                k = np.nanargmax(yd)
                off = dense_offsets[k]
                val = yd[k]
            else:
                off = x[0]
                val = yy[0]
            peak_gain[i, j] = val
            peak_off[i, j] = off
            peak_sig[i, j] = np.interp(off, offsets, sig[:, j])
    return peak_gain, peak_sig, peak_off
```

Validation:

```powershell
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m30_m20 --metric spectrum_sg_peak
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m24_m18 --metric spectrum_sg_peak
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_3c --metric spectrum_sg_peak
```

Check that:

- Existing no-flag plot output is unchanged.
- SG peak grid has NaNs where cells failed/skipped.
- Peak marker moves to true spectrum maximum, not the fixed trailing `ws = wp - 100 MHz` value.
- Output filename distinguishes metric, for example `gain_map_spectrum_sg_peak.png`.

## Next step 2: maps with 1 percent Lj scatter

Goal: run pump maps with scattering in the junctions by adding Gaussian noise to each `Lj` value with sigma 1%.

Important physics/code relation:

```text
Ic = phi0_reduced / Lj
```

If `Lj` changes, `Ic` must change consistently. Do not only perturb `Lj` in metadata. The solver uses `ipm.Ic` in nonlinear `gamma(t) = Ic / phi0 * cos(...)`.

Best implementation target:

- Add scatter support where Josephson branches are created/written, preferably in `experiments/exp10_jc_doc_python_design_builders.py`.
- `CircuitBuilder.josephson_inductor(name, n1, n2, Lj)` currently appends `JosephsonBranch(name, n1, n2, Lj, PHI0_REDUCED / Lj)`.
- `CircuitBuilder.write()` writes both `Ic` and `Lj` to `ipm_arrays.npz` plus summaries.

Recommended API:

```python
def apply_lj_scatter(cb: CircuitBuilder, sigma: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    factors = rng.normal(loc=1.0, scale=sigma, size=len(cb.josephson))
    factors = np.clip(factors, 0.5, 1.5)  # generous guard; 1% should never hit
    for jj, fac in zip(cb.josephson, factors):
        jj.Lj = float(jj.Lj * fac)
        jj.Ic = float(PHI0_REDUCED / jj.Lj)
    return {
        "lj_scatter_sigma": sigma,
        "lj_scatter_seed": seed,
        "lj_scatter_factor_min": float(factors.min()),
        "lj_scatter_factor_max": float(factors.max()),
        "lj_scatter_factor_std": float(factors.std(ddof=0)),
    }
```

Where to expose it:

1. `experiments/ripple_common.py::build_design(design, ipm_dir)` is the user-facing helper used by the runbook for 2c/3c.
2. Add optional args there, for example `lj_scatter_sigma: float = 0.0`, `lj_scatter_seed: int | None = None`.
3. Pass scatter metadata into `CircuitBuilder.write()` so `summary.json` / `ipm_summary.json` record seed and sigma.
4. For CLI/manual runs, either:
   - add a tiny script `experiments/build_scattered_ipm_design.py`, or
   - use a one-line Python command after extending `ripple_common.build_design`.

Suggested design output paths:

```text
outputs/ipm_python_design_ljscatter_s1pct_seed1
outputs/ipm_python_design_3c_ljscatter_s1pct_seed1
```

Suggested build commands after adding API:

```powershell
python -c "import sys; sys.path.insert(0,'experiments'); import ripple_common as rc; rc.build_design('2c','outputs/ipm_python_design_ljscatter_s1pct_seed1', lj_scatter_sigma=0.01, lj_scatter_seed=1)"
python -c "import sys; sys.path.insert(0,'experiments'); import ripple_common as rc; rc.build_design('3c','outputs/ipm_python_design_3c_ljscatter_s1pct_seed1', lj_scatter_sigma=0.01, lj_scatter_seed=1)"
```

Then run maps by changing only `--ipm-dir` and `--outdir`:

```powershell
$C = @(
  "--executor", "inprocess",
  "--mode", "warmstart",
  "--inproc-pump-backend", "schur_cpu_mt",
  "--inproc-preconditioner", "real_coupled_fast",
  "--inproc-fold-predictor", "secant",
  "--inproc-fail-fast",
  "--fold-skip-patience", "2",
  "--signal-detuning-mhz", "100",
  "--sidebands", "6",
  "--n-power", "50",
  "--n-frequency", "50",
  "--overwrite"
)

python experiments/exp10_full_ipm_pump_map_warmstart.py @C `
  --ipm-dir outputs/ipm_python_design_ljscatter_s1pct_seed1 `
  --pump-power-min-dbm -30 --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 `
  --outdir outputs/exp10_ljscatter_s1pct_seed1_2c_m30_m20

python experiments/exp10_full_ipm_pump_map_warmstart.py @C `
  --ipm-dir outputs/ipm_python_design_3c_ljscatter_s1pct_seed1 `
  --pump-power-min-dbm -36 --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 `
  --outdir outputs/exp10_ljscatter_s1pct_seed1_3c
```

If spectrum maxima are needed on scattered maps too, add these flags to `$C`:

```powershell
"--signal-spectrum", "--signal-workers", "6"
```

Validation for scatter:

```powershell
python -c "import numpy as np; z=np.load('outputs/ipm_python_design_ljscatter_s1pct_seed1/ipm_arrays.npz'); Lj=z['Lj']; Ic=z['Ic']; phi0=z['phi0_reduced'][0]; print(Lj.min(), Lj.max(), Lj.std()/Lj.mean()); print(np.max(np.abs(Ic - phi0/Lj)))"
python experiments/plot_exp10_gain_map.py outputs/exp10_ljscatter_s1pct_seed1_2c_m30_m20
```

Expected:

- Scatter factor stats in `summary.json` show sigma near `0.01` for many junctions. Raw `Lj.std()/Lj.mean()` is only a quick check and can include intentional base-design variation.
- `max(abs(Ic - phi0/Lj))` near floating point roundoff.
- `summary.json` records `lj_scatter_sigma=0.01` and seed.

## Tests to run

Focused tests/checks:

```powershell
python -m pytest tests/test_pump_basis.py --basetemp C:\tmp\pytest-twpa
python -m pytest tests/test_fxjtwpa_node_order.py --basetemp C:\tmp\pytest-twpa
```

For SG plot changes, also run:

```powershell
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m30_m20
python experiments/plot_exp10_gain_map.py outputs/exp10_spectrum_2c_m30_m20 --metric spectrum_sg_peak
```

For scatter, a small smoke map is safer before a 50x50:

```powershell
python experiments/exp10_full_ipm_pump_map_warmstart.py `
  --executor inprocess --mode warmstart `
  --inproc-pump-backend schur_cpu_mt `
  --inproc-preconditioner real_coupled_fast `
  --inproc-fold-predictor secant `
  --inproc-fail-fast --fold-skip-patience 2 `
  --ipm-dir outputs/ipm_python_design_ljscatter_s1pct_seed1 `
  --n-power 5 --n-frequency 5 `
  --pump-power-min-dbm -30 --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 `
  --signal-detuning-mhz 100 --overwrite `
  --outdir outputs/exp10_ljscatter_s1pct_seed1_2c_smoke5
```

## Pitfalls

- Do not treat `pump_report.json` strict convergence alone as truth. Existing fold acceptance is reached `source_scale == 1` and `psi/phi0 < 1e3`.
- Do not use the 2c power window for 3c unless deliberately exploring over-fold cells. 3c folds lower; use `-36` to `-20` dBm for full coverage or `-36` to `-28` dBm for fold-centered 3c maps.
- Do not change pump basis defaults casually. For unbiased 4WM IPM/JTWPA use `positive_odd_jc`, K=10, Nt=40, linear-phasor seed.
- If perturbing `Lj`, keep `Ic` synchronized. Gain code uses `Ic`, not `Lj`, for nonlinear stiffness.
- `outputs/` is regenerable and gitignored; docs/code are the durable artifacts.
