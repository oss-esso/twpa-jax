# twpa-solver

Harmonic-balance simulation of travelling-wave parametric amplifiers (TWPA /
JTWPA / KITWPA) and the coupled-line IPM devices, in Python.

Given a circuit — a Josephson junction ladder with its transmission lines,
couplers and ports — the solver finds the periodic steady state under a strong
pump, then computes small-signal gain, gain maps over pump power and frequency,
and large-signal compression (P1dB).

Everything is NumPy/SciPy sparse. No Julia dependency, no GPU requirement.

---

## Install

```powershell
python -m pip install -e .
python -m pip install -e ".[fast]"   # adds pypardiso, ~4x faster factorization
```

Python 3.10+. Verify:

```powershell
python -m pytest -q
```

Expect ~367 passing. Two failures in `tests/test_column_matrices_tracer.py` are
known and unrelated to the solver — see [Known issues](#known-issues).

On Windows, run pytest with a scratch directory outside the repository to avoid
an ACL problem on nested temp dirs:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa
```

---

## Quickstart

Build a circuit, look at it with the pump off, and confirm the solver runs:

```powershell
# 1. Build an IPM design from parameters (writes matrices + netlist)
python -m twpa_solver.builders.ipm --outdir outputs/my_design --write-matrices --coupler-mode cached

# 2. Passive S-parameters of a shipped design, with plots
python workflows/build_design_and_passive.py --design-dir designs/ipm_2c_fixed

# 3. Load a circuit and inspect it
python -c "from twpa_solver.core import load_circuit; c = load_circuit('designs/ipm_2c_fixed'); print(c.node_count, c.branch_count, c.port_to_index)"
```

---

## Concepts

### The model

Every circuit reduces to one node-flux equation:

```
C xddot + G xdot + K x + Bphi i_J(Bphi.T x) = i_src
```

`x` holds the non-ground node fluxes. `C`, `G`, `K` are node-admittance stamps
(capacitance, conductance, inverse inductance). `Bphi` is the node-to-branch
incidence matrix for the **nonlinear** branches only.

One asymmetry matters throughout: **Josephson inductances are not stamped into
`K`.** They become nonlinear `Bphi` branches carrying `Ic = phi0_reduced / Lj`,
evaluated through a branch law at solve time. Their `Cj` shunt capacitance *is*
in `C`. Ordinary inductors stamp `1/L` into `K`; a mutually coupled pair stamps
`B Lpair^-1 B.T`.

### Circuit directories

A circuit lives in a directory that `twpa_solver.core.load_circuit` reads:

| File | Contents |
| --- | --- |
| `C.npz`, `G.npz`, `K.npz`, `Bphi.npz` | scipy sparse CSR matrices |
| `ipm_arrays.npz` | `nodes`, `Ic`, `Lj`, `phi0_reduced`, `port_numbers`, `port_indices` |
| `ipm_summary.json` | parameters, element counts, metadata |
| `ipm_elements.csv` | the netlist, with `role` and `cell_index` per element |
| `ipm_ports.csv` | port table |

`load_circuit` reads arrays by name and ignores unknown keys, so builders can
add fields without breaking consumers.

### Ports

For the IPM/2c devices: **1** signal in, **2** signal out, **3** pump rail
reference, **4** pump source. The pump counter-propagates relative to the
signal.

Pump injection is **port 4**, not port 1. Using port 1 for both leaves a
promoted-pump residual of sqrt(2). Signal scattering is 1 → 2.

### Pump basis

The harmonic-balance pump reconstructs the real waveform with a positive-phasor
convention:

```
psi_pump(t) = 2 * Re sum_k X_k exp(+i k omega_p t)
```

For an unbiased 4-wave-mixing device the correct mode list is **odd**
(`[1,3,5,...]`), not dense (`[1,2,3,...]`). Dense harmonics truncate the high
odd pump content. `twpa_solver.pump.basis` is the single source of truth;
`resolve_pump_basis` with `policy="positive_odd_jc"` gives the odd list.

Biased/DC/3WM devices break that symmetry and need `dense_real` plus a DC
solution.

---

## Shipped designs

Eight IPM circuit artifacts under `designs/`, ready to run:

| Directory | Notes |
| --- | --- |
| `ipm_2c_fixed` | the live 2-coupler device, 6136 nodes, 2508 junctions |
| `ipm_3c_fixed` | 3-coupler |
| `ipm_7c_fixed`, `ipm_7c_new`, `ipm_7c_old`, `ipm_7c_lj158_cg66`, `ipm_7c_oldlen_newcl` | 7-coupler variants |
| `ipm_7c_ideal_m25db_8ghz` | built with the geometryless `ideal` coupler |

These are the authoritative circuits. Anything you generate under `outputs/` is
yours; do not treat an old output directory as a design.

---

## Building circuits

Four builders, documented in full in
[`docs/circuit_builders.md`](docs/circuit_builders.md):

| Module | Devices |
| --- | --- |
| `twpa_solver.builders.ipm` | IPM / 2c / 3c / 7c coupled-line JTWPA |
| `twpa_solver.builders.jc_doc` | JPA, DPJPA, FXJPA, JTWPA, FQJTWPA, FQJTWPA_diss, FXJTWPA |
| `twpa_solver.builders.le_gal_2025` | effective-SNAIL line |
| `twpa_solver.builders.scattered` | Lj-scattered copy of an existing design |

```powershell
python -m twpa_solver.builders.ipm --outdir outputs/d1 --write-matrices --coupler-mode cached
python -m twpa_solver.builders.jc_doc --outdir outputs/jc --cases jc_jpa jc_jtwpa
```

The three IPM coupler modes — `cached`, `optimize`, `ideal` — are **not
interchangeable**. A design built with one is not reproduced by another.

### Per-cell profiles and scatter

`Lj` and `Cg` can vary cell by cell across arbitrary blocks, with `Cj` derived
so the plasma frequency stays constant, plus independent percentage scatter on
all three. Full reference:
[`docs/component_profiles_and_scatter.md`](docs/component_profiles_and_scatter.md).

```powershell
python -m twpa_solver.builders.ipm --outdir outputs/tapered --write-matrices `
  --coupler-mode cached `
  --lj-profile "rows=0-2:const:150p" `
  --lj-profile "rows=3-5:linear:123.9p->140p" `
  --cg-profile "all:half_cosine:66f->72f" `
  --lj-scatter-sigma 0.01 --cj-scatter-sigma 0.005 --cg-scatter-sigma 0.02 `
  --scatter-seed 7
```

To re-emit one of the shipped designs with a profile applied:

```python
from twpa_solver.builders.ipm import build_variant_design
from twpa_solver.builders.profiles import parse_profile_shorthand as shorthand
from twpa_solver.builders.scatter import ScatterSpec

build_variant_design(
    "designs/ipm_2c_fixed", "outputs/tapered_2c",
    lj_segments=[shorthand("all:linear:123.9p->140p")],
    lj_scatter=ScatterSpec(0.01), seed=3, overwrite=True,
)
```

---

## Workflows

End-to-end entry points live in [`workflows/`](workflows/) and compose the
solver and plotting backends. Detail in [`docs/workflows.md`](docs/workflows.md).

### 1. Design + passive response

```powershell
python workflows/build_design_and_passive.py --design-dir designs/ipm_2c_fixed
```

Writes `passive_sparameters.npz` and the S21/S24 and S11/S21/S31/S41 figures
into the design directory. Convention is `S[frequency, output_port,
source_port]`. Pump is off, so this is the linear response — the first thing to
check when a device behaves oddly.

### 2. Gain map over pump power and frequency

`scripts/run_gain_map.py` is the orchestrator;
`workflows/run_gain_map_and_plots.py` wraps it and produces the standard plot
catalogue.

```powershell
python workflows/run_gain_map_and_plots.py `
  --design designs/ipm_2c_fixed `
  --run-dir outputs/gain_map_2c `
  --n-power 20 --n-frequency 20 `
  --pump-power-min-dbm -35 --pump-power-max-dbm -23 `
  --pump-freq-min-ghz 7.5 --pump-freq-max-ghz 8.5 `
  --pump-port 4 --source-port 1 --out-port 2 `
  --pump-mode-policy positive_odd_jc --pump-mode-count 10 `
  --sidebands 10
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--executor {inprocess,subprocess}` | `inprocess` is the default and the fast path |
| `--mode {cold,warmstart,both}` | warm-start each cell from its neighbour |
| `--traversal {column,backbone,nearest,serpentine,floodfill}` | grid traversal order |
| `--signal-backend {direct,schur}` | use `direct` for large maps; `schur` densifies and can exhaust memory |
| `--frequency-chunk-size` | parallel chunking; forced to 0 by non-column traversals |
| `--attenuation-db` | flat line loss; omit to use the measured model |

Pump power is converted to on-chip **peak** current after subtracting line
loss. The validated conversion is `--pump-current-jc-scale 1.0`; the parser
default of `2.0` is a historical parity convention, so pass it explicitly.

Prune a finished map before archiving:

```powershell
python scripts/prune_map_solutions.py <run-dir> --top-k 100 --purge-point-dirs --apply
```

### 3. Signal spectrum at one pump point

```powershell
python workflows/run_signal_spectrum.py --design designs/ipm_2c_fixed --run-dir outputs/spectrum
```

### 4. Compression / P1dB

`scripts/run_compression.py` sweeps signal power at a fixed pump point and
locates the 1 dB compression point by nonlinear solves inside a bracket.

```powershell
python scripts/run_compression.py --fixture jtwpa --signal-ghz 6.6 --output-dir outputs/compression_jtwpa

python scripts/run_compression.py --circuit-dir designs/ipm_2c_fixed --signal-ghz 7.44 `
  --pump-freq-ghz 7.1 --pump-port 4 --output-dir outputs/compression_2c
```

| Flag | Meaning |
| --- | --- |
| `--fixture {jpa,jtwpa,fqjtwpa}` | built-in reference devices, no circuit dir needed |
| `--multitone-basis {matched,three_tone,lattice}` | `matched` retains pump harmonics and is the default |
| `--multitone-sidebands` | basis size; memory scales as `(n_pump_modes + 2S + 1)^2` |
| `--multitone-backend {auto,full,schur_cpu_mt}` | `schur_cpu_mt` for loaded circuits |
| `--signal-workers` | capped automatically against free RAM |
| `--p1db-power-tol-db` | 0 falls back to log-linear interpolation |
| `--check-stability` | without it, `stability_status` stays `NOT_CHECKED` |

Signal frequency is mandatory for a single run. Fixtures default to zero line
attenuation; loaded circuits use the measured loss model unless
`--attenuation-db` is given.

### 5. Comparing against measurement

The Themis measurement cubes ship under `docs/development/*Themis*/`.

```powershell
python scripts/align_map_to_measurement.py --help
python scripts/compare_map_to_measurement.py --help
```

`align_map_to_measurement.py` fits the calibration offsets `(df, dP, dG)` as
nuisance parameters rather than hand-tuning them, and writes a four-panel
comparison including the loss surface. Fit one band at a time — per-section
fits are far better identified than a whole-map fit, which has to compromise
between comb lobes.

---

## Line loss

Pump power → on-chip current uses a measured insertion-loss fit, not a flat
value:

```
att_dB(f) = 27.3882 + 0.4579*sqrt(f) + 0.8354*f     (f in GHz)
```

The constant is fixed coupling loss, `sqrt(f)` skin effect, `f` dielectric;
RMS 0.37 dB against `docs/loss_A10.csv`. The constant is required — the data
show ~26 dB at f=0, and a pure `A*sqrt(f)+B*f` fits terribly. At 8 GHz the
model gives ~35.4 dB, matching the older band-calibrated flat value.

`InsertionLossModel` / `default_loss_model()` in `twpa_solver.loss` expose it;
`InsertionLossModel.fit_csv` re-fits from the CSV.

---

## Repository layout

```
src/twpa_solver/
  core/        CircuitMatrices, load/save, branch laws
  builders/    ipm, jc_doc, le_gal_2025, scattered, profiles, scatter
  pump/        harmonic-balance pump solve, basis policy, preconditioners
  signal/      small-signal gain, Floquet, passive S, quantum efficiency
  multitone/   large-signal compression, observables, stability
  plotting/    figure backends
  parity/      cross-check helpers
  loss.py      measured insertion-loss model

workflows/     end-to-end entry points (start here)
scripts/       drivers, campaigns, plotting, measurement comparison
designs/       eight live IPM circuit artifacts
docs/          reference documentation and measurement data
tests/         the gates
experiments/   the few legacy modules the drivers still import
```

---

## Branches

| Branch | Contents |
| --- | --- |
| `main` | the production surface: solver, designs, docs, workflows, tests |
| `dev` | everything, including diagnostic dumps, the full experiment history, and development notes |

Work on `dev` if you need the historical experiment scripts or the raw
diagnostic artifacts. `main` is what you hand to someone who wants to use the
solver.

---

## Testing

```powershell
python -m pytest -q                                    # fast suite
python -m pytest -q --run-slow                         # + dense HB physics gates
python -m pytest -q --basetemp D:\tmp\twpa --run-slow  # Windows-safe scratch dir
```

Running without `--run-slow` is not complete validation.

---

## Known issues

- **`tests/test_column_matrices_tracer.py`** — 2 failures, pre-existing and
  unrelated to the solver. They concern the diagnostic matrix tracer, not any
  physics path.
- **`tests/physics/test_compression_low_signal_limit.py`** is an expected
  failure: the default JPA fixture has not been found at >3 dB gain.
- **`test_fxjtwpa_node_order.py`** skips unless the FXJTWPA seed artifacts are
  present; those are generated, not shipped.

## Cautions

Conclusions from measurement, not style preferences.

- **JosephsonCircuits.jl is not a physical reference.** It is another
  simulator, and two of the seven parity designs are its own documentation
  examples, so gating against it was circular. Agreement with JC measures
  numerical drift between two codes — a useful regression check, nothing more.
- **Production sideband bases are not self-converged.** JTWPA gain is
  non-monotone in sideband count (30.7, 24.2, 26.6, 27.5 dB at S=2,4,6,10), so
  S=10 cannot be selected by agreement. Treat every published P1dB as carrying
  an unquantified basis-truncation uncertainty.
- **Saturation has no external reference.** Its correctness rests on
  Manley–Rowe, power balance, basis self-convergence, and the small-signal
  Floquet limit — not on any measurement.
- **Quote damping rates against `omega_p`.** A bare stability exponent is not
  interpretable.
- **`--check-stability` is opt-in.** A deep-saturation solution without it is
  not a stability claim.

---

## Further reading

| Document | Covers |
| --- | --- |
| [`docs/solver_architecture.md`](docs/solver_architecture.md) | how the three solve stages fit together, and where to make a change |
| [`docs/circuit_builders.md`](docs/circuit_builders.md) | all four builders in detail |
| [`docs/component_profiles_and_scatter.md`](docs/component_profiles_and_scatter.md) | per-cell profiles, shapes, scatter streams |
| [`docs/workflows.md`](docs/workflows.md) | the end-to-end entry points |
| [`docs/pump_current_conversions.tex`](docs/pump_current_conversions.tex) | the two pump-current conventions |
| `CLAUDE.md` | working notes, measured results, and open questions |
