# twpa_jax — agent notes

## Language standard

Use ATS technical English in all conversations and documentation. Write in
plain, precise, and unambiguous technical language. Avoid idioms, marketing
language, unnecessary theory, and unexplained abbreviations. Use terminology
consistently in user-facing documentation, implementation notes, and status
updates.

## Declarative circuit designs

See `docs/design_format.md` for the schema and compiler boundary. Nested
`repeat` is limited to depth two. Compiler cursor collisions are hard errors,
and files under `designs/*.yaml` describe concrete devices rather than
parametric templates.
The checked-in `designs/` tree is source-only; generated matrices, plots, and
resolved artifacts belong under `outputs/` or another disposable build path.

## Branch and artifact policy

`main` is the clean, installable package branch. `dev` is the development and
cross-machine synchronization branch and may contain experiments, generated
artifacts, and incomplete work. `.gitignore` is not a branch boundary: ignored
files are not synchronized, and tracked files merge normally.

Never merge `dev` wholesale into `main`. Keep production changes in focused
commits and promote only those commits with `git cherry-pick` or a manually
curated merge. Before promotion, check that development-only paths are absent
from the destination. Generated outputs belong under ignored `outputs/` (or
another disposable path); source design YAML belongs under `designs/`.

## Design-independent saturation validation status (2026-07-29)

Do not use JosephsonCircuits.jl or the Themis measurement cube as physical
references for finite-signal saturation. JC is regression-only; Themis has no
signal-power axis. The CSV field `compression_model_depletion_only` is the
depletion model's dB power gain. The underlying
`depletion_only_model` API still returns linear gain.

The design-independent campaign is incomplete. The existing JVP and lossless
observable tests are numerical/unit checks, not external physical validation;
the current lossless Manley–Rowe test is degenerate because it has no generated
signal/idler photon-flux scale. Do not report the eight-check campaign as
passed until the finite-signal lossless fixture and the requested measured
slopes/figures have been run.

The driver now uses voltage-per-input-current ratios for `gain_vs_off_db`,
matching Floquet normalization; absolute one-port `S` values are affine and
must not be subtracted to form gain-vs-off. The sideband basis builder also
allocates `omega_max` as a function of requested sideband count, so production
S=10 bases are not clipped. A serial JPA lattice-Q=3 measurement found sector
slopes near `1.01, 2.02, 3.03`, while its gain curve was non-monotone. JPA JVP
finite differences gave slope `1.9414` and minimum error `1.566e-12`. The
production JTWPA S=10 run exceeded 300 s without an artifact; S=10/S=12 basis
self-convergence remains unevaluated. Manley–Rowe remains unresolved.
The Manley–Rowe output now includes its photon-flux scale and marks
sub-`1e-28` cancellation-dominated points as not evaluable instead of turning
the denominator floor into a physical error.

Package `twpa_solver` (under `src/`) is the production solver; `scripts/run_gain_map.py`
is the pump/gain-map orchestrator. The solver was extracted from the `experiments/`
research scripts (exp08 pump solve, exp09 gain, exp10 maps, exp14 parity), which
now serve as validation provenance — most notes below apply to the solver modules.

Run the complete test suite, including compression and distributed multitone
physics gates, with a temporary directory outside the repository:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow
```

## Production multitone compression campaigns (exp20-exp22)

### Phase 5 basis-convergence status (2026-07-29)

`scripts/multitone_convergence_study.py` now accepts `--device jtwpa|fqjtwpa` and
uses the Phase 3 bracketed nonlinear P1dB refinement for every setting. The
required matrix includes Q=1/2/3, pump-order, torus-scale, three-tone versus
lattice, and odd-only versus dense pump modes.

Two earlier blockers are fixed: the study hardcoded the plain `real_coupled`
preconditioner (a fresh `splu` every Newton step) and buffered every result to
write the CSV once at the end, so an overrun produced no output at all. The
multitone solves now use `real_coupled_fast` (pump solve stays `real_coupled`,
production-identical): measured 8.53 s -> 3.51 s on one jpa setting, P1dB
bit-identical at -116.179752. The CSV is written per setting, and
`--per-setting-budget-s` records `TIMEOUT`/`FAILED`/`OOM` rows instead of
stalling or losing the matrix.

jtwpa matrix, 8 of 9 settings (`outputs/phase5_convergence/jtwpa_full.csv`,
banded backend, 15 coarse points, refined P1dB):

| knob | top two settings | \|ΔP1dB\| | verdict |
| --- | --- | ---: | --- |
| signal_order_max Q | Q=2 -110.500076, Q=3 -110.440913 | **0.059163 dB** | PASS |
| torus_scale | sc=1 -110.440913, sc=2 -110.440913 | **0.000000 dB** | PASS |
| pump_order | order=1 -108.784341, order=3 -110.440913 | 1.656572 dB | **not evaluable** |
| odd vs dense | dense order-5 did not run | — | **not evaluated** |

`torus_scale` is converged exactly -- doubling `n_p`/`n_delta` moves nothing.
The `pump_order` number spans a **gap**: `odd` order 2 (modes `[1,3]`) fails its
pump solve reproducibly in ~1.4 s across two independent runs, while `[1]` and
`[1,3,5]` both converge, so 1-vs-3 are not adjacent settings and 1.66 dB is not
a convergence measurement. The dense order-5 point (the plan's **mandatory**
odd-vs-dense comparison) exhausted memory on this 7 GB machine. Consistency
check that did pass: `three_tone` and `lattice` Q=1/order=1 agree to all six
reported decimals (-112.748281), as they must, being the same basis.

**This does not discharge the production-basis caveat.** Every setting in this
matrix solves at 3.756-7.894 dB small-signal gain against production's 27.541 dB,
and the study builds `build_lattice_basis` where production uses
`build_sideband_matched_basis` at S=10. It converges the lattice family at a much
weaker operating point. The exp20/21 production basis still must not be described
as having passed the 0.2 dB gate.

`scripts/run_compression.py` defaults to the pump-harmonic-retaining
`--multitone-basis matched --multitone-sidebands 2`. A three-tone basis is only
valid with a fundamental-only pump basis; the driver raises if any pump `(h,0)`
mode would be silently dropped. Signal frequency is mandatory for a single run.
Fixture runs default to zero line attenuation; loaded production circuits use
the Themis loss model unless `--attenuation-db` is explicit.

Reproduce the four-device compression curves and S=2/S=4 P1dB check:

```powershell
python experiments/exp20_multitone_compression.py --output-dir outputs/exp20_multitone_compression_converged
python experiments/exp20_summary.py --input-dir outputs/exp20_multitone_compression_converged --output-dir outputs/exp20_summary_converged
```

Run frequency-resolved P1dB and spatial attribution campaigns:

```powershell
python experiments/exp21_p1db_vs_frequency.py --output-dir outputs/exp21_p1db_vs_frequency_converged --signal-workers 4
python experiments/exp22_spatial_attribution.py --output-dir outputs/exp22_spatial_attribution_converged
```

### P1dB refinement versus interpolation (measured 2026-07-29)

`--p1db-power-tol-db` (default 0.1) locates P1dB by nonlinear solves inside the
coarse bracket; 0 falls back to log-linear interpolation. The driver now emits
**both** numbers from one sweep (`p1db` and `p1db_interpolated_dbm`), so the
delta is single-variable. Validation: each run's interpolated value reproduces
the published exp20 number to 5.1e-9 dB (jtwpa) and 6.0e-9 dB (2c).

| device | published exp20 | refined | delta |
| --- | ---: | ---: | ---: |
| jtwpa 6.6 GHz | -111.458017 | -111.118089 | +0.339928 dB |
| 2c 7.440816 GHz | -95.083826 | -94.857885 | +0.225941 dB |

This **supersedes an earlier +0.461 / +2.920 dB claim**, which was measured on a
nine-point coarse grid that no published number used; that grid put 2c's first
crossing 8.4 dB off, making its "+2.920" a grid artifact. The real errors are
0.23-0.34 dB: above the 0.2 dB scale, single-signed (refined is always higher,
so devices compress later than published), and *not* grounds for re-running
exp20/21 wholesale. See `docs/development/saturation_solver_p1db_measurement.md`.

Two diagnostics emitted alongside are **not yet trustworthy**:
`manley_rowe_rel_err` peaks at the *smallest* signal power (0.533 jtwpa, 0.500
2c -- near-identical across very different devices, so a fixed factor rather
than physics). The depletion-model CSV unit defect is fixed: it now emits the
plan model's power gain in dB and exp22 consumes that value without a second
conversion.

### Finite-signal stability (measured 2026-07-29)

Runs without `--check-stability` keep `stability_status = "NOT_CHECKED"`;
deep-saturation solutions are not stability claims.

The previously reported "stable, dominant exponent +2.87e-6 s^-1" is
**withdrawn**. `multitone/stability.py` was passing pump-harmonic keys where
`assemble_conversion_matrix` expects a sideband ladder, and its sigma_min was
owned by a near-DC sideband: measured bit-identical with the pump on and off up
to psi/phi0 = 57000. The committed tests could not catch it because they ran on
`Ic=0` -- a linear circuit -- at a zero state. Both defects are fixed and
mutation-verified; the near-DC case now returns INCONCLUSIVE with a reason
rather than STABLE. Production points are clear of the guard (closest sideband
0.52 GHz for jtwpa, 0.10 GHz for 2c, against ~7 MHz thresholds).

Measured at the three exp22 operating points, all exponents negative (decaying),
now state-dependent as they must be:

| device | zero signal | P1dB | deepest saturation |
| --- | ---: | ---: | ---: |
| jtwpa | -3.3785e+08 (STABLE) | -1.7638e+08 (INCONCL) | -1.1316e+08 (INCONCL) |
| 2c | -1.1034e-02 (STABLE) | -1.6045e-02 (STABLE) | -1.8755e-02 (STABLE) |

**No unstable point was found.** Always quote these against omega_p: jtwpa's
|sigma|/omega_p ~ 8e-3 is real damping, 2c's ~4e-13 is numerically marginal. A
bare exponent is not interpretable. jtwpa's INCONCLUSIVE points are
`refine_complex_resonance` not converging, not the near-DC guard. Details in
`docs/development/saturation_solver_stability_measurement.md`.

Final exp20 evidence (2026-07-29):

| device | S | gain dB | gain - JC dB | input P1dB dBm | depletion at P1dB dB | recovery rungs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| jpa | 2 | 13.305144 | +0.002813 | -132.732343 | -3.368085 | adaptive 1, previous 24 |
| jtwpa | 10 | 27.541036 | +0.000836 | -111.458017 | -0.351776 | adaptive 1, previous 20, rescaled 2, substep 2 |
| fqjtwpa | 6 | 28.534877 | -0.001823 | -107.386000 | -0.598642 | adaptive 1, previous 23, substep 1 |
| 2c | 10 | 15.916611 | no JC reference | -95.083826 | -0.622737 | adaptive 1, previous 24 |

Final exp21 directories are under
`outputs/exp21_p1db_vs_frequency_converged/{jtwpa,fqjtwpa,2c}`. JTWPA reached
P1dB at 8/10 frequencies (minimum -112.280936 dBm at 6.888889 GHz), FQJTWPA at
8/10 (minimum -107.881062 dBm at 7.533333 GHz), and 2c at 10/10 (minimum
-94.011690 dBm at 7.422222 GHz). `NO_GAIN_AT_OPERATING_POINT` is retained at
the dead-band frequencies rather than reporting a P1dB.

Final exp22 spatial artifacts are under
`outputs/exp22_spatial_attribution_converged`: JTWPA contains 6,141 rows
(2,047 branches x three operating points) and 2c contains 7,524 rows (2,508 x
three). The 2c nonlinear branches are monotone but segmented by five intentional
node-number gaps; spatial mapping validates order, not literal node adjacency.

### Multitone preconditioner backend (measured 2026-07-28)

`--multitone-preconditioner real_coupled_fast` (the default) now really is fast:
it routes to `pump.backends.fast_coupled.FastCoupledPreconditioner`, which caches
the scatter map + symbolic factorization and reruns only the numeric factor per
Newton step. It previously forwarded to the plain
`assemble_real_coupled_preconditioner`, i.e. a fresh `spla.splu` every step.
Both the Schur backend (`multitone/schur.py`, cached on the partition) and the
full backend (`multitone/problem.py`, cached on the `cache` field that survives
`dataclasses.replace`) use it. Identical matrix, verified to 1.0e-11 relative.

jtwpa S=10, per Newton step, single process:

| path | before | after |
| --- | --- | --- |
| full backend (`--fixture`) | 6.148 s | 0.93 s (6.6x) |
| schur backend (`--circuit-dir`) | 5.004 s | 1.00 s (5.0x) |

`pypardiso` is present (0.4.7) and is the default factor backend. SuperLU still
works via the automatic fallback but is a bad trade at this size — measured on
jtwpa S=10: refactor 3.94 s vs 0.92 s (4.3x slower, because `spla.splu` has no
symbolic-reuse API and this problem refactors every Newton step) for only 7% less
peak RSS (2.81 vs 3.01 GB). Since peak sets the worker count and both give the
same count, PARDISO wins ~3.8x at equal concurrency. Set `TWPA_REQUIRE_PARDISO=1`
to make a silent fallback a hard error during timed campaigns.

The assembly scatter used to be a sparse matrix `W` of shape `(M.nnz, 2*n_ells*
nnzk)` so assembly was one spmv. It carried no information beyond target
indices -- every stored value is +-1 and the source indices are contiguous
per-ell ranges -- and cost ~630 MB at S=10 to hold 94 MB of indices. It is now
four int32 target arrays per (mode, mode) block (`_index_contributions`), which
cut peak RSS 3.04 -> 2.51 GB for +87 ms of assembly (scattered writes replace a
sequential-write spmv). Gate: `tests/test_fast_coupled_assembly.py` compares the
assembled matrix against `problem.real_coupled_matrix`, per quadrant, since a
dropped sign would only show as a slower preconditioner and no physics gate
would catch it.

**Memory scales as (n_pump_modes + 2S + 1)^2**, not as the packed dimension: the
coupled Jacobian is block-dense in tone index. Same jtwpa circuit, same n=2048,
gain-map pump (H=10) vs multitone S=10 (H=31): M.nnz 2.46M -> 23.6M, exactly the
9.6x H^2 ratio, while the dimension grew only 3.1x. The gain map is cheap because
`signal/floquet.py::solve_gain_one` treats the sidebands as a separate *linear*
factor-once system; saturation cannot, since the sidebands act back on the pump.
Per-worker peak: 2.80 GB at S=10, ~1.6 GB at S=6, ~0.9 GB at S=2.

`scripts/run_compression.py --signal-workers` is capped by
`multitone.resources.fast_coupled_footprint` against BOTH `--resource-budget-gb`
and actual free RAM. The old cap was a hardcoded 3.0 GB/worker independent of
basis size, which underestimated S=10 by 38% and OOMed a 15.3 GB machine at
2 workers. Tests: `tests/test_multitone_resources.py` pins the estimate to the
measurement and asserts it never reads below it.

`--precond-reuse N` / `--precond-reuse-refresh-gmres` expose modified-Newton
preconditioning (reuse one factor across N Newton steps; the update is always
taken against the true Jacobian, so the converged solution cannot change).
Default 1 = refactor every step. **Measured 2026-07-29: N>1 is a large net
loss, leave it at 1.** On the jpa compression fixture, N=2 saved 26 of 59
factorizations but cost 841 extra GMRES iterations (243 -> 1084); N=3 saved 29
and cost 986. The exact preconditioner converges GMRES in ~3 iterations, and one
GMRES iteration (jvp + triangular solve, 303 ms at S=6) costs about as much as
one factorization (346 ms) -- so there is no cheap-preconditioner regime to
amortize, and a stale exact factor is far worse than a fresh one. Wall time on
jpa: 3313 ms (N=1) vs 3574 (N=2) vs 3545 (N=3); gain identical to all digits.
The pump solve is pinned to `precond_reuse=1` so its iterate path is unchanged.

### Factor backends and worker count (measured 2026-07-29)

`--factor-backend {pardiso,banded}` (default `pardiso`). `banded` reorders the
coupled Jacobian **node-major** (`node * 2*n_tones + super_block`) and stores the
factors as a LAPACK general band instead of a general sparse LU. Packed
tone-major the matrix spans everything; node-major it collapses onto a band
~3 tone-blocks wide, because the device is a 1-D chain. That is a property of
the circuit, not of an ordering search -- RCM does worse (max bandwidth 147 vs
89 on jtwpa S=2). Bandwidth is measured from the assembled pattern, never
assumed. jtwpa S=10, 3 power points, single process:

| backend | peak RSS | wall | workers in 7 GB | net throughput |
| --- | ---: | ---: | ---: | ---: |
| pardiso | 2.51 GB | 147.7 s | 2 | 1.55x |
| banded | 1.84 GB | 173.7 s | 3 | **1.95x** |

Converged gain agrees to 4.4e-10 dB (27.541036124719 vs 27.541036124280), as it
must -- this only changes the preconditioner. `banded` is worth it only when the
smaller footprint buys another worker; for a single-frequency run (exp22) it is
a pure 1.18x loss, so the default stays `pardiso` and exp21 passes it
explicitly. Worker throughput does not scale linearly -- measured on jtwpa S=2,
1/2/3/4/6 workers give 1.00/1.55/2.30/2.00/2.41x (+-15% run-to-run), so it is
bandwidth-bound and plateaus around 3 workers on this 6-core part.

**PARDISO is pinned to one thread** (`TWPA_PARDISO_THREADS`, default 1) because
MKL intermittently fails reordering with error -3. That pin costs single-run
latency: at S=6 the factor is 340 ms at 1 thread, 228 ms at 2 (1.49x), and
**1374 ms at 6** -- MKL PARDISO at high thread counts on this AMD part is 4x
slower than serial. Raising it only helps when running one worker; with several
workers, extra threads lose to extra workers (3 workers x 1 thread = 5.00 cyc/s
vs 3 x 2 threads = 3.66).

**The sideband-selection justification is void as of 2026-07-29.** Production
bases (jtwpa S=10, fqjtwpa S=6, 2c S=10) were picked by gating small-signal gain
against a JosephsonCircuits.jl reference. JC is another simulator with no
reference of its own, and `jc_jtwpa`/`jc_fqjtwpa` are JC's own documentation
designs, so that gate was circular; the user has retired it. Do not cite the
"0.2 dB JC-reference gate", and do not present JC agreement as validation --
it measures numerical drift between two codes, which is a useful regression
check and nothing more.

What this leaves unresolved: JTWPA gain is **non-monotone in S** (30.7152,
24.2021, 26.5563, 27.5410 dB at S=2,4,6,10). Without an external reference,
S=10 cannot be selected from that sequence by agreement -- it requires a
self-convergence argument (S=10 against S=12/14) that has never been run, and
non-monotonicity means S=10 may still be climbing. 2c never had a JC reference
at all and inherited S=10 by analogy. Same-S multitone/Floquet parity remains a
valid internal check and should still pass.

Until production-basis self-convergence is measured, treat every published P1dB
as carrying an unquantified basis-truncation uncertainty.

**2c basis self-convergence, measured 2026-08-06** (`experiments/exp54_basis_self_convergence.py`,
`outputs/exp54_basis_self_convergence/basis_convergence_full.json`): S=10 vs
S=12 vs S=14 on `designs/ipm_2c_fixed`, fixed operating point fp=7.100 GHz,
Ip=7.2311e-6 A, three signal frequencies (6.0/7.2/8.4 GHz), full
`--stop-after-p1db` nonlinear search at each point. **Converged**: G0 and P1dB
agree to <1e-6 dB between S=10/12/14 at 6.0 and 7.2 GHz; at 8.4 GHz the
refined nonlinear P1dB solve itself failed to converge at S=12/14
(`p1db_state_status=SIGNAL_CONTINUATION_FAILED`, falls back to the smooth
interpolated P1dB) but that fallback value still agrees to 5e-9 dB between
S=12 and S=14, so the basis truncation itself is not the source of any
disagreement. This does not touch the JTWPA gap above (different device,
non-monotone in S) but does resolve "2c inherited S=10 by analogy" — 2c's
S=10 choice at this operating point is now directly verified, not inherited.

Same campaign reran the gain-matched P1dB-vs-measurement comparison
([[2c-model-compresses-early-confirmed]]) at 100 MHz spacing (13 points,
6.0-7.6 GHz) instead of the earlier ~200 MHz/18-point grid:
**mean delta -18.20 dB, median -18.75 dB** — reproduces the earlier -18.25 dB
finding almost exactly at 2x the frequency resolution. The model still
compresses ~18 dB earlier than the hardware at matched gain; this is not a
basis-truncation artifact (S=10 is converged) and not a frequency-resolution
artifact (denser grid gives the same number). A first pass at this rerun (via
`experiments/exp53_gain_matched_p1db_comparison.py`) gave a spurious +41 dB
from comparing the model's external/instrument-referred `p1db` field directly
against the measured table's on-chip (loss-subtracted) power — the model
field already has the run's attenuation added back on top of on-chip power
(`_current_to_dbm`, `run_compression.py:444-448`); fixed by subtracting the
run's own recorded `attenuation_db` before comparing on-chip to on-chip.

The earlier 2c blocker was a settings/wiring mismatch, not a fold or intrinsic
continuation wall. The gain-map pump is injected at port 4 while signal
scattering is 1 -> 2; using port 1 for both gave a promoted-pump residual of
sqrt(2). With `--pump-port 4`, the saved pump residual is 7.97e-12. The S=10
Schur multitone point converged adaptively at lambda=0.25/0.625/1.0 in
82.1/108.7/224.7 s with final coefficient residual 1.45e-14, producing
15.9166 dB small-signal gain at 7.440816 GHz. Production circuit runs default
to `--multitone-backend schur_cpu_mt`; fixtures retain the full backend.

Running without `--run-slow` is not complete validation.

Module map: pump solve `twpa_solver.pump.hb` + `twpa_solver.pump`
(`HarmonicNewtonKrylovSolver`, `NewtonKrylovSettings`, `FullPumpProblem`); pump
basis `twpa_solver.pump.basis`; gain `twpa_solver.signal`; circuits
`twpa_solver.core`; loss `twpa_solver.loss`.

## Line loss model (`src/twpa_solver/loss.py`)

`run_gain_map.py` converts pump dBm → on-chip peak current after subtracting
line loss. Loss defaults to the measured `docs/loss_A10.csv` fit, not a flat
35 dB:

    att_dB(f) = 27.3882 + 0.4579*sqrt(f) + 0.8354*f    (f in GHz)

`InsertionLossModel` / `default_loss_model()` expose it; `InsertionLossModel.fit_csv`
re-fits the CSV. C = fixed coupling loss, A*sqrt(f) = skin effect, B*f = dielectric
(RMS 0.37 dB). The constant C is required — the CSV has ~26 dB loss at f=0, so a
pure `A*sqrt(f)+B*f` fits terribly (RMS 4.6 dB, B<0). Sanity: model at 8 GHz ≈
35.4 dB, matching the old band-calibrated flat 35 dB.

`default_loss_model()`/this A10 fit is now named `pump_line_loss_model()` — it
is the PUMP feedline's loss, not a generic loss. The SIGNAL feedline is a
separate, physically distinct line: `signal_line_loss_model()` fits
`docs/development/loss_B1.csv` (`att_dB(f) = 50.0 + 3.3*sqrt(f) + 0.14*f`, RMS
2.80e-5 dB). Before 2026-08-05, `scripts/measured_psat_pipeline.py` subtracted
a fabricated flat `SIGNAL_LINE_LOSS_DB = 72.5` from the Themis cube's signal
axis instead — off by +9.4 to +15.3 dB across the band and erasing a real
5.95 dB tilt (still present in `experiments/exp30_themis_map_and_pump_inference.py`
and `exp45_curves_two_way.py`, both unfixed legacy scripts —
`outputs/presentation/2c_themis_map.png` is NOT on the corrected calibration).
Which physical line gets which model is forced by energy conservation, not a
free fit choice: signal+idler output at compression cannot exceed the pump
(`P_sat + 3 dB <= P_pump`), and only `pump_line_loss_model()` on the pump
feedline satisfies it — `signal_line_loss_model()` there gives P_sat 16.5 dB
*above* the pump, which is impossible. See
`docs/development/psat_comparison_fix_plan.md` Phase 2 and
`scripts/measured_psat_pipeline.py::energy_conservation_gate` (Phase 7), which
now asserts this on every run instead of trusting it.

## Port power convention: matched travelling-wave, not Norton (reverted 2026-08-06)

`src/twpa_solver/ports.py` is the single source of truth for current<->dBm.
`designs/ipm_2c_fixed`'s `G` matrix has **exactly four nonzeros**, all
`0.02 S = 50 Ω`, one per port — every drive is an ideal current source in
parallel with `G0 = 1/Z0`. This topology is ambiguous between two standard
port conventions: a **Norton generator** (the injected current is the
source's own short-circuit current, splitting `I/2` across a separately
matched load, `P = I^2 Z0/8`) or a **matched wave port** (the injected
current *is* the incident wave's own amplitude, `G0` only absorbing
reflections, `P = I^2 Z0/2`). The `G` stamp alone cannot distinguish them.

Default is **`legacy_traveling_wave`** (`P = I^2 Z0/2`), per the design intent
confirmed 2026-08-06: `I` is the incident wave amplitude. `convention="norton"`
remains selectable for comparison but is not physically justified for the
current topology.

**Open inconsistency, not yet resolved:** `multitone/observables.py::extract_port_waves`
(lines ~72-75) independently derives the *outgoing* wave from the actual
solved node state via KCL as `current = injected_current - voltage/z0_ohm`
— i.e. it subtracts the port resistor's own draw, which is the Norton
picture, not the travelling-wave one. This was not changed in the revert.
The two pictures agree only once the separate port resistor is actually
removed from the netlist (planned, not yet done) — until then, the
source-side current-to-power conversion (`ports.py`, now travelling-wave) and
the solved-state wave extraction (`observables.py`, still Norton-KCL) are
using two different physical pictures of the same port. Do not "fix" either
side unilaterally without redoing this section.

`port_available_power_w(current_a, z0_ohm)` / `port_current_from_power_a(...)`
are the conversion functions; `convention="norton"` reproduces the
2026-08-05..06 Norton-era numbers bit-for-bit (`LEGACY_TW_OFFSET_DB =
10*log10(4)` is still the exact offset between the two, now in the opposite
direction: Norton reads `-6.0206 dB` relative to the default). `--power-convention`
(default `legacy_traveling_wave`) is wired through `scripts/run_compression.py`,
`scripts/run_gain_map.py`, and `src/twpa_solver/loss.py::dbm_to_peak_current_a`.
Gain (`gain_vs_off_db`) is a pump-on/pump-off ratio, invariant under the
source-scale convention, so this only ever relabels absolute powers (P1dB,
saturation dBm, etc.), never gain itself.

**Superseded:** the "Norton, resolved 2026-08-05" entry that previously lived
here (default `norton`, `P = I^2 Z0/8`) is reverted. That entry's own
evidence — `pump_outgoing_power_w` matching `I^2 Z0/8` to the last digit — is
still true and still in the code (it comes from the same unresolved KCL
subtraction in `observables.py` noted above), it just no longer determines
the *source-side* default. Do not cite the old entry as settled; the port
topology question is open until the resistor-removal work above lands.

**This supersedes "Pump-current conversion (validated 2026-07-18)"** — that
entry validated `--pump-current-jc-scale` against JosephsonCircuits.jl, a
factor-of-two check in *current* against another simulator
([[jc-is-not-a-reference]]); it never addressed whether the port termination
itself was Norton or travelling-wave, and `--pump-current-jc-scale`
(`docs/development/pump_current_conversions.tex`) remains a **separate,
orthogonal** knob layered on top of whichever power convention is selected —
do not conflate the two when reading that doc, which now carries a dated
addendum for this fix rather than being rewritten.

**Depletion cross-check (Phase 7,** `measured_psat_pipeline.py::model_depletion_cross_check`**):**
an energy-accounted depletion estimate (`(P_sat+3dB)/P_pump`, assuming
signal+idler output ~ P_sat + 3 dB) agrees with the solver's own
`pump_depletion_all_port_db` field to within 1.2-1.5x across a 7-point
provisional-operating-point sweep — see [[pump-power-norton-6db]] for the
resolved memory entry and current numbers.

`--attenuation-db` defaults to `None` (= use the model); pass a float to force a
flat value. Only `run_gain_map.py` is wired to the model; the `experiments/exp10_*`
scripts still use their local flat `dbm_to_peak_current_a`. Tests:
`tests/test_loss_model.py`.

## Pump-mode-policy layer (`twpa_solver.pump.basis`)

The harmonic-balance pump solve (`twpa_solver.pump.hb`, solver
`twpa_solver.pump.HarmonicNewtonKrylovSolver`) reconstructs the **real** pump
waveform with the JosephsonCircuits.jl (JC) positive-phasor convention:

    psi_pump(t) = 2 * Re sum_{k in modes} X_k * exp(+i k omega_p t)

`twpa_solver.pump.basis` (`resolve_pump_basis`, `PumpBasis`) is the single source
of truth for the pump-mode basis. The pump solve and the gain solve
(`twpa_solver.signal`) both consume it. `scripts/run_gain_map.py` drives both.

### Why this exists
JC's nonlinear pump for an unbiased 4WM device uses the **odd** mode list
`[1,3,5,...,2K-1]` (K = `Nmodulationharmonics`), e.g. `[1,3,...,19]` for the
JTWPA. The legacy code hardcoded dense harmonics `[1,2,...,H]`, which truncated
the high odd pump content and left a ~0.89 dB JTWPA gain mismatch vs JC. Using
the JC odd basis fixes it: **JTWPA gain RMS dropped to ~0.0006 dB.**

### Pump-solve knobs (`resolve_pump_basis` / pump-solve CLI)
- `policy`: `dense_real | positive_odd_jc | positive_phasor_explicit | auto_jc`
  (default `dense_real` preserves the legacy `[1..H]` behavior).
- `mode_count K` — for `positive_odd_jc` -> `[1,3,...,2K-1]`
  (`positive_odd_modes`).
- explicit modes `1,3,5,...` — for `positive_phasor_explicit`
  (`parse_explicit_modes`).
- promote-from an existing lower-basis solution
  (`promote_solution_to_basis`): shared modes copied, new modes zero-filled,
  then a single full-scale Newton solve (no continuation).
- `nt` must be `>= 2*max(mode)+1` (JC uses Nt=40 for max mode 19).

### Metadata persisted (pump_report.json metadata + pump_solution.npz)
`pump_modes`, `pump_basis="positive_phasor"`, `real_reconstruction_factor=2`,
`omega_p`, `phase_convention="exp_plus_i_k_omega_t"`, `pump_mode_policy`,
`pump_source_mode` (via `PumpBasis.to_metadata`). `pump_solution.npz` stores
`X_real`/`X_imag` as **float32, `savez_compressed`** (~2.1x smaller than the old
float64/uncompressed 1.5 MB/point — matters at 10k points/map), plus `pump_modes`
(and legacy `harmonics`). The gain solve reloads these via
`twpa_solver.pump.basis.load_pump_basis_from_solution`, which upcasts back to
complex128 (float32 would otherwise leak complex64 into scipy). float32's ~1e-7
relative precision is far below the ~1e-3 dB gain-map tolerance. Recompress legacy
maps in place with `scripts/recompress_pump_solutions.py <dir> --apply` (dry-run by
default, idempotent).

### Gain diagnostics (`twpa_solver.signal`)
`gamma_hat_summary.csv` — per-ell branch spectrum of
`gamma(t)=cos(psi_p/phi0)*Ic/phi0` (`compute_gamma_hat`):
`ell,nbranches,l2_abs,l2_abs_over_zero_l2,max_abs,mean_abs,mean_real,mean_imag,conj_symmetry_rel_err`.
For a correct real pump, `conj_symmetry_rel_err == 0` (gamma_hat[-ell] =
conj(gamma_hat[ell])).

### Quantum efficiency (`twpa_solver.signal.quantum_efficiency`)
`calc_qe(S, S_noise=None)` / `calc_qe_ideal(S)` are direct ports of
JosephsonCircuits.jl's `calcqe`/`calcqeideal`, vectorized (no separate `!`
in-place variant — numpy's row-sum already gives the cache-efficient behavior
the Julia loop hand-rolled). Both expect `S` in the **photon ladder-operator
basis**, not the classical voltage-ratio S this repo's `solve_gain_one(_schur)`
returns: signal and idler sidebands sit at different frequencies, so converting
requires the Manley-Rowe reweighting `S_ladder[m,n] = S_classical[m,n] *
sqrt(freq[n]/freq[m])` before calling calc_qe (see
`experiments/exp19_calcqe_validation.py::ladder_basis_weights`). Validated on
ipm_2c_fixed/jc_jtwpa/jc_fqjtwpa by building the 2x2 [signal,idler]x[signal,idler]
sub-matrix via two `solve_gain_one_schur` calls (excite signal_m, excite
idler_m); the resulting unitarity check `|S_ss|^2-|S_is|^2==1` (lossless
non-degenerate amp) holds to ~2%/~0.4% for ipm_2c_fixed/jc_fqjtwpa but is off
~47% for jc_jtwpa — expected, not a bug: a real unitarity check needs the full
multi-sideband S (jc_jtwpa solves 10 sidebands), and the 2x2 truncation used
for this compact demo drops power leaking to other sidebands. Tests:
`tests/test_quantum_efficiency.py` (ports the Julia docstring examples exactly).

### Policy selection per design family
- Unbiased 4WM (JPA, JTWPA, FQJTWPA): `positive_odd_jc`, K = `Nmodulationharmonics`.
- Biased / DC / 3WM (FXJPA): symmetry broken -> use **`dense_real`** (all-mode
  phasor basis) + a DC solution.
- Complex/lossy (FQJTWPA_diss): complex C **just works** — physical node fluxes
  stay real, loss only makes D(omega) complex. Use `positive_odd_jc` + complex
  matrices (loads automatically). (Gain currently ~0.9 dB off near threshold; JC
  lossy-pump convention still to reconcile.)
- Multi-pump (DPJPA): needs true 2D-lattice HB -> use the standalone
  `exp14_dpjpa_multitone.py` (modes are (k1,k2) tuples). `auto_jc` in exp08 still
  raises for multi-pump (scalar policy can't represent it).
- DC + mutual-inductor distributed (FXJTWPA): **MATCHED (RMS 0.0 dB)** via an
  imported JC pump nodeflux seed. The blocker was never the fold or the stiff
  mutual K (exp10's mutual stamp is algebraically identical to JC's `calcinvLn`,
  doctest in `capindmat.jl`). It was **node ordering**: exp10 inserts nodes per
  cell as (node, node+3, node+2, node+1, node+4) -- unsorted -- while JC orders
  by sorted node number. The identity seed left a real ~45 pump residual on the
  SQUID nodes; the sorted-rank permutation drops it to ~5e-9. Pipeline:
  `exp14_build_jc_warmstart.py` (raw seed) -> `exp14_fxjtwpa_fix_seed.py`
  (applies the node-order permutation to pump X **and** DC node fluxes) ->
  `exp09 --pump-dir outputs/exp14_fxjtwpa_seed_fixed/pump --dc-solution .../dc
  --source-port 1 --out-port 2 --sidebands 4 --signal-m 0 --idler-m -2`.
  Test: `tests/test_fxjtwpa_node_order.py`.

### Preconditioners (`NewtonKrylovSettings.preconditioner`)
- `mean_tangent` (default), `linear`, `none` — block-diagonal.
- `spectral_coupled` — assembles the mode-coupled (k-q) complex Jacobian, one LU.
- `real_coupled` — exact full real-packed Jacobian incl. the conjugate (k+q) term;
  GMRES converges in ~1 iteration. Use for stiff DC/mutual designs.
  `run_gain_map.py`'s in-process engine defaults to `real_coupled`.

## Pseudo-arclength metric fix and fold-vs-numerical measurement (2026-08-06)

`solve_arclength` (`src/twpa_solver/pump/solver.py`) previously normalised
its tangent with an unscaled Euclidean metric mixing node flux (`X`, ~1e-13
Wb on a real device) with the dimensionless source scale `lambda` (~1.0). The
state's contribution to the arclength constraint was ~5e-26 of the lambda
contribution -- ten orders below double-precision roundoff -- so the function
was natural-parameter continuation in disguise: fold detection was
structurally impossible (`lam_dot`'s sign could never flip) and the corrector
returned `terminal_reason="minimum_step"` identically for a genuine fold and
a merely sharp turn. Fixed: a state-scale-derived metric
(`metric_x(a,b) = Re<a,b> / state_scale**2`, same construction as the
already-correct `trace_arclength_from_two_points`), a Govaerts & Pryce
one-refinement bordered solve (accurate through a near-singular Jacobian,
where plain block elimination is not), and a least-squares (`lsqr`) fallback
in `FastCoupledPreconditioner` for an exactly-singular factor (the
JosephsonCircuits.jl `QRfactorization()` analogue -- a numerical-technique
borrow only, not a physics comparison; see `jc-is-not-a-reference`). Full
plan: `docs/development/arclength_metric_fix_and_fold_test_function_plan.md`.

New: `src/twpa_solver/pump/singularity.py` -- `jacobian_min_eigenvalue`
(shift-invert Arnoldi, `scipy.sparse.linalg.eigs(sigma=0)`, around the exact
real-packed Jacobian's factor/solve as `OPinv`; falls back to the original
inverse-power iteration if shift-invert fails to converge --
`jacobian_min_eigenvalue_with_estimator` also returns which estimator
produced the value), `jacobian_det_signature` (sign/log|det| via SuperLU's
`U` diagonal + permutation parity), and `bordered_conditioning` (the
fold-vs-branch-point discriminator: estimates the condition number of the
same bordered system `solve_arclength`'s own corrector solves -- a fold
leaves it well-conditioned even though `J` is singular, Keller 1977; a
branch point, a rank-2 degeneracy, is not regularized by one border and
degenerates the bordered system too). Diagnostic driver:
`scripts/scan_branch_singularity.py` (marches one frequency column in power;
`solve_arclength`'s `on_step` hook now feeds all three functions at every
*accepted* continuation step via `singularity_scan_steps.csv`, not just once
at the run's final endpoint -- the original single-endpoint measurement was
comparing a healthy point on the branch to a healthy baseline and could not
have detected a fold even where one existed; not wired into the production
map). Full plan for the re-measurement below:
`docs/development/arclength_fold_resolution_plan.md`.

**Both fp=7.9 GHz "confirmed NUMERICAL" and fp=7.0 GHz "genuine SNAKING"
above are WITHDRAWN (measured 2026-08-07)** -- both verdicts were artifacts
of the single-endpoint measurement bug just described, not of the metric
fix itself. Re-measured with the per-step instrument on
`designs/ipm_2c_fixed`:

- **fp=7.9 GHz: CONFIRMED genuine fold, `I_bound ~= 1.163e-05 A`.** With a
  budget matched to where folds are actually detected
  (`--arclength-max-steps 60`) plus a 150-step post-fold extension
  (`docs/development/arclength_fold_resolution_plan.md` Phase 2,
  `outputs/phase2_fold_rounding_check/singularity_scan.csv`): -22.0 through
  -19.5 dBm now reach `target_lam=1.0` (the larger base budget alone lets
  the corrector detect folds it previously ran out of steps before
  reaching); at -19.0 and -18.5 dBm a fold *is* detected
  (`fold_lambda`=0.9511/0.8984) but even 150 further steps cannot cross back
  to target -- the budget is exhausted rounding the fold, not reaching it.
  `I_bound = fold_lambda * injected_current` = 1.1628e-05 / 1.1633e-05 A,
  matching this section's earlier (differently-derived) `1.1929e-05 A`
  figure to ~2.5%. This is the device's physical pump ceiling at this
  frequency, not a solver artifact.
- **fp=7.0 GHz: mistuning CONFIRMED, not a branch point/snaking.** The
  original measurement's `terminal_reason=minimum_step` with no `lam_dot`
  sign flip, from -22.75 dBm onward, is what a mistuned arclength metric
  looks like, not what snaking looks like. With Phase 1's periodic metric
  rescale (`solve_arclength(..., rescale_every=5)`,
  `outputs/phase3_rescale_classification/singularity_scan.csv`),
  `terminal_reason=minimum_step` **never occurs** from -22.5 through
  -21.0 dBm (24% above the old 8.640e-06 A boundary) -- the corrector makes
  productive progress and now detects real folds (`fold_lambda`
  0.81-0.93) instead of collapsing. Reaching `target_lam=1.0` from here is
  the same already-solved fp=7.9 GHz fold-rounding problem, not a
  fp=7.0-specific blocker. `bordered_conditioning` was inconclusive at this
  measurement's `--eig-iters 5` (too few power iterations to have
  converged at that setting) and is not needed for this verdict.

**Deflation is NOT needed for either frequency** -- the decisive prior
claim ("Deflation ... is the indicated next tool") is retracted along with
the SNAKING verdict it was based on. Full write-up including the
intermediate wrong turns (a first fold-rounding attempt with too small a
base budget that never triggered the extension at all) and raw per-step
data: `docs/development/arclength_fold_resolution_plan.md`.

Still not re-run at full resolution: the 19-frequency `--fold-follow` sweep
(§2d's "zero folds everywhere" was measured with the pre-Phase-1 metric).
A reduced 4-point version (7.6/7.9/8.2/8.5 GHz, `--recovery-arclength-
rescale-every 5`, `outputs/phase4_fold_follow_reduced/fold_curve.csv`) found
real folds at 3 of 4 points (fold power -19.4 to -21.5 dBm) -- directly
contradicting "zero folds everywhere" and corroborating the retraction
above. `fold_power` (`solver.py`) gained the same `rescale_every` parameter
as `solve_arclength` for this; without it, a re-run would have repeated the
exact mistuning failure mode. Any fp=7.0 GHz fold *location* number from
sessions before this one (e.g. "lambda~0.97") remains unreliable and not
re-affirmed; the full-resolution 19-point sweep is still open follow-up work.

`run_gain_map.py`'s `--fold-policy arclength` recovery now accepts
`--recovery-arclength-rescale-every` (default `0`, disabled) and
`--recovery-arclength-max-steps-after-fold` (default `0`, disabled --
mathematically identical to `solve_arclength`'s own `None` default, not
merely empirically so: the extension is `max(effective_max_steps, fold_step
+ N)`, and `fold_step <= effective_max_steps` always holds by the loop's own
invariant, so `N=0` can never raise it). When a fold is detected but the
extended budget still cannot reach `target_lam=1.0`, the cell's row records
`pump_arclength_fold_current_a` (the fold's physical boundary current)
instead of a bare failure, on whichever of `--inproc-fail-fast` or the final
reseed fallback ends up returning the cell.

## Continuation-method suite (`run_gain_map.py` + `solver.py`)

Opt-in inter-cell traversal / predictor / recovery / fold-policy layers plus
advanced intra-cell continuation, from `docs/reports/pump_map_continuation_methods.tex`
and its expanded test matrix. **Defaults reproduce the legacy `column` pass
byte-for-byte** (regression: `tests/test_traversal.py::test_column_order...`,
existing gate/CLI tests). Everything below is off unless a flag selects it.

- **Traversal** `--traversal {column,backbone,nearest,serpentine,floodfill}`
  (`+ --backbone-direction {ltr,rtl,center_out,two_ended}`). `column` is the
  legacy per-frequency-column pass. The others share one in-process
  `solved[(i,j)]->X` store across BOTH axes, so they **force
  `--frequency-chunk-size 0`** (single process; the Schur cache stays small to
  bound RAM, so a backbone row rebuilds the per-frequency partition as it
  sweeps). Orchestrator: `run_map_traversal` (not `run_warm_pass_inprocess`,
  which stays the `column` path).
- **Predictors** `--predictor {copy,power_secant,freq_secant,corner,plane,portfolio}`
  (`+ --portfolio-policy {best,ranked}`). Pure math in
  `src/twpa_solver/pump/predictors.py`; `portfolio` ranks candidates by
  `problem.norms(X,1)` residual (`engine.residual_norm`). Tests:
  `tests/test_predictors.py`.
- **Recovery** `--recovery {reseed,alt_parent,bridge,ladder}`
  (`+ --bridge-steps`, `--bridge-mode {diagonal,freq_first,power_first,adaptive}`).
  Bridge = physical-parameter continuation from a solved parent to the target
  along (P,f), `InProcessEngine.solve_bridge`.
- **Power substep** `--column-power-substep` (`+ --column-power-substep-init-db`
  0.1, `--column-power-substep-min-db` 0.005, `--column-power-substep-deadline-s`
  120). On a failed warm cell, adaptive natural-parameter continuation **along the
  map power axis** from the last converged state: walk up in adaptive dBm
  micro-steps (geometric in current, grow x1.5 / halve on fail), warm-starting
  each; `InProcessEngine.solve_power_substep`. This is the diagnostics'
  "0.005-0.01 dB steps cross the wall" finding operationalized inside the map:
  the coarse 0.30 dB power grid overshoots gain-lobe crests that finer stepping
  crosses. A step-independent stall (step < min_db) is recorded
  (`pump_power_substep_stall_dbm`, sets `verified_fold`) as a real
  numerical/fold boundary rather than retried. **Demonstrated:** 2c themis
  column fp=8.099 GHz, coarse map died at -29.36 dBm (3-4 PASS); substep
  recovered to ~-25.75 dBm (16 PASS, +3.9 dB, gains through the multi-lobe
  ripple) before a genuine boundary. The intra-loop `last_good_X` is
  retained-shape (Schur), so substep solves are shape-compatible with no disk
  round-trip (the disk-load seed path in `--initial-pump-dir` is full-shape and
  is a separate concern). Compare a recovered map to the Themis measurement with
  `scripts/compare_map_to_measurement.py` (peak-gain + collapse-power envelope,
  aligned by the ~+0.99 GHz / few-dB calibration offsets;
  `docs/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK`).
- **Calibration-shift map fit** (`scripts/align_map_to_measurement.py`): instead of
  hand-tuning the calibration offsets, fit them as nuisance parameters on the 2-D
  peak-gain maps. Model `G_meas(f,P) ~= G_sim(f-df, P-dP) + dG`; for weighted LSQ
  `dG` is analytic per `(df,dP)`, so a coarse+fine 2-D grid search over `(df,dP)`
  remains (`align_maps`). Reduces the Themis cube (`105C5_*GHz.npy`: transmission
  over power x signal-freq per pump freq) to a peak-gain map and plots it (the raw
  `docs/14.18.08_Themis_...` ships data only, no plot); resamples the sim with
  `RegularGridInterpolator` at shifted coords, masks non-overlap + NaN (failed) sim
  cells, ROI-weights so the amplified ridge (not the flat background) drives the fit,
  `--loss {l2,huber}`. Writes JSON + a measurement-map PNG + a 4-panel comparison
  (meas / aligned sim / residual / **loss surface**, clipped color so the min basin
  is visible). Fit one section only with `--fit-freq-ghz LO HI` / `--fit-power-dbm
  LO HI` (hard-mask the measurement grid outside the window); `--min-overlap-frac`
  (default 0.25) rejects tiny-overlap corner fits (a soft 1/overlap penalty alone
  lets a ~9-cell corner with near-zero local residual win -- the guard is relative
  to the floor-weighted window, whose max achievable overlap is only ~0.3-0.4
  because the sim fails at high power, so 0.25 is the practical ceiling not 0.5).
  **Demonstrated (14.18.08 vs `map_2c_scan_6p0_8p5_100x70`):** full-map best
  df=-0.30 GHz, dP=+2.55 dB, dG=-1.30 dB, RMS 6.0 dB, df-elongated (weakly
  identified) basin. **Per-section fits are far better identified:** restricting to
  one comb branch gives **df ~= 0 GHz** (the two datasets' combs are already
  frequency-aligned -- no ~+1 GHz offset like 17.03.10; the full-map df=-0.30 was a
  comb-alias compromise since a single df can't align all lobes when the comb phase
  drifts) and **dP ~= +2.5..+3.3 dB robustly across 6.2-7.45 GHz** (the one real
  calibration offset), with a compact single-minimum loss surface. RMS stays ~2-4 dB
  per lobe because the sim still (a) does not reach the measured high-power lobes
  (numerical-boundary cap) and (b) has a slightly different comb periodicity -- a
  model-fidelity/coverage gap, not a fit bug. Band edge 7.45+ GHz has ~no sim
  coverage (junk dG). Tests: `tests/test_align_map.py`.
- **Forced-gain column resume** (`scripts/resume_column_force_gain.py` +
  `InProcessEngine.solve_point(force_gain=True)`): diagnostic to test whether a
  column's high-power wall is the real device fold or a numerical boundary. Marches
  one column (`--column-freq-ghz`, nearest grid col; omit = all columns) up in
  power, warm-starting each cell from the previous, and runs the gain solve on the
  **last Newton iterate regardless of convergence** — the normal path only gains
  converged pumps (`solve_point` gate is `converged or force_gain`; it also returns
  the last-iterate `X` so the warm chain continues past the wall). Never skips;
  stops a column after `--force-max-nonfinite` (default 3) consecutive non-finite
  pump states (warm chain diverged). Writes per-cell dirs, a per-column CSV
  (`pump_converged`, `forced_gain`, `gain_db` columns) and a PNG (gain vs power,
  converged pts vs forced pts). Takes the SAME engine/grid flags as
  `run_gain_map.py`. **Demonstrated:** 2c fp=8.099 GHz — converged branch 9.2→12.3→
  15.7 dB (-32..-30 dBm), then -29 dBm pump FAILS (coeff_rel 0.089) and the forced
  gain **collapses to 1.5 dB**, i.e. the non-converged waveform does not sustain the
  gain (evidence the wall is near a genuine transition, not a mere solver miss). The
  in-loop warm state is retained-shape (Schur), so force-marching through a
  non-converged iterate is shape-compatible with no disk round-trip. Tests:
  `tests/test_run_gain_map_cli.py::test_force_gain_*`.
- **Fold policy** `--fold-policy {patience,cross_axis,bridge_gate,combined,arclength}`
  — when a failed cell counts toward the per-column fold short-circuit; `combined`
  is the report's recommended ladder (power/freq parent + portfolio + bridge before
  counting); `arclength` rounds the fold.
- **Intra-cell** (`solver.py`) `--inproc-continuation {adaptive_secant,adaptive_tangent,affine,ptc}`:
  tangent/Euler predictor (`dR/dlambda=-S`, `source_coeffs(1)`),
  affine-ish step control, and pseudo-transient (`solve_pseudo_transient`).
  Pseudo-arclength (`solve_arclength`, bordering algorithm, modified-Newton) and
  the `fold_power` locator drive `--fold-policy arclength` and `--fold-follow`
  (writes `fold_curve.csv`, no gain map). **Key perf detail:** the advanced linear
  solves use `problem.assemble_real_coupled_preconditioner` (near-direct) via
  `_linear_solver`; the mean-tangent block factors leave GMRES grinding on the
  coupled system. Tests: `tests/test_advanced_continuation.py`.
  arclength/fold-follow are functional but **experimental** on the stiff 2c device
  (fold-follow may report no fold in range; the arclength target endpoint is
  linearly interpolated, so it is used as a warm guess, not a polished root).
- **Mid-GMRES deadline abort** (`solver.py` `gmres_call`): `solve_deadline_s`
  checks elapsed wall time on every GMRES iteration via `callback_type="pr_norm"`,
  raising `GmresDeadlineExceeded` from inside the callback instead of only
  checking between Newton iterations (the old scheme let one pathological GMRES
  call run ~200s past a 14s budget with `gmres_total` in the thousands).
- **Adaptive-continuation fallback resumes, not restarts** (`solve_adaptive_continuation`):
  when lambda-bisection shrinks below `min_step` (near a genuine fold in source
  scale, reduction ratio stuck near 1.0 at every lambda=1 attempt), it falls back
  to a fixed-step ladder (`solve_continuation`). This used to pass the *original*
  seed and lambda=0, discarding every state the adaptive phase had already
  converged and re-deriving the cheap low-lambda region from scratch -- on a real
  map column (fp=7.329 GHz, -28.25 dBm,
  `outputs/measurement_match_debug_01/column_debug_col3_trim`, debug-logged) this
  burned the whole 14s per-point deadline getting back only to lambda=0.75 after
  the adaptive phase had already reached lambda=0.9375 converged. Fixed:
  `solve_continuation` takes a `lambda_start` (default 0.0, so all other callers
  are unaffected) and the fallback resumes from `(X_current, lambda_current)`,
  sized to the remaining span at the original `1/fallback_fixed_steps`
  granularity (`math.ceil(remaining / step_size)` steps, not the full
  `fallback_fixed_steps`). Column-level `--column-arclength-recovery` (separate
  from this intra-solve fallback) already retries fresh on every failing cell,
  not once per column. Tests: `tests/test_adaptive_continuation_fallback.py`.

The engine's `X` is Schur-reduced (retained-port shape, constant across
frequencies), so chained warm starts and residual ranking all use the same
`engine._make_solve_problem(...)` representation — never the full-node problem.

### Campaign (`scripts/run_campaign.ps1`)
Sequential 2c campaign (`outputs/ipm_python_design`) mirroring the current
production run (`outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10`:
50x50, -35..-23 dBm x 7.5..8.5 GHz, spectrum, sb10). Each config runs
`run_gain_map -> plot_gain_map (--top-k 3, maps + candidate S21 sweeps) ->
prune_map_solutions (--top-k 100 --purge-point-dirs --apply)`. `-DryRun` prints
commands; `-Only id1,id2` runs a subset. ~16 configs, est. ~20-26 h; pruned to
~0.1-0.2 GB/run.

### Standard gain-map flag set (2026-08-03)

Every production gain map uses this block. Pass it to `scripts/run_gain_map.py`,
or forward it through `workflows/run_gain_map_and_plots.py` (which injects
`--circuit-dir`, `--outdir`, `--executor inprocess` and adds the plot
catalogue). Only the grid bounds and the directories change per run.

```
--executor inprocess --mode warmstart
--inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast
--inproc-fold-predictor secant --inproc-fail-fast --fold-skip-patience 2
--pump-current-jc-scale 1.0
--n-power 20 --n-frequency 20 --frequency-chunk-size 10
--pump-power-min-dbm -26 --pump-power-max-dbm -16
--pump-freq-min-ghz 7.6 --pump-freq-max-ghz 7.85
--signal-detuning-mhz 500 --no-signal-spectrum
--log-level INFO --overwrite
```

Most of these already match the parser default, but **six do not** and omitting
them silently changes results or cost:

| flag | default | reason it is set |
| --- | --- | --- |
| `--inproc-fail-fast` | `False` | stop a bad cell instead of grinding |
| `--fold-skip-patience 2` | `0` | enable skip counting |
| `--signal-detuning-mhz 500` | `100` | wider signal offset |
| `--no-signal-spectrum` | spectrum **on** | the default runs a full signal sweep at *every* grid point and dominates wall time |
| `--log-level INFO` | `DEBUG` | DEBUG logs are enormous on a 400-point map |
| `--overwrite` | `False` | reruns land in the same directory |

Leave `--attenuation-db` unset (measured loss_A10 model applies) and
`--loss-model` at `auto`. `--pump-port` defaults to 4, correct for 2c.

`--pump-current-jc-scale` is kept in the block for explicitness only: 1.0 is now
both the parser default and the validated conversion. Passing `2.0` would be a
6 dB error.

Caveat: `--fold-skip-patience 2` has previously culled the 2c broadband
operating region into needle-only maps when skip counting false-tripped. Check
the PASS coverage of a sparse map before quoting it.

## Validation provenance (experiments/)

The solver's numerics are pinned to JosephsonCircuits.jl by the exp13/exp14
parity runs below. These live in `experiments/` + `outputs/`; they are the
reference the solver was tuned against, not part of the production path.

### 7-design parity status (outputs/exp14_seven_design_summary/)
6/7 MATCHED < 0.0024 dB: jpa, jtwpa, fqjtwpa, fxjpa, dpjpa, **fxjtwpa (RMS 0.0)**.
fqjtwpa_diss SOLVED ~0.89 dB (lossy convention) -- the only remaining mismatch.
JC reference
curves: `outputs/exp14_jc_refs/` via `exp14_jc_doc_curve_dump.jl` (generic; splats
each case `hbsolve_kwargs` so DC/3WM/4WM work).

### Reproduce parity
JC reference curves: `outputs/exp13_compare/jc_jpa_curve.csv`,
`outputs/exp13_jtwpa_fast_scale2/jc_jtwpa_curve_21pt.csv`. "Pump scale 2" means
pump source current = 2 x the design's AC pump current. Runs land under
`outputs/exp14_*`; build the table with
`python experiments/exp14_seven_design_summary.py`.

## Matrix-tracing driver (`scripts/run_gain_map_column_matrices.py`)

Diagnostic one-column driver that runs `run_gain_map` in-process (forces
`--n-frequency 1`, single `--column-frequency-ghz`) and archives the sparse
matrices flowing through solver frames (`src/twpa_solver/`, `experiments/`,
`run_gain_map.py`), plus the static design matrices. The production engine flags
are baked in (`PRODUCTION_ENGINE_FLAGS`: schur_cpu_mt + real_coupled_fast +
secant + fail-fast + patience 2 + cache 1 + detuning 100 MHz + direct/superlu +
mode-count 10 + nt 40 + no-spectrum); pass extra run_gain_map flags (e.g.
`--log-level INFO`) after `--extra-map-args` (REMAINDER, must be last).

**Capture uses `sys.setprofile` (call/return), NOT `sys.settrace` (per-line) —
do not switch it back.** Line tracing the stiff assembly loops slowed one pump
solve **~40x** (factor 2.2s -> 92s; the hot `assemble_real_coupled_fast`/Schur
assembly lives *inside* a traced module, so scoping settrace to solver files did
not help). `setprofile` scans a target frame's `f_locals` only on `return`
(args + assembled results both present) -> ~4x over untraced (92.8s -> 8.95s for
3 points), archiving matrices that cross function boundaries. Trade-off: a matrix
built and discarded within one function without surviving to its return is not
archived (per-line count 1103 -> boundary count ~42). The callback is **fail-safe**
(catches `Exception` per matrix, logs + skips, reports `matrix_save_errors`) so a
save error never aborts the solve; only `KeyboardInterrupt`/`SystemExit` propagate.
Because the hook is hot, a **Ctrl-C tends to surface inside `dispatch`** -- that is a
real interrupt, not a tracer bug. On Windows the archive paths are deep: use a
**short `--outdir`** to stay under MAX_PATH (a deep default outdir can hit an OSError
that is now skipped rather than fatal). Tests: `tests/test_column_matrices_tracer.py`.

## Tests
`tests/` covers the solver: `test_loss_model.py` (loss fit),
`test_pump_basis.py` (pump-mode basis), `test_fxjtwpa_node_order.py`,
`test_exp10_gate.py` (map gate), `test_column_matrices_tracer.py` (setprofile
matrix tracer). Run with `--basetemp` off the repo to dodge a Windows ACL issue
on `.pytest_tmp`.

## Le Gal benchmark readout contract

The effective-SNAIL benchmark uses a branch law shifted to its solved static
equilibrium. Its 31 fF SNAIL capacitance is stamped across the branch, while
223.5 fF is the ground capacitance; `sqrt(L/Cg)` is about 62.3765 ohm.
`shunt_conductance_s` is an SI conductance, not a loss tangent.

Benchmark HB gain must be read through `multitone.observables.tone_s21` and
reported as the pump-on/pump-off ratio when compression is discussed. Do not
recreate gain with `|i omega X|/V_in`; that old path was biased by 12.041 dB.
`power_balance` accepts `z0_ohm`. The `conversion_manley_rowe_*` fields are
restricted to pump/signal/idler; `all_tone_manley_rowe_*` exposes the retained
tone scope. Legacy names remain for compatibility.

The corrected Level-2 campaign currently measures a passive or weakly
deamplifying effective line at the sampled points; it has not reproduced the
paper's two gain lobes or a finite P1dB. This is a measured negative result,
not evidence against the paper.

## Component profiles and scatter

The IPM builder supports ordered per-cell `Lj` and `Cg` profiles and
independent multiplicative `Lj`, `Cj`, and `Cg` scatter, all on the JTL cell
index (2508 on 2c). See `docs/development/component_profiles_and_scatter.md`
for the shape catalogue, selection rules, Cg boundary-halving mapping, and the
stable RNG stream contract.

There is **no** `Cj` profile: nominal `Cj` is derived from the `Lj` profile so
the plasma frequency stays constant. Independent `Cj` *scatter* intentionally
breaks that, and must not be "corrected" to track `Lj`. Scatter sigma is a
fraction of each cell's own nominal, and the `Lj` stream is bit-identical to
the pre-profile `apply_lj_scatter` at the same seed.

`sine`/`cosine` are the one place `start`/`end` are not the first and last
values — for those two they bound the oscillation envelope, since periodic and
endpoint-anchored are mutually exclusive.

`build_variant_design` re-emits a stored design with a profile applied. Its
topology gate (`assert_source_topology`) must run on a **nominal** rebuild;
gating the profiled netlist against the source would reject every real variant.
`designs/ipm_2c_fixed` passes at 16312/16312 elements, `C/G/K/Bphi` maxdiff
0.0, with `--coupler-mode cached` (`ideal`/`optimize` do not reproduce it).
Other design dirs must be checked before use as variant sources.

Gates: `tests/test_component_profiles.py`, `test_component_scatter.py`,
`test_ipm_role_tags.py`, `test_ipm_component_plan.py`, `test_variant_design.py`
(95 tests, each verified by mutation).
# Dielectric dissipation

On-chip dielectric loss is stored as `C * (1 - 1j*tan_delta)`, so
`Im(C) = -C*tan_delta`. Lossy circuits resolve to `conductance_abs_omega`,
which uses `G + |omega| C tan_delta` and preserves
`D(-omega) = conj(D(omega))`. The `jj_cj` junction-capacitance role is always
lossless; the tangent applies to substrate and coupling capacitors.

`plasma_locked` Cj scatter derives Cj from the Lj factor, preserving each
cell's plasma frequency. Because `conductance_abs_omega` is non-analytic in
complex omega, Tier-1 stability remains available but Tier-2 complex-omega
resonance refinement is unavailable for lossy circuits.

## 7.9 GHz PERIOD1 branch and Floquet stability (measured 2026-08-11)

Full evidence and the staged plan:
`docs/development/high_power_79ghz_period1_floquet_plan.md`.

### The `-23.421053` dBm "HB failure" is a power-step artifact

`scripts/recover_period1_branch.py` reached `-24.25`, `-24.00`, `-23.80`,
`-23.60` and `-23.421053` dBm from the `-24.473684` dBm column checkpoint by
**plain Newton** (`method="natural"`); PALC was never invoked. Same basis
`[1,3,...,19]`, `--pump-port 4`, `designs/ipm_2c_fixed`. The original column
(`.hybrid_outputs/hb_up_7p9_m35_to_m21`) stepped `1.0526 dB` and stalled;
`0.179`-`0.250` dB steps converge. The converged branch there is benign:
`branch_current_max_over_ic` runs `0.5589 -> 0.6023` and `branch_min_cos_phase`
`0.8292 -> 0.7983`, smooth and monotone.

**Do not quote `0.862` utilization or `0.507` min-cos from
`hb_up_to_failure.csv` row 11.** Those are the diverging Newton iterate, not a
solution. The converged values at that power are `0.6023` and `0.7983`.

The PALC fold remains `I_bound = 1.1628e-05 A`
([[arclength-metric-bug-and-snaking-verdict]]). The TD ramp bracket
`1.140e-05 A <= I_boundary <= 1.160e-05 A`
(`docs/development/h3_physical_boundary_79.md`) is now **UNCONFIRMED**: the
requested `delta_theta = 0.01` floor run terminated at its 100-period restart
checkpoint without a final summary, and the available floor margin is only
7.1x rather than the required 10x. The disputed point is `7.3489e-06 A`, i.e.
**63% of the PALC boundary current, 3.98 dB below it**.

### The TD campaign's `UNRESOLVED_LONG_TRANSIENT` labels are protocol artifacts

`.hybrid_outputs/overnight_7p9_dynamics_v1/campaign_summary.json`:
`d1_late = 4.25e-4 +/- 1e-6` **constant across 11 dB** of pump power
(`-35` through `-24.0` dBm), and `tau_periods = 396..410` at every one of
those powers. That is a numerical floor; fitting a decay slope on it produces a
constant, meaningless timescale. The campaign's own controls at `-23.8` dBm:

| `delta_theta` | ramp | init | d1_late |
| ---: | ---: | --- | ---: |
| 0.05 | 20 | zero-pump | 2.413e-3 |
| 0.05 | 80 | zero-pump | 1.025e-3 |
| 0.025 | 40 | zero-pump | 1.179e-3 |
| 0.05 | 0 | TD restart | 4.678e-4 (**PERIOD1**) |
| 0.05 | 40 | **HB orbit** | **4.436e-4** |

The matched-protocol rerun does not support timestep convergence: at `-23.8`
dBm, `delta_theta = 0.05` gives `1.075e-3`, while `0.025` gives `1.179e-3`.
The ramp-length effect remains, and HB-orbit init lands at the same floor the
classifier calls PERIOD1 at `-35` dBm (`6.313e-4` at `-23.421053` dBm).
**TD preserves the PERIOD1 orbit at the disputed powers, but `d1` is not a
stability discriminant.**
Above the transition `d1` is non-monotone and falls back to `1.2e-2` by
`-17` dBm, which is not a period-doubling ordering. The one physical signal up
there is `mean_phase_winding_cycles`, `1e-8` below vs `-8.12e-2` at `-15` dBm.

**Retracted:** the inference that ~1000-period relaxation implies a Floquet
multiplier near `0.9992`. The same timescale is present 11 dB lower where
nothing is marginal.

Three instrument defects behind this:
- `scripts/run_overnight_7p9_dynamics.py:105` hardcodes `implicit_trapezoid`,
  which is A-stable but **not L-stable** (`R(z) -> -1`), so it applies no
  numerical damping at any frequency. In a circuit whose only dissipation is
  four port resistors, ramp-injected content never decays.
  `h1_transient_branch_transfer.py:1557` already offers `BDF` and `Radau`.
- Phase 1 now makes the post-ramp `max_abs_phi` envelope slope the primary
  classifier (`>1e-5/period` means growth); recurrence distances and their
  trend remain secondary diagnostics only.
- A separate audit found `UNRESOLVED_SLOW_RELAXATION` while the `max_abs_phi`
  envelope grew monotonically by `2.7x`; retire `d1` and use the envelope slope
  as the transient diagnostic.
- `scripts/floquet_stability_sweep.py` now prints the phase as ASCII `angle=`
  and writes its JSON before the summary block.

### `|lambda|` is not a decidable discriminant on this circuit

`designs/ipm_2c_fixed` resolves to `has_loss = False`: four 50-ohm port
resistors are the only dissipation in a 6136-node / 16312-element network, so
`default_loss_model_for` returns `current_complex_c` (analytic, Tier-2 legal).
Dense half-zone Hill scan (700 points, S=4, `gamma_nt=1024`, top-20 minima
refined):

| P [dBm] | max `\|lambda\|` | `1 - \|lambda\|` | f [GHz] | label |
| ---: | ---: | ---: | ---: | --- |
| -35.000000 | 0.99999994 | 6e-8 | 3.90277 | PERIOD_DOUBLING_CANDIDATE |
| -24.473684 | 0.99999772 | 2.3e-6 | 2.69643 | NEIMARK_SACKER_CANDIDATE |
| -23.421053 | 1.00000000 | 2e-11 | 3.47645 | NEIMARK_SACKER_CANDIDATE |

`-35` dBm, unquestionably stable, looks *more* marginal than `-24.47` dBm and is
labelled period-doubling. Exactly-real roots (`Im(omega)=0`, `|lambda|=1`) also
appear (`2.977`/`2.797` GHz and `2.320`/`2.337` GHz): port-decoupled internal
modes. A 16-point seeded power sweep shows the port-coupled branch flat at
`0.9550..0.9573` with no approach, while the secant intermittently falls onto a
power-independent neutral root at `3.5912` GHz -- a branch-tracking failure.
`src/twpa_solver/stability/tracking.py` exists but is wired only to the
monodromy scan, not to the Hill sweep.

**Consequence: do not report a `|lambda|` crossing on a lossless circuit.**
Making stability decidable requires physical dissipation first.

### The time-domain monodromy route fails on 2c and should not be retried as-is

`.hybrid_outputs/floquet_7p9_2c_v1_smoke2/floquet_results.json`: dimension
`12271`, `k=2`, `which="LM"`, `ncv` default 20 ->
`"ARPACK error -1: No convergence (81 iterations, 0/2 eigenvectors converged)"`
after `651` matvecs / `121 s`; `spectral_radius = NaN`. Two independent
blockers, both properties of the operator rather than defects in
`src/twpa_solver/stability/`:
- thousands of multipliers within `1e-8` of the unit circle -- Arnoldi cannot
  split a 2-D subspace out of that cluster;
- one-period closure error `3.922e-03` at 64 steps/period, against a target
  `1 - |lambda| ~ 1e-8`. Trapezoid is 2nd order, so `~1e5` steps/period would be
  needed.

The formulation in `docs/development/floquet_implementation.md` is sound; the
eigensolver strategy is what fails. Prefer the Hill route
(`src/twpa_solver/signal/stability.py`) -- the conversion matrix *is* the Hill
matrix and the gain solve already factorizes it.

### Mode comb sets the required Hill scan density

Unpumped linear circuit, `solve_linear_scattering` with
`extra_K = Bphi diag(gamma_off) Bphi^T` at zero flux, 7.85-7.95 GHz:
port 4->3 group delay `0.121 ns` (1.0 pump period, `|S|` -0.05..-0.67 dB);
port 4->2 group delay `5.885 ns` (**46.5 pump periods**, `|S|` -8.5..-23.2 dB).
The pump-shifted mode comb measured in Phase 0 is **~241.7 MHz**. Use about
700 points for a full Hill-zone scan; `--n-points 200` is thin but not aliased,
and the Phase 1 guard rejects scans below approximately 175 points.

### The Themis 14.18.08 cube measures the collapse boundary directly

`docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK`: 51 pump
frequencies `5.980`-`7.997` GHz at ~40 MHz spacing, so it **brackets 7.9 GHz**.
(`17.03.10_...` spans only `7.043`-`7.373` GHz and is useless for this.) Each
`.npy` is a pickled dict, load with `np.load(..., allow_pickle=True).item()`:

```text
Frequency    (2001,)      4.0 - 12.0 GHz
Response     (31, 2001)   dB, "cali on unpumped device" -> IS gain_vs_off_db
PumpPower    (31,)        -29.0800 .. -19.0267 dBm, 0.3351 dB step
SignalPower  scalar       -30 dBm
```

`Response` is a pump-on/pump-off ratio, so it is **invariant under the source
power convention** and directly comparable to the solver's `gain_vs_off_db`.

**The device collapses abruptly and totally.** Median response over the whole
4-12 GHz span falls from ~0 dB to ~-30 dB in one 0.335 dB pump step -- the line
stops transmitting, it is not gain rolling off. At 7.916 GHz: 20.694 dB peak /
-0.052 median at -22.7129 dBm, then 2.258 / -11.241 at -22.3778 dBm.

The boundary is a **sawtooth comb in pump frequency**: period ~265 MHz (resets
at 6.141, 6.383, 6.666, 6.908, 7.150, 7.432, 7.714, 7.997 GHz), depth 5.36 dB,
envelope -24.388 to -19.027 dBm. The upper edge is **censored** by the
instrument's max pump power -- at 6.908, 7.150, 7.714, 7.755 and 7.997 GHz the
device never collapsed in range. Never report a censored frequency as a
boundary. Peak gain immediately before collapse spans 8.4-33.2 dB, median ~20 dB
(33.15 dB at 7.835 GHz).

**First external contact with the PALC fold -- promising, not yet a
confirmation.** An exhaustive search found exactly **three** `fold_curve.csv`
files; there is **no 19-point sweep on disk** (a `-18.6..-23.0 dBm` 19-point
figure circulating in session notes is unsupported -- do not cite it):

| file | points | content |
| --- | ---: | --- |
| `outputs/phase4_fold_follow_reduced/` | 4 | 7.6: `-21.496451` (`lam=0.5311`); 7.9: `-19.435096` (`lam=0.6734`); 8.2: no fold; 8.5: `-19.825254` |
| `outputs/campaign_diss/2c_single_column_7p6_fold_trace/` | 1 | 7.6: `-22.109839` (`lam=0.4949`) |
| `outputs/continuation_diagnostics/f00_fold_follow/` | 3 | 7.786/7.969/8.153: **all empty** (retracted pre-metric-fix run) |

| f_p [GHz] | model fold [dBm] | source | measured bracket | verdict |
| ---: | ---: | --- | --- | --- |
| 7.6 | -21.496451 | phase4 | [-21.7076, -21.3724] @ 7.593 | **inside** |
| 7.6 | -22.109839 | campaign_diss | [-21.7076, -21.3724] @ 7.593 | 0.40 dB below |
| 7.9 | -19.435096 | phase4 | [-21.3724, -21.0373] @ 7.876 | model high 1.6-3.3 dB |
| 8.2 | no fold found | phase4 | measurement ends 7.997 | no comparison |

**That table assumes `df = 0` and `dP = 0`, which is not admissible.** The
measurement is a real device: fab tolerance on `Lj`/`Cg` shifts the comb in
frequency, line calibration shifts the power. Both offsets are small in their
own units, but the comb converts small `df` into large `dP` -- measured envelope
slope is **-19.1 dB/GHz at 7.6 GHz and -20.8 dB/GHz at 7.9 GHz**, i.e. 0.19-0.21
dB per 10 MHz, so the measurement's own 40 MHz grid step is already +/-0.4 dB of
irreducible power ambiguity.

**The comparison currently has ZERO degrees of freedom.** 8.5 GHz is outside the
measured band and 8.2 GHz has no fold, so only 2 model points land inside; there
are 2 calibration unknowns. Constrained scan against the 46 non-censored
measured points (bracket midpoints):

| window | best rms | at | `dP` within 0.25 dB of best |
| --- | ---: | --- | --- |
| `df`<=50 MHz, `dP`<=1.5 dB | 1.006 dB | df=-0.024 GHz, dP=-0.76 dB | -1.50..+0.40 (span **1.90 dB**) |
| `df`<=100 MHz, `dP`<=3.0 dB | 0.975 dB | df=-0.100 GHz, dP=+0.76 dB | -3.00..+1.48 (span **4.48 dB**) |

Two retractions follow:

1. **The 0.12 dB agreement at 7.6 GHz is a single-point coincidence.** Adding
   the 7.9 GHz point, no small `(df,dP)` reconciles both better than ~1.0 dB rms
   -- 3x the measurement's own 0.335 dB bracket.
2. **`dP` is NOT determined** (uncertain by 1.9-4.5 dB; the window edge, not the
   data, does the constraining). The earlier inference that `dP ~ 0` rather than
   `align_map_to_measurement.py`'s `+2.5..+3.3 dB` is **withdrawn**. Both remain
   admissible.

The 0.61 dB inter-run model spread is still real and still blocking, but it is
now the second problem, not the first.

Offset-free content that survives: both sides have frequencies with no boundary
in range (model 8.2 GHz; measurement 6.908/7.150/7.714/7.755/7.997); every model
fold and the whole measured envelope sit in the same -19..-24 dBm window (so the
model is not wrong by tens of dB); and **curve shape** (265 MHz comb period,
5.36 dB depth) is `dP`-invariant and only translates under `df`, so it is the
comparison to make. Any `(df,dP)` fit needs >=10 usable model fold frequencies
before it carries information.

Two consequences:

- **Fit `(df, dP)` on the boundary, not the gain lobes.** A 51-point collapse
  envelope against a 51-point fold curve is over-determined and the parameters
  separate, unlike `scripts/align_map_to_measurement.py`'s gain-lobe fit, whose
  `dP = +2.5..+3.3 dB` was measured at 6.2-7.45 GHz. Whether it applies at 7.6+
  GHz is **open** -- the two model folds available there cannot separate `df`
  from `dP` (DOF = 0, see above). Do not argue `dP` from one or two fold points;
  it needs the full 51-frequency curve.
**Observable mismatch: the two sides do not measure the same thing.** The
production column reports `gain_vs_off_db` at ONE tone `f_s = f_p - 500 MHz`;
the measurement takes `max` over 4-12 GHz. Measured bias `G_peak - G@(fp-500)`:
**1.79 -> 11.90 dB at 7.876 GHz** (+9.34 dB across the sweep), 1.15 -> 8.97 dB
at 7.916 GHz. It grows because the gain lobes move in frequency as pump phase
accumulates (peak wanders 4.824 -> 6.800 -> 6.408 -> 8.256 GHz at 7.876 GHz).
Worse, at 7.876 GHz the single-tone trace **falls** (16.31 -> 15.55 -> 13.62 dB)
while the peak still **rises** (21.81 -> 23.48 -> 25.52 dB) -- a fixed-offset
probe reports rollover where the device is still gaining. No slope, shape
statistic or normalization removes a power-dependent non-monotone 12 dB bias.
**Match the observable at the source**: run with a signal spectrum and reduce it
with the measurement's own rule (same span, same pump-exclusion window). Cost
~0.36 s per signal frequency.

**The measured device follows `G ~ (1 - I/I_th)^-2`.** Fitting `1/sqrt(G_lin)`
against pump amplitude over amplifying pre-collapse points and extrapolating to
zero predicts the observed collapse power:

| f_p [GHz] | n | R^2 | `P_th` fit | collapse observed | error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7.835 | 7 | 0.9853 | -20.718 | -20.702 | **0.02 dB** |
| 7.876 | 11 | 0.9009 | -21.634 | -21.372 | 0.26 dB |
| 7.916 | 16 | 0.9666 | -22.435 | -22.713 | 0.28 dB |
| 7.795 | 11 | 0.7630 | -18.459 | -19.697 | 1.24 dB (post-reset, exclude) |

So the collapse **is** the parametric threshold, established from the gain
trajectory alone with no model input.

**Curve-shape comparison protocol** (each step kills a nuisance parameter by
construction, not by fitting):
1. **Landmark referencing (primary).** `u = P - P_boundary` on both sides -> the
   boundary sits at `u=0` and `dP` never enters. Overlay `G(u)`.
2. **Threshold-form scalars (primary quantitative).** Fit `1/sqrt(G_lin)` vs
   pump amplitude on both sides; compare **intercept (threshold) and slope**.
   Over-determined, physically meaningful. Gate on `R^2>=0.9`, `n>=6`, exclude
   post-comb-reset frequencies.
3. **`dG/dP` log-slope, secondary only.** Kills additive `dG` exactly but not
   `dP`, and amplifies noise on a 0.335 dB grid. Use Savitzky-Golay.
4. **Global `(df,dP)` registration last**, under the DOF gate above.

Rejected: **DTW** (absorbs real physics discrepancy into an x-warp, always
matches), **Procrustes** (removes rotation/isotropic scale, meaningless for
dBm-vs-dB axes), **min-max normalization** (discards the dB scale under test),
**correlation coefficient alone** (blind to a uniform offset).

- **Gain at the boundary is a calibration-free test.** Both sides are
  pump-on/pump-off ratios, so it is independent of the port convention and the
  line-loss model. The 7.9 GHz column ran `--no-signal-spectrum`, so only a
  single-frequency `gain_vs_off_db = 13.124` dB at 7.4 GHz exists; that is not
  the peak and must not be compared against the measured 8.4-33.2 dB range.

The `-23.421053` dBm point is below the measured collapse at 7.9 GHz (~-21.6
dBm interpolated) as well as below the model fold. It is near no boundary at
all. Separately, the TD campaign's rising `mean_phase_winding_cycles` at high
power now has an external counterpart: total loss of transmission is what
junction phase running looks like on a VNA.

### Loss channels: three, not interchangeable (resolved 2026-08-11)

| channel | stamped in | analytic in omega | available to `ipm_2c_fixed` |
| --- | --- | --- | --- |
| dielectric `tan_delta` | `Im(C)` = `C*(1-1j*tan_delta)` | **no** -> `conductance_abs_omega` | **yes, the only one** |
| real shunt conductance | `G` | yes | **no** -- IPM has none. `shunt_conductance_s` is a `build_effective_snail_line` parameter only (`builders/le_gal_2025.py:47`, stamped uniform node-to-ground at `:113`); Le Gal benchmark knob, not IPM. |
| RCSJ / quasiparticle resistance | -- | -- | **not implemented in `src/`**. Junctions are purely reactive (ideal JJ + `Cj`); `pump/diagnostics.py:37` names quasiparticle switching as out of model. |

`CircuitMatrices.has_loss` (`core/circuit.py:106`) tests **only** `Im(C)`, never
`G`. So `has_loss = False` on 2c means "no dielectric loss in C", **not** "no
dissipation" -- the four 50-ohm port resistors are in `G` and do dissipate.

Consequence for stability work: adding `tan_delta` flips
`default_loss_model_for` to `conductance_abs_omega` and **kills Tier-2**
complex-omega refinement. Choose the loss model explicitly:
`conductance_abs_omega` = Tier-1 only, preserves `D(-w)=conj(D(w))`, production
convention; `current_complex_c` = Tier-1 **and** Tier-2 (polynomial in omega,
analytic) but breaks conjugate symmetry, so it is a stability-analysis
convention and must never produce a published gain or compression number.
Quantify the two at Tier-1, where both are legal, rather than assuming the gap
is small.

### Do not enable a new HB ansatz

No PERIOD2, period-N, or torus basis is justified by any current evidence. The
scaffolding (`pump/floquet.py`, `pump/periodic_branch.py`,
`signal/period_doubled.py`, `scripts/run_period_doubled_branch.py`) stays
dormant until a tracked multiplier crossing is resolved, timestep-converged,
sideband-converged, and corroborated by an L-stable TD run at the Phase 2
boundary. If a complex pair does cross, use the **auxiliary-generator** closure
on the existing `multitone` two-frequency lattice (two extra real unknowns
`(A_a, omega_a)`, two extra real equations `Y_AG=0` in an outer loop) rather
than building a torus basis from scratch.

## HB ansatz validity measured against the FDTD kernel (2026-08-15)

`scripts/chaos/measure_ansatz_validity.py` reduces a signal-driven FDTD
campaign to the fraction of in-band spectral power sitting on the production
tone lattice `n*f_p + m*f_s`. That fraction is the ceiling on any HB result at
that point and is calibration-free (a ratio inside one spectrum). Canonical
output: `outputs/chaos/ansatz_validity/ansatz_validity.csv`, 68 points, four
devices.

**Below the transition the ansatz is exact.** on-lattice 1.0000 -> 0.9233
(jc_jtwpa, -30.5 -> -28.2 dBm, 13.9 -> 36.1 dB gain); 1.0000 -> 0.9346
(jc_fqjtwpa); 1.0000 (ipm_2c_fixed 0.450-0.575, guarcello -70..-54 dBm).
Off-lattice floor 112-226 dB below pump.

**The collapse is NOT a period doubling.** At collapse on-lattice falls to
0.056/0.051 in one 0.4 dB step; admitting half-integers adds only **4.8 / 4.5
percentage points** (thirds 9.5 / 10.2). Residue is a continuum: top-20
off-lattice bins hold 2.4%, best single extra generator explains 26%, floor
13-18 dB below pump. **Keep the period-N scaffolding dormant**
(`build_half_pump_basis`, `pump/floquet.py`, `pump/periodic_branch.py`,
`signal/period_doubled.py`) -- this is the direct test of the hypothesis they
serve and it fails.

**A torus does appear, in a narrow window.** jc_jtwpa -29.3 -> -28.2 dBm:
off-lattice 0.009 -> 0.077 with single-generator fit 0.999 -> 0.627. So a
quasi-periodic (auxiliary-generator) extension buys ~1.1 dB of pump range and
no more. Real, small, and not the first thing to build.

### Fixed continuation ladders fail where solutions exist (2026-08-15)

`run_compression.py::_solve_pump_from_scratch` hardcodes
`solve_continuation(..., continuation_steps=4)` with no adaptive fallback. At
jc_jtwpa `f_p=7.12 GHz`, `I_p=4.603781e-06 A` (where FDTD shows a clean
period-1 state, on-lattice 0.9233):

| strategy | outcome | lambda | coeff_rel | s |
| --- | --- | ---: | ---: | ---: |
| fixed 4 | stalled | 1.0000 | 1.762e-01 | 11.7 |
| fixed 8 | stalled | 0.8750 | 3.869e-02 | 20.9 |
| fixed 16 | stalled | 0.9375 | 1.889e-02 | 38.0 |
| fixed 32 | stalled | 0.9375 | 2.144e-02 | 62.3 |
| **adaptive** | **reached** | **1.0000** | **8.420e-12** | 94.1 |

Refining a *fixed* ladder does not help; the adaptive run needed accepted steps
down to `dlambda = 0.004454`. Two follow-on defects: the adaptive fallbacks in
`_solve_compression` (pump-only reference) and
`multitone/compression.py::solve_signal_power_point` both use `min_step=0.01`,
above what this point needs, and both pass `x_init=None`, discarding the
promoted pump seed. Separately `AffineSourcePath.signal_turn_on` starts from
**zero** and ramps pump+signal together, so every signal-power point re-runs the
pump ramp -- it fails even at a 1e-11 A probe.

With the pump supplied via `--pump-solution-dir`, the S=10 multitone state at
-28.2 dBm converges in **1 Newton iteration** (coeff_rel 6.23e-12), gain
48.36 dB. So the torus solver was never the blocker.

### JTWPA sideband self-convergence, measured (2026-08-15)

jc_jtwpa at `f_p=7.12 GHz`, `I_p=3.873843e-06 A`, small-signal multitone gain:
S=4/6/8/10/12 -> 27.539 / 29.067 / 30.300 / 30.428 / **30.440** dB. Monotone;
S=10 -> S=12 moves **0.012 dB**. This closes "S=10 cannot be selected" *at this
operating point* only; the older non-monotone sequence (30.7152/24.2021/26.5563/
27.5410 at S=2/4/6/10) was a different point and stays unexplained.

### `--sidebands` defaults to 6; the HB columns are NOT S=10

`run_gain_map.py:4379` has `--sidebands` default **6**, and
`scripts/run_hb_column_until_failure.py` does not override it. Every
`hb_up_to_failure.csv` gain is therefore an S=6 truncation, ~1.4 dB below
converged on jc_jtwpa. **Same-S multitone/Floquet parity is INTACT** -- at the
column's own pump current 3.8806468570637416e-06 A, multitone S=6 gives
29.141 dB against the column's 29.140947 dB (**0.0002 dB**), with utilisation
matching to 11 significant figures. An earlier "parity fails by 1.3 dB" claim
was S=6 vs S=10 and is retracted.

### The HB-vs-FDTD gain gap was the FDTD timestep

**`dt_norm = 0.01` under-resolves the small-signal response on `jc_jtwpa`.**
Compared at the FDTD run's own achieved on-chip pump current (so no power
convention enters), `gain_vs_off_db` both sides (pump-on/pump-off ratio, so also
loss-model-invariant), probe 3e-09 A:

| P_p [dBm] | I_p [A] | dt=0.01 | dt=0.005 | model S=6 | S=10 | S=12 | residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -30.1 | 3.701475e-06 | 19.996 | 24.916 | 26.752 | 27.736 | 27.743 | 2.83 |
| -29.7 | 3.873843e-06 | 25.891 | 29.585 | 29.067 | 30.428 | 30.440 | 0.86 |

Halving the step moves the FDTD **+4.9 / +3.7 dB toward the model**, cutting the
gap 2.7x and 5.3x. Utilisation moves <0.5% over the same refinement
(0.5856 -> 0.5883, 0.6114 -> 0.6125) -- the pump was never in question, only the
small-signal response, exactly where the utilisation agreement had localised it.

**Not yet converged.** Two points cannot fit an order; a third at
`dt_norm = 0.0025` is required before any residual is called physical. Precedent
(`rf_squid_2393_3wm`) needed 4x finer than default.

**RETRACTED:** an earlier entry here reported a "4.8-8.4 dB gap, model reads
high" from the `dt_norm=0.01` data. Most of it was discretisation. The
floor-corrected FDTD values it used (19.37 / 25.63) carried a +-1.4 dB
quadrature assumption that is now moot, the timestep effect being several times
larger.

Eliminated along the way: analysis window (100 -> 800 periods moves <=0.5 dB);
sideband truncation (0.012 dB S=10->S=12); multitone/Floquet parity (0.0002 dB
above). Partly real: the campaign's 3e-08 A probe is compressed 1.2 dB (-30.1) /
2.3 dB (-29.7).

**Consequence: `jc_jtwpa` and `jc_fqjtwpa` FDTD results at `dt_norm = 0.01` are
not timestep-converged for gain.** The linear-limit check that would have caught
this is degenerate on both (`Ic = 0` removes their only inductive path), so
Phase 0.2 of `docs/development/post_ansatz_measurement_solver_plan.md` -- a
finite-linear-inductance variant of that gate -- is now the priority item.

**The FDTD has a signal-tone floor at 2.3-2.5e-7 V** that does not shrink with
window length -- undamped ramp transient, consistent with `implicit_trapezoid`
not being L-stable. Output amplitude at the signal tone over four decades of
drive (jc_jtwpa, -30.10 / -29.70 dBm):

| I_sig [A] | 3e-08 | 3e-09 | 3e-10 | 3e-11 | 3e-12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| -30.10 | 6.07e-6 | 7.46e-7 | 2.73e-7 | 2.34e-7 | 2.30e-7 |
| -29.70 | 1.10e-5 | 1.47e-6 | 3.52e-7 | 2.57e-7 | 2.49e-7 |

Flat to 1.6-3.4% over the last factor of ten, so a 3e-12 A probe reports 63.7 /
64.4 dB of "gain". **Probes at/below 3e-10 A measure the floor -- do not use
them.** 3e-09 A sits only ~16% above it in voltage, so the quadrature
correction carries +-1.4 dB. The floor is higher at the higher pump power, as
parametrically amplified residue should be.

Constraints on any explanation: the *pump* states agree (utilisation 0.6167
Floquet / 0.6162 multitone / 0.6110 FDTD, within 1%), so it is the small-signal
response. The untested candidate is the FDTD timestep -- `jc_jtwpa` has **no**
linear-limit validation because that check is degenerate at `Ic=0` (it removes
the only inductive path), and the one device where the check did apply needed a
4x finer step. Quote TD and HB gains separately until closed.

One further inconsistency found by the same comparison:
- **fqjtwpa `pump_branch_current_max_over_ic` is unusable**: 0.086-0.151 across
  its whole converged column against 0.62-0.89 from the FDTD, and non-monotone
  in pump power. jtwpa's column is fine. Do not use the fqjtwpa value as a
  boundary diagnostic until explained.

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
