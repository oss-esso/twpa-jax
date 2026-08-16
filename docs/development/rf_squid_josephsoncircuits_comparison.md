# RF-SQUID 3WM Circuit Comparison

This note records the differences found between the predecessor's
`JosephsonCircuits.jl` scripts in `D:\Projects\Thesis\Harmonia` and the current
`twpa_jax` RF-SQUID 3WM model. The Julia scripts were inspected without
modification.

Compared scripts:

- `D:\Projects\Thesis\Harmonia\RF_squid.jl`
- `D:\Projects\Thesis\Harmonia\RF_squid_test_ws_num.jl`
- `D:\Projects\Thesis\Harmonia\rf_SQUID_2D_plot.jl`

## Circuit and topology

The predecessor scripts construct a nominal 2393-cell chain using
`add_RF_JTL_element!(..., Lrf; Lp=8.9e-12)`, followed by a separate `Lw` element
after every RF-JTL element. They use:

```text
Ic = 0.93 uA
Lrf = 58.6 pH
Lp = 8.9 pH
Lw = 37.0 pH
Cj = 15 fF
```

Both ends have 50 ohm resistive terminations. Port 1 is the input and port 2 is
the output. The output node also receives `Lg = 20 nH` to ground for the DC
return.

The helper implementation is now known explicitly. For the predecessor call
with `Lp = 8.9 pH`, one cell beginning at node `j` stamps:

```julia
Cg:       j       -> ground
Lp:       j       -> j+1
JJ + Cj:  j+1     -> j+2
Lrf:      j       -> j+2
```

The helper returns node `j+2`. The subsequent script-level `Lw` element then
stamps:

```julia
Lw:       j+2     -> j+3
```

The exact nonlinear cell is therefore a ground capacitor at the incoming node,
followed by a propagation section in which the series `Lp + JJ` branch is in
parallel with `Lrf`, followed by the separate series `Lw` inductance. The
`Lp + JJ` branch and the direct `Lrf` branch form the intended RF-SQUID loop;
the Josephson junction is not a self-loop.

The helper uses `sigma = 0.0` by default, so the predecessor cell is lossless
in the JosephsonCircuits model. No resistor is inserted into the junction branch
by this function. The wrapper `add_RF_JTL!` simply repeats the same element
construction `N_cell` times.

This confirms the intended local branch topology in the predecessor scripts. A
remaining comparison task is only to verify that the current production builder
stamps the same graph and branch orientation.

### Confirmed cell-count discrepancy

All three scripts contain:

```julia
total_cells = 2393
num_full_periods = div(total_cells, 24)
remainder_cells = rem(total_cells, 24)
```

but only append `num_full_periods` blocks. Therefore they generate

```text
99 * 24 = 2376 cells
```

and silently omit the remaining 17 cells. The remainder variable is never used.
The current production design is intended to represent all configured cells.

### Periodic ground-capacitance sequence

The predecessor uses four groups of six cells per 24-cell block:

```julia
[Cg1, Cg3, Cg1, Cg2]
```

with values:

```text
Cg1 = 10.5 fF
Cg2 = 50.4 fF
Cg3 = 68.2 fF
```

Thus the numerical sequence is:

```text
10.5 fF, 68.2 fF, 10.5 fF, 50.4 fF
```

The variable names `Cg2` and `Cg3` are reversed relative to the paper labels if
the paper convention is `C2 = 68.2 fF` and `C3 = 50.4 fF`. The numerical order
may still be intentional, but the naming is a provenance hazard.

## Flux-bias convention

The predecessor uses a direct-current approximation:

```julia
target_flux = 0.33 * Phi0
Idc = target_flux / 58.6e-12
```

This gives approximately `Idc = 11.64 uA`. It treats the applied flux as
`Phi_ext = Lm * Idc` and does not explicitly solve the self-consistent
RF-SQUID equation:

```text
phi_dc = phi_ext - beta_L * sin(phi_dc)
```

The current model uses the self-consistent convention. Therefore the two models
can have different static junction phases even when both are labelled
`Phi_ext = 0.33 Phi0`.

The Julia source applies both DC bias and AC pump at port 1:

```julia
(mode=(0,), port=1, current=Idc)
(mode=(1,), port=1, current=Ip)
```

The resulting DC distribution depends on the complete RF-SQUID helper topology
and the output-side DC return.

## Pump and signal settings

| Script | Pump frequency | Pump current |
|---|---:|---:|
| `RF_squid.jl` | 12.311 GHz | 3 uA |
| `RF_squid_test_ws_num.jl` | 12.28 GHz | 2 uA |
| `rf_SQUID_2D_plot.jl` | 12.101--12.301 GHz | 2.1--2.7 uA |

The current reproduction target is 12.08 GHz and uses calibrated pump power as
the main input, so the predecessor's fixed-current points are not directly
equivalent to current `-60 dBm` or `-56 dBm` coordinates.

The predecessor's broad signal sweep is approximately 2.01--13.01 GHz. The 2D
script locks the signal near the degenerate region:

```text
fs = fp / 2 - 10 MHz
```

## Harmonic and nonlinear settings

The predecessor uses:

```julia
Npumpharmonics = (8,)
Nmodulationharmonics = (4,)
dc = true
threewavemixing = true
fourwavemixing = true
```

The explicit DC source and `dc=true` indicate that the JosephsonCircuits solve
includes a DC component. This dynamic pump-induced DC component must not be
confused with the prescribed external flux bias.

The predecessor enables both 3WM and 4WM, so it is not a pure 3WM-only
calculation. The current validation workflow uses a dense pump basis and tracks
3WM idler and unwanted-product channels explicitly.

## Gain convention

The predecessor extracts absolute pumped transmission:

```julia
10 * log10(abs2(S21))
```

The current workflow reports pumped transmission relative to a pump-off reference.
Those values cannot be compared directly without matching the port normalization
and subtracting the same cold-device baseline.

The predecessor's 2D sweep also stores complete solver objects for every point in
`results_matrix` and passes them to `save_hdf5`. The current map workflow stores
compact summaries and retains full states only for restart-critical points.

## Matching and stopbands

`RF_squid.jl` and `RF_squid_test_ws_num.jl` calculate a matching capacitor at
`fp/2` from `Lg`, but the corresponding AC capacitor is commented out. The
calculated matching element is therefore not active in the shown circuit.

The predecessor scripts do not explicitly report the pump-off-relative gain,
idler conversion, or unwanted-product suppression used in the current 3WM
validation workflow.

## Execution note

The Julia runtime and the local package at `D:\Projects\Thesis\Harmonia.jl` were
located. An attempted execution of `RF_squid.jl` reached the full 2393-cell
calculation but did not finish within the available run window; it produced only
its startup message. The other two scripts were not launched after that
incomplete first run.

No predecessor script was edited while preparing this comparison.




 After applying the JosephsonCircuits current-factor correction:

   Quantity                                         JosephsonCircuits run                                Our run
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Pump frequency                                              12.311 GHz                              12.08 GHz
  ───────────────────────────────  ───────────────────────────────────────  ─────────────────────────────────────
   Nominal pump current                                              3 µA               equivalent to about 6 µA
  ───────────────────────────────  ───────────────────────────────────────  ─────────────────────────────────────
   Corrected pump power                                      ≈ −60.45 dBm                                −60 dBm
  ───────────────────────────────  ───────────────────────────────────────  ─────────────────────────────────────
   Cell count actually simulated                                     2376         2393 nominal production design
  ───────────────────────────────  ───────────────────────────────────────  ─────────────────────────────────────
   Flux bias                        0.33 Φ₀, direct-current approximation    0.33 Φ₀, self-consistent convention

  Thus the pump amplitudes are effectively very close: the Julia run is only about 0.45 dB below our −60 dBm run, while its frequency is 0.231 GHz higher.

  The generated Julia spectrum is here:

  D:/tmp/harmonia_rf_squid_run/RF_squid_gain_spectrum.png

  It visually shows approximately 17–18 dB gain across roughly 4–8 GHz. Our validated on-chip run at 12.08 GHz, −60 dBm gave:

  mean gain, 4–8 GHz: 8.53 dB
  maximum gain, 3–9 GHz: 9.57 dB

  However, the Julia run did not converge:

  iterations: 1000
  relative residual: 7.77e-6
  infinity residual: 8.66e-2

Therefore the apparent ~17–18 dB Julia gain is not yet a valid quantitative comparison. It is a nonconverged JosephsonCircuits result, and it also uses a different cell count,
frequency, gain normalization, and flux-bias implementation.

## Controlled `Cj` comparison at the predecessor operating point

Two additional runs isolated the junction-capacitance change while keeping the
predecessor conditions fixed:

| Variant | Cells | Pump frequency | Passed pump current | `Cj` | Solver result |
|---|---:|---:|---:|---:|---|
| `current_op_legacy_cells` | 2376 | 12.311 GHz | 3.0 µA | 15 fF | did not converge after 1000 iterations |
| `current_op_legacy_cells_cj20` | 2376 | 12.311 GHz | 3.0 µA | 20 fF | did not converge after 1000 iterations |

The reported nonlinear residual worsened substantially when `Cj` was changed
from 15 fF to 20 fF:

| Variant | `norm(F)/norm(x)` | Infinity norm |
|---|---:|---:|
| 15 fF | `7.77e-6` | `8.66e-2` |
| 20 fF | `2.48e-4` | `1.21e1` |

The corresponding spectra are stored in the repository output area:

- `outputs/jc_rf_squid/rf_squid_current_op_legacy_cells/gain_spectrum.png`
- `outputs/jc_rf_squid/rf_squid_current_op_legacy_cells_cj20/gain_spectrum.png`

The 20 fF run showed substantially lower apparent gain than the 15 fF run.
This is physically plausible because increasing `Cj` changes the junction
plasma frequency, distributed dispersion, impedance, and phase matching.
However, both runs are nonconverged, and the 20 fF residual is much worse.
Consequently, the gain reduction is evidence of strong parameter sensitivity,
not yet a converged quantitative gain comparison.
