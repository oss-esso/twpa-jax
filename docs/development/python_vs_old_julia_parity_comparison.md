# Python vs Old Julia Parity Comparison

Comparison artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\parity_comparison_25x25
```

Inputs:

- old Julia: `outputs/new_twpa_solver/old_julia_reference_25x25`
- Python parity: `outputs/new_twpa_solver/python_old_julia_parity_25x25`

## Metrics

- aligned cells: 625
- missing old cells: 0
- missing Python cells: 0
- mean absolute gain difference on mutually valid cells: 12.044 dB
- max absolute gain difference on mutually valid cells: 23.770 dB
- valid-mask agreement count: 477
- valid-mask disagreement count: 148
- best old valid point: fp=6.583333333 GHz, Pext=-23.5 dBm, gain=18.663 dB
- best Python valid point: fp=6.0 GHz, Pext=-28.0 dBm, gain=-6.192 dB
- source-power mismatch max: 0.0 dB
- pump-current mismatch max: 1.69e-21 A

## Conclusion

Python old-Julia parity mode matches the old source-power/current convention
to numerical precision, but it does **not** reproduce the old Julia gain map.

The mismatch is expected from the current model:

- old Julia uses an old-IPM JosephsonCircuits netlist with 2508 junctions;
- Python parity uses a 32-cell reduced independent residual;
- old Harmonia couplers include mutual `K` elements and, in newer references,
  CPW-derived distributed coupled-cell parameters;
- Python parity uses a compact coupled-inductor surrogate;
- old run used pump harmonics `(10,)` and modulation harmonics `(5,)`;
- Python parity used pump harmonics 5 and conversion sidebands 3.

Generated plots:

- `plots/old_julia_gain.png`
- `plots/python_parity_gain.png`
- `plots/gain_difference.png`
- `plots/old_julia_valid_mask.png`
- `plots/python_valid_mask.png`
- `plots/mask_difference.png`

## Direct Answers

| Question | Answer | Evidence artifact |
|---|---|---|
| Are we creating the same output as the 25x25 Julia map? | No. Axes and source-current convention match; gain and valid masks do not. | `parity_summary.md` |
| If yes, on which metrics? | Source power and pump current match. | `parity_rows.csv` |
| If no, what differs? | Gain differs by mean 12.044 dB on mutually valid cells; masks disagree in 148 cells. | `gain_difference_grid.csv`, `valid_mask_difference_grid.csv` |
| Are we using the right geometry from the `.jl` files? | Partially. Constants and conventions are ported; full netlist/distributed coupler is not. | `config.json`, docs |
| Which `.jl` details are ported? | axes, 32 dB offset, old current formula, port-equivalent metadata, selected old constants. | `rows.csv` |
| Which `.jl` details are not ported? | 2508-junction netlist, JosephsonCircuits K coupler graph, CPW coupler generator, old HB internals. | this report |
