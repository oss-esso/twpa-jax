# Gain-map workflows

The canonical entry point is `workflows/run_gain_map_and_plots.py`.

`--fast` runs the ordinary production HB map and is the default when neither mode
flag is supplied. `--slow` runs the bounded HB-recovery plus transient physical
boundary controller. Slow mode does not integrate every point: it uses HB for the
bulk of a column and invokes TD only after the bounded HB path reaches an
obstruction.

Examples:

```powershell
python workflows/run_gain_map_and_plots.py --fast `
  --design designs/ipm_2c_fixed --n-frequency 20 --n-power 20 `
  --frequency-workers 4 --outdir outputs/gain_maps/fast_2c

python workflows/run_gain_map_and_plots.py --slow `
  --design designs/ipm_2c_fixed --n-frequency 20 --n-power 20 `
  --frequency-workers 4 --outdir outputs/gain_maps/slow_2c
```

All generated files belong below the selected output root. Compact summaries use
the same raw and physically eligible coverage accounting in both modes. Physical
boundary, transient numerical failure, and unresolved statuses remain distinct.
Frequency workers are isolated processes; power points remain sequential within a
column. Completed column subtrees are reusable on restart.
