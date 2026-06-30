# Pump Speedup Single-Point Benchmark

Recommended opt-in mode: `linear_seed_adaptive` (8.38x pump speedup).

| variant | accepted | pump s | speedup | gain dB | gain delta dB | coeff rel | time rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_cold_fixed | True | 5.51241 | 1 | 0.703249 | nan | 7.8e-13 | 0.000493 |
| linear_seed_fixed | False | 6.09767 | 0.904 | 0.703249 | 2.44e-12 | 7.58e-13 | 0.000493 |
| linear_seed_adaptive | True | 0.657802 | 8.38 | 0.703249 | -5.18e-10 | 1.51e-10 | 0.000493 |
| linear_seed_adaptive_tol1e8 | True | 0.643037 | 8.57 | 0.703249 | -5.18e-10 | 1.51e-10 | 0.000493 |
