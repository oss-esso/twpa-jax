# Equivalence Probes

Equivalence probes compare S-parameters from the existing zero-pump path with S-parameters from direct `hblinsolve`.

## Compared Paths

Old path:

```julia
sim_hb = hbsolve(...)
s_hb = sim_hb.linearized.S(...)
```

New path:

```julia
sim_lin = hblinsolve(...)
s_lin = ...
```

The probes compute `max_abs_diff`, `relative_diff`, and pass/fail status.

## Smoke Equivalence Results

| Family | Target | Frequencies | `hbsolve_s` | `hblinsolve_s` | `max_abs_diff` | Status | Evidence |
|---|---|---:|---:|---:|---:|---|---|
| JTL | `harmonia_jtl_linear_jc_smoke` | 11 | 2.304 | 0.309 | 0.0 | PASS | `outputs\jc_profiles\jc3m_m1_jtl_hbsolve_vs_hblinsolve\jtl_hbsolve_vs_hblinsolve_report.json` |
| RF-JTL | `harmonia_rf_jtl_linear_jc_smoke` | 5 | 3.372 | 0.282 | 0.0 | PASS | `outputs\jc_profiles\jc3m_m5_rf_jtl_hbsolve_vs_hblinsolve\rf_jtl_hbsolve_vs_hblinsolve_report.json` |
| ETHZ-JTL | `harmonia_ethz_jtl_linear_jc_smoke` | 5 | 3.411 | 0.302 | 0.0 | PASS | `outputs\jc_profiles\jc3m_m6_ethz_jtl_hbsolve_vs_hblinsolve\ethz_jtl_hbsolve_vs_hblinsolve_report.json` |

These small smoke probes include cold/setup effects. They prove equality for the tested configurations; they do not prove a universal speedup.

## Largest Exact Old-Vs-Direct Comparisons

From `outputs\jc_profiles\jc3m_scaled_direct_linear_showcase\scaled_direct_linear_showcase.csv`:

| Family | Cells | Elements | Nodes | JC tuples | Direct warm time | Old-path warm time | `max_abs_diff_vs_hbsolve` |
|---|---:|---:|---:|---:|---:|---:|---:|
| JTL | 3000 | 6004 | 3002 | 9004 | 0.207 s | 0.211 s | 0.0 |
| RF-JTL | 2393 | 9576 | 4788 | 11969 | 2.470 s | 2.466 s | 0.0 |
| ETHZ-JTL | 2048 | 6653 | 3326 | 8700 | 0.225 s | 0.297 s | 0.0 |

## What These Probes Prove

They prove exact numerical agreement for tested zero-pump linear cases. They also show that warm direct timings are not consistently much faster than warm `hbsolve(...).linearized.S(...)` timings at larger sizes.

## What They Do Not Prove

They do not validate nonlinear pumped gain workflows, nonzero pump current, different frequency windows, or `lumped_jpa_linear`.
