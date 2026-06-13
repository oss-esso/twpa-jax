# JC3M Old-IPM 5x5 Reference Inventory

Reference root:

```text
D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_old_ipm_power_map_5x5_marked
```

This folder is the stored JosephsonCircuits reference for backend substitution comparisons.

| Artifact | Present | Purpose |
|---|---:|---|
| `report_old_ipm_power_map_rows.csv` | yes | row-level map data, statuses, solver logs, gain trace JSON |
| `raw_gain_max_db_grid.csv` | yes | raw maximum S21 gain grid |
| `convergence_mask_grid.csv` | yes | cells classified as converged |
| `finite_mask_grid.csv` | yes | finite S-parameter/gain mask |
| `solver_warning_mask_grid.csv` | yes | solver warning mask |
| `residual_norm_grid.csv` | yes | parsed nonlinear residual norm grid |
| `infinity_norm_grid.csv` | yes | parsed infinity norm grid |
| `status_grid.csv` | yes | per-cell status labels |
| `convergence_masked_gain_max_db_grid.csv` | yes | gain grid blanking nonconverged cells |
| `report_old_ipm_power_map_summary.md` | yes | summary table and status counts |
| `reproduction_status_summary.md` | yes | convergence interpretation and artifact list |
| `old_ipm_gain_map_power_frequency_marked_summary.md` | yes | plot summary |
| `old_ipm_gain_map_current_frequency_marked_summary.md` | yes | plot summary |
| `plots/` | yes | marked gain-map figures |

The backend comparison uses this folder as the baseline. It does not create a new reference baseline.
