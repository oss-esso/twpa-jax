# Final pre-map gate: storage and 7.9 GHz TD-to-HB handoff

## Storage audit

The completed 7.0 GHz hybrid output was approximately 86.3 MB. The dominant
files were:

| Artifact | Approximate size |
|---|---:|
| `td_bridge_01/late_time_phase.npz` | 78.0 MB |
| eight full HB `pump_solution.npz` files | 7.5 MB |
| TD CSV/checkpoint/plots/JSON | <1 MB |

Compact mode is now enabled by the hybrid CLI:

- TD writes `td_compact.npz`, scalar stroboscopic metrics, spectrum, and the
  restart checkpoint; full phase history, CSV, and plots are omitted.
- HB writes compact per-point summaries and retains only the latest successful
  float64 pump state needed to continue or start TD.
- Failed/obsolete point states are removed from the active run directory.
- Default production map storage remains unchanged and continues to use
  float32; hybrid restart states explicitly use float64.

Based on the measured breakdown, the completed-column payload is expected to be
approximately 1 MB rather than 86 MB, dominated by one float64 HB state and a
restart checkpoint.

## 7.9 GHz handoff attempt

Source: `outputs/boundary_79_11p4_hold10/restart_checkpoints/transient_restart.npz`,
the existing 11.40 µA PERIOD_1 run. A minimal fixed-drive continuation was run
for six periods at Δθ=0.01, followed by the existing five-period Fourier
projection and production HB restart.

| Quantity | Result |
|---|---:|
| drive | 11.40 µA |
| fixed-drive continuation | 6 periods, 3770 steps |
| projection error | 1.13e−3 |
| projected HB residual | 2.07e−2 |
| production restart residual | 1.24e−2 |
| restart Newton iterations | 4 |
| restart runtime | 3.27 s |
| status | `HB_RESTART_VALIDATION_FAILED` |

The earlier coarse-step attempt had a 9.86% projection error, so it was not
used as evidence. The validated-step attempt still fails the production
residual gate and therefore does not establish `TD_TO_HB_RESTART_VALIDATED`.

## Final gate

Storage policy is implemented and tested, but the required one-time TD-to-HB
handoff has not passed. No solver physics was changed and no overnight map was
started.

```text
NOT_READY_FOR_OVERNIGHT_MAP
```
