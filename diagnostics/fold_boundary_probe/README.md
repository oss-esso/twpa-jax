# 6.91 GHz pump-branch boundary probe

This diagnostic follows the last converged state from
`outputs/quick_7c_691_arclength_no_skip_v3` on a 0.01 dB grid. It was run with
the same 7-row circuit, ten positive odd pump modes, Schur backend, and exact
real-coupled preconditioner as the 50-point campaign.

## Localized boundary

| External pump power (dBm) | Pump solve | Gain (dB) |
| ---: | :---: | ---: |
| -19.132857 | PASS | 25.6413 |
| -19.122857 | PASS | 24.6840 |
| -19.112857 | PASS | 24.0343 |
| -19.102857 | PASS | 23.6051 |
| -19.092857 | FAIL | n/a |

The 0.01 dB natural-parameter continuation stops between -19.102857 and
-19.092857 dBm. The first missing production point, -18.857143 dBm, is about
0.24 dB beyond this numerical boundary. This is not by itself proof that the
periodic HB branch ends there: the later 2c comparison in
`diagnostics/2c_measurement_comparison` crossed analogous apparent walls by
reducing the continuation step to 0.005--0.01 dB.

The saved converged pump states live below `warm/points/`. The narrow Tier-1
Floquet sweep for the last state is `stability_boundary_narrow.json`; its
deepest real-axis resonance is at 6.632759 GHz with
`sigma_min = 4.605282e3`. Targeted complex refinement converged to
`6.662963 + 0.087738j GHz`, or growth rate `-5.512751e8 1/s` under the
repository's `exp(+i omega t)` convention. The last available periodic orbit
is damped, so there is no evidence for a growing Floquet mode on that orbit.
Whether the forced periodic branch truly terminates or merely enters a very
narrow continuation basin remains open.

## Recovery attempts

- Relaxing the four-step stall guard and lowering the minimum line-search step
  improved the residual but stopped at a nonzero floor (`1.812e-2`).
- Pseudo-transient solves over five shift scales stopped at the same floor.
- Pump quadrature at `nt = 40, 80, 160` did not restore a root.
- Linear, softened-Josephson, phase-rotated, and amplitude-scaled seeds did not
  converge. The best phase-flipped seed stopped at `1.863e-3`.
- Two-point pseudo-arclength continuation spent 300 s in a snaking cluster,
  detecting 17 turns over normalized drive `0.848..0.894`; the first missing
  target is `0.906`. It did not produce a target crossing.

## Measurement comparison

`docs/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK/Images/Gains_data.mat`
contains the same qualitative transition. At 7.043 GHz, extracted peak gain
rises to 17.79 dB at -21.754 dBm and drops to -9.51 dB at the next power step
(-21.453 dBm). Across all 51 measured pump frequencies, the largest adjacent
gain drop has median -25.86 dB, with the pre-drop pump power ranging from
-23.857 to -19.050 dBm (median -21.453 dBm).

The measured transition is shifted in absolute power and frequency relative
to this 7-row model, but it independently shows that the amplifying regime
does not continue smoothly through the high-power side.
