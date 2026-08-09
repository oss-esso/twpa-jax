# Full-IPM H3 decay-aware revalidation

## Overall status

`H3_UNRESOLVED_SLOW_RELAXATION`

The full-IPM transition is not confirmed persistent by this milestone. The 11.6
uA trajectory was followed to 120 pump periods and its period-1 distance did not
settle to the validated periodic floor. It first decayed, then levelled and
oscillated. The 16.0 uA trajectory reached period 30 with healthy checkpointed
integration but was stopped before the requested long hold completed; it is not
classified as physical persistence.

## 11.6 uA

The run started from the production-corrected H3 checkpoint, not the rejected
float32 legacy G1 checkpoint. It used the validated implicit trapezoid solver at
`Delta_theta=0.01`.

- ramp: 10 periods
- hold observation: through total period 120
- integration: successful for the completed continuation
- additional continuation: periods 70--120 from a restart checkpoint
- late `d1`: approximately `0.0030`--`0.0034`
- late `d2`: approximately `0.0054`--`0.0069`
- late `d3`: approximately `0.0079`--`0.0103`
- early continuation `d1`: `0.00807` decreasing to about `0.00302`
- decay-aware fit: initially negative slopes, final window slope
  `b=+9.8e-3`; no meaningful positive relaxation time can be assigned
- classification: `UNRESOLVED_SLOW_RELAXATION`
- mean phase winding: `6.99e-5` cycles over the continuation window

The trajectory is therefore not a clean PERIOD_1 convergence and not a
stationary persistent-nonperiodic state. The late oscillatory/nondecaying trend
requires a longer or more targeted observation before deciding between slow
relaxation and persistence.

## 16.0 uA

The same validated H3 start was used with a 30-period ramp and an intended
80-period hold. The run reached a restart checkpoint at approximately period 30
with no transient integrator failure. It was stopped before completion because
the full-IPM sparse trapezoid cost made the remaining bounded hold impractical.
No final `d_n` trend or physical class is assigned.

The previous 10-period H3 result remains historical evidence only; it is not
enough for the decay-aware classification required here.

## Implementation

`h1_transient_branch_transfer.py` now supports periodic restart checkpoints and
records a conservative decay-aware trend estimate. The resume utility is
`scripts/continue_h3_hold.py`. Checkpoints are written under the gitignored
`outputs/` tree every 10 pump periods.

The original G1 checkpoint was explicitly rejected by the current production
validation gate (`2.90e-4` residual). The production-corrected H3 checkpoint was
used for all completed work.
