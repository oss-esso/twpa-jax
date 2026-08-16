# Reproduction dossier

## Scientific question and contribution

The paper studies a *single* microwave-driven Josephson junction and shows that subharmonic Shapiro steps and the chaotic windows between them form a highly organized, self-similar structure related to a devil's staircase. It combines current-voltage curves, Lyapunov exponents, Poincare sections, fractal box-counting, and fine parameter scans. The route to chaos on subharmonic steps follows a Feigenbaum period-doubling cascade.

## Dynamical system and normalization

The normalized RCSJ equations are Eqs. (1)-(2): a driven damped pendulum written as first-order voltage/phase equations. Current is normalized by `Ic`; voltage by the stated Josephson/plasma voltage scale; time by the inverse plasma frequency. The damping parameter is related to the McCumber parameter. The principal numerical example uses `beta = 0.3`, normalized radiation frequency `omega = 0.5`, and amplitude `A = 0.8`. A tiny white-current-noise term may be used for numerical tests but most results are deterministic.

## Numerical protocol

The authors use fourth-order Runge-Kutta with time step `1/32`. Transient durations range from approximately `10^3` to `10^5` normalized time units and averaging intervals from `10^4` to `10^5`, depending on the fine structure being resolved. The DC current sweep can require steps as small as `10^-5` to `10^-8`. The continuation direction matters: start above the critical-current region and decrease the DC current, using the final state at one bias point as the initial state for the next. That branch-following procedure is important for reproducing the plotted staircase.

## Diagnostics

- Current-voltage characteristic: mean normalized voltage versus DC current.
- Largest Lyapunov exponent: positive values identify chaos; the sum of the two exponents obeys the damping sum rule given in the text.
- Poincare sections: the number of points/curves on a phase-locked step reflects the denominator of the rational locking ratio.
- Box counting: fit the staircase/chaotic-set scaling to obtain the reported fractal dimension `D = 0.868 +/- 0.012`.
- High-resolution period-doubling sequences: extract successive bifurcation parameter spacings and orbit scales, then form the Feigenbaum ratios in Eqs. (6)-(7), approaching the universal constants.

## Mathematical structure

Equation (3) writes the rational/continued-fraction hierarchy of Shapiro/subharmonic locking. Equation (4) gives the fitted step/window-width scaling law. Eqs. (6)-(7) are the Feigenbaum scaling ratios. All exact equations are in `equations_index.md` and the page text.

## Figure interpretation

The early figures show the CVC with alternating rational subharmonic plateaus and chaotic windows (the authors nickname this structure). Subsequent plots measure the widths and establish the fractal/scaling behavior. Poincare sections connect each rational step with a finite periodic orbit. Zoomed chaotic windows reveal repeated/self-similar organization. A Farey/devil's-staircase construction explains why the chaos is structured rather than randomly interspersed. High-resolution plots then demonstrate period doubling and Feigenbaum universality. Final parameter scans show that the structured chaos persists when radiation amplitude and damping are changed.

## Reproduction requirement most likely to be missed

Very coarse current sweeps or resetting the initial state independently at every bias point will erase the fine windows and can produce a qualitatively different plot. Use the paper's continuation direction, long settling/averaging times, and progressively refine the DC-current step around each subharmonic window.
