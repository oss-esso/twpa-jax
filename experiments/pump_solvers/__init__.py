"""Experimental, GPU-compatible / matrix-free pump-solver backends.

These backends are opt-in and do not change the legacy exp08 solve path. The
goal is to beat the high-power fold runtime of the ``real_coupled`` full-Jacobian
backend (~4.42 s/point at -22 dBm) while preserving exp09 gain to < 0.01 dB.

Modules
-------
schur_partition : node partition (retained nonlinear/port block vs eliminated
    linear-internal block) + per-harmonic eliminated-block factorization and
    back-substitution.
schur_operators : ``SchurReducedProblem`` -- a drop-in replacement for
    ``FullIPMPumpProblem`` that the existing Newton-Krylov solver can drive,
    with all operators reduced to retained nodes and the linear elimination
    applied matrix-free via prefactored eliminated blocks.
jvp_backends : analytic (production) vs finite-difference (sanity) vs optional
    JAX directional-derivative JVP, with a cross-check helper.
"""

from __future__ import annotations
