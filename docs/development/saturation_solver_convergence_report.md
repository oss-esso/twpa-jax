# Saturation solver basis-convergence report

The convergence driver now measures P1dB with real nonlinear bracket refinement,
rather than interpolating the coarse grid. It exposes the required device and
basis dimensions through `--device jtwpa|fqjtwpa`, and its matrix includes Q,
pump order, torus scale, three-tone/lattice, and odd/dense pump-policy cases.

The automated convergence gate passes. A real jtwpa Q=1, odd-order-3 lattice
point completed with `p1db_method=refined` and P1dB `-103.0455 dBm`.

The complete jtwpa matrix and the reduced Q=1/2/3 comparison were attempted.
Both exceeded ten minutes before producing a result file: Q=3 entered repeated
adaptive substeps and Newton stalls at continuation increments down to roughly
0.02--0.04. The fqjtwpa matrix was not started after this demonstrated runtime
limit. Consequently, top-two `|Delta P1dB|` values and a 0.2 dB production-basis
verdict are not available; this is residual uncertainty, not a pass.

