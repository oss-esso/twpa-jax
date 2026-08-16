# Reproduction dossier

## Scientific question and contribution

This is the detailed numerical-method paper behind the later CPR studies. It maps when a realistic 990-cell rf-SQUID JTWPA changes from periodic amplification into chaotic dynamics as pump intensity, signal frequency/power, DC bias, and CPR skewness are varied. Fourier transforms (FTs) and Poincare sections (PSs) are used as primary diagnostics. Appendix A provides the full finite-difference update system, making this the most directly reproducible time-domain solver specification in the set.

## Circuit parameters

The main line has `N = 990` rf-SQUID cells. Per cell: `Lg,n = 120 pH`, `Cg,n = 24 fF`, and a Josephson junction with `Ic = 2 uA`, `RJ = 20 kOhm`, `CJ = 200 fF`; the stated plasma frequency is `27.7 GHz`. The input cell has pump + signal voltage sources, `Ri = 50 Ohm`, `Ci = 24 fF`. The output has `Rl = 50 Ohm`, `Cl = 1 nF` and the measured voltage `Vout`.

## Governing model

Equation (1) is the RCSJ law. Equation (2) gives the phase-dependent Josephson inductance for a sinusoidal CPR, and the text gives its DC-bias dependence. The line equations are not reduced to coupled modes; Kirchhoff equations for every cell are discretized directly. Equation (4) defines effective nonlinear coefficients `beta` and `gamma` used later as diagnostics; Equation (5) introduces the transparency-dependent CPR.

Appendix A, Eqs. (A1)-(A62), is the authoritative solver derivation. It includes current balances, RCSJ and inductor equations, mesh voltage equations, normalization, central finite differences, interior recurrence coefficients, both boundary conditions, the final tridiagonal matrix, and current updates. `equations_index.md` contains all numbered appendix equations with page-local context.

## Numerical algorithm

The integration uses an implicit finite-difference method and a tridiagonal solve at each step. Dimensionless `Delta t = 0.01`, `tmax = 20000`; the paper maps these to about 0.06 ps and 120 ns for the stated junction parameters. The initial state in Appendix A sets the phase at the two starting time levels to zero. The long run suppresses initial transients before FT/PS analysis.

A direct implementation sequence is:

1. Normalize the electrical variables exactly as Appendix A defines them.
2. Assemble interior coefficients (`A21-A26`) for all cells.
3. Apply the left input boundary (`A27-A43`) with pump, signal, source impedance/capacitance.
4. Apply the right load boundary (`A44-A56`).
5. Solve the tridiagonal phase system (`A57-A58`) each time step.
6. Update currents using (`A59-A60`) and iterate.
7. Restore dimensional output voltage/current and discard transient data.
8. FFT `Vout`; compute gain `20 log10[Vout(nu_sign)/Vsign]`.
9. Build a Poincare section from `Vout'` sampled when `Vout=0`. A tight cluster indicates periodic motion; broad scattered points coincide with chaos.

## Main scans and expected structure

The standard example uses `nu_pump = 7 GHz`, a weak signal near `6.42 GHz` at `-100 dBm`, and initially zero DC bias. In the pump sweep, moderate stable gain is observed below roughly `-54 dBm`, followed by an abrupt regime change around `-54` to `-53.5 dBm`; above that, very large apparent gain is accompanied by broadband/complex spectra and should not be interpreted as usable amplifier gain.

Signal-frequency sweeps from roughly 4 to 10 GHz show broadband response and many mixing products. Narrow vertical gain features/ripple are attributed to reflection/impedance-mismatch effects rather than a new nonlinear mechanism.

With DC bias, the response repeats with the flux/phase period expected from the rf-SQUID loop. The paper gives a current scale near `Phi0/Lg ~ 17.1 uA`; stable and chaotic windows repeat as the phase winds. The 2D `(Ppump, Ibias)` maps show that gain alone cannot classify the operating state, whereas the PS spread changes abruptly in chaotic regions.

## Transparency-dependent CPR

The later sections replace `Ic sin(phi)` with the paper's Eq. (5), controlled by transparency/skewness `tau`. Sweeps use representative `tau = 0.25, 0.5, 0.75, 0.99`. Increasing skewness tends to reduce the maximum gain while widening the pump range over which the response stays regular, and makes the onset of chaos sharper. At very high skewness, some DC-bias periodic stability windows disappear.

## Figure interpretation

Figure 1 is the full electrical line including input/output boundary cells. Figures 2-3 establish the FT/PS classification versus pump, signal frequency and DC bias, including two-dimensional stability/gain maps. Figure 4 maps the effective nonlinear coefficients and their fluctuations. Figures 5-7 repeat the dynamical analysis for non-sinusoidal transparency-dependent CPRs. Appendix figures/diagrams and every extracted caption are indexed separately.

## Strong reproduction checks

Do not validate only on a gain trace. Reproduce simultaneously: (a) a narrow-line periodic FT below the instability; (b) a compact PS; (c) sudden spectral broadening and PS dispersion at the instability; (d) the bias-periodic windows; and (e) the transparency-induced change in the stability/gain tradeoff. These jointly test the nonlinear dynamics and boundary implementation.
