# Reproduction dossier

## Scientific question and contribution

This study asks how adding a second harmonic to the Josephson current-phase relation (CPR) changes gain and dynamical stability of a 990-cell JTWPA. The full time-domain line is integrated without reducing it to a few coupled modes. Gain curves are interpreted together with Fourier spectra, phase portraits, and Poincare sections, allowing the authors to distinguish useful periodic amplification from high apparent gain caused by chaotic dynamics. The main reported design point is a negative second-harmonic weight near `Jc2/Jc1 = -0.6`, which can provide roughly 13 dB gain without dispersion engineering.

## Circuit and boundary conditions

The line has `N = 990` cells. Each cell contains an rf-SQUID: one Josephson junction in parallel with `Lg,n = 120 pH`, with `Cg,n = 24 fF` to ground. Junction parameters are `CJ = 200 fF`, `RJ = 20 kOhm`. The input uses a 50-ohm source; the output is a 50-ohm load separated by `Cl = 1 nF` for DC decoupling. Boundary matching is emphasized because it materially changes nonlinear/chaotic behavior in Josephson transmission lines.

## Governing equations

Equation (1) is the RCSJ current through each junction. The nonlinear supercurrent is replaced by

`I(phi) = Jc1 sin(phi) + Jc2 sin(2 phi)`

[Eq. (2)]. The associated Josephson potential is Eq. (3). Eqs. (4)-(5) give the ground-state condition and show when non-trivial minima occur. The ratio `g = Jc2/Jc1` is used to organize the CPR family. Eqs. (6)-(7) derive the maximum CPR amplitude; this is then used to renormalize the coefficients so the *physical critical current remains fixed at `Ic = 2 uA`* while only the CPR shape changes. That normalization is essential for a fair reproduction.

All numbered equations and page-local extraction are in `equations_index.md`.

## Time-domain solver

The authors solve the coupled cell equations using an implicit finite-difference method with a tridiagonal algorithm, a standard approach for Josephson transmission lines. In normalized plasma-frequency units the step is `Delta t = 1e-2` and the total simulated interval is `tmax = 2e4`. Use the same line/load boundary equations as the prior numerical method cited by the paper rather than integrating 990 uncoupled RCSJ equations.

## Analysis workflow

1. Choose `Jc1 = 1` and sweep `Jc2` in the stated range, but renormalize the whole CPR so `max_phi I(phi) = Ic = 2 uA`.
2. Integrate the complete 990-cell line for pump + weak signal.
3. Remove transient portions consistently with the paper and compute the Fourier transform of `Vout`.
4. Define gain as `20 log10[Vout(nu_sign)/Vsign]`.
5. Form a Poincare section from values of `Vout'` when `Vout = 0`; use the standard deviation of the Poincare-section ordinate as a compact chaos indicator.
6. Sweep pump power and second-harmonic weight to reproduce the two-dimensional stability/gain map.

A principal operating example uses `nu_pump = 7 GHz`, `nu_sign = 6 GHz`, `Psign = -100 dBm`, and zero DC bias. For `Jc2 = -0.6`, the pump sweep develops useful high gain before the chaotic transition near the value reported in the paper (around `-59.5 dBm` in the plotted example). Both a 4WM idler (`2 nu_p - nu_s = 8 GHz`) and a 3WM difference tone (`nu_p - nu_s = 1 GHz`) appear even without a DC/magnetic bias because the altered CPR changes the symmetry/nonlinearity.

## Interpretation of the main figures

Figure 1 establishes the unit cell, ground-state map, an example CPR, and its potential. Figure 2 is the detailed dynamical diagnosis at negative second harmonic: output FT, gain versus pump, a spectral map versus pump, phase-space portrait, and Poincare section. Figure 3 maps gain and the Poincare-spread chaos diagnostic over pump power and `Jc2`. The negative-`Jc2` region can raise gain but also brings stronger sensitivity to the onset of chaos. The small-phase expansion discussed in the text, `gamma(phi,g) ~ (1+2g)phi - (1+8g)phi^3`, explains special behavior around `g=-1/2` and `g=-1/8`.

## Reproduction checks

A successful reproduction should recover: (i) the ground-state changes at the predicted `g`; (ii) comparable gain topology over `(Ppump, Jc2)`; (iii) a sharp increase in Poincare-section spread coincident with broadband/complex Fourier spectra; and (iv) the high-gain negative-second-harmonic region without changing the fixed physical `Ic`.
