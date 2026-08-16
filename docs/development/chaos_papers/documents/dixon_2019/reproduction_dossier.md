# Reproduction dossier

## Scientific question and contribution

The paper tests whether the conventional three-tone coupled-mode-equation (CME) description of a three-wave-mixing rf-SQUID JTWPA is sufficient when the full Josephson nonlinearity is allowed to generate every frequency component. The reference calculation is a full transient WRspice simulation. Its central result is that pump harmonics and pump-mediated sum-frequency tones drain energy from the intended pump/signal/idler manifold and can reduce the predicted signal gain substantially. The authors then enlarge the CME state systematically (CME-1 through CME-5) until it approaches the WRspice result.

## Device model and exact operating point

The simulated transmission line contains 2000 rf-SQUID cells. Each cell has geometric inductance `Lg = 57 pH`, junction critical current `Ic = 5 uA`, junction capacitance `CJ = 60 fF`, and capacitance to ground `C0 = 100 fF`. The array is flux biased into a three-wave-mixing regime. The principal test uses a pump at `fp = 12 GHz`, approximately `Irms_p(0) = 1.97 uA` (`~ -70 dBm` in the paper's convention), and a weak signal at `fs = 7.2 GHz`, approximately `Irms_s(0) = 0.07 uA` (`~ -96 dBm`). The desired idler is `fi = fp - fs = 4.8 GHz`.

## Mathematical model

The derivation begins from a multi-tone travelling-wave flux ansatz and the rf-SQUID wave equation. The linear dispersion is retained, while the nonlinear terms are organized through the quadratic coefficient `beta`; for the pure 3WM comparison the cubic term is set to zero. The exact numbered equations, including all appendices, are indexed in `equations_index.md`; use the original-layout text in `source_fulltext_layout.txt` whenever a symbol is ambiguous.

The important structure is:

- Multi-tone state: complex envelopes `A_j(x)` multiplying `exp[i(k_j x - omega_j t)] + c.c.`.
- Linear dispersion: `k(omega) = omega / [omega0 sqrt(1 - omega^2/omegaJ^2)]`, with `omega0 = 1/sqrt(Lg C0)` and `omegaJ = 1/sqrt(Lg CJ)`.
- Conventional CME-1 retains only pump, signal, idler.
- CME-2 additionally retains second pump harmonic and pump-mediated sum tones; CME-3, CME-4 and CME-5 continue the same rule through the third, fourth, and fifth pump harmonics. The full extended systems are written in Appendices A-C.
- Equation (14) maps the CME travelling-wave amplitude back to the RMS current convention used by WRspice.

## Numerical methodology

1. Build the 2000-cell rf-SQUID line in WRspice with the same component values and DC flux bias.
2. Drive pump and signal at the input; perform a transient long enough to reach steady response.
3. Record the current entering every cell/node and FFT each record. This produces a position-frequency map with no prior restriction on generated tones.
4. Independently integrate CME-1 through CME-5 along propagation coordinate `x`; the paper uses MATLAB `ode45`.
5. Convert CME amplitudes to RMS current using Eq. (14), then compare the same tones against the WRspice FFT.
6. Repeat at the reduced pump amplitude used by the authors (`~0.67 uA`) to enforce the small-phase assumption of the analytic derivation.

## Reproduction targets and interpretation

At the high pump amplitude, WRspice visibly generates pump harmonics through at least the fifth harmonic and associated sum-frequency products. The signal gain is much smaller than the simplest three-tone theory predicts. The phase swing across the junction becomes too large for the truncated small-phase CME assumptions. At the lower pump amplitude the approximation is better: each successive CME extension transfers more energy into the additional modes and moves the signal prediction toward WRspice. A decisive reproduction check is therefore *monotonic improvement of the multi-tone CME relative to CME-1*, not merely matching one gain number.

The paper reports a representative conventional-CME prediction near 20 dB at a propagation location around node 1175 while WRspice is around 8.9 dB; CME-5 substantially closes that discrepancy. Exact curves and node locations should be read from the figure index and full source text rather than inferred from this summary.

## Figure logic

Figure 1 defines the repeated rf-SQUID cell. Figure 2 is the key unconstrained WRspice position-frequency map and establishes the existence of harmonics/sum tones. Figure 3 compares selected CME-5 mode amplitudes with WRspice. Figure 4 diagnoses the junction phase swing and motivates lowering the pump. Figure 5 shows how successively enlarged CME sets reproduce additional generated modes. Figure 6 shows the corresponding convergence of the signal-amplitude/gain prediction. `figures_tables_index.md` contains every extracted caption plus nearby textual discussion.

## Reproduction pitfalls

The analytic comparison is only meaningful when its approximations match the transient simulation: small junction phase, consistent dispersion and amplitude normalization, sufficient transient settling, and identical input amplitudes. Do not force WRspice to retain only the analytic tones—the unrestricted spectrum is precisely the reference being tested.
