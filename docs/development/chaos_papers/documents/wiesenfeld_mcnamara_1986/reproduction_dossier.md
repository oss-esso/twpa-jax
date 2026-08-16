# Reproduction dossier

## Scientific question and contribution

This is a general bifurcation-theory paper rather than a JTWPA model. It proves that a periodically driven dynamical system becomes a narrow-band small-signal amplifier near a codimension-one instability. The location and scaling of the susceptibility peaks depend primarily on the bifurcation class: saddle-node/transcritical, pitchfork, period doubling, or Hopf. The authors validate the period-doubling result with an analog Duffing oscillator and then interpret Josephson parametric amplification as an example of the same near-bifurcation susceptibility.

## General method

The starting point is a periodically forced nonlinear ODE `xdot = F(x,t)` (or a parameterized form). Introduce a weak additive or parametric modulation with frequency `Omega`. Linearize about the stable unperturbed periodic orbit `x0(t)`. Solve the linear variational equation using Floquet theory: each homogeneous mode has a periodic Floquet eigenvector multiplied by `exp(mu_k t)`. Expand the forcing and periodic vectors in Fourier series, use variation of constants, and derive the spectral response.

As a control parameter approaches a bifurcation, one Floquet exponent (or a conjugate pair) approaches the imaginary axis. That near-critical denominator dominates the response. The resulting power spectra have resonance centers determined by the imaginary part of the critical exponent and widths determined by its small negative real part. This is the reusable calculation to implement for any periodically driven circuit.

## Bifurcation-specific predictions

The paper derives explicit spectra for saddle-node, transcritical, pitchfork, period-doubling and Hopf instabilities in Section III. Near period doubling, the critical Floquet multiplier approaches `-1`, producing susceptibility peaks around odd half-harmonics of the base drive. Near Hopf, a complex-conjugate Floquet pair creates sideband-like resonances. Near saddle-node/pitchfork classes, critical response appears around integer/base harmonics according to the symmetry of the orbit and forcing.

All numbered equations that text extraction can identify are listed in `equations_index.md`. Because this is an older typeset PDF, some glyphs are imperfect in machine extraction; the preserved original-layout text is the source of record and equation labels/page numbers allow an agent to inspect the PDF when exact typography matters.

## Duffing validation

The paper uses a driven Duffing oscillator tuned close to period doubling. A very small periodic modulation is added at/near an odd half-harmonic. Analog simulations measure the output power spectrum while detuning the small signal and moving the control parameter toward the bifurcation. The predicted resonance enhancement and narrowing agree with the analog data.

To reproduce digitally: (1) use the exact Duffing constants printed with the validation figures; (2) locate the period-doubling threshold of the unperturbed drive; (3) choose a sequence of distances `epsilon` from threshold; (4) inject the small modulation and sweep its detuning `Delta`; (5) FFT the steady state; (6) fit peak height/width to the formulas of Section III.

## Josephson-paramp connection

Section IV maps the driven Josephson junction to the same bifurcation picture. The biased three-photon mode is associated with the half-pump/period-doubling susceptibility, while the unbiased four-photon mode is associated with a saddle-node or phase instability. The discussion also explains why symmetry matters for the unbiased three-photon case. The paper's claim is not that every paramp must literally cross into chaos; it is that operating close to the relevant loss of stability produces a large linear susceptibility to the correct small-signal frequency.

## Virtual Hopf sequence

The paper further discusses a "virtual Hopf" sequence: Floquet multipliers can remain close to the unit circle while their phase changes with the control parameter. This lets the narrow high-gain response tune continuously in center frequency. Eqs. (4.7)-(4.9) connect Floquet multipliers to the real and imaginary parts of the exponents.

## Reproduction limits

The divergent gain in the formulas is a *linear-response asymptote*. Finite input amplitude, nonlinear saturation and the actual post-bifurcation dynamics regularize it. Reproduce the small-signal regime first; do not compare the formula directly with a large-signal saturated simulation.
