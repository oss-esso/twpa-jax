# Reproduction dossier

## Scientific question and contribution

Levinsen connects three topics in the ac-driven Stewart-McCumber Josephson model: subharmonic generation, Feigenbaum period-doubling/chaos, and the operation/noise of Josephson parametric amplifiers. The paper uses the Duffing approximation for analytic insight and an analog computer for the complete nonlinear junction model. A major conceptual point is that odd subharmonics are naturally supported by the sine nonlinearity and can themselves seed bifurcation trees; spontaneous symmetry breaking resolves an apparent contradiction between older three-photon-paramp theory and period-doubling results at zero nominal DC current.

## Base junction model

Figure 1 is the RCSJ/Stewart-McCumber equivalent circuit driven by `I = Idc + ID sin(omega_D t)`. Equation (1) gives the normalized driven junction equation, with currents normalized by `Ic`, voltage by `R Ic`, and time/frequency by the definitions printed directly after the equation. The paper then uses a cubic/Duffing expansion to obtain analytic conditions for subharmonic generation and stability; every numbered equation is collected in `equations_index.md`.

## Analytical logic to reproduce

1. Expand the Josephson `sin(phi)` nonlinearity to a Duffing-type equation near the relevant operating regime.
2. Assume a subharmonic trial solution and derive amplitude/phase consistency equations.
3. Examine the symmetry constraints at zero DC bias. The key observation is that a symmetric equation can possess stable solutions that break the symmetry spontaneously; therefore an averaged quantity can vanish while the oscillation itself acquires a half-frequency component.
4. Use trigonometric multiple-angle identities (including the explicit third-harmonic identity and its generalization later in the paper) to show why odd subharmonics arise naturally from the full sine CPR.
5. Track secondary period doublings of those subharmonic states, connecting them to the Feigenbaum scenario.

## Analog-computer methodology

The full driven Stewart-McCumber model is solved on an EMRI-ACS-122D hybrid analog computer. Reproducing it today can be done digitally by integrating the same normalized ODE, but one must preserve the exact parameter values, scan direction, and initial-state continuation used in each figure. The analog spectra in the paper are the observable to match: fundamental, subharmonic lines, then successively halved frequencies and chaotic broadband response.

## Parametric-amplifier model

The later section constructs an externally pumped Josephson parametric amplifier including source and load/tuned-circuit dynamics, and examines both three-photon and four-photon cases. Gain is defined as the squared output-to-input current ratio. Near the half-harmonic instability the analog model gives very large gains (the paper shows values above 30 dB), but after further subharmonic bifurcation/chaos the coherent gain collapses.

The authors also inject white noise and compare signal/noise behavior. Their conclusion is *negative*: although chaotic states exist, the model does not support chaotic noise as the explanation of the experimentally observed gain-dependent noise-temperature rise; applied white noise and signal are amplified together.

## Figure interpretation

Figure 1 defines the junction circuit. The next figures show a period-doubling cascade and its spectral signatures, explicit odd subharmonic generation, and a bifurcation tree starting from an odd subharmonic. The final amplifier plots demonstrate high gain close to instability and the noise-injection test. `figures_tables_index.md` gives all captions and nearby paper discussion.

## Practical modern reproduction

Use double-precision ODE integration with an adaptive high-order integrator or sufficiently fine fixed RK step, but verify against the normalized analog-computer equations rather than changing the model. For each plotted spectrum: integrate many pump periods after transients, sample uniformly, FFT with enough record length to resolve half/quarter/eighth and odd-subharmonic lines, and repeat while continuing the final state along the control-parameter sweep.
