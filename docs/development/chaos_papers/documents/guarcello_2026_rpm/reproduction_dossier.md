# Reproduction dossier

## Scientific question and contribution

This paper extends the authors' time-domain JTWPA studies to non-sinusoidal CPRs *and* resonant phase matching (RPM). It compares a conventional sinusoidal junction, a CPR with positive second harmonic, and one with negative second harmonic. The full nonlinear dynamics reveal different chaos thresholds and spectral clutter. The second part adds periodically coupled resonators and verifies in LTspice that RPM raises gain in all three CPR cases.

## Base JTWPA

The equivalent line uses 990 repeated cells. Table I gives the numerical values used in the simulations: `CJ = 200 fF`, `RJ = 20 kOhm`, `Ic = 2 uA`, geometric/parallel inductance `Lp = 120 pH`, shunt capacitance `C0 = 24 fF`, and `Z0 = 50 Ohm`. The paper works without an externally applied flux; the drive is electrical.

The junction is represented by a parallel Josephson element, shunt resistor, and junction capacitor. Eq. (1) decomposes total current, Eq. (2) is the Josephson voltage-phase relation. Eqs. (3)-(5) define three CPR families: sinusoidal; a transparency-dependent skewed CPR; and a first-plus-second-harmonic CPR. In the latter two, a scale factor `alpha` is selected so `max_phi IJ(phi) = Ic`.

## Time-domain methodology

For the large 990-cell study the paper uses a FORTRAN implementation of an implicit finite-difference/tridiagonal Josephson-line algorithm. The dimensionless integration step is `0.01` and total time is `20000`; the full CPR is evaluated directly rather than Taylor-expanded. Fourier spectra are post-processed in Mathematica. Pump is `7 GHz`, swept approximately from `-70` to `-45 dBm`; the signal is `-100 dBm` and signal frequencies in the broader study lie between about 4 and 10 GHz.

Figure 2 compares three CPRs (`Jc2/Jc1 = 0`, `+0.6`, `-0.6`). The top panels show gain versus pump power and the lower panels show output Fourier content. Yellow/regular and gray/chaotic regions distinguish useful amplification from unstable dynamics. The negative second harmonic transfers pump energy more effectively and reaches useful gain at lower pump power, but it develops a richer spectrum and chaos earlier.

## Resonant phase matching implementation

The RPM network is a parallel LC resonator connected to the line through a coupling capacitance and repeated every ten line cells. The paper's implementation uses 99 sections of 10 cells plus RPM loading. Values shown with the RPM schematic are `Cc = C0 = 24 fF`, `Cr = 3 pF`, `Lr = 170 pH`, with a resonance designed around 7 GHz.

The RPM reproduction is done in LTspice. The non-standard Josephson CPR is implemented using a custom/controlled current source so that the simulator evaluates the desired supercurrent law. Sweep pump frequency/power around the resonator-assisted phase-matching point; the example operating points selected in the paper are approximately:

- sinusoidal: `fp = 6.95 GHz`, `Pp = -55 dBm`;
- `Jc2/Jc1 = +0.6`: `fp = 6.95 GHz`, `Pp = -58 dBm`;
- `Jc2/Jc1 = -0.6`: `fp = 7.05 GHz`, `Pp = -64 dBm`.

The reported RPM gain increases are roughly 8->11 dB, 7->12 dB, and 12->15 dB for the three cases respectively, while the non-sinusoidal cases—especially negative second harmonic—produce more intermodulation lines.

## Figure interpretation

Figure 1 is the equivalent JTWPA circuit and CPR context. Figure 2 is the stability/gain/spectral comparison for the three CPR shapes. Figure 3 introduces the RPM-loaded cell/line and parameter values. Figure 4 compares gain spectra with RPM and output spectral content at the selected pump operating points. The figure/table index preserves all captions and local discussion.

## Reproduction caveat

The model is classical and deterministic. The broad spectral background described in relation to chaotic dynamics is not a calculation of quantum added noise. A reproduction should therefore report spectral complexity/chaos separately from quantum-noise metrics.
