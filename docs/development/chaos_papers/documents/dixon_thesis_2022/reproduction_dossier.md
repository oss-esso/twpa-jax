# Reproduction dossier

## Scope of the thesis

This thesis is the expanded source behind the 2019 many-mode JTWPA paper. It begins with nonlinear wave mixing, transmission-line theory, Josephson junction/rf-SQUID physics and JPA/JTWPA context; reports experiments on a nonlinear resonator and a travelling-wave device; validates WRspice building blocks; constructs a complete JTWPA simulation; then extends the three-tone coupled-mode theory to many modes and develops quasi-phase-matching/poling as a way to suppress unwanted harmonics.

The corpus preserves the entire 180-page source as page-separated JSONL and original-layout text. For an LLM agent, use the structured dossier to find the relevant chapter, then retrieve the exact page text/equation by page and label.

## Chapter-by-chapter reproduction map

### Chapter 1 — prerequisite theory and literature

The nonlinear optics analogy is developed from polarization `P = epsilon0 chi E` to the nonlinear expansion `P^(1)+P^(2)+...`, the driven wave equation, two/multi-tone fields, and explicit second-order mixing terms. The derivation shows how sum-frequency, difference-frequency and second-harmonic processes arise and introduces phase mismatch. Transmission-line equations, Josephson relations and rf-SQUID behavior then provide the circuit counterpart. Reproduce these derivations symbolically before coding the later CME model; the numbered equations begin at Eq. (1.1) and are indexed separately.

### Chapter 2 — rf-SQUID JTWPA theory

The thesis derives the wave equation for a superconducting transmission line loaded by junction/rf-SQUID nonlinearities, develops the flux-bias control of centrosymmetry, and reduces it to a three-tone three-wave-mixing model. It then studies practical gain bandwidth, pump/signal dependence, phase matching and compression. Figures in this chapter are a useful analytic baseline before the full circuit simulation.

### Chapter 3 — experiments

The first device is a nonlinear resonant structure. The workflow covers cryogenic measurement configuration, resonator/Fano fitting, power-dependent bifurcation, flux dependence, then wave-mixing observations. A higher-order tone is traced to a three-wave-mixing condition.

The travelling-wave experiment then uses one- and two-tone drives. With a single strong tone the measured spectrum exhibits cascaded harmonics. With two input tones it exhibits sums, differences, and up-converted products. These observations motivate the many-mode model: the physical device does not remain inside a pump/signal/idler basis.

For reproduction, the `figures_tables_index.md` entries for Chapter 3 should be treated as the measurement checklist: reproduce the calibration/characterization plots first, then the nonlinear spectra as power/flux/frequency are swept.

### Chapter 4 — circuit simulation

WRspice is validated from the bottom up. Linear LC and capacitively shunted transmission lines are checked first. The Josephson RCSJ element is checked through DC and AC behavior including Shapiro steps. The full rf-SQUID JTWPA is then assembled and transient currents are recorded along its length. FFT processing (including the windowing procedure described in the chapter) turns those time traces into position-frequency maps.

The key comparison reproduces the three-tone theory and the unrestricted WRspice calculation at the same pump/signal operating point. Pump second-harmonic generation and additional tones explain why the simple analytic model overestimates signal gain.

### Chapter 5 — resolving the many-mode discrepancy

This chapter systematically tests the approximations behind the conventional CME. First it examines junction phase excursions and shows that the initial high-pump case leaves the small-phase regime. Pump and signal are reduced and transient/ramp issues are checked so the circuit simulation lies inside the analytic assumptions.

The CME is then generalized by adding pump harmonics and all pump-mediated mixing tones in an ordered sequence. Mode/mixing matrices make the bookkeeping explicit. Numerical integration of these enlarged equations converges toward WRspice as more modes are included.

Finally, the thesis introduces quasi-phase matching/poling. By periodically changing the sign/effective nonlinear interaction, unwanted second-harmonic accumulation can be suppressed while the desired three-wave process remains coherent. The final figures compare poling designs, harmonic suppression, signal gain and WRspice/CME agreement; the goal is a flatter high-gain response (around the 20 dB design scale discussed in the chapter) while controlling parasitic tones.

### Appendices — direct implementation material

Appendix A contains the detailed wave-mixing/JTWPA derivations omitted from the main narrative. Appendix B contains concrete simulation material, including RCSJ MATLAB code and WRspice netlist content. Appendix C contains the full extended CME equations and MATLAB implementation for CME-5/quasi-phase-matching. For reproduction work, these appendices should be treated as executable specifications, with the main chapters providing interpretation and validation targets.

## Reproduction sequence recommended by the thesis itself

1. Reproduce linear transmission-line dispersion/impedance.
2. Validate one Josephson junction: DC response, AC response and Shapiro steps.
3. Validate an rf-SQUID cell and its flux bias.
4. Build the full JTWPA and reproduce the unrestricted transient FFT map.
5. Reproduce the conventional three-tone CME at exactly the same operating point.
6. Check phase excursion and other approximation conditions before comparing.
7. Add higher modes in the ordered CME hierarchy and verify convergence to WRspice.
8. Reproduce the poling/quasi-phase-matching section and verify both harmonic suppression and restored desired gain.

## What counts as a successful reproduction

Matching only a final gain curve is insufficient. The thesis establishes a chain of evidence: linear transmission -> single-junction nonlinear benchmarks -> full spectrum along the line -> phase/small-signal validity -> many-mode energy redistribution -> gain convergence -> quasi-phase-matching mitigation. An implementation that passes those intermediate plots is much more likely to be physically equivalent to the thesis model.
