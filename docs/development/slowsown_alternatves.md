Yes. At **75k elements**, I would start treating the current H2 transient solver as a **proof-of-physics implementation**, not automatically as the production large-scale transient backend.

The encouraging part is that the literature does **not** say “trapezoidal integration cannot scale.” JoSIM itself uses modified nodal analysis plus trapezoidal integration and was specifically designed for large superconducting circuits. ([colab.ws][1]) The scaling problem is much more likely to be **what linear/nonlinear system you solve at every timestep**.

Your jump is roughly (30\times) in circuit size. If the current 6136-variable/2500-element ratio stays similar, we're talking about order (10^5)-(2\times10^5) dynamical unknowns. At that point, repeated generic sparse LU factorizations inside every implicit timestep become the thing I would worry about—not whether the scheme is trapezoidal versus BDF.

## The main alternatives

| Approach                                               |               Scales to ~75k elements? | Handles actual post-fold dynamics? | How I rate it for us                            |
| ------------------------------------------------------ | -------------------------------------: | ---------------------------------: | ----------------------------------------------- |
| **PETSc TS + Newton–Krylov**                           |                    Excellent potential |                                Yes | **Best scalable replacement for H2 backend**    |
| **SUNDIALS IDA + Krylov**                              |                    Excellent potential |                                Yes | **Very strong alternative**                     |
| **JoSIM / WRspice / PSCAN2**                           | Proven superconducting transient tools |                                Yes | Excellent external benchmark / possibly backend |
| **JSICsim**                                            |                Claims up to (10^6) JJs |                                Yes | Very interesting if obtainable/usable           |
| **Xyce + custom JJ model**                             |    Designed for huge parallel circuits |                                Yes | Powerful, but significant integration work      |
| **Shooting Newton / PSS**                              |                            Potentially |               Periodic states only | Strong alternative to HB for high nonlinearity  |
| **Circuit-envelope transient**                         |             Potentially extremely good |    Slow dynamics around RF carrier | Very interesting future route                   |
| **HB + Floquet + sparse transient only at boundaries** |                  Probably best overall |                        Yes, hybrid | **What I would ultimately aim for**             |

### PETSc TS is probably the most relevant large-scale alternative

PETSc's `TS` framework is explicitly designed for scalable implicit ODE/DAE integration and supports BDF, Crank–Nicolson/theta methods, implicit Runge–Kutta families, adaptive timesteps, and arbitrary implicit residuals

[
F(t,u,\dot u)=0.
]

The nonlinear step is handled by PETSc's SNES machinery and the linear solves by KSP/preconditioners, so it is built around exactly the large sparse Newton–Krylov situation we have. ([petsc.org][2])

That means we could retain essentially our physical equations and replace

[
\boxed{\text{SciPy sparse factorization at every Newton step}}
]

with

[
\boxed{\text{matrix-free JVP + GMRES + circuit-specific preconditioner}}.
]

And *that* is the architectural change that could make (75\mathrm{k}) elements plausible.

SUNDIALS IDA is the other obvious option. It directly targets DAEs, uses variable-order BDF, supports GMRES/FGMRES/BiCGStab and user preconditioners, and its documentation explicitly says Krylov methods are generally preferable—and often the only feasible choice—for very large DAEs. ([sundials.readthedocs.io][3])

Between the two, I lean **PETSc for us** because the linear-algebra/preconditioning infrastructure is more central and flexible.

---

# Why our circuit is actually favorable for Newton–Krylov time stepping

At an implicit timestep, the Newton matrix has essentially the structure

[
J_{\mathrm{step}}
=================

a(\Delta t)C
+b(\Delta t)G
+K
+
B_\phi
\operatorname{diag}
\left[
\frac{I_c}{\phi_0}
\cos\phi
\right]
B_\phi^T,
]

with details depending on the time discretization.

That is quite nice:

* (C,G,K,B_\phi) have fixed sparsity;
* only the Josephson differential stiffness changes with state;
* each JJ contributes locally;
* the topology is predominantly spatial/local;
* consecutive timesteps have very similar Jacobians.

So I would **not refactorize the exact Jacobian from scratch at every Newton iteration** at 75k elements.

I'd use something like

[
P
=

aC+bG+K+
B_\phi D_{\rm approx}B_\phi^T
]

as a reusable preconditioner, updating it only occasionally.

The exact nonlinear Jacobian stays in the JVP.

That's philosophically almost identical to what you've already done successfully in HB:

[
\boxed{\text{exact matrix-free nonlinear operator + approximate structured inverse}}.
]

We'd be reusing a solver idea we already know works on this topology.

---

# There are dedicated superconducting simulators that already attack this problem

This is worth taking seriously rather than assuming we must implement everything.

A 2024 comparison specifically evaluates **JoSIM, JSIM, WRspice, PSCAN2 and JSICsim** for superconducting circuit transient simulation, including accuracy and speed. ([ScienceDirect][4])

JoSIM is particularly relevant because it uses

[
\boxed{\text{MNA + trapezoidal integration}}
]

just like the broad direction of H2. It is modern C++, dedicated to superconducting circuits, and its original paper explicitly emphasizes large simulations that were impractical in JSIM. ([colab.ws][1])

WRspice has native Josephson junction support, including RSJ-style models and a more microscopic tunnel-junction model. ([wrcad.com][5]) Importantly for us, **WRspice and PSCAN2 have already been used specifically for JTWPA transient simulation** in a recent comparative JTWPA study. ([arXiv][6])

And JSICsim is especially eye-catching: its publication reports parallel circuit simulation, speedups over PSCAN2/WRspice/JoSIM in its benchmarks, and support up to **one million junctions**. I would treat that as a published claim rather than assume we'll reproduce it on our topology, but it makes 75k JJs clearly not an absurd scale for specialized transient software. ([ResearchGate][7])

So I would absolutely benchmark at least **JoSIM or WRspice** against our H2 circuit at some point.

Not necessarily replace our solver with them—our exact circuit model and product integration matter—but they can tell us whether our transient scaling is fundamentally poor or merely immature.

---

# Xyce is the nuclear option for large-scale circuit transient

Xyce is Sandia's parallel SPICE-compatible simulator and was designed specifically for extremely large circuit simulations, using a DAE formulation and MPI/Trilinos infrastructure. ([Xyce][8])

The catch is Josephson support.

I couldn't find a native built-in JJ device comparable to WRspice. Xyce does support adding custom Verilog-A/C++ device models via ADMS, but its own documentation still characterizes that route as somewhat developmental and notes that sophisticated models may require manual work. ([Xyce][9])

So:

[
\boxed{\text{Xyce could probably scale extremely well}}
]

but

[
\boxed{\text{getting our exact JJ/KI model into it is a real engineering project}.}
]

I would not choose it first for the thesis unless PETSc/IDA fails.

---

# There is also an alternative to long transient integration: Shooting Newton

This is a more interesting change to the **whole stack**.

Instead of solving a periodic orbit with Fourier coefficients, define the one-period flow map

[
\Phi_T(x_0;\mu)
]

and solve

[
\boxed{
F_{\rm shoot}(x_0,\mu)
======================

\Phi_T(x_0;\mu)-x_0
=0.
}
]

That's **shooting Newton / periodic steady state (PSS)**.

It is not obscure: Cadence SpectreRF has both HB and Shooting Newton as its two production periodic-steady-state engines. Cadence describes them as complementary, with HB strong for RF/distributed systems and Shooting Newton often attractive for strongly nonlinear time-domain behavior. ([community.cadence.com][10])

The dimensional tradeoff is interesting:

HB roughly works in a space of

[
N_{\rm state}\times N_{\rm harmonics},
]

while shooting's nonlinear unknown is only roughly

[
N_{\rm state}.
]

But every residual evaluation requires integrating an entire period.

For 75k elements that could be attractive **if the time integrator itself scales**, especially once the waveform becomes strongly nonlinear and HB needs many harmonics.

And you can make shooting matrix-free:

[
Jv
==

D\Phi_T(x_0)v-v
]

via tangent integration or finite-difference flow-map actions.

BifurcationKit already implements matrix-free shooting and continuation for large systems, including periodic-orbit continuation and bifurcation detection. ([BifurcationKit][11])

So an alternative whole architecture would be

[
\boxed{
\text{scalable transient flow map}
+
\text{Newton–Krylov shooting}
+
\text{PALC}.
}
]

That would entirely replace the HB branch solver for the pump.

I **wouldn't do that now**, because your HB solver is already very fast and validated. But it's the most credible alternative if high-drive HB becomes increasingly awkward.

---

# Circuit envelope is potentially the most interesting speed trick

There is another RF-industry approach that fits our problem unusually well.

Rather than resolve every 7.9-GHz carrier cycle during a 40- or 80-period amplitude ramp, represent

[
x(t,T)
\approx
\sum_k X_k(T)e^{ik\omega_pt},
]

where (T) is a **slow time** and the Fourier coefficients evolve.

You then integrate the **envelopes**

[
X_k(T)
]

instead of resolving every RF oscillation.

This is usually called **circuit-envelope / envelope-following analysis**. Commercial RF simulators use it, and mathematically it sits between HB and full transient: harmonic balance resolves the fast carrier while transient evolution describes the slowly varying amplitudes. ([cadence.com][12])

For our use case:

[
\text{slow pump ramp}
+
\text{GHz carrier}
]

is almost exactly the scenario where envelope methods can save enormous amounts of work.

The problem is that when you hit

* period doubling,
* new incommensurate tones,
* phase running,
* chaos,

the assumed carrier basis may stop being adequate.

So I see envelope simulation as an excellent **fast branch-transfer method while the state remains near synchronous**, with full transient as fallback when the spectrum changes qualitatively.

This is perhaps the most interesting long-term research option if runtime becomes the dominant concern.

---

# But I think the best final stack is neither full transient nor shooting everywhere

For your actual map, I would aim for:

[
\boxed{
\text{HB continuation}
+
\text{periodic-state stability}
+
\text{sparse transient only at physical transitions}.
}
]

Specifically:

[
\begin{aligned}
\text{HB} &\rightarrow \text{periodic operating state}\
\text{Floquet/Hill} &\rightarrow \text{is it dynamically stable?}\
\text{transient} &\rightarrow \text{where does it go when stability/existence is lost?}
\end{aligned}
]

There is an established literature on extracting Floquet stability directly from an HB representation using Hill's method, including work specifically aimed at nonlinear circuit simulation. ([ScienceDirect][13])

The important implication for scalability is:

**we would not transient-integrate every pump point.**

At most points:

[
\text{HB solution}
\rightarrow
\text{stability check}.
]

Only around a boundary:

[
\text{transient branch transfer}.
]

That is much closer to the computational architecture I'd want for a 75k-element industrialized solver.

And the JTWPA literature strongly supports avoiding full-transient simulation for ordinary map calculation: recent work comparing JTWPA tools uses WRspice/PSCAN2 in time domain but reports the frequency-domain approach as dramatically faster for ordinary gain calculations; one follow-up reports roughly four orders of magnitude advantage in the example considered. ([arXiv][6])

So full transient should be the exception, not the map engine.

---

# What I would do practically

I would **let H3 finish with the validated current solver**. At 2500 elements it is now a trustworthy physics oracle, and changing backend mid-H3 would just destabilize the investigation.

In parallel, though, I would open a very small **scalability spike** before we invest further in custom transient infrastructure:

1. Benchmark one fixed H2 constant-drive simulation at approximately 2.5k, 10k, maybe 25k elements using the current solver. Measure timestep time, Newton time, factorization time and RAM.
2. Prototype the same implicit DAE residual in **PETSc TS or SUNDIALS IDA**, initially without exotic optimizations.
3. Then replace direct Newton linear solves with **GMRES + a reused linear-network preconditioner** and measure scaling.
4. Export one representative circuit to **JoSIM or WRspice** and measure the same transient as an external reference.

That will tell us very quickly whether:

[
\boxed{\text{our equations are expensive}}
]

or merely

[
\boxed{\text{our current sparse-direct implementation is expensive}.}
]

My expectation is strongly the latter.

### If I had to choose today for the 75k target

I would plan around:

[
\boxed{
\underbrace{\text{your current HB + PALC}}*{\text{keep}}
+
\underbrace{\text{HB/Floquet stability}}*{\text{add later}}
+
\underbrace{\text{PETSc/IDA JFNK transient}}_{\text{replace current large-scale TD backend}}
}
]

with JoSIM/WRspice as external validation.

I would **not** plan around scaling the current SciPy sparse-direct trapezoidal implementation unchanged by (30\times), and I would also **not throw away trapezoidal integration itself**. JoSIM is a useful reminder that trapezoidal stepping can be perfectly viable at large scale when the circuit stamping and sparse linear algebra underneath it are designed for that scale. ([colab.ws][1])

The next thing I'd investigate technically is therefore not “which integrator has a higher order?” but:

[
\boxed{
\text{Can we make one transient Newton step matrix-free and preconditioned
using the same circuit structure that made HB scalable?}
}
]

That is probably the difference between a 6k-variable research prototype and a (10^5)-variable production transient engine.

[1]: https://colab.ws/articles/10.1109%2Ftasc.2019.2897312?utm_source=chatgpt.com "JoSIM—Superconductor SPICE Simulator | CoLab"
[2]: https://petsc.org/release/manualpages/TS/?utm_source=chatgpt.com "Time Stepping ODE and DAE Solvers (TS) — PETSc 3.25.3 documentation"
[3]: https://sundials.readthedocs.io/en/develop/ida/Introduction_link.html?utm_source=chatgpt.com "5.1. Introduction — Documentation for SUNDIALS documentation"
[4]: https://www.sciencedirect.com/science/article/abs/pii/S0921453424001370?utm_source=chatgpt.com "Performance evaluation of superconductor integrated circuit simulators - ScienceDirect"
[5]: https://www.wrcad.com/manual/wrsmanual/node146.html?utm_source=chatgpt.com "Josephson Junction Model"
[6]: https://arxiv.org/abs/2402.12037?utm_source=chatgpt.com "Numerical simulations of Josephson Traveling Wave Parametric Amplifiers (JTWPAs): comparative study of open-source tools"
[7]: https://www.researchgate.net/publication/365115288_JSICsim_-An_Analog_Simulator_for_Superconductor_Integrated_Circuit?utm_source=chatgpt.com "JSICsim -An Analog Simulator for Superconductor Integrated Circuit | Request PDF"
[8]: https://xyce.sandia.gov/?utm_source=chatgpt.com "Xyce – Sandia National Laboratories"
[9]: https://xyce.sandia.gov/documentation-tutorials/xyce-adms-users-guide/?utm_source=chatgpt.com "Xyce/ADMS Users Guide – Xyce"
[10]: https://community.cadence.com/cadence_blogs_8/b/cic/posts/small-signal-analyses-using-hb-and-shooting-newton-methods-in-spectrerf-option?utm_source=chatgpt.com "Small-Signal Analyses Using HB and Shooting Newton Methods in SpectreRF Option - Analog/Custom Design - Cadence Blogs - Cadence Community"
[11]: https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/?utm_source=chatgpt.com "🏠 Home · Bifurcation Analysis in Julia"
[12]: https://www.cadence.com/en_US/home/tools/custom-ic-analog-rf-design/circuit-simulation/spectre-rf-option.html?utm_source=chatgpt.com "Spectre RF Option | Cadence"
[13]: https://www.sciencedirect.com/science/article/pii/S1631072110001142?utm_source=chatgpt.com "A harmonic-based method for computing the stability of periodic solutions of dynamical systems - ScienceDirect"
