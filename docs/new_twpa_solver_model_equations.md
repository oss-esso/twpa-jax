# New TWPA Solver Model Equations

The simulator model is the reduced node-flux system

```text
C phi_ddot(t) + G phi_dot(t) + i_lin(phi(t)) + i_nl(phi(t)) = i_src(t).
```

`CircuitModel` stores:

- `capacitance_f`: global C matrix.
- `conductance_s`: global G matrix, including port loading.
- `linear_stiffness_h_inv`: linear inductive K matrix.
- `josephson_incidence`: D matrix for Josephson branches.
- `josephson`: explicit branch law object.
- `ports`: named port nodes and reference impedances.
- `pump_nodes`: coupler injection nodes.

## Josephson Law

The implemented branch current is

```text
i_J(psi) = Ic sin(psi / varphi0)
d i_J / d psi = Ic / varphi0 cos(psi / varphi0)
```

with `varphi0 = Phi0 / (2 pi)`. Node current contribution is

```text
i_JTL(phi) = D [ Ic .* sin(D^T phi / varphi0) ].
```

RF-SQUID and kinetic-inductance classes expose the same `current()` and
`derivative()` interface for future topology blocks.

## Linear S-Parameters

For small signals at angular frequency `omega`, the voltage admittance is

```text
Y_node(omega) = G + j omega C + K / (j omega).
```

Internal nodes are Schur-eliminated to get `Y_port`. The S-matrix uses real
reference impedances:

```text
S = (I - Z0 Y_port) (I + Z0 Y_port)^-1.
```

## Pump Harmonic Balance

`PumpAFTResidual` uses real coefficients:

```text
phi(t) = sum_h a_h cos(h omega_p t) + b_h sin(h omega_p t).
```

It evaluates the nonlinear branch current in time, projects the residual back
onto the cos/sin basis, and returns a scaled real residual vector. Harmonics,
time samples, source amplitude, source phase, and residual scale are explicit
config fields.

## Pumped Conversion Matrix

After pump HB, the linearized perturbation equation is

```text
C delta_phi_ddot + G delta_phi_dot + K(t) delta_phi = delta_i_src,
K(t) = K_lin + d i_nl / d phi | phi_p(t).
```

`build_conversion_sparameters` Fourier-expands `K(t)` and builds the sideband
grid

```text
omega_m = omega_s + m omega_p.
```

It forms a sideband admittance matrix, Schur-eliminates internal nodes, converts
to S-parameters, and carries the pump convergence status into the result.
