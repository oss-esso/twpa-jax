# Showcase Designs for Direct Linear Backend Speedups

## Purpose

This document defines the design cases to use when demonstrating the direct
`hblinsolve` backend work.

The direct backend phase applies to linear response workflows. It should be used
for pump-off S-parameters, topology checks, fast linear campaigns, and dataset
sanity checks. It should not be presented as a full nonlinear gain/HB speedup.

## Tier 1 — Proven current templates

These are already integrated and tested:

| Case | Backend flag | Status |
|---|---|---|
| JTL linear | `jtl_linear_backend = "hblinsolve_direct"` | integrated |
| RF-JTL linear | `rf_jtl_linear_backend = "hblinsolve_direct"` | integrated |
| ETHZ-JTL linear | `ethz_jtl_linear_backend = "hblinsolve_direct"` | integrated |

These are the correctness anchor cases.

## Tier 2 — Scaled internal templates

Use the same templates, but sweep size parameters to show scaling.

Recommended first scale sweep:

| Template | Small | Medium | Large |
|---|---:|---:|---:|
| JTL | current default | 64 cells | 256+ cells |
| RF-JTL | current default | 64 cells | 256+ cells |
| ETHZ-JTL | current default | scaled sections | largest safe local run |

Acceptance criteria:

- `PASS` status.
- finite S-parameters.
- backend telemetry records direct `hblinsolve`.
- no fallback or surrogate path.
- compare old `hbsolve` path against direct `hblinsolve` on at least small/medium cases.

## Tier 3 — Real literature-scale targets

These are not necessarily immediate full simulations on the laptop. They are reference
targets for explaining why the infrastructure matters.

### MIT/Keysight-style JTWPA

Reference scale: around 1648 unit cells. This is useful as a realistic large JTWPA
target where linear response and topology validation are recurring campaign tasks.

### Uniform/Floquet JTWPA

Reference scales: around 2047 junctions for a uniform design and around 3998 junctions
for a Floquet design. These are good production-scale targets for future nonlinear
HB and X-parameter workflows.

### rf-IPM / rf-SQUID Harmonia design

This is the natural Harmonia-native showcase because it stresses automated topology
generation, disorder/defect handling, and complex microwave structures.

## Reporting rule

Report the direct backend speedup as:

> exact-equivalent linear response acceleration

Do not report it as:

> full nonlinear HB gain acceleration

unless a separate nonlinear benchmark proves that claim.
