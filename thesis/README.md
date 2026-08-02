# Thesis sources

LaTeX sources for the thesis on the `twpa_solver` harmonic-balance solver.

## Build

```powershell
cd thesis
latexmk -pdf main.tex
```

Produces `main.pdf`. `latexmk -C` cleans. Build artifacts are gitignored;
`main.pdf` is deliberately **not** tracked, since committing it makes every
rebuild a spurious diff.

Requires a standard TeX distribution. Verified with MiKTeX on Windows; all
packages used are in TeX Live's `-full` scheme.

## Layout

| File | Contents |
| --- | --- |
| `main.tex` | Document skeleton and chapter includes |
| `preamble.tex` | Packages, macros, callout environments |
| `bibliography.bib` | Starter reference set |
| `chapters/` | One file per chapter |

### Chapters

| File | Chapter |
| --- | --- |
| `00_notation.tex` | Symbol table mapping notation to code identifiers |
| `01_introduction.tex` | Motivation, contributions, structure |
| `02_twpa_theory.tex` | Josephson element, travelling-wave line, mixing, phase matching, saturation mechanisms, and where analytic theory stops |
| `03_circuit_model.tex` | Node-flux formulation, stamps, ports, loss, builders, profiles and scatter |
| `04_pump_harmonic_balance.tex` | **The algorithm.** All eight nested loops at reimplementation depth |
| `05_numerical_methods.tex` | Schur reduction, factorisation backends, ordering, measured performance |
| `06_small_signal_floquet.tex` | Floquet sidebands, conversion matrix, gain definitions, quantum efficiency |
| `07_saturation.tex` | Tone lattice, multitone residual, compression, conservation diagnostics, stability |
| `08_validation_and_limits.tex` | What is established, what is retracted, what has no external reference |
| `A1_conventions_and_pitfalls.tex` | Every convention and the specific error it guards against |

## Conventions used in the text

Three callout environments carry specific meanings, and the distinction is
load-bearing:

- **`implnote`** — what the code actually does, quoted from the source. Used
  wherever the implementation departs from the idealised statement in the
  surrounding text.
- **`measured`** — a number that came out of a run, not out of an argument.
- **`caveat`** — a limitation of the method or of the present validation state.

Chapter 4 in particular is written so that the mathematical specification and
the running code can be compared line by line. Three genuine divergences are
documented there and summarised in a table at the end of that chapter:

1. GMRES is **left**-preconditioned (SciPy's `M=`), not right;
2. the static DC branch current is **subtracted** from the branch law;
3. `source_time()` ignores `k_src` — a latent defect on a diagnostic-only path.

Every other formula in that chapter was verified against the running code, in
several cases to machine precision.

## Templating

`preamble.tex` is deliberately template-agnostic: no `\documentclass`, no font
selection. When the institutional template arrives, replace `main.tex`'s
`\documentclass` and the "Layout" block at the top of `preamble.tex`; the
chapter files need no changes.
