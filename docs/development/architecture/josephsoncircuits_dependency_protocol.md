# JosephsonCircuits.jl Dependency Protocol

JosephsonCircuits.jl is the authoritative harmonic-balance solver backend used through Harmonia.jl.

## Modes

- `upstream_release`: pinned/default baseline.
- `local_develop`: local editable checkout for thesis experiments.
- `thesis_fork`: documented fork branch for solver diagnostics or extensions.

## Rule

Do not edit `.julia/packages/JosephsonCircuits/...`.

All solver changes must happen in:

D:\Projects\Thesis\JosephsonCircuits.jl

## Remotes
upstream: https://github.com/kpobrien/JosephsonCircuits.jl.git
origin: thesis fork under the user's GitHub account
Safe workflow
Baseline upstream behavior.
Add Harmonia/Python golden tests before changing solver behavior.
Use Pkg.develop(path="../JosephsonCircuits.jl") only on a dedicated Harmonia.jl branch.
Keep status/failure metadata in Harmonia/Python unless JosephsonCircuits itself must expose missing internals.
Every solver patch must record:
upstream commit;
fork branch;
reason;
physics impact;
before/after golden outputs;
tests added.


Then:

```powershell
cd $PY

git add docs/architecture/josephsoncircuits_dependency_protocol.md
git commit -m "Document JosephsonCircuits dependency protocol"


## Current local solver wiring

Harmonia.jl successfully imports JosephsonCircuits.jl from the sibling editable checkout:

```text
D:\Projects\Thesis\JosephsonCircuits.jl\src\JosephsonCircuits.jl

No tracked Harmonia.jl files changed during the local wiring check, so there is no Harmonia commit for this step.

Interpretation: the current Julia environment already resolves JosephsonCircuits.jl to the local compatible checkout, or Pkg.develop(path="../JosephsonCircuits.jl") was a no-op.

Operational rule: before solver modifications, always verify with:

using JosephsonCircuits
println(pathof(JosephsonCircuits))

Do not edit JosephsonCircuits.jl before golden Harmonia/JosephsonCircuits output tests exist.