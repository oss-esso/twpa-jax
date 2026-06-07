# Meeting Note — Direct Linear Backend Phase

I completed a concrete cleanup and acceleration phase in the Harmonia/JosephsonCircuits pipeline.

Some linear CircuitIR cases were using the full `hbsolve` wrapper and then reading `linearized.S`, even though the pump current was zero. I added opt-in direct `hblinsolve` backends for JTL, RF-JTL, and ETHZ-JTL linear workflows.

For each family, I first compared the old path against direct `hblinsolve`. All compared S-parameters matched exactly, with `max_abs_diff = 0.0`.

Largest exact old-vs-direct comparisons:

- JTL 3000 cells: 6004 elements, 9004 JosephsonCircuits tuples.
- RF-JTL 2393 cells: 9576 elements, 11969 tuples.
- ETHZ-JTL 2048 cells: 6653 elements, 8700 tuples.

I also ran direct-only extreme linear-response checks:

- JTL 30000 cells: 60004 elements, 90004 tuples, about 40 s.
- RF-JTL 5000 cells: 20004 elements, 25004 tuples, about 12 s.
- ETHZ-JTL 10000 cells: 32497 elements, 42496 tuples, about 6.5 s.

RF-JTL at 10000 cells produced non-finite S-parameters, which we recorded as a real numerical boundary rather than hiding it.

The conclusion is not “full nonlinear HB is now faster.” The correct conclusion is that the linear-response path is now explicit, equivalence-tested, scalable, and status-tracked. This gives a cleaner foundation for topology validation, pump-off S-parameter campaigns, dataset sanity checks, and later nonlinear HB/calibration work.
