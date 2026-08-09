Below is the implementation plan I would give directly to the coding agent. It is intentionally **architecture-level rather than file/function-name-specific**: the agent must first discover how the existing solver is structured and then map these requirements onto it. The underlying HB residual, AFT/JVP structure, continuation hierarchy, Schur reduction, and preconditioning assumptions are those in the supplied algorithm description.  The failure classification and already-completed work are taken from the current problem report. 

# Implementation plan: robust fold-aware continuation for production TWPA pump solves

## Objective

Refactor the current pump continuation workflow so that:

[
\boxed{
\text{one physical solution branch is traced once per pump frequency}
}
]

rather than performing a separate target-normalized continuation problem for every desired pump power.

The solver must:

* retain the existing matrix-free HB formulation;
* retain the existing Schur reduction;
* retain the existing Newton–Krylov/JVP machinery;
* retain the existing working weighted pseudo-arclength metric;
* retain the existing refined bordered-solve machinery;
* retain warm starts and direct Newton solves where they are advantageous;
* replace target-relative continuation with continuation in a fixed physical drive coordinate;
* automatically adjust arclength step length;
* identify and refine folds;
* distinguish a target that is genuinely beyond a fold from a numerical failure;
* avoid wasting hundreds of continuation points waiting for a returning branch that may not exist;
* optionally continue beyond a fold for branch exploration;
* eventually support continuation of the fold itself versus pump frequency;
* expose enough diagnostics to determine whether high-drive loss of the synchronous pump solution is physical, harmonic-truncation-induced, or caused by a change of dynamical state.

The current problem is not primarily “make pseudo-arclength stronger.” The current pseudo-arclength implementation appears capable of passing a genuine simple fold. The architectural issue is that the continuation problem is currently posed separately relative to every requested final pump amplitude. 

---

# 0. First task: inspect the codebase and establish invariants

Before changing anything, map the existing implementation into the following conceptual components.

Do not rename or reorganize code merely to match this plan.

Identify where the implementation currently provides:

1. HB residual evaluation
   [
   R(X;\lambda)
   ============

   \mathcal D X+N(X)-\lambda S.
   ]

2. Matrix-free Jacobian action
   [
   v\mapsto Jv.
   ]

3. If available, transpose-Jacobian or adjoint action
   [
   v\mapsto J^\mathsf Tv.
   ]

4. Schur-reduced residual and Jacobian operations.

5. Newton corrector.

6. GMRES or equivalent Krylov solve.

7. Preconditioner construction and reuse.

8. Natural continuation.

9. Pseudo-arclength continuation.

10. Tangent construction.

11. Weighted arclength inner product.

12. Periodic metric rescaling.

13. Bordered linear solve.

14. One-refinement bordered solve.

15. Minimum-eigenvalue / singularity diagnostics.

16. Per-accepted-step telemetry callback.

17. Inter-power warm starts.

18. Inter-frequency warm starts, if any.

19. Fold-event detection.

20. Current map-level recovery logic.

The agent must produce a short internal mapping before modifying behavior.

### Hard invariants

Do not replace the HB physics.

The governing nonlinear system remains

[
R(X;\mu,f_p)
============

\mathcal D(f_p)X+N(X)-S(\mu,f_p).
]

The nonlinear AFT remains

[
X
\rightarrow
x(t)
\rightarrow
\psi(t)
\rightarrow
I_c\sin(\psi/\phi_0)
\rightarrow
N(X).
]

The exact JVP remains derived from the true HB residual, not from the preconditioner.

The preconditioner remains an approximation to the Jacobian, not the mathematical continuation operator.

Do not construct a dense full Jacobian solely to implement pseudo-arclength.

Do not remove the existing simple direct solve at an already-nearby operating point.

Do not remove the current metric fix. The previous unscaled metric made the state component effectively irrelevant and turned pseudo-arclength into natural continuation with bookkeeping; that has already been identified and corrected. 

---

# 1. Replace target-normalized (\lambda) with a global physical drive coordinate

This is the highest-priority architectural change.

At present, conceptually, a requested pump current (I_{\rm target}) defines

[
I(\lambda)=\lambda I_{\rm target},
\qquad
0\leq\lambda\leq1.
]

That means the exact same physical fold (I_f) appears at

[
\lambda_f
=========

\frac{I_f}{I_{\rm target}}.
]

Consequently every requested target power creates a differently scaled representation of the same physical branch.

Stop doing that for fold-aware branch tracing.

Introduce a fixed reference current

[
I_{\rm ref}
]

and define

[
\boxed{
\mu=\frac{I_p}{I_{\rm ref}}.
}
]

Recommended default:

[
I_{\rm ref}=I_{c,\rm nominal}
]

or another device-wide, fixed current already naturally available to the model.

The exact choice is less important than these requirements:

* it must not depend on the current target pump power;
* it must remain fixed throughout a branch;
* it should make (\mu=O(1)) in the physically interesting regime;
* all requested pump powers must be convertible into (\mu_{\rm target}).

Then define a reference source vector corresponding to (I_{\rm ref}):

[
S_{\rm ref}.
]

The continuation residual becomes

[
\boxed{
R(X,\mu;f_p)
============

\mathcal D(f_p)X
+
N(X)
----

\mu S_{\rm ref}(f_p).
}
]

Therefore,

[
R_\mu
=====

# \frac{\partial R}{\partial\mu}

-S_{\rm ref}.
]

This derivative is exact, constant with respect to (X), and extremely cheap.

### Frequency-dependent source networks

If source delivery through the modeled input network depends on frequency, that is fine.

For each fixed pump frequency (f_p), construct the source corresponding to (I_{\rm ref}) using exactly the same physical source convention currently used.

The continuation parameter still scales that frequency-specific reference source:

[
S(\mu,f_p)=\mu S_{\rm ref}(f_p).
]

### Map conversion

Every requested dBm point should be converted first into physical current using the existing source convention:

[
P\rightarrow I_p.
]

Then

[
\mu_{\rm target}
================

\frac{I_p}{I_{\rm ref}}.
]

The continuation solver should no longer know or care that the original request was expressed in dBm.

---

# 2. Separate three concepts that are currently too tightly coupled

From now on distinguish:

### Physical drive coordinate

[
\mu.
]

This says how much pump is physically applied.

### Continuation coordinate

[
s.
]

This parameterizes distance along a solution branch.

### Requested map target

[
\mu_{\rm target}.
]

This is merely a point at which the caller wants a solution.

These must never again be treated as the same thing.

Pseudo-arclength should solve for

[
Y=
\begin{bmatrix}
X\
\mu
\end{bmatrix}
]

as a function of (s).

The branch may have

[
\frac{d\mu}{ds}>0,
]

then

[
\frac{d\mu}{ds}=0
]

at a fold, then

[
\frac{d\mu}{ds}<0.
]

That is valid continuation behavior.

A negative (\dot\mu) must never by itself be classified as failure.

---

# 3. Define one authoritative augmented-state metric

The pseudo-arclength metric must remain explicit.

For state perturbations (a,b), retain a scaled state inner product of the form

[
\langle a,b\rangle_X
====================

\frac{
\operatorname{Re}(a^\dagger b)
}{
x_{\rm scale}^2
}.
]

For the full continuation state

[
Y=(X,\mu),
]

use

[
\boxed{
\langle
(\delta X_1,\delta\mu_1),
(\delta X_2,\delta\mu_2)
\rangle_W
=========

\frac{
\operatorname{Re}(\delta X_1^\dagger\delta X_2)
}{
x_{\rm scale}^2
}
+
w_\mu^2\delta\mu_1\delta\mu_2.
}
]

Because (\mu) is dimensionless and (O(1)), start with

[
w_\mu=1.
]

Do not add another arbitrary drive scale unless evidence shows it is necessary.

The corresponding norm is

[
|Y|_W
=====

\sqrt{\langle Y,Y\rangle_W}.
]

Every quantity related to pseudo-arclength must use the same metric:

* tangent normalization;
* tangent orientation;
* predictor distance;
* corrector hyperplane;
* tangent-angle calculation;
* curvature calculation.

Do not mix Euclidean and weighted inner products inside the continuation algorithm.

---

# 4. Make metric rescaling mathematically continuous

The periodic metric rescaling that fixed the 7.0 GHz failure must remain. 

However, changing (x_{\rm scale}) changes the meaning of one unit of arclength.

Therefore metric rescaling must also transform the tangent and effective step size.

Suppose immediately before rescaling the accepted tangent is

[
t_{\rm old}
===========

(\dot X,\dot\mu)
]

with

[
|t_{\rm old}|*{W*{\rm old}}=1.
]

The current predictor displacement would be

[
\Delta Y
========

ds_{\rm old}t_{\rm old}.
]

After changing the metric, calculate

[
q=
|t_{\rm old}|*{W*{\rm new}}.
]

Then renormalize

[
t_{\rm new}
===========

\frac{t_{\rm old}}{q}
]

and change the arclength step to

[
\boxed{
ds_{\rm new}=ds_{\rm old}q.
}
]

Then

[
ds_{\rm new}t_{\rm new}
=======================

ds_{\rm old}t_{\rm old}.
]

So changing the metric does **not** suddenly change the physical predictor.

This should be tested explicitly.

### Prevent metric chatter

Do not allow (x_{\rm scale}) to jump arbitrarily every few points.

Use one or more of:

* rescale only after a configurable number of accepted steps;
* rescale only if the proposed scale differs materially from the current one;
* limit the scale change per rescaling event;
* smooth the proposed scale with the previous scale.

Store both:

[
x_{\rm scale,old},
\qquad
x_{\rm scale,new}.
]

Metric changes must appear in continuation telemetry.

---

# 5. Define the continuation tangent as a full augmented tangent

At every accepted point compute

[
t=
\begin{bmatrix}
t_X\
t_\mu
\end{bmatrix}
=============

\begin{bmatrix}
dX/ds\
d\mu/ds
\end{bmatrix}.
]

The tangent satisfies

[
Jt_X+R_\mu t_\mu=0.
]

Because

[
R_\mu=-S_{\rm ref},
]

[
\boxed{
Jt_X-S_{\rm ref}t_\mu=0.
}
]

Add an orientation/normalization constraint based on the previous tangent.

A robust bordered tangent system is conceptually

[
\begin{bmatrix}
J&R_\mu\
t_{X,\rm old}^{T}W_X&t_{\mu,\rm old}w_\mu^2
\end{bmatrix}
\begin{bmatrix}
\widetilde t_X\
\widetilde t_\mu
\end{bmatrix}
=============

\begin{bmatrix}
0\
1
\end{bmatrix}.
]

Then normalize:

[
t=
\frac{\widetilde t}
{|\widetilde t|_W}.
]

Use the existing robust bordered solver rather than introducing a naive inversion of (J).

The currently implemented bordered refinement should be reused for this operation. 

---

# 6. Enforce tangent orientation

After obtaining the new normalized tangent, calculate

[
c=
\langle t_{\rm new},t_{\rm old}\rangle_W.
]

If

[
c<0,
]

flip the complete tangent:

[
t_{\rm new}\leftarrow -t_{\rm new}.
]

Never flip only (t_X) or only (t_\mu).

This prevents arbitrary null-vector sign choices from looking like physical direction reversals.

Only after this orientation correction may a sign change in (t_\mu) be interpreted as evidence of a fold.

---

# 7. Predictor

Given an accepted branch point

[
Y_n=
(X_n,\mu_n)
]

and normalized tangent (t_n), use the Euler tangent predictor

[
\boxed{
Y_{\rm pred}
============

Y_n+ds,t_n.
}
]

Therefore

[
X_{\rm pred}=X_n+ds,t_{X,n},
]

[
\mu_{\rm pred}=\mu_n+ds,t_{\mu,n}.
]

The existing secant predictor can remain as:

* startup fallback;
* tangent failure fallback;
* diagnostic comparison.

But once two reliable PALC points exist, the tangent predictor should be authoritative.

---

# 8. Corrector equation

The corrector solves simultaneously:

[
R(X,\mu)=0
]

and

[
g(X,\mu)=0.
]

Use the standard orthogonal pseudo-arclength hyperplane through the predictor:

[
\boxed{
g(X,\mu)
========

\left\langle
t_n,
Y-Y_{\rm pred}
\right\rangle_W
=0.
}
]

Expanded,

[
g
=

\frac{
\operatorname{Re}
[
t_{X,n}^{\dagger}(X-X_{\rm pred})
]
}{
x_{\rm scale}^2
}
+
w_\mu^2
t_{\mu,n}
(\mu-\mu_{\rm pred}).
]

The augmented Newton equation is

[
\boxed{
\begin{bmatrix}
J&R_\mu\
c_X^\mathsf T&c_\mu
\end{bmatrix}
\begin{bmatrix}
\Delta X\
\Delta\mu
\end{bmatrix}
=============

*

\begin{bmatrix}
R\
g
\end{bmatrix}
}
]

where

[
c_X^\mathsf T\Delta X
=====================

\frac{
\operatorname{Re}
(t_X^\dagger\Delta X)
}{
x_{\rm scale}^2
}
]

and

[
c_\mu=w_\mu^2t_\mu.
]

Do not build the augmented matrix densely.

Use the existing bordered linear-system machinery.

---

# 9. Corrector stopping criteria

The corrector must check both equations independently.

Require:

[
r_R
===

\text{normalized HB residual}
<
\varepsilon_R
]

and

[
r_g
===

\frac{|g|}
{\max(ds,\epsilon)}
<
\varepsilon_g.
]

Do not accept a point solely because the HB residual is small.

A point with

[
R\approx0
]

but a large arclength-constraint error may have converged to a different nearby branch location.

Store:

* final HB residual;
* arclength residual;
* Newton iterations;
* total Krylov iterations;
* final damping;
* number of preconditioner rebuilds;
* corrector wall time.

---

# 10. Introduce actual adaptive pseudo-arclength step control

The current fixed (ds) should become only the initial value.

Use

[
ds_{\rm initial}
]

as the nominal starting scale.

Retain a relative hard floor of approximately

[
ds_{\min}
=========

10^{-6}ds_{\rm initial}
]

unless current empirical evidence supports another value.

Introduce a configurable upper limit

[
ds_{\max}.
]

A reasonable initial conservative default is several times (ds_{\rm initial}), not orders of magnitude larger.

The exact defaults should remain configuration parameters.

## After a successful corrector

Let

[
n_k
]

be the number of Newton iterations.

Choose a target corrector effort

[
n_*
]

around 3–5 Newton iterations.

Define

[
q_N
===

\left(
\frac{n_*}
{\max(n_k,1)}
\right)^\beta
]

with

[
\beta\approx\frac12.
]

Clamp the multiplicative change, for example to approximately

[
0.5\leq q_N\leq1.5.
]

Then calculate a provisional

[
ds_{\rm trial}=ds_kq_N.
]

The exact factors should be tunable.

The important behavior is:

* very easy correction → modest growth;
* average correction → little change;
* difficult correction → shrink.

Do not jump directly by huge factors.

---

# 11. Add curvature-based step control

Newton iteration count alone responds after a poor predictor.

Also measure branch curvature before making the next predictor.

With consecutive oriented normalized tangents,

[
t_{k-1},
\qquad
t_k,
]

calculate

[
c_k
===

\operatorname{clip}
(
\langle t_{k-1},t_k\rangle_W,
-1,
1
).
]

Then

[
\boxed{
\theta_k=\arccos(c_k).
}
]

Small (\theta) means a nearly straight branch.

Large (\theta) means the branch is bending strongly.

Use the tangent angle to further cap (ds).

Suggested starting behavior:

| Tangent change             | Action                    |
| -------------------------- | ------------------------- |
| (\theta<5^\circ)           | allow Newton-based growth |
| (5^\circ-15^\circ)         | no special action         |
| (15^\circ-30^\circ)        | prevent growth            |
| (30^\circ-45^\circ)        | shrink moderately         |
| (>45^\circ)                | shrink aggressively       |
| extreme discontinuous jump | reject/recompute tangent  |

These numbers are starting heuristics, not physics constants.

Store (\theta_k).

A fold should naturally produce increased curvature and therefore smaller steps without a fold-specific hardcoded (ds).

---

# 12. Failure handling for a continuation step

If the corrector fails:

Do not immediately declare branch failure.

Restore the last accepted point exactly.

Do not use the failed state as a future predictor.

Reduce

[
ds\leftarrow \rho ds
]

with an initial

[
\rho\approx0.5.
]

Retry from a freshly generated predictor.

Do not simply reuse the failed predictor after shrinking (ds).

Recalculate

[
Y_{\rm pred}
============

Y_{\rm accepted}
+
ds_{\rm new}t_{\rm accepted}.
]

If the failure involved an invalid or stale preconditioner, permit a forced rebuild before retrying.

Continue until:

[
ds<ds_{\min}
]

or another explicit resource limit is reached.

Distinguish:

* corrector failed;
* linear solve failed;
* residual evaluation failed;
* nonfinite state;
* preconditioner failure;
* wall-clock deadline;
* minimum step reached.

Do not collapse these into one generic continuation failure.

---

# 13. Accepted-step state must be transactional

A continuation point becomes authoritative only after all of the following succeed:

[
R\text{ converged}
]

[
g\text{ converged}
]

[
X,\mu\text{ finite}
]

[
t\text{ computed}
]

[
t\text{ normalized}
]

and required telemetry has been generated.

Keep:

[
Y_{n-1},
t_{n-1}
]

and

[
Y_n,
t_n
]

until the next point is fully accepted.

A failed attempted point must not modify:

* accepted branch position;
* previous tangent;
* fold bracket;
* metric-scale history;
* target-bracketing state.

---

# 14. Primary fold detection must use the tangent, not a noisy eigenvalue

A fold candidate exists when oriented consecutive accepted tangents satisfy

[
t_{\mu,n-1}t_{\mu,n}<0.
]

Equivalently, (\dot\mu) changes sign.

This is the primary geometric fold detector.

Do not use determinant signs of a large Jacobian.

Do not use the smallest eigenvalue alone as the primary fold detector.

The Jacobian is not guaranteed to be symmetric or normal.

At a fold:

[
t_\mu=0.
]

Meanwhile

[
Jt_X=0.
]

So the tangent itself supplies a right-null-vector approximation near the fold.

---

# 15. Add a fold candidate confidence check

Once a (t_\mu) sign change is detected, perform diagnostics around the event.

Required information:

[
\mu_{-},\quad
\mu_{+},
]

[
t_{\mu,-},\quad
t_{\mu,+},
]

[
|R_-|,\quad
|R_+|,
]

smallest-singularity diagnostic if available,

bordered-system conditioning diagnostic if available.

A fold candidate should only become a confirmed fold if:

* both surrounding PALC points are genuinely converged;
* tangent orientation is consistent;
* (t_\mu) crosses zero;
* ordinary Jacobian singularity evidence increases;
* augmented/bordered system remains solvable.

That is the pattern already seen at 7.9 GHz. 

---

# 16. Refine each fold instead of merely noting a sign change

Do not store fold location as one of the neighboring PALC points.

Refine it.

The goal is to find

[
(X_f,\mu_f)
]

such that

[
R(X_f,\mu_f)=0
]

and

[
t_{\mu,f}\approx0.
]

Initially implement a safeguarded PALC-based refinement rather than immediately building the full minimally augmented fold solver.

Maintain a bracket in arclength:

[
s_a<s_f<s_b
]

with

[
t_\mu(s_a)t_\mu(s_b)<0.
]

Then repeatedly generate intermediate points using a secant estimate of the zero of (t_\mu):

[
s_*=
s_a
---

t_{\mu,a}
\frac{s_b-s_a}
{t_{\mu,b}-t_{\mu,a}}.
]

Safeguard it so it cannot approach either bracket edge excessively.

Correct onto the branch at that local pseudo-arclength location.

Recompute the tangent.

Update the bracket according to the sign of (t_\mu).

Stop when at least one configured fold criterion is met, ideally both:

[
|t_\mu|<\varepsilon_{\rm fold,t}
]

and

[
|\mu_b-\mu_a|
<
\varepsilon_{\rm fold,\mu}.
]

Choose (\varepsilon_{\rm fold,\mu}) based on the map resolution.

The refined fold should be substantially more precise than the power-grid spacing.

---

# 17. Implement the mathematically correct simple-fold test

For a suspected simple fold, obtain approximate right and left null vectors:

[
Jv\approx0,
]

[
w^\mathsf TJ\approx0.
]

Normalize them.

The important transversality quantity is

[
\boxed{
\tau
====

\frac{
|w^\mathsf TR_\mu|
}{
|w||R_\mu|
}.
}
]

Since

[
R_\mu=-S_{\rm ref},
]

[
\boxed{
\tau
====

\frac{
|w^\mathsf TS_{\rm ref}|
}{
|w||S_{\rm ref}|
}.
}
]

A regular simple fold has:

[
\dim\ker J=1
]

and

[
\tau\neq0.
]

A possible branch point or higher degeneracy is indicated when:

* more than one independent singular direction approaches zero, or
* (\tau) becomes numerically indistinguishable from zero, or
* the bordered system also becomes badly singular.

This should ultimately replace “run the bordered condition estimate for more iterations and infer a taxonomy” as the primary mathematical classification.

Keep the conditioning diagnostic as supporting telemetry.

---

# 18. Do not confuse eigenvalues and singular values

For singularity classification, prefer information about

[
\sigma_{\min}(J)
]

over

[
\min|\lambda_i(J)|.
]

The HB Jacobian can be strongly non-normal.

A tiny eigenvalue often indicates something useful, but

[
|J^{-1}|
========

\frac{1}{\sigma_{\min}(J)}
]

is the quantity directly associated with numerical singularity.

Do not make an expensive smallest-singular-value solve mandatory at every continuation point.

Use it:

* near fold candidates;
* for diagnostics;
* for validation runs.

If the existing implementation has only eigenvalue diagnostics, retain them but label them appropriately.

---

# 19. Change what “target reached” means

PALC should **not** be required to land exactly on

[
\mu=\mu_{\rm target}.
]

Instead detect when consecutive accepted branch points bracket the target:

[
(\mu_n-\mu_t)
(\mu_{n+1}-\mu_t)
\leq0.
]

When this occurs on a known branch segment, construct an interpolated state.

A first estimate can use linear interpolation in (\mu):

[
\alpha=
\frac{
\mu_t-\mu_n
}{
\mu_{n+1}-\mu_n
},
]

[
X_{\rm guess}
=============

(1-\alpha)X_n+\alpha X_{n+1}.
]

Then solve the ordinary fixed-(\mu) HB problem:

[
\boxed{
R(X,\mu_t)=0
}
]

using this guess.

That final fixed-parameter Newton solve produces the authoritative map state.

Therefore PALC discovers the geometry of the branch; ordinary Newton gives exact requested map samples.

---

# 20. Introduce branch segments

Every fold divides a connected branch into monotonic-(\mu) segments.

For example:

[
\text{segment 0}
\rightarrow
\text{fold 1}
\rightarrow
\text{segment 1}
\rightarrow
\text{fold 2}
\rightarrow
\text{segment 2}.
]

Each accepted branch point must carry:

* connected branch ID;
* segment ID;
* arclength (s);
* (\mu);
* (I_p);
* equivalent dBm;
* tangent sign (t_\mu);
* number of folds passed.

A target may intersect multiple segments.

That means it may have multiple HB solutions.

Do not silently overwrite them.

---

# 21. Explicitly distinguish production continuation from branch exploration

This distinction is essential.

Create two conceptual continuation objectives.

## Production target mode

Goal:

> Find the operating state connected continuously to the low-drive physical solution.

Start at low drive.

Trace increasing physical drive.

If

[
\mu_{\rm target}
]

is reached before a fold, return that solution.

If a confirmed simple fold occurs at

[
\boxed{
\mu_f<\mu_{\rm target},
}
]

stop.

Do **not** automatically spend another 150 or 500 steps following the returning branch.

Return a semantic result equivalent to:

> Target is beyond the first fold of the low-drive synchronous branch.

This is not a generic numerical failure.

This is the main behavioral change required for the current 7.9 GHz problem.

## Branch exploration mode

Goal:

> Map the connected HB solution component regardless of monotonic drive.

Here, continue through folds.

Permit:

[
t_\mu<0.
]

Continue until one of:

* another target intersection;
* second fold;
* configured number of folds;
* configured arclength span;
* configured maximum accepted points;
* minimum (ds);
* harmonic-resolution invalidity;
* nonfinite state;
* user-defined physical drive bounds;
* explicit resource deadline.

Only this mode should spend substantial effort on post-fold branches.

---

# 22. Remove “post-fold step budget” as the primary solution mechanism

A larger post-fold step budget may remain as a safety bound.

It must no longer be the strategy for answering:

> Does the production target exist?

If the first fold occurs below the target, the production continuation has already answered the question for the low-drive connected branch.

A second fold might eventually turn the branch toward higher (\mu), but that is **branch exploration**, not production-target continuation.

---

# 23. Build one continuation branch per pump frequency

This is the map-level architectural change.

For each fixed pump frequency:

1. Construct the linear operators and frequency-specific cached data.

2. Construct (S_{\rm ref}).

3. Determine all requested target powers in that frequency column.

4. Convert them to

   [
   \mu_{t,1},\mu_{t,2},\ldots.
   ]

5. Sort targets by physical drive.

6. Establish one low-drive branch seed.

7. Trace one branch in ((X,\mu)).

8. Whenever the branch brackets a requested target, run an exact fixed-(\mu) solve.

9. Store that map point.

10. Continue until:

* all desired principal-branch targets have been reached, or
* a confirmed fold lies below the remaining targets.

This eliminates repeated rediscovery of the same fold for every power point.

---

# 24. Make the branch object reusable

The continuation result for one frequency should conceptually contain:

### Branch metadata

[
f_p,
\quad
I_{\rm ref},
\quad
\text{metric convention},
\quad
\text{harmonic basis}.
]

### Accepted points

For every point:

[
s,
\mu,
I_p,
P_{\rm dBm},
X.
]

### Numerical telemetry

For every point:

* HB residual;
* arclength residual;
* Newton iterations;
* GMRES iterations;
* elapsed corrector time;
* (ds);
* state scale;
* tangent angle;
* (t_\mu);
* preconditioner rebuild count.

### Events

* fold candidate;
* refined fold;
* target intersection;
* metric rescale;
* harmonic-resolution warning;
* continuation failure.

### Branch topology

* branch ID;
* segment ID;
* fold count;
* orientation.

Do not require full (X) to be persisted for every intermediate point if disk cost is excessive.

At minimum retain enough states to:

* continue;
* interpolate target crossings;
* restart near folds;
* reproduce diagnostics.

Checkpoint strategically.

---

# 25. Implement branch-aware target status semantics

For every requested map target, distinguish at least:

### Converged

A fixed-(\mu) HB solution was obtained on the principal branch.

### Beyond principal fold

A confirmed simple fold exists at

[
\mu_f<\mu_{\rm target}
]

before the target could be reached on the low-drive branch.

### Numerical continuation failure

No confirmed fold explains the failure.

### Harmonic resolution insufficient

Continuation may have converged numerically, but truncation diagnostics are unacceptable.

### Alternative branch candidate

A target is intersected again after a fold in explicit branch-exploration mode.

### Dynamically unclassified

An HB root exists but its dynamical stability has not been established.

Preserve the project's external PASS/PARTIAL/FAIL conventions where necessary, but retain the detailed internal reason.

For example:

[
\text{status}=PARTIAL
]

with

[
\text{reason}=\text{BEYOND_PRINCIPAL_FOLD}
]

is vastly more informative than a generic NaN.

---

# 26. Add a direct diagnostic plot of the continuation branch

For every fold-debugging run produce at minimum:

Horizontal axis:

[
\mu
]

or physical pump current.

Vertical axis:

one or more scalar branch observables.

Useful observables include:

[
|X|_W,
]

fundamental pump amplitude,

maximum junction phase amplitude,

selected output pump amplitude,

total harmonic norm.

Color or mark:

* accepted PALC points;
* folds;
* map targets;
* target intersections;
* failed attempted steps.

The crucial plot is a genuine branch diagram.

A plot indexed only by continuation step does not show branch topology clearly.

---

# 27. Add maximum Josephson phase diagnostics

For every accepted state reconstruct the junction phase waveform.

For each junction (b),

[
\phi_b(t)
=========

\frac{
\psi_b(t)
}{
\phi_0
}.
]

Store or aggregate:

[
\max_{b,t}|\phi_b(t)|.
]

Also preferably store quantiles across junctions rather than all waveforms.

For example:

* median maximum phase;
* 90th percentile;
* 99th percentile;
* global maximum;
* junction index of global maximum.

This gives direct physical context for a fold occurring near strong junction excitation.

---

# 28. Add harmonic-tail diagnostics before trusting a high-drive fold

A mathematically perfect continuation result is not enough if the HB basis is under-resolved.

For every accepted high-drive point calculate a harmonic energy/amplitude tail.

One simple diagnostic is

[
\boxed{
\eta_X
======

\frac{
\displaystyle\sum_{k\in K_{\rm tail}}
|X_k|^2
}{
\displaystyle\sum_{k\in K}
|X_k|^2
}.
}
]

Define (K_{\rm tail}) as the highest one or several retained harmonics.

Also evaluate the tangent nonlinearity spectrum.

From

[
\gamma_b(t)
===========

\frac{I_{c,b}}{\phi_0}
\cos\phi_b(t),
]

calculate

[
\widehat\gamma_{\ell,b}.
]

Define a similar tail measure

[
\eta_\gamma.
]

This is especially important because the Jacobian may require higher harmonic resolution before the state waveform obviously looks under-resolved.

---

# 29. Add a fold-resolution convergence workflow

Before calling a fold physical, rerun a small number of representative folds using increased numerical resolution.

At minimum compare:

[
H
\rightarrow
H+\Delta H
]

and/or

[
N_t
\rightarrow
2N_t.
]

For each resolution record

[
\mu_f(H,N_t).
]

Define relative fold movement

[
\delta_f
========

\frac{
|\mu_f^{(2)}-\mu_f^{(1)}|
}{
\max(|\mu_f^{(1)}|,\epsilon)
}.
]

The fold is resolution-stable only if (\delta_f) is below a chosen engineering tolerance.

Also compare:

* state norm at fold;
* maximum phase;
* residual;
* tail measures;
* null-vector diagnostics.

Do this first at:

* 7.9 GHz;
* one frequency where no problematic fold occurs;
* another frequency where a fold was independently observed.

Do **not** immediately run the entire 19-frequency sweep at doubled harmonics.

---

# 30. Establish a clear criterion for declaring the current 7.9 GHz case solved

The first milestone is complete when the new branch trace at 7.9 GHz demonstrates all of the following:

* low-drive branch matches the existing solver;
* (\mu) increases smoothly initially;
* a fold is approached;
* (t_\mu\rightarrow0);
* (t_\mu) changes sign;
* ordinary Jacobian singularity evidence grows;
* bordered system remains usable;
* the fold is refined;
* the calculated fold drive is stable against a reasonable HB-resolution increase;
* targets below the fold are recovered;
* targets above the fold are classified as `BEYOND_PRINCIPAL_FOLD` rather than burning a post-fold step budget.

If those are satisfied, the original production problem has been numerically resolved.

---

# 31. After the first fold, determine whether the returning branch is worth tracing

Only after the above should branch exploration be used.

Start from the refined fold and move onto the post-fold segment.

Because

[
t_\mu<0
]

there, (\mu) should initially decrease.

Verify that this is smooth and not an orientation artifact.

Use adaptive (ds).

The expectation is:

* shrink naturally around the fold;
* grow again once the branch straightens.

This is the test that adaptive (ds) is working correctly.

A healthy post-fold branch should not remain stuck forever at the tiny step size needed immediately at the turning point.

---

# 32. Detect a second fold explicitly

Do not infer a second fold merely because (\mu) starts rising somewhere.

Use the same criterion:

[
t_\mu
]

must cross zero again.

Refine it.

Then the connected component has an S-like topology.

If a second fold exists and

[
\mu
]

subsequently rises past a requested high-power target, the target can have another periodic HB solution.

The branch data model must permit storing both:

[
X_{\rm lower}(\mu_t)
]

and

[
X_{\rm upper}(\mu_t).
]

Never silently choose one.

---

# 33. If the branch never comes back, stop with a geometric criterion

In branch-exploration mode define an arclength exploration budget, not just a number of steps.

For example track

[
\Delta s_{\rm explored}.
]

Also monitor physical progress in

[
\mu.
]

Terminate a post-fold search when several conditions collectively indicate diminishing value, such as:

* large arclength traveled;
* no second (t_\mu=0) event;
* (\mu) continues moving away from all targets;
* branch curvature has become small;
* adaptive (ds) has become large and many easy steps have been accepted.

This is much more meaningful than:

> we tried another 150 steps.

---

# 34. Implement fold continuation in frequency only after single-frequency tracing is trustworthy

Once simple folds have been reliably detected at several frequencies, stop rediscovering them independently.

Treat the fold as a codimension-one object in

[
(f_p,\mu).
]

The desired output is

[
\boxed{
\mu_f(f_p).
}
]

Equivalent power boundary:

[
P_f(f_p).
]

This is the actual fold boundary of the pump map.

---

# 35. Use a minimally augmented fold formulation

Conceptually introduce a scalar singularity function

[
\sigma(X,\mu,f)=0
]

that vanishes when (J) is singular in the relevant one-dimensional direction.

Then solve

[
\boxed{
R(X,\mu,f)=0
}
]

[
\boxed{
\sigma(X,\mu,f)=0.
}
]

Use a bordered/minimally-augmented formulation so that the solver does not introduce a second full (N)-dimensional unknown null vector unless necessary.

Reuse the same large-system infrastructure:

* matrix-free actions;
* Schur reduction;
* preconditioning;
* bordered solves.

Do not assemble an enormous dense codimension-two Jacobian.

---

# 36. Seed the fold curve from a known refined fold

Use the best-characterized frequency first, likely 7.9 GHz.

Starting data:

[
(X_f,\mu_f,f_p).
]

Obtain a second nearby fold either by:

* independently refining the adjacent-frequency fold once, or
* making a small frequency predictor and correcting the fold equations.

Then obtain a secant/tangent direction in the fold manifold.

Continue the fold through frequency.

---

# 37. The output of fold continuation should become a first-class map boundary

For every pump frequency, the map orchestrator can ask:

[
\mu_{\rm target}
<
\mu_f(f_p)?
]

If yes, ordinary principal-branch solving is expected to be possible.

If

[
\mu_{\rm target}>
\mu_f(f_p),
]

do not repeatedly launch expensive principal-branch recovery attempts.

The fold curve becomes a physics-informed skip boundary.

This is much better than the previous heuristic `fold-skip-patience` behavior documented in the earlier continuation workflow. 

---

# 38. Preserve a safety band around the fold boundary

Do not treat an interpolated fold curve as mathematically exact.

Define a numerical uncertainty margin

[
\Delta\mu_f.
]

Targets well below

[
\mu_f-\Delta\mu_f
]

can use standard warm Newton solves.

Targets near

[
|\mu_t-\mu_f|
<
\Delta\mu_f
]

should use a fold-aware continuation path.

Targets well above

[
\mu_f+\Delta\mu_f
]

should be classified as beyond the principal fold unless explicit alternative-branch exploration is requested.

---

# 39. Integrate this with warm-start map traversal rather than replacing warm starts

PALC should not become the default method for every easy map point.

Use the cheapest solver compatible with the local branch geometry.

Far below the fold:

[
\text{neighbor warm start}
\rightarrow
\text{direct Newton}.
]

If direct Newton fails unexpectedly:

[
\text{local continuation recovery}.
]

Near the fold:

[
\text{branch PALC}.
]

If target is beyond known fold:

[
\text{do not repeatedly retry principal-branch Newton}.
]

This preserves the performance benefits of the existing map workflow.

---

# 40. Reuse branches across neighboring target powers

If a branch has already been traced through a region, later map requests at the same frequency should query the existing branch cache.

Do not rerun PALC.

For a target (\mu_t):

find two stored branch points on the required segment with

[
\mu_a\leq\mu_t\leq\mu_b.
]

Interpolate.

Run fixed-(\mu) Newton.

Store exact solution.

The branch becomes a reusable continuation backbone.

---

# 41. Reuse information across neighboring frequencies

Once the one-frequency implementation is proven, use neighboring-frequency states as seeds.

For frequency

[
f_j
]

and adjacent

[
f_{j+1},
]

a state at similar physical (\mu) is likely a much better starting guess than a linear seed.

Use frequency continuation cautiously because frequency changes the dynamic operator:

[
D_k(f_p).
]

Do not blindly copy without a correction solve.

But the nearby branch geometry should become an available predictor.

---

# 42. Add a transient-crossing diagnostic as a separate phase

This does **not** need to block the PALC refactor.

Once the first physical fold is established, determine what the actual time-domain system does when driven through it.

Choose one problematic frequency, initially 7.9 GHz.

Start from a stable low-drive state sufficiently below the fold.

Slowly ramp physical pump current through

[
I_f.
]

Continue to a target above the fold.

Record a long enough time interval after the ramp for transients to decay or reveal persistent non-periodicity.

Do not initialize the high-power transient from zero unless specifically testing basin dependence.

---

# 43. Classify the post-fold transient state

After transients, test whether the waveform is:

### Pump-periodic

[
x(t+T_p)\approx x(t).
]

Then Fourier-project it into the existing HB basis and use that as an initial condition for a fixed-(\mu) Newton solve.

If Newton converges:

there is another synchronous HB solution.

### Period doubled

[
x(t+2T_p)\approx x(t)
]

but

[
x(t+T_p)\not\approx x(t).
]

The existing pump-harmonic basis cannot represent the attracting state.

### Higher-period

[
x(t+nT_p)\approx x(t).
]

Requires an expanded fundamental period.

### Quasiperiodic

No finite small integer period but distinct additional incommensurate spectral components.

Requires torus/multifundamental HB.

### Irregular/chaotic

No stable discrete quasiperiodic spectrum and strong sensitivity/persistent broadband behavior.

Ordinary periodic HB is no longer the right target model.

This diagnostic tells you whether searching harder for another (T_p)-periodic root is scientifically meaningful.

---

# 44. Explicitly detect Josephson phase winding

This is important for the future high-drive model.

The existing Fourier representation assumes a bounded periodic node-flux waveform:

[
x(t+T_p)=x(t).
]

For every junction calculate its phase change across one pump period:

[
\Delta\phi_b
============

\phi_b(t+T_p)-\phi_b(t).
]

Define winding number

[
\boxed{
m_b
===

\frac{\Delta\phi_b}{2\pi}.
}
]

After transients, test whether

[
m_b\approx0
]

for all junctions.

If a junction has persistent integer or near-integer nonzero winding,

[
m_b\neq0,
]

then the state contains a running phase.

The bounded periodic flux ansatz cannot represent that state correctly.

Classify it separately.

Do not respond by simply adding more pump harmonics.

A future winding-aware formulation would require something of the form

[
\phi_b(t)
=========

m_b\omega_pt
+
\phi_{b,\rm per}(t).
]

That belongs to the later transient/JJ-switching extension rather than this PALC milestone.

---

# 45. Do not implement deflation in the first patch

Deflation is not necessary to fix the current simple-fold continuation problem.

The immediate order must be:

[
\boxed{
\text{physical }\mu
\rightarrow
\text{adaptive PALC}
\rightarrow
\text{fold detection}
\rightarrow
\text{fold refinement}
\rightarrow
\text{target reachability}.
}
]

Only consider deflation after evidence indicates that another (T_p)-periodic root exists at the target.

Such evidence could be:

* transient settling into another synchronous state;
* second branch observed through another method;
* experiment requiring a periodic high-drive state not connected to the principal branch.

---

# 46. If deflation is later added, keep it matrix-free

For a known root (X_1), define a weighted distance

[
d_1(X)
======

|X-X_1|_W.
]

A simple deflation factor can be

[
D(X)
====

\alpha+d_1(X)^{-p}.
]

Define

[
G(X)=D(X)R(X).
]

The JVP is

[
\boxed{
J_Gv
====

D(X)J_Rv
+
[\nabla D(X)^\mathsf Tv]R(X).
}
]

Therefore no dense Jacobian is required.

For multiple known solutions,

[
D(X)
====

\prod_j
\left[
\alpha+|X-X_j|_W^{-p}
\right].
]

But defer all of this until another synchronous branch is actually motivated.

---

# 47. Add stability analysis after branch geometry is trustworthy

Existence of an HB root does not imply physical observability.

Eventually linearize the time-domain equations around the periodic pump state:

[
C\delta\ddot x
+
G\delta\dot x
+
\left[
K+
B_\phi
\operatorname{diag}\gamma(t)
B_\phi^\mathsf T
\right]
\delta x
========

0.

]

with

[
\gamma_b(t)
===========

\frac{I_{c,b}}{\phi_0}
\cos\phi_b(t).
]

Propagate the variational dynamics over one pump period.

Obtain the monodromy operator

[
M.
]

Its eigenvalues

[
\rho_i
]

are Floquet multipliers.

A periodic pump state is dynamically stable if the physically relevant multipliers satisfy

[
|\rho_i|<1
]

subject to any neutral modes implied by the formulation.

This should eventually label each HB branch segment as:

* stable;
* unstable;
* unknown.

Do not block the continuation refactor on this feature.

---

# 48. High-value stability events

Eventually detect:

### Fold / saddle-node of periodic solutions

A multiplier approaches

[
+1.
]

### Period doubling

A multiplier approaches

[
-1.
]

### Neimark–Sacker / torus bifurcation

A complex-conjugate pair crosses

[
|\rho|=1.
]

This will connect the HB branch structure to the transient classification.

---

# 49. Required telemetry schema

For every accepted PALC point collect at least:

| Quantity                | Purpose                    |
| ----------------------- | -------------------------- |
| frequency               | operating coordinate       |
| (\mu)                   | physical drive             |
| current                 | physical interpretation    |
| dBm equivalent          | map interpretation         |
| arclength (s)           | branch coordinate          |
| (ds)                    | step adaptation            |
| state norm              | branch plotting            |
| (t_\mu)                 | fold detection             |
| tangent angle           | curvature                  |
| state scale             | metric diagnostics         |
| HB residual             | correctness                |
| PALC residual           | correctness                |
| Newton iterations       | adaptive control           |
| GMRES iterations        | solver cost                |
| GMRES failures          | conditioning               |
| preconditioner rebuilds | performance                |
| corrector time          | performance                |
| singularity estimate    | fold diagnostics           |
| bordered conditioning   | fold-vs-degeneracy support |
| (\eta_X)                | HB resolution              |
| (\eta_\gamma)           | Jacobian resolution        |
| max JJ phase            | physics                    |
| branch ID               | topology                   |
| segment ID              | topology                   |
| fold count              | topology                   |

Do not make diagnostics accessible only through logs.

Make them machine-readable.

---

# 50. Required event telemetry

Every unusual event should have a structured event record.

Events include:

* predictor created;
* corrector accepted;
* corrector rejected;
* step reduced;
* step enlarged;
* metric rescaled;
* tangent reoriented;
* fold candidate;
* fold refinement started;
* fold confirmed;
* target bracketed;
* target exact solve converged;
* target exact solve failed;
* harmonic warning;
* minimum step reached;
* resource deadline reached;
* branch exploration stopped.

Each event should contain:

[
s,\mu,ds
]

and a reason.

This will make future continuation failures diagnosable without rerunning expensive jobs.

---

# 51. Unit test 1: scalar simple fold

Use

[
R(x,\mu)=x^2+\mu-1=0.
]

Exact branch:

[
\mu=1-x^2.
]

Exact fold:

[
x=0,\qquad\mu=1.
]

Starting on the positive-(x) branch below the fold, PALC must:

* approach (\mu=1);
* pass through it;
* produce (t_\mu=0);
* flip sign of (t_\mu);
* continue onto the negative-(x) branch;
* keep the bordered corrector nonsingular.

This is a mandatory regression test.

---

# 52. Unit test 2: target beyond a simple fold

Using the same system, request

[
\mu_t>1.
]

Production target mode must:

* find/refine the fold at (\mu=1);
* terminate;
* classify the target as beyond the principal fold.

It must **not** walk indefinitely along the returning branch hoping to reach (\mu_t).

This specifically tests the logic required by the 7.9 GHz case.

---

# 53. Unit test 3: two-fold S-curve

Use a scalar system with two folds, for example an appropriate cubic normal form such as

[
R(x,\mu)=x^3-x+\mu.
]

PALC must:

* pass fold 1;
* follow the returning segment;
* pass fold 2;
* identify all three monotonic-(\mu) segments;
* permit a target to intersect multiple segments;
* return distinct roots rather than overwriting them.

---

# 54. Unit test 4: no fold

Use a monotonically solvable nonlinear equation.

The adaptive PALC solver should grow (ds) when the branch is easy.

No false fold event should occur.

---

# 55. Unit test 5: metric-rescaling invariance

Trace the same branch with a deliberate metric scale update.

Immediately before rescaling:

[
\Delta Y_{\rm old}
==================

ds_{\rm old}t_{\rm old}.
]

Immediately after:

[
\Delta Y_{\rm new}
==================

ds_{\rm new}t_{\rm new}.
]

Require

[
|\Delta Y_{\rm old}-\Delta Y_{\rm new}|
]

to be near numerical roundoff/tolerance.

This prevents reintroducing a metric-related continuation bug.

---

# 56. Unit test 6: tangent orientation

Artificially multiply one computed tangent by (-1).

Orientation logic must restore continuity.

A numerical null-vector sign flip must not generate a fake fold.

---

# 57. Unit test 7: adaptive (ds)

Use a curve with one nearly straight region and one strongly curved region.

Verify:

[
ds_{\rm straight}

>

ds_{\rm curved}.
]

After leaving the curved region, verify that (ds) recovers.

The controller should not become permanently trapped at its minimum near a fold.

---

# 58. Unit test 8: exact target landing

Trace through

[
\mu_t
]

without landing exactly on it.

Bracket it.

Interpolate.

Run the fixed-parameter solve.

Verify:

[
|\mu-\mu_t|=0
]

by construction and

[
R(X,\mu_t)
]

meets the standard HB tolerance.

---

# 59. Unit test 9: fold transversality

On the simple-fold scalar example evaluate

[
w^\mathsf TR_\mu.
]

Confirm it is nonzero.

Add a deliberately degenerate test problem where the transversality condition fails.

Ensure the classifier does not label both identically.

---

# 60. Integration test: easy TWPA pump point

Choose a pump frequency/power far from the problematic regime.

Compare:

* existing direct/warm solution;
* new physical-(\mu) continuation solution.

Require agreement within existing solver tolerance for:

[
X,
]

pump transmission,

derived gain inputs if applicable.

The continuation refactor must not alter easy-regime physics.

---

# 61. Integration test: 7.0 GHz

This is the metric-regression case.

The previous premature minimum-step failure must remain fixed.

Verify that:

* periodic metric rescaling works;
* no reappearance of the old false “snaking” interpretation occurs;
* adaptive (ds) behaves sensibly;
* any actual later fold is detected geometrically.

This protects fix #8 in the current problem report. 

---

# 62. Integration test: 7.9 GHz

This is the primary acceptance test.

Produce:

[
\mu(s),
]

[
t_\mu(s),
]

[
ds(s),
]

HB residual,

Newton iterations,

singularity diagnostics.

The expected qualitative pattern is:

[
\mu\uparrow
]

then

[
t_\mu\rightarrow0
]

then

[
\mu\downarrow.
]

Confirm the refined fold.

Check the target powers corresponding to the previously problematic region.

The solver must automatically distinguish:

* targets below the fold;
* target near the fold;
* targets beyond the principal fold.

No manually selected `+150 post-fold steps` should be needed for this classification.

---

# 63. Integration test: harmonic-resolution convergence at 7.9 GHz

Repeat only the fold neighborhood with:

* current harmonic count;
* increased harmonic count;

and preferably:

* current (N_t);
* increased (N_t).

Compare

[
\mu_f.
]

If fold movement is material, do not call the physical boundary validated yet.

Instead report:

> continuation works, but fold location is harmonic-resolution dependent.

That would be a physics/numerics-model issue, not a continuation bug.

---

# 64. Performance requirement

The refactor must not create a dense augmented Jacobian.

Memory complexity should remain dominated by:

* existing HB state;
* existing sparse preconditioner;
* existing Schur factors;
* Krylov vectors;
* small continuation vectors.

Adding one scalar continuation unknown should have negligible memory impact.

A PALC bordered solve should reuse existing expensive factorizations whenever mathematically valid.

---

# 65. Preconditioner policy near the fold

Do not rebuild the preconditioner at every PALC Newton iteration by default.

Retain the existing reuse policy where effective.

However force a rebuild when:

* accumulated Newton convergence degrades substantially;
* GMRES iteration count jumps;
* metric/state has moved materially;
* a previous linear solve reports numerical failure.

Preconditioner failure at an exact Jacobian singularity must not automatically imply augmented-system singularity.

This distinction is exactly why the bordered formulation exists.

The existing singular-preconditioner least-squares fallback should remain available. 

---

# 66. Do not use the ordinary Jacobian inverse to implement the fold crossing

At the fold,

[
J^{-1}
]

does not exist.

Any algorithm whose core operation is effectively

[
J^{-1}b
]

without a properly augmented regularization will fail exactly where PALC is needed.

The augmented/bordered solve must remain the fundamental corrector operation around the fold.

---

# 67. Stopping conditions must be objective-specific

Do not use one generic `max_steps` as the dominant stopping mechanism.

### Production branch tracing stops when

* all requested targets have been solved; or
* first confirmed principal fold lies below remaining targets; or
* genuine numerical failure occurs.

### Branch exploration stops when

* requested branch topology is found;
* configured fold count reached;
* arclength budget reached;
* physical (\mu) bounds exceeded;
* resolution diagnostics fail;
* minimum (ds) reached;
* resource budget reached.

### Fold refinement stops when

* (t_\mu) sufficiently near zero;
* (\mu)-bracket sufficiently small;
* residuals meet tolerance.

Different tasks require different stopping logic.

---

# 68. Production rollout order

Implement in this exact order:

**Milestone A — parameterization**

Introduce fixed-reference (\mu).

Do not yet change the rest of PALC.

Prove easy points match existing results.

**Milestone B — branch representation**

Store accepted points in ((s,\mu,X,t)) form.

Add target crossing detection.

**Milestone C — adaptive (ds)**

Implement Newton-effort and curvature adaptation.

Validate on toy systems.

**Milestone D — fold events**

Implement (t_\mu) crossing detection and fold refinement.

**Milestone E — production semantics**

Stop at a principal fold below target instead of following the returning branch by default.

This milestone should solve the immediate practical problem.

**Milestone F — diagnostic validation**

Add transversality and harmonic-tail diagnostics.

**Milestone G — per-frequency branch reuse**

Trace once per frequency and extract multiple map powers.

**Milestone H — fold curve**

Continue (\mu_f(f_p)).

**Milestone I — transient crossing**

Determine the actual post-fold dynamical regime.

**Milestone J — stability**

Floquet classification.

**Milestone K — optional alternate-root search**

Deflation only if motivated.

Do not combine all milestones into one giant patch.

---

# 69. What the agent must not do

Do not:

* rewrite the HB solver;
* replace JVP with a dense Jacobian;
* replace GMRES merely for this feature;
* replace Schur reduction;
* remove working preconditioners;
* remove metric rescaling;
* treat negative (t_\mu) as failure;
* infer a fold from Newton failure alone;
* infer a branch point from a noisy eigenvalue alone;
* assume a returning branch must eventually reach the target;
* increase post-fold steps as the primary fix;
* add deflation before the branch geometry is understood;
* add time-domain simulation as a prerequisite for the PALC fix;
* silently treat a post-fold mathematical solution as physically stable;
* silently label targets above a fold as generic NaNs.

---

# 70. What the first final report from the agent should contain

After implementing through **Milestone E**, stop and report before proceeding to fold-curve or transient work.

The report must show the following for 7.9 GHz:

[
\mu(s)
]

plot,

[
t_\mu(s)
]

plot,

[
ds(s)
]

plot,

Newton iteration count versus (s),

refined fold location

[
\mu_f,
]

physical fold current

[
I_f,
]

equivalent source/device power according to the existing convention,

classification of each tested high-power target.

It should explicitly answer:

> Is (-19.0) dBm below, at, or beyond the first principal fold?

and

> Is (-18.5) dBm below, at, or beyond the first principal fold?

It should also state whether the fold moved materially when the harmonic/time-grid resolution was increased.

For 7.0 GHz it should show that the previous metric failure remains eliminated.

Only after those results are clean should the agent proceed to two-parameter fold continuation.

---

# Final intended architecture

When this is finished, the pump-map workflow should conceptually look like

[
\boxed{
f_p
\rightarrow
\text{trace physical pump branch}
\rightarrow
\text{detect/refine folds}
\rightarrow
\text{extract requested powers}
\rightarrow
\text{small-signal simulation}
}
]

rather than

[
f_p
\rightarrow
P_1
\rightarrow
\lambda:0\to1
]

[
f_p
\rightarrow
P_2
\rightarrow
\lambda:0\to1
]

[
f_p
\rightarrow
P_3
\rightarrow
\lambda:0\to1
]

with every target independently rediscovering the same nonlinear branch.

The important conceptual end state is:

[
\boxed{
\text{the branch is the primary numerical object;}
\quad
\text{map powers are samples of that branch.}
}
]

That is the change most likely to turn the current fold problem from an expensive recovery failure into useful physical information about the JTWPA operating boundary.

---

# Progress log

## Section 0 (codebase mapping, 2026-08-07)

`FullPumpProblem.residual_coeffs`/`source_coeffs` (`src/twpa_solver/pump/problem.py:193-239`)
already implement exactly `R(X,lambda) = D X + N(X) - lambda*S` with
`S = 0.5*lambda*pump_current_a` at the pump node -- `R_lambda = -S/lambda`
is linear/cheap, matching Section 1's assumed structure. Confirmed the
plan's central diagnosis directly: `pump_current_a` is a `FullPumpProblem`
constructor field, and `run_gain_map.py::InProcessEngine._build_problem`
(723-749) is called once per grid cell with that cell's own target current
as `pump_current_a` -- every map point genuinely does define its own
`lambda:0->1` problem for the nominally-same physical branch, exactly as
Section 1 states. `fold_power`/`run_fold_follow` (`solver.py:1627`,
`run_gain_map.py:2125-2160`) were already a partial, ad hoc instance of
Milestone A's idea (one problem built at a fixed per-frequency reference
current, reused for the fold search) -- they just weren't generalized into a
reusable primitive or connected to the per-cell `_recover` path, which still
uses each cell's own target current (`cur_t`) as its lambda=1 reference
(`run_gain_map.py:2041-2071`). `solve_arclength` (1110-1351) already
satisfies several plan requirements ahead of schedule: primary fold
detection is the `lam_dot` sign flip (Section 14), not an eigenvalue: sign;
tangent orientation is already flip-corrected via the same-metric dot
product (Section 6); the metric fix, relative step floor, periodic rescale,
and post-fold budget extension are already in place
(`arclength_fold_resolution_plan.md` Phases 0-2, prior session). Not yet
present: adaptive Newton-effort/curvature step control (only a crude
`step_size *= 1.25` growth on cheap correctors), fold refinement (raw
`lam_dot`-sign-flip lambda is stored, never bracketed/polished), branch/
segment objects, transversality check, mu-vs-frequency fold curve,
production/branch-exploration mode split, harmonic-tail diagnostics,
winding detection, stability/Floquet analysis. None of Sections 45-46
(deflation) apply yet, consistent with Section 45's own instruction.

## Milestone A (parameterization, 2026-08-07) -- DONE

Added `HarmonicNewtonKrylovSolver.solve_arclength_mu` (`solver.py`, after
`solve_arclength`): a thin reparameterization, not a rewrite. It computes
`k = i_ref / problem.pump_current_a` once and calls the existing
`solve_arclength` with every lambda-valued boundary argument (`mu0`,
`target_mu`, `ds`) multiplied by `k`, converting the returned `lambda`/
`fold_lambda` back to `mu` by dividing by `k`. This satisfies the "retain
the existing HB residual" invariant literally -- `problem.py` is untouched,
the corrector/metric/bordering/fold-detection code in `solve_arclength` runs
unmodified -- while giving the caller a `mu = I_physical/i_ref` coordinate
whose meaning does not depend on whatever `pump_current_a` a particular
`FullPumpProblem` instance happens to carry. `k=1` (`i_ref ==
problem.pump_current_a`) reduces to calling `solve_arclength` directly,
proven bit-identical by test.

Deviation from the plan's literal Section 1 (which describes building a
separate `S_ref` source vector): not needed. `source_coeffs` is already
exactly linear in its scale argument, so rescaling the scalar `lambda`
argument by the fixed constant `k` is mathematically identical to building
`S_ref = i_ref * S_per_amp` and working in `mu` natively -- doing it as a
boundary-value rescale avoids touching `problem.py` at all, a strictly
smaller diff for the same physics.

Tests (`tests/test_advanced_continuation.py`):
`test_solve_arclength_mu_k1_matches_solve_arclength` (bit-identical `k=1`
regression), `test_solve_arclength_mu_rescales_target_and_step_by_reference_current`
(`i_ref=2x pump_current_a`, `target_mu=0.5` reaches the exact same physical
state as the `k=1` case's `target_lam=1.0`), `test_solve_arclength_mu_fold_mu_scales_with_reference_current`
(fold location on the known-folding fixture scales by `1/k` as required).
3/3 pass.

## Milestone B (branch representation + target bracketing, 2026-08-07) -- DONE

Added `BranchPoint`, `ContinuationBranch`, `trace_branch`, `bracket_target`,
`resolve_target_on_branch` (`solver.py`, end of file, after `fold_power`).
`trace_branch` calls `solve_arclength_mu` once (target `mu_max`) and uses
its existing `on_step` hook to accumulate every accepted `(s, mu, X, t_mu)`
into `ContinuationBranch.points` -- no new corrector code, no dense
Jacobian, no change to `solve_arclength`'s algorithm. `bracket_target` scans
`points` for the two consecutive points straddling a requested `mu_target`
(first crossing only -- multi-segment/multi-crossing selection is
Section 20's segment-ID work, deferred). `resolve_target_on_branch` linearly
interpolates the bracketing states as a warm guess and runs the existing,
unmodified `solve_one` at the exact fixed-`mu` (converted to the problem's
native `lambda` via the same `k`) -- this is Section 19's "PALC discovers
geometry, ordinary Newton gives the exact sample" split.

`s` (cumulative `step_size`) is documented as a step-index proxy, not a
metric-exact arclength -- Section 4's metric-rescale-consistent `ds`/tangent
transform is deferred to Milestone C, so `s` is not yet meaningful across a
rescale event. `BranchPoint.t_mu` is the raw inner-`solve_arclength`
`lam_dot` (sign is exactly the mu-tangent's sign since `k>0`; magnitude is
in the lambda-parametrization, not separately mu-normalized) -- documented
rather than mis-labeled as an exact mu-rate.

Tests: `test_trace_branch_stores_points_and_resolve_matches_direct_solve`
(a target resolved off a traced branch matches a fresh `solve_one` at the
same point to `atol=1e-8`), `test_bracket_target_returns_none_beyond_traced_extent`,
`test_trace_branch_records_fold_and_multiple_targets_reuse_one_trace`
(single trace on the known-folding fixture, `branch.info["fold_mu"]`
matches the fixture's documented `~0.7808` fold to `5e-3`, and **two**
different targets both resolve correctly off the **same** trace -- the
concrete demonstration that one branch answers multiple map targets instead
of each target re-running its own continuation problem). 3/3 pass.
Full `tests/test_advanced_continuation.py`: 20/20 pass.

Not yet wired into `run_gain_map.py`/`_recover` or `run_fold_follow` --
Milestones A/B add the primitives; production wiring is Milestone G/E per
the plan's own ordering (production semantics before wiring, per the
rollout order in Section 68). `_recover`'s `fold_policy=arclength` branch
and `run_fold_follow` still call `solve_arclength`/`fold_power` directly and
are unaffected by this milestone.

## Milestone C (adaptive `ds`, 2026-08-07) -- DONE

Extracted the main loop's predictor+bordering-corrector block into
`_corrector_step` (`solver.py`, pure refactor -- verified byte-identical
behavior against the full pre-existing test suite before adding anything
new). Added module-level `_adaptive_step_size` implementing Sections 10-11:
Newton-effort factor `q_N = (newton_target / max(used_newton,1)) **
growth_exponent`, clamped to `step_growth_clamp` (default `(0.5, 1.5)`),
then a tangent-angle `theta_k` (computed every accepted step from
`metric_x(Xdot_old, Xdot_new) + lam_dot_old*lam_dot_new`, stored as
telemetry in `info["theta_last_deg"]` regardless of mode) further caps
growth per the plan's threshold table (no special action <15deg, prevent
growth 15-30deg, shrink moderately 30-45deg, shrink aggressively >45deg).
Gated behind `solve_arclength(..., step_control="legacy" | "adaptive")` --
default `"legacy"` reproduces the exact prior "grow 1.25x if used_newton<=3,
cap at ds_initial" rule bit-for-bit (pinned by
`test_solve_arclength_default_step_control_is_legacy`). `ds_max` (default
`None` -> `ds_initial`, matching the legacy ceiling) lets `"adaptive"` mode
grow past the old hardcoded cap once opted in.

Did not implement the plan's "extreme discontinuous jump -> reject/recompute
tangent" row -- the tangent at the accepted point is already the correct one
(computed from that point's own factor, not a stale predictor factor per the
existing code's own established invariant), so there is nothing wrong to
recompute; the >45deg "shrink aggressively" bucket already produces the
intended effect (a much smaller next step) without an extra solve. Documented
deviation, not silently dropped.

Tests: `test_adaptive_step_size_grows_on_easy_corrector_and_shrinks_on_hard`,
`test_adaptive_step_size_curvature_caps_growth_near_a_fold`,
`test_adaptive_step_size_respects_ds_min_and_ds_max` (pure-function unit
tests of `_adaptive_step_size`), `test_solve_arclength_adaptive_reaches_same_solution_as_legacy`
(both modes converge to the identical target point -- adaptive changes the
path, not the physics), `test_solve_arclength_adaptive_ds_max_allows_larger_steps_than_ds_initial`,
`test_solve_arclength_default_step_control_is_legacy`. 6/6 pass.

## Milestone D (fold events, 2026-08-07) -- DONE

Added `_refine_fold` (`solver.py`, reuses `_corrector_step`): secant
refinement per Section 16, predicting from the FIXED anchor point (the last
accepted point before the sign flip, with its own tangent) at successive
secant-estimated arclength offsets toward the zero of the tangent's
lambda-component, narrowing a `(sa, sb)` arclength bracket each iteration.
Stops on `|lambda_dot| < fold_t_tol` or bracket width `< fold_lambda_tol`,
whichever first. Gated behind `solve_arclength(..., refine_fold=True)`
(default `False`, `info["fold_refined"] = None` unless requested). Does
**not** implement Section 17's transversality test (`tau`, the minimally
augmented simple-fold check) -- that is Milestone F's diagnostic-validation
scope per Section 68's own ordering, not this one.

Two bugs found and fixed during validation (both would have silently
produced a non-converging refinement that still looked plausible from the
`fold_lambda` alone):

1. The default `fold_lambda_tol` heuristic was `max(1e-8, |lamc - lam| *
   1e-3)` -- scaled off the lambda displacement across the fold-detecting
   step. That displacement is exactly what goes to zero AT a fold
   (`lambda_dot -> 0` there by definition), so the default tolerance
   degenerated to near-zero right where it needed to be usable. Fixed to
   scale off the arclength step actually taken instead
   (`max(1e-8, step_size * 1e-4)`), which stays well-conditioned regardless
   of how close the bracketing step landed to the true fold.
2. The secant safeguard rejected any estimate landing within 10% of either
   bracket edge, falling back to plain bisection instead. Secant
   legitimately produces estimates close to an edge as the bracket narrows
   toward the root (expected -- that IS convergence), so this safeguard was
   rejecting good steps and replacing them with the ~2x-slower bisection
   rate on every single iteration, discovered when a refinement pinned at
   `iterations=30` (`max_iter` exhausted) despite a well-posed bracket.
   Fixed to only bisect on genuine extrapolation (`s_star` outside `[sa,
   sb]`), which is what a safeguard is for.

With both fixes, the known `_build_folding_problem` fixture (fold at
`lambda~=0.7808`, bracketing accepted points `ds=0.02` apart) refines to a
bracket width `<1e-5` well inside the step grid, converged, matching the
known fold location to `<5e-3`.

Tests: `test_refine_fold_disabled_by_default_leaves_fold_refined_none`,
`test_refine_fold_narrows_bracket_below_the_step_grid`,
`test_refine_fold_tolerances_are_configurable`,
`test_solve_arclength_mu_refine_fold_converts_units` (also exercises
`trace_branch`'s forwarding of the new Milestone C/D kwargs). 4/4 pass.
Full `tests/test_advanced_continuation.py`: 30/30 pass. Full repo suite
(`pytest -p no:cacheprovider`) re-run after both milestones: no regressions
beyond the two pre-existing, unrelated `test_loss_model.py` port-convention
failures already present before this session's changes.

Not yet wired into `fold_power`/`run_fold_follow`/`run_gain_map.py` --
same as Milestone B, production wiring is Milestone E/G territory. The
19-frequency fold-follow sweep in
`docs/development/arclength_fold_resolution_plan.md`'s Phase 4 section (run
in parallel with this implementation work, not using Milestone D) used
`fold_power`'s raw bracket-point `fold_lambda`, not a Milestone-D-refined
root -- 18/19 frequencies show a real fold, -18.6 to -23.0 dBm across
7.6-8.5 GHz.

## Milestone D.5 (local topology of the mu~0.5253 feature, 2026-08-07) -- DONE

Before Milestone E, the pre-E validation campaign (memory
`fold-plan-ad-validation-narrow-feature`) found a step-resolution-dependent
extra tangent-sign flip near mu~0.525 at 7.9 GHz on `designs/ipm_2c_fixed`,
separate from the main mu~0.674 fold. The user's working hypothesis was a
clean local S-bend (`t_mu: + -> - -> +`) that a coarse `ds` steps straight
over -- this milestone tests that hypothesis directly with a standalone
diagnostic (`scripts/validate_fold_plan_d5.py`, does not touch production
code), before committing Milestone E's design to it.

**Detection reproduced the sign-flip pair, but at a much finer resolution
than originally measured:** a local high-resolution forward trace
(`ds=0.005`, restarted from an already-converged anchor at mu=0.49) found
exactly 2 sign-flip events, both within `mu=[0.525275, 0.525283]`ish -- an
extraordinarily narrow feature: `delta_mu_pair = 8.5e-6`,
`delta_s_pair = 1.3e-5` in arclength. A second, even finer forward trace at
`ds=0.0025` found **zero** events in the same window -- exactly the
"coarser step jumps clean over it" pattern repeating one order of magnitude
finer than the original ds=0.02-vs-0.01 measurement. An independent
backward trace (`trace_arclength_from_two_points`, seeded in reverse order,
`ds=0.0025`) found its own pair at `mu=[0.525299, 0.525347]` -- close to but
not pinpoint-matching the forward pair, unlike the main fold's refinement
(which reproduces to 1e-9..1e-11 regardless of direction/step size).

**The multistability probe (fold_plan.md's own decisive test) does NOT
confirm a genuine 3-valued local S-bend.** Three fixed-mu Newton solves at
`mu_star=0.525313` (inside the pair's overlap), seeded from the
before/inside/after segments of the ds=0.005 trace:

| seed | seed mu | converged | relative distance from seg 0 | relative distance from seg 1 |
| --- | ---: | --- | ---: | ---: |
| 0 (before) | 0.525316 | yes | -- | 0.443% |
| 1 (inside) | 0.525275 | yes | 0.443% | -- |
| 2 (after) | 0.525313 | yes | 0.443% | **1.2e-9** |

Segments 1 and 2 converge to the **same root** (1.2e-9 apart -- noise
floor), not two distinct branch sheets. Only segment 0 is materially
different (0.443%). A genuine local fold pair predicts 3 distinct roots;
this measurement gives 2. Corroborating: the first bracket's secant
refinement never converged in 40 iterations (bracket width unchanged at
8.5e-6, unlike every other fold refinement in this plan, which converges in
1-8 iterations to 1e-9..1e-11); `det_sign` does flip cleanly (-1 -> +1)
between the two sampled points (a real sign change happens somewhere in the
window) but `min_eigenvalue` stays negative at both samples rather than
bracketing a clean zero crossing the way the main fold's did.

**Verdict:** the mu~0.5253 event is not the clean, safely-recoverable local
S-bend the pre-E campaign's report hypothesized. It looks like one genuine,
very sharp transition (segment 0 really is a materially different state)
sitting in a region numerically thin enough (`delta_mu_pair` at the scale of
1e-5, comparable to the corrector's own tolerance) that different step
sizes and trace directions report an unstable, non-reproducible extra
wobble in tangent sign around it, rather than a bona fide second fold with
a third coexisting solution branch. Full data:
`D:/tmp/d5_campaign/d5_summary.json`,
`D:/tmp/d5_campaign/d5_followup_summary.json`.

User decision (given this result): Milestone E's fold-pair classification
must be **gated on the multistability probe result itself**, not on sign-flip
count alone -- a confirmed second sign flip is necessary but not sufficient
to call something a safely-recoverable local fold pair.

## Milestone E (production reachability semantics, 2026-08-07) -- DONE

`solve_arclength`/`solve_arclength_mu` already never intentionally stop at a
fold -- they only fail to reach `target_mu` via a genuine corrector failure
(`minimum_step`), the step/deadline budget, or a singular Jacobian; the
"stop at first fold" framing from the original Milestone E sketch was never
actually solve_arclength's behavior. What was missing, per Milestone D.5's
finding, was a way to tell a genuine device pump ceiling apart from a local
branch wiggle the trace already recovered from a few points later -- and to
do that classification using the multistability probe as the actual gate,
not a sign-flip count.

Added to `solver.py`, built directly on Milestone B's `ContinuationBranch`
(pure post-hoc analysis of already-recorded points -- no new continuation
solves needed for detection itself):

- `find_fold_candidates(branch) -> list[FoldCandidate]`: every `t_mu` sign
  change on a traced branch, in traversal order, from consecutive
  `BranchPoint.t_mu` signs already stored. Recovers every candidate, unlike
  `solve_arclength`'s own `info["fold_lambda"]`, which only ever records the
  first. Each `FoldCandidate` also carries `kind="TANGENT_SIGN_FLIP"` and
  `validation="UNVALIDATED"` -- inert placeholders for Milestone F's later
  promotion hierarchy (tangent sign flip -> fold candidate -> validated
  bifurcation event -> classified event); nothing here promotes them.
- `classify_fold_pair(solver, problem, branch, candidate_a, candidate_b,
  ...) -> dict`: the Milestone D.5 multistability probe as a reusable
  primitive. Given a `"+ -> -"` candidate and the following `"- -> +"`
  candidate, picks `mu_star` inside the GENERAL triple-interval overlap
  `[max(min_i), min(max_i)]` of the three segments (before/inside/after --
  works for either orientation, not a shape-specific peak-dip-peak formula),
  seeds one fixed-mu Newton solve (`solve_one`) per segment, and returns
  `status=FOLD_TOPOLOGY_LOCAL_PAIR` only if all three converge AND every
  pairwise relative state distance exceeds `distinctness_tol` (default
  `1e-6`) -- else `status=FOLD_TOPOLOGY_UNRESOLVED` with a `reason` (seed
  non-convergence, no mu overlap, or a collapsed pair).
- `classify_adjacent_fold_candidate_pairs(solver, problem, branch, ...) ->
  list[dict]`: classifies EVERY adjacent pair
  (`(E0,E1),(E1,E2),(E2,E3),...`), not a disjoint two-at-a-time grouping.
  Each result carries `candidate_a_id`/`candidate_b_id` so it traces back to
  its two candidates.

**Three corrections made after the first implementation, per explicit user
review before proceeding to F:**

1. **`FOLD_TOPOLOGY_DEGENERATE` renamed to `FOLD_TOPOLOGY_UNRESOLVED`.** The
   D.5 evidence (2 roots not 3, refinement stalling, location depending on
   direction/step size) shows the region is numerically unresolved at the
   tested settings -- it does NOT establish a mathematically degenerate
   bifurcation (corank/transversality failure), which is untested and is
   Milestone F's scope. The status name must not claim more than the
   evidence supports.
2. **Disjoint pairing (`E0-E1, E2-E3, ...`) replaced with adjacent pairing
   (`E0-E1, E1-E2, E2-E3, ...`)** (`classify_branch_fold_pairs` renamed to
   `classify_adjacent_fold_candidate_pairs`). Disjoint pairing can silently
   miss the meaningful pair: if the real sequence is `[noise, fold A, fold
   B]`, disjoint pairing tests `(noise, foldA)` and leaves `foldB` trailing
   unclassified, when `(foldA, foldB)` is the pair that matters.
3. **`FoldEvent` renamed to `FoldCandidate`, `find_fold_events` to
   `find_fold_candidates`.** A raw tangent-sign flip is not yet evidence of
   a fold (point 1) -- naming it "fold" prejudges the classification
   Milestone F is supposed to do.

Fixing (1) also surfaced a real bug in `classify_fold_pair`'s original
mu-overlap formula (`mu_lo = min(inside), mu_hi = min(max(before),
max(after))`): that formula assumes a peak-dip-peak shape and is WRONG for
the opposite orientation (a middle bump between two lower segments, which
adjacent pairing can now produce at odd-indexed pairs, since consecutive
candidates alternate direction). Replaced with the general triple-interval
overlap `[max(min_i), min(max_i)]` over all three segments, correct
regardless of orientation.

Deliberately NOT built: a live state machine inside `solve_arclength`'s
inner loop (ASCENDING/FOLD_PROBE/etc., as first sketched). The functions
above are a strictly after-the-fact analysis of a completed
`ContinuationBranch` -- cheaper to reason about, cheaper to test (no new
continuation solves for detection), and does not touch the tuned 300-line
corrector loop at all. Also not built: wiring into
`run_gain_map.py`'s map-cell resolution -- same as Milestones B-D, this is a
solver capability; production map-loop consumption is out of this
milestone's scope.

Tests (`tests/test_advanced_continuation.py`, 8, renamed/reworked in place):
`test_find_fold_candidates_detects_all_sign_changes` (also asserts the
placeholder `kind`/`validation` fields), `test_find_fold_candidates_monotone_branch_has_none`,
`test_classify_fold_pair_distinct_roots_is_local_fold_pair`,
`test_classify_fold_pair_collapsed_roots_is_unresolved` (reproduces the D.5
measurement's exact inside==after collapse with a stub solver),
`test_classify_fold_pair_reports_unresolved_when_a_seed_fails_to_converge`,
`test_classify_fold_pair_reports_unresolved_when_no_mu_overlap`,
`test_classify_adjacent_fold_candidate_pairs_tests_every_consecutive_pair`
(3 candidates -> asserts BOTH `(E0,E1)` and `(E1,E2)` are tested, not a
disjoint 1-pair grouping), `test_classify_adjacent_fold_candidate_pairs_empty_when_fewer_than_two_candidates`.
`classify_fold_pair`/`classify_adjacent_fold_candidate_pairs` are tested
against a stub `solve_one` (echoes its seed back as an already-converged
root) rather than a real folding fixture -- the thing under test is the
orchestration (segment selection, `mu_star` placement, pairwise-distance
gating), and a stub gives deterministic control over root distinctness that
a real 1-node fixture cannot (that fixture only has a single, non-multivalued
fold; real-solve coverage for `solve_one`/`solve_arclength` itself is the
existing `test_arclength_*`/`test_refine_fold_*` tests). 38/38
`test_advanced_continuation.py` pass; full repo suite re-run after this
milestone (both before and after the three corrections): 522 passed, 2
pre-existing unrelated `test_loss_model.py` port-convention failures
(confirmed present on a clean stash of this session's changes), 3 skipped
(`--run-slow`), 1 xfail.

### Real-solver validation report (`scripts/validate_fold_plan_e_report.py`, 2026-08-07)

User-requested lightweight follow-up before proceeding to F: run the actual
PRODUCTION `trace_branch`/`find_fold_candidates`/
`classify_adjacent_fold_candidate_pairs` (not a parallel diagnostic
reimplementation) against three already-validated branch settings. No new
simulation campaign -- same settings as D.5/A-D, just calling the real
entry points this time.

**7.9 GHz problematic** (`ds=0.005`, `mu_max=0.60`, 216 points, 230s): 2
candidates, both at the same razor-thin mu~0.5253 feature D.5 found
(`E0: mu=[0.525283,0.525275]`, `E1: mu=[0.525275,0.525278]`). Both
candidates' individual secant refinements still fail to converge in 40
iterations (bracket width unchanged), matching D.5. The `(E0,E1)` adjacent
pair: `status=UNRESOLVED_NEAR_FOLD`, reason `"no mu overlap"` --
**under the corrected general overlap formula, the "before" segment's own
traced coverage does not even reach far enough to overlap "inside"/"after",
so no valid `mu_star` exists and the multistability probe never runs.**
This is a cleaner failure than the original D.5 measurement's own ad-hoc
script got (that used the old shape-specific formula and picked a `mu_star`
slightly outside the "before" segment's actual coverage, which is how it
got as far as running 3 Newton solves before finding only 2 distinct
roots) -- the corrected classifier now refuses to extrapolate rather than
producing a borderline-valid probe.

**7.0 GHz** (`ds=0.02`, `mu_max=1.0`, same settings as the A-D campaign's D7
trace, 221 points, 353s, `terminal=max_steps`): **0 candidates.** The
earlier D7 trace (same settings) found one clean fold near mu~0.7332 after
282 accepted points (budget-extended by `max_steps_after_fold` once
detected); this re-run's step budget was consumed by rejected/retried steps
before reaching that far in mu (no fold-triggered budget extension since
nothing was ever detected) -- ordinary continuation-path variance between
runs, not a false negative on a known-clean region, and specifically NOT
the failure mode being checked for (the classifier did not invent any
candidates from harmless tangent noise here).

**Boring no-fold** (`ds=0.02`, `mu_max=0.3`, same as D8, 24 points, 20s):
**0 candidates, 0 pairs**, as required.

Full log: `D:/tmp/e_real_solver_report_log.txt`.

Milestones A-E are now DONE. Superseded: the original Milestone E sketch
above ("stop at a principal fold below target") assumed `solve_arclength`
needed new stopping behavior; it does not (it never intentionally stops at
a fold in the first place -- see the Milestone E section above). What
Milestone E actually needed, per the D.5 measurement, was a
multistability-gated classifier for what a detected fold-candidate pair
means, which is what got built (then corrected per the epistemic-naming /
adjacent-pairing / candidate-not-fold review above). Remaining, not done:
Milestone F (transversality + harmonic-tail diagnostics -- specifically
`sigma_min(J)`, a left near-null vector `w`, and the transversality quantity
`tau = |w^T R_mu| / (|w| |R_mu|)`, per Section 68's ordering, to actually
classify the still-`UNVALIDATED` candidates as `SIMPLE_FOLD` /
`POSSIBLE_BRANCH_POINT` / `HIGHER_DEGENERACY` / etc.) and wiring
`classify_adjacent_fold_candidate_pairs` into `run_gain_map.py`'s map-cell
resolution (both solver capability and production consumption were kept
separate throughout B-E; explicitly deferred until F tells us what these
candidates actually are -- do not let this logic decide `SKIP_PAST_FOLD` or
reject production gain-map points before then).

---

## Milestone F (Jacobian-level bifurcation classification, 2026-08-08) -- DONE

E answers a kinematic question ("did `t_mu` change sign"); F answers the
algebraic one E deliberately deferred: does an actual singularity of the HB
Jacobian `J=F_X` exist at a `FoldCandidate`, what is its local nullity, and
does it satisfy the two generic simple-fold conditions (parameter
transversality `tau`, quadratic fold coefficient `alpha`)? Per the user's
explicit design: singular values (not eigenvalues) via a left/right
near-null vector pair `(w,v)`, both genuinely computed (not faked from one
side), `sigma1/sigma2` for corank, `tau=|w^T F_mu|/(|w||F_mu|)` for
transversality, and `alpha=w^T F_XX[v,v]` via a centered-difference JVP
(never a dense tensor). No hardcoded universal thresholds -- every tolerance
is a `BifurcationTolerances` field with a documented default, and every
classification row reports the raw diagnostics alongside the label.

### Implementation

`src/twpa_solver/pump/singularity.py` gained `SingularTriplet` +
`smallest_singular_triplets(problem, X, k=2)`: the standard trick of
eigendecomposing the symmetric augmented matrix `[[0,J],[J^T,0]]`
(eigenvalues `+-sigma_i(J)`, eigenvector `[u_i;v_i]` for `+sigma_i` gives
exactly the left/right singular vectors) via ARPACK shift-invert `eigsh`
around `sigma=0`. Forward action is two free sparse matvecs once `J` is an
explicit matrix; inverse action reuses the production factor's own `.solve`
when available (`SchurReducedProblem.assemble_real_coupled_fast`,
PARDISO/banded-backed -- `_assembled_jacobian_matrix_and_solve` dispatches
on `hasattr(problem, "assemble_real_coupled_fast")`, falling back to
`FullPumpProblem.real_coupled_jacobian` for the plain test-fixture path) for
`J^-1`, and a fresh `spla.splu(J^T)` for `J^-T` (no backend exposes a
transpose-solve, so this is the one genuinely new cost -- only paid at
detected fold candidates, never per continuation step). `sigma_ref` (a cheap
power-iteration largest-singular-value estimate, no solves) normalizes
"small" to a dimensionless `sigma1_hat=sigma1/sigma_ref`, since the raw
scale is problem-dependent. Returns `converged=False` (empty
`sigma`/`u`/`v`) on any ARPACK/factorization failure rather than faking a
result -- per the user's explicit instruction not to claim a simple fold
without real evidence.

`src/twpa_solver/pump/bifurcation.py` (new module) consumes
`FoldCandidate`/`ContinuationBranch` (Milestone E) and `SingularTriplet`
(above) to produce the six-way classification:

- `transversality(problem, w) -> (beta, tau, nonzero)`: `beta=-w^T S_ref`
  (source enters linearly, `F_mu=-S_ref`), `tau=|beta|/|S_ref|` (already
  dimensionless since `w` is unit-norm).
- `quadratic_fold_coefficient(problem, X, v, w, state_scale)`: centered
  difference of `J(X+-eps v) v` over a geometric `eps` sweep (never builds
  `F_XX` as a tensor -- only assembles `J` at two nearby points per `eps`),
  requires the middle three sweep values to plateau (relative spread below
  `alpha_plateau_rel_tol`) before reporting `resolved=True`, and gates
  "clearly nonzero" on a self-normalizing SNR (mean-abs / spread across the
  plateau window, `alpha_snr_min`) rather than an absolute magic number,
  since alpha carries no natural unit.
- `local_stencil_profile`/`_build_singularity_profile`: the cheap
  sigma1/sigma2 profile at a small window (`P-2..P+2`) of branch points
  around each candidate -- the expensive `w`/`v`/`tau`/`alpha` work is only
  ever done once, at the profile's sigma1 minimum.
- `cluster_candidates_into_regions`: merges adjacent candidates into one
  `SingularRegion` unless some profiled point strictly between their
  stencil windows has `sigma1_hat >= sigma1_hat_max` (a visible recovery to
  "not singular") -- reuses the same threshold `classify_region`'s first
  gate uses rather than inventing a second unrelated number, and naturally
  merges a razor-thin candidate pair with no profiled points between their
  windows (nothing to prove they are separate).
- `classify_region`: `SIMPLE_FOLD | POSSIBLE_BRANCH_POINT |
  POSSIBLE_HIGHER_DEGENERACY | NONSINGULAR_TANGENT_FLIP |
  SINGULAR_UNCLASSIFIED | DIAGNOSTIC_FAILURE`, short-circuiting through
  `sigma1_hat -> sigma_gap -> tau -> alpha` in increasing cost order, per
  the user's step-8 decision tree exactly (including: alpha unresolved ->
  `SINGULAR_UNCLASSIFIED` (weak evidence), alpha resolved but not clearly
  nonzero -> `POSSIBLE_HIGHER_DEGENERACY` (positive evidence of vanishing
  curvature) -- these are deliberately different outcomes, not the same
  "unknown" bucket).
- `classify_fold_candidates(solver, problem, branch)`: the orchestrator.
  **Event-driven**: zero candidates -> returns `[]` without ever calling
  `smallest_singular_triplets` (verified by a monkeypatch-raises test).
  `bordered_conditioning` (existing `singularity.py` diagnostic) is
  attached as a non-gating comparison field only -- a failed tangent solve
  degrades it to `None`, never blocks a classification.

Tests: `tests/test_singularity.py` (2 new: `smallest_singular_triplets`
shrinks near the toy fixture's known fold vs away from it; left/right
vectors are unit-norm and finite) and `tests/test_bifurcation.py` (12 new:
one test per branch of the six-way decision tree via hand-built
`SingularTriplet`s + monkeypatched `transversality`/`quadratic_fold_coefficient`
so the pure control flow is independent of real linear algebra noise;
`cluster_candidates_into_regions` merge/separate/anchor-selection tests; the
event-driven empty-candidates guard; and one full real-solver integration
test on the 1-DOF near-resonant fixture (`_build_folding_problem`, the same
one `test_advanced_continuation.py`/`test_singularity.py` already use) that
traces a known genuine fold end to end and asserts `classify_fold_candidates`
calls it `SIMPLE_FOLD` -- if F could not do that, per the user's own
instruction, it would not be ready regardless of what it said about 0.525).
57/57 pass across the three touched test files; full suite 536/536 relevant
(2 pre-existing unrelated `test_loss_model.py` failures, unchanged baseline).

### A real bug found and fixed during validation: the alpha eps-scale was inverted

The first `quadratic_fold_coefficient` scaled the finite-difference step as
`eps = eps_base * scale / max(|X|, state_scale)` (divide) -- correct-looking
in isolation, but backwards: since the perturbation direction `v` is already
unit-norm, the ABSOLUTE step must scale WITH the state's own magnitude, not
against it (`eps = eps_base * scale * max(|X|, state_scale)`, multiply).
This was invisible on the toy fixture (`X` there is O(1), so divide and
multiply by `~1` are nearly the same number) but broke completely on
`designs/ipm_2c_fixed`, where node flux is `X ~ 1.76e-14` Wb: dividing gave
`eps ~ 1e6-1e8`, so large that `X +- eps*v` is dominated entirely by
`+-eps*v`, and because the device is unbiased (`dc_branch_flux == 0`,
confirmed) and the branch nonlinearity is `cos(psi/phi0)` (exactly even),
`J(X+eps v)` and `J(X-eps v)` came out bit-identical -- every `alpha_value`
in the sweep was exactly `0.0`, not noisy, at every eps. Fixed by flipping
divide to multiply; verified the toy-fixture `SIMPLE_FOLD` test still passes
(`alpha=-0.303`, `snr=1.75e9`, essentially unaffected as predicted).

### A second "finding" that was WITHDRAWN one day later: alpha was never structurally zero -- it was a matrix-aliasing bug

**Retracted.** The paragraph below this heading originally claimed alpha was
"genuinely unresolvable... a genuine, reproducible measurement, not scale
noise" on `designs/ipm_2c_fixed`, based on a hand-written debug script that
compared `J(X+eps v)` against `J(X-eps v)` across a 20-order-of-magnitude
eps sweep and found them bit-identical every time. **That comparison was
invalid**: both snapshots came from `singularity.py::_assembled_jacobian_matrix`,
which on the production `SchurReducedProblem` path returns
`factor.M` directly from `problem.assemble_real_coupled_fast(tangent)` --
and that factor is a single object the production `FastCoupledPreconditioner`
**caches and mutates in place** across calls (by design, for production's
own single-solve-then-discard usage, correct and cheap there). Two
"snapshots" taken by calling `_assembled_jacobian_matrix` twice in a row
were `Ma is Mb` -- **literally the same Python object** -- so subtracting
them was always exactly zero regardless of `X`. Confirmed directly:
`_assembled_jacobian_matrix(problem, X)` and `_assembled_jacobian_matrix(problem,
X+V)` returned `id()`-identical objects on the real circuit. Fixed by
`.copy()`-ing the matrix on that code path (singularity.py); regression
test `test_assembled_jacobian_matrix_is_not_aliased_across_calls`
(test_singularity.py) reproduces the aliasing with a stub cached factor and
asserts two calls no longer share an object. This also retroactively
explains why the earlier finite-difference `alpha` (Milestone F, before
F.5) came out exactly `0.0` at mu~0.5253: `quadratic_fold_coefficient_fd`'s
eps-sweep loop calls `_assembled_jacobian_matrix` repeatedly and was
comparing the aliased object against itself, not measuring anything about
the device. See Milestone F.5 below for the corrected, exact result.

### Real-solver validation report (`scripts/validate_fold_plan_f_report.py`, 2026-08-08)

Runs the actual production functions (`trace_branch`, `find_fold_candidates`,
`classify_fold_candidates`) against real branches on `designs/ipm_2c_fixed`,
one isolated process per branch (`--which {region_0525,main_fold,freq70,boring}`,
each its own process) -- **the original combined single-process script (all
three branches sequentially) was OOM-killed (exit 137)** on this machine,
almost certainly because the 7.9 GHz `mu_max=0.75` trace crosses the genuine
physical fold near mu~0.674 and burns its whole `max_steps_after_fold`
budget on post-fold recovery (this plan's own prior measurement: "even 150
further steps cannot cross back to target"), while
`smallest_singular_triplets` simultaneously holds a second ~50k x 50k `J^-T`
factorization alongside the production factor's own `J^-1`. Splitting into
separate processes (and shrinking near-fold `max_steps_after_fold` to 20,
since rounding the fold is not needed -- only enough points to bracket the
sign flip) let each run's peak memory release before the next started.

| region | mu | sigma1 | sigma2 | sigma1/sigma2 | tau | alpha (finite-diff, RETRACTED) | bordered_cond | class (SUPERSEDED, see F.5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7.9 GHz E0/1 (mu~0.5253) | 0.525275 | 3247.3 | 2.815e5 | 0.01154 | 0.006612 | n/a (aliasing bug) | 5.572e9 | ~~`SINGULAR_UNCLASSIFIED`~~ -> `SIMPLE_FOLD` |
| 7.9 GHz E0..5 (mu~0.525-0.674, merged) | 0.662673 | 430.2 | 3.612e5 | 0.001191 | 0.002613 | n/a (aliasing bug) | 1.955e9 | ~~`SINGULAR_UNCLASSIFIED`~~ (not rerun with F.5) |
| toy fixture (mu~0.7808, positive control) | 0.7132 | (sigma1_hat=2.6e-4) | -- | 0.0103 | 0.999995 | -0.303 (snr=1.75e9) | 39.3 | **`SIMPLE_FOLD`** |
| boring (mu<=0.3) | -- | -- | -- | -- | -- | -- | -- | 0 regions (event-driven, 0.0s) |

**The `alpha`/class columns for the two real-2c rows above are superseded --
see Milestone F.5 below for the corrected exact result.** What still stands
from this table: both real-2c rows pass the first three gates cleanly --
`sigma1_hat` tiny (`1.66e-9` and `2.21e-10` respectively, far below
`sigma1_hat_max=1e-2`), `sigma_gap` well separated (corank 1 likely), `tau`
clearly nonzero (rules out `POSSIBLE_BRANCH_POINT` at both). mu~0.5253 is a
genuine near-singularity of the exact Jacobian that is not a branch point --
that conclusion is unaffected by the alpha bug below.

**A genuine calibration finding, not yet acted on**: the merged
`E0..5` region reveals that the default `sigma1_hat_max=1e-2` does not
discriminate on this device -- a point picked at random earlier in the same
trace (mu~0.09, far from any candidate) measured `sigma1_hat~1.7e-7`,
already three orders of magnitude below the threshold, so `sigma1_hat` never
recovers to "not singular" under this default anywhere on the branch, and
`cluster_candidates_into_regions` merged all 6 candidates from mu~0.525
through mu~0.674 into one region rather than keeping the mu~0.525 and
mu~0.674 features separate. `sigma_ref` on this device is enormous
(`~2e12`, dominated by the strong linear diagonal terms), so `sigma1_hat`
naturally sits many orders of magnitude below `1e-2` everywhere, healthy
points included; discriminating "singular" from "healthy" here needs a
threshold calibrated against this device's own healthy-point baseline
(~`1e-7`), not the current absolute default. Reported per the user's own
instruction ("we need to see their actual scale on this problem before
locking production criteria") -- not fixed here, since retuning defaults
from one device's data risks overfitting them to `designs/ipm_2c_fixed`
specifically.

**7.0 GHz targeted control at mu~0.7332: not reproduced (3 attempts, this
session).** Exact `D7` settings (`ds=0.02, step_control="adaptive",
mu_max=1.0, max_steps=250, max_steps_after_fold=80, rescale_every=5`) found
the fold once during the A-D campaign (282 points, budget-extended) but 0
candidates on every later rerun at these or nearby settings, including this
session's two attempts (`mu_max=0.78`/`0.75`, `terminal=max_steps` once and
`terminal=target` once -- the second cleanly reached the target mu without
ever registering a sign flip, not merely running out of budget before
reaching it). This is a **pre-existing, already-documented run-to-run path
variance** (memory `fold-plan-d5-e-multistability-gate`, Milestone E's own
report), not something Milestone F introduced or a new blocker -- deferred,
not resolved. The toy-fixture `SIMPLE_FOLD` result above is Milestone F's
positive control instead: a real (not stubbed) solve of a known genuine
fold, correctly classified through all four gates.

Figures: `outputs/fold_plan_milestone_f/sigma1_profile_0p525.png` (mu(s),
t_mu(s), log10(sigma1_hat(s)) over the mu~0.5253 window -- a single smooth
trough in sigma1, consistent with E0/E1 being one Jacobian singularity, not
two, corroborating the multistability-probe-driven merge decision E already
made) and `sigma1_profile_main_fold.png` (same, over the merged 0.525-0.674
window).

## Milestone F.5 (exact AFT quadratic/cubic fold coefficients, 2026-08-08) -- DONE

Milestone F's finite-difference `alpha` was always a stand-in; the user
asked for the exact analytic replacement plus a cubic coefficient, using
the same closed-form AFT machinery the residual/JVP already use (no dense
`F_XX`/`F_XXX` tensor):

- `bifurcation.py::d2n_coeffs`/`d3n_coeffs`: `N(X) = Bphi[Ic sin(psi/phi0)]`,
  `psi(X+eps v)(t) = psi(X)(t) + eps*dpsi_v(t)` (linear in `v`), so
  differentiating twice/thrice at `eps=0` gives closed forms
  `Bphi[-(Ic/phi0^2) sin(psi/phi0) (dpsi_v)^2]` and
  `Bphi[-(Ic/phi0^3) cos(psi/phi0) (dpsi_v)^3]` -- each just one more AFT
  projection (synthesize -> incidence -> nonlinearity -> incidence ->
  project), evaluated at the same perturbation direction `v` twice/thrice.
  `F_XX[v,v] == D2N[X][v,v]` exactly since the linear circuit part has zero
  second derivative. Polymorphic over both problem types via `_ic_phi0`
  (`JosephsonBranchArray.Ic` in the toy test fixture,
  `JosephsonBranchLaw.critical_current` in production -- same formula,
  different field name; `EffectiveSnailBranchLaw`/`CompositeBranchLaw`,
  other designs, are a different nonlinearity and raise rather than
  silently assume the wrong branch law).
- `exact_quadratic_fold_coefficient`/`exact_cubic_fold_coefficient`: `alpha
  = w^T F_XX[v,v]`, `gamma = w^T F_XXX[v,v,v]`, no `eps` anywhere. The
  original finite-difference version survives as `quadratic_fold_coefficient_fd`,
  a validation/cross-check tool only.
- `singularity.py::singular_triplet_residuals`: `r_right = |Jv-sigma
  w|/sigma_ref`, `r_left = |J^Tw-sigma v|/sigma_ref` -- quantifies how much
  to trust `(w,v)` before using `tau` to rule out a branch point, per the
  user's explicit request. Reported, does not change the estimator.
- `bifurcation.py::_local_sigma1_contrast`: `C_sigma = sigma1(candidate) /
  median(sigma1 at 2 nearby healthy branch points, offset +-15 points)` --
  reported for interpretation, deliberately NOT used to recalibrate
  `sigma1_hat_max` (the user's explicit instruction: one device's data
  would overfit the classifier).
- New classification: `QUADRATICALLY_DEGENERATE_SINGULARITY` (alpha_hat ~ 0
  AND residuals clean AND gamma_hat clearly nonzero) -- deliberately not
  named anything with "cusp" in it; cusp classification needs the
  two-parameter continuation, this only reports local one-parameter
  evidence.

**Validation** (`tests/test_bifurcation.py`, 22 tests): exact alpha agrees
with the finite-difference cross-check to `rel=1e-3` at both a known real
fold (toy fixture, lambda~0.7808) and a nonsingular point on the same
branch. One test per branch of the now-seven-way decision tree.

**A real bug caught mid-validation, on the FIRST attempt to exercise F.5 on
the real 2c circuit**: `d2n_coeffs`/`d3n_coeffs` read `problem.branch.Ic`,
which only exists on the toy fixture's `JosephsonBranchArray` -- the
production branch is `JosephsonBranchLaw` (`critical_current` field),
raising `AttributeError` immediately. Fixed by `_ic_phi0`'s polymorphic
lookup. This is exactly why the toy-fixture cross-validation tests, which
all passed, did not catch it: the toy fixture never exercises the
production branch-law class.

**A second, more serious bug caught on the SECOND attempt**: after fixing
the attribute name, the real-2c run succeeded but a hand-written debug
script comparing `_assembled_jacobian_matrix` snapshots at different `X`
found them bit-identical -- which led to the (now-retracted, see the
withdrawn section above) claim that alpha was "genuinely unresolvable" on
this device. That comparison was itself broken: `_assembled_jacobian_matrix`
was returning the SAME cached, in-place-mutated object
(`FastCoupledPreconditioner.M`) on every call, so `Ma is Mb` regardless of
`X`. Fixed by copying the matrix on that code path (singularity.py); this
also explains the original finite-difference alpha's "exactly 0.0 across
20 orders of magnitude of eps" finding from Milestone F -- the FD loop
called `_assembled_jacobian_matrix` repeatedly and was always comparing the
aliased object against itself, never measuring the device.

**Corrected real-2c result at mu~0.5253** (`scripts/validate_fold_plan_f_report.py
--which region_0525`, same branch as before, exact alpha this time):

| quantity | value |
| --- | ---: |
| mu | 0.525275 |
| sigma1_hat | 1.665e-9 |
| sigma1/sigma2 | 0.01154 |
| tau | 0.006612 |
| alpha (exact) | -2.08e21 |
| alpha_hat | -1.066e9 |
| singular residual r_right | 6.948e-18 |
| singular residual r_left | 6.196e-18 |
| sigma1 local contrast (vs 2 healthy neighbors) | 0.0199 (candidate ~50x more singular than its local background) |
| **class** | **`SIMPLE_FOLD`** |

**This reverses Milestone F's own tentative conclusion.** mu~0.5253 is not
`SINGULAR_UNCLASSIFIED` and shows no sign of vanishing curvature -- `alpha`
is enormous and unambiguous (`alpha_hat` nine orders of magnitude above the
`1e-6` gate), and the singular-vector residuals (`~7e-18`) are so far below
`tau=0.0066` that trusting `tau` to rule out a branch point is fully
justified. All four generic simple-fold conditions are satisfied. The
"razor-thin candidate pair, refinement stalls, location depends on
direction/step size" symptoms from D.5 are consistent with an ordinary but
NUMERICALLY NARROW fold (small arclength/mu extent), not a degenerate one --
**the cusp hypothesis this milestone was built to test is not supported by
the corrected data.** Not rerun yet: the merged 0.525-0.674 region (was
also computed with the buggy FD alpha; its sigma1_hat/sigma_gap/tau
conclusions stand, its alpha/class do not).

**Milestones A-F.5 are now DONE.** Per the user's explicit standing
constraint, still not integrated anywhere production-facing:
`classify_fold_candidates`/`classify_adjacent_fold_candidate_pairs` remain a
solver-only capability, never wired into `run_gain_map.py`'s map-cell
resolution or `--fold-policy`. Before any such wiring: (1) calibrate
`sigma1_hat_max`/`sigma_gap_max` against real healthy-vs-candidate data (now
available), (2) get a reproducible 7.0 GHz positive control, (3) rerun the
merged 0.525-0.674 region with exact alpha to close out the one stale row
above. Do not let this logic decide `SKIP_PAST_FOLD` or reject production
gain-map points before then.

## H0-lite (frequency reconnaissance, 2026-08-08) -- INCONCLUSIVE, NOT Hypothesis A

Deliberately scoped-down substitute for the originally-requested two-parameter
bordered fold corrector (`scripts/validate_fold_plan_h0_lite.py`): five
independent cold `region_0525`-recipe traces (`ds=0.005, mu_max=0.60,
max_steps=1200, max_steps_after_fold=300`, `ref_dbm=-16.0`) at
`f_p = 7.85, 7.875, 7.90, 7.925, 7.95` GHz, each its own process (the
Milestone F OOM lesson), picking the classified region nearest `mu~0.5253`
at each frequency.

| f_p GHz | mu_f | sigma1_hat | sigma1/sigma2 | tau | alpha_hat | # candidates | class |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7.850 | 0.558968 | 1.761e-11 | 0.0001406 | 0.009554 | 1.414e+05 | 18 | SIMPLE_FOLD |
| 7.875 | 0.585543 | 6.262e-11 | 0.0005008 | 0.01027 | 3.361e+06 | 12 | SIMPLE_FOLD |
| 7.900 | 0.525275 | 1.665e-09 | 0.01154 | 0.006612 | -1.066e+09 | 2 | SIMPLE_FOLD |
| 7.925 | -- | -- | -- | -- | -- | 0 | no fold in mu<=0.60 |
| 7.950 | -- | -- | -- | -- | -- | 0 | no fold in mu<=0.60 |

**This does not meet the success criterion.** Every found region classifies
`SIMPLE_FOLD` (reinforcing that wherever this feature is found, it is a
genuine one-sided-generic singularity, not something else), but `mu_f(f_p)`
is not smooth or monotone (0.559 -> 0.586 -> 0.525 for 7.85/7.875/7.90), the
candidate count at fixed `ds=0.005` swings 18 -> 12 -> 2 -> 0 -> 0 across a
100 MHz span, and `sigma1_hat`/`alpha_hat` both move 2-4 orders of magnitude
between adjacent 25 MHz steps (`alpha_hat` also changes sign: positive at
7.85/7.875, negative at 7.90). 7.925 and 7.95 GHz have no candidate at all
inside the `mu<=0.60` window traced -- consistent with either the fold
moving past 0.60 or genuinely vanishing; not distinguished here since
extending the window is outside H0-lite's scope.

This is the same symptom `fold_plan.md` Milestone D.5 already documented at
fixed frequency, step-size-dependent location (memory
`fold-plan-ad-validation-narrow-feature`) -- now shown to recur across
frequency at FIXED step size too. It is closer to this plan's Hypothesis C
(a resolution/tracking artifact of a razor-thin feature) than Hypothesis A
(one smooth fold curve). It is not evidence the mu~0.5253 SIMPLE_FOLD
classification itself is wrong (F.5's own point, with clean residuals
`r_right/r_left ~ 7e-18`, stands on its own) -- it is evidence that
"nearest classified region to mu~0.5253" is not a reliable way to track
ONE continuous fold curve across frequency when several tangent-sign-flip
candidates cluster in the same window, exactly the D.5 local-S-bend
scenario repeating itself in a second parameter. Per the user's own
stop condition ("If one nearby frequency suddenly produces no fold or
wildly different classification, that's useful too; investigate only
then") -- this is that case. Stopping here rather than interpreting further
without new instruction. Figure: `outputs/fold_plan_milestone_h0_lite/mu_f_vs_freq.png`.

### Aliasing audit (2026-08-08) -- no further instances found

Per the user's request after the Milestone F.5 matrix-aliasing bug, audited
every place in `singularity.py`/`bifurcation.py` that could retain a
reference to the production `FastCoupledPreconditioner`'s cached, in-place-
mutated `.M` buffer across more than one call (the exact pattern that
produced the bug). Findings: `_assembled_jacobian_matrix_and_solve` (the one
already fixed with `.copy()`) is the ONLY function in either module that
calls `assemble_real_coupled_fast` and returns the resulting matrix to a
caller; every other consumer (`jacobian_min_eigenvalue`,
`jacobian_det_signature`, `bordered_conditioning`,
`smallest_singular_triplets`) uses its `factor`/`.solve` synchronously
within one call and never persists it past that call's return.
`branch.points[i].X` (`solver.py::trace_branch`) already copies
(`np.array(Xc, copy=True)`). `_build_singularity_profile`'s cached
`SingularTriplet`s hold `u`/`v` vectors produced by `array / norm` (a fresh
array each time, never a view onto `factor.M`), so no aliasing there either.
No broad refactor performed; no additional regression tests needed beyond
`test_assembled_jacobian_matrix_is_not_aliased_across_calls`
(`tests/test_singularity.py`), which already covers the one real instance.

## Milestone G0 (single-column production recovery prototype, 2026-08-08) -- DONE, coverage nearly doubled

`scripts/g0_column_recovery.py`. Pump-only, one frequency (7.9 GHz), the
real production power grid (20 points, -26..-16 dBm, matching the CLAUDE.md
"Standard gain-map flag set" range). Two new `InProcessEngine` methods
(`run_gain_map.py`, alongside the existing `solve_power_substep`/
`solve_bridge`): `solve_arclength_forward` (local pseudo-arclength
continuation seeded at a converged state, `i_ref=target_current` so
`solve_arclength_mu` reduces to `solve_arclength` exactly and stops the
instant `target_mu=1.0` is reached -- no separate bracket step needed) and
`solve_frequency_substep` (`solve_power_substep`'s adaptive-step pattern,
but stepping frequency additively at fixed current instead of current
geometrically at fixed frequency).

Baseline = the real production warm-start dispatch (`solve_point`,
fail-fast chained from the last converged point), skip-patience disabled so
every one of the 20 points gets a genuine attempt (patience would otherwise
hide untried points from the coverage count). For each baseline failure,
four tiers in increasing cost order, each seeded from the nearest
CONVERGED point (baseline or a prior recovery in this same pass):

1. previous-power warm-start Newton -- **not re-run**: `solve_direct` (the
   baseline's own "warm" dispatch) is bit-identical to
   `solve_one(problem, last_good_X, 1.0)`, so whenever a warm anchor
   existed the baseline attempt already WAS this tier; re-running it would
   be a deterministic, wasted repeat.
2. `solve_power_substep` (adaptive fixed-frequency power continuation).
3. `solve_arclength_forward` (local pseudo-arclength from the anchor).
4. nearby-frequency same-power anchor search (6 small offsets,
   +-0.005/0.01/0.02 GHz) + `solve_frequency_substep` back to 7.9 GHz.

Every tier that produces a candidate re-runs the REAL `solve_point` so a
"recovery" always means the actual production convergence gate passed, gain
stays skipped exactly as production skips it on a non-converged pump, and
files are written normally.

**Result: baseline 7/20 (35.0%) -> final 13/20 (65.0%) -- all six recoveries
via Tier 2 alone** (`recovered by tier: {'power_substep': 6}`); Tiers 3/4
were never reached because Tier 2 already closed every recoverable gap.
Recovered points cost 4.8-9.5 s each (36.7 s total across the six) --
cheap. The remaining six failures (-19.158 through -16.0 dBm) all report
`tier2_terminal_reason=step_floor` -- the step-independent stall
`solve_power_substep`'s own docstring defines as "a numerical/fold
boundary" -- and cost ~80 s each (all four tiers exhausted) confirming it,
for 595.7 s of the 595.7 s total recovery wall time (baseline itself: 29.4
s). This upper boundary (~-19.4 dBm onset) lines up with the already-
measured genuine fold at mu~0.674 (`I_bound~=1.163e-05 A`, this doc's
arclength-metric section) far better than with the mu~0.5253 fold F.5
classified (~-21.6 dBm) -- independent corroboration, from a completely
different code path, that ~-19.4 dBm and not ~-21.6 dBm is this column's
real high-power wall; the mu~0.5253 fold is passable (production already
recovers past it via plain power substep), the mu~0.674 one is not.

**Runtime verdict**: the recovery ladder essentially doubles usable
coverage for ~37 s of genuinely productive extra work; the ~596 s total
figure is dominated by exhaustively confirming six points are a real wall,
not by the recoveries themselves -- an "acceptable runtime increase" for
the coverage gained, but a production integration should consider a
cheaper early-exit once Tier 2 alone reports `step_floor` twice in a row
(this prototype does not add that -- out of scope, per the milestone's own
instruction not to build new fold-policy integration here).

No new fold classification or two-parameter fold tracking was performed;
`classify_fold_candidates` is not called anywhere in this milestone. Per
the standing constraint, this recovery ladder itself is still **not**
wired into `run_gain_map.py`'s production `--fold-policy`/map path -- G0 is
a prototype/measurement, not an integration.

## Milestone G1 (productionize + boundary cutoff + fold confirmation, 2026-08-08) -- DONE

`scripts/g1_column_recovery.py`. Reuses G0's four tiers and both new
`InProcessEngine` methods unchanged; adds a per-point route label (`DIRECT
| POWER_SUBSTEP | ARCLENGTH_RECOVERY | FREQUENCY_RECOVERY |
FAILED_NUMERICAL | PAST_CONNECTED_BRANCH_BOUNDARY`) and a column-level
failure cache: the first point where the full four-tier ladder is
exhausted becomes a `FAILED_NUMERICAL` **candidate boundary**; every
subsequent point at or above that power runs only a cheap probe (Tier 1,
already free, + Tier 2, ~5-16 s) and is labeled `PAST_CONNECTED_BRANCH_
BOUNDARY` on failure -- Tiers 3/4 are skipped entirely rather than
re-paying ~80-90 s to reconfirm a wall already established.

**Re-run of the 7.9 GHz column with the cutoff**: identical coverage to G0
(13/20, same wall onset -19.158 dBm), but total recovery wall time dropped
**595.7 s -> 181.9 s (3.3x)** -- post-boundary points now cost ~10 s
(cheap probe) instead of ~80 s (full ladder). Confirms the cutoff changes
nothing about which points converge, only how much it costs to find out
they don't.

**Fold classifier at mu~0.674, once** (`scripts/validate_fold_plan_f_report.py
--which main_fold`, now running with the F.5-fixed exact alpha):
`sigma1/sigma2=0.001191`, `tau=0.002613`, `alpha=6.413e+20`
(`alpha_hat=3.288e+08`), singular-vector residuals `r_right=1.6e-16`,
`r_left=3.5e-18` -- **class SIMPLE_FOLD, unambiguous** by the same margins
seen at mu~0.5253. (The classifier's own clustering still merges all six
tangent-sign-flip candidates from mu~0.525 through mu~0.674 into one
region anchored at the sigma1-minimum point, mu=0.6627, at this trace's
`ds=0.005` -- the same "does not cleanly separate two nearby features"
symptom H0-lite already found; the reported diagnostics are for the
deepest/dominant singularity in that merged window, which sits inside the
0.662-0.674 cluster, not the 0.525 one.) Three independent lines of
evidence now agree this is the column's real high-power wall: the
prior PALC turning point near mu~0.674, G0/G1's production recovery
ladder exhausting at ~-19.4 dBm regardless of tier, and now the corrected
Jacobian-singularity classifier itself.

**Three-column test** (7.9 GHz hard reference, 7.7 GHz taken as the "easy"
frequency, 7.0 GHz as the independently-documented problematic one; same
20-point -26..-16 dBm grid at each):

| freq GHz | direct | +substep | +PALC | +freq detour | total runtime | wall onset dBm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.9 | 7/20 (35%) | 13/20 (65%) | 13/20 (65%) | 13/20 (65%) | 209.4 s | -19.158 |
| 7.7 | 10/20 (50%) | 13/20 (65%) | 14/20 (70%) | 15/20 (75%) | 341.5 s | -18.105 |
| 7.0 | 4/20 (20%) | 7/20 (35%) | 7/20 (35%) | 7/20 (35%) | 333.2 s | -22.316 |

**7.7 GHz was not actually "easy"**: Tier 3 fired once (a 53 s local PALC
recovery) and Tier 4 fired once (a 107 s nearby-frequency detour), each
recovering exactly one point neither Tier 2 nor a cheaper tier could reach
-- the initial assumption that Tiers 3/4 would essentially never trigger
(true at 7.9/7.0 GHz, where they fired zero times) does not generalize to
every frequency. Tier 2 remains the dominant mechanism everywhere (6/13,
3/15, 3/7 of each column's total passes), but is not the *whole* story;
keep Tiers 3/4 in the ladder rather than dropping them.

**The wall onset is not smooth/monotone across these three points**
(-22.316, -19.158, -18.105 dBm at 7.0/7.9/7.7 GHz) -- consistent with, but
not yet confirming, the same non-smoothness H0-lite already found for the
mu~0.5253 feature; three points is not enough to characterize
`I_max,connected(f_p)` as an envelope, and this milestone deliberately
does not attempt to (that needs the 5-7 frequency follow-up, not run
here). All three columns DO behave similarly enough on the recovery-
mechanism question (Tier 2 dominant, cutoff logic saves 3x+ regardless of
how much of the column is recoverable) to justify moving forward per the
user's own conditional.

Still no fold classification wired into `run_gain_map.py`'s map path; G1
is a prototype/measurement exactly like G0. The remaining six unrecovered
7.9 GHz points (-19.158 through -16.0 dBm) are, per the user's own
framing, now a different research question (what dynamical state the
circuit occupies above the connected HB branch) rather than a solver
problem -- not pursued in this milestone.

## Milestone G1.5 (junction headroom audit + one resolution check, 2026-08-08) -- DONE, fold is basis-converged

`scripts/g1_5_headroom_audit.py` + `scripts/g1_5_resolution_check.py`. No
new continuation: reconstructs `phi_j(t) = psi_j(t)/phi0`,
`|sin(phi_j(t))|`, `cos(phi_j(t))` from the already-converged G1 pump
solutions (`bifurcation.py::_psi_total_time`/`_ic_phi0`, the same AFT
reconstruction F.5's exact alpha already uses, evaluated once instead of
differentiated), at the highest converged point on each of the three G1
columns:

| freq GHz | I_source/Ic_ref | max\|sin phi\| | max\|phi\| rad | min cos(phi) |
| ---: | ---: | ---: | ---: | ---: |
| 7.0 | 3.25 | 0.416 | 0.429 | 0.909 |
| 7.7 | 4.90 | 0.669 | 0.733 | 0.743 |
| 7.9 | 4.25 | 0.794 | 0.917 | 0.608 |

`I_source/Ic_ref` (source current / median junction critical current) is
2.5-5x -- a naive reading of this alone would call every column absurdly
over-driven, exactly the trap the user's audit was designed to catch: in
this distributed line the port current is not the branch current through
any single junction, so this ratio is not the right "how hard are we
driving it" number. The junction-level numbers are the right ones, and at
every wall they are all comfortably sub-critical: `max|sin(phi)|` peaks at
0.79 (7.9 GHz), `max|phi|` peaks at 0.92 rad against a `pi/2=1.5708` rad
threshold, `min cos(phi)` never drops below 0.61 (nowhere near the
`cos(phi)->0` stiffness-collapse regime). The 7.9 GHz sweep over all 13
converged points (`outputs/fold_plan_milestone_g1_5/headroom_vs_power_
79ghz.png`) shows a smooth, physically ordinary rise from 0.45 to ~0.85 in
`max|sin(phi)|` with increasing power, not a sudden pre-wall spike.
Notably 7.0 GHz's wall (the LOWEST-power wall of the three, -22.316 dBm)
occurs at the LEAST junction stress of the three (`max|sin(phi)|=0.42`) --
this column's wall is the least explicable by local saturation, if
anything.

Per the user's own decision rule, this is squarely the "still looks
under-driven" branch, calling for exactly one resolution experiment (not a
harmonic-resolution campaign): retrace the SAME mu~0.674 local fold at
7.9 GHz (`ds=0.005, mu_max=0.68, max_steps=400, max_steps_after_fold=20`,
identical to the `main_fold` branch config used throughout Milestones F/G1)
at baseline resolution (`--pump-mode-count 10 --nt 40`, production
default, basis_dim=25180) and at richer resolution (`--pump-mode-count 14
--nt 80`, basis_dim=35252 -- 40% more modes, 2x the time-domain samples).

**Result: the fold does not move.** All six tangent-sign-flip candidates
match to 8-9 significant figures between the two resolutions:

| candidate mu | baseline | rich | delta |
| --- | ---: | ---: | ---: |
| (main, largest-mu) | 0.6734293212 | 0.6734293303 | 9.1e-9 |
| | 0.6717483683 | 0.6717484119 | 4.4e-8 |
| | 0.6626787689 | 0.6626787763 | 7.4e-9 |
| | 0.6626779090 | 0.6626779188 | 9.8e-9 |
| | 0.5252792567 | 0.5252792572 | 4.4e-10 |
| | 0.5252766643 | 0.5252766589 | 5.4e-9 |

This is the user's own stated "much harder to dismiss as numerical"
outcome. Combined with F.5's already-established `SIMPLE_FOLD`
classification (a genuine singularity of the equations as solved), the
mu~0.674 fold at 7.9 GHz is now established as **basis-converged, not an
HB-truncation artifact** -- the junction-headroom picture (comfortably
under-driven) does not, on its own, make the fold suspicious once its
location is shown to be independent of resolution at this precision.

**This settles the question G1.5 was raised to answer.** No further
resolution campaign is warranted from this single result; per the user's
own framing this was "exactly one resolution experiment", not the start of
a broader harmonic-convergence study. G2/the pump map is no longer blocked
on this question.

## Milestone G2 (fold traversal experiment, 2026-08-08) -- NOT_TRAVERSED, but not a corrector failure either: a densely-folded region immediately past the known fold

`scripts/g2_fold_traversal.py`. Goal: demonstrate that
`solve_arclength_mu`/`trace_branch` (unmodified) follows the connected
solution manifold continuously through the already-validated
mu~=0.6734293212 SIMPLE_FOLD at 7.9 GHz -- `t_mu>0 -> ~0 at the fold ->
t_mu<0` for a sustained run of accepted post-fold states, not just a touch.

**Solver change (additive, opt-in, zero default-path impact):** added
`step_telemetry` (a callback firing once per accepted PALC step) threaded
through `solve_arclength` -> `solve_arclength_mu` -> `trace_branch`, plus an
optional `gmres_iter_sink` on `_linear_solver`/`_corrector_step` so a caller
can read the true per-step GMRES iteration count. Every one of these
parameters defaults to `None`/unused and every existing test in
`tests/test_advanced_continuation.py` still passes unmodified (42/42); six
new tests added there cover the telemetry payload, the mu-unit conversion
through `solve_arclength_mu`, and forwarding through `trace_branch`
(matching an independent fold location via a `t_lam_own` sign flip).

**Method:** two-stage trace at 7.9 GHz, `--pump-mode-count 10 --nt 40`
(production/basis-converged, [[fold-plan-g1-5-fold-is-basis-converged]]),
`REF_DBM=-16.0` (same `i_ref` convention as F/G1/G1.5, so
`KNOWN_FOLD_MU=0.6734293212367346` applies directly). Stage 1: cheap
uncorrected warmup `mu: 0 -> 0.62` (`ds=0.01`, no telemetry) to get a
trustworthy pre-fold anchor without retracing from `mu=0` at fine
resolution. Stage 2: instrumented trace from that anchor, `mu_max=0.75`,
`ds=0.005`, `step_control=adaptive`, `rescale_every=5`, `refine_fold=True`,
`max_steps_after_fold=200`, `max_steps=350`, `max_wall_s=1200`,
`step_telemetry` capturing HB residual, PALC constraint residual, Newton/
GMRES counts, tangent angle, and both tangent conventions (`t_lam_pred`,
the tangent that predicted the point -- `BranchPoint.t_mu`'s convention --
and `t_lam_own`, the point's own freshly computed tangent) per accepted
step.

**Result: the known fold reproduces almost exactly, but does not lead to a
clean second segment.** First tangent-sign flip: mu=0.6734525 (delta
2.3e-5 from the known value -- excellent independent cross-check from a
materially different numerical path, arriving warm from mu=0.62 rather
than traced fresh from mu=0). Immediately after it, `t_lam_own` goes
sharply negative (-0.98) for ~8 accepted steps, descending only to
mu~=0.6715 (`delta_mu_retreat=0.0021` -- a shallow retreat) before flipping
back positive at step 38 and re-ascending. From there the trace does NOT
settle into either branch: over the remaining ~250 accepted steps
(arclength s=0.08 to 0.24) mu stays confined to a band `[0.671, 0.691]`,
crossing zero **20 more times** (21 total sign flips in the whole run),
many with small `|t_lam_own|` (18 of 289 points have `|t_lam_own|<0.15`)
consistent with the branch dwelling near a near-flat, highly convoluted
local shelf rather than a sequence of well-separated isolated folds. The
trace exhausts its step budget (`terminal_reason=max_steps`) still inside
this band, never reaching `target_mu=0.75`.

**This is not a corrector failure.** Every one of the 289 accepted steps
individually satisfies both convergence conditions to nine-plus digits:
max HB residual (`coeff_rel`) `9.7e-9`, max `|PALC constraint residual|`
`5.4e-17`. The corrector never hits `minimum_step`/`singular_jacobian`/
`deadline`; every predicted point it accepted really is a converged root of
`F(X,mu)=0` on the pseudo-arclength hyperplane. The state-continuity
metric `dX_n = ||X_{n+1}-X_n||_W / max(ds_n, eps)` spikes as high as 12.0
against a median of 0.68 at several points, but every spike coincides with
`ds` having been driven to ~1e-5 to 1e-6 by the adaptive step controller
reacting to local curvature -- consistent with dividing a modest state
step by a tiny denominator near a near-degenerate region, not necessarily
a literal jump to a disconnected solution sheet (distinguishing the two
would need a Milestone-D.5-style multistability probe at several of these
points, explicitly out of scope here: "do not add new fold classifiers").

**Revised verdict: `FIRST_FOLD_TRAVERSED_ENTERED_MULTIFOLD_REGION`**, not
`NOT_TRAVERSED` (user correction, 2026-08-08). The run genuinely crossed
the validated fold (`t_mu: + -> - `at mu=0.6734525, delta 2.3e-5 from the
known value) and continued for 7 further converged accepted states on the
negative side before the next reversal -- that first crossing is a real
traversal, not a touch-and-fail. What did not succeed was escaping the
region beyond it toward `mu=0.75`: after the first reversal the branch
re-ascends and spends the rest of the bounded budget (250 more accepted
steps) inside a `[0.671, 0.691]`-wide band, crossing zero 20 further times,
never reaching the target. `classify_traversal` in
`scripts/g2_fold_traversal.py` now reports this three-way status
(`NOT_TRAVERSED` / `FIRST_FOLD_TRAVERSED_ENTERED_MULTIFOLD_REGION` /
`TRAVERSED`) instead of the original strict binary.

### Predictor-invariance check and a frozen-metric rerun (2026-08-08)

Before accepting "densely multiply-folded region" at face value, the user
flagged a real suspect in the telemetry: `state_scale` changes exactly
every `rescale_every=5` accepted steps, and every single time it does,
the freshly-rescaled `t_lam_pred` lands on exactly `+-1/sqrt(2)`
(58/58 events). This is not numerical coincidence -- it is a provable
consequence of the rescale formula (`solver.py`'s periodic-rescale block
reuses the *same* `state_scale = raw_state_norm/|lam_dot|` construction as
the initial metric derivation, which algebraically forces
`metric_x_new(Xdot,Xdot) == lam_dot**2` whenever that term dominates the
`max()`, so the renormalized `|lam_dot|` is `1/sqrt(2)` on the nose, every
time, regardless of how close to a fold the branch actually is).

**`predictor_invariance()`** (`scripts/g2_fold_traversal.py`, derived
purely from the already-saved CSV -- no rerun needed) checks whether the
rescale preserves the *physical* augmented predictor `ds*(Xdot, lam_dot)`.
Because a rescale event is a pure scalar multiply of the whole tangent
(same direction, new magnitude), the check reduces to a scalar formula
readable straight off consecutive telemetry rows:
`e_pred = |ds_after/(ds_before*rescale_norm) - 1|`, with
`rescale_norm = t_lam_own[i] / t_lam_pred[i+1]`.

**Result: badly violated.** 57 rescale events on the original run; median
`e_pred=0.23` (23% relative error in the physical predictor at a *typical*
rescale), only 19/57 (33%) under 10% error, and at the fold itself (row
29, mu=0.6737, where `|lam_dot|` was naturally tiny at ~0.05 right before
rescale forced it to 0.707): `e_pred=1.67` -- a 167% distortion of the
predictor exactly at the most sensitive point on the branch. This is a
real, confirmed defect in the periodic-rescale implementation: it
renormalizes the tangent under the new metric but never compensates `ds`,
so `ds*t` (the actual predicted displacement) is not preserved across a
rescale, contrary to what the corrector's unit-tangent-under-the-metric
invariant implicitly assumes stays physically meaningful step to step.

**Frozen-metric rerun** (`python scripts/g2_fold_traversal.py
--rescale-every 0`, otherwise identical config/anchor/budget,
`outputs/fold_plan_milestone_g2_frozen_metric/`): same 7.9 GHz circuit,
same mu=0.62 anchor, `rescale_every=0` for the whole traversal stage.

| | adaptive metric (rescale_every=5) | frozen metric (rescale_every=0) |
| --- | ---: | ---: |
| rescale events | 57 | 0 |
| total accepted steps | 289 | 293 |
| total tangent-sign flips | 21 | 27 |
| terminal_reason | max_steps | max_steps |
| final mu | 0.6840 | 0.6876 |
| max dX_weighted / median | 12.04 / 0.68 (17.7x) | 1.39 / 0.95 (1.46x) |
| jump_flag | True | False |
| max HB residual | 9.7e-9 | 9.9e-9 |
| max \|PALC residual\| | 5.4e-17 | 3.6e-17 |

Two separate, cleanly separable findings:

1. **The rescale bug is real and it did locally corrupt the fold crossing.**
   With the metric frozen, the SAME mu~=0.6734 fold (found at flip index
   41, mu=0.673405, delta 2.4e-5 from the known value -- reproduces again)
   is crossed *smoothly*: `t_lam_own` decays gradually
   (-0.153, -0.165, -0.134, -0.094, -0.040) over 5 accepted steps before
   turning, a textbook fold approach. With rescale on, the same physical
   fold was crossed as an abrupt near-discontinuous kink
   (`t_lam_own: -0.050 -> -0.978` in a single accepted step) -- and that
   is exactly what produced the spurious `dX_weighted=12x` "jump" alarm in
   the original run, which vanishes entirely when the metric is frozen
   (max/median ratio drops from 17.7x to 1.46x, essentially flat). The
   rescale bug is worth fixing on its own merits.
2. **But per the user's own Outcome-2 criterion, the densely-folded region
   itself is NOT an artifact of that bug.** With rescale fully disabled,
   the branch still enters a multiply-folded band (comparably wide,
   comparably or even more densely oscillating: 27 flips vs 21, more
   arclength consumed reaching the same `max_steps` budget) and still
   never escapes toward `mu=0.75`. Residuals stay fully converged in both
   runs throughout (HB <1e-8, PALC <1e-16 at every single accepted step in
   both). This is the user's own stated stopping condition: "the same
   folded band remains... stop tuning PALC. The algorithm has done its
   job. The connected periodic branch itself simply wanders through this
   region and doesn't provide a useful route toward mu=0.75."

**Per the task's explicit "bounded budget, do not search indefinitely"
instruction, and now per the user's own Outcome-2 stopping criterion, this
was not extended further, and no further PALC metric/step-control tuning
is planned off the back of this result.** 350 steps / up to 1200s wall
(both runs) already exceeds every prior single-fold trace in this plan.

Outputs: `outputs/fold_plan_milestone_g2/` (adaptive-metric run, 289 rows)
and `outputs/fold_plan_milestone_g2_frozen_metric/` (frozen-metric run,
293 rows), each with `g2_fold_traversal_steps.csv` + 4 plots. Rerun
classification/plots from a saved CSV without resolving via
`python scripts/g2_fold_traversal.py --reanalyze-csv <path>`; rerun the
predictor-invariance check via
`python scripts/g2_fold_traversal.py --check-predictor-invariance <path>`;
rerun the traversal itself at an arbitrary `rescale_every` via
`python scripts/g2_fold_traversal.py --rescale-every <N>`.
