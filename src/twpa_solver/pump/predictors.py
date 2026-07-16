"""Inter-cell state predictors for the pump/gain map warm pass.

A predictor turns one or more already-converged neighbour states into an initial
guess ``X`` (complex harmonic coefficients, shape ``(H, n)``) for a new map cell,
so the target Newton solve starts closer to the root. Physics is untouched: only
the initial guess changes.

All functions are pure and operate on ``numpy`` complex arrays. They return
``None`` when the required neighbours are missing or shapes disagree, so the
caller can fall back to a plain copy / seed. The residual-ranked portfolio
(``rank_candidates``) takes a caller-supplied residual evaluator so this module
stays free of any solver/problem dependency.

Predictor catalogue (see ``docs/reports/pump_map_continuation_methods.tex`` and
the campaign test matrix):

- ``copy``            : zeroth-order, X_pred = X_parent.
- ``axis_secant``     : 1st-order extrapolation along a scalar abscissa
                        (pump current for the power axis, frequency for the
                        frequency axis).
- ``corner``          : 2-D affine, X[i,j] = X[i-1,j] + X[i,j-1] - X[i-1,j-1].
- ``plane``           : local least-squares plane X(P,f) ~ a0 + aP dP + af df.
- ``rank_candidates`` : residual-ranked portfolio over any of the above.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _shapes_match(*arrays: np.ndarray | None) -> bool:
    """True iff every argument is a non-None array sharing one shape."""
    shape = None
    for a in arrays:
        if a is None:
            return False
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            return False
    return shape is not None


def copy_predictor(parent: np.ndarray) -> np.ndarray:
    """Zeroth-order predictor: copy the parent state."""
    guess = np.array(parent, dtype=np.complex128, copy=True)
    logger.debug("predictor_copy shape=%s", guess.shape)
    return guess


def axis_secant(
    x_prev2: np.ndarray | None,
    x_prev1: np.ndarray | None,
    t_prev2: float | None,
    t_prev1: float | None,
    t_target: float,
) -> np.ndarray | None:
    """Linear extrapolation of the state along a scalar abscissa ``t``.

    ``x_prev1`` sits at ``t_prev1`` and ``x_prev2`` at ``t_prev2`` (the two most
    recent converged states along one axis). Predict at ``t_target``:

        X_pred = x_prev1 + beta * (x_prev1 - x_prev2),
        beta   = (t_target - t_prev1) / (t_prev1 - t_prev2).

    ``t`` is the natural continuation coordinate for that axis: the injected
    pump current for the power axis (source is linear in it) or the pump
    frequency for the frequency axis. Returns ``None`` if inputs are missing,
    shapes disagree, or the abscissa is degenerate.
    """
    if not _shapes_match(x_prev2, x_prev1):
        logger.debug("predictor_axis_secant_unavailable reason=shape_mismatch")
        return None
    if t_prev2 is None or t_prev1 is None:
        logger.debug("predictor_axis_secant_unavailable reason=missing_abscissa")
        return None
    denom = t_prev1 - t_prev2
    if abs(denom) < 1e-30:
        logger.debug("predictor_axis_secant_unavailable reason=degenerate_abscissa")
        return None
    beta = (t_target - t_prev1) / denom
    guess = x_prev1 + beta * (x_prev1 - x_prev2)
    logger.debug("predictor_axis_secant_complete beta=%s shape=%s", beta, guess.shape)
    return guess


def corner_predictor(
    x_i_jm1: np.ndarray | None,
    x_im1_j: np.ndarray | None,
    x_im1_jm1: np.ndarray | None,
) -> np.ndarray | None:
    """2-D affine (corner) predictor.

    Given the three grid neighbours below, left, and diagonal of the target
    cell ``(i, j)``:

        X[i, j] ~ X[i-1, j] + X[i, j-1] - X[i-1, j-1].

    Exact for any state that is affine in the two grid indices. ``x_i_jm1`` is
    the same-power / previous-frequency neighbour, ``x_im1_j`` the
    previous-power / same-frequency neighbour, ``x_im1_jm1`` the diagonal.
    Returns ``None`` if any neighbour is missing or shapes disagree.
    """
    if not _shapes_match(x_i_jm1, x_im1_j, x_im1_jm1):
        logger.debug("predictor_corner_unavailable reason=shape_mismatch")
        return None
    guess = x_im1_j + x_i_jm1 - x_im1_jm1
    logger.debug("predictor_corner_complete shape=%s", guess.shape)
    return guess


def plane_predictor(
    samples: list[tuple[float, float, np.ndarray]],
    p_target: float,
    f_target: float,
    *,
    min_samples: int = 3,
) -> np.ndarray | None:
    """Local least-squares plane predictor.

    Fit each complex coefficient independently as an affine function of the two
    physical parameters ``(P, f)`` from nearby converged cells:

        X(P, f) ~ a0 + aP * (P - P_target) + af * (f - f_target),

    so the prediction at the target is simply ``a0``. Needs at least
    ``min_samples`` (>= 3) samples that all share one state shape and span a
    non-degenerate ``(P, f)`` plane. Returns ``None`` otherwise (caller falls
    back to a lower-order predictor).
    """
    usable = [(p, f, x) for (p, f, x) in samples if x is not None]
    if len(usable) < max(3, min_samples):
        logger.debug("predictor_plane_unavailable reason=insufficient_samples n=%d", len(usable))
        return None
    shape = usable[0][2].shape
    if any(x.shape != shape for _, _, x in usable):
        logger.debug("predictor_plane_unavailable reason=shape_mismatch")
        return None

    m = len(usable)
    design = np.empty((m, 3), dtype=float)
    rhs = np.empty((m, int(np.prod(shape))), dtype=np.complex128)
    for row, (p, f, x) in enumerate(usable):
        design[row, 0] = 1.0
        design[row, 1] = float(p) - float(p_target)
        design[row, 2] = float(f) - float(f_target)
        rhs[row, :] = np.asarray(x, dtype=np.complex128).ravel()

    # Reject a rank-deficient design (all samples on one P or one f line): the
    # plane is under-determined and the fit would be meaningless.
    if np.linalg.matrix_rank(design, tol=1e-9) < 3:
        logger.debug("predictor_plane_unavailable reason=rank_deficient")
        return None

    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    a0 = coeffs[0, :]  # value at the target (design centred on target)
    guess = a0.reshape(shape).astype(np.complex128, copy=False)
    logger.debug("predictor_plane_complete samples=%d shape=%s", len(usable), guess.shape)
    return guess


def rank_candidates(
    candidates: dict[str, np.ndarray | None],
    residual_fn: Callable[[np.ndarray], float],
) -> list[tuple[str, np.ndarray, float]]:
    """Rank predictor candidates by target residual, ascending.

    ``candidates`` maps a predictor name to its guess (``None`` entries are
    dropped). ``residual_fn(X)`` returns a scalar residual norm of ``X`` at the
    target operating point (e.g. ``problem.norms(X, 1.0)["coeff_rel"]``). The
    returned list is sorted so the lowest-residual (best) candidate is first.
    Non-finite residuals sort last.
    """
    scored: list[tuple[str, np.ndarray, float]] = []
    for name, guess in candidates.items():
        if guess is None:
            continue
        try:
            rho = float(residual_fn(guess))
        except (ValueError, FloatingPointError):
            rho = float("inf")
        if not np.isfinite(rho):
            rho = float("inf")
        scored.append((name, guess, rho))
    scored.sort(key=lambda item: item[2])
    logger.debug("predictor_candidates_ranked candidates=%r", [(n, r) for n, _, r in scored])
    return scored
