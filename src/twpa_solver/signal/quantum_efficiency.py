from __future__ import annotations

import numpy as np


def calc_qe(s: np.ndarray, s_noise: np.ndarray | None = None) -> np.ndarray:
    """Quantum efficiency matrix for a scattering matrix in the field ladder
    operator basis.

    Port of JosephsonCircuits.jl's `calcqe`/`calcqe!`. Row i is normalized by
    the total power leaving output mode i across every input mode in `s`
    (plus every noise mode in `s_noise`, if given): `qe[i, j] = |s[i, j]|^2 /
    sum_k(|s[i, k]|^2 + |s_noise[i, k]|^2)`.

    Unlike the Julia source, this is a single vectorized function: numpy's
    row-sum already gives the cache-efficient behavior the Julia `calcqe!`
    loop was written to obtain by hand, so the `!`-suffixed in-place variant
    and the two-pass denominator loop are not ported separately.

    Args:
        s: Scattering matrix, shape (n_out, n_in).
        s_noise: Additional noise scattering matrix sharing `s`'s output
            dimension, shape (n_out, n_noise). Its power is added to the
            denominator only.

    Returns:
        Real-valued quantum efficiency matrix, same shape as `s`.

    Raises:
        ValueError: If `s_noise` is given and its output dimension does not
            match `s`'s.
    """
    s = np.asarray(s)
    denom = np.sum(np.abs(s) ** 2, axis=1)

    if s_noise is not None:
        s_noise = np.asarray(s_noise)
        if s_noise.shape[0] != s.shape[0]:
            raise ValueError(
                f"s_noise output dimension {s_noise.shape[0]} != s output "
                f"dimension {s.shape[0]}"
            )
        denom = denom + np.sum(np.abs(s_noise) ** 2, axis=1)

    return np.abs(s) ** 2 / denom[:, None]


def calc_qe_ideal(s: np.ndarray) -> np.ndarray:
    """Ideal (best possible) quantum efficiency for each element of a
    scattering matrix.

    Port of JosephsonCircuits.jl's `calcqeideal`/`calcqeideal!`.

    Args:
        s: Scattering matrix.

    Returns:
        Real-valued ideal quantum efficiency, same shape as `s`: 1.0 where
        `|s| <= 1`, else `1 / (2 - 1 / |s|^2)`.
    """
    s = np.asarray(s)
    abs2_s = np.abs(s) ** 2
    return np.where(abs2_s <= 1.0, 1.0, 1.0 / (2.0 - 1.0 / abs2_s))
