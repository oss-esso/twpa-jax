"""Conservative resource estimates for multitone solves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceEstimate:
    """Estimated peak memory footprint before numerical allocation."""

    coefficient_state_bytes: int
    waveform_bytes: int
    jvp_workspace_bytes: int
    matrix_dimension: int
    predicted_factor_nnz: int
    checkpoint_bytes: int
    total_bytes: int
    preconditioner: str
    n_tones: int
    n_torus: int
    n_retained: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1024**3


class ResourceLimitExceeded(MemoryError):
    """Raised when a requested solve exceeds its configured memory budget."""


def estimate(
    basis: object,
    grid: object,
    n_retained: int,
    n_branches: int,
    preconditioner: str,
) -> ResourceEstimate:
    """Estimate memory from basis/grid dimensions without allocating arrays."""
    n_tones = int(basis.n_tones)
    n_torus = int(basis.n_p) * int(basis.n_delta)
    n_grid_nodes = int(getattr(grid, "n", getattr(grid, "n_nodes", 0)))
    if n_grid_nodes <= 0 or n_retained <= 0 or n_branches < 0:
        raise ValueError("grid nodes, retained nodes, and branch count are invalid")
    coefficient_state = 16 * n_tones * n_retained
    waveform = 8 * n_torus * (n_grid_nodes + n_branches)
    jvp_workspace = 16 * n_tones * n_retained + 8 * n_torus * n_branches
    matrix_dimension = 2 * n_tones * n_retained
    if preconditioner == "floquet_sector":
        tone_orders = getattr(basis, "signal_order", lambda _: 0)
        sector_sizes: dict[int, int] = {}
        for tone in basis.tones:
            order = int(tone_orders(tone))
            sector_sizes[order] = sector_sizes.get(order, 0) + 1
        matrix_dimension = 2 * max(sector_sizes.values()) * n_retained
    elif preconditioner not in {"none", "linear", "mean_tangent", "real_coupled_fast"}:
        raise ValueError(f"unknown preconditioner {preconditioner!r}")
    predicted_factor_nnz = matrix_dimension * max(1, min(matrix_dimension, 2 * n_branches + 3))
    checkpoint = coefficient_state
    total = coefficient_state + waveform + jvp_workspace + 16 * predicted_factor_nnz + checkpoint
    return ResourceEstimate(
        coefficient_state,
        waveform,
        jvp_workspace,
        matrix_dimension,
        predicted_factor_nnz,
        checkpoint,
        total,
        preconditioner,
        n_tones,
        n_torus,
        n_retained,
    )


def guard(resource: ResourceEstimate, budget_gb: float) -> None:
    """Raise before allocation when ``resource`` exceeds ``budget_gb``."""
    if budget_gb <= 0.0:
        raise ValueError("budget_gb must be positive")
    if resource.total_gb > budget_gb:
        raise ResourceLimitExceeded(
            f"estimated multitone memory {resource.total_gb:.3f} GiB exceeds "
            f"budget {budget_gb:.3f} GiB"
        )
