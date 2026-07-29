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


# Calibration for the ``real_coupled_fast`` path, measured on jtwpa S=10
# (n_tones=31, n=2560, n_branches=2559, packed dim 158720) with pypardiso:
#   M.nnz            23,705,080   ->  284 MB   (12 B/nnz: float64 + int32)
#   W.nnz            47,219,696   ->  567 MB   (W.nnz == 2 * M.nnz)
#   PARDISO factors                ~ 1.85 GB   (~6.5x M.nnz)
#   steady RSS                       2.80 GB
#   peak RSS                         3.01 GB   (transient COO build of W)
# Identical to 2 decimals for the Schur backend (n=2048, dim 126976), because
# the footprint is dominated by the tone-coupled matrix, not the node count.
# The khat block for a chain circuit carries ~2.4 nnz per retained node, and
# M has 4 real quadrants over an H x H tone grid, so M.nnz ~ 4 H^2 (2.4 n).
_KHAT_NNZ_PER_NODE = 2.4
_BYTES_PER_NNZ = 12.0  # float64 data + int32 index
# Rounded UP from the measured ~6.5: an underestimate overcommits workers and
# drives the machine into swap, while an overestimate only leaves a core idle.
_FACTOR_FILL_RATIO = 7.0
# W is built from preallocated int32 row/col plus float64 value triplets and
# then compressed to CSR, so the build transiently costs a fraction of the
# final array on top of it. Rounded up from the measured (3.01-2.80)/0.567.
_SCATTER_BUILD_OVERHEAD = 1.5


@dataclass(frozen=True)
class FastCoupledFootprint:
    """Measured-calibrated peak memory for one ``real_coupled_fast`` solve."""

    matrix_bytes: int
    scatter_bytes: int
    factor_bytes: int
    steady_bytes: int
    peak_bytes: int
    matrix_nnz: int
    matrix_dimension: int

    @property
    def steady_gb(self) -> float:
        return self.steady_bytes / 1024**3

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1024**3


def fast_coupled_footprint(
    n_tones: int, n_retained: int, *, base_bytes: int = 200 * 1024**2
) -> FastCoupledFootprint:
    """Estimate peak RSS of one ``real_coupled_fast`` multitone solve.

    ``base_bytes`` covers the interpreter, numpy/scipy, and the circuit itself.
    The transient peak is what matters for deciding worker counts: the scatter
    map is built from int64 COO triplets before being compressed to CSR, so a
    worker briefly needs well above its steady footprint.
    """
    if n_tones <= 0 or n_retained <= 0:
        raise ValueError("n_tones and n_retained must be positive")
    matrix_nnz = int(4 * n_tones**2 * _KHAT_NNZ_PER_NODE * n_retained)
    matrix_bytes = int(matrix_nnz * _BYTES_PER_NNZ)
    scatter_bytes = 2 * matrix_bytes
    factor_bytes = int(matrix_nnz * _FACTOR_FILL_RATIO * _BYTES_PER_NNZ)
    steady = base_bytes + matrix_bytes + scatter_bytes + factor_bytes
    # The W build transiently holds int64 COO triplets for the same nnz count.
    peak = steady + int(scatter_bytes * (_SCATTER_BUILD_OVERHEAD - 1.0))
    return FastCoupledFootprint(
        matrix_bytes=matrix_bytes,
        scatter_bytes=scatter_bytes,
        factor_bytes=factor_bytes,
        steady_bytes=steady,
        peak_bytes=peak,
        matrix_nnz=matrix_nnz,
        matrix_dimension=2 * n_tones * n_retained,
    )


def available_memory_gb() -> float | None:
    """Return free physical memory in GiB, or ``None`` if it cannot be read."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / 1024**3
    except (OSError, AttributeError, ValueError):
        return None


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
    n_grid_nodes = int(
        getattr(grid, "n", getattr(grid, "n_nodes", n_retained))
    )
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
    elif preconditioner not in {
        "none",
        "linear",
        "mean_tangent",
        "spectral_coupled",
        "real_coupled_fast",
    }:
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
