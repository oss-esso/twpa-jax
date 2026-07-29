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
#   scatter map                    ->   94 MB   (4 int32 targets per khat entry)
#   PARDISO factors                ~ 1.85 GB   (~6.5x M.nnz)
#   peak RSS                         2.51 GB
# Identical to 2 decimals for the Schur backend (n=2048, dim 126976), because
# the footprint is dominated by the tone-coupled matrix, not the node count.
# The khat block for a chain circuit carries ~2.4 nnz per retained node, and
# M has 4 real quadrants over an H x H tone grid, so M.nnz ~ 4 H^2 (2.4 n).
_KHAT_NNZ_PER_NODE = 2.4
_BYTES_PER_NNZ = 12.0  # float64 data + int32 index
# Everything that is neither the matrix, the scatter map, nor the factors:
# gamma/khat arrays, AFT waveforms, GMRES vectors, LAPACK workspace. It scales
# with the same H^2 * n as the matrix. The split between this and the fill ratio
# below is not independently measured -- only the two totals are (pypardiso
# 2.51 GB, banded 1.84 GB on jtwpa S=10) -- but the band size is fixed by
# geometry, so those two measurements determine both terms.
_SOLVER_WORKSPACE_BYTES_PER_NNZ = 24.0
# Sparse-LU factor size as a multiple of the matrix bytes, back-solved from the
# pypardiso measurement given the workspace term above. An underestimate
# overcommits workers and drives the machine into swap, while an overestimate
# only leaves a core idle, so round up on ties.
_FACTOR_FILL_RATIO = 5.0
# The scatter map holds four int32 M.data indices per khat entry, i.e. one
# index per real quadrant. Against M.nnz ~ 4 H^2 (khat_nnz) that is exactly
# one int32 per nonzero, so it tracks M at 4/12 of the bytes per nnz. It used
# to be a sparse matrix of +-1 values costing 2x M's bytes; see
# fast_coupled._index_contributions.
_SCATTER_BYTES_PER_MATRIX_NNZ = 4.0
# Assembly stages values into one buffer and scatters them; nothing transiently
# doubles an array the size of the matrix any more, so the peak sits just above
# steady. Rounded up from the measured 2.51/2.42.
_ASSEMBLY_PEAK_OVERHEAD = 1.06


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


# The banded backend stores the factors as a LAPACK general band over the
# node-major reordering instead of a general sparse LU. Measured on jtwpa S=10:
# bandwidth kl=ku=3*(2*n_tones), leading dimension 2*kl+ku+1, one buffer of
# (2*kl+ku+1) * dim float64 factored in place, plus one int32 index per matrix
# nonzero. Peak RSS 1.85 GB against 2.51 GB for pypardiso on the same solve.
_BAND_BLOCKS = 3  # measured band half-width in units of the 2*n_tones block


def _banded_factor_bytes(n_tones: int, n_retained: int, matrix_nnz: int) -> int:
    block = 2 * n_tones
    dimension = block * n_retained
    half_bandwidth = _BAND_BLOCKS * block
    leading = 3 * half_bandwidth + 1
    return leading * dimension * 8 + matrix_nnz * 4


def fast_coupled_footprint(
    n_tones: int,
    n_retained: int,
    *,
    base_bytes: int = 200 * 1024**2,
    factor_backend: str = "pardiso",
) -> FastCoupledFootprint:
    """Estimate peak RSS of one ``real_coupled_fast`` multitone solve.

    ``base_bytes`` covers the interpreter, numpy/scipy, and the circuit itself.
    The peak is what matters for deciding worker counts, and it is dominated by
    the sparse factors -- everything else scales with the same ``H^2`` but at a
    far smaller constant.
    """
    if n_tones <= 0 or n_retained <= 0:
        raise ValueError("n_tones and n_retained must be positive")
    if factor_backend not in ("pardiso", "banded"):
        raise ValueError(f"unknown factor backend: {factor_backend!r}")
    matrix_nnz = int(4 * n_tones**2 * _KHAT_NNZ_PER_NODE * n_retained)
    matrix_bytes = int(matrix_nnz * _BYTES_PER_NNZ)
    scatter_bytes = int(matrix_nnz * _SCATTER_BYTES_PER_MATRIX_NNZ)
    if factor_backend == "banded":
        factor_bytes = _banded_factor_bytes(n_tones, n_retained, matrix_nnz)
    else:
        factor_bytes = int(matrix_nnz * _FACTOR_FILL_RATIO * _BYTES_PER_NNZ)
    workspace_bytes = int(matrix_nnz * _SOLVER_WORKSPACE_BYTES_PER_NNZ)
    steady = (
        base_bytes + matrix_bytes + scatter_bytes + factor_bytes + workspace_bytes
    )
    peak = int(steady * _ASSEMBLY_PEAK_OVERHEAD)
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
