# experiments/pump_basis.py
"""Pump-mode basis policies shared by exp08 (pump solve) and exp09 (gain).

A "pump basis" is the explicit list of positive integer pump-harmonic indices
used to represent the physical real pump waveform via the JosephsonCircuits.jl
(JC) convention:

    psi_pump(t) = real_reconstruction_factor * Re sum_{k in modes} X_k exp(+i k omega_p t)

with real_reconstruction_factor = 2. This is identical to the legacy dense
behavior when modes = [1, 2, ..., H]; the only generalization is that `modes`
may be an arbitrary sorted set of distinct positive integers, e.g. the JC odd
mode list [1, 3, 5, ..., 2K-1].

Policies
--------
dense_real
    Legacy behavior. modes = [1, 2, ..., H].
positive_phasor_explicit
    Explicit positive pump modes from --pump-modes.
positive_odd_jc
    JC odd mode list: modes = [1, 3, 5, ..., 2K-1] for K = --pump-mode-count.
auto_jc
    Choose the mode list from design metadata when available, else from
    symmetry rules (single pump, unbiased 4WM -> positive odd; multi-pump ->
    unsupported with a clear reason).

All four policies share the same positive-phasor real reconstruction. The
`policy` string is recorded so downstream tooling can reproduce the choice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

POLICIES = (
    "dense_real",
    "positive_phasor_explicit",
    "positive_odd_jc",
    "auto_jc",
)
PHASE_CONVENTION = "exp_plus_i_k_omega_t"
REAL_RECONSTRUCTION_FACTOR = 2
BASIS_NAME = "positive_phasor"


@dataclass
class PumpBasis:
    """Resolved pump-mode basis plus the metadata needed to reconstruct it."""

    modes: list[int]
    policy: str
    omega_p: float
    basis: str = BASIS_NAME
    real_reconstruction_factor: int = REAL_RECONSTRUCTION_FACTOR
    phase_convention: str = PHASE_CONVENTION
    source_mode: int = 1

    def __post_init__(self) -> None:
        self.modes = [int(m) for m in self.modes]
        if len(self.modes) == 0:
            raise ValueError("pump basis needs at least one mode")
        if any(m < 1 for m in self.modes):
            raise ValueError(f"all pump modes must be >= 1, got {self.modes}")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError(f"duplicate pump modes: {self.modes}")
        self.modes = sorted(self.modes)
        self.source_mode = int(self.source_mode)
        if self.source_mode not in self.modes:
            raise ValueError(
                f"source_mode {self.source_mode} not in modes {self.modes}"
            )

    @property
    def k(self) -> np.ndarray:
        """Mode indices as float array (matches the harmonic grid convention)."""
        return np.asarray(self.modes, dtype=float)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def max_mode(self) -> int:
        return max(self.modes)

    @property
    def source_row(self) -> int:
        return self.modes.index(self.source_mode)

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "pump_modes": list(self.modes),
            "pump_basis": self.basis,
            "real_reconstruction_factor": self.real_reconstruction_factor,
            "omega_p": self.omega_p,
            "phase_convention": self.phase_convention,
            "pump_mode_policy": self.policy,
            "pump_source_mode": self.source_mode,
        }
        logger.debug(
            "pump_basis_to_metadata n_modes=%s policy=%s omega_p=%s "
            "phase_convention=%s source_mode=%s",
            self.n_modes, self.policy, self.omega_p,
            self.phase_convention, self.source_mode,
        )
        return metadata


def positive_odd_modes(k: int) -> list[int]:
    """Return [1, 3, 5, ..., 2k-1] (the JC odd pump mode list)."""
    if k < 1:
        logger.debug("positive_odd_modes_invalid k=%s", k)
        raise ValueError(f"--pump-mode-count must be >= 1, got {k}")
    modes = [2 * i - 1 for i in range(1, k + 1)]
    logger.debug("positive_odd_modes k=%s n_modes=%s max_mode=%s", k, len(modes), modes[-1])
    return modes


def parse_explicit_modes(spec: str) -> list[int]:
    """Parse a comma-separated mode list like '1,3,5,7'."""
    toks = [t for t in spec.replace(" ", "").split(",") if t != ""]
    if not toks:
        logger.debug("parse_explicit_modes_invalid spec=%r", spec)
        raise ValueError(f"could not parse --pump-modes from {spec!r}")
    modes = [int(t) for t in toks]
    logger.debug("parse_explicit_modes spec=%r n_modes=%s", spec, len(modes))
    return modes


def _unwrap_design_meta(design_meta: dict[str, Any]) -> dict[str, Any]:
    """Return the innermost design metadata block used for basis decisions."""
    meta = design_meta
    decision_keys = {
        "features",
        "pump_modes",
        "pump_sources",
        "Nmodulationharmonics",
        "Npumpharmonics",
    }

    while isinstance(meta, dict):
        if decision_keys & set(meta):
            return meta
        nested = meta.get("metadata")
        if not isinstance(nested, dict) or nested is meta:
            return meta
        meta = nested

    return {}


def _modes_from_design_meta(
    design_meta: dict[str, Any],
    *,
    harmonics: int,
) -> list[int] | None:
    """Try to read an explicit positive pump mode list from design metadata.

    Returns None if the design does not record enough to derive a scalar
    positive mode list. Raises for multi-pump designs that need multi-index
    pump-mode tuples (not representable by a scalar mode policy).
    """
    meta = _unwrap_design_meta(design_meta)
    features = meta.get("features", {})

    if features.get("multi_pump", False):
        raise ValueError(
            "auto_jc: multi-pump design needs multi-index pump-mode tuples; "
            "not supported by the scalar positive-phasor mode policy"
        )

    # An explicit positive integer mode list, if present, wins.
    explicit = meta.get("pump_modes")
    if explicit:
        flat = [int(m) for m in explicit if int(m) >= 1]
        if flat:
            return sorted(set(flat))

    if features.get("needs_dc", False):
        return list(range(1, harmonics + 1))

    pump_sources = meta.get("pump_sources", [])
    for src in pump_sources if isinstance(pump_sources, list) else []:
        if not isinstance(src, dict):
            continue
        mode = src.get("mode")
        modes = mode if isinstance(mode, (list, tuple)) else [mode]
        if any(int(m) == 0 for m in modes if m is not None):
            return list(range(1, harmonics + 1))

    # JC records Nmodulationharmonics; for an unbiased single-pump 4WM design
    # the nonlinear pump modes are the odd list [1, 3, ..., 2K-1].
    nmod = meta.get("Nmodulationharmonics")
    if nmod:
        k = int(nmod[0] if isinstance(nmod, (list, tuple)) else nmod)
        if k >= 1:
            return positive_odd_modes(k)

    return None


def resolve_pump_basis(
    *,
    policy: str,
    omega_p: float,
    harmonics: int,
    mode_count: int | None,
    explicit_modes: str | None,
    design_meta: dict[str, Any] | None = None,
    source_mode: int = 1,
) -> PumpBasis:
    """Resolve CLI options + design metadata into a concrete PumpBasis."""
    logger.debug(
        "pump_basis_resolve_start policy=%s omega_p=%s harmonics=%s mode_count=%s "
        "explicit_modes=%r",
        policy, omega_p, harmonics, mode_count, explicit_modes,
    )
    if policy not in POLICIES:
        raise ValueError(f"unknown pump-mode-policy {policy!r}, choose from {POLICIES}")

    if policy == "dense_real":
        modes = list(range(1, harmonics + 1))

    elif policy == "positive_odd_jc":
        k = mode_count if mode_count is not None else harmonics
        modes = positive_odd_modes(k)

    elif policy == "positive_phasor_explicit":
        if not explicit_modes:
            raise ValueError(
                "positive_phasor_explicit requires --pump-modes '1,3,5,...'"
            )
        modes = parse_explicit_modes(explicit_modes)

    elif policy == "auto_jc":
        modes = None
        if explicit_modes:
            modes = parse_explicit_modes(explicit_modes)
        elif design_meta is not None:
            modes = _modes_from_design_meta(design_meta, harmonics=harmonics)
        if modes is None:
            if mode_count is not None:
                modes = positive_odd_modes(mode_count)
            else:
                # Symmetry default for an unbiased single-pump 4WM design.
                modes = positive_odd_modes(harmonics)
    else:  # pragma: no cover - guarded above
        raise ValueError(policy)

    basis = PumpBasis(
        modes=modes,
        policy=policy,
        omega_p=omega_p,
        source_mode=source_mode,
    )
    logger.debug("pump_basis_resolve_complete policy=%s modes=%r source_mode=%s", policy, basis.modes, source_mode)
    return basis


def load_pump_basis_from_solution(
    pump_dir: str | Path,
    fallback_omega_p: float | None = None,
) -> tuple[np.ndarray, PumpBasis]:
    """Load a saved pump solution and reconstruct its PumpBasis.

    Returns (X, basis) where X has shape (n_modes, n_nodes) complex, row i
    corresponding to basis.modes[i]. Works for both legacy dense solutions
    (which only stored a `harmonics` array) and new mode-aware solutions.
    """
    d = Path(pump_dir)
    logger.debug("pump_basis_load_start pump_dir=%s", d)
    sol_path = d / "pump_solution.npz"
    if not sol_path.exists():
        raise FileNotFoundError(f"missing pump solution: {sol_path}")

    sol = np.load(sol_path)
    # Force float64/complex128: solutions may be stored float32 to save space,
    # and float32 would otherwise promote to complex64 and infect scipy solves.
    X = np.asarray(sol["X_real"], dtype=np.float64) + 1j * np.asarray(
        sol["X_imag"], dtype=np.float64
    )

    metadata: dict[str, Any] = {}
    report_path = d / "pump_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            metadata = json.load(f).get("metadata", {})

    if "pump_modes" in sol.files:
        modes = [int(m) for m in np.asarray(sol["pump_modes"]).ravel()]
    elif metadata.get("pump_modes"):
        modes = [int(m) for m in metadata["pump_modes"]]
    elif "harmonics" in sol.files:
        modes = [int(m) for m in np.asarray(sol["harmonics"]).ravel()]
    else:
        modes = list(range(1, X.shape[0] + 1))

    if len(modes) != X.shape[0]:
        raise ValueError(
            f"pump solution has {X.shape[0]} rows but {len(modes)} modes in {d}"
        )

    omega_p = float(metadata.get("omega_p", fallback_omega_p or 0.0))
    policy = str(metadata.get("pump_mode_policy", "dense_real"))
    source_mode = int(metadata.get("pump_source_mode", min(modes)))

    basis = PumpBasis(
        modes=modes,
        policy=policy,
        omega_p=omega_p,
        source_mode=source_mode,
    )
    logger.debug("pump_basis_load_complete shape=%s modes=%r omega_p=%s", X.shape, basis.modes, basis.omega_p)
    return X, basis


def promote_solution_to_basis(
    X_src: np.ndarray,
    src_basis: PumpBasis,
    dst_basis: PumpBasis,
) -> np.ndarray:
    """Warm-start: map a lower-basis pump solution into a richer basis.

    Rows whose mode index exists in both bases are copied; new modes are zero.
    """
    n_nodes = X_src.shape[1]
    X_dst = np.zeros((dst_basis.n_modes, n_nodes), dtype=np.complex128)
    src_index = {m: i for i, m in enumerate(src_basis.modes)}
    for j, m in enumerate(dst_basis.modes):
        if m in src_index:
            X_dst[j] = X_src[src_index[m]]
    logger.debug("pump_basis_promote_complete src_modes=%r dst_modes=%r shape=%s", src_basis.modes, dst_basis.modes, X_dst.shape)
    return X_dst
