"""Residual builders for nonlinear TWPA solves."""

from __future__ import annotations

from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.jax_aft_hb import JaxPumpAFTResidual

__all__ = ["JaxPumpAFTResidual", "PumpAFTConfig", "PumpAFTResidual"]
