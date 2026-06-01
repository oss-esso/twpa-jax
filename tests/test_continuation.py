"""
Tests for twpa.solvers.continuation.

These tests define the expected behavior of the continuation layer used to
solve nonlinear HB problems robustly by ramping a parameter from an easy
starting point to the target operating point.

The tests are API-tolerant in naming, but require the module to expose a public
way to run continuation over a scalar parameter, usually through one of:

    - continuation_solve
    - solve_continuation
    - parameter_continuation
    - continuation

Expected behavior
-----------------
A continuation solver should:

    - visit a monotone sequence of parameter values;
    - use the previous solution as the next initial guess;
    - return the final state near the target solution;
    - expose per-step status/history when possible;
    - stop or report failure when an intermediate solve fails;
    - support both real and complex state vectors.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

import twpa.solvers.continuation as cont


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Cannot convert object to mapping: {type(obj)!r}")


class ToySolverFailure(RuntimeError):
    pass


def _make_param_grid(
    start: float = 0.0,
    stop: float = 1.0,
    n_steps: int = 6,
) -> np.ndarray:
    for name in [
        "make_continuation_grid",
        "continuation_grid",
        "parameter_grid",
        "linear_parameter_grid",
        "make_parameter_schedule",
    ]:
        if hasattr(cont, name):
            out = _call_with_supported_kwargs(
                getattr(cont, name),
                start=start,
                start_value=start,
                lambda_start=start,
                stop=stop,
                end=stop,
                target=stop,
                target_value=stop,
                lambda_target=stop,
                n_steps=n_steps,
                steps=n_steps,
                num_steps=n_steps,
            )
            return np.asarray(out, dtype=float)

    return np.linspace(start, stop, n_steps)


def _extract_final_state(result: Any) -> np.ndarray:
    value = _get_attr_or_key(
        result,
        "x",
        "state",
        "solution",
        "final_state",
        "final_solution",
        default=None,
    )
    if value is not None:
        return np.asarray(value)

    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "shape") or isinstance(item, (list, tuple)):
                arr = np.asarray(item)
                if arr.ndim >= 1:
                    return arr

    return np.asarray(result)


def _extract_success(result: Any) -> bool:
    value = _get_attr_or_key(
        result,
        "success",
        "converged",
        "passed",
        "ok",
        default=None,
    )
    if value is None:
        return True
    return bool(value)


def _extract_history(result: Any) -> list[Any]:
    value = _get_attr_or_key(
        result,
        "history",
        "steps",
        "records",
        "step_results",
        "continuation_history",
        default=None,
    )
    if value is None and isinstance(result, tuple) and len(result) >= 2:
        candidate = result[1]
        if isinstance(candidate, list):
            value = candidate

    if value is None:
        return []

    return list(value)


def _extract_parameter_values(result: Any) -> np.ndarray:
    history = _extract_history(result)
    values = []

    for item in history:
        value = _get_attr_or_key(
            item,
            "parameter",
            "parameter_value",
            "lambda_value",
            "lam",
            "alpha",
            "scale",
            default=None,
        )
        if value is not None:
            values.append(float(value))

    if values:
        return np.asarray(values, dtype=float)

    value = _get_attr_or_key(
        result,
        "parameter_values",
        "parameters",
        "lambda_values",
        "schedule",
        default=None,
    )
    if value is not None:
        return np.asarray(value, dtype=float)

    return np.asarray([], dtype=float)


def _continuation_solve(
    solve_at_parameter: Callable[..., Any],
    x0: Any,
    parameter_values: Sequence[float],
    *,
    fail_fast: bool = True,
    max_step_retries: int = 0,
) -> Any:
    for name in [
        "continuation_solve",
        "solve_continuation",
        "parameter_continuation",
        "run_continuation",
        "continuation",
    ]:
        if hasattr(cont, name):
            return _call_with_supported_kwargs(
                getattr(cont, name),
                solve_at_parameter=solve_at_parameter,
                solve_fn=solve_at_parameter,
                solver=solve_at_parameter,
                step_solver=solve_at_parameter,
                residual_solver=solve_at_parameter,
                x0=x0,
                initial_state=x0,
                initial_guess=x0,
                parameter_values=np.asarray(parameter_values, dtype=float),
                parameters=np.asarray(parameter_values, dtype=float),
                schedule=np.asarray(parameter_values, dtype=float),
                lambdas=np.asarray(parameter_values, dtype=float),
                lambda_values=np.asarray(parameter_values, dtype=float),
                fail_fast=fail_fast,
                stop_on_failure=fail_fast,
                max_step_retries=max_step_retries,
                retries=max_step_retries,
            )

    raise AttributeError(
        "twpa.solvers.continuation must expose a continuation solver such as "
        "continuation_solve, solve_continuation, or parameter_continuation."
    )


def _toy_real_solver_factory(target_log: list[dict[str, Any]]) -> Callable[..., Any]:
    def solver(x0: Any, parameter: float | None = None, **kwargs: Any) -> Mapping[str, Any]:
        if parameter is None:
            parameter = kwargs.get("parameter_value", kwargs.get("lambda_value", kwargs.get("alpha")))
        if parameter is None:
            raise TypeError("solver did not receive a parameter value")

        x0_arr = np.asarray(x0, dtype=float)
        target = np.array([parameter, parameter**2], dtype=float)

        target_log.append(
            {
                "parameter": float(parameter),
                "x0": x0_arr.copy(),
                "target": target.copy(),
            }
        )

        return {
            "success": True,
            "x": target,
            "state": target,
            "residual_norm": 0.0,
            "n_iter": 1,
            "parameter": float(parameter),
        }

    return solver


def _toy_complex_solver_factory(target_log: list[dict[str, Any]]) -> Callable[..., Any]:
    def solver(x0: Any, parameter: float | None = None, **kwargs: Any) -> Mapping[str, Any]:
        if parameter is None:
            parameter = kwargs.get("parameter_value", kwargs.get("lambda_value", kwargs.get("alpha")))
        if parameter is None:
            raise TypeError("solver did not receive a parameter value")

        x0_arr = np.asarray(x0, dtype=np.complex128)
        target = np.array(
            [
                parameter + 1j * parameter**2,
                np.exp(1j * parameter),
            ],
            dtype=np.complex128,
        )

        target_log.append(
            {
                "parameter": float(parameter),
                "x0": x0_arr.copy(),
                "target": target.copy(),
            }
        )

        return {
            "success": True,
            "x": target,
            "residual_norm": 0.0,
            "n_iter": 2,
            "parameter": float(parameter),
        }

    return solver


def test_parameter_grid_is_linear_and_includes_endpoints() -> None:
    grid = _make_param_grid(start=0.0, stop=1.0, n_steps=6)

    np.testing.assert_allclose(grid, np.linspace(0.0, 1.0, 6), rtol=0.0, atol=0.0)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


def test_parameter_grid_supports_decreasing_direction() -> None:
    grid = _make_param_grid(start=1.0, stop=0.0, n_steps=5)

    np.testing.assert_allclose(grid, np.linspace(1.0, 0.0, 5), rtol=0.0, atol=0.0)
    assert np.all(np.diff(grid) < 0.0)


def test_parameter_grid_single_step_if_supported() -> None:
    try:
        grid = _make_param_grid(start=0.25, stop=1.0, n_steps=1)
    except (ValueError, AssertionError):
        pytest.skip("Single-step continuation grids are intentionally unsupported.")

    assert grid.shape == (1,)
    assert grid[0] == pytest.approx(1.0)


def test_real_continuation_reaches_final_target() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.linspace(0.0, 1.0, 6)

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)

    x_final = _extract_final_state(result)

    assert _extract_success(result)
    np.testing.assert_allclose(x_final, np.array([1.0, 1.0]), rtol=1e-14, atol=1e-14)
    assert len(calls) == len(schedule)


def test_complex_continuation_reaches_final_target() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_complex_solver_factory(calls)
    schedule = np.linspace(0.0, 0.8, 5)

    result = _continuation_solve(
        solver,
        x0=np.zeros(2, dtype=np.complex128),
        parameter_values=schedule,
    )

    x_final = _extract_final_state(result)
    expected = np.array(
        [
            0.8 + 1j * 0.8**2,
            np.exp(1j * 0.8),
        ],
        dtype=np.complex128,
    )

    assert _extract_success(result)
    np.testing.assert_allclose(x_final, expected, rtol=1e-14, atol=1e-14)
    assert len(calls) == len(schedule)


def test_previous_solution_is_used_as_next_initial_guess() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.linspace(0.0, 1.0, 5)

    _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)

    assert len(calls) == len(schedule)

    np.testing.assert_allclose(calls[0]["x0"], np.zeros(2), rtol=0.0, atol=0.0)

    for idx in range(1, len(calls)):
        previous_target = calls[idx - 1]["target"]
        current_initial = calls[idx]["x0"]
        np.testing.assert_allclose(current_initial, previous_target, rtol=1e-14, atol=1e-14)


def test_parameter_values_are_visited_in_order() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.array([0.0, 0.1, 0.4, 0.55, 1.0])

    _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)

    visited = np.asarray([c["parameter"] for c in calls], dtype=float)
    np.testing.assert_allclose(visited, schedule, rtol=0.0, atol=0.0)


def test_history_contains_step_information_when_available() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.linspace(0.0, 1.0, 4)

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)
    history = _extract_history(result)

    if not history:
        pytest.skip("Continuation history is optional but recommended.")

    assert len(history) == len(schedule)

    parameter_values = _extract_parameter_values(result)
    if parameter_values.size:
        np.testing.assert_allclose(parameter_values, schedule, rtol=0.0, atol=0.0)


def test_result_object_is_mapping_or_dataclass_like_when_not_raw_array() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=[0.0, 1.0])

    if isinstance(result, np.ndarray):
        pytest.skip("Raw ndarray final-state result is acceptable.")

    if isinstance(result, tuple):
        pytest.skip("Tuple continuation result is acceptable.")

    mapping = _as_mapping(result)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0


def test_result_object_is_json_serializable_when_mapping_like() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=[0.0, 0.5, 1.0])

    try:
        mapping = _as_mapping(result)
    except TypeError:
        pytest.skip("Non-mapping raw result is acceptable.")

    def convert(value: Any) -> Any:
        if isinstance(value, complex):
            return {"real": float(np.real(value)), "imag": float(np.imag(value))}
        if isinstance(value, np.ndarray):
            if np.iscomplexobj(value):
                return {
                    "real": np.real(value).tolist(),
                    "imag": np.imag(value).tolist(),
                }
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    import json

    json.dumps(convert(mapping), default=str)


def test_failure_in_step_is_reported_or_raised() -> None:
    calls: list[float] = []

    def failing_solver(x0: Any, parameter: float | None = None, **kwargs: Any) -> Mapping[str, Any]:
        if parameter is None:
            parameter = kwargs.get("parameter_value", kwargs.get("lambda_value", kwargs.get("alpha")))
        if parameter is None:
            raise TypeError("solver did not receive a parameter value")

        calls.append(float(parameter))

        if parameter >= 0.5:
            raise ToySolverFailure("intentional continuation failure")

        return {
            "success": True,
            "x": np.asarray([parameter], dtype=float),
            "parameter": float(parameter),
        }

    schedule = np.array([0.0, 0.25, 0.5, 0.75])

    try:
        result = _continuation_solve(
            failing_solver,
            x0=np.zeros(1),
            parameter_values=schedule,
            fail_fast=True,
        )
    except ToySolverFailure:
        assert calls == [0.0, 0.25, 0.5]
        return
    except RuntimeError as exc:
        assert "failure" in str(exc).lower() or "intentional" in str(exc).lower()
        return

    assert not _extract_success(result)


def test_non_fail_fast_can_return_partial_result_if_supported() -> None:
    calls: list[float] = []

    def sometimes_failing_solver(x0: Any, parameter: float | None = None, **kwargs: Any) -> Mapping[str, Any]:
        if parameter is None:
            parameter = kwargs.get("parameter_value", kwargs.get("lambda_value", kwargs.get("alpha")))
        if parameter is None:
            raise TypeError("solver did not receive a parameter value")

        calls.append(float(parameter))

        if parameter == pytest.approx(0.5):
            return {
                "success": False,
                "x": np.asarray(x0),
                "parameter": float(parameter),
                "message": "intentional unsuccessful step",
            }

        return {
            "success": True,
            "x": np.asarray([parameter], dtype=float),
            "parameter": float(parameter),
        }

    schedule = np.array([0.0, 0.5, 1.0])

    try:
        result = _continuation_solve(
            sometimes_failing_solver,
            x0=np.zeros(1),
            parameter_values=schedule,
            fail_fast=False,
        )
    except Exception as exc:
        pytest.skip(f"Non-fail-fast behavior is optional: {type(exc).__name__}: {exc}")

    assert calls[0] == pytest.approx(0.0)
    assert not _extract_success(result) or len(calls) >= 2


def test_adaptive_continuation_class_if_available() -> None:
    class_names = [
        "ContinuationSolver",
        "ParameterContinuation",
        "ContinuationRunner",
    ]

    cls = None
    for name in class_names:
        if hasattr(cont, name):
            cls = getattr(cont, name)
            break

    if cls is None:
        pytest.skip("Object-oriented continuation runner is optional.")

    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    runner = _call_with_supported_kwargs(
        cls,
        solve_at_parameter=solver,
        solve_fn=solver,
        solver=solver,
        parameter_values=np.linspace(0.0, 1.0, 4),
        schedule=np.linspace(0.0, 1.0, 4),
        x0=np.zeros(2),
        initial_state=np.zeros(2),
    )

    run_method = None
    for method_name in ["run", "solve", "execute"]:
        if hasattr(runner, method_name):
            run_method = getattr(runner, method_name)
            break

    if run_method is None:
        pytest.skip("Continuation runner object has no run/solve/execute method.")

    result = _call_with_supported_kwargs(
        run_method,
        x0=np.zeros(2),
        initial_state=np.zeros(2),
        parameter_values=np.linspace(0.0, 1.0, 4),
        schedule=np.linspace(0.0, 1.0, 4),
    )

    x_final = _extract_final_state(result)
    np.testing.assert_allclose(x_final, np.array([1.0, 1.0]), rtol=1e-14, atol=1e-14)


def test_step_result_class_if_available() -> None:
    class_names = [
        "ContinuationStep",
        "ContinuationStepResult",
        "StepResult",
    ]

    cls = None
    for name in class_names:
        if hasattr(cont, name):
            cls = getattr(cont, name)
            break

    if cls is None:
        pytest.skip("Continuation step-result dataclass is optional.")

    step = _call_with_supported_kwargs(
        cls,
        parameter=0.5,
        parameter_value=0.5,
        lambda_value=0.5,
        x=np.array([0.5]),
        state=np.array([0.5]),
        solution=np.array([0.5]),
        success=True,
        converged=True,
        residual_norm=0.0,
        n_iter=1,
    )

    mapping = _as_mapping(step)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    success = _get_attr_or_key(step, "success", "converged", default=True)
    assert bool(success)


def test_continuation_rejects_empty_parameter_values() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    with pytest.raises((ValueError, AssertionError, IndexError)):
        _continuation_solve(solver, x0=np.zeros(2), parameter_values=[])


def test_continuation_rejects_nonfinite_parameter_values() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    with pytest.raises((ValueError, AssertionError, FloatingPointError)):
        _continuation_solve(solver, x0=np.zeros(2), parameter_values=[0.0, np.nan, 1.0])


def test_continuation_rejects_none_solver() -> None:
    with pytest.raises((TypeError, ValueError, AttributeError)):
        _continuation_solve(None, x0=np.zeros(1), parameter_values=[0.0, 1.0])  # type: ignore[arg-type]


def test_continuation_propagates_nan_initial_state_or_reports_failure() -> None:
    def solver(x0: Any, parameter: float | None = None, **kwargs: Any) -> Mapping[str, Any]:
        x0_arr = np.asarray(x0, dtype=float)
        if np.any(~np.isfinite(x0_arr)):
            raise FloatingPointError("nonfinite initial state")
        return {
            "success": True,
            "x": np.asarray([parameter], dtype=float),
            "parameter": float(parameter),
        }

    try:
        result = _continuation_solve(
            solver,
            x0=np.array([np.nan]),
            parameter_values=[0.0, 1.0],
        )
    except (FloatingPointError, ValueError, AssertionError):
        return

    assert not _extract_success(result)


def test_continuation_handles_numpy_scalar_parameters() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    schedule = np.asarray([np.float64(0.0), np.float64(0.5), np.float64(1.0)])
    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)

    x_final = _extract_final_state(result)
    np.testing.assert_allclose(x_final, np.array([1.0, 1.0]), rtol=1e-14, atol=1e-14)


def test_continuation_with_single_parameter_value_uses_initial_guess_once() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=[0.75])

    x_final = _extract_final_state(result)

    assert len(calls) == 1
    np.testing.assert_allclose(x_final, np.array([0.75, 0.75**2]), rtol=1e-14, atol=1e-14)


def test_solver_receives_parameter_by_some_supported_name() -> None:
    received: list[dict[str, Any]] = []

    def recording_solver(x0: Any, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        received.append({"args": args, "kwargs": kwargs})

        parameter = None
        for key in ["parameter", "parameter_value", "lambda_value", "lam", "alpha", "scale"]:
            if key in kwargs:
                parameter = kwargs[key]
                break

        if parameter is None and args:
            parameter = args[0]

        if parameter is None:
            raise TypeError("No continuation parameter received")

        return {
            "success": True,
            "x": np.asarray([parameter], dtype=float),
            "parameter": float(parameter),
        }

    result = _continuation_solve(
        recording_solver,
        x0=np.zeros(1),
        parameter_values=[0.0, 0.5, 1.0],
    )

    x_final = _extract_final_state(result)

    assert len(received) == 3
    np.testing.assert_allclose(x_final, np.array([1.0]), rtol=1e-14, atol=1e-14)


def test_monotone_parameter_values_are_preserved_not_resorted() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.array([1.0, 0.8, 0.4, 0.0])

    result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)
    x_final = _extract_final_state(result)

    visited = np.asarray([c["parameter"] for c in calls], dtype=float)

    np.testing.assert_allclose(visited, schedule, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(x_final, np.array([0.0, 0.0]), rtol=1e-14, atol=1e-14)


def test_duplicate_parameter_values_are_allowed_or_rejected_consistently() -> None:
    calls: list[dict[str, Any]] = []
    solver = _toy_real_solver_factory(calls)
    schedule = np.array([0.0, 0.5, 0.5, 1.0])

    try:
        result = _continuation_solve(solver, x0=np.zeros(2), parameter_values=schedule)
    except (ValueError, AssertionError):
        return

    x_final = _extract_final_state(result)
    np.testing.assert_allclose(x_final, np.array([1.0, 1.0]), rtol=1e-14, atol=1e-14)
    assert len(calls) == len(schedule)