"""Differential kinematics for serial manipulators.

This module builds on the geometric Jacobian already provided by Robot.
It owns task-space row selection, forward Cartesian velocity propagation, and
numerical velocity-level inverse kinematics.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import sympy as sp
from sympy import Matrix


__all__ = [
    "VelocityIKSolution",
    "task_jacobian",
    "cartesian_velocity",
    "solve_velocity_ik",
]


_TASK_ROWS = {
    "vx": 0,
    "vy": 1,
    "vz": 2,
    "wx": 3,
    "wy": 4,
    "wz": 5,
}

_TASK_PRESETS = {
    "linear": ("vx", "vy", "vz"),
    "angular": ("wx", "wy", "wz"),
    "twist": ("vx", "vy", "vz", "wx", "wy", "wz"),
}


@dataclass(frozen=True)
class VelocityIKSolution:
    """Result returned by solve_velocity_ik()."""

    qd: Matrix
    unconstrained_qd: Matrix
    desired_velocity: Matrix
    achieved_velocity: Matrix
    residual: Matrix
    residual_norm: float
    rank: int
    condition_number: float
    method: str
    success: bool
    limited: bool
    message: str


def _normalize_task(task):
    if isinstance(task, str):
        name = task.lower()
        if name not in _TASK_PRESETS:
            allowed = ", ".join(sorted(_TASK_PRESETS))
            raise ValueError(
                f"Unknown task preset {task!r}; expected one of: {allowed}."
            )
        return _TASK_PRESETS[name]

    if isinstance(task, (bytes, bytearray)) or not isinstance(task, Sequence):
        raise TypeError(
            "task must be a preset string or a sequence of component names."
        )

    if len(task) == 0:
        raise ValueError("task must contain at least one component.")

    components = []
    for component in task:
        if not isinstance(component, str):
            raise TypeError("task components must be strings.")
        name = component.lower()
        if name not in _TASK_ROWS:
            allowed = ", ".join(_TASK_ROWS)
            raise ValueError(
                f"Unknown task component {component!r}; expected one of: {allowed}."
            )
        components.append(name)

    if len(set(components)) != len(components):
        raise ValueError("task must not contain duplicate components.")

    return tuple(components)


def _robot_interface(robot):
    missing = [
        name for name in ("J", "qs", "dof")
        if not hasattr(robot, name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise TypeError(
            "robot must provide the differential-kinematics interface "
            f"(missing: {joined})."
        )

    try:
        J = Matrix(robot.J)
        qs = list(robot.qs)
        dof = int(robot.dof)
    except Exception as exc:
        raise TypeError(
            "robot must expose compatible J, qs, and dof attributes."
        ) from exc

    if J.rows != 6:
        raise ValueError("robot.J must have exactly 6 rows.")
    if J.cols != dof:
        raise ValueError("robot.J column count must equal robot.dof.")
    if len(qs) != dof:
        raise ValueError("len(robot.qs) must equal robot.dof.")

    return J, qs, dof


def _as_vector(value, size, *, name):
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and any(item is None for item in value)
    ):
        raise ValueError(f"{name} must not contain None.")

    try:
        vector = Matrix(value)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a vector-like object with {size} components."
        ) from exc

    if vector.shape == (size, 1):
        result = Matrix(vector)
    elif vector.shape == (1, size):
        result = Matrix(vector.T)
    elif len(vector) == size and (vector.rows == size or vector.cols == size):
        result = Matrix(list(vector)).reshape(size, 1)
    else:
        raise ValueError(f"{name} must contain exactly {size} components.")

    if any(value is None for value in result):
        raise ValueError(f"{name} must not contain None.")

    return result


def _normalize_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping or None.")

    normalized = {}
    for key, value in parameters.items():
        if not isinstance(key, sp.Basic):
            raise TypeError("parameter keys must be SymPy objects.")
        if value is None:
            raise ValueError("parameter values must not be None.")
        try:
            normalized[key] = sp.sympify(value)
        except Exception as exc:
            raise TypeError(
                f"Parameter value for {key!r} could not be sympified."
            ) from exc
    return normalized


def _configuration_substitutions(q, qs, dof):
    if q is None:
        return {}
    q_vector = _as_vector(q, dof, name="q")
    return dict(zip(qs, list(q_vector)))


def _evaluate_expression(expr, q, qs, dof, parameters):
    result = Matrix(expr)
    q_substitutions = _configuration_substitutions(q, qs, dof)
    if q_substitutions:
        result = result.subs(q_substitutions)

    parameter_substitutions = _normalize_parameters(parameters)
    if parameter_substitutions:
        result = result.subs(parameter_substitutions)

    return Matrix(result)


def _require_resolved_real_finite(expr, *, name):
    matrix = Matrix(expr)
    if matrix.free_symbols:
        symbols = ", ".join(sorted(str(s) for s in matrix.free_symbols))
        raise ValueError(
            f"{name} contains unresolved symbols: {symbols}."
        )

    for value in matrix:
        value = sp.sympify(value)
        numeric = sp.N(value)
        if numeric.is_finite is not True:
            raise ValueError(f"{name} must contain only finite values.")
        if numeric.is_real is not True:
            raise ValueError(f"{name} must contain only real values.")

    return matrix


def _to_numpy_vector(expr, *, name):
    matrix = _require_resolved_real_finite(expr, name=name)
    try:
        array = np.asarray(matrix, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to real floats.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _to_numpy_matrix(expr, *, name):
    matrix = _require_resolved_real_finite(expr, name=name)
    try:
        array = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to real floats.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_positive_real(value, *, name):
    if isinstance(value, bool) or not np.isscalar(value):
        raise TypeError(f"{name} must be a finite positive real scalar.")

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be a finite positive real scalar."
        ) from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return numeric


def _validate_finite_real(value, *, name):
    if isinstance(value, bool) or not np.isscalar(value):
        raise TypeError(f"{name} must be a finite real scalar.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _normalize_method(method, damping):
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    method = method.lower()
    if method not in {"pinv", "dls"}:
        raise ValueError("method must be either 'pinv' or 'dls'.")

    if method == "pinv":
        if damping is not None:
            raise ValueError("damping must be None when method='pinv'.")
        return method, None

    if damping is None:
        raise ValueError("damping is required when method='dls'.")
    return method, _validate_positive_real(damping, name="damping")


def _normalize_joint_velocity_limits(limits, dof):
    if limits is None:
        return None

    if isinstance(limits, (str, bytes, bytearray)) or not isinstance(
        limits, Sequence
    ):
        raise TypeError("joint_velocity_limits must be a sequence or None.")

    if len(limits) != dof:
        raise ValueError(
            "joint_velocity_limits must contain exactly robot.dof entries."
        )

    normalized = []
    for index, item in enumerate(limits, start=1):
        if item is None:
            raise ValueError(
                "joint_velocity_limits entries must not be None."
            )

        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
        ):
            if len(item) != 2:
                raise ValueError(
                    f"Joint velocity limit {index} must contain two bounds."
                )
            lower = _validate_finite_real(
                item[0], name=f"joint_velocity_limits[{index - 1}][0]"
            )
            upper = _validate_finite_real(
                item[1], name=f"joint_velocity_limits[{index - 1}][1]"
            )
            if lower >= upper:
                raise ValueError(
                    "Asymmetric joint velocity limits must satisfy lower < upper."
                )
        else:
            vmax = _validate_positive_real(
                item, name=f"joint_velocity_limits[{index - 1}]"
            )
            lower, upper = -vmax, vmax

        normalized.append((lower, upper))

    return tuple(normalized)


def _svd_diagnostics(J):
    U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)

    if singular_values.size == 0:
        threshold = 0.0
        rank = 0
        condition_number = math.inf
        return U, singular_values, Vt, threshold, rank, condition_number

    sigma_max = float(singular_values[0])
    threshold = (
        max(J.shape) * np.finfo(J.dtype).eps * sigma_max
        if sigma_max > 0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > threshold))

    if rank < min(J.shape) or singular_values[-1] <= threshold:
        condition_number = math.inf
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    return U, singular_values, Vt, threshold, rank, condition_number


def task_jacobian(robot, q=None, *, task="twist", parameters=None):
    """Return the selected geometric task Jacobian as a SymPy matrix."""
    components = _normalize_task(task)
    J, qs, dof = _robot_interface(robot)

    rows = [_TASK_ROWS[name] for name in components]
    J_task = J[rows, :]
    return _evaluate_expression(
        J_task,
        q,
        qs,
        dof,
        parameters,
    )


def cartesian_velocity(
    robot,
    q,
    qd,
    *,
    task="twist",
    parameters=None,
):
    """Propagate joint velocity into the selected Cartesian task velocity."""
    components = _normalize_task(task)
    _, _, dof = _robot_interface(robot)

    q_vector = _as_vector(q, dof, name="q")
    qd_vector = _as_vector(qd, dof, name="qd")
    parameter_substitutions = _normalize_parameters(parameters)

    J_task = task_jacobian(
        robot,
        q_vector,
        task=components,
        parameters=parameter_substitutions,
    )

    if parameter_substitutions:
        qd_vector = qd_vector.subs(parameter_substitutions)

    result = Matrix(J_task * qd_vector)
    return _require_resolved_real_finite(
        result,
        name="Cartesian velocity",
    )


def solve_velocity_ik(
    robot,
    q,
    velocity,
    *,
    task="twist",
    method="pinv",
    damping=None,
    joint_velocity_limits=None,
    parameters=None,
    tol=1e-9,
):
    """Solve velocity-level inverse kinematics for a selected task."""
    components = _normalize_task(task)
    method, damping = _normalize_method(method, damping)
    tol = _validate_positive_real(tol, name="tol")

    _, _, dof = _robot_interface(robot)
    q_vector = _as_vector(q, dof, name="q")
    velocity_vector = _as_vector(
        velocity,
        len(components),
        name="velocity",
    )

    parameter_substitutions = _normalize_parameters(parameters)

    J_symbolic = task_jacobian(
        robot,
        q_vector,
        task=components,
        parameters=parameter_substitutions,
    )
    if parameter_substitutions:
        velocity_vector = velocity_vector.subs(parameter_substitutions)

    J = _to_numpy_matrix(J_symbolic, name="Task Jacobian")
    desired = _to_numpy_vector(
        velocity_vector,
        name="Desired task velocity",
    )

    U, singular_values, Vt, threshold, rank, condition_number = (
        _svd_diagnostics(J)
    )

    if method == "pinv":
        gains = np.zeros_like(singular_values)
        nonzero = singular_values > threshold
        gains[nonzero] = 1.0 / singular_values[nonzero]
    else:
        gains = singular_values / (singular_values**2 + damping**2)

    unconstrained = Vt.T @ (gains * (U.T @ desired))
    if not np.all(np.isfinite(unconstrained)):
        raise RuntimeError(
            "Velocity IK produced non-finite joint velocities."
        )

    final_qd = np.array(unconstrained, dtype=float, copy=True)
    limits = _normalize_joint_velocity_limits(
        joint_velocity_limits,
        dof,
    )

    limited = False
    if limits is not None:
        lower = np.asarray([pair[0] for pair in limits], dtype=float)
        upper = np.asarray([pair[1] for pair in limits], dtype=float)
        clipped = np.clip(final_qd, lower, upper)
        limited = not np.array_equal(clipped, final_qd)
        final_qd = clipped

    achieved = J @ final_qd
    residual = desired - achieved
    residual_norm = float(np.linalg.norm(residual))

    if not (
        np.all(np.isfinite(achieved))
        and np.all(np.isfinite(residual))
        and math.isfinite(residual_norm)
    ):
        raise RuntimeError(
            "Velocity IK produced non-finite task-space diagnostics."
        )

    success = residual_norm <= tol

    if success and limited:
        message = "Desired task velocity achieved within tolerance after limiting."
    elif success:
        message = "Desired task velocity achieved within tolerance."
    elif limited:
        message = (
            "Desired task velocity was not achieved within tolerance after "
            "joint-velocity limiting."
        )
    else:
        message = "Desired task velocity was not achieved within tolerance."

    return VelocityIKSolution(
        qd=Matrix(final_qd),
        unconstrained_qd=Matrix(unconstrained),
        desired_velocity=Matrix(desired),
        achieved_velocity=Matrix(achieved),
        residual=Matrix(residual),
        residual_norm=residual_norm,
        rank=rank,
        condition_number=condition_number,
        method=method,
        success=success,
        limited=limited,
        message=message,
    )
