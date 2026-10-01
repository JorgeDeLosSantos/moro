"""Sampled end-effector workspace analysis."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import sympy as sp


__all__ = ["Workspace", "sample_workspace"]


@dataclass
class Workspace:
    """Sampled joint configurations and corresponding end-effector points."""

    points: np.ndarray
    configurations: np.ndarray
    joint_limits: tuple
    seed: int | None = None

    def __post_init__(self):
        self.points = _as_real_array(self.points, name="points", ndim=2)
        self.configurations = _as_real_array(
            self.configurations,
            name="configurations",
            ndim=2,
        )

        if self.points.shape[0] < 1:
            raise ValueError("Workspace must contain at least one sample.")
        if self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        if self.configurations.shape[0] != self.points.shape[0]:
            raise ValueError(
                "points and configurations must contain the same number of samples."
            )
        if self.configurations.shape[1] < 1:
            raise ValueError("configurations must contain at least one DOF.")

        self.joint_limits = _normalize_joint_limits(
            self.joint_limits,
            self.configurations.shape[1],
        )
        self.seed = _normalize_seed(self.seed)

        lower = np.array([pair[0] for pair in self.joint_limits], dtype=float)
        upper = np.array([pair[1] for pair in self.joint_limits], dtype=float)
        if np.any(self.configurations < lower) or np.any(self.configurations > upper):
            raise ValueError(
                "Every configuration must lie within the stored joint_limits."
            )

    @property
    def samples(self):
        return self.points.shape[0]

    @property
    def dof(self):
        return self.configurations.shape[1]

    @property
    def bounds(self):
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        return tuple(
            (float(lower), float(upper))
            for lower, upper in zip(mins, maxs)
        )

    def __repr__(self):
        return (
            f"Workspace(samples={self.samples}, dof={self.dof}, "
            f"seed={self.seed})"
        )


def _as_real_array(value, *, name, ndim):
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numerical.") from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values.") from exc

    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(arr, dtype=float, copy=True)


def _normalize_seed(seed):
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be None or an integer that is not Boolean.")
    return int(seed)


def _normalize_samples(samples):
    if isinstance(samples, bool) or not isinstance(samples, Integral):
        raise ValueError("samples must be a positive integer that is not Boolean.")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be greater than 0.")
    return samples


def _normalize_limit_scalar(value, *, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar.")
    if not isinstance(value, Real):
        try:
            value = float(value)
        except Exception as exc:
            raise ValueError(f"{name} must be a finite real scalar.") from exc
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _normalize_joint_limits(joint_limits, dof):
    try:
        limits = list(joint_limits)
    except TypeError as exc:
        raise ValueError("joint_limits must contain one pair per robot DOF.") from exc

    if len(limits) != dof:
        raise ValueError(
            f"joint_limits must contain exactly {dof} pairs. Got {len(limits)}."
        )

    normalized = []
    for index, pair in enumerate(limits, start=1):
        if pair is None:
            raise ValueError("Partial joint-limit overrides are not supported.")
        try:
            values = list(pair)
        except TypeError as exc:
            raise ValueError(
                f"Joint limit for joint {index} must be a 2-value pair."
            ) from exc
        if len(values) != 2:
            raise ValueError(
                f"Joint limit for joint {index} must contain exactly 2 values."
            )
        lower = _normalize_limit_scalar(
            values[0], name=f"lower limit for joint {index}"
        )
        upper = _normalize_limit_scalar(
            values[1], name=f"upper limit for joint {index}"
        )
        if lower >= upper:
            raise ValueError(
                f"Joint limit for joint {index} must satisfy lower < upper."
            )
        normalized.append((lower, upper))

    return tuple(normalized)


def _prepare_position_model(robot, parameters=None):
    qs = tuple(robot.qs)
    position = sp.Matrix(robot.T[:3, 3])

    if parameters is not None:
        try:
            position = position.subs(parameters)
        except Exception as exc:
            raise ValueError(
                "parameters must be a SymPy-compatible substitution mapping."
            ) from exc

    allowed_symbols = set(qs)
    for q in qs:
        allowed_symbols.update(getattr(q, "free_symbols", set()))

    unresolved = set(position.free_symbols) - allowed_symbols
    if unresolved:
        missing = ", ".join(str(symbol) for symbol in sorted(unresolved, key=str))
        raise ValueError(
            "Cannot sample workspace because the following symbols have no "
            f"numerical value: {missing}. Provide them using 'parameters'."
        )

    try:
        func = sp.lambdify(qs, position, modules="numpy")
    except Exception as exc:
        raise ValueError("Could not compile numerical forward kinematics.") from exc
    return func


def _evaluate_position(func, q, *, sample_index):
    try:
        raw = func(*q)
    except Exception as exc:
        raise ValueError(
            f"Forward kinematics failed for workspace sample {sample_index}."
        ) from exc

    try:
        raw_arr = np.asarray(raw)
    except Exception as exc:
        raise ValueError(
            f"Workspace sample {sample_index} produced invalid Cartesian data."
        ) from exc

    if np.iscomplexobj(raw_arr):
        raise ValueError(
            f"Workspace sample {sample_index} produced a complex Cartesian position."
        )

    try:
        p = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Workspace sample {sample_index} produced non-numerical Cartesian data."
        ) from exc

    if p.size != 3:
        raise ValueError(
            f"Workspace sample {sample_index} must produce exactly 3 Cartesian values."
        )
    if not np.all(np.isfinite(p)):
        raise ValueError(
            f"Workspace sample {sample_index} produced non-finite Cartesian data."
        )
    return p


def sample_workspace(
    robot,
    *,
    samples=1000,
    joint_limits=None,
    parameters=None,
    seed=None,
):
    """Sample end-effector positions uniformly from a finite joint-space domain."""
    samples = _normalize_samples(samples)
    seed = _normalize_seed(seed)

    effective_limits = _normalize_joint_limits(
        robot.joint_limits if joint_limits is None else joint_limits,
        robot.dof,
    )

    lower = np.array([pair[0] for pair in effective_limits], dtype=float)
    upper = np.array([pair[1] for pair in effective_limits], dtype=float)

    position_func = _prepare_position_model(robot, parameters=parameters)

    rng = np.random.default_rng(seed)
    configurations = rng.uniform(
        low=lower,
        high=upper,
        size=(samples, robot.dof),
    )

    points = np.empty((samples, 3), dtype=float)
    for index, q in enumerate(configurations):
        points[index] = _evaluate_position(
            position_func,
            q,
            sample_index=index,
        )

    return Workspace(
        points=points,
        configurations=configurations,
        joint_limits=effective_limits,
        seed=seed,
    )
