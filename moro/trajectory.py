"""Numerical point-to-point trajectory generation.

This module provides compact, robot-independent polynomial trajectory
generation in joint space and Cartesian position space.
"""

from dataclasses import dataclass

import numpy as np


__all__ = [
    "JointTrajectory",
    "PositionTrajectory",
    "joint_trajectory",
    "position_trajectory",
]


_METHODS = {"linear", "cubic", "quintic"}


def _normalize_method(method):
    """Normalize and validate an interpolation method name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    normalized = method.lower()
    if normalized not in _METHODS:
        raise ValueError(
            "method must be one of 'linear', 'cubic', or 'quintic'."
        )
    return normalized


def _as_real_float_array(value, *, name, ndim=None):
    """Return an independent finite real float array."""
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numerical.") from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(arr, dtype=float, copy=True)


def _prepare_time(t):
    """Validate and normalize an explicit trajectory time vector."""
    time = _as_real_float_array(t, name="t", ndim=1)
    if time.size < 2:
        raise ValueError("t must contain at least two samples.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("t must be strictly increasing.")
    return time


def _prepare_joint_endpoint(value, *, name):
    """Normalize a scalar or vector joint endpoint to shape (dof,)."""
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector.") from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    if raw.ndim == 0:
        arr = _as_real_float_array([value], name=name, ndim=1)
    elif raw.ndim == 1:
        arr = _as_real_float_array(value, name=name, ndim=1)
    else:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector.")

    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _prepare_joint_boundary(value, dof, *, name):
    """Normalize a joint derivative boundary condition."""
    if value is None:
        return None

    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must contain exactly {dof} real finite value(s)."
        ) from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    if raw.ndim == 0:
        if dof != 1:
            raise ValueError(
                f"{name} must contain exactly {dof} values; scalar "
                "broadcasting is not supported."
            )
        arr = _as_real_float_array([value], name=name, ndim=1)
    elif raw.ndim == 1:
        arr = _as_real_float_array(value, name=name, ndim=1)
    else:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector.")

    if arr.size != dof:
        raise ValueError(
            f"{name} must contain exactly {dof} values. Got {arr.size}."
        )
    return arr


def _prepare_cartesian_vector(value, *, name):
    """Normalize a Cartesian vector to exactly three components."""
    arr = _as_real_float_array(value, name=name, ndim=1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")
    return arr


def _prepare_cartesian_boundary(value, *, name):
    """Normalize an optional Cartesian derivative boundary condition."""
    if value is None:
        return None
    return _prepare_cartesian_vector(value, name=name)


def _prepare_boundary_conditions(
    dimension,
    *,
    method,
    v0,
    vf,
    a0,
    af,
):
    """Validate method-specific boundary conditions and apply defaults."""
    if method == "linear":
        if any(value is not None for value in (v0, vf, a0, af)):
            raise ValueError(
                "linear interpolation supports endpoint positions only; "
                "velocity and acceleration boundary conditions must be None."
            )
        return None, None, None, None

    if method == "cubic":
        if a0 is not None or af is not None:
            raise ValueError(
                "cubic interpolation does not support acceleration "
                "boundary conditions."
            )
        zeros = np.zeros(dimension, dtype=float)
        return (
            zeros.copy() if v0 is None else v0,
            zeros.copy() if vf is None else vf,
            None,
            None,
        )

    zeros = np.zeros(dimension, dtype=float)
    return (
        zeros.copy() if v0 is None else v0,
        zeros.copy() if vf is None else vf,
        zeros.copy() if a0 is None else a0,
        zeros.copy() if af is None else af,
    )


def _polynomial_trajectory(
    x0,
    xf,
    t,
    *,
    method,
    v0=None,
    vf=None,
    a0=None,
    af=None,
):
    """Evaluate linear, cubic, or quintic vector polynomials."""
    method = _normalize_method(method)
    time = _prepare_time(t)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    xf = np.asarray(xf, dtype=float).reshape(-1)

    if x0.shape != xf.shape or x0.size < 1:
        raise ValueError("x0 and xf must have the same nonzero dimension.")

    v0, vf, a0, af = _prepare_boundary_conditions(
        x0.size,
        method=method,
        v0=v0,
        vf=vf,
        a0=a0,
        af=af,
    )

    duration = float(time[-1] - time[0])
    tau = ((time - time[0]) / duration)[:, None]
    delta = xf - x0

    if method == "linear":
        x = x0 + tau * delta
        xd = np.broadcast_to(delta / duration, x.shape).copy()
        xdd = np.zeros_like(x)
        return time, x, xd, xdd

    if method == "cubic":
        c0 = x0
        c1 = duration * v0
        c2 = 3.0 * delta - duration * (2.0 * v0 + vf)
        c3 = -2.0 * delta + duration * (v0 + vf)

        x = c0 + c1 * tau + c2 * tau**2 + c3 * tau**3
        xd = (c1 + 2.0 * c2 * tau + 3.0 * c3 * tau**2) / duration
        xdd = (2.0 * c2 + 6.0 * c3 * tau) / duration**2
        return time, x, xd, xdd

    c0 = x0
    c1 = duration * v0
    c2 = 0.5 * duration**2 * a0
    c3 = (
        10.0 * delta
        - 6.0 * duration * v0
        - 4.0 * duration * vf
        - 1.5 * duration**2 * a0
        + 0.5 * duration**2 * af
    )
    c4 = (
        -15.0 * delta
        + 8.0 * duration * v0
        + 7.0 * duration * vf
        + 1.5 * duration**2 * a0
        - duration**2 * af
    )
    c5 = (
        6.0 * delta
        - 3.0 * duration * v0
        - 3.0 * duration * vf
        - 0.5 * duration**2 * a0
        + 0.5 * duration**2 * af
    )

    x = (
        c0
        + c1 * tau
        + c2 * tau**2
        + c3 * tau**3
        + c4 * tau**4
        + c5 * tau**5
    )
    xd = (
        c1
        + 2.0 * c2 * tau
        + 3.0 * c3 * tau**2
        + 4.0 * c4 * tau**3
        + 5.0 * c5 * tau**4
    ) / duration
    xdd = (
        2.0 * c2
        + 6.0 * c3 * tau
        + 12.0 * c4 * tau**2
        + 20.0 * c5 * tau**3
    ) / duration**2
    return time, x, xd, xdd


@dataclass
class JointTrajectory:
    """Numerical joint-space trajectory sampled at explicit times."""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    method: str

    def __post_init__(self):
        self.t = _prepare_time(self.t)
        self.q = _as_real_float_array(self.q, name="q", ndim=2)
        self.qd = _as_real_float_array(self.qd, name="qd", ndim=2)
        self.qdd = _as_real_float_array(self.qdd, name="qdd", ndim=2)
        self.method = _normalize_method(self.method)

        if self.q.shape != self.qd.shape or self.q.shape != self.qdd.shape:
            raise ValueError("q, qd, and qdd must have identical shapes.")
        if self.q.shape[0] != self.t.size:
            raise ValueError(
                "q, qd, and qdd must have one row per time sample."
            )
        if self.q.shape[1] < 1:
            raise ValueError("Joint trajectories must contain at least one DOF.")

    @property
    def samples(self):
        return self.t.size

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    @property
    def dof(self):
        return self.q.shape[1]

    def __repr__(self):
        return (
            f"JointTrajectory(samples={self.samples}, dof={self.dof}, "
            f"duration={self.duration}, method='{self.method}')"
        )


@dataclass
class PositionTrajectory:
    """Numerical Cartesian-position trajectory sampled at explicit times."""

    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    method: str

    def __post_init__(self):
        self.t = _prepare_time(self.t)
        self.p = _as_real_float_array(self.p, name="p", ndim=2)
        self.v = _as_real_float_array(self.v, name="v", ndim=2)
        self.a = _as_real_float_array(self.a, name="a", ndim=2)
        self.method = _normalize_method(self.method)

        expected = (self.t.size, 3)
        if self.p.shape != expected or self.v.shape != expected or self.a.shape != expected:
            raise ValueError(
                "p, v, and a must all have shape (len(t), 3)."
            )

    @property
    def samples(self):
        return self.t.size

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    def __repr__(self):
        return (
            f"PositionTrajectory(samples={self.samples}, "
            f"duration={self.duration}, method='{self.method}')"
        )


def joint_trajectory(
    q0,
    qf,
    t,
    *,
    method="quintic",
    qd0=None,
    qdf=None,
    qdd0=None,
    qddf=None,
):
    """Generate a numerical point-to-point joint-space trajectory."""
    method = _normalize_method(method)
    q0_vec = _prepare_joint_endpoint(q0, name="q0")
    qf_vec = _prepare_joint_endpoint(qf, name="qf")

    if q0_vec.size != qf_vec.size:
        raise ValueError(
            "q0 and qf must contain the same number of joint coordinates."
        )

    dof = q0_vec.size
    qd0_vec = _prepare_joint_boundary(qd0, dof, name="qd0")
    qdf_vec = _prepare_joint_boundary(qdf, dof, name="qdf")
    qdd0_vec = _prepare_joint_boundary(qdd0, dof, name="qdd0")
    qddf_vec = _prepare_joint_boundary(qddf, dof, name="qddf")

    time, q, qd, qdd = _polynomial_trajectory(
        q0_vec,
        qf_vec,
        t,
        method=method,
        v0=qd0_vec,
        vf=qdf_vec,
        a0=qdd0_vec,
        af=qddf_vec,
    )
    return JointTrajectory(time, q, qd, qdd, method)


def position_trajectory(
    p0,
    pf,
    t,
    *,
    method="quintic",
    v0=None,
    vf=None,
    a0=None,
    af=None,
):
    """Generate a numerical point-to-point Cartesian position trajectory."""
    method = _normalize_method(method)
    p0_vec = _prepare_cartesian_vector(p0, name="p0")
    pf_vec = _prepare_cartesian_vector(pf, name="pf")
    v0_vec = _prepare_cartesian_boundary(v0, name="v0")
    vf_vec = _prepare_cartesian_boundary(vf, name="vf")
    a0_vec = _prepare_cartesian_boundary(a0, name="a0")
    af_vec = _prepare_cartesian_boundary(af, name="af")

    time, p, v, a = _polynomial_trajectory(
        p0_vec,
        pf_vec,
        t,
        method=method,
        v0=v0_vec,
        vf=vf_vec,
        a0=a0_vec,
        af=af_vec,
    )
    return PositionTrajectory(time, p, v, a, method)
