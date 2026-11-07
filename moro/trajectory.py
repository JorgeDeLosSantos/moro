"""Numerical point-to-point and multi-waypoint trajectory generation.

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
    "joint_waypoint_trajectory",
    "position_waypoint_trajectory",
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


def _validated_segment_result(time, x, xd, xdd):
    """Reject nonfinite polynomial evaluations before results are assembled."""
    if not all(np.all(np.isfinite(array)) for array in (x, xd, xdd)):
        raise ValueError("Polynomial segment evaluation produced nonfinite values.")
    return time, x, xd, xdd


def _evaluate_polynomial_segment(
    x0,
    xf,
    t0,
    tf,
    sample_times,
    *,
    method,
    v0=None,
    vf=None,
    a0=None,
    af=None,
):
    """Evaluate a segment at arbitrary samples, using its true time bounds.

    Empty or single-sample grids are valid for internal piecewise assembly.
    Unlike point-to-point generation, the sample endpoints need not equal
    the segment endpoints.
    """
    method = _normalize_method(method)
    boundaries = _as_real_float_array(
        [t0, tf], name="segment boundaries", ndim=1
    )
    t0, tf = boundaries
    if tf <= t0:
        raise ValueError("Segment end time must be greater than start time.")
    duration = float(tf - t0)
    if not np.isfinite(duration):
        raise ValueError("Segment duration must be finite.")

    time = _as_real_float_array(sample_times, name="sample_times", ndim=1)
    if np.any(np.diff(time) <= 0):
        raise ValueError("sample_times must be strictly increasing.")
    if time.size and (time[0] < t0 or time[-1] > tf):
        raise ValueError("sample_times must lie within segment boundaries.")

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

    tau = ((time - t0) / duration)[:, None]
    delta = xf - x0

    if method == "linear":
        x = x0 + tau * delta
        xd = np.broadcast_to(delta / duration, x.shape).copy()
        xdd = np.zeros_like(x)
        return _validated_segment_result(time, x, xd, xdd)

    if method == "cubic":
        c0 = x0
        c1 = duration * v0
        c2 = 3.0 * delta - duration * (2.0 * v0 + vf)
        c3 = -2.0 * delta + duration * (v0 + vf)

        x = c0 + c1 * tau + c2 * tau**2 + c3 * tau**3
        xd = (c1 + 2.0 * c2 * tau + 3.0 * c3 * tau**2) / duration
        xdd = (2.0 * c2 + 6.0 * c3 * tau) / duration**2
        return _validated_segment_result(time, x, xd, xdd)

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
    return _validated_segment_result(time, x, xd, xdd)


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
    """Evaluate a point-to-point polynomial on the full supplied time grid."""
    method = _normalize_method(method)
    time = _prepare_time(t)
    return _evaluate_polynomial_segment(
        x0,
        xf,
        time[0],
        time[-1],
        time,
        method=method,
        v0=v0,
        vf=vf,
        a0=a0,
        af=af,
    )


def _prepare_waypoints(waypoints, *, cartesian):
    """Normalize time-major waypoint rows without consulting a Robot."""
    values = _as_real_float_array(waypoints, name="waypoints")
    if cartesian:
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Cartesian waypoints must have shape (K, 3).")
    else:
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError(
                "Joint waypoints must have shape (K, dof), "
                "or (K,) for one degree of freedom."
            )

    if values.shape[0] < 2:
        raise ValueError("waypoints must contain at least two rows.")
    return values


def _prepare_waypoint_times(waypoint_times, count):
    """Validate one finite, strictly increasing arrival time per waypoint."""
    times = _as_real_float_array(
        waypoint_times, name="waypoint_times", ndim=1
    )
    if times.size != count:
        raise ValueError(
            f"waypoint_times must contain {count} times; got {times.size}."
        )
    # Subtracting two finite but extreme values can overflow; treat such
    # durations as invalid input without emitting an avoidable RuntimeWarning.
    with np.errstate(over="ignore", invalid="ignore"):
        durations = np.diff(times)
    if np.any(durations <= 0) or not np.all(np.isfinite(durations)):
        raise ValueError(
            "waypoint_times must be strictly increasing with finite durations."
        )
    return times


def _prepare_waypoint_samples(
    waypoint_times, *, sample_times, samples_per_segment
):
    """Resolve exact sample times or an inclusive grid for each segment."""
    if sample_times is not None and samples_per_segment is not None:
        raise ValueError(
            "sample_times and samples_per_segment cannot both be specified."
        )

    if sample_times is not None:
        times = _as_real_float_array(
            sample_times, name="sample_times", ndim=1
        )
        if times.size < 2:
            raise ValueError("sample_times must contain at least two values.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("sample_times must be strictly increasing.")
        if times[0] != waypoint_times[0] or times[-1] != waypoint_times[-1]:
            raise ValueError(
                "sample_times must begin and end at the first and last "
                "waypoint_times, respectively."
            )
        return times

    count = 51 if samples_per_segment is None else samples_per_segment
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 2
    ):
        raise ValueError("samples_per_segment must be an integer >= 2.")

    segments = [
        np.linspace(t0, tf, count)[:-1]
        for t0, tf in zip(waypoint_times[:-1], waypoint_times[1:])
    ]
    times = np.concatenate((*segments, waypoint_times[-1:]))
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(
            "Automatic sampling requires distinguishable, increasing "
            "floating-point times; reduce samples_per_segment or rescale time."
        )
    return times


def _estimate_shape_preserving_velocities(waypoints, waypoint_times):
    """Estimate PCHIP-style interior slopes with resting endpoint velocities.

    The harmonic mean is evaluated using normalized time weights and the
    smaller slope magnitude to avoid unnecessary intermediate overflow.
    Each coordinate is treated independently; reversals and plateaus stop.
    """
    result = np.zeros_like(waypoints, dtype=float)
    if waypoints.shape[0] <= 2:
        return result

    durations = np.diff(waypoint_times)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        secants = np.diff(waypoints, axis=0) / durations[:, None]
    if not np.all(np.isfinite(secants)):
        raise ValueError(
            "shape_preserving requires finite waypoint secant slopes."
        )

    previous = secants[:-1]
    following = secants[1:]
    same_direction = (
        ((previous > 0) & (following > 0))
        | ((previous < 0) & (following < 0))
    )
    if not np.any(same_direction):
        return result

    # Durations may be very large: rescale before forming the PCHIP
    # weights w1 = 2*h_next + h_prev, w2 = h_next + 2*h_prev.
    h_previous = durations[:-1]
    h_next = durations[1:]
    scale = np.maximum(h_previous, h_next)
    previous_scaled = h_previous / scale
    next_scaled = h_next / scale
    w1 = 2.0 * next_scaled + previous_scaled
    w2 = next_scaled + 2.0 * previous_scaled
    weight = w1 / (w1 + w2)

    magnitudes_previous = np.abs(previous[same_direction])
    magnitudes_next = np.abs(following[same_direction])
    smaller = np.minimum(magnitudes_previous, magnitudes_next)

    # m = 1 / (weight/|d_prev| + (1-weight)/|d_next|).
    # The scaled form avoids reciprocal overflow for tiny secants.
    local_weight = np.broadcast_to(weight[:, None], previous.shape)[
        same_direction
    ]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = (
            local_weight * (smaller / magnitudes_previous)
            + (1.0 - local_weight) * (smaller / magnitudes_next)
        )
        estimated = smaller / denominator

    if not np.all(np.isfinite(estimated)):
        raise ValueError(
            "shape_preserving generated nonfinite waypoint velocities."
        )
    result[1:-1][same_direction] = (
        np.copysign(estimated, previous[same_direction])
    )
    return result


def _prepare_waypoint_derivatives(
    values, waypoint_times, *, method, derivative_policy,
    velocities, accelerations,
):
    """Return shared waypoint derivatives for piecewise Hermite segments."""
    if not isinstance(derivative_policy, str):
        raise ValueError(
            "derivative_policy must be 'zero' or 'shape_preserving'."
        )
    policy = derivative_policy.lower()
    if policy not in ("zero", "shape_preserving"):
        raise ValueError(
            "derivative_policy must be 'zero' or 'shape_preserving'."
        )

    if policy == "shape_preserving":
        if method != "cubic":
            raise ValueError(
                "derivative_policy='shape_preserving' requires method='cubic'."
            )
        if velocities is not None or accelerations is not None:
            raise ValueError(
                "shape_preserving does not accept explicit velocities "
                "or accelerations."
            )
        return _estimate_shape_preserving_velocities(
            values, waypoint_times
        ), None

    if method == "linear":
        if velocities is not None or accelerations is not None:
            raise ValueError(
                "Linear waypoint interpolation does not accept velocities "
                "or accelerations."
            )
        return None, None

    if method == "cubic" and accelerations is not None:
        raise ValueError(
            "Cubic waypoint interpolation does not accept accelerations."
        )

    expected = values.shape

    def normalize(data, name):
        if data is None:
            return np.zeros(expected, dtype=float)
        array = _as_real_float_array(data, name=name)
        if len(expected) == 2 and expected[1] == 1 and array.ndim == 1:
            array = array[:, None]
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        return array

    v = normalize(velocities, "velocities")
    a = normalize(accelerations, "accelerations") if method == "quintic" else None
    return v, a


def _piecewise_polynomial_trajectory(
    waypoints, waypoint_times, sample_times, *,
    method, velocities=None, accelerations=None,
):
    """Evaluate piecewise polynomials with shared waypoint derivatives.

    The right-hand segment owns every interior knot. The final knot belongs
    to the final segment; derivatives there use its left-hand limit.
    """
    if method == "linear":
        durations = np.diff(waypoint_times)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            slopes = np.diff(waypoints, axis=0) / durations[:, None]
        if not np.all(np.isfinite(slopes)):
            raise ValueError("Waypoint segment velocities must be finite.")

    positions = np.empty((sample_times.size, waypoints.shape[1]), dtype=float)
    velocity_samples = np.empty_like(positions)
    acceleration_samples = np.empty_like(positions)

    for i in range(waypoints.shape[0] - 1):
        start = np.searchsorted(sample_times, waypoint_times[i], side="left")
        stop = (
            sample_times.size
            if i == waypoints.shape[0] - 2
            else np.searchsorted(sample_times, waypoint_times[i + 1], side="left")
        )
        kwargs = {}
        if method != "linear":
            kwargs.update(v0=velocities[i], vf=velocities[i + 1])
        if method == "quintic":
            kwargs.update(a0=accelerations[i], af=accelerations[i + 1])

        _, x, xd, xdd = _evaluate_polynomial_segment(
            waypoints[i],
            waypoints[i + 1],
            waypoint_times[i],
            waypoint_times[i + 1],
            sample_times[start:stop],
            method=method,
            **kwargs,
        )
        positions[start:stop] = x
        velocity_samples[start:stop] = xd
        acceleration_samples[start:stop] = xdd

    return sample_times, positions, velocity_samples, acceleration_samples


def _prepare_waypoint_request(
    waypoints,
    waypoint_times,
    *,
    cartesian,
    method,
    derivative_policy,
    velocities,
    accelerations,
    sample_times,
    samples_per_segment,
):
    """Prepare a validated robot-independent piecewise polynomial request."""
    method = _normalize_method(method)
    values = _prepare_waypoints(waypoints, cartesian=cartesian)
    times = _prepare_waypoint_times(waypoint_times, values.shape[0])
    derivatives = _prepare_waypoint_derivatives(
        values,
        times,
        method=method,
        derivative_policy=derivative_policy,
        velocities=velocities,
        accelerations=accelerations,
    )
    samples = _prepare_waypoint_samples(
        times,
        sample_times=sample_times,
        samples_per_segment=samples_per_segment,
    )
    return values, times, samples, derivatives



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

def joint_waypoint_trajectory(
    waypoints,
    waypoint_times,
    *,
    method="quintic",
    derivative_policy="zero",
    velocities=None,
    accelerations=None,
    sample_times=None,
    samples_per_segment=None,
):
    """Generate piecewise linear, cubic or quintic joint-space trajectories.

    The 'zero' policy stops at every waypoint unless complete derivative
    arrays are provided. The cubic-only 'shape_preserving' policy estimates
    interior velocities, with zero velocities at both global endpoints.
    """
    values, times, samples, (v, a) = _prepare_waypoint_request(
        waypoints,
        waypoint_times,
        cartesian=False,
        method=method,
        derivative_policy=derivative_policy,
        velocities=velocities,
        accelerations=accelerations,
        sample_times=sample_times,
        samples_per_segment=samples_per_segment,
    )
    normalized_method = _normalize_method(method)
    t, q, qd, qdd = _piecewise_polynomial_trajectory(
        values, times, samples,
        method=normalized_method, velocities=v, accelerations=a,
    )
    return JointTrajectory(t, q, qd, qdd, normalized_method)


def position_waypoint_trajectory(
    waypoints,
    waypoint_times,
    *,
    method="quintic",
    derivative_policy="zero",
    velocities=None,
    accelerations=None,
    sample_times=None,
    samples_per_segment=None,
):
    """Generate piecewise polynomial Cartesian-position trajectories.

    Cartesian waypoints and derivative matrices must have shape (K, 3).
    The cubic-only 'shape_preserving' policy estimates coordinate-wise
    interior velocities and rests at both endpoints.
    """
    values, times, samples, (v, a) = _prepare_waypoint_request(
        waypoints,
        waypoint_times,
        cartesian=True,
        method=method,
        derivative_policy=derivative_policy,
        velocities=velocities,
        accelerations=accelerations,
        sample_times=sample_times,
        samples_per_segment=samples_per_segment,
    )
    normalized_method = _normalize_method(method)
    t, p, v_samples, a_samples = _piecewise_polynomial_trajectory(
        values, times, samples,
        method=normalized_method, velocities=v, accelerations=a,
    )
    return PositionTrajectory(t, p, v_samples, a_samples, normalized_method)
