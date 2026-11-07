"""Private analytic core for synchronized trapezoidal joint motion.

The unit-distance progress profile is robot-independent. The public joint
trajectory generator will map progress and its physical-time derivatives to
joint coordinates in a later increment (0.7.0 T-04/B2).
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class _ScalarTrapezoidalPlan:
    """Effective phase parameters for normalized progress from 0 to 1."""

    kind: str
    t_accel: float
    t_cruise: float
    t_decel: float
    duration: float
    minimum_duration: float | None
    peak_speed: float
    acceleration: float


def _positive_finite_scalar(value, *, name):
    """Validate strictly positive scalar time or kinematic limit values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite positive real scalar."
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar.")
    return number


def _resolve_trapezoidal_phases(
    v_max, a_max, *, duration=None, stationary=False
):
    """Solve symmetric unit-progress phases, optionally stretching time.

    For a nonstationary move, s runs from 0 to 1 with zero endpoint speed.
    For a stationary hold, s remains zero and duration must be explicit.

    Both limits are bounds on physical-time derivatives of scalar progress,
    *not* necessarily on any one joint's velocity or acceleration.
    """
    velocity_limit = _positive_finite_scalar(v_max, name="v_max")
    acceleration_limit = _positive_finite_scalar(a_max, name="a_max")

    if duration is not None:
        duration = _positive_finite_scalar(duration, name="duration")

    if not isinstance(stationary, (bool, np.bool_)):
        raise ValueError("stationary must be a boolean.")

    if stationary:
        if duration is None:
            raise ValueError("A stationary hold requires an explicit duration.")
        return _ScalarTrapezoidalPlan(
            kind="stationary",
            t_accel=0.0,
            t_cruise=duration,
            t_decel=0.0,
            duration=duration,
            minimum_duration=None,
            peak_speed=0.0,
            acceleration=0.0,
        )

    triangular_peak = math.sqrt(acceleration_limit)
    if velocity_limit >= triangular_peak:
        kind = "triangular"
        peak = triangular_peak
        accel_time = 1.0 / triangular_peak
        cruise_time = 0.0
    else:
        kind = "trapezoidal"
        peak = velocity_limit
        accel_time = peak / acceleration_limit
        if accel_time <= 0.0:
            raise ValueError(
                "Profile acceleration phase is too short to represent."
            )
        # 1 = peak * (accel_time + cruise_time).
        cruise_distance = 1.0 - peak * accel_time
        cruise_time = cruise_distance / peak
        if cruise_time == 0.0:
            kind = "triangular"

    min_duration = 2.0 * accel_time + cruise_time
    if not (
        math.isfinite(min_duration)
        and min_duration > 0.0
        and math.isfinite(accel_time)
        and accel_time > 0.0
        and math.isfinite(cruise_time)
        and cruise_time >= 0.0
    ):
        raise ValueError(
            "Limits produce nonrepresentable or nonfinite profile phase times."
        )

    if duration is None:
        requested_duration = min_duration
    else:
        # Treat discrepancies at the scale of arithmetic rounding as equality,
        # without accepting materially infeasible requested durations.
        if duration < min_duration:
            if not math.isclose(duration, min_duration, rel_tol=8e-16):
                raise ValueError(
                    f"duration must be >= the minimum feasible duration "
                    f"({min_duration:.17g})."
                )
            requested_duration = min_duration
        else:
            requested_duration = duration

    alpha = requested_duration / min_duration
    if not math.isfinite(alpha) or alpha < 1.0:
        raise ValueError("Requested duration produces an invalid time scale.")

    effective_accel_time = accel_time * alpha
    effective_cruise_time = cruise_time * alpha
    effective_peak = peak / alpha
    # Divide sequentially to avoid overflow in alpha**2.
    effective_acceleration = (acceleration_limit / alpha) / alpha

    if not all(
        math.isfinite(x)
        for x in (
            effective_accel_time,
            effective_cruise_time,
            effective_peak,
            effective_acceleration,
        )
    ) or any(
        x <= 0.0 for x in (
            effective_accel_time, effective_peak, effective_acceleration
        )
    ):
        raise ValueError(
            "Requested duration produces nonrepresentable profile values."
        )

    if effective_accel_time > requested_duration / 2.0:
        raise ValueError(
            "Profile phases are not representable at this duration."
        )

    # Retain the triangular convention exactly when the central phase is zero.
    if kind == "triangular":
        effective_accel_time = requested_duration / 2.0
        effective_cruise_time = 0.0
    else:
        # Remove a tiny accumulated roundoff from the phase sum.
        effective_cruise_time = (
            requested_duration - 2.0 * effective_accel_time
        )
        if effective_cruise_time < 0.0:
            raise ValueError("Computed cruise phase is negative.")

    return _ScalarTrapezoidalPlan(
        kind=kind,
        t_accel=effective_accel_time,
        t_cruise=effective_cruise_time,
        t_decel=effective_accel_time,
        duration=requested_duration,
        minimum_duration=min_duration,
        peak_speed=effective_peak,
        acceleration=effective_acceleration,
    )


def _evaluate_trapezoidal_phases(plan, elapsed_times):
    """Evaluate normalized progress, velocity and acceleration analytically.

    The input times are physical-time offsets from the movement start.
    The acceleration at interior phase boundaries belongs to the right
    phase; at the final endpoint the last phase owns the value.
    """
    if not isinstance(plan, _ScalarTrapezoidalPlan):
        raise TypeError("plan must be a _ScalarTrapezoidalPlan.")
    try:
        raw = np.asarray(elapsed_times)
    except Exception as exc:
        raise ValueError("elapsed_times must contain real values.") from exc
    if np.iscomplexobj(raw):
        raise ValueError("elapsed_times must contain real values.")
    try:
        times = np.asarray(elapsed_times, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("elapsed_times must contain real values.") from exc
    if times.ndim != 1:
        raise ValueError("elapsed_times must be a one-dimensional array.")
    if not np.all(np.isfinite(times)):
        raise ValueError("elapsed_times must contain finite values.")
    if np.any(times < 0.0) or np.any(times > plan.duration):
        raise ValueError("elapsed_times must be within [0, duration].")

    progress = np.zeros_like(times)
    velocity = np.zeros_like(times)
    acceleration = np.zeros_like(times)
    if plan.kind == "stationary":
        return progress, velocity, acceleration

    ta = plan.t_accel
    tc = plan.t_cruise
    total = plan.duration
    peak = plan.peak_speed
    rate = plan.acceleration

    accelerating = times < ta
    cruising = (times >= ta) & (times < ta + tc)
    decelerating = ~(accelerating | cruising)

    tau = times[accelerating]
    velocity[accelerating] = rate * tau
    progress[accelerating] = 0.5 * velocity[accelerating] * tau
    acceleration[accelerating] = rate

    tau = times[cruising] - ta
    velocity[cruising] = peak
    progress[cruising] = 0.5 * peak * ta + peak * tau

    remaining = total - times[decelerating]
    velocity[decelerating] = rate * remaining
    progress[decelerating] = 1.0 - 0.5 * velocity[decelerating] * remaining
    acceleration[decelerating] = -rate

    progress[times == 0.0] = 0.0
    progress[times == total] = 1.0
    velocity[(times == 0.0) | (times == total)] = 0.0

    if not (
        np.all(np.isfinite(progress))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(acceleration))
    ):
        raise ValueError("Profile evaluation produced nonfinite values.")
    return progress, velocity, acceleration
