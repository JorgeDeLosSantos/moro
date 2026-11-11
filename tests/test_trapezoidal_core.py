"""Analytic tests for the private T-04/B1 scalar motion-profile core."""

import dataclasses

import numpy as np
import pytest

from moro._trapezoidal import (
    _ScalarTrapezoidalPlan,
    _evaluate_trapezoidal_phases,
    _resolve_trapezoidal_phases,
)


def test_trapezoidal_known_phases_and_right_hand_acceleration():
    plan = _resolve_trapezoidal_phases(v_max=0.5, a_max=5 / 6)
    assert dataclasses.is_dataclass(plan)
    assert plan.kind == "trapezoidal"
    assert plan.t_accel == pytest.approx(0.6)
    assert plan.t_cruise == pytest.approx(1.4)
    assert plan.t_decel == pytest.approx(0.6)
    assert plan.duration == pytest.approx(2.6)
    assert plan.minimum_duration == pytest.approx(2.6)
    assert plan.peak_speed == pytest.approx(0.5)
    assert plan.acceleration == pytest.approx(5 / 6)

    times = [0.0, 0.3, plan.t_accel, 1.3,
             plan.t_accel + plan.t_cruise, 2.3, plan.duration]
    s, sd, sdd = _evaluate_trapezoidal_phases(plan, times)
    np.testing.assert_allclose(s, [
        0.0, 0.0375, 0.15, 0.5, 0.85, 0.9625, 1.0,
    ], atol=1e-14)
    np.testing.assert_allclose(sd, [
        0.0, 0.25, 0.5, 0.5, 0.5, 0.25, 0.0,
    ], atol=1e-14)
    np.testing.assert_allclose(sdd, [
        5 / 6, 5 / 6, 0.0, 0.0, -5 / 6, -5 / 6, -5 / 6,
    ], atol=1e-14)


def test_triangular_profile_and_peak_boundary_convention():
    plan = _resolve_trapezoidal_phases(v_max=2.0, a_max=1.0)
    assert plan.kind == "triangular"
    assert plan.t_accel == pytest.approx(1.0)
    assert plan.t_cruise == 0.0
    assert plan.duration == pytest.approx(2.0)
    assert plan.peak_speed == pytest.approx(1.0)
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    s, sd, sdd = _evaluate_trapezoidal_phases(plan, times)
    np.testing.assert_allclose(s, [0, 0.125, 0.5, 0.875, 1])
    np.testing.assert_allclose(sd, [0, 0.5, 1, 0.5, 0])
    np.testing.assert_allclose(sdd, [1, 1, -1, -1, -1])


@pytest.mark.parametrize(
    "speed,accel,kind",
    [
        (1.0, 1.0, "triangular"),
        (0.2, 1.0, "trapezoidal"),
        (5.0, 2.0, "triangular"),
        (0.5, 0.25, "triangular"),
        (0.25, 0.25, "trapezoidal"),
    ],
)
def test_auto_selection_and_exact_normalized_endpoints(speed, accel, kind):
    plan = _resolve_trapezoidal_phases(speed, accel)
    assert plan.kind == kind
    assert plan.t_accel + plan.t_cruise + plan.t_decel == pytest.approx(
        plan.duration
    )
    samples = np.linspace(0.0, plan.duration, 101)
    s, sd, sdd = _evaluate_trapezoidal_phases(plan, samples)
    np.testing.assert_allclose(s[[0, -1]], [0.0, 1.0], atol=0)
    np.testing.assert_allclose(sd[[0, -1]], [0.0, 0.0], atol=0)
    assert np.min(np.diff(s)) >= -1e-14
    assert np.min(s) >= -1e-14 and np.max(s) <= 1.0 + 1e-14
    assert np.max(sd) <= speed * (1 + 1e-12)
    assert np.max(np.abs(sdd)) <= accel * (1 + 1e-12)


@pytest.mark.parametrize("speed,accel", [(0.5, 5/6), (2.0, 1.0)])
def test_time_scaling_preserves_positions_and_scales_derivatives(speed, accel):
    base = _resolve_trapezoidal_phases(speed, accel)
    scaled = _resolve_trapezoidal_phases(
        speed, accel, duration=2 * base.duration
    )
    assert scaled.kind == base.kind
    assert scaled.duration == pytest.approx(2 * base.duration)
    assert scaled.minimum_duration == pytest.approx(base.duration)
    assert scaled.t_accel == pytest.approx(2 * base.t_accel)
    assert scaled.t_cruise == pytest.approx(2 * base.t_cruise)
    assert scaled.peak_speed == pytest.approx(base.peak_speed / 2)
    assert scaled.acceleration == pytest.approx(base.acceleration / 4)

    times = np.array([
        0.0,
        base.t_accel / 2,
        base.t_accel,
        base.t_accel + base.t_cruise / 2,
        base.t_accel + base.t_cruise,
        base.duration,
    ])
    s0, v0, a0 = _evaluate_trapezoidal_phases(base, times)
    s1, v1, a1 = _evaluate_trapezoidal_phases(scaled, 2 * times)
    np.testing.assert_allclose(s1, s0, atol=1e-14)
    np.testing.assert_allclose(v1, v0 / 2, atol=1e-14)
    np.testing.assert_allclose(a1, a0 / 4, atol=1e-14)


def test_duration_minimum_is_preserved_and_shorter_duration_rejected():
    plan = _resolve_trapezoidal_phases(0.5, 5/6)
    same = _resolve_trapezoidal_phases(
        0.5, 5/6, duration=plan.minimum_duration
    )
    assert plan == same
    with pytest.raises(ValueError, match="minimum feasible duration"):
        _resolve_trapezoidal_phases(
            0.5, 5/6, duration=0.9 * plan.minimum_duration
        )


def test_stationary_hold_has_only_cruise_time_and_zero_progress():
    plan = _resolve_trapezoidal_phases(
        v_max=1.0, a_max=2.0, duration=3.0, stationary=True
    )
    assert plan.kind == "stationary"
    assert plan.minimum_duration is None
    assert plan.t_accel == 0
    assert plan.t_decel == 0
    assert plan.t_cruise == plan.duration == 3.0
    assert plan.peak_speed == plan.acceleration == 0.0
    s, sd, sdd = _evaluate_trapezoidal_phases(
        plan, [0, 0.2, 2.8, 3.0]
    )
    np.testing.assert_array_equal(s, 0.0)
    np.testing.assert_array_equal(sd, 0.0)
    np.testing.assert_array_equal(sdd, 0.0)
    with pytest.raises(ValueError, match="stationary hold requires"):
        _resolve_trapezoidal_phases(1.0, 2.0, stationary=True)


def test_scalar_profile_is_immutable():
    plan = _resolve_trapezoidal_phases(1.0, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.duration = 9.0


@pytest.mark.parametrize(
    "name,value",
    [
        ("v_max", 0.0),
        ("v_max", -1.0),
        ("v_max", np.nan),
        ("v_max", np.inf),
        ("v_max", True),
        ("v_max", 1 + 2j),
        ("v_max", [1.0]),
        ("a_max", 0.0),
        ("a_max", np.inf),
        ("a_max", False),
        ("a_max", "3"),
        ("duration", -1.0),
        ("duration", 0.0),
        ("duration", np.nan),
        ("duration", np.inf),
        ("duration", True),
        ("duration", [2.0]),
    ],
)
def test_invalid_scalar_limits_and_duration(name, value):
    kwargs = {"v_max": 1.0, "a_max": 1.0, "duration": 3.0}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        _resolve_trapezoidal_phases(**kwargs)


@pytest.mark.parametrize("bad", [None, 1, "yes", [True]])
def test_stationary_flag_requires_boolean(bad):
    with pytest.raises(ValueError, match="stationary"):
        _resolve_trapezoidal_phases(1.0, 1.0, stationary=bad)


@pytest.mark.parametrize(
    "times",
    [
        -0.1,
        [[0.0, 1.0]],
        [np.nan],
        [np.inf],
        [-1.0],
        [5.0],
        [1 + 2j],
        ["wrong"],
    ],
)
def test_evaluator_rejects_invalid_times(times):
    plan = _resolve_trapezoidal_phases(0.5, 5/6)
    with pytest.raises(ValueError, match="elapsed_times"):
        _evaluate_trapezoidal_phases(plan, times)


def test_evaluator_accepts_nonuniform_and_unsorted_times():
    plan = _resolve_trapezoidal_phases(0.5, 5/6)
    t = np.array([plan.duration, 0.4, 0.0, 2.1])
    s, sd, sdd = _evaluate_trapezoidal_phases(plan, t)
    assert s.shape == sd.shape == sdd.shape == (4,)
    assert s[0] == 1.0
    assert s[2] == 0.0


def test_evaluator_requires_phase_plan():
    with pytest.raises(TypeError, match="_ScalarTrapezoidalPlan"):
        _evaluate_trapezoidal_phases(None, [0, 1])


@pytest.mark.parametrize(
    "velocity,acceleration",
    [
        (1e-150, 1e-300),
        (1e150, 1e300),
    ],
)
def test_extreme_but_representable_triangles(velocity, acceleration):
    plan = _resolve_trapezoidal_phases(velocity, acceleration)
    assert plan.kind == "triangular"
    assert np.isfinite(plan.duration)
    assert np.isfinite(plan.peak_speed)
    t = [0.0, plan.t_accel, plan.duration]
    s, sd, sdd = _evaluate_trapezoidal_phases(plan, t)
    np.testing.assert_allclose(s, [0, 0.5, 1], atol=1e-14)
    assert np.isfinite(sd).all() and np.isfinite(sdd).all()


def test_nonrepresentable_time_scales_rejected():
    with pytest.raises(ValueError):
        _resolve_trapezoidal_phases(1e-300, 1e300)
    with pytest.raises(ValueError):
        _resolve_trapezoidal_phases(1e154, 1e308, duration=1e308)
