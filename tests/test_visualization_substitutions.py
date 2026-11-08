"""API-001/A1: private visualization input normalization, without public dispatch."""

import numpy as np
import pytest
import sympy as sp

from moro import Robot
from moro.abc import l1, l2, q1, q2, t
from moro.visualization.evaluation import evaluate_robot
from moro.visualization.substitutions import (
    _get_visualization_parameters,
    _normalize_visualization_configurations,
    _normalize_visualization_parameters,
    _normalize_visualization_substitutions,
)


@pytest.fixture
def rr():
    return Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))


def test_dh_parameter_discovery_ignores_dynamic_joint_time_and_dynamics(rr):
    assert q1.free_symbols == {t}
    assert _get_visualization_parameters(rr) == frozenset((l1, l2))
    rr.masses = [sp.Symbol("m_a"), sp.Symbol("m_b")]
    rr.gravity = (0, -sp.Symbol("g"), 0)
    assert _get_visualization_parameters(rr) == frozenset((l1, l2))


def test_mixed_rp_and_symbolic_dh_expressions():
    L, h, alpha = sp.symbols("L h alpha")
    robot = Robot(
        (L / 2, alpha, h + 1, q1, "r"),
        (0, 0, q2, 0, "p"),
    )
    assert robot.qs == [q1, q2]
    assert _get_visualization_parameters(robot) == frozenset((L, h, alpha))

    substitutions = _normalize_visualization_substitutions(
        robot, [0.5, 0.3],
        parameters={L: 2, h: 0.2, alpha: sp.pi / 2},
    )
    assert substitutions[q1] == 0.5
    assert substitutions[q2] == 0.3
    assert substitutions[L] == 2.0
    assert substitutions[h] == 0.2
    assert substitutions[alpha] == pytest.approx(np.pi / 2)
    scene = evaluate_robot(robot, substitutions)
    assert len(scene.frames) == 3


def test_complete_legacy_mapping_and_partial_mapping_with_parameters(rr):
    legacy = {q1: 0.5, q2: -0.3, l1: 1, l2: 0.8}
    normalized = _normalize_visualization_substitutions(rr, legacy)
    assert normalized == {q1: 0.5, q2: -0.3, l1: 1.0, l2: 0.8}
    separate = _normalize_visualization_substitutions(
        rr, {q1: 0.5, q2: -0.3}, {l1: 1.0, l2: 0.8}
    )
    assert separate == normalized


def test_numeric_configuration_accepts_numpy_and_sympy_pi(rr):
    values = _normalize_visualization_substitutions(
        rr, np.array([sp.pi / 6, 0], dtype=object),
        parameters={l1: sp.sqrt(2), l2: np.float64(0.8)},
    )
    assert values[q1] == pytest.approx(np.pi / 6)
    assert values[l1] == pytest.approx(np.sqrt(2))


def test_scalar_symbol_joint_coordinate_is_supported():
    q, L = sp.symbols("q L", real=True)
    robot = Robot((L, 0, 0, q, "r"))
    assert _get_visualization_parameters(robot) == frozenset((L,))
    assert _normalize_visualization_substitutions(
        robot, [0.5], parameters={L: 2}
    ) == {L: 2.0, q: 0.5}


def test_no_parameters_for_fully_numeric_model():
    robot = Robot((1.0, 0, 0, q1, "r"))
    assert _get_visualization_parameters(robot) == frozenset()
    assert _normalize_visualization_substitutions(robot, [0.2]) == {
        q1: 0.2
    }


def test_reject_missing_coordinates_and_named_parameters(rr):
    with pytest.raises(ValueError, match="Missing joint coordinates"):
        _normalize_visualization_substitutions(
            rr, {q1: 0.2, l1: 1.0, l2: 0.8}
        )
    with pytest.raises(ValueError, match="Missing model parameters: l_2"):
        _normalize_visualization_substitutions(
            rr, [0.2, -0.3], parameters={l1: 1}
        )


def test_extra_legacy_mapping_keys_are_permitted_but_explicit_unknown_rejected(rr):
    extra = sp.Symbol("unused")
    complete = {
        q1: 0, q2: 0, l1: 1, l2: 1, extra: 2,
    }
    assert _normalize_visualization_substitutions(rr, complete)[extra] == 2.0
    with pytest.raises(ValueError, match="Unknown model parameter: unused"):
        _normalize_visualization_substitutions(
            rr, [0, 0], parameters={l1: 1, l2: 1, extra: 2}
        )


def test_joint_cannot_be_passed_in_explicit_parameters(rr):
    with pytest.raises(ValueError, match="parameters keys must be atomic"):
        _normalize_visualization_substitutions(
            rr, [0, 0], parameters={q1: 0, l1: 1, l2: 1}
        )


def test_explicit_parameters_and_configuration_never_override(rr):
    with pytest.raises(ValueError, match="Duplicate substitutions"):
        _normalize_visualization_substitutions(
            rr, {q1: 0.2, q2: 0, l1: 1.0}, parameters={l1: 1, l2: 2}
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1j, True, "1.0"])
def test_reject_nonfinite_nonreal_or_nonnumeric_parameter(rr, bad):
    with pytest.raises(ValueError, match="finite real"):
        _normalize_visualization_substitutions(
            rr, [0, 0], parameters={l1: bad, l2: 1}
        )


@pytest.mark.parametrize("bad", [
    [0, float("nan")],
    [0, float("inf")],
    [0, True],
    [0, "0.5"],
    [0, 1j],
    [0],
    [[0, 0]],
])
def test_reject_invalid_numeric_joint_vectors(rr, bad):
    with pytest.raises(ValueError):
        _normalize_visualization_substitutions(
            rr, bad, parameters={l1: 1, l2: 1}
        )


def test_mapping_values_and_parameter_keys_are_checked(rr):
    with pytest.raises(ValueError, match="Configuration value"):
        _normalize_visualization_substitutions(
            rr, {q1: float("nan"), q2: 0},
            parameters={l1: 1, l2: 2},
        )
    with pytest.raises(ValueError, match="parameters must be a mapping"):
        _normalize_visualization_substitutions(rr, [0, 0], parameters=[l1, l2])
    with pytest.raises(ValueError, match="parameters keys must be atomic"):
        _normalize_visualization_parameters(rr, {l1 + l2: 1})
    with pytest.raises(ValueError, match="Configuration mappings"):
        _normalize_visualization_substitutions(
            rr, {"q1": 0, q2: 0}, parameters={l1: 1, l2: 2}
        )


def test_dh_expressions_must_evaluate_to_finite_real():
    L = sp.Symbol("L")
    robot = Robot((1 / L, 0, 0, q1, "r"))
    with pytest.raises(ValueError, match="must evaluate to a finite real"):
        _normalize_visualization_substitutions(
            robot, [0.5], parameters={L: 0}
        )


def test_explicit_time_in_dh_is_not_a_constant_parameter():
    robot = Robot((t, 0, 0, q1, "r"))
    assert t not in _get_visualization_parameters(robot)
    with pytest.raises(ValueError, match="depends on explicit time"):
        _normalize_visualization_substitutions(robot, [0.5])


def test_unknown_dynamic_dh_function_needs_complete_mapping():
    external = sp.Function("external")(t)
    robot = Robot((external, 0, 0, q1, "r"))
    with pytest.raises(ValueError, match="Unresolved time-dependent DH"):
        _normalize_visualization_substitutions(robot, [0.5])
    result = _normalize_visualization_substitutions(
        robot, {q1: 0.5, external: 1.0}
    )
    assert result[external] == 1.0


def test_composite_joint_expression_mapping_works_but_vector_does_not():
    L, offset = sp.symbols("L offset", real=True)
    robot = Robot((L, 0, 0, q1 + offset, "r"))
    assert offset in _get_visualization_parameters(robot)
    combined = _normalize_visualization_substitutions(
        robot, {q1: 0.4}, parameters={L: 1.0, offset: 0.2}
    )
    assert combined[q1] == 0.4
    assert evaluate_robot(robot, combined)
    with pytest.raises(ValueError, match="composite joint expressions"):
        _normalize_visualization_substitutions(
            robot, [0.4], parameters={L: 1, offset: 0.2}
        )


def test_composite_joint_with_plain_symbols_requires_full_mapping():
    q, offset = sp.symbols("q offset")
    L = sp.Symbol("L")
    robot = Robot((L, 0, 0, q + offset, "r"))
    assert _normalize_visualization_substitutions(
        robot, {q: 0.4, offset: 0.2, L: 1.0}
    )[q] == 0.4
    with pytest.raises(ValueError, match="Cannot separate parameters"):
        _normalize_visualization_substitutions(
            robot, {q: 0.4}, parameters={offset: 0.2, L: 1}
        )


def test_symbolic_dh_fragment_key_does_not_replace_required_symbol():
    L = sp.Symbol("L")
    robot = Robot((L / 2, 0, 0, q1, "r"))
    # The expanded transform may contain L*cos(q1)/2, not the literal L/2.
    # Accepting L/2 as the only substitution could thus fail downstream.
    with pytest.raises(ValueError, match="Missing model parameters: L"):
        _normalize_visualization_substitutions(
            robot, {L / 2: 1.0, q1: 0.5}
        )


def test_animation_normalizes_parameters_once_and_returns_separate_mappings(rr):
    original_parameters = {l1: 1.0, l2: 0.8}
    trajectory = np.array([[0.0, 0.2], [0.1, 0.3], [0.2, 0.4]])
    trajectory_before = trajectory.copy()
    frames = _normalize_visualization_configurations(
        rr, trajectory, parameters=original_parameters
    )
    assert len(frames) == 3
    assert all(frame[l1] == 1.0 for frame in frames)
    assert [frame[q1] for frame in frames] == [0.0, 0.1, 0.2]
    assert frames[0] is not frames[1]
    frames[0][l1] = 100.0
    assert frames[1][l1] == 1.0
    assert original_parameters == {l1: 1.0, l2: 0.8}
    np.testing.assert_array_equal(trajectory, trajectory_before)


def test_animation_accepts_generator_of_legacy_mappings(rr):
    configs = (
        {q1: float(i), q2: 0, l1: 1, l2: 1}
        for i in range(3)
    )
    frames = _normalize_visualization_configurations(rr, configs)
    assert [frame[q1] for frame in frames] == [0.0, 1.0, 2.0]


def test_animation_reports_failing_frame_and_empty_input(rr):
    with pytest.raises(ValueError, match="Frame 1: Missing joint coordinates"):
        _normalize_visualization_configurations(
            rr, [
                {q1: 0, q2: 0},
                {q1: 0.1},
            ],
            parameters={l1: 1, l2: 1},
        )
    with pytest.raises(ValueError, match="at least one frame"):
        _normalize_visualization_configurations(
            rr, [], parameters={l1: 1, l2: 1}
        )
    with pytest.raises(ValueError, match="sequence, not a mapping"):
        _normalize_visualization_configurations(
            rr, {q1: 0, q2: 0}, parameters={l1: 1, l2: 1}
        )
