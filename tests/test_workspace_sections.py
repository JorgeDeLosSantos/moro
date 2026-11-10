"""Numerical Cartesian workspace-section regression tests (API-003/D3)."""

import numpy as np
import pytest

from moro.workspace import Workspace, WorkspaceSection, section_workspace


def make_workspace():
    # Deliberately include duplicate Cartesian points with distinct q rows.
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.1],
        [1.0, 2.0, 0.1],
        [2.0, 0.0, 0.2],
        [3.0, 1.0, -0.1],
        [4.0, 0.0, 0.3],
    ])
    configurations = np.column_stack([
        np.linspace(-0.8, 0.8, len(points)),
        np.linspace(0.0, 0.5, len(points)),
    ])
    return Workspace(points, configurations, ((-1, 1), (0, 1)), seed=7)


@pytest.mark.parametrize("plane,level,thickness,expected", [
    ("xy", 0.1, 0.2, [0, 1, 2, 3]),
    ("xz", 0.0, 0.0, [0, 3, 5]),
    ("yz", 1.0, 0.0, [1, 2]),
    ("xy", 0.3, 0.0, [5]),
    ("xy", 99.0, 0.1, []),
])
def test_section_filters_preserve_row_indices_and_coordinates(
    plane, level, thickness, expected
):
    source = make_workspace()
    result = section_workspace(
        source, plane=plane, level=level, thickness=thickness
    )
    np.testing.assert_array_equal(result.indices, expected)
    np.testing.assert_array_equal(result.points, source.points[expected])
    np.testing.assert_array_equal(
        result.configurations, source.configurations[expected]
    )
    assert result.points.shape == (len(expected), 3)
    assert result.configurations.shape == (len(expected), source.dof)
    assert result.samples == len(expected)
    assert result.dof == source.dof
    assert result.source_samples == source.samples
    assert (result.plane, result.level, result.thickness) == (
        plane, level, thickness
    )


def test_default_plane_and_inclusive_slab_endpoints():
    source = make_workspace()
    s = section_workspace(source, level=0.1, thickness=0.2)
    assert s.plane == "xy"
    np.testing.assert_array_equal(s.indices, [0, 1, 2, 3])


def test_independent_data_and_no_reordering_or_deduplication():
    source = make_workspace()
    initial_points = source.points.copy()
    initial_configurations = source.configurations.copy()
    section = section_workspace(source, plane="yz", level=1, thickness=0)
    np.testing.assert_array_equal(section.indices, [1, 2])
    np.testing.assert_array_equal(section.points[0], section.points[1])
    section.points[0, 0] = -100
    section.configurations[0, 0] = 0.9
    section.indices[0] = 0
    np.testing.assert_array_equal(source.points, initial_points)
    np.testing.assert_array_equal(source.configurations, initial_configurations)


@pytest.mark.parametrize("plane", [None, "", "xyz", "XY", 2, "arbitrary"])
def test_invalid_plane(plane):
    with pytest.raises(ValueError, match="plane"):
        section_workspace(make_workspace(), plane=plane, level=0, thickness=1)


@pytest.mark.parametrize("value", [None, True, False, "0", np.nan, np.inf,
                                     -np.inf, 1 + 0j, [1], object()])
@pytest.mark.parametrize("field", ["level", "thickness"])
def test_invalid_real_scalar_inputs(value, field):
    kw = {"level": 0.0, "thickness": 1.0}
    kw[field] = value
    with pytest.raises(ValueError, match=field):
        section_workspace(make_workspace(), **kw)


def test_negative_thickness_rejected_and_type_checked():
    with pytest.raises(ValueError, match="thickness"):
        section_workspace(make_workspace(), level=0, thickness=-0.01)
    with pytest.raises(TypeError, match="Workspace"):
        section_workspace(object(), level=0, thickness=1)


def test_direct_section_validation_and_empty_section():
    empty = section_workspace(make_workspace(), plane="xy", level=12, thickness=0)
    assert isinstance(empty, WorkspaceSection)
    assert empty.samples == 0
    assert empty.indices.shape == (0,)
    assert empty.points.shape == (0, 3)
    assert empty.configurations.shape == (0, 2)

    kwargs = dict(
        plane="xy", level=0, thickness=0, indices=np.array([0, 2]),
        points=np.zeros((2, 3)), configurations=np.zeros((2, 2)),
        source_samples=3,
    )
    valid = WorkspaceSection(**kwargs)
    assert valid.samples == 2 and valid.source_samples == 3

    for changes in [
        {"indices": [0, 0]},
        {"indices": [2, 0]},
        {"indices": [0, 3]},
        {"indices": [0.0, 2.0]},
        {"indices": [True, False]},
        {"indices": [[0], [2]]},
        {"points": np.zeros((1, 3))},
        {"points": np.zeros((2, 2))},
        {"configurations": np.zeros((1, 2))},
        {"configurations": np.zeros((2, 0))},
        {"source_samples": 0},
        {"source_samples": True},
    ]:
        with pytest.raises(ValueError):
            WorkspaceSection(**(kwargs | changes))


def test_section_preserves_generated_grid_row_order():
    from moro.abc import q1, q2
    from moro.core import Robot
    from moro.workspace import sample_workspace

    robot = Robot((1.0, 0, 0, q1, "r"), (0, 0, q2, 0, "p"))
    robot.joint_limits = [(-1, 1), (0, 0.5)]
    sampled = sample_workspace(robot, method="grid", grid_shape=(3, 5))
    section = section_workspace(
        sampled, plane="xy", level=0.25, thickness=0
    )
    np.testing.assert_array_equal(section.indices, [2, 7, 12])
    np.testing.assert_array_equal(
        section.configurations, sampled.configurations[[2, 7, 12]]
    )
