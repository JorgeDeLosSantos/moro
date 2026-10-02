"""Cross-cutting release invariants for Moro 0.5.0."""

from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

import moro
from moro import version as version_module
from moro.abc import q1, q2
from moro.core import Robot
from moro.differential_kinematics import __all__ as differential_all
from moro.dynamics import DynamicsSolution, __all__ as dynamics_all
from moro.inverse_kinematics import (
    __all__ as ik_all,
    solve_position_ik,
)
from moro.trajectory import (
    JointTrajectory,
    PositionTrajectory,
    __all__ as trajectory_all,
)
from moro.transformations import (
    __all__ as transformations_all,
    is_homogeneous_transform,
    is_rotation_matrix,
)
from moro.util import is_SE3, is_SO3, ishtm, isrot
from moro.visualization import RobotVisualizer, __all__ as visualization_all
from moro.workspace import Workspace, __all__ as workspace_all


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_package_and_project_versions_are_synchronized():
    project = _project_metadata()

    assert moro.__version__ == version_module.__version__
    assert project["version"] == moro.__version__
    assert moro.__version__ == "0.5.0"


def test_supported_python_and_direct_runtime_dependencies_are_release_ready():
    project = _project_metadata()
    dependencies = {
        dependency.split(";")[0]
        .split("[")[0]
        .split("<")[0]
        .split(">=")[0]
        .split("==")[0]
        .strip()
        .lower()
        for dependency in project["dependencies"]
    }

    assert project["requires-python"] == ">=3.11"
    assert {"sympy", "numpy", "matplotlib", "scipy"} <= dependencies


def test_root_package_remains_intentionally_compact():
    expected = {
        "__version__",
        "Robot",
        "enable_vprinting",
        "rotx",
        "roty",
        "rotz",
        "rot2eul",
        "eul2rot",
        "axa2rot",
        "rot2axa",
        "htmtra",
        "htmrot",
        "rot2htm",
        "rt2htm",
        "htm2rot",
        "htm2tra",
        "invhtm",
        "dh",
    }

    assert set(moro.__all__) == expected
    assert not hasattr(moro, "solve_pose_ik")
    assert not hasattr(moro, "joint_trajectory")
    assert not hasattr(moro, "simulate")
    assert not hasattr(moro, "sample_workspace")


def test_new_modules_export_the_accepted_public_api():
    assert set(differential_all) == {
        "VelocityIKSolution",
        "task_jacobian",
        "cartesian_velocity",
        "singular_values",
        "jacobian_rank",
        "condition_number",
        "is_singular",
        "manipulability",
        "solve_velocity_ik",
    }
    assert set(ik_all) >= {
        "solve_position_ik",
        "solve_position_trajectory",
        "solve_pose_ik",
        "IKSolution",
        "IKTrajectorySolution",
        "PoseIKSolution",
    }
    assert set(trajectory_all) == {
        "JointTrajectory",
        "PositionTrajectory",
        "joint_trajectory",
        "position_trajectory",
    }
    assert set(dynamics_all) == {
        "DynamicsSolution",
        "inverse_dynamics",
        "forward_dynamics",
        "state_derivative",
        "simulate",
    }
    assert set(workspace_all) == {"Workspace", "sample_workspace"}
    assert "plot_workspace" in visualization_all

    for name in (
        "is_rotation_matrix",
        "is_homogeneous_transform",
        "rot2quat",
        "quat2rot",
        "rot2rotvec",
        "rotvec2rot",
        "vex",
    ):
        assert name in transformations_all


def test_descriptive_transform_predicates_and_legacy_wrappers_coexist():
    R = sp.eye(3)
    T = sp.eye(4)

    assert is_rotation_matrix(R) is True
    assert is_homogeneous_transform(T) is True

    for legacy, value in (
        (is_SO3, R),
        (isrot, R),
        (is_SE3, T),
        (ishtm, T),
    ):
        with pytest.warns(DeprecationWarning):
            result = legacy(value)
        assert isinstance(result, bool)
        assert result is True


def test_dynamic_model_compatibility_alias_is_deprecated():
    robot = Robot((0, 0, 0, q1, "r"))
    robot.masses = [1.0]
    robot.cm_positions = [(0.5, 0, 0)]
    robot.inertia_tensors = [sp.diag(0, 0, 0.1)]
    robot.gravity = (0, -9.81, 0)

    matrix_model = robot.dynamic_model()
    equations = robot.euler_lagrange_equations()

    assert isinstance(matrix_model, sp.Equality)
    assert isinstance(equations, list)

    with pytest.warns(DeprecationWarning):
        compatibility_model = robot.dynamic_model_matrix_form()

    assert compatibility_model == matrix_model


def test_cross_module_time_major_and_cartesian_shape_conventions():
    joint = JointTrajectory(
        t=[0.0, 1.0],
        q=[[0.0], [1.0]],
        qd=[[0.0], [0.0]],
        qdd=[[0.0], [0.0]],
        method="quintic",
    )
    position = PositionTrajectory(
        t=[0.0, 1.0],
        p=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
        v=np.zeros((2, 3)),
        a=np.zeros((2, 3)),
        method="quintic",
    )
    dynamics = DynamicsSolution(
        t=[0.0, 1.0],
        q=[[0.0], [1.0]],
        qd=[[0.0], [0.0]],
        qdd=[[0.0], [0.0]],
        success=True,
        message="ok",
        method="RK45",
    )
    workspace = Workspace(
        points=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
        configurations=[[0.0], [1.0]],
        joint_limits=((-0.1, 1.1),),
        seed=1,
    )

    assert joint.q.shape == (2, 1)
    assert position.p.shape == (2, 3)
    assert dynamics.q.shape == (2, 1)
    assert workspace.configurations.shape == (2, 1)
    assert workspace.points.shape == (2, 3)


def test_position_ik_04_style_workflow_remains_compatible():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (1.0, 0, 0, q2, "r"),
    )
    q_reference = [0.5, -0.7]
    target = np.asarray(
        robot.T[:3, 3].subs(dict(zip(robot.qs, q_reference))),
        dtype=float,
    ).reshape(3)

    solution = solve_position_ik(
        robot,
        target,
        q0=[0.4, -0.6],
        method="lm",
        tol=1e-9,
        max_iter=100,
    )

    assert solution.converged is True
    achieved = np.asarray(
        robot.T[:3, 3].subs(dict(zip(robot.qs, solution.q))),
        dtype=float,
    ).reshape(3)
    np.testing.assert_allclose(achieved, target, atol=1e-8)


def test_mapping_based_visualization_04_style_workflow_remains_compatible():
    robot = Robot(
        (1.0, 0, 0, q1, "r"),
        (1.0, 0, 0, q2, "r"),
    )
    viz = RobotVisualizer(robot)

    fig, ax = viz.plot({q1: 0.2, q2: -0.1}, backend="matplotlib")

    assert ax.figure is fig
    plt.close(fig)
