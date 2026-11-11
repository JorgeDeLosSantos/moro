# moro

[![PyPI version](https://img.shields.io/pypi/v/moro.svg)](https://pypi.org/project/moro/)
[![License](https://img.shields.io/github/license/JorgeDeLosSantos/moro.svg)](https://github.com/JorgeDeLosSantos/moro/blob/master/LICENSE.txt)

`moro` is a Python library for symbolic modeling, analysis, numerical simulation, and visualization of serial robot manipulators.

It is designed primarily for robotics education and for workflows where inspecting the underlying kinematic and dynamic expressions is as important as evaluating them numerically.

## Features

* **Robot modeling:** Define serial manipulators with revolute and prismatic joints using Denavit-Hartenberg parameters.
* **Transformations:** Work with `SO(3)` rotations, `SE(3)` homogeneous transformations, Euler/Tait-Bryan angles, axis-angle, quaternions, and rotation vectors.
* **Forward kinematics:** Compute symbolic end-effector and intermediate-frame transformations.
* **Differential kinematics:** Evaluate task Jacobians, Cartesian velocities, velocity IK, singular values, numerical rank, condition number, and manipulability.
* **Inverse kinematics:** Solve numerical Cartesian position IK and full-pose IK with Jacobian-based methods; position-only CCD remains available.
* **Trajectory generation:** Generate point-to-point and multi-waypoint linear, cubic, and quintic joint/Cartesian-position trajectories, plus synchronized trapezoidal joint profiles.
* **Dynamics:** Derive symbolic equations of motion, evaluate inverse/forward dynamics and sampled trajectory efforts numerically, and integrate motion with `solve_ivp`.
* **Workspace sampling:** Explore approximate reachable Cartesian workspaces with seeded random, Sobol, Latin-hypercube, or grid joint-space samplers and Cartesian plane sections.
* **Visualization:** Plot and animate robot configurations with Three.js by default; visualize workspaces and sections in Three.js with optional grids, or select Matplotlib explicitly.

## Installation

Install the latest stable release from PyPI:

```bash
pip install moro
```

To install the current code directly from GitHub:

```bash
pip install git+https://github.com/JorgeDeLosSantos/moro.git@master
```

The `develop` branch contains ongoing work and may include unreleased API changes.

Moro requires **Python 3.11 or newer**.

Its main runtime dependencies are SymPy, NumPy, Matplotlib, SciPy, and IPython (for Three.js HTML output).

## Quick Start

The following example creates a symbolic planar 2R manipulator and evaluates its forward kinematics and Jacobian at one configuration:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)

T = robot.T
J = robot.J

values = {
    l1: 1.0,
    l2: 1.0,
    q1: 0.5,
    q2: 0.8,
}

T_num = T.subs(values).evalf()
J_num = J.subs(values).evalf()
```

The same symbolic model can be visualized:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
viz.plot(values)  # interactive Three.js viewer (Jupyter / Colab)
```

Three.js is the default for plotting and animating configurations.
When you need a Matplotlib figure or animation, select it explicitly:

```python
fig, ax = viz.plot(values, backend="matplotlib")
```

With the default `VisualizationStyle()`, Three.js displays revolute
cylinders and prismatic boxes when the robot's joint metadata is available;
Matplotlib retains simple spherical markers. The interactive HTML viewer
loads Three.js resources in the browser, so it needs access to its CDN.

## Inverse Kinematics

Position IK:

```python
from moro.inverse_kinematics import solve_position_ik

solution = solve_position_ik(
    robot,
    [1.5, 0.5, 0.0],
    q0=[0.1, 0.1],
    parameters={l1: 1.0, l2: 1.0},
)
```

For full-pose IK, `solve_pose_ik()` accepts a numerical homogeneous
transform as its target, with an initial joint estimate. See the
[full-pose IK example](https://jorgedelossantos.github.io/moro/examples/full-pose-inverse-kinematics.html)
for a complete worked problem.

## Trajectories

Generate a smooth joint-space trajectory with:

```python
import numpy as np
from moro.trajectory import joint_trajectory

traj = joint_trajectory(
    [0.0, 0.0],
    [0.8, -0.4],
    np.linspace(0.0, 2.0, 101),
    method="quintic",
)
```

The result uses time-major arrays such as `traj.q.shape == (N, dof)` and can be animated with physical timestamps using `viz.animate(traj.q, times=traj.t)` in Three.js. For multi-waypoint motion and synchronized trapezoidal profiles, use `joint_waypoint_trajectory()` and `joint_trapezoidal_trajectory()`, respectively.

## Dynamics and Simulation

After assigning masses, centers of mass, inertia tensors, and gravity, the symbolic manipulator equation is obtained with:

```python
model = robot.dynamic_model()
```

`dynamic_model()` returns the matrix-form equation

```text
M(q) qdd + C(q, qd) qd + G(q) = tau
```

The former per-joint Euler-Lagrange behavior is available as:

```python
equations = robot.euler_lagrange_equations()
```

Numerical dynamics and simulation live in `moro.dynamics`.
After defining physical properties (masses, centers of mass, inertia tensors,
and gravity), evaluate states and trajectories with:

```python
from moro.dynamics import (
    inverse_dynamics,
    forward_dynamics,
    inverse_dynamics_trajectory,
    simulate,
)

q = [0.2, -0.1]
qd = [0.0, 0.0]
qdd = [0.5, -0.2]
parameters = {l1: 1.0, l2: 1.0}

forces = inverse_dynamics(robot, q, qd, qdd, parameters=parameters)
acceleration = forward_dynamics(robot, q, qd, forces, parameters=parameters)
effort_history = inverse_dynamics_trajectory(robot, traj, parameters=parameters)

solution = simulate(robot, (0.0, 2.0), q, qd0=qd, tau=forces, parameters=parameters)
```

For a fully specified physical model, see the
[numerical dynamics example](https://jorgedelossantos.github.io/moro/examples/numerical-dynamics.html).

## Workspace Sampling

```python
from moro.workspace import sample_workspace, section_workspace
from moro.visualization import plot_workspace, plot_workspace_section

workspace = sample_workspace(
    robot, samples=1024, method="sobol", seed=42,
    joint_limits=[(-3.14, 3.14), (-3.14, 3.14)],
    parameters={l1: 1.0, l2: 1.0},
)
viewer = plot_workspace(workspace)  # Three.js
# Matplotlib instead: fig, ax = plot_workspace(workspace, backend="matplotlib")

section = section_workspace(workspace, plane="xy", level=0.0, thickness=0.05)
section_viewer = plot_workspace_section(section, show_grid=True)
```

Workspace sampling operates in joint coordinates, not uniformly in Cartesian space. The result stores both `workspace.configurations` and their corresponding Cartesian `workspace.points`. Sections select points near a plane; they do not reconstruct exact workspace boundaries.

## Documentation

The complete documentation is available at:

https://jorgedelossantos.github.io/moro/

It includes Getting Started guides, User Guide, worked examples, API Reference, mathematical Theory notes, and contributor documentation.

## Roadmap

Want to know what may come next? See the [Moro Roadmap Wiki](https://github.com/JorgeDeLosSantos/moro/wiki/Roadmap).

## Bug Reports and Contributions

If you encounter a bug, have a question, or want to request a feature, please open an issue in the [GitHub Issue Tracker](https://github.com/JorgeDeLosSantos/moro/issues).

Contributions are welcome. See the contributor documentation included in the project documentation for the recommended development workflow.
