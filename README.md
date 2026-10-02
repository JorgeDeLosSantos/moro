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
* **Trajectory generation:** Generate linear, cubic, and quintic point-to-point trajectories in joint or Cartesian position space.
* **Dynamics:** Derive symbolic equations of motion, evaluate inverse/forward dynamics numerically, and integrate motion with `solve_ivp`.
* **Workspace sampling:** Approximate reachable Cartesian workspace from finite joint ranges with reproducible joint-space sampling.
* **Visualization:** Plot and animate robot configurations using Matplotlib or Three.js, and plot sampled workspaces with Matplotlib.

## Installation

Install the latest stable release from PyPI:

```bash
pip install moro
```

To install the current development version from GitHub:

```bash
pip install git+https://github.com/JorgeDeLosSantos/moro.git
```

Moro 0.5.x requires **Python 3.11 or newer**.

Its main runtime dependencies are SymPy, NumPy, Matplotlib, and SciPy.

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
viz.plot(values)
```

For interactive visualization in a notebook:

```python
viz.plot(values, backend="threejs")
```

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

Full-pose IK is available through:

```python
from moro.inverse_kinematics import solve_pose_ik

solution = solve_pose_ik(
    robot,
    target_transform,
    q0=initial_guess,
    parameters=parameters,
)
```

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

The result uses time-major arrays such as `traj.q.shape == (N, dof)` and can be passed directly to `RobotVisualizer.animate()`.

## Dynamics and Simulation

After assigning masses, centers of mass, inertia tensors, and gravity, the symbolic manipulator equation is obtained with:

```python
model = robot.dynamic_model()
```

In Moro 0.5.0, `dynamic_model()` returns the matrix-form equation

```text
M(q) qdd + C(q, qd) qd + G(q) = tau
```

The former per-joint Euler-Lagrange behavior is available as:

```python
equations = robot.euler_lagrange_equations()
```

Numerical dynamics and simulation live in `moro.dynamics`:

```python
from moro.dynamics import inverse_dynamics, forward_dynamics, simulate

forces = inverse_dynamics(robot, q, qd, qdd, parameters=parameters)
acceleration = forward_dynamics(robot, q, qd, forces, parameters=parameters)

solution = simulate(
    robot,
    (0.0, 2.0),
    q0,
    qd0=qd0,
    tau=controller,
    parameters=parameters,
)
```

## Workspace Sampling

```python
from moro.workspace import sample_workspace
from moro.visualization import plot_workspace

workspace = sample_workspace(robot, samples=5000, seed=42)
fig, ax = plot_workspace(workspace)
```

Workspace samples are uniform in joint coordinates, not Cartesian space. The result stores both `workspace.configurations` and their corresponding Cartesian `workspace.points`.

## Documentation

The complete documentation is available at:

https://jorgedelossantos.github.io/moro/

It includes Getting Started guides, User Guide, worked examples, API Reference, mathematical Theory notes, and contributor documentation.

## Roadmap

Want to know what may come next? See the [Moro Roadmap Wiki](https://github.com/JorgeDeLosSantos/moro/wiki/Roadmap).

## Bug Reports and Contributions

If you encounter a bug, have a question, or want to request a feature, please open an issue in the [GitHub Issue Tracker](https://github.com/JorgeDeLosSantos/moro/issues).

Contributions are welcome. See the contributor documentation included in the project documentation for the recommended development workflow.
