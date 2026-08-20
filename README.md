# moro

[![PyPI version](https://img.shields.io/pypi/v/moro.svg)](https://pypi.org/project/moro/)
[![License](https://img.shields.io/github/license/JorgeDeLosSantos/moro.svg)](https://github.com/JorgeDeLosSantos/moro/blob/master/LICENSE.txt)

`moro` is a Python library for symbolic modeling, analysis, and visualization of serial robot manipulators.

It is designed primarily for robotics education and for workflows where inspecting the underlying kinematic and dynamic expressions is as important as evaluating them numerically.

## Features

* **Robot modeling:** Define serial manipulators with revolute and prismatic joints using Denavit-Hartenberg parameters.
* **Transformations:** Work with `SO(3)` rotation matrices, `SE(3)` homogeneous transformations, Euler angles, and axis-angle representations.
* **Forward kinematics:** Compute symbolic end-effector and intermediate-frame transformations.
* **Differential kinematics:** Compute geometric Jacobians for the end-effector and other points.
* **Inverse kinematics:** Solve numerical Cartesian position IK using Levenberg-Marquardt, Newton-Raphson, or CCD.
* **IK trajectories:** Solve ordered sequences of Cartesian position targets using warm-started inverse kinematics.
* **Dynamics:** Derive symbolic equations of motion and the standard `M(q) qdd + C(q, qd) qd + G(q) = tau` model.
* **Visualization:** Plot and animate robot configurations using Matplotlib or an interactive Three.js backend.

## Installation

Install the latest stable release from PyPI:

```bash
pip install moro
```

To install the current development version from the `develop` branch:

```bash
pip install git+https://github.com/JorgeDeLosSantos/moro.git@develop
```

`moro` requires Python 3.9 or newer.

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

The same symbolic model can also be visualized:

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

A Cartesian position target can be solved numerically with:

```python
from moro.inverse_kinematics import solve_position_ik

solution = solve_position_ik(
    robot,
    [1.5, 0.5, 0.0],
    q0=[0.1, 0.1],
    parameters={
        l1: 1.0,
        l2: 1.0,
    },
)

if solution.converged:
    print(solution.q)
else:
    print(solution.message)
```

Current inverse-kinematics support is focused on Cartesian position. Full-pose IK with orientation constraints is not yet included.

## Dynamics

Dynamic models can be built by assigning masses, centers of mass, inertia tensors, and gravity to an existing `Robot` model.

For example:

```python
import sympy as sp

from moro.abc import m1, m2, lc1, lc2, g

I1, I2 = sp.symbols("I1 I2", positive=True)

robot.masses = [m1, m2]
robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
robot.inertia_tensors = [
    sp.diag(0, 0, I1),
    sp.diag(0, 0, I2),
]
robot.gravity = (0, -g, 0)

M = robot.inertia_matrix()
C = robot.coriolis_matrix()
G = robot.gravity_vector()

model = robot.dynamic_model_matrix_form()
```

The current dynamics API derives symbolic equations of motion and supports inverse-dynamics-style evaluation. Forward dynamics integration is not currently included.

## Documentation

The complete documentation is available at:

https://jorgedelossantos.github.io/moro/

It includes:

* Getting Started guides;
* a practical User Guide;
* complete worked examples;
* API Reference;
* mathematical Theory notes;
* contributor documentation and naming conventions.

## Roadmap

Want to know what may come next? See the [Moro Roadmap Wiki](https://github.com/JorgeDeLosSantos/moro/wiki/Roadmap).

## Bug Reports and Contributions

If you encounter a bug, have a question, or want to request a feature, please open an issue in the [GitHub Issue Tracker](https://github.com/JorgeDeLosSantos/moro/issues).

Contributions are welcome. See the contributor documentation included in the project documentation for the recommended development workflow.
