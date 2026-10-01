# Workspace Sampling

This example introduces sampled workspace analysis with three complementary cases.

## 1. Planar 2R workspace

```python
import numpy as np

from moro import Robot
from moro.abc import q1, q2
from moro.workspace import sample_workspace
from moro.visualization import plot_workspace

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
)

robot.joint_limits = [
    (-np.pi, np.pi),
    (-np.pi / 2, np.pi / 2),
]

workspace = sample_workspace(
    robot,
    samples=5000,
    seed=1,
)
```

Inspect the sampled dataset:

```python
workspace.samples
workspace.dof
workspace.bounds
workspace.points.shape
workspace.configurations.shape
```

Because this robot is planar, automatic visualization selects the $xy$ plane:

```python
fig, ax = plot_workspace(workspace)
```

The plotted points approximate the reachable position workspace over the selected joint ranges.

## 2. Mixed revolute/prismatic workspace

The same API applies to mixed joint types:

```python
robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.0, 0, q2, 0, "p"),
)

robot.joint_limits = [
    (-np.pi, np.pi),
    (0.0, 0.5),
]

workspace = sample_workspace(
    robot,
    samples=3000,
    seed=2,
)

fig, ax = plot_workspace(
    workspace,
    projection="3d",
)
```

The revolute and prismatic coordinates are sampled uniformly over their respective numerical intervals.

## 3. Effect of joint limits

Workspace depends on both robot geometry and the joint-space domain.

```python
wide = sample_workspace(
    robot,
    samples=3000,
    joint_limits=[
        (-np.pi, np.pi),
        (0.0, 0.5),
    ],
    seed=3,
)

restricted = sample_workspace(
    robot,
    samples=3000,
    joint_limits=[
        (-np.pi / 4, np.pi / 4),
        (0.1, 0.3),
    ],
    seed=3,
)
```

The explicit limits apply only to the sampling call. They do not modify `robot.joint_limits`.

Comparing the two point clouds illustrates that

$$
\mathcal W
=
\{p(q)\mid q\in\mathcal Q\}
$$

depends directly on the chosen domain $\mathcal Q$.

## Symbolic geometry

Symbolic link parameters can be resolved numerically at sampling time:

```python
workspace = sample_workspace(
    robot_symbolic,
    samples=2000,
    parameters={
        l1: 1.0,
        l2: 0.8,
    },
    seed=4,
)
```

The symbolic robot model remains unchanged.

## Interpretation

The point cloud is sampled uniformly in joint coordinates, not Cartesian space. It should therefore be interpreted as sampled reachability rather than an exact boundary, area, volume, or probability density in Cartesian space.
