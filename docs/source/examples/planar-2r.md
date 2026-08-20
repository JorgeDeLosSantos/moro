# Planar 2R Manipulator

This example introduces a complete kinematic workflow with `moro` using a simple planar two-link manipulator.

We will:

1. define the robot from its Denavit-Hartenberg parameters;
2. compute its forward kinematics;
3. extract the end-effector position;
4. compute the geometric Jacobian;
5. evaluate the symbolic expressions at a numerical configuration;
6. visualize the resulting robot pose.

The goal is not to derive the equations manually, but to show how the main kinematic tools in `moro` fit together in a typical analysis workflow.

## Problem

Consider a planar manipulator composed of two revolute joints and two rigid links.

Let

$$
q_1,\; q_2
$$

be the joint variables, and

$$
l_1,\; l_2
$$

the link lengths.

The robot moves entirely in the \(xy\)-plane, while all joint rotation axes are parallel to the \(z\)-axis.

Using the classical Denavit-Hartenberg convention, the robot can be described by:

| Link | \(a_i\) | \(\alpha_i\) | \(d_i\) | \(\theta_i\) | Joint |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | \(l_1\) | \(0\) | \(0\) | \(q_1\) | revolute |
| 2 | \(l_2\) | \(0\) | \(0\) | \(q_2\) | revolute |

We want to obtain the symbolic kinematics of the manipulator and then evaluate them for a specific configuration.

## Robot model

First import the required objects:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2
```

The robot is created by passing one DH row per joint:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The resulting model can be inspected directly:

```python
robot
```

which returns:

```text
Robot RR
```

The number of degrees of freedom is:

```python
robot.dof
```

which returns:

```text
2
```

The joint variables used by the model are available through:

```python
robot.qs
```

The DH table can also be inspected with:

```python
robot.dh_table
```

At this point, the model is fully symbolic. No numerical values have been assigned to the link lengths or joint variables.

## Forward kinematics

The complete homogeneous transformation of the end-effector with respect to the base frame is available through:

```python
T = robot.T
T
```

For this planar 2R manipulator, the transformation has the structure

$$
T_2^0 =
\begin{bmatrix}
\cos(q_1+q_2) &
-\sin(q_1+q_2) &
0 &
l_1\cos q_1+l_2\cos(q_1+q_2)
\\
\sin(q_1+q_2) &
\cos(q_1+q_2) &
0 &
l_1\sin q_1+l_2\sin(q_1+q_2)
\\
0 & 0 & 1 & 0
\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

The transformation of each intermediate frame can also be obtained individually:

```python
T10 = robot.T_i0(1)
T20 = robot.T_i0(2)
```

Since frame `{2}` is the end-effector frame,

```python
T20 == robot.T
```

evaluates to `True`.

Accessing intermediate transformations is useful when the position or orientation of a specific link frame is also required.

## End-effector position

The Cartesian position of the end-effector corresponds to the translation part of the homogeneous transformation:

```python
p = robot.T[:3, 3]
p
```

For this robot,

$$
p(q)=
\begin{bmatrix}
l_1\cos q_1+l_2\cos(q_1+q_2)
\\
l_1\sin q_1+l_2\sin(q_1+q_2)
\\
0
\end{bmatrix}.
$$

The zero third component reflects the fact that the manipulator is planar.

Because this expression remains symbolic, it can be evaluated repeatedly for different geometric parameters and joint configurations without rebuilding the robot model.

## Jacobian matrix

The geometric Jacobian of the end-effector is available through:

```python
J = robot.J
J
```

For a two-degree-of-freedom robot, the result is a \(6\times2\) matrix.

The first three rows correspond to the linear velocity Jacobian:

```python
Jv = J[:3, :]
Jv
```

and the last three rows correspond to the angular velocity Jacobian:

```python
Jw = J[3:, :]
Jw
```

For the planar 2R manipulator, the linear part is

$$
J_v =
\begin{bmatrix}
-l_1\sin q_1-l_2\sin(q_1+q_2)
&
-l_2\sin(q_1+q_2)
\\
l_1\cos q_1+l_2\cos(q_1+q_2)
&
l_2\cos(q_1+q_2)
\\
0 & 0
\end{bmatrix},
$$

while the angular part is

$$
J_\omega =
\begin{bmatrix}
0 & 0
\\
0 & 0
\\
1 & 1
\end{bmatrix}.
$$

The Jacobian relates joint velocities to the end-effector linear and angular velocities:

$$
\begin{bmatrix}
v\\
\omega
\end{bmatrix}
=
J(q)\dot q.
$$

For this mechanism, all angular motion occurs about the \(z\)-axis.

## Numerical evaluation

Now consider the numerical values

$$
l_1=1.0,
\qquad
l_2=0.8,
$$

and the joint configuration

$$
q_1=\frac{\pi}{4},
\qquad
q_2=-\frac{\pi}{6}.
$$

Create a substitution dictionary:

```python
from sympy import pi

values = {
    l1: 1.0,
    l2: 0.8,
    q1: pi / 4,
    q2: -pi / 6,
}
```

The end-effector transformation can now be evaluated numerically:

```python
T_num = robot.T.subs(values).evalf()
T_num
```

The corresponding Cartesian position is:

```python
p_num = p.subs(values).evalf()
p_num
```

Likewise, the Jacobian at the same configuration is:

```python
J_num = robot.J.subs(values).evalf()
J_num
```

The same symbolic expressions can therefore be reused for any other configuration simply by changing the substitution dictionary.

For example:

```python
another_configuration = {
    l1: 1.0,
    l2: 0.8,
    q1: 0,
    q2: pi / 2,
}

robot.T.subs(another_configuration).evalf()
```

No new `Robot` instance is required.

## Visualization

The same numerical values used in the kinematic evaluation can also be passed directly to the visualization tools.

Create a `RobotVisualizer`:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
```

### Matplotlib

A static Matplotlib visualization can be created with:

```python
fig, ax = viz.plot(
    values,
    backend="matplotlib",
)
```

If running in a standard Python script, the figure can be displayed with:

```python
import matplotlib.pyplot as plt

plt.show()
```

The generated scene includes the robot links, joints, and coordinate frames.

### Three.js

For an interactive visualization in a notebook:

```python
viz.plot(
    values,
    backend="threejs",
)
```

The Three.js viewer allows the robot to be inspected interactively using orbit controls and includes preset front, top, and isometric views.

Since the robot model is symbolic, the same visualizer can be reused with different configurations:

```python
viz.plot(
    {
        l1: 1.0,
        l2: 0.8,
        q1: pi / 2,
        q2: -pi / 3,
    },
    backend="threejs",
)
```

## Discussion

This simple example illustrates the central workflow used throughout `moro`.

A single symbolic `Robot` model provides access to both forward and differential kinematics:

```python
robot.T
robot.J
```

The same model can then be evaluated numerically through standard SymPy substitutions:

```python
robot.T.subs(values)
robot.J.subs(values)
```

and the same substitution dictionary can be reused by the visualization system:

```python
viz.plot(values)
```

This separation between symbolic modeling and numerical evaluation is particularly useful in robotics education and analysis. The robot equations can be derived once and then explored at many different parameter values and configurations.

The planar 2R manipulator is intentionally simple, but the workflow remains essentially the same for larger serial manipulators:

```text
define the robot
        ↓
compute symbolic kinematics
        ↓
compute differential kinematics
        ↓
evaluate a configuration
        ↓
visualize the result
```

More advanced examples will extend this workflow to spatial manipulators, numerical inverse kinematics, Cartesian trajectories, and dynamic modeling.

## See also

- **User Guide → Robot Modeling** — creating serial robot models with DH parameters.
- **User Guide → Forward Kinematics** — frame transformations and Cartesian poses.
- **User Guide → Jacobians** — geometric Jacobians and point Jacobians.
- **User Guide → Visualization** — static plots and robot animations.
- **Theory → Forward Kinematics** — mathematical background.
- **Theory → Differential Kinematics** — Jacobian theory and velocity relationships.