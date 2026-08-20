# Anthropomorphic RRR Manipulator

This example extends the basic kinematic workflow to a spatial serial manipulator.

We will use a three-degree-of-freedom anthropomorphic RRR robot and compute its forward kinematics, inspect intermediate frames, obtain the end-effector pose, compute the geometric Jacobian, evaluate the model numerically, and visualize the resulting configuration.

The objective is to show that the same workflow used for a planar robot can be applied directly to a spatial manipulator.

## Problem

Consider a serial manipulator with three revolute joints.

The first joint rotates about the base vertical axis, while the second and third joints form an articulated arm.

Let

\[
q_1,\; q_2,\; q_3
\]

be the joint variables, and let

\[
d_1,\; l_2,\; l_3
\]

represent the geometric dimensions of the robot.

Using the classical Denavit-Hartenberg convention, the robot is described by:

| Link | \(a_i\) | \(\alpha_i\) | \(d_i\) | \(\theta_i\) | Joint |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | \(0\) | \(\pi/2\) | \(d_1\) | \(q_1\) | revolute |
| 2 | \(l_2\) | \(0\) | \(0\) | \(q_2\) | revolute |
| 3 | \(l_3\) | \(0\) | \(0\) | \(q_3\) | revolute |

Unlike the planar 2R manipulator, this mechanism moves in three-dimensional space.

We want to obtain its symbolic kinematic model and then inspect one numerical configuration.

## Robot model

First import the required objects:

```python
from sympy import pi

from moro import Robot
from moro.abc import q1, q2, q3, d1, l2, l3
```

Create the robot:

```python
robot = Robot(
    (0, pi / 2, d1, q1, "r"),
    (l2, 0, 0, q2, "r"),
    (l3, 0, 0, q3, "r"),
)
```

Inspect the resulting object:

```python
robot
```

which returns:

```text
Robot RRR
```

The number of degrees of freedom is:

```python
robot.dof
```

which returns:

```text
3
```

The joint variables are:

```python
robot.qs
```

and the DH table can be inspected with:

```python
robot.dh_table
```

At this point, the robot model is fully symbolic.

## Forward kinematics

The homogeneous transformation of the end-effector with respect to the base frame is available through:

```python
T = robot.T
T
```

For this robot, the transformation depends on all three joint variables and contains both position and orientation information.

The end-effector position has the structure

\[
p =
\begin{bmatrix}
\left(l_2\cos q_2+l_3\cos(q_2+q_3)\right)\cos q_1
\\
\left(l_2\cos q_2+l_3\cos(q_2+q_3)\right)\sin q_1
\\
d_1+l_2\sin q_2+l_3\sin(q_2+q_3)
\end{bmatrix}.
\]

This expression already shows the characteristic structure of an anthropomorphic manipulator:

- \(q_1\) rotates the arm around the base axis;
- \(q_2\) and \(q_3\) determine the radial and vertical position of the arm.

The complete transformation also contains the orientation of the final frame.

## Intermediate frames

For spatial manipulators, inspecting intermediate frames is often useful.

The transformations of the three link frames with respect to the base are:

```python
T10 = robot.T_i0(1)
T20 = robot.T_i0(2)
T30 = robot.T_i0(3)
```

The last transformation is equivalent to the complete end-effector transformation:

```python
T30 == robot.T
```

which evaluates to:

```text
True
```

The rotation matrices can also be accessed directly:

```python
R10 = robot.R_i0(1)
R20 = robot.R_i0(2)
R30 = robot.R_i0(3)
```

Similarly, the origin of each frame in base coordinates can be obtained with:

```python
r1 = robot.r_o(1)
r2 = robot.r_o(2)
r3 = robot.r_o(3)
```

These intermediate quantities are particularly useful when studying the geometry of spatial robots or when building Jacobians.

## End-effector pose

A homogeneous transformation contains both orientation and position.

The orientation of the end-effector is:

```python
R = robot.T[:3, :3]
R
```

while its Cartesian position is:

```python
p = robot.T[:3, 3]
p
```

These quantities can also be obtained from the corresponding frame transformation:

```python
R = robot.R_i0(3)
p = robot.r_o(3)
```

The position depends on the three joint coordinates:

\[
p(q_1,q_2,q_3)=
\begin{bmatrix}
x
\\
y
\\
z
\end{bmatrix}.
\]

For this particular robot,

\[
x=
\left(l_2\cos q_2+l_3\cos(q_2+q_3)\right)\cos q_1,
\]

\[
y=
\left(l_2\cos q_2+l_3\cos(q_2+q_3)\right)\sin q_1,
\]

and

\[
z=
d_1+l_2\sin q_2+l_3\sin(q_2+q_3).
\]

Unlike the planar 2R example, the end-effector position can now vary in all three Cartesian coordinates.

## Jacobian matrix

The geometric Jacobian is available through:

```python
J = robot.J
J
```

For a three-degree-of-freedom manipulator, the result is a \(6\times3\) matrix.

The first three rows represent the linear velocity Jacobian:

```python
Jv = J[:3, :]
Jv
```

and the final three rows represent the angular velocity Jacobian:

```python
Jw = J[3:, :]
Jw
```

The complete relationship is:

\[
\begin{bmatrix}
v
\\
\omega
\end{bmatrix}
=
J(q)\dot q.
\]

In this case, the Jacobian describes motion in three-dimensional space, so both its linear and angular parts contain information that is not present in a purely planar mechanism.

The Jacobian columns correspond to the contribution of each revolute joint to the motion of the end-effector.

## Numerical evaluation

Now assign numerical values to the robot geometry:

\[
d_1=1.0,
\qquad
l_2=1.2,
\qquad
l_3=0.9.
\]

Consider the joint configuration:

\[
q_1=\frac{\pi}{6},
\qquad
q_2=-\frac{\pi}{9},
\qquad
q_3=\frac{7\pi}{36}.
\]

These values correspond to approximately:

\[
q_1=30^\circ,
\qquad
q_2=-20^\circ,
\qquad
q_3=35^\circ.
\]

Create the substitution dictionary:

```python
values = {
    d1: 1.0,
    l2: 1.2,
    l3: 0.9,
    q1: pi / 6,
    q2: -pi / 9,
    q3: 7 * pi / 36,
}
```

The end-effector transformation can now be evaluated numerically:

```python
T_num = robot.T.subs(values).evalf()
T_num
```

The Cartesian position is:

```python
p_num = p.subs(values).evalf()
p_num
```

The orientation matrix is:

```python
R_num = R.subs(values).evalf()
R_num
```

and the Jacobian at the same configuration is:

```python
J_num = robot.J.subs(values).evalf()
J_num
```

The same symbolic model can be reused for any other configuration by modifying only the substitution dictionary.

## Inspecting intermediate frames numerically

The intermediate transformations can also be evaluated using the same numerical values:

```python
T10_num = robot.T_i0(1).subs(values).evalf()
T20_num = robot.T_i0(2).subs(values).evalf()
T30_num = robot.T_i0(3).subs(values).evalf()
```

Similarly, the frame origins are:

```python
r1_num = robot.r_o(1).subs(values).evalf()
r2_num = robot.r_o(2).subs(values).evalf()
r3_num = robot.r_o(3).subs(values).evalf()
```

This makes it possible to inspect the complete spatial geometry of the manipulator, not only the end-effector pose.

## Visualization

A spatial robot is especially useful for illustrating the interactive visualization tools.

Create a visualizer:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
```

### Matplotlib

The configuration can be rendered with Matplotlib:

```python
fig, ax = viz.plot(
    values,
    backend="matplotlib",
)
```

A different initial camera orientation can be selected with:

```python
fig, ax = viz.plot(
    values,
    backend="matplotlib",
    view_init=(30, 45),
)
```

When running in a standard Python script:

```python
import matplotlib.pyplot as plt

plt.show()
```

The resulting 3D plot shows the spatial arrangement of the links and coordinate frames.

### Three.js

For an interactive representation:

```python
viz.plot(
    values,
    backend="threejs",
)
```

The Three.js viewer is especially useful for this robot because the configuration is genuinely three-dimensional.

The robot can be inspected using:

- free orbit navigation;
- Front view;
- Top view;
- Isometric view;
- orthographic projection;
- perspective projection.

The coordinate frames help reveal how the orientation changes from one link to the next.

## Reusing the model

The symbolic robot can be evaluated at another configuration without creating a new `Robot` instance:

```python
another_configuration = {
    d1: 1.0,
    l2: 1.2,
    l3: 0.9,
    q1: pi / 3,
    q2: pi / 6,
    q3: -pi / 4,
}
```

The new end-effector pose is obtained with:

```python
robot.T.subs(another_configuration).evalf()
```

and the same configuration can be visualized immediately:

```python
viz.plot(
    another_configuration,
    backend="threejs",
)
```

This reuse of the symbolic model is one of the main advantages of the workflow.

## Discussion

The anthropomorphic RRR manipulator extends the workflow introduced with the planar 2R robot to three-dimensional kinematics.

The same `Robot` object provides access to:

```python
robot.T
robot.T_i0(i)
robot.R_i0(i)
robot.r_o(i)
robot.J
```

These quantities describe different aspects of the same symbolic kinematic model.

The complete workflow remains:

```text
define the DH model
        ↓
compute symbolic frame transformations
        ↓
inspect end-effector pose
        ↓
compute the geometric Jacobian
        ↓
substitute numerical values
        ↓
visualize the configuration
```

The main difference with the planar example is geometric rather than procedural. Once the robot is defined, the same API can be applied to both planar and spatial serial manipulators.

This RRR robot is also a useful model for more advanced analyses.

For example, the end-effector position

\[
p(q_1,q_2,q_3)
\]

can be used as the basis for numerical inverse kinematics, and the geometric Jacobian can be used by iterative inverse-kinematics algorithms.

The next examples build directly on this model to solve Cartesian position targets numerically.

## See also

- **Planar 2R Manipulator** — introductory symbolic kinematic workflow.
- **User Guide → Robot Modeling** — defining revolute and prismatic serial manipulators.
- **User Guide → Forward Kinematics** — frame transformations and end-effector poses.
- **User Guide → Jacobians** — geometric Jacobians for serial manipulators.
- **User Guide → Visualization** — Matplotlib and Three.js rendering.
- **User Guide → Inverse Kinematics** — numerical Cartesian position IK.
- **Theory → Forward Kinematics** — mathematical background.
- **Theory → Differential Kinematics** — Jacobian theory.