# Quick Start

This guide introduces the basic workflow for modeling and analyzing a serial robot with `moro`.

We will use a simple planar two-link manipulator with two revolute joints and perform a few common operations:

* create the robot model;
* inspect its basic properties;
* compute the forward kinematics;
* compute the geometric Jacobian;
* evaluate symbolic expressions at a particular configuration;
* visualize the robot.

The goal is to provide a first overview of the library rather than a detailed explanation of each concept. More complete discussions are available in the **User Guide** and **Theory** sections.

## Importing moro

We will use the `Robot` class together with a few predefined symbolic variables provided by `moro.abc`:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2
from moro.visualization import RobotVisualizer
```

The variables `q1` and `q2` represent the joint variables, while `l1` and `l2` represent the lengths of the two links.

`moro.abc` provides several commonly used symbolic variables as a convenience. You can also define your own symbols directly with SymPy, as shown later in the User Guide.

## Creating a robot

A serial manipulator is created by passing one Denavit-Hartenberg row for each joint to `Robot`.

Each row follows the convention:

```text
(a_i, alpha_i, d_i, theta_i, joint_type)
```

where `joint_type` can be:

* `"r"` for a revolute joint;
* `"p"` for a prismatic joint.

For a planar 2R manipulator with link lengths $l_1$ and $l_2$, we can write:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The resulting object contains the geometric description of the robot and can be used to derive its kinematic quantities.

## Inspecting the robot model

Some basic information about the robot can be inspected directly from the model.

The number of degrees of freedom is available through:

```python
robot.dof
```

which returns:

```text
2
```

The joint variables can be accessed with:

```python
robot.qs
```

For this robot, the result corresponds to:

```text
[q1, q2]
```

You can also inspect the Denavit-Hartenberg parameters:

```python
robot.dh_table
```

This is useful for verifying that the robot model has been defined as intended before performing further calculations.

## Forward kinematics

The homogeneous transformation matrix from the base frame to the end-effector is available through the `T` property:

```python
T = robot.T
T
```

For the planar 2R manipulator, `T` is a symbolic $4\times4$ homogeneous transformation matrix whose elements depend on $q_1$, $q_2$, $l_1$, and $l_2$.

Because the result is symbolic, you can inspect it directly, simplify it, or substitute numerical values later.

Intermediate transformations are also available when needed. For example, the transformation from the base frame to frame `{1}` can be obtained with:

```python
robot.T_i0(1)
```

For a more detailed discussion of transformations between robot frames, see the **Forward Kinematics** section of the User Guide.

## Computing the Jacobian

The geometric Jacobian of the end-effector is available through the `J` property:

```python
J = robot.J
J
```

The result is again a symbolic matrix.

For a manipulator with two joints, the Jacobian has two columns, one associated with each joint variable. The upper part describes the contribution of the joints to the linear velocity of the end-effector, while the lower part describes their contribution to its angular velocity.

The Jacobian can therefore be inspected symbolically before assigning a particular robot configuration.

## Evaluating a robot configuration

Once the symbolic model has been obtained, numerical values can be substituted using SymPy.

Consider the configuration:

$$
l_1 = 1,\qquad
l_2 = 1,\qquad
q_1 = 0.5,\qquad
q_2 = 0.8.
$$

We can store these values in a dictionary:

```python
values = {
    l1: 1,
    l2: 1,
    q1: 0.5,
    q2: 0.8,
}
```

The forward kinematics can then be evaluated with:

```python
T_num = T.subs(values)
T_num
```

Similarly, the Jacobian can be evaluated at the same configuration:

```python
J_num = J.subs(values)
J_num
```

If floating-point values are preferred explicitly, SymPy's `evalf()` method can be applied:

```python
T_num.evalf()
```

and:

```python
J_num.evalf()
```

This symbolic-first workflow makes it possible to derive the robot equations once and then evaluate them at as many configurations as needed.

## Visualizing the robot

The same robot model can be visualized using `RobotVisualizer`.

First create a visualizer associated with the robot:

```python
viz = RobotVisualizer(robot)
```

A particular configuration can then be plotted using the same dictionary of numerical values:

```python
viz.plot(values)
```

By default, `RobotVisualizer` uses the Matplotlib backend.

The Three.js backend can also be selected explicitly:

```python
viz.plot(values, backend="threejs")
```

The visualization module can also animate sequences of robot configurations. These capabilities are covered in more detail in the **Visualization** section of the User Guide.

## Where to go next

You now have the basic workflow required to start working with `moro`:

```text
define the robot
        ↓
inspect the model
        ↓
compute symbolic kinematics
        ↓
compute the Jacobian
        ↓
substitute numerical values
        ↓
visualize the configuration
```

From here, the **User Guide** develops each topic in more detail:

* **Robot Modeling** — define revolute and prismatic serial manipulators and inspect their parameters.
* **Transformations** — work with rotation matrices and homogeneous transformations.
* **Forward Kinematics** — compute and inspect transformations throughout the kinematic chain.
* **Jacobians** — compute geometric Jacobians for the end-effector and other points.
* **Inverse Kinematics** — solve position inverse kinematics problems numerically.
* **Dynamics** — derive inverse dynamic models using the available formulations.
* **Visualization** — plot and animate robot configurations with the available backends.

For complete worked problems, see the **Examples** section.

For detailed information about individual classes, functions, properties, and methods, see the **API Reference**.

For the mathematical foundations behind these operations, see the **Theory** section.
