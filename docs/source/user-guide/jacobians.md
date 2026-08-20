# Jacobians

`moro` provides tools for computing geometric Jacobians for the end-effector, arbitrary points on the manipulator, and link centers of mass.

The geometric Jacobian relates joint velocities to the linear and angular velocity of a point attached to the robot.

This section focuses on how to compute and inspect Jacobians in `moro`. For the mathematical derivation and interpretation of differential kinematics, see **Theory → Differential Kinematics**.

## End-effector Jacobian

Once a `Robot` has been defined, the geometric Jacobian of the end-effector is available through the `J` property.

Consider a planar 2R manipulator:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The end-effector Jacobian is:

```python
J = robot.J
J
```

For a robot with (n) degrees of freedom, the geometric Jacobian has size:

[
6 \times n.
]

It can be written as:

[
J =
\begin{bmatrix}
J_v \
J_\omega
\end{bmatrix},
]

where:

* (J_v) relates joint velocities to the linear velocity of the point;
* (J_\omega) relates joint velocities to its angular velocity.

For `robot.J`, the point of interest is the origin of the final frame `{n}`, that is, the end-effector origin.

## Linear and angular components

The linear and angular parts of a geometric Jacobian can be obtained directly by slicing the resulting SymPy matrix:

```python
Jv = robot.J[:3, :]
Jw = robot.J[3:, :]
```

The upper part,

```python
Jv
```

contains the linear-velocity contribution, while:

```python
Jw
```

contains the angular-velocity contribution.

For a joint-velocity vector:

[
\dot q =
\begin{bmatrix}
\dot q_1 \
\dot q_2 \
\vdots \
\dot q_n
\end{bmatrix},
]

the corresponding end-effector velocities satisfy:

[
v = J_v \dot q,
]

and:

[
\omega = J_\omega \dot q.
]

Because the Jacobian is symbolic, these relationships can be inspected before assigning numerical values to the joint coordinates.

## Jacobian of an arbitrary point

The Jacobian does not need to be restricted to the origin of the end-effector frame.

Use:

```python
robot.J_point(point, i)
```

to compute the geometric Jacobian of a point attached to link `i`.

The coordinates in `point` must be expressed with respect to frame `{i}`.

For example:

```python
Jp = robot.J_point([0.2, 0, 0], 2)
```

computes the Jacobian of a point located (0.2) units along the (x_2)-axis from the origin of frame `{2}`.

The point can be specified using a list, tuple, or a compatible SymPy vector.

For the origin of frame `{i}`, use:

```python
Ji = robot.J_point([0, 0, 0], i)
```

The end-effector Jacobian itself is equivalent to:

```python
robot.J_point([0, 0, 0], robot.dof)
```

This makes `J_point()` the most general Jacobian interface for points rigidly attached to the robot links.

## Revolute and prismatic joint contributions

The columns of the geometric Jacobian depend on the corresponding joint type.

For a revolute joint (j), the contribution to a point located at (p) is:

[
J_{v_j}
=======

z_{j-1}
\times
\left(
p-r_{O_{j-1}}
\right),
]

and:

[
J_{\omega_j}
============

z_{j-1}.
]

For a prismatic joint:

[
J_{v_j}
=======

z_{j-1},
]

and:

[
J_{\omega_j}
============

0.

]

`moro` builds these contributions automatically from the joint types stored in the `Robot` model.

For a point attached to link `i`, joints located after that link do not affect its motion. Their corresponding Jacobian columns are therefore zero.

For example, in a three-joint robot, the Jacobian of a point attached to link 2 has no contribution from joint 3.

The mathematical derivation of these expressions is covered in **Theory → Differential Kinematics**.

## Center-of-mass Jacobians

`moro` also provides Jacobians for the centers of mass of individual links.

These are especially useful in dynamic modeling.

Before computing them, the center-of-mass position of each link must be defined:

```python
robot.cm_positions = [
    (...),
    (...),
]
```

Each center-of-mass position is expressed with respect to the corresponding link frame.

For example, for a planar 2R robot:

```python
from moro.abc import lc1, lc2

robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
```

### Full center-of-mass Jacobian

The geometric Jacobian of the center of mass of link `i` is:

```python
Jcm = robot.J_cm_i(i)
```

For example:

```python
Jcm1 = robot.J_cm_i(1)
Jcm2 = robot.J_cm_i(2)
```

Each result has size:

[
6 \times n.
]

### Linear center-of-mass Jacobian

The linear part is available directly through:

```python
Jv_cm = robot.Jv_cm_i(i)
```

For example:

```python
Jv1 = robot.Jv_cm_i(1)
```

This is equivalent to the first three rows of the full center-of-mass Jacobian.

### Angular center-of-mass Jacobian

The angular part is:

```python
Jw_cm = robot.Jw_cm_i(i)
```

For example:

```python
Jw1 = robot.Jw_cm_i(1)
```

This corresponds to the lower three rows of `robot.J_cm_i(i)`.

These quantities are used internally when constructing dynamic terms such as the manipulator inertia matrix.

## Evaluating a configuration

As with the forward-kinematics results, Jacobians returned by `moro` are symbolic SymPy matrices.

Consider the configuration:

[
l_1 = 1,\qquad
l_2 = 0.8,\qquad
q_1 = \frac{\pi}{4},\qquad
q_2 = -\frac{\pi}{6}.
]

Define:

```python
from sympy import pi

values = {
    l1: 1,
    l2: 0.8,
    q1: pi / 4,
    q2: -pi / 6,
}
```

The end-effector Jacobian can then be evaluated with:

```python
J_num = robot.J.subs(values)
```

and converted explicitly to floating-point values with:

```python
J_num.evalf()
```

The same approach applies to arbitrary-point Jacobians:

```python
Jp = robot.J_point([0.2, 0, 0], 2)

Jp_num = Jp.subs(values).evalf()
```

and to center-of-mass Jacobians:

```python
Jcm2_num = robot.J_cm_i(2).subs(values).evalf()
```

provided all symbolic center-of-mass parameters are also included in the substitution dictionary.

## Working with symbolic results

Because the Jacobians are SymPy matrices, they can be manipulated with standard symbolic operations.

For example:

```python
from sympy import simplify

J = simplify(robot.J)
```

The linear and angular parts can be extracted:

```python
Jv = J[:3, :]
Jw = J[3:, :]
```

Individual columns can also be inspected:

```python
J1 = J[:, 0]
J2 = J[:, 1]
```

This is useful when studying how individual joints contribute to the velocity of the end-effector.

You can also inspect the rank of a numerically evaluated Jacobian:

```python
J_num = robot.J.subs(values)

J_num.rank()
```

or compute symbolic expressions involving its entries.

For larger symbolic models, simplification may become computationally expensive, so it is often useful to simplify only the expressions that are relevant to the current analysis.

## A worked example

Consider again a planar two-link robot:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The end-effector Jacobian is:

```python
J = robot.J
```

Its linear component is:

```python
Jv = J[:3, :]
```

and its angular component is:

```python
Jw = J[3:, :]
```

Now consider a point located halfway along the second link, assuming `l2` represents its full length:

```python
J_mid = robot.J_point(
    [-l2 / 2, 0, 0],
    2,
)
```

This point has a different linear Jacobian from the end-effector because its position relative to the joints is different.

Its angular Jacobian, however, is determined by the same revolute joints that affect link 2.

We can compare the linear parts:

```python
Jv_end = robot.J[:3, :]
Jv_mid = J_mid[:3, :]
```

Now evaluate both at a specific configuration:

```python
from sympy import pi

values = {
    l1: 1,
    l2: 0.8,
    q1: pi / 4,
    q2: -pi / 6,
}
```

```python
Jv_end_num = Jv_end.subs(values).evalf()
Jv_mid_num = Jv_mid.subs(values).evalf()
```

This illustrates an important distinction:

```text
same robot
    +
same joint configuration
    ↓
different point on the robot
    ↓
different linear Jacobian
```

The `J_point()` interface makes it possible to compute these quantities without defining a separate robot model.

## Notes and conventions

When working with Jacobians in `moro`, keep the following points in mind:

* `robot.J` is the geometric Jacobian of the end-effector origin;
* geometric Jacobians have size (6\times n);
* the first three rows correspond to linear velocity;
* the last three rows correspond to angular velocity;
* `robot.J_point(point, i)` expects the point coordinates to be expressed in frame `{i}`;
* the resulting Jacobian describes the point motion with respect to the base frame;
* revolute and prismatic joints are handled automatically according to the robot model;
* joints located after the link containing the point contribute zero columns;
* center-of-mass Jacobians require `robot.cm_positions` to be defined;
* Jacobian results are symbolic SymPy matrices unless numerical values are substituted.

The Jacobians returned by `moro` are geometric Jacobians. Analytical Jacobians associated with specific minimal orientation parameterizations are outside the current scope of this interface.

## See also

* **Forward Kinematics** — compute the frame positions and axes used in geometric Jacobian construction.
* **Robot Modeling** — define revolute and prismatic joints and their parameters.
* **Inverse Kinematics** — use Jacobian-based numerical methods to solve position inverse kinematics problems.
* **Dynamics** — use center-of-mass Jacobians when deriving the manipulator dynamic model.
* **Theory → Differential Kinematics** — mathematical derivation and interpretation of the geometric Jacobian.
* **API Reference → Robot** — complete signatures and descriptions for `J`, `J_point()`, `J_cm_i()`, `Jv_cm_i()`, and `Jw_cm_i()`.
