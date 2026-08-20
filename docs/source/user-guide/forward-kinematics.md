# Forward Kinematics

Once a `Robot` has been defined, `moro` can compute the homogeneous transformations that describe the pose of each robot frame.

These transformations form the basis of forward kinematics: given the joint variables, determine the position and orientation of the robot links and end-effector.

This section focuses on how to obtain and work with those transformations in `moro`. For the mathematical derivation of forward kinematics, see the corresponding section in **Theory**.

## Computing the end-effector pose

The homogeneous transformation of the end-effector frame `{n}` with respect to the base frame `{0}` is available through the `T` property:

```python id="9xg6h2"
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The forward kinematics of the complete manipulator can then be obtained with:

```python id="p6mw8j"
T = robot.T
T
```

For this planar 2R manipulator, the result corresponds to:

[
T_2^0,
]

the pose of frame `{2}` with respect to the base frame `{0}`.

Because the robot parameters are symbolic, the resulting matrix remains symbolic in terms of (l_1), (l_2), (q_1), and (q_2).

The end-effector transformation is equivalent to:

```python id="g2fj9y"
robot.T_i0(robot.dof)
```

## Transformations with respect to the base frame

The transformation of any intermediate frame `{i}` with respect to the base frame can be obtained with:

```python id="w7e4md"
robot.T_i0(i)
```

For example:

```python id="4sae5u"
T10 = robot.T_i0(1)
T20 = robot.T_i0(2)
```

These correspond to:

[
T_1^0
]

and:

[
T_2^0.
]

For the base frame itself:

```python id="x4h1cs"
robot.T_i0(0)
```

returns the (4\times4) identity matrix:

[
T_0^0 = I.
]

This makes it possible to inspect any stage of the kinematic chain rather than only the final end-effector pose.

## Transformations between arbitrary robot frames

For a more general transformation between two robot frames, use:

```python id="8e4fua"
robot.T_ij(i, j)
```

This returns:

[
T_i^j,
]

the pose of frame `{i}` expressed with respect to frame `{j}`.

For example:

```python id="0wem7v"
T20 = robot.T_ij(2, 0)
T21 = robot.T_ij(2, 1)
T02 = robot.T_ij(0, 2)
```

These represent:

[
T_2^0,\qquad
T_2^1,\qquad
T_0^2.
]

The last example describes the base frame relative to frame `{2}` and therefore corresponds to the inverse transformation of (T_2^0).

Both frame indices must satisfy:

```text id="kp6vk2"
0 <= i <= robot.dof
0 <= j <= robot.dof
```

If both indices are equal:

```python id="g4u4oa"
robot.T_ij(1, 1)
```

the result is the identity matrix.

## Frame orientations

If only the orientation of a frame is required, use:

```python id="v3qb1w"
robot.R_i0(i)
```

This returns the rotation matrix:

[
R_i^0,
]

which describes the orientation of frame `{i}` with respect to the base frame.

For example:

```python id="b7d7cq"
R10 = robot.R_i0(1)
R20 = robot.R_i0(2)
```

The rotation matrix can also be obtained manually from the corresponding homogeneous transformation:

```python id="q6ek2j"
R20_from_T = robot.T_i0(2)[:3, :3]
```

but `R_i0()` provides a more direct interface.

## Frame origins and axes

Forward kinematics also provides access to geometric quantities associated with each frame.

### Frame origins

The position of the origin of frame `{i}` with respect to the base frame is available through:

```python id="ybwjk5"
robot.r_o(i)
```

For example:

```python id="2llhdd"
r1 = robot.r_o(1)
r2 = robot.r_o(2)
```

Each result is a three-component column vector.

For the 2R planar manipulator, `robot.r_o(2)` gives the position of the end-effector origin.

This is equivalent to extracting the translation part of the corresponding homogeneous transformation:

```python id="epkgxc"
r2_from_T = robot.T_i0(2)[:3, 3]
```

### Frame z-axes

The direction of the (z_i) axis expressed in the base frame is available through:

```python id="3i7u97"
robot.z(i)
```

For example:

```python id="pl3t8v"
z0 = robot.z(0)
z1 = robot.z(1)
```

The base-frame (z)-axis is:

[
z_0 =
\begin{bmatrix}
0\
0\
1
\end{bmatrix}.
]

These axis vectors are especially useful when working with differential kinematics and geometric Jacobians.

## Evaluating a configuration

A symbolic forward-kinematics result can be evaluated at a particular robot configuration using SymPy substitution.

Consider:

[
l_1 = 1,\qquad
l_2 = 1,\qquad
q_1 = 0.5,\qquad
q_2 = 0.8.
]

Create a substitution dictionary:

```python id="zqd1j5"
values = {
    l1: 1,
    l2: 1,
    q1: 0.5,
    q2: 0.8,
}
```

The end-effector pose can then be evaluated with:

```python id="v9nd3m"
T_num = robot.T.subs(values)
T_num
```

To obtain floating-point values explicitly:

```python id="vug4fm"
T_num.evalf()
```

Intermediate frames can be evaluated in exactly the same way:

```python id="cq4f4k"
T10_num = robot.T_i0(1).subs(values).evalf()
```

as can positions and orientations:

```python id="lnxdwc"
r2_num = robot.r_o(2).subs(values).evalf()
R20_num = robot.R_i0(2).subs(values).evalf()
```

This makes it possible to derive the kinematics symbolically once and then evaluate the model for multiple configurations.

## Working with symbolic results

Since the transformations returned by `moro` are SymPy matrices, the standard SymPy operations can be applied directly.

For example, a matrix can be simplified with:

```python id="7eot8g"
from sympy import simplify

T_simplified = simplify(robot.T)
```

Individual matrix elements can also be accessed:

```python id="9at2ye"
x = robot.T[0, 3]
y = robot.T[1, 3]
z = robot.T[2, 3]
```

For the planar 2R robot, `x` and `y` correspond to the symbolic coordinates of the end-effector position.

You can then manipulate these expressions normally:

```python id="ggxi7j"
x_simplified = simplify(x)
```

or evaluate them separately:

```python id="ywv3yx"
x.subs(values).evalf()
```

This symbolic-first workflow is useful when the equations themselves are part of the analysis.

## A worked example

Consider again the planar 2R manipulator:

```python id="5ba9rc"
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The complete forward kinematics is:

```python id="e0hkc0"
T20 = robot.T
```

The intermediate frame transformation is:

```python id="yyj1fj"
T10 = robot.T_i0(1)
```

The relative transformation between frames `{1}` and `{2}` is:

```python id="h1c91g"
T21 = robot.T_ij(2, 1)
```

The end-effector position can be obtained directly:

```python id="lo0gv2"
p = robot.r_o(2)
```

and its orientation with:

```python id="4ru3ny"
R = robot.R_i0(2)
```

Now evaluate the robot at:

[
l_1 = 1,\qquad
l_2 = 0.8,\qquad
q_1 = \frac{\pi}{4},\qquad
q_2 = -\frac{\pi}{6}.
]

```python id="3g4daz"
from sympy import pi

values = {
    l1: 1,
    l2: 0.8,
    q1: pi / 4,
    q2: -pi / 6,
}
```

The numerical end-effector pose is:

```python id="3o5pfm"
T20_num = T20.subs(values).evalf()
```

The numerical position is:

```python id="qajvqc"
p_num = p.subs(values).evalf()
```

and the numerical orientation is:

```python id="x7un4j"
R_num = R.subs(values).evalf()
```

This same robot model can now be reused directly for Jacobian computation, inverse kinematics, dynamics, or visualization.

## Notes and conventions

When working with forward kinematics in `moro`, keep the following conventions in mind:

* frame `{0}` is the base frame;
* frame `{n}` is the final frame associated with the end-effector;
* `robot.T` represents (T_n^0);
* `robot.T_i0(i)` represents (T_i^0);
* `robot.T_ij(i, j)` represents (T_i^j);
* valid robot frame indices range from `0` to `robot.dof`;
* `robot.R_i0(i)` returns the orientation of frame `{i}` with respect to `{0}`;
* `robot.r_o(i)` returns the position of the origin of frame `{i}` expressed in `{0}`;
* `robot.z(i)` returns the direction of the (z_i) axis expressed in `{0}`;
* symbolic parameters remain symbolic until numerical values are substituted;
* angles follow the conventions established when the robot model is created.

The transformation matrices used by `Robot` follow the Denavit-Hartenberg convention described in the **Theory** section.

## See also

* **Robot Modeling** — define serial manipulators and inspect their structural parameters.
* **Transformations** — work directly with rotation matrices and homogeneous transformations.
* **Jacobians** — use frame positions and axes to compute differential kinematics.
* **Inverse Kinematics** — solve for joint configurations that produce a desired end-effector position.
* **Visualization** — display robot configurations obtained from the kinematic model.
* **Theory → Forward Kinematics** — mathematical derivation and interpretation of the forward-kinematics equations.
* **API Reference → Robot** — complete reference for the kinematic methods and properties of `Robot`.
