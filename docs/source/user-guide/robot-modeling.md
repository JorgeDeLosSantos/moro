# Robot Modeling

`moro` represents serial robotic manipulators through the `Robot` class.

A robot model is created from its Denavit-Hartenberg parameters and joint types. Once the model has been defined, the same `Robot` object can be used throughout the library to compute forward kinematics, Jacobians, inverse kinematics, dynamics, and visualizations.

This section focuses on how to define and inspect a robot model. The mathematical details of the Denavit-Hartenberg convention are covered separately in the **Theory** section.

## Creating a serial robot

A robot is created by passing one Denavit-Hartenberg row for each joint:

```python
from moro import Robot
```

For example, a planar two-link manipulator with two revolute joints can be defined as:

```python
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Each row describes the relative transformation between two consecutive frames in the serial kinematic chain.

The number of rows passed to `Robot` determines the number of degrees of freedom of the manipulator.

## Denavit-Hartenberg rows

Each Denavit-Hartenberg row must contain either four or five elements.

The four-parameter form is:

```text
(a_i, alpha_i, d_i, theta_i)
```

The five-parameter form additionally specifies the joint type:

```text
(a_i, alpha_i, d_i, theta_i, joint_type)
```

The parameters correspond to the classical Denavit-Hartenberg convention:

| Position | Parameter    | Description                 |
| -------- | ------------ | --------------------------- |
| 1        | $a_i$        | Link length                 |
| 2        | $\alpha_i$   | Link twist                  |
| 3        | $d_i$        | Link offset                 |
| 4        | $\theta_i$   | Joint angle                 |
| 5        | `joint_type` | Revolute or prismatic joint |

If the joint type is omitted, the joint is assumed to be revolute.

For example, these two definitions are equivalent:

```python
robot1 = Robot(
    (l1, 0, 0, q1),
    (l2, 0, 0, q2),
)
```

and:

```python
robot2 = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Numeric angular parameters must be given in radians.

For a detailed explanation of the frame assignment and the meaning of each parameter, see **Theory → Denavit-Hartenberg Convention**.

## Revolute and prismatic joints

`moro` currently supports two joint types:

* `"r"` for revolute joints;
* `"p"` for prismatic joints.

For a revolute joint, the joint variable is taken from the $\theta_i$ parameter.

For example:

```python
from moro.abc import q1

robot = Robot(
    (1, 0, 0, q1, "r"),
)
```

Here, `q1` represents the rotational joint coordinate.

For a prismatic joint, the joint variable is taken from the $d_i$ parameter:

```python
robot = Robot(
    (0, 0, q1, 0, "p"),
)
```

Here, `q1` represents the translational displacement of the joint.

Joint type identifiers are case-insensitive, so `"R"` and `"P"` are also accepted. For consistency, lowercase `"r"` and `"p"` are recommended in user code and documentation.

## Using symbolic parameters

Robot models can contain symbolic parameters, which makes it possible to derive kinematic and dynamic expressions before assigning numerical values.

### Using predefined symbols

For convenience, `moro.abc` provides several commonly used symbolic variables:

```python
from moro.abc import q1, q2, l1, l2
```

These can be used directly when constructing a robot:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The joint variables available from `moro.abc`, such as `q1` and `q2`, are time-dependent symbolic quantities. This makes them suitable for models that may later be reused for dynamic analysis.

### Defining your own symbols

Using `moro.abc` is optional. You can define your own parameters directly with SymPy.

For example:

```python
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols

l1, l2 = symbols("l1 l2", positive=True)
q1, q2 = dynamicsymbols("q1 q2")
```

These symbols can then be used normally:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

This approach is useful when a model requires custom variable names or assumptions.

## Inspecting the robot model

Once a robot has been created, several properties can be used to inspect its structure.

### Denavit-Hartenberg parameters

The original Denavit-Hartenberg parameters are available through:

```python
robot.dh_parameters
```

For the previous 2R manipulator, this returns one tuple for each row used to construct the robot.

A tabular representation is available through:

```python
robot.dh_table
```

This is useful for checking the model before performing further calculations.

### Joint types

The joint types can be inspected with:

```python
robot.joint_types
```

For a 2R manipulator:

```python
["r", "r"]
```

For a mixed revolute-prismatic manipulator, the result could instead be:

```python
["r", "p"]
```

### Individual transformations

`Robot` also stores the homogeneous transformation associated with each Denavit-Hartenberg row.

They can be inspected through:

```python
robot.Ts
```

The resulting list contains the relative transformation matrices between consecutive frames.

More detailed operations involving these transformations are covered in **Forward Kinematics**.

## Joint variables and degrees of freedom

The number of degrees of freedom is available through:

```python
robot.dof
```

For the planar 2R example:

```python
robot.dof
# 2
```

The joint variables detected from the robot definition are available through:

```python
robot.qs
```

For the same robot:

```python
robot.qs
# [q1, q2]
```

The user does not need to provide the joint variables separately. `moro` determines them from the joint types and the corresponding Denavit-Hartenberg parameters.

For a revolute joint, the joint coordinate is obtained from $\theta_i$, while for a prismatic joint it is obtained from $d_i$.

## Joint limits

Each `Robot` model also stores one pair of limits for every joint.

These can be inspected through:

```python
robot.joint_limits
```

By default, `moro` assigns:

* $(-\pi, \pi)$ to revolute joints;
* $(0, 1000)$ to prismatic joints.

For example:

```python
from sympy import pi

robot = Robot(
    (1, 0, 0, q1, "r"),
    (0, 0, q2, 0, "p"),
)

robot.joint_limits
```

corresponds to:

```python
[(-pi, pi), (0, 1000)]
```

These default values are convenience ranges rather than physical limits of a particular mechanism.

For a real robot, the limits should normally be replaced with values that represent the actual admissible motion.

For example:

```python
robot.joint_limits = [
    (-pi / 2, pi / 2),
    (0, 0.5),
]
```

The number of limit pairs must match the number of degrees of freedom, and each joint limit must be specified as a pair:

```text
(lower_limit, upper_limit)
```

Angular limits are expressed in radians.

Joint limits become especially relevant when solving inverse kinematics problems, where they can be used to constrain the admissible joint configurations.

## A mixed revolute-prismatic example

Consider a two-degree-of-freedom manipulator with one revolute joint followed by one prismatic joint:

```python
from moro import Robot
from moro.abc import q1, q2, l1

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (0, 0, q2, 0, "p"),
)
```

The model can then be inspected with:

```python
robot.dof
```

which returns:

```text
2
```

The joint types are:

```python
robot.joint_types
```

```text
["r", "p"]
```

and the joint variables are:

```python
robot.qs
```

corresponding to:

```text
[q1, q2]
```

The Denavit-Hartenberg table can be inspected with:

```python
robot.dh_table
```

and the default joint limits can be checked with:

```python
robot.joint_limits
```

This same model can then be passed directly to the kinematics, inverse kinematics, dynamics, and visualization workflows provided by `moro`.

## Notes and limitations

The current robot modeling capabilities in `moro` are centered on serial manipulators.

At present:

* robot geometry is described using the classical Denavit-Hartenberg convention;
* revolute and prismatic joints are supported;
* both symbolic and numerical Denavit-Hartenberg parameters can be used;
* numeric angular quantities are expressed in radians;
* one joint coordinate is associated with each Denavit-Hartenberg row;
* URDF and other robot-description formats are not currently supported;
* branched, parallel, or closed-loop kinematic structures are outside the current scope.

Creating a `Robot` primarily defines its kinematic structure.

Dynamic properties such as link masses, centers of mass, inertia tensors, and the gravity vector are configured separately when dynamic modeling is required. These quantities are introduced in the **Dynamics** section of the User Guide.

## See also

* **Theory → Denavit-Hartenberg Convention** — mathematical background for the robot description used by `moro`.
* **Forward Kinematics** — compute transformations and poses throughout the robot chain.
* **Jacobians** — compute geometric Jacobians for the end-effector and other points.
* **Inverse Kinematics** — solve for joint configurations subject to robot constraints.
* **Dynamics** — add physical parameters and derive the robot dynamic model.
* **Visualization** — plot and animate robot configurations.
* **API Reference → Robot** — complete reference for the `Robot` class.
