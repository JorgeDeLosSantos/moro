# Forward kinematics

Forward kinematics determines the pose of a robot or one of its intermediate frames from the values of its joint variables.

For a serial manipulator, the joint variables are collected in the configuration vector

$$
\vec q=
\begin{bmatrix}
q_1 &
q_2 &
\cdots &
q_n
\end{bmatrix}^{T}.
$$

The notation and transformation rules used in this section follow the conventions established in [Mathematical notation and conventions](notation.md), [Rotations](rotations.md), [Homogeneous transformations](homogeneous-transformations.md), and [Denavit--Hartenberg convention](denavit-hartenberg.md).

## Robot configuration

The configuration of an $n$-joint serial manipulator is described by

$$
\boxed{
\vec q\in\mathcal Q
}
$$

where $\mathcal Q$ denotes the robot's configuration space.

Each coordinate $q_i$ represents the variable associated with joint $i$.

For a revolute joint,

$$
q_i=\theta_i,
$$

while for a prismatic joint,

$$
q_i=d_i.
$$

Therefore,

$$
q_i=
\begin{cases}
\theta_i, & \text{revolute joint},\\[2mm]
d_i, & \text{prismatic joint}.
\end{cases}
$$

The configuration space contains the admissible joint configurations of the robot.

When joint limits are present, they restrict the allowed values of the corresponding coordinates. For example,

$$
q_i^{\min}
\leq
q_i
\leq
q_i^{\max}.
$$

The vector $\vec q$ describes the internal configuration of the manipulator. It should not be confused with the pose of the end effector.

Different joint configurations may, in general, produce the same end-effector pose.

## Forward kinematic map

The complete forward kinematics of a serial manipulator may be represented as a map

$$
\boxed{
f:\mathcal Q\rightarrow SE(3)
}
$$

such that

$$
\boxed{
T_n^0=f(\vec q).
}
$$

The transformation

$$
T_n^0
$$

describes the pose of the terminal frame $\{n\}$ with respect to the base frame $\{0\}$.

It has the form

$$
T_n^0(\vec q)
=
\begin{bmatrix}
R_n^0(\vec q) &
\vec r_{O_n}^{\,0}(\vec q)\\
0 & 1
\end{bmatrix}.
$$

Thus, the forward kinematic map contains both:

- the orientation
  $$
  R_n^0(\vec q),
  $$
- and the position
  $$
  \vec r_{O_n}^{\,0}(\vec q).
  $$

## Position and pose kinematics

In some applications, the complete pose is required:

$$
\boxed{
\vec q
\longmapsto
T_n^0.
}
$$

In other applications, only the position of the terminal point is relevant.

The positional component of the forward kinematics may therefore be written as

$$
\boxed{
f_p:\mathcal Q\rightarrow\mathbb R^3
}
$$

with

$$
\boxed{
\vec r_{O_n}^{\,0}
=
f_p(\vec q).
}
$$

Similarly, the orientation component can be regarded as

$$
R_n^0
=
f_R(\vec q),
$$

with

$$
f_R:\mathcal Q\rightarrow SO(3).
$$

These maps are different components of the same forward kinematic model.

```{note} id="k3x8hq"
A robot may have multiple configurations that produce the same position or pose. This non-uniqueness becomes especially important in inverse kinematics.
```

## Serial kinematic chains

For a serial manipulator, each pair of consecutive reference frames is related by a homogeneous transformation

$$
T_i^{i-1}.
$$

The pose of frame $\{i\}$ with respect to the base is obtained by composing all previous transformations:

$$
\boxed{
T_i^0
=
T_1^0
T_2^1
\cdots
T_i^{i-1}.
}
$$

For the terminal frame,

$$
\boxed{
T_n^0
=
T_1^0
T_2^1
\cdots
T_n^{n-1}.
}
$$

When classic Denavit--Hartenberg parameters are used,

$$
T_i^{i-1}
=
T_i^{i-1}
(a_i,\alpha_i,d_i,\theta_i),
$$

with

$$
T_i^{i-1}
=
R_z(\theta_i)
T_z(d_i)
T_x(a_i)
R_x(\alpha_i).
$$

For a rigid serial manipulator, one of $d_i$ or $\theta_i$ depends on the joint variable $q_i$, depending on whether the joint is prismatic or revolute.

Thus, the forward kinematic process can be summarized as

$$
\boxed{
\vec q
\rightarrow
T_i^{i-1}
\rightarrow
T_n^0
\rightarrow
\begin{cases}
\text{position},\\
\text{orientation}.
\end{cases}
}
$$

## Intermediate frames

Forward kinematics is not limited to the terminal frame.

For any intermediate frame $\{i\}$,

$$
T_i^0
=
T_1^0
T_2^1
\cdots
T_i^{i-1}.
$$

The corresponding orientation is

$$
R_i^0,
$$

and the position of its origin is

$$
\vec r_{O_i}^{\,0}.
$$

Therefore,

$$
T_i^0
=
\begin{bmatrix}
R_i^0 &
\vec r_{O_i}^{\,0}\\
0 & 1
\end{bmatrix}.
$$

Intermediate transformations are useful for describing the pose of individual links and for later calculations involving velocities, Jacobians, centers of mass, and dynamics.

## Points attached to a link

Let $P$ be a point rigidly attached to link $i$.

Suppose its local position is known in frame $\{i\}$:

$$
\vec r_P^{\,i}.
$$

Its homogeneous representation is

$$
\tilde r_P^{\,i}
=
\begin{bmatrix}
\vec r_P^{\,i}\\
1
\end{bmatrix}.
$$

The position of $P$ with respect to the base frame is then

$$
\boxed{
\tilde r_P^{\,0}
=
T_i^0
\tilde r_P^{\,i}.
}
$$

Equivalently, in Cartesian form,

$$
\boxed{
\vec r_P^{\,0}
=
R_i^0\vec r_P^{\,i}
+
\vec r_{O_i}^{\,0}.
}
$$

This relation allows the forward kinematic model to determine the position of arbitrary points attached to the robot, not only reference-frame origins.

Examples include:

- points located along a link,
- centers of mass,
- tool points,
- geometric points used for visualization or analysis.

## Terminal and tool frames

The terminal DH frame $\{n\}$ does not necessarily coincide with the physical tool or end-effector frame.

If a separate tool frame $\{E\}$ is defined through a constant transformation

$$
T_E^n,
$$

then its pose with respect to the robot base is

$$
\boxed{
T_E^0
=
T_n^0T_E^n.
}
$$

Likewise, if the robot base is itself located with respect to an external world frame $\{W\}$,

$$
T_0^W
$$

may be used to obtain

$$
\boxed{
T_E^W
=
T_0^W
T_n^0
T_E^n.
}
$$

This separation allows the robot's internal kinematic model to remain independent of application-specific base and tool offsets.

## Revolute and prismatic joints

The mathematical structure of the forward kinematic chain is the same for revolute and prismatic joints.

The difference lies in which parameter varies with configuration.

For a revolute joint,

$$
\theta_i=q_i,
$$

and the relative transformation depends on $q_i$ through a rotation.

For a prismatic joint,

$$
d_i=q_i,
$$

and the relative transformation depends on $q_i$ through a translation.

Thus, both joint types can be incorporated into the same product

$$
T_n^0(\vec q)
=
T_1^0(q_1)
T_2^1(q_2)
\cdots
T_n^{n-1}(q_n).
$$

## Forward kinematics in Moro

Moro's `Robot` class provides direct access to the main quantities involved in forward kinematics.

The most important relationships are:

| Mathematical quantity | Moro |
|---|---|
| DH parameters | `dh_parameters` |
| DH table | `dh_table` |
| Joint variables | `qs` |
| Joint variable $q_i$ | `q(i)` |
| Number of degrees of freedom | `dof` |
| Joint types | `joint_types` |
| $T_i^0$ | `T_i0(i)` |
| $T_n^0$ | `T` |
| $R_i^0$ | `R_i0(i)` |
| $\vec r_{O_i}^{\,0}$ | `r_o(i)` |
| Direction of $z_i$ expressed in $\{0\}$ | `z(i)` |

The transformation

$$
T_i^0
$$

corresponds directly to

```python
robot.T_i0(i)
```

while the pose of the terminal frame is available through

```python
robot.T
```

The orientation and origin position of an intermediate frame can be obtained using

```python
robot.R_i0(i)
robot.r_o(i)
```

respectively.

The internal transformations between consecutive frames are used by Moro to construct the complete kinematic chain, but implementation details such as private internal caches are not part of the public mathematical interface.

```{note} id="xj7qwf"
This section introduces the mathematical role of the main `Robot` properties and methods. Complete signatures, validation rules, and return types belong to the API reference.
```

## Example: planar 2R manipulator

Consider again the planar two-link manipulator with revolute joint variables

$$
\vec q
=
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix}
$$

and link lengths $a_1$ and $a_2$.

Using the classic DH convention,

| $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
|---:|---:|---:|---:|---:|
| 1 | $a_1$ | $0$ | $0$ | $q_1$ |
| 2 | $a_2$ | $0$ | $0$ | $q_2$ |

The first relative transformation is

$$
T_1^0
=
\begin{bmatrix}
\cos q_1 & -\sin q_1 & 0 & a_1\cos q_1\\
\sin q_1 & \cos q_1 & 0 & a_1\sin q_1\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

The second is

$$
T_2^1
=
\begin{bmatrix}
\cos q_2 & -\sin q_2 & 0 & a_2\cos q_2\\
\sin q_2 & \cos q_2 & 0 & a_2\sin q_2\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

The terminal pose is

$$
T_2^0
=
T_1^0T_2^1.
$$

After multiplication,

$$
T_2^0
=
\begin{bmatrix}
\cos(q_1+q_2) &
-\sin(q_1+q_2) &
0 &
a_1\cos q_1+a_2\cos(q_1+q_2)
\\
\sin(q_1+q_2) &
\cos(q_1+q_2) &
0 &
a_1\sin q_1+a_2\sin(q_1+q_2)
\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

The position of the terminal origin is therefore

$$
\boxed{
\vec r_{O_2}^{\,0}
=
\begin{bmatrix}
a_1\cos q_1+a_2\cos(q_1+q_2)\\
a_1\sin q_1+a_2\sin(q_1+q_2)\\
0
\end{bmatrix}.
}
$$

Its orientation is

$$
\boxed{
R_2^0
=
R_z(q_1+q_2).
}
$$

Because this is a planar manipulator, the terminal orientation can also be represented by the scalar angle

$$
\phi=q_1+q_2.
$$

Thus, the planar forward kinematic map may be written as

$$
\boxed{
(q_1,q_2)
\longmapsto
(x,y,\phi).
}
$$

with

$$
x
=
a_1\cos q_1
+
a_2\cos(q_1+q_2),
$$

$$
y
=
a_1\sin q_1
+
a_2\sin(q_1+q_2),
$$

and

$$
\phi=q_1+q_2.
$$

### The same model in Moro

A symbolic planar 2R robot can be created using the same DH parameters:

```python
from sympy import symbols
from moro import Robot

a1, a2 = symbols("a1 a2")
q1, q2 = symbols("q1 q2")

robot = Robot(
    (a1, 0, 0, q1, "r"),
    (a2, 0, 0, q2, "r"),
)
```

The complete forward kinematics is then available through

```python
robot.T
```

while the pose of an intermediate frame can be obtained using

```python
robot.T_i0(1)
```

and its position and orientation using

```python
robot.r_o(1)
robot.R_i0(1)
```

The same mathematical relationships derived above are therefore available directly through Moro's kinematic interface.

## Forward versus inverse kinematics

Forward kinematics starts with the robot configuration

$$
\vec q
$$

and computes a pose or position:

$$
\boxed{
\vec q
\longrightarrow
T_n^0.
}
$$

Inverse kinematics asks the opposite question:

$$
\boxed{
T_{\mathrm{desired}}
\longrightarrow
\vec q.
}
$$

or, for a position-only problem,

$$
\boxed{
\vec r_{\mathrm{desired}}
\longrightarrow
\vec q.
}
$$

Unlike forward kinematics, which produces a deterministic pose for a given configuration, inverse kinematics may have:

- no solution,
- one solution,
- several solutions,
- or infinitely many solutions.

The inverse problem is discussed separately in the corresponding theory section.

## Summary

The main forward-kinematics conventions used throughout Moro are:

| Concept | Convention |
|---|---|
| Robot configuration | $\vec q=[q_1,\ldots,q_n]^T$ |
| Configuration space | $\vec q\in\mathcal Q$ |
| Full forward kinematics | $T_n^0=f(\vec q)$ |
| Position kinematics | $\vec r_{O_n}^{\,0}=f_p(\vec q)$ |
| Orientation kinematics | $R_n^0=f_R(\vec q)$ |
| Intermediate-frame pose | $T_i^0=T_1^0\cdots T_i^{i-1}$ |
| Terminal-frame pose | $T_n^0=T_1^0\cdots T_n^{n-1}$ |
| Point attached to link $i$ | $\tilde r_P^0=T_i^0\tilde r_P^i$ |
| Tool frame | $T_E^0=T_n^0T_E^n$ |
| Revolute coordinate | $q_i=\theta_i$ |
| Prismatic coordinate | $q_i=d_i$ |
| Moro terminal pose | `Robot.T` |
| Moro intermediate pose | `Robot.T_i0(i)` |
| Moro frame orientation | `Robot.R_i0(i)` |
| Moro frame-origin position | `Robot.r_o(i)` |

Forward kinematics provides the geometric foundation for differential kinematics, Jacobian analysis, inverse kinematics, and robot dynamics.