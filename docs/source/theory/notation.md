# Mathematical notation and conventions

This section summarizes the notation and mathematical conventions used throughout Moro's documentation.

The goal is not to introduce all the mathematical concepts required for robot kinematics and dynamics, but to establish a consistent notation that will be used in the following sections.

## Reference frames

Reference frames are denoted by braces:

$$
\{0\}, \qquad \{1\}, \qquad \{2\}, \ldots
$$

The origin of frame $\{i\}$ is denoted by $O_i$.

For a serial manipulator, frame $\{0\}$ usually represents the base frame, while frames $\{1\},\ldots,\{n\}$ are associated with the successive links or joints of the robot.

Unless otherwise stated, mathematical descriptions of links and joints use indices starting from $1$.

## Scalars, vectors, and matrices

Scalar quantities are written using ordinary italic symbols, for example

$$
q_i, \qquad \theta_i, \qquad d_i.
$$

Vectors are represented with an arrow:

$$
\vec{r}, \qquad
\vec{v}, \qquad
\vec{\omega}.
$$

Matrices are represented by uppercase symbols:

$$
R, \qquad
T, \qquad
J.
$$

Vectors are assumed to be **column vectors** unless otherwise stated.

For example,

$$
\vec{r}
=
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}.
$$

## Position vectors

The notation

$$
\vec{r}_{P/Q}^{A}
$$

denotes the vector directed from point $Q$ to point $P$, with its components expressed in reference frame $\{A\}$.

Geometrically,

$$
\vec{r}_{P/Q}
=
\overrightarrow{QP}.
$$

The superscript indicates only the frame in which the vector components are expressed.

Therefore,

$$
\vec{r}_{P/Q}^{A}
$$

and

$$
\vec{r}_{P/Q}^{B}
$$

represent the same geometric vector expressed using two different coordinate systems.

Relative position vectors satisfy

$$
\vec{r}_{P/Q}^{A}
=
\vec{r}_{P}^{A}
-
\vec{r}_{Q}^{A}.
$$

### Position of a point with respect to a frame origin

When the reference point is the origin of the frame in which the vector is expressed, the notation is simplified as

$$
\boxed{
\vec{r}_{P}^{A}
\equiv
\vec{r}_{P/O_A}^{A}
}
$$

Thus,

$$
\vec{r}_{P}^{0}
$$

denotes the position of point $P$ with respect to the origin $O_0$, expressed in frame $\{0\}$.

For example,

$$
\vec{r}_{P}^{0}
=
\begin{bmatrix}
x_P \\
y_P \\
z_P
\end{bmatrix}_{\{0\}}.
$$

The shorter notation $\vec{P}^{A}$ is sometimes used in robotics literature and classroom derivations. Moro's documentation uses $\vec{r}_{P}^{A}$ as the preferred notation because it explicitly identifies the quantity as a position vector.

## Rotation matrices

The orientation of frame $\{i\}$ with respect to frame $\{j\}$ is denoted by

$$
R_i^j.
$$

The superscript identifies the **reference frame**, while the subscript identifies the frame whose orientation is being described.

Equivalently, $R_i^j$ transforms the coordinates of a vector expressed in frame $\{i\}$ into coordinates expressed in frame $\{j\}$:

$$
\boxed{
\vec{r}^{j}
=
R_i^j
\vec{r}^{i}
}
$$

for the same geometric vector $\vec{r}$.

For example,

$$
\vec{r}_{P/Q}^{0}
=
R_1^0
\vec{r}_{P/Q}^{1}.
$$

The inverse transformation is

$$
R_j^i
=
\left(R_i^j\right)^{-1}
=
\left(R_i^j\right)^T.
$$

Rotation matrices compose according to the reference frames. For example,

$$
R_2^0
=
R_1^0 R_2^1.
$$

This notation follows the convention used throughout Moro's documentation.

```{note}
Some robotics texts place the reference-frame superscript to the left of the matrix and write the same relationship as

$$
{}^jR_i.
$$

Thus, the notation used in Moro,

$$
R_i^j,
$$

corresponds conceptually to

$$
{}^jR_i
$$

in that alternative convention.
```

## Homogeneous transformation matrices

A homogeneous transformation describing frame $\{i\}$ with respect to frame $\{j\}$ is denoted by

$$
T_i^j.
$$

It contains both the relative orientation and the relative position of the two frames:

$$
T_i^j
=
\begin{bmatrix}
R_i^j & \vec{r}_{O_i}^{j} \\
0 & 1
\end{bmatrix},
$$

where

* $R_i^j$ represents the orientation of frame $\{i\}$ with respect to frame $\{j\}$,
* $\vec{r}_{O_i}^{j}$ represents the position of the origin $O_i$ with respect to $O_j$, expressed in frame $\{j\}$.

Homogeneous transformations compose in the same way as rotation matrices:

$$
T_2^0
=
T_1^0 T_2^1.
$$

For a serial manipulator,

$$
T_n^0
=
T_1^0
T_2^1
\cdots
T_n^{n-1}.
$$

The mathematical properties and construction of homogeneous transformations are discussed in more detail in the corresponding theory section.

## Joint variables

The configuration of a robot with $n$ joints is represented by the generalized-coordinate vector

$$
\vec{q}
=
\begin{bmatrix}
q_1 &
q_2 &
\cdots &
q_n
\end{bmatrix}^{T}.
$$

The meaning of each $q_i$ depends on the joint type.

For a revolute joint,

$$
q_i = \theta_i,
$$

where $\theta_i$ is an angular displacement.

For a prismatic joint,

$$
q_i = d_i,
$$

where $d_i$ is a linear displacement.

Their first and second time derivatives are written as

$$
\dot{\vec{q}}
=
\begin{bmatrix}
\dot q_1 &
\dot q_2 &
\cdots &
\dot q_n
\end{bmatrix}^{T},
$$

and

$$
\ddot{\vec{q}}
=
\begin{bmatrix}
\ddot q_1 &
\ddot q_2 &
\cdots &
\ddot q_n
\end{bmatrix}^{T}.
$$

These vectors represent joint velocities and joint accelerations, respectively.

## Linear and angular quantities

The position, linear velocity, and linear acceleration of a point $P$, expressed in frame $\{A\}$, are denoted by

$$
\vec{r}_P^{A},
\qquad
\vec{v}_P^{A},
\qquad
\vec{a}_P^{A}.
$$

When the reference point needs to be made explicit, the same relative-position convention may be used:

$$
\vec{r}_{P/Q}^{A}.
$$

Angular velocity and angular acceleration are denoted by

$$
\vec{\omega},
\qquad
\vec{\alpha}.
$$

Additional subscripts and superscripts may be included when it is necessary to identify the body or frame involved and the frame in which the vector is expressed.

For example,

$$
\vec{\omega}_i^{0}
$$

denotes the angular velocity associated with link or frame $i$, expressed in frame $\{0\}$.

## Time derivatives

A dot over a variable denotes differentiation with respect to time:

$$
\dot q
=
\frac{dq}{dt},
\qquad
\ddot q
=
\frac{d^2q}{dt^2}.
$$

Similarly,

$$
\dot{\vec r},
\qquad
\ddot{\vec r}
$$

denote the first and second time derivatives of a vector representation.

When differentiating vectors expressed in moving reference frames, the reference frame must be considered explicitly. The corresponding relationships are introduced in the sections dealing with velocity and differential kinematics.

## Units

Moro does not impose a particular system of physical units.

Users are responsible for maintaining a consistent unit system throughout a model.

For example, if SI units are used, typical quantities are expressed as

* length in meters,
* angles in radians,
* time in seconds,
* mass in kilograms,
* forces in newtons,
* moments and torques in newton-meters.

Angles supplied to trigonometric functions and robot joint variables are assumed to be expressed in **radians**, unless explicitly stated otherwise.

```{important}
Moro does not automatically perform unit conversions. All quantities participating in the same computation must use a consistent system of units.
```

## Summary of conventions

The main notation used throughout Moro can be summarized as follows:

| Notation              | Meaning                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| $\{A\}$              | Reference frame $A$                                                    |
| $O_A$                 | Origin of frame $\{A\}$                                               |
| $\vec{r}_{P/Q}^{A}$  | Vector from $Q$ to $P$, expressed in $\{A\}$                           |
| $\vec{r}_{P}^{A}$    | Position of $P$ relative to $O_A$, expressed in $\{A\}$                |
| $R_i^j$               | Orientation of frame $\{i\}$ with respect to frame $\{j\}$            |
| $T_i^j$               | Homogeneous transformation from frame $\{i\}$ to frame $\{j\}$        |
| $\vec{q}$             | Vector of generalized joint coordinates                                |
| $\dot{\vec{q}}$       | Vector of joint velocities                                             |
| $\ddot{\vec{q}}$      | Vector of joint accelerations                                          |
| $\vec{\omega}_i^{A}$ | Angular velocity associated with frame or link $i$, expressed in $\{A\}$ |

These conventions are used consistently in the remaining theoretical sections and examples.
