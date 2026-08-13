# Denavit--Hartenberg convention

The Denavit--Hartenberg convention provides a systematic way to describe the geometry of a serial manipulator using a small set of parameters associated with consecutive reference frames.

Moro uses the **classic Denavit--Hartenberg convention**.

The notation and transformation rules in this section follow the conventions established in [Mathematical notation and conventions](notation.md), [Rotations](rotations.md), and [Homogeneous transformations](homogeneous-transformations.md).

## Classic Denavit--Hartenberg formulation

For two consecutive reference frames $\{i-1\}$ and $\{i\}$, the relative pose is described by four parameters:

$$
a_i,\qquad
\alpha_i,\qquad
d_i,\qquad
\theta_i.
$$

In Moro, these parameters are always ordered as

$$
\boxed{
(a_i,\alpha_i,d_i,\theta_i).
}
$$

The parameters define the homogeneous transformation

$$
\boxed{
T_i^{i-1}
=
R_z(\theta_i)
T_z(d_i)
T_x(a_i)
R_x(\alpha_i).
}
$$

Thus, the transformation from frame $\{i\}$ to frame $\{i-1\}$ is obtained through the following sequence:

1. rotate by $\theta_i$ about $z_{i-1}$,
2. translate by $d_i$ along $z_{i-1}$,
3. translate by $a_i$ along $x_i$,
4. rotate by $\alpha_i$ about $x_i$.

The resulting homogeneous transformation is

$$
\boxed{
T_i^{i-1}
=
\begin{bmatrix}
\cos\theta_i &
-\sin\theta_i\cos\alpha_i &
\sin\theta_i\sin\alpha_i &
a_i\cos\theta_i
\\
\sin\theta_i &
\cos\theta_i\cos\alpha_i &
-\cos\theta_i\sin\alpha_i &
a_i\sin\theta_i
\\
0 &
\sin\alpha_i &
\cos\alpha_i &
d_i
\\
0 & 0 & 0 & 1
\end{bmatrix}.
}
$$

This is the matrix returned by Moro's `dh(a, alpha, d, theta)` function.

```{important}
Moro implements the **classic** Denavit--Hartenberg convention. Modified Denavit--Hartenberg parameters use a different frame assignment and transformation order.
```

## Assignment of reference frames

The Denavit--Hartenberg convention associates one reference frame with each stage of the serial kinematic chain.

The most important assignment rule is

$$
\boxed{
z_{i-1}
\text{ is aligned with the axis of joint }i.
}
$$

Therefore:

- a revolute joint $i$ rotates about $z_{i-1}$,
- a prismatic joint $i$ translates along $z_{i-1}$.

The $x_i$ axis is chosen along the common normal between $z_{i-1}$ and $z_i$, pointing from $z_{i-1}$ toward $z_i$ whenever the geometry determines a unique common normal.

The origin $O_i$ is normally placed at the intersection between $x_i$ and $z_i$.

Finally, $y_i$ completes a right-handed Cartesian frame:

$$
\hat y_i
=
\hat z_i\times\hat x_i.
$$

## Geometric meaning of the DH parameters

Once the frames are assigned, the four parameters have a precise geometric interpretation.

### Joint angle

The parameter

$$
\theta_i
$$

is the angle from $x_{i-1}$ to $x_i$, measured about $z_{i-1}$ according to the right-hand rule.

A positive value therefore corresponds to a positive rotation about $+z_{i-1}$.

### Joint offset

The parameter

$$
d_i
$$

is the signed translation along $z_{i-1}$.

A positive value corresponds to translation in the $+z_{i-1}$ direction.

### Link length

The parameter

$$
a_i
$$

is the signed distance along $x_i$ between the axes $z_{i-1}$ and $z_i$.

It is commonly interpreted as a link-length parameter, although its exact geometric meaning depends on the selected frame assignment.

A positive value corresponds to translation in the $+x_i$ direction.

### Link twist

The parameter

$$
\alpha_i
$$

is the angle from $z_{i-1}$ to $z_i$, measured about $x_i$ according to the right-hand rule.

A positive value therefore corresponds to a positive rotation about $+x_i$.

The four parameters may be summarized as

$$
\boxed{
\begin{aligned}
\theta_i &: \text{rotation about } +z_{i-1},\\
d_i &: \text{translation along } +z_{i-1},\\
a_i &: \text{translation along } +x_i,\\
\alpha_i &: \text{rotation about } +x_i.
\end{aligned}
}
$$

## Revolute and prismatic joints

The variable DH parameter depends on the joint type.

For a revolute joint,

$$
\boxed{
\theta_i=q_i,
}
$$

while $d_i$ remains constant.

For a prismatic joint,

$$
\boxed{
d_i=q_i,
}
$$

while $\theta_i$ remains constant.

The geometric parameters $a_i$ and $\alpha_i$ are constant for a rigid serial manipulator.

Thus,

$$
\begin{array}{c|c|c}
\text{Joint type} & \text{Variable parameter} & \text{Constant parameter}\\
\hline
\text{Revolute} & \theta_i & d_i\\
\text{Prismatic} & d_i & \theta_i
\end{array}
$$

## The base frame

For the first joint,

$$
z_0
$$

is aligned with the axis of joint 1.

The remaining orientation of frame $\{0\}$ may be chosen conveniently, provided the DH assignment remains consistent.

In particular, $x_0$ is usually selected using a geometrically meaningful direction or in a way that simplifies the resulting DH table.

The $y_0$ axis is then determined by the right-hand rule.

The base frame is therefore not always unique.

```{note}
Different valid choices of the base frame may lead to different DH parameter tables while still describing the same physical robot.
```

## The terminal frame

For an $n$-joint manipulator, the axis $z_{n-1}$ is fixed by the final joint.

The orientation of $z_n$ is not necessarily constrained by another physical joint, so the final frame $\{n\}$ usually has some freedom.

When possible, $z_n$ may be chosen parallel to $z_{n-1}$ to simplify the final parameters, often producing

$$
\alpha_n=0.
$$

The terminal DH frame does not necessarily have to coincide with the physical tool or end-effector frame.

A separate tool frame $\{E\}$ may be introduced:

$$
\{0\}
\rightarrow
\{1\}
\rightarrow
\cdots
\rightarrow
\{n\}
\rightarrow
\{E\}.
$$

Its pose with respect to the base frame is then

$$
\boxed{
T_E^0
=
T_n^0T_E^n.
}
$$

Likewise, a world frame $\{W\}$ may be defined separately from the robot base:

$$
T_E^W
=
T_0^W
T_n^0
T_E^n.
$$

This distinction keeps the kinematic model of the robot separate from application-specific base and tool offsets.

## DH parameter tables

A DH table contains one row for each relative transformation between consecutive reference frames.

Moro uses the column order

$$
\boxed{
(a_i,\alpha_i,d_i,\theta_i).
}
$$

A generic table therefore has the form

| $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
|---:|---:|---:|---:|---:|
| 1 | $a_1$ | $\alpha_1$ | $d_1$ | $\theta_1$ |
| 2 | $a_2$ | $\alpha_2$ | $d_2$ | $\theta_2$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $n$ | $a_n$ | $\alpha_n$ | $d_n$ | $\theta_n$ |

The parameters in row $i$ define

$$
T_i^{i-1}.
$$

Thus,

$$
T_i^{i-1}
=
dh(a_i,\alpha_i,d_i,\theta_i).
$$

The order used in the table intentionally matches the function signature

```python
dh(a, alpha, d, theta)
```

used by Moro.

```{note}
Some textbooks use a different column order, such as $(\theta_i,d_i,a_i,\alpha_i)$. The mathematical convention is unchanged as long as each parameter is interpreted consistently.
```

## Forward composition

Once the relative DH transformations have been constructed,

$$
T_1^0,\qquad
T_2^1,\qquad
\ldots,\qquad
T_n^{n-1},
$$

the pose of the final frame with respect to the base frame is

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

More generally,

$$
T_k^j
=
T_i^jT_k^i.
$$

This composition follows directly from the homogeneous-transformation convention used throughout Moro.

## Sign conventions

The signs of the DH parameters depend on the orientations assigned to the reference-frame axes.

Angles follow the right-hand rule:

- $\theta_i$ is positive about $+z_{i-1}$,
- $\alpha_i$ is positive about $+x_i$.

Translations follow the positive directions of their corresponding axes:

- $d_i>0$ means translation along $+z_{i-1}$,
- $a_i>0$ means translation along $+x_i$.

DH parameters are not required to be positive.

For example,

$$
a_i<0
$$

is mathematically valid if the selected frame assignment produces a translation in the $-x_i$ direction.

Similarly, negative values of $d_i$, $\theta_i$, and $\alpha_i$ are valid.

## Units

Moro does not impose a specific unit of length.

The parameters

$$
a_i,\qquad d_i
$$

may be expressed in meters, millimeters, or another unit, provided the same unit system is used consistently throughout the model.

The angular parameters

$$
\alpha_i,\qquad\theta_i
$$

are interpreted in **radians** by the `dh` function.

Moro does not automatically convert DH angle values from degrees.

## Special geometric cases

The assignment of the $x_i$ axis depends on the relative geometry of $z_{i-1}$ and $z_i$.

### Skew axes

If $z_{i-1}$ and $z_i$ are neither parallel nor intersecting, they have a unique common normal.

The $x_i$ axis is selected along this common normal, pointing from $z_{i-1}$ toward $z_i$.

In this case,

$$
a_i
$$

is the signed distance between the two joint axes along $x_i$.

### Intersecting axes

If $z_{i-1}$ and $z_i$ intersect, the common-normal length is zero.

Therefore,

$$
\boxed{
a_i=0.
}
$$

The $x_i$ axis is chosen perpendicular to the plane defined by the two joint axes, with a direction consistent with the remaining DH frame assignment.

### Parallel axes

If $z_{i-1}$ and $z_i$ are parallel, infinitely many common normals exist.

The $x_i$ axis may therefore be selected along a convenient common normal.

A useful choice is usually one that simplifies the DH parameters and the resulting symbolic expressions.

### Collinear axes

If $z_{i-1}$ and $z_i$ are collinear, then

$$
a_i=0.
$$

Depending on the relative direction assigned to the two axes, the twist may be chosen as

$$
\alpha_i=0
$$

or

$$
\alpha_i=\pi.
$$

There is additional freedom in the choice of $x_i$ in this case.

## Non-uniqueness of DH assignments

A Denavit--Hartenberg description of a robot is not necessarily unique.

Different valid frame assignments may produce different parameter tables while describing exactly the same physical mechanism.

What matters is that:

- each joint axis is assigned consistently,
- the DH geometric rules are respected,
- the parameter table matches the selected frames,
- the resulting transformations correctly describe the kinematic chain.

When several valid assignments are possible, it is usually convenient to prefer the one that produces the simplest parameters, for example by introducing zeros or simple constant angles where appropriate.

```{important}
Two different DH tables are not necessarily inconsistent. They may correspond to two different, but equally valid, assignments of reference frames.
```

## Example: planar 2R manipulator

Consider a planar manipulator with two revolute joints and link lengths $a_1$ and $a_2$.

Both joint axes are parallel to the $z$ direction and perpendicular to the plane of motion.

A convenient DH table is

| $i$ | $a_i$ | $\alpha_i$ | $d_i$ | $\theta_i$ |
|---:|---:|---:|---:|---:|
| 1 | $a_1$ | $0$ | $0$ | $q_1$ |
| 2 | $a_2$ | $0$ | $0$ | $q_2$ |

Because both joints are revolute,

$$
\theta_1=q_1,
\qquad
\theta_2=q_2.
$$

The first transformation is

$$
T_1^0
=
dh(a_1,0,0,q_1),
$$

or

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

Similarly,

$$
T_2^1
=
dh(a_2,0,0,q_2),
$$

so

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

The pose of the second frame with respect to the base is

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
0&0&1&0\\
0&0&0&1
\end{bmatrix}.
$$

Therefore, the position of the terminal point is

$$
\boxed{
x
=
a_1\cos q_1
+
a_2\cos(q_1+q_2),
}
$$

$$
\boxed{
y
=
a_1\sin q_1
+
a_2\sin(q_1+q_2).
}
$$

This simple example illustrates the complete DH workflow:

$$
\boxed{
\text{frame assignment}
\rightarrow
\text{DH table}
\rightarrow
T_i^{i-1}
\rightarrow
T_n^0.
}
$$

The resulting transformation is the basis for the forward kinematic model discussed in the following section.

## Summary of conventions used by Moro

| Convention | Moro |
|---|---|
| DH formulation | Classic Denavit--Hartenberg |
| Parameter order | $(a_i,\alpha_i,d_i,\theta_i)$ |
| Row $i$ defines | $T_i^{i-1}$ |
| Joint $i$ axis | $z_{i-1}$ |
| $\theta_i$ | Rotation about $z_{i-1}$ |
| $d_i$ | Translation along $z_{i-1}$ |
| $a_i$ | Translation along $x_i$ |
| $\alpha_i$ | Rotation about $x_i$ |
| Revolute variable | $\theta_i=q_i$ |
| Prismatic variable | $d_i=q_i$ |
| Positive angular direction | Right-hand rule |
| Angular units | Radians |
| Length units | User-defined but consistent |
| DH matrix | `dh(a, alpha, d, theta)` |
| Forward composition | $T_n^0=T_1^0T_2^1\cdots T_n^{n-1}$ |

These conventions are used throughout Moro when describing serial manipulators using Denavit--Hartenberg parameters.