# Homogeneous transformations

Homogeneous transformation matrices provide a compact way to represent the position and orientation of one reference frame with respect to another.

This section introduces the conventions used by Moro for homogeneous transformations and their composition.

The notation follows the conventions established in [Mathematical notation and conventions](notation.md) and [Rotations](rotations.md).

## Pose between reference frames

The pose of a reference frame $\{i\}$ with respect to a reference frame $\{j\}$ is represented by the homogeneous transformation matrix

$$
T_i^j
=
\begin{bmatrix}
R_i^j & \vec{r}_{O_i}^{\,j}\\
0 & 1
\end{bmatrix}.
$$

Here,

- $R_i^j$ represents the orientation of frame $\{i\}$ with respect to frame $\{j\}$,
- $\vec{r}_{O_i}^{\,j}$ represents the position of the origin $O_i$ with respect to the origin $O_j$, expressed in frame $\{j\}$.

Using the position-vector notation introduced previously,

$$
\vec{r}_{O_i}^{\,j}
\equiv
\vec{r}_{O_i/O_j}^{\,j}.
$$

Operationally, $T_i^j$ transforms homogeneous coordinates expressed in frame $\{i\}$ into homogeneous coordinates expressed in frame $\{j\}$.

Thus,

$$
\boxed{
T_i^j:
\{i\}\rightarrow\{j\}
}
$$

in the sense of coordinate transformation.

## Cartesian and homogeneous coordinates

The Cartesian coordinates of a point $P$, expressed in frame $\{i\}$, are written as

$$
\vec{r}_P^{\,i}
=
\begin{bmatrix}
x_P\\
y_P\\
z_P
\end{bmatrix}.
$$

Its homogeneous representation is

$$
\boxed{
\tilde{r}_P^{\,i}
=
\begin{bmatrix}
\vec{r}_P^{\,i}\\
1
\end{bmatrix}
}
$$

or, explicitly,

$$
\tilde{r}_P^{\,i}
=
\begin{bmatrix}
x_P\\
y_P\\
z_P\\
1
\end{bmatrix}.
$$

Moro's documentation keeps the tilde notation to distinguish Cartesian vectors in $\mathbb{R}^3$ from their homogeneous representations in $\mathbb{R}^4$.

Thus,

$$
\vec{r}_P^{\,i}\in\mathbb{R}^3,
\qquad
\tilde{r}_P^{\,i}\in\mathbb{R}^4.
$$

## Transforming a point

Given

$$
T_i^j
=
\begin{bmatrix}
R_i^j & \vec{r}_{O_i}^{\,j}\\
0 & 1
\end{bmatrix},
$$

the homogeneous coordinates of a point transform according to

$$
\boxed{
\tilde{r}_P^{\,j}
=
T_i^j
\tilde{r}_P^{\,i}.
}
$$

Expanding the matrix product gives

$$
\begin{bmatrix}
\vec{r}_P^{\,j}\\
1
\end{bmatrix}
=
\begin{bmatrix}
R_i^j & \vec{r}_{O_i}^{\,j}\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\vec{r}_P^{\,i}\\
1
\end{bmatrix},
$$

so that

$$
\boxed{
\vec{r}_P^{\,j}
=
R_i^j\vec{r}_P^{\,i}
+
\vec{r}_{O_i}^{\,j}.
}
$$

This equation shows explicitly the two components of a rigid transformation:

1. rotation of the point coordinates,
2. translation of the frame origin.

The homogeneous formulation combines both operations into a single matrix multiplication.

## Relative position vectors

A relative position vector between two points does not depend on the translational offset between reference-frame origins.

For points $P$ and $Q$,

$$
\vec{r}_{P/Q}^{\,j}
=
\vec{r}_P^{\,j}
-
\vec{r}_Q^{\,j}.
$$

Using the rigid-transformation equations for both points,

$$
\vec{r}_{P/Q}^{\,j}
=
R_i^j
\vec{r}_{P/Q}^{\,i}.
$$

Therefore,

$$
\boxed{
\vec{r}_{P/Q}^{\,j}
=
R_i^j
\vec{r}_{P/Q}^{\,i}.
}
$$

The translational part cancels because $\vec{r}_{P/Q}$ represents a displacement between two points rather than the absolute position of a point relative to a frame origin.

This is why relative displacement vectors are normally transformed directly using rotation matrices.

## Free vectors and homogeneous coordinates

Homogeneous coordinates can also represent a free vector $\vec{v}$ using

$$
\tilde{v}
=
\begin{bmatrix}
\vec{v}\\
0
\end{bmatrix}.
$$

The zero fourth component prevents the translational part of a homogeneous transformation from affecting the vector:

$$
T_i^j
\begin{bmatrix}
\vec{v}^{\,i}\\
0
\end{bmatrix}
=
\begin{bmatrix}
R_i^j\vec{v}^{\,i}\\
0
\end{bmatrix}.
$$

However, in Moro's robotics formulations, vectors such as directions, relative displacements, and angular velocities are normally transformed directly using $R_i^j$ rather than by augmenting them with a zero homogeneous coordinate.

```{note}
Homogeneous coordinates with fourth component equal to zero are mathematically useful for representing free vectors, but they are not the primary convention used in Moro's kinematic formulations.
```

## Rigid transformations and SE(3)

A homogeneous rigid transformation has the structure

$$
T
=
\begin{bmatrix}
R & \vec{p}\\
0 & 1
\end{bmatrix},
$$

where

$$
R\in SO(3)
$$

and

$$
\vec{p}\in\mathbb{R}^3.
$$

The set of all such rigid-body transformations forms the **special Euclidean group**

$$
SE(3).
$$

Thus,

$$
T\in SE(3).
$$

For the purposes of Moro, this notation is used to identify a valid rigid-body transformation that combines a three-dimensional rotation and translation.

A deeper treatment of Lie groups and Lie algebras is outside the scope of this section.

## Composition of homogeneous transformations

Homogeneous transformations compose according to their intermediate reference frames.

Suppose

$$
T_i^j
$$

transforms coordinates from frame $\{i\}$ to frame $\{j\}$, and

$$
T_k^i
$$

transforms coordinates from frame $\{k\}$ to frame $\{i\}$.

Then

$$
\tilde{r}_P^{\,j}
=
T_i^j\tilde{r}_P^{\,i}
$$

and

$$
\tilde{r}_P^{\,i}
=
T_k^i\tilde{r}_P^{\,k}.
$$

Substitution gives

$$
\tilde{r}_P^{\,j}
=
T_i^jT_k^i\tilde{r}_P^{\,k}.
$$

Therefore,

$$
\boxed{
T_k^j
=
T_i^jT_k^i.
}
$$

This is completely analogous to the composition rule for rotation matrices.

For example,

$$
T_2^0
=
T_1^0T_2^1.
$$

For a serial chain,

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

This relation is fundamental in the forward kinematics of serial manipulators.

## Inverse of a homogeneous transformation

The inverse transformation reverses the direction of the coordinate mapping:

$$
\boxed{
T_j^i
=
\left(T_i^j\right)^{-1}.
}
$$

Let

$$
T_i^j
=
\begin{bmatrix}
R_i^j & \vec{r}_{O_i}^{\,j}\\
0 & 1
\end{bmatrix}.
$$

Since

$$
(R_i^j)^{-1}
=
(R_i^j)^T,
$$

the inverse homogeneous transformation is

$$
\boxed{
T_j^i
=
\begin{bmatrix}
(R_i^j)^T &
-(R_i^j)^T\vec{r}_{O_i}^{\,j}\\
0 & 1
\end{bmatrix}.
}
$$

Using

$$
R_j^i=(R_i^j)^T,
$$

this may also be written as

$$
T_j^i
=
\begin{bmatrix}
R_j^i &
\vec{r}_{O_j}^{\,i}\\
0 & 1
\end{bmatrix},
$$

where

$$
\boxed{
\vec{r}_{O_j}^{\,i}
=
-
R_j^i
\vec{r}_{O_i}^{\,j}.
}
$$

```{important}
Inverting a homogeneous transformation does **not** generally consist of simply transposing the rotation matrix and changing the sign of the translation vector.

The translation must also be expressed in the new reference frame:

$$
\vec{p}_{\mathrm{inv}}
=
-R^T\vec{p}.
$$
```

## Pure translations

A pure translation by

$$
\vec{d}
=
\begin{bmatrix}
d_x\\
d_y\\
d_z
\end{bmatrix}
$$

is represented by

$$
\boxed{
T(\vec{d})
=
\begin{bmatrix}
I & \vec{d}\\
0 & 1
\end{bmatrix}.
}
$$

Explicitly,

$$
T(\vec{d})
=
\begin{bmatrix}
1 & 0 & 0 & d_x\\
0 & 1 & 0 & d_y\\
0 & 0 & 1 & d_z\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

Elementary translations along the coordinate axes may be written as

$$
T_x(a),
\qquad
T_y(b),
\qquad
T_z(c).
$$

For example,

$$
T_x(a)
=
\begin{bmatrix}
1 & 0 & 0 & a\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

In Moro, the function `htmtra(x, y, z)` constructs a pure homogeneous translation.

For example,

```python
htmtra(x=a, y=b, z=c)
```

corresponds to

$$
T
=
\begin{bmatrix}
1 & 0 & 0 & a\\
0 & 1 & 0 & b\\
0 & 0 & 1 & c\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

## Pure rotations

A pure rotation embedded in homogeneous form is represented by

$$
\boxed{
T_R
=
\begin{bmatrix}
R & 0\\
0 & 1
\end{bmatrix}.
}
$$

For an elementary rotation around the $z$ axis,

$$
T_R
=
\begin{bmatrix}
R_z(\theta) & 0\\
0 & 1
\end{bmatrix}.
$$

In Moro, `htmrot(theta, axis)` constructs the homogeneous representation of an elementary rotation, while `rot2htm(R)` embeds a general $3\times3$ rotation block into a homogeneous matrix.

The mathematical discussion assumes that $R$ represents a valid rotation matrix. Validation details belong to the corresponding API documentation.

## Fixed and moving transformations

A pure rotation or translation matrix does not by itself determine whether the operation is interpreted relative to fixed or moving axes.

That interpretation depends on the side on which the transformation is composed.

Let $T$ denote the current pose of a frame.

If a new transformation $A$ is defined with respect to the fixed or global frame,

$$
\boxed{
T_{\mathrm{new}}
=
A\,T.
}
$$

The new transformation multiplies on the left.

If $A$ is defined with respect to the current local or moving frame,

$$
\boxed{
T_{\mathrm{new}}
=
T\,A.
}
$$

The new transformation multiplies on the right.

This is the homogeneous-transformation counterpart of the fixed-axis and moving-axis composition rules discussed for rotation matrices.

For example, a translation along the global $x$ axis is represented by

$$
T_{\mathrm{new}}
=
T_x(a)T,
$$

while a translation along the local $x$ axis of the current frame is represented by

$$
T_{\mathrm{new}}
=
TT_x(a).
$$

Similarly, a rotation about the global $z$ axis is composed on the left, while a rotation about the current local $z$ axis is composed on the right.

## Noncommutativity

Rigid transformations do not commute in general.

For arbitrary transformations $T_A$ and $T_B$,

$$
\boxed{
T_AT_B
\neq
T_BT_A.
}
$$

This is particularly important when rotations and translations are combined.

For example, rotating a frame and then translating it generally produces a different result from translating it and then rotating it.

Pure translations expressed with respect to the same reference frame do commute:

$$
T(\vec{a})T(\vec{b})
=
T(\vec{b})T(\vec{a})
=
T(\vec{a}+\vec{b}).
$$

However, a rotation and a translation do not generally commute:

$$
T(\vec{d})T_R
\neq
T_RT(\vec{d}).
$$

This dependence on order is fundamental in robot kinematics and becomes particularly important in the Denavit--Hartenberg convention.

## Notation for generic transformations

When the reference frames are important, Moro's documentation uses the indexed notation

$$
T_i^j.
$$

When the identity of the frames is not relevant to the discussion, a generic transformation may be written simply as

$$
T.
$$

Similarly,

$$
R_i^j
$$

is used for a rotation relating specific reference frames, while

$$
R
$$

may be used for a generic rotation matrix.

## Summary of conventions

The homogeneous-transformation conventions used throughout Moro may be summarized as follows:

| Notation or convention | Meaning |
|---|---|
| $T_i^j$ | Pose of frame $\{i\}$ with respect to frame $\{j\}$ |
| $T_i^j:\{i\}\rightarrow\{j\}$ | Coordinate mapping from frame $\{i\}$ to frame $\{j\}$ |
| $\vec{r}_P^{\,i}$ | Cartesian position vector in $\mathbb{R}^3$ |
| $\tilde{r}_P^{\,i}$ | Homogeneous point coordinates in $\mathbb{R}^4$ |
| Fourth homogeneous component for points | $1$ |
| Fourth homogeneous component for free vectors | $0$ |
| Point transformation | $\tilde{r}_P^{\,j}=T_i^j\tilde{r}_P^{\,i}$ |
| Cartesian point transformation | $\vec{r}_P^{\,j}=R_i^j\vec{r}_P^{\,i}+\vec{r}_{O_i}^{\,j}$ |
| Relative displacement transformation | $\vec{r}_{P/Q}^{\,j}=R_i^j\vec{r}_{P/Q}^{\,i}$ |
| Composition | $T_k^j=T_i^jT_k^i$ |
| Inverse | $T_j^i=(T_i^j)^{-1}$ |
| Inverse translation | $-R^T\vec{p}$ |
| Fixed/global operation | Multiply on the left |
| Moving/local operation | Multiply on the right |
| Rigid transformation space | $SE(3)$ |

These conventions provide the foundation for the Denavit--Hartenberg formulation and the forward kinematics of serial manipulators.