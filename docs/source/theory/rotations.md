# Rotations

Rotations are one of the fundamental mathematical tools used to describe the orientation of rigid bodies and reference frames in robotics.

This section introduces the conventions used by Moro for rotation matrices and common orientation representations.

The discussion assumes the notation introduced in [Mathematical notation and conventions](notation.md).

## Rotation matrices

The orientation of a reference frame $\{i\}$ with respect to a reference frame $\{j\}$ is represented by the rotation matrix

$$
R_i^j.
$$

With the conventions used throughout Moro,

$$
\boxed{
\vec{v}^{\,j}
=
R_i^j
\vec{v}^{\,i}
}
$$

where $\vec{v}^{\,i}$ and $\vec{v}^{\,j}$ represent the coordinates of the same geometric vector expressed in frames $\{i\}$ and $\{j\}$, respectively.

Thus, $R_i^j$ transforms vector coordinates from frame $\{i\}$ to frame $\{j\}$.

Moro uses:

- column vectors,
- left multiplication by transformation matrices,
- right-handed coordinate systems,
- positive rotations according to the right-hand rule.

## Properties of rotation matrices

A three-dimensional rotation matrix belongs to the special orthogonal group $SO(3)$:

$$
R \in SO(3).
$$

This means that

$$
R^T R = RR^T = I
$$

and

$$
\det(R)=1.
$$

Consequently,

$$
R^{-1}=R^T.
$$

For the relative orientation between two frames,

$$
\boxed{
R_j^i
=
\left(R_i^j\right)^T
}
$$

and therefore

$$
\vec{v}^{\,i}
=
R_j^i\vec{v}^{\,j}.
$$

## Geometric interpretation

The columns of $R_i^j$ correspond to the unit axes of frame $\{i\}$ expressed in frame $\{j\}$:

$$
R_i^j
=
\begin{bmatrix}
\hat{x}_i^{\,j} &
\hat{y}_i^{\,j} &
\hat{z}_i^{\,j}
\end{bmatrix}.
$$

For example, the first column gives the direction of the $x$ axis of frame $\{i\}$ expressed using the coordinates of frame $\{j\}$.

Because the axes of a Cartesian reference frame form an orthonormal basis, the columns and rows of a rotation matrix are mutually orthogonal unit vectors.

## Elementary rotations

Moro follows the right-hand rule for positive rotations.

A rotation of angle $\theta$ about the $x$ axis is represented by

$$
R_x(\theta)
=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\theta & -\sin\theta \\
0 & \sin\theta & \cos\theta
\end{bmatrix}.
$$

A rotation about the $y$ axis is

$$
R_y(\theta)
=
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}.
$$

A rotation about the $z$ axis is

$$
R_z(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

For example,

$$
R_z\left(\frac{\pi}{2}\right)
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}
=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}.
$$

This corresponds to a positive $90^\circ$ rotation according to the right-hand rule.

The functions `rotx`, `roty`, `rotz`, and `rot` implement these elementary rotations.

## Composition of reference-frame orientations

Relative orientations compose according to their reference frames.

For three frames $\{0\}$, $\{1\}$, and $\{2\}$,

$$
\vec{v}^{\,0}
=
R_1^0\vec{v}^{\,1}
$$

and

$$
\vec{v}^{\,1}
=
R_2^1\vec{v}^{\,2}.
$$

Substitution gives

$$
\vec{v}^{\,0}
=
R_1^0R_2^1\vec{v}^{\,2},
$$

and therefore

$$
\boxed{
R_2^0
=
R_1^0R_2^1
}
$$

More generally,

$$
R_n^0
=
R_1^0
R_2^1
\cdots
R_n^{n-1}.
$$

This composition rule is used extensively in the kinematic description of serial manipulators.

## Active rotations and coordinate transformations

A rotation matrix can be interpreted in two closely related ways.

### Active rotation

In an active interpretation, the coordinate frame remains fixed while the geometric vector itself is rotated:

$$
\boxed{
\vec{v}'
=
R\vec{v}
}
$$

For example, applying $R_z(\theta)$ actively rotates a vector by $\theta$ around the positive $z$ axis.

### Change of representation

In a coordinate-transformation interpretation, the geometric vector remains unchanged while its coordinate representation changes:

$$
\boxed{
\vec{v}^{\,j}
=
R_i^j
\vec{v}^{\,i}
}
$$

Both interpretations use the same matrix algebra. The distinction lies in the geometric meaning assigned to the operation.

```{important}
When interpreting a rotation matrix, always distinguish between rotating a geometric object and expressing the same object in a different reference frame.
```

## Successive rotations about fixed axes

Consider several rotations applied successively about axes belonging to a fixed reference frame.

Suppose a first rotation $R_a(\alpha)$ is followed by a rotation $R_b(\beta)$.

The resulting transformation is

$$
\vec{v}'
=
R_b(\beta)
R_a(\alpha)
\vec{v}.
$$

Therefore,

$$
\boxed{
R
=
R_b(\beta)R_a(\alpha)
}
$$

A new rotation about a fixed axis is multiplied on the **left**.

For example, three successive rotations about the fixed $x$, $y$, and $z$ axes give

$$
R
=
R_z(\gamma)
R_y(\beta)
R_x(\alpha).
$$

With column vectors, the rightmost operation acts first.

## Successive rotations about moving axes

Now consider rotations about axes attached to a reference frame that moves after every rotation.

Suppose the first rotation is about the initial $x$ axis:

$$
R_1
=
R_x(\alpha).
$$

A second rotation about the new, already-rotated $y$ axis is composed on the right:

$$
R_2
=
R_x(\alpha)R_y(\beta).
$$

A third rotation about the new $z$ axis gives

$$
R
=
R_x(\alpha)
R_y(\beta)
R_z(\gamma).
$$

Therefore, for rotations about moving axes,

$$
\boxed{
R_{\text{new}}
=
R_{\text{old}}R_{\text{local}}.
}
$$

The matrices appear in the same order as the intrinsic rotations are performed.

The two composition rules may be summarized as

$$
\boxed{
\begin{aligned}
\text{fixed axis:}\qquad
&
R_{\text{new}}
=
R_{\text{axis}}R_{\text{old}},
\\[4pt]
\text{moving axis:}\qquad
&
R_{\text{new}}
=
R_{\text{old}}R_{\text{axis}}.
\end{aligned}
}
$$

## Intrinsic and extrinsic rotation sequences

A sequence of rotations about moving axes is called an **intrinsic** sequence.

A sequence of rotations about fixed axes is called an **extrinsic** sequence.

For an intrinsic sequence $ABC$ with angles $(\phi,\theta,\psi)$,

$$
\boxed{
R
=
R_A(\phi)
R_B(\theta)
R_C(\psi)
}
$$

where the first, second, and third rotations are performed about successive moving axes.

An equivalent orientation can be obtained using an extrinsic sequence with reversed axis and angle order:

$$
\boxed{
\operatorname{Intrinsic}\ ABC(\phi,\theta,\psi)
\equiv
\operatorname{Extrinsic}\ CBA(\psi,\theta,\phi)
}
$$

For example,

$$
\operatorname{Intrinsic}\ XYZ(\phi,\theta,\psi)
\equiv
\operatorname{Extrinsic}\ ZYX(\psi,\theta,\phi).
$$

```{note}
A sequence name such as `XYZ` or `ZXZ` is ambiguous unless it is also specified whether the rotations are intrinsic or extrinsic.
```

Moro currently interprets Euler-angle sequences as **intrinsic rotations**.

## Euler angles

Three successive elementary rotations can be used to parameterize an arbitrary three-dimensional orientation.

Moro denotes the three angles of a selected sequence by

$$
(\phi,\theta,\psi),
$$

where:

- $\phi$ is the first angle of the sequence,
- $\theta$ is the second angle,
- $\psi$ is the third angle.

These symbols are associated with their **position in the sequence**, not permanently with the $x$, $y$, or $z$ axes.

For example, for an intrinsic `ZXZ` sequence,

$$
R
=
R_z(\phi)
R_x(\theta)
R_z(\psi),
$$

while for an intrinsic `YZY` sequence,

$$
R
=
R_y(\phi)
R_z(\theta)
R_y(\psi).
$$

This convention allows the same notation to be used consistently across different Euler-angle sequences.

## Proper Euler angles

Proper Euler sequences use the same axis for the first and third rotations.

There are six possible proper Euler sequences:

$$
ZXZ,\qquad
ZYZ,\qquad
XYX,\qquad
XZX,\qquad
YXY,\qquad
YZY.
$$

In general,

$$
R
=
R_a(\phi)
R_b(\theta)
R_a(\psi),
\qquad
a\neq b.
$$

Moro currently supports these six proper Euler sequences.

For example, for `ZXZ`,

$$
\boxed{
R
=
R_z(\phi)
R_x(\theta)
R_z(\psi)
}
$$

under Moro's intrinsic-rotation convention.

## Tait--Bryan angles

Tait--Bryan sequences use three different rotation axes.

The six possible sequences are

$$
XYZ,\qquad
XZY,\qquad
YXZ,\qquad
YZX,\qquad
ZXY,\qquad
ZYX.
$$

Their general form is

$$
R
=
R_a(\phi)
R_b(\theta)
R_c(\psi),
$$

with

$$
a\neq b,
\qquad
b\neq c,
\qquad
a\neq c.
$$

Tait--Bryan sequences are widely used in applications involving concepts such as roll, pitch, and yaw.

However, the terms **roll**, **pitch**, and **yaw** should not be treated as universal synonyms for $\phi$, $\theta$, and $\psi$. Their precise interpretation depends on the selected sequence and on whether the rotations are intrinsic or extrinsic.

```{note}
Tait--Bryan sequences are not yet supported by Moro's Euler-angle conversion functions. They are included here to clarify the general convention and to distinguish them from proper Euler sequences.
```

## Canonical Euler-angle ranges

Euler-angle representations are not unique. In general, several angle triples may represent the same physical orientation.

When extracting Euler angles from a rotation matrix, Moro uses a single canonical solution.

For proper Euler sequences, the canonical ranges are

$$
\boxed{
\phi,\psi\in[-\pi,\pi],
\qquad
\theta\in[0,\pi].
}
$$

For Tait--Bryan sequences, the intended canonical convention is

$$
\boxed{
\phi,\psi\in[-\pi,\pi],
\qquad
\theta\in
\left[-\frac{\pi}{2},\frac{\pi}{2}\right].
}
$$

Returning one canonical representation makes the result deterministic and avoids requiring users to handle multiple equivalent solutions for a single orientation.

```{note}
A canonical Euler-angle triple is not the unique mathematical representation of an orientation. It is one representative chosen from a family of equivalent angle triples.
```

## Euler-angle singularities

Every three-angle Euler parameterization contains singular configurations in which the first and third rotations cannot be determined independently.

For proper Euler sequences, singularities occur when

$$
\theta=0
$$

or

$$
\theta=\pi.
$$

At these configurations, only a combination of $\phi$ and $\psi$ can be determined from the orientation matrix.

Moro resolves this ambiguity by selecting the convention

$$
\boxed{
\psi=0
}
$$

and computing the remaining angle so that the returned triple still represents the correct orientation.

For Tait--Bryan sequences, singularities occur at

$$
\theta
=
\pm\frac{\pi}{2}.
$$

These configurations are commonly associated with the phenomenon known as **gimbal lock**.

Singularities do not indicate that the physical orientation is invalid. They indicate that the chosen three-angle parameterization loses one degree of representational freedom at those configurations.

## Axis--angle representation

An alternative representation of orientation consists of a unit rotation axis

$$
\hat{u}
=
\begin{bmatrix}
u_x\\
u_y\\
u_z
\end{bmatrix},
\qquad
\|\hat{u}\|=1,
$$

and a rotation angle $\theta$.

The pair

$$
(\hat{u},\theta)
$$

represents a rotation of $\theta$ radians about the axis $\hat{u}$ according to the right-hand rule.

## Skew-symmetric matrix

For a vector

$$
\vec{u}
=
\begin{bmatrix}
u_x\\
u_y\\
u_z
\end{bmatrix},
$$

the associated skew-symmetric matrix is

$$
[\vec{u}]_\times
=
\begin{bmatrix}
0 & -u_z & u_y\\
u_z & 0 & -u_x\\
-u_y & u_x & 0
\end{bmatrix}.
$$

It satisfies

$$
[\vec{u}]_\times\vec{v}
=
\vec{u}\times\vec{v}.
$$

## Rodrigues' rotation formula

Given a unit axis $\hat{u}$ and an angle $\theta$, the corresponding rotation matrix can be obtained from Rodrigues' formula:

$$
\boxed{
R
=
I
+
\sin\theta[\hat{u}]_\times
+
(1-\cos\theta)[\hat{u}]_\times^2
}
$$

This expression provides the basis for converting between axis--angle and rotation-matrix representations.

The functions `axa2rot` and `rot2axa` provide these conversions in Moro.

## Canonical axis--angle representation

Moro uses a canonical axis--angle representation with

$$
\boxed{
\theta \in [0,\pi].
}
$$

Restricting the rotation angle to this interval removes the redundancy associated with the equivalence

$$
(\hat{u},\theta)
\equiv
(-\hat{u},-\theta).
$$

For rotations satisfying

$$
0<\theta<\pi,
$$

the rotation axis is uniquely determined.

### Identity rotation

When

$$
\theta=0,
$$

the corresponding orientation is

$$
R=I.
$$

In this case, the rotation axis is geometrically undefined because every unit axis produces the same zero rotation.

To provide a deterministic representation, Moro uses

$$
\boxed{
\hat{u}
=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}
}
$$

when converting the identity rotation to axis--angle form.

Therefore,

$$
\operatorname{rot2axa}(I)
\longrightarrow
\left(
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix},
0
\right).
$$

```{note}
The choice of the $x$ axis for the identity rotation is purely conventional. It does not carry geometric information about the rotation.
```

### Rotation by pi

When

$$
\theta=\pi,
$$

the rotation axis is determined only up to sign.

The representations

$$
(\hat{u},\pi)
$$

and

$$
(-\hat{u},\pi)
$$

describe the same physical orientation.

Therefore, Moro does not assign geometric significance to the sign of the axis in this case.

```{important}
When comparing axis--angle representations for rotations of $\pi$ radians, the axes $\hat{u}$ and $-\hat{u}$ must be considered equivalent.
```

The cases near $\theta=0$ and $\theta=\pi$ are handled explicitly by Moro because the general expressions used to recover the rotation axis from a matrix become numerically ill-conditioned near these configurations.

## Summary of conventions used by Moro

The rotation conventions used throughout Moro may be summarized as follows:

| Convention | Moro |
|---|---|
| Vector representation | Column vectors |
| Matrix action | Left multiplication |
| Coordinate systems | Right-handed |
| Positive rotation | Right-hand rule |
| Relative orientation | $R_i^j$ |
| Coordinate transformation | $\vec v^{\,j}=R_i^j\vec v^{\,i}$ |
| Composition | $R_k^j=R_i^jR_k^i$ |
| Fixed-axis rotation | New rotation multiplies on the left |
| Moving-axis rotation | New rotation multiplies on the right |
| Euler interpretation | Intrinsic |
| Current Euler sequences | Proper Euler |
| Proper Euler middle-angle range | $[0,\pi]$ |
| Proper Euler first/third range | $[-\pi,\pi]$ |
| Singular Euler convention | $\psi=0$ |
| Rotation direction | Right-hand rule |
| Axis normalization | $\|\hat{u}\|=1$ |
| Angle range | $\theta\in[0,\pi]$ |
| Positive rotation | Right-hand rule |
| Identity rotation | $\theta=0$, $\hat{u}=[1,0,0]^T$ |
| Rotation by $\pi$ | $\hat{u}$ and $-\hat{u}$ are equivalent |
| Matrix construction | Rodrigues' rotation formula |

These conventions are used consistently throughout Moro's kinematics, transformations, and visualization tools.