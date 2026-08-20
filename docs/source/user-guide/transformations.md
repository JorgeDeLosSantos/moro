# Transformations

`moro` provides a collection of utilities for working with rotation matrices, homogeneous transformations, Euler angles, axis-angle representations, and Denavit-Hartenberg transformations.

These functions are useful both when working directly with geometric transformations and when building or inspecting robot models.

This section focuses on practical usage. For the mathematical background behind rotation matrices and homogeneous transformations, see the **Theory** section.

## Elementary rotations

Rotation matrices about the coordinate axes can be generated with:

```python
from moro import rotx, roty, rotz
```

For example, a rotation of $\pi/2$ radians about the $z$-axis is:

```python
from sympy import pi

Rz = rotz(pi / 2)
Rz
```

Similarly:

```python
Rx = rotx(pi / 4)
Ry = roty(pi / 3)
```

By default, angular arguments are interpreted in radians.

Angles can also be provided in degrees by setting `deg=True`:

```python
R = rotz(90, deg=True)
```

Symbolic angles are supported as well:

```python
from sympy import symbols

theta = symbols("theta")

R = roty(theta)
```

The resulting object is a SymPy matrix, so it can be simplified, multiplied, differentiated, or evaluated numerically using the usual SymPy tools.

## General rotation helper

The `rot()` function provides a common interface for rotations about any principal axis:

```python
from moro.transformations import rot
```

Its general form is:

```python
rot(theta, axis="z", deg=False)
```

For example:

```python
Rx = rot(pi / 4, axis="x")
Ry = rot(pi / 4, axis="y")
Rz = rot(pi / 4, axis="z")
```

The axis identifier is case-insensitive:

```python
R = rot(90, axis="Z", deg=True)
```

Using `rot()` can be convenient when the rotation axis is selected programmatically.

## Euler angles

`moro` can convert between proper Euler angles and rotation matrices.

The two main functions are:

```python
from moro import eul2rot, rot2eul
```

### From Euler angles to a rotation matrix

A rotation matrix can be built from three Euler angles with:

```python
R = eul2rot(phi, theta, psi, seq="zxz")
```

The supported proper Euler sequences are:

```text
xyx
xzx
yxy
yzy
zxz
zyz
```

For example:

```python
from sympy import pi

R = eul2rot(
    pi / 4,
    pi / 3,
    pi / 6,
    seq="zxz",
)
```

The sequence is case-insensitive:

```python
R = eul2rot(30, 45, 60, seq="ZYZ", deg=True)
```

By default, angles are interpreted in radians. Use `deg=True` when specifying degrees.

`moro` uses column vectors and active rotations. For a proper Euler sequence `aba`, the convention is:

$$
R = R_a(\phi) R_b(\theta) R_a(\psi).
$$

Tait-Bryan sequences such as `xyz` or `zyx` are not currently supported by `eul2rot()` and `rot2eul()`.

### From a rotation matrix to Euler angles

The inverse operation is:

```python
solutions = rot2eul(R, seq="zxz")
```

In the general case, Euler-angle representations are not unique. Therefore, `rot2eul()` returns two equivalent solutions:

```python
[
    (phi1, theta1, psi1),
    (phi2, theta2, psi2),
]
```

Both solutions reconstruct the same rotation matrix using the selected sequence.

For example:

```python
solutions = rot2eul(R, seq="zxz")

for solution in solutions:
    print(solution)
```

Angles can also be returned in degrees:

```python
solutions = rot2eul(R, seq="zxz", deg=True)
```

Near Euler singularities, the first and third angles cannot be determined independently. In those cases, `rot2eul()` returns a single representative solution and sets the third angle to zero.

A numerical tolerance can be controlled with the `tol` argument:

```python
solutions = rot2eul(R, seq="zxz", tol=1e-8)
```

For a more detailed discussion of Euler-angle non-uniqueness and singularities, see the corresponding section in **Theory**.

## Axis-angle representation

A rotation can also be represented using:

* a rotation axis $\mathbf{k}$;
* a rotation angle $\theta$.

### From axis-angle to a rotation matrix

Use `axa2rot()`:

```python
from moro import axa2rot

R = axa2rot([0, 0, 1], pi / 2)
```

The axis may be provided as:

* a list;
* a tuple;
* a SymPy column matrix;
* a SymPy row matrix.

For example:

```python
from sympy import Matrix

k = Matrix([1, 1, 0])

R = axa2rot(k, pi / 4)
```

The axis is normalized internally, so it does not need to have unit length.

The zero vector is not a valid rotation axis and will raise an error.

### From a rotation matrix to axis-angle

Use:

```python
from moro import rot2axa

k, theta = rot2axa(R)
```

For example:

```python
k, theta = rot2axa(rotz(pi / 3))
```

The returned axis is normalized.

The angle is returned in radians by default:

```python
k, theta = rot2axa(R)
```

or in degrees with:

```python
k, theta = rot2axa(R, deg=True)
```

For numerical matrices, `rot2axa()` uses a tolerance when validating the rotation and handling values close to special cases:

```python
k, theta = rot2axa(R, tol=1e-8)
```

For the identity rotation, the rotation axis is not uniquely defined. `moro` uses the $x$-axis as a representative choice and returns an angle of zero.

Rotations close to $\pi$ are handled separately because the usual general formula becomes numerically sensitive in that region.

## Skew-symmetric matrices

The `skew()` function constructs the skew-symmetric matrix associated with a three-dimensional vector:

```python
from moro.transformations import skew
```

For a vector:

$$
\mathbf{u}
=
\begin{bmatrix}
u_x \\
u_y \\
u_z
\end{bmatrix},
$$

the corresponding matrix is:

$$
[\mathbf{u}]_\times
=
\begin{bmatrix}
0 & -u_z & u_y \\
u_z & 0 & -u_x \\
-u_y & u_x & 0
\end{bmatrix}.
$$

For example:

```python
S = skew([1, 2, 3])
```

Symbolic vectors can also be used:

```python
from sympy import symbols

ux, uy, uz = symbols("ux uy uz")

S = skew([ux, uy, uz])
```

This representation is especially useful when working with cross products and Rodrigues' rotation formula.

## Homogeneous translations and rotations

`moro` provides helpers for creating pure translations and pure rotations in homogeneous coordinates.

### Pure translation

Use:

```python
from moro import htmtra
```

For example:

```python
T = htmtra(1, 2, 3)
```

is equivalent to a translation vector:

$$
\mathbf{p}
=
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}.
$$

Keyword arguments can also be used:

```python
T = htmtra(x=1, z=2)
```

Symbolic translations are supported:

```python
x, y, z = symbols("x y z")

T = htmtra(x, y, z)
```

### Pure rotation

Use:

```python
from moro import htmrot
```

For example:

```python
T = htmrot(pi / 2, axis="z")
```

This produces a $4\times4$ homogeneous transformation with zero translation.

As with the elementary rotation functions, degrees can be used explicitly:

```python
T = htmrot(90, axis="z", deg=True)
```

## Building homogeneous transformations

A homogeneous transformation can also be created from an existing rotation matrix.

### From a rotation matrix

Use:

```python
from moro import rot2htm

R = rotz(pi / 4)

T = rot2htm(R)
```

The resulting transformation has zero translation.

### From rotation and translation

Use:

```python
from moro import rt2htm
```

For example:

```python
R = rotz(pi / 4)
p = [1, 2, 3]

T = rt2htm(R, p)
```

The translation vector may be specified as:

* a list;
* a tuple;
* a $3\times1$ SymPy matrix;
* a $1\times3$ SymPy matrix.

The vector is converted internally to a column vector.

This is often the most convenient way to construct a general homogeneous transformation when the orientation and position are already known separately.

## Extracting rotation and translation

The rotation and translation components of a homogeneous transformation can be extracted with:

```python
from moro import htm2rot, htm2tra
```

For example:

```python
R = htm2rot(T)
p = htm2tra(T)
```

`htm2rot()` returns the upper-left $3\times3$ rotation block.

`htm2tra()` returns the $3\times1$ translation vector.

These functions are useful when only one part of a previously computed transformation is required.

## Inverting homogeneous transformations

The inverse of a rigid-body homogeneous transformation can be computed with:

```python
from moro import invhtm

T_inv = invhtm(T)
```

For:

$$
T =
\begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix},
$$

the inverse is computed using the rigid-body structure:

$$
T^{-1}
=
\begin{bmatrix}
R^T & -R^Tp \\
0 & 1
\end{bmatrix}.
$$

This avoids applying a general-purpose matrix inversion algorithm.

For example:

```python
R = rotz(pi / 4)
T = rt2htm(R, [1, 2, 0])

T_inv = invhtm(T)
```

The product:

```python
T * T_inv
```

should simplify to the $4\times4$ identity matrix.

## Denavit-Hartenberg transformations

The `dh()` function constructs the classical Denavit-Hartenberg homogeneous transformation associated with one row of DH parameters:

```python
from moro import dh
```

Its signature is:

```python
dh(a, alpha, d, theta)
```

For example:

```python
T1 = dh(
    a=1,
    alpha=0,
    d=0,
    theta=pi / 4,
)
```

Symbolic parameters can also be used:

```python
a, alpha, d, theta = symbols("a alpha d theta")

T = dh(a, alpha, d, theta)
```

This is the same transformation convention used internally by the `Robot` class when constructing a serial manipulator from DH rows.

For a detailed explanation of the convention itself, see **Theory → Denavit-Hartenberg Convention**.

## A worked example

Consider a rigid-body pose composed of:

* a rotation of $45^\circ$ about the $z$-axis;
* a translation of $2$ units along $x$;
* a translation of $1$ unit along $y$.

First construct the rotation:

```python
R = rotz(45, deg=True)
```

Then combine it with the translation:

```python
T = rt2htm(
    R,
    [2, 1, 0],
)
```

The resulting homogeneous transformation contains both the orientation and position.

We can recover its components:

```python
R_recovered = htm2rot(T)
p_recovered = htm2tra(T)
```

and compute the inverse transformation:

```python
T_inv = invhtm(T)
```

A point represented in homogeneous coordinates can then be transformed using standard matrix multiplication.

For example:

```python
from sympy import Matrix

p_local = Matrix([1, 0, 0, 1])

p_global = T * p_local
```

Applying the inverse transformation recovers the original coordinates:

```python
T_inv * p_global
```

This pattern appears repeatedly throughout robot kinematics, where transformations are used to express positions and orientations between different reference frames.

## Notes and conventions

When using transformation utilities in `moro`, keep the following conventions in mind:

* angles are interpreted in radians unless `deg=True` is explicitly supported and provided;
* rotation matrices operate on column vectors;
* rotation functions represent active rotations;
* Euler-angle functions currently support proper Euler sequences only;
* Euler-angle representations are generally not unique;
* homogeneous transformations use the standard rigid-body form

$$
T =
\begin{bmatrix}
R & p \\
0 & 1
\end{bmatrix};
$$

* vector inputs accepted by functions such as `rt2htm()`, `skew()`, and `axa2rot()` are normalized internally to column-vector form when appropriate;
* functions such as `rot2htm()`, `rt2htm()`, `htm2rot()`, and `invhtm()` validate input dimensions but do not perform a complete mathematical membership test for $SO(3)$ or $SE(3)$.

The mathematical interpretation of these conventions is developed in the **Theory** section.

## See also

* **Theory → Rotations** — mathematical background for rotation matrices and orientation representations.
* **Theory → Homogeneous Transformations** — rigid-body transformations and frame changes.
* **Theory → Denavit-Hartenberg Convention** — derivation and interpretation of DH transformations.
* **Robot Modeling** — define serial manipulators from DH parameters.
* **Forward Kinematics** — combine transformations throughout a robot chain.
* **API Reference → Transformations** — complete signatures, parameters, and return values for transformation utilities.
