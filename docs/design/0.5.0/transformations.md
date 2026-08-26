# Detailed design: `moro.transformations` for Moro 0.5.0

This document records detailed design decisions for the `moro.transformations` work planned for Moro 0.5.0. It complements `docs/roadmap/0.5.0/transformations.md`, which remains the higher-level scope document.

The intent is to preserve the compact, functional, SymPy-first and educational character of the module while completing the orientation utilities required by the 0.5.0 roadmap.

## Status

Design in progress.

The Euler/Tait-Bryan section below reflects decisions already accepted during detailed design. Other sections will be completed incrementally.

## 1. Design principles

The transformations API should remain functional and representation-oriented rather than introducing a class hierarchy for orientations or poses.

Public conversion functions should use explicit, descriptive names and return SymPy objects consistent with the rest of Moro.

Geometric inspection and operation preconditions are separate concerns:

- public `is_*` predicates inspect and may use ternary semantics for symbolic inputs;
- internal validators enforce operation preconditions and either accept the input or raise a clear exception.

The module should not silently project invalid numerical matrices onto `SO(3)` or `SE(3)`.

## 2. Euler and Tait-Bryan angles

### 2.1 Public API

Moro 0.5.0 will retain the existing public names and extend their capabilities:

```python
eul2rot(phi, theta, psi, seq="zxz", deg=False, intrinsic=True)

rot2eul(
    R,
    seq="zxz",
    deg=False,
    intrinsic=True,
    tol=1e-9,
)
```

No separate public function family will be introduced for Tait-Bryan angles.

### 2.2 Supported sequences

The API will support all twelve standard three-axis sequences.

Proper Euler sequences:

```text
xyx, xzx, yxy, yzy, zxz, zyz
```

Tait-Bryan sequences:

```text
xyz, xzy, yxz, yzx, zxy, zyx
```

Sequence matching is case-insensitive. Letter case will not encode intrinsic versus extrinsic rotations; that distinction is expressed explicitly by `intrinsic`.

### 2.3 Rotation convention

Moro uses active rotations and column vectors.

For an intrinsic sequence `seq="abc"`, the public convention is

\[
R = R_a(\phi)R_b(\theta)R_c(\psi).
\]

The current behavior is therefore preserved when `intrinsic=True`.

Extrinsic rotations are related to the intrinsic convention through

\[
\operatorname{Ext}_{abc}(\phi,\theta,\psi)
=
\operatorname{Int}_{cba}(\psi,\theta,\phi).
\]

This equivalence should be used internally to avoid duplicating the complete mathematical implementation.

### 2.4 Intrinsic and extrinsic rotations

`intrinsic=True` is the default in order to preserve the current Moro convention and compatibility with existing code.

`intrinsic=False` interprets the named sequence with respect to fixed axes.

For `eul2rot()`, an extrinsic request may therefore be implemented conceptually as

```python
_eul2rot_intrinsic(psi, theta, phi, seq[::-1])
```

For a nonsingular `rot2eul()` result, the reversed intrinsic sequence may be solved and every returned angle triple reversed back to the requested public convention while preserving solution order.

The singular case requires additional handling because the public singular representative must always set the third angle to zero.

### 2.5 Number and ordering of inverse solutions

`rot2eul()` will continue to expose the non-uniqueness of three-angle orientation representations.

In the nonsingular case it returns two equivalent solutions:

```python
[
    (phi1, theta1, psi1),
    (phi2, theta2, psi2),
]
```

The first solution is the principal solution. The second corresponds to the other branch of the same parameterization.

This behavior is retained intentionally for educational value and backward compatibility.

In a singular configuration, `rot2eul()` returns one representative solution only.

### 2.6 Principal ranges

For proper Euler sequences, the principal intermediate angle satisfies

\[
\theta_1 \in [0,\pi].
\]

The second branch uses the equivalent negative intermediate angle, conceptually

\[
\theta_2=-\theta_1,
\]

with the corresponding equivalent shifts in the outer angles.

For Tait-Bryan sequences, the principal intermediate angle satisfies

\[
\theta_1\in[-\pi/2,\pi/2].
\]

The second branch lies outside the principal interval and may be computed robustly as

\[
\theta_2=\operatorname{atan2}
\left(s_\theta,-\sqrt{1-s_\theta^2}\right),
\]

whereas the principal branch uses

\[
\theta_1=\operatorname{atan2}
\left(s_\theta,+\sqrt{1-s_\theta^2}\right).
\]

Equivalently,

\[
\theta_2=
\begin{cases}
\pi-\theta_1, & \theta_1\ge0,\\
-\pi-\theta_1, & \theta_1<0.
\end{cases}
\]

No additional post-normalization will be imposed on `phi` and `psi`; their natural `atan2` representatives will be retained.

### 2.7 Singularities and representative convention

Proper Euler sequences are singular at

\[
\theta=0,\pi.
\]

Tait-Bryan sequences are singular at

\[
\theta=\pm\pi/2.
\]

At a singularity, the two outer angles cannot be recovered independently. Moro will use a single representative with the public third angle set to zero:

```python
[(phi_eq, theta_singular, 0)]
```

This rule applies uniformly to:

- proper Euler sequences;
- Tait-Bryan sequences;
- intrinsic rotations;
- extrinsic rotations.

`phi_eq` represents the observable combination of the two outer angles and should not be documented as necessarily equal to the original `phi` used to create the rotation.

For extrinsic rotations, the singular representative must be constructed so that the public third angle remains zero. A mechanical reversal of an intrinsic singular triple is insufficient because it could move the zero to the first public angle.

### 2.8 Tait-Bryan inverse structure

For the intrinsic convention

\[
R=R_a(\phi)R_b(\theta)R_c(\psi),
\]

every Tait-Bryan sequence can use the same inverse algorithmic pattern. A sequence-specific matrix element provides `sin(theta)`, and sequence-specific `atan2` pairs provide the two outer angles.

The general-case configuration is:

| `seq` | `sin(theta)` | `phi1` | `psi1` |
| --- | --- | --- | --- |
| `xyz` | `R[0,2]` | `atan2(-R[1,2], R[2,2])` | `atan2(-R[0,1], R[0,0])` |
| `xzy` | `-R[0,1]` | `atan2(R[2,1], R[1,1])` | `atan2(R[0,2], R[0,0])` |
| `yxz` | `-R[1,2]` | `atan2(R[0,2], R[2,2])` | `atan2(R[1,0], R[1,1])` |
| `yzx` | `R[1,0]` | `atan2(-R[2,0], R[0,0])` | `atan2(-R[1,2], R[1,1])` |
| `zxy` | `R[2,1]` | `atan2(-R[0,1], R[1,1])` | `atan2(-R[2,0], R[2,2])` |
| `zyx` | `-R[2,0]` | `atan2(R[1,0], R[0,0])` | `atan2(R[2,1], R[2,2])` |

The second solution is obtained by changing the sign of both arguments in the `atan2` expressions for the two outer angles and by using the negative square-root branch for `cos(theta)`.

### 2.9 Tait-Bryan singular representatives

For the intrinsic convention and `psi=0`, one valid set of sequence-specific singular representatives is:

Positive singularity, `theta=+pi/2`:

| `seq` | `phi_eq` |
| --- | --- |
| `xyz` | `atan2(R[1,0], R[1,1])` |
| `xzy` | `atan2(R[2,1], R[2,2])` |
| `yxz` | `atan2(R[0,1], R[0,0])` |
| `yzx` | `atan2(R[0,2], R[2,2])` |
| `zxy` | `atan2(R[1,0], R[0,0])` |
| `zyx` | `atan2(R[1,2], R[1,1])` |

Negative singularity, `theta=-pi/2`:

| `seq` | `phi_eq` |
| --- | --- |
| `xyz` | `atan2(-R[1,0], R[1,1])` |
| `xzy` | `atan2(-R[2,1], R[2,2])` |
| `yxz` | `atan2(-R[0,1], R[0,0])` |
| `yzx` | `atan2(R[0,2], R[2,2])` |
| `zxy` | `atan2(R[1,0], R[0,0])` |
| `zyx` | `atan2(-R[1,2], R[1,1])` |

These formulas should be verified by reconstruction tests for every sequence and both singular signs before implementation is considered complete.

### 2.10 Internal implementation strategy

The implementation should preserve the distinction between proper Euler and Tait-Bryan mathematics while sharing lower-level helpers.

Recommended internal organization:

```text
_PROPER_EULER_SEQUENCES
_TAIT_BRYAN_SEQUENCES
_EULER_SEQUENCES

_PROPER_EULER_CONFIG
_TAIT_BRYAN_CONFIG

_normalize_euler_sequence()
_validate_euler_tol()
_classify_trig_value()
_sqrt_one_minus_square()
_signed_matrix_element()
_atan2_from_config()
_negated_pair()
_convert_euler_solutions_to_degrees()

_rot2proper_euler()
_rot2tait_bryan()
```

`_rot2proper_euler()` and `_rot2tait_bryan()` should remain separate because the two families recover the intermediate angle from different trigonometric quantities:

```text
Proper Euler:
    recover cos(theta)
    sin(theta) = sqrt(1 - cos(theta)^2)

Tait-Bryan:
    recover sin(theta)
    cos(theta) = sqrt(1 - sin(theta)^2)
```

This separation keeps the mathematics explicit while still avoiding duplicated tolerance, clipping, `atan2`, and sign-handling code.

Internal inverse helpers should also communicate whether the returned result is singular, for example conceptually as

```python
solutions, singular = _rot2tait_bryan(...)
```

or an equivalent lightweight internal representation. A public result dataclass is not required.

### 2.11 Validation contract

`eul2rot()` is a constructor and does not require geometric validation of an input matrix.

`rot2eul()` requires a valid rotation matrix.

The input must:

- be convertible to a SymPy matrix;
- have shape `(3, 3)`;
- satisfy the `SO(3)` precondition.

For numerical matrices, the future shared rotation-matrix validator should use `tol` to check orthogonality and determinant and to tolerate small floating-point deviations.

For symbolic matrices:

- established `SO(3)` membership is accepted;
- established non-membership is rejected;
- indeterminate membership is rejected by `rot2eul()` because this is an operation precondition rather than an inspection predicate.

No automatic projection onto `SO(3)` will be performed.

`tol` must be a positive real value.

`intrinsic` must be a boolean value. Coercion from arbitrary truthy/falsy objects is not part of the public contract.

### 2.12 Numerical classification near singularities

Sequence-specific values theoretically constrained to `[-1, 1]` should be treated with tolerance-aware clipping.

For example, a floating-point value such as

\[
1+10^{-12}
\]

may be accepted and clipped to `1` when `tol=1e-9`, while a deviation larger than tolerance should be rejected.

The current `_classify_euler_cos()` behavior should therefore be generalized into a shared trigonometric classifier rather than duplicated for `sin(theta)`.

### 2.13 Compatibility with Moro 0.4.0

The following behavior is preserved:

- public names `eul2rot()` and `rot2eul()`;
- default `seq="zxz"`;
- default radian input/output;
- all six existing proper Euler sequences;
- two equivalent solutions in nonsingular inverse conversion;
- one representative solution at a singularity;
- third angle equal to zero in the singular representative.

The following capabilities are added:

- six Tait-Bryan sequences;
- explicit extrinsic rotations through `intrinsic=False`.

The following behavior is intentionally stricter:

- `rot2eul()` will validate membership in `SO(3)` instead of validating only matrix shape.

This is considered a correctness improvement rather than a redesign of the existing public API.

### 2.14 Test requirements

Tests should emphasize reconstruction rather than direct equality of angle triples.

At minimum, tests should cover:

- all six proper Euler sequences;
- all six Tait-Bryan sequences;
- intrinsic and extrinsic conventions;
- radian and degree modes;
- reconstruction from both nonsingular returned solutions;
- principal intermediate-angle ranges;
- positive and negative Tait-Bryan singularities;
- `theta=0` and `theta=pi` proper-Euler singularities;
- the invariant that the third public angle is zero in every singular result;
- preservation of first/second solution ordering for extrinsic conversions;
- tolerance-aware numerical classification near singularities;
- rejection of invalid `SO(3)` matrices;
- symbolic matrices whose membership can be established;
- rejection of symbolically indeterminate rotation matrices when used as an operation precondition.

## 3. Rotation-matrix and homogeneous-transform validation

Detailed design pending.

## 4. Axis-angle refinements

Detailed design pending.

## 5. Quaternions

Detailed design pending.

## 6. Rotation vectors and `SO(3)` logarithmic/exponential maps

Detailed design pending.

## 7. `skew()` and `vex()`

Detailed design pending.

## 8. Compatibility and migration

Detailed design pending completion of the remaining transformation features.

## 9. Overall test strategy

Detailed design pending completion of the remaining transformation features.
