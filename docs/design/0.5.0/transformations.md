# Detailed design: `moro.transformations` for Moro 0.5.0

This document records detailed design decisions for the `moro.transformations` work planned for Moro 0.5.0. It complements `docs/roadmap/0.5.0/transformations.md`, which remains the higher-level scope document.

The intent is to preserve the compact, functional, SymPy-first and educational character of the module while completing the orientation utilities required by the 0.5.0 roadmap.

## Status

Detailed design complete for the accepted Moro 0.5.0 transformations scope.

The Euler/Tait-Bryan, validation, axis-angle, quaternion, rotation-vector, `skew()`/`vex()`, compatibility, and test-strategy sections below reflect decisions accepted during detailed design.

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
seq = seq[::-1]
phi, psi = psi, phi
```

followed by the same intrinsic construction path.

For a nonsingular `rot2eul()` result, the reversed intrinsic sequence is solved and every returned angle triple is reversed back to the requested public convention while preserving solution order.

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

The sequence constants should be organized as:

```python
_PROPER_EULER_SEQUENCES = (
    "xyx", "xzx",
    "yxy", "yzy",
    "zxz", "zyz",
)

_TAIT_BRYAN_SEQUENCES = (
    "xyz", "xzy",
    "yxz", "yzx",
    "zxy", "zyx",
)

_EULER_SEQUENCES = (
    *_PROPER_EULER_SEQUENCES,
    *_TAIT_BRYAN_SEQUENCES,
)
```

`_normalize_euler_sequence()` should normalize case, reject non-string inputs with `TypeError`, and reject unsupported strings with `ValueError`.

#### Proper Euler configuration

The existing `_PROPER_EULER_CONFIG` structure should be retained rather than redesigned. Each sequence continues to provide:

```text
cos_index
phi
psi
singular_positive
singular_negative
```

`cos_index` identifies the matrix element equal to `cos(theta)`, while the other entries define the `atan2` pairs required for the general and singular branches.

#### Tait-Bryan configuration

A parallel `_TAIT_BRYAN_CONFIG` should be added. Its sequence entries should provide:

```text
sin_term
phi
psi
singular_positive
singular_negative
singular_positive_sign
singular_negative_sign
```

For example, the `xyz` entry has the conceptual structure:

```python
"xyz": {
    "sin_term": (1, 0, 2),
    "phi": ((-1, 1, 2), (1, 2, 2)),
    "psi": ((-1, 0, 1), (1, 0, 0)),
    "singular_positive": ((1, 1, 0), (1, 1, 1)),
    "singular_negative": ((-1, 1, 0), (1, 1, 1)),
    "singular_positive_sign": +1,
    "singular_negative_sign": -1,
}
```

The singular sign entries encode the observable relation

\[
\chi = \phi + \sigma\psi,
\]

with `sigma` equal to `+1` or `-1`. They are used when converting a singular intrinsic result into the requested extrinsic representative while preserving the public `psi=0` rule.

For intrinsic Tait-Bryan sequences, the singular relation signs are:

| `seq` | `theta=+pi/2` | `theta=-pi/2` |
| --- | ---: | ---: |
| `xyz` | `+1` | `-1` |
| `xzy` | `-1` | `+1` |
| `yxz` | `-1` | `+1` |
| `yzx` | `+1` | `-1` |
| `zxy` | `+1` | `-1` |
| `zyx` | `-1` | `+1` |

#### Shared helpers

The current proper-Euler implementation already contains several helpers that should be reused or generalized:

```text
_normalize_euler_sequence()
_validate_euler_tol()
_classify_trig_value()
_sqrt_one_minus_square()
_signed_matrix_element()
_atan2_from_config()
_negated_pair()
_convert_euler_solutions_to_degrees()
_get_singular_relation_sign()
```

`_classify_euler_cos()` should be generalized to `_classify_trig_value()` so that it only knows that a theoretical trigonometric value belongs to `[-1, 1]`. Interpretation of `+1` and `-1` remains the responsibility of the proper-Euler or Tait-Bryan solver.

The generalized classifier should continue to support:

- symbolic simplification;
- tolerance-aware clipping for floating-point values;
- rejection of excursions outside `[-1,1]` beyond tolerance;
- distinction between positive singular, negative singular, general, and symbolically undecidable cases.

`_euler_sqrt_term()` should similarly be generalized to `_sqrt_one_minus_square(value)`, including the current floating-point protection

```python
max(0.0, 1.0 - value**2)
```

so that the helper can represent either

```text
sin(theta) = sqrt(1 - cos(theta)^2)
```

or

```text
cos(theta) = sqrt(1 - sin(theta)^2).
```

The existing `_signed_matrix_element()`, `_atan2_from_config()`, and `_negated_pair()` abstractions remain suitable for both sequence families.

#### Internal inverse solvers

The two mathematical solvers should remain separate:

```python
_rot2proper_euler(R, seq, tol)
_rot2tait_bryan(R, seq, tol)
```

They should always work in radians and in the intrinsic convention.

The separation is intentional because the two families recover the intermediate angle from different quantities:

```text
Proper Euler:
    recover cos(theta)
    sin(theta) = sqrt(1 - cos(theta)^2)

Tait-Bryan:
    recover sin(theta)
    cos(theta) = sqrt(1 - sin(theta)^2)
```

Both solvers should return a lightweight internal result of the form

```python
solutions, singular_case
```

where:

```python
singular_case is None
```

in the general case, and

```python
singular_case == "positive"
singular_case == "negative"
```

for the two singular branches.

A public result dataclass is not required.

The degree conversion should not occur inside these solvers. It should happen once at the public `rot2eul()` boundary.

A small dispatcher should route intrinsic requests:

```python
def _rot2eul_intrinsic(R, seq, tol):
    if seq in _PROPER_EULER_SEQUENCES:
        return _rot2proper_euler(R, seq, tol)
    return _rot2tait_bryan(R, seq, tol)
```

#### Extrinsic singularities

For a requested extrinsic sequence `abc`, the internal solver uses the equivalent intrinsic sequence `cba`.

In the nonsingular case, each internal solution

```python
(alpha, theta, gamma)
```

is converted to the public extrinsic solution

```python
(gamma, theta, alpha)
```

while preserving first/second solution order.

In the singular case, this mechanical reversal is not sufficient because it would place the internally fixed zero in the first public angle. Instead, let the internal singular relation be

\[
\chi = \alpha + \sigma\gamma.
\]

The intrinsic solver returns the representative

\[
(\alpha_{eq},\theta_s,0),
\]

so that

\[
\alpha_{eq}=\chi.
\]

The public extrinsic convention requires the third public angle to be zero, which corresponds to setting the first internal angle to zero. Therefore

\[
\gamma_{eq}=\sigma\alpha_{eq}.
\]

The public extrinsic singular result is consequently

\[
(\sigma\alpha_{eq},\theta_s,0).
\]

For proper Euler sequences, the singular relation sign is sequence-independent:

```text
positive singularity (theta=0):   sigma = +1
negative singularity (theta=pi):  sigma = -1
```

For Tait-Bryan sequences, the sign comes from `_TAIT_BRYAN_CONFIG` according to the internal reversed sequence and singular branch.

A helper such as

```python
_get_singular_relation_sign(seq, singular_case)
```

may encapsulate this rule.

#### Public `rot2eul()` flow

The public function should conceptually follow this order:

```python
def rot2eul(R, seq="zxz", deg=False, intrinsic=True, tol=1e-9):
    seq = _normalize_euler_sequence(seq)
    _validate_euler_tol(tol)

    if not isinstance(intrinsic, bool):
        raise TypeError("intrinsic must be a bool.")

    R = Matrix(R)
    _validate_rotation_matrix(R, tol=tol)

    if intrinsic:
        solutions, singular_case = _rot2eul_intrinsic(R, seq, tol)

    else:
        internal_seq = seq[::-1]
        internal_solutions, singular_case = _rot2eul_intrinsic(
            R,
            internal_seq,
            tol,
        )

        if singular_case is None:
            solutions = [
                (psi, theta, phi)
                for phi, theta, psi in internal_solutions
            ]
        else:
            alpha_eq, theta, _ = internal_solutions[0]
            sign = _get_singular_relation_sign(
                internal_seq,
                singular_case,
            )
            solutions = [(sign * alpha_eq, theta, 0)]

    if deg:
        solutions = _convert_euler_solutions_to_degrees(solutions)

    return solutions
```

This pseudocode records the intended control flow rather than prescribing exact implementation syntax.

#### Public `eul2rot()` flow

`eul2rot()` does not require separate proper-Euler and Tait-Bryan algorithms. Once the sequence has been validated, the same elementary-rotation construction handles all twelve sequences.

Conceptually:

```python
def eul2rot(
    phi,
    theta,
    psi,
    seq="zxz",
    deg=False,
    intrinsic=True,
):
    seq = _normalize_euler_sequence(seq)

    if not isinstance(intrinsic, bool):
        raise TypeError("intrinsic must be a bool.")

    if deg:
        phi, theta, psi = deg2rad(
            Matrix([phi, theta, psi]),
            evalf=False,
        )

    if not intrinsic:
        seq = seq[::-1]
        phi, psi = psi, phi

    return (
        rot(phi, seq[0])
        * rot(theta, seq[1])
        * rot(psi, seq[2])
    )
```

#### Legacy private wrappers

The current wrappers

```text
_rot2zxz()
_rot2zyz()
_rot2xyx()
_rot2xzx()
_rot2yxy()
_rot2yzy()
```

only delegate to `_rot2proper_euler()` and should be removed in 0.5.0. Creating corresponding wrappers for all twelve sequences would add unnecessary duplication and obscure the configuration-driven design.

The intended final Euler structure is therefore:

```text
Public API
  eul2rot()
  rot2eul()

Intrinsic mathematical core
  _rot2proper_euler()
  _rot2tait_bryan()
  _rot2eul_intrinsic()

Configuration
  _PROPER_EULER_SEQUENCES
  _TAIT_BRYAN_SEQUENCES
  _EULER_SEQUENCES
  _PROPER_EULER_CONFIG
  _TAIT_BRYAN_CONFIG

Shared mechanics
  sequence normalization
  tolerance validation
  trig classification
  complementary trig term
  configured atan2 extraction
  second-solution sign handling
  singular relation sign handling
  degree conversion
```

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

### 3.1 Public inspection predicates

Moro 0.5.0 will expose:

```python
is_rotation_matrix(R, *, tol=1e-9)
is_homogeneous_transform(T, *, tol=1e-9)
```

These are inspection predicates, not operation preconditions.

For symbolic inputs they use ternary semantics:

```text
True   -> validity can be established
False  -> invalidity can be established
None   -> validity cannot be decided conclusively
```

A predicate should not raise merely because a matrix is geometrically invalid. Conversion failures for objects that cannot reasonably be interpreted as matrices may still raise `TypeError`.

### 3.2 Shared tolerance policy

A single private helper should validate tolerances across the transformations module:

```python
_validate_tol(tol)
```

The accepted tolerance must be:

- scalar;
- numeric;
- real;
- strictly positive.

Examples that should be accepted include ordinary positive floats and exact positive SymPy numbers such as `Rational` values.

Examples that should be rejected include:

```text
0
negative values
symbolic tolerances
complex values
```

Invalid types should raise `TypeError`; numeric but non-positive values should raise `ValueError`.

The helper may normalize the accepted tolerance to a Python `float` for subsequent numerical comparisons.

### 3.3 `is_rotation_matrix()` contract

A valid rotation matrix belongs to `SO(3)` and therefore satisfies

\[
R^T R = I,
\qquad
\det(R)=1.
\]

`is_rotation_matrix()` should also require shape `(3, 3)` and real components for numerical inputs.

The public behavior is:

```text
not convertible to Matrix -> TypeError
shape != (3, 3)          -> False
numerical and non-real    -> False
numerically valid         -> True
numerically invalid       -> False
symbolically valid        -> True
symbolically invalid      -> False
symbolically indeterminate-> None
```

#### Numerical path

A matrix may be classified as numerical when every element has `is_number is True`.

Within that path:

- any element with `is_real is False` makes the matrix invalid;
- if any element has `is_real is None`, the predicate should avoid assuming reality and should fall back to symbolic/indeterminate reasoning;
- otherwise values may be evaluated numerically for tolerance checks.

The orthogonality condition should use an element-wise tolerance check on

\[
E = R^T R-I.
\]

The matrix is numerically orthogonal when every element satisfies

\[
|E_{ij}|\le tol.
\]

The determinant condition should similarly require

\[
|\det(R)-1|\le tol.
\]

A separate matrix norm is not required for 0.5.0; the element-wise criterion is explicit and easy to document.

#### Symbolic path

The symbolic path should evaluate

\[
R^T R-I
\]

and

\[
\det(R)-1
\]

using measured symbolic simplification.

Each required zero condition may be classified through `expr.is_zero` after `simplify()`:

```text
True  -> condition established
False -> condition violated
None  -> condition indeterminate
```

The aggregate predicate is:

- `False` if any required condition is demonstrably false;
- `True` if all required conditions are demonstrably true;
- `None` otherwise.

The predicate should not initially rely on an aggressive chain of symbolic transformations such as repeated `trigsimp`, `factor`, or `cancel`. Additional simplification strategies may be introduced only if tests reveal important Moro-generated matrices that `simplify()` cannot recognize.

Matrices constructed from Moro elementary rotation functions should normally be recognizable as valid, while a fully generic symbolic `3 x 3` matrix may legitimately remain indeterminate.

### 3.4 `_validate_rotation_matrix()`

Operations that require a valid rotation matrix should use:

```python
_validate_rotation_matrix(R, *, tol=1e-9)
```

The helper should:

1. validate and normalize `tol`;
2. convert `R` to a SymPy matrix;
3. delegate geometric classification to `is_rotation_matrix()`;
4. return the normalized matrix when validity is established;
5. raise when the precondition is not satisfied.

Conceptually:

```text
True  -> return Matrix(R)
False -> ValueError: R is not a valid rotation matrix
None  -> ValueError: validity could not be established
```

The `False` and `None` cases should use distinguishable error messages because a demonstrably invalid matrix and a symbolically indeterminate matrix represent different situations.

This helper should be the canonical precondition path for orientation conversions such as:

```text
rot2eul()
rot2axa()
rot2quat()
rot2rotvec()
```

### 3.5 `is_homogeneous_transform()` contract

A valid rigid homogeneous transformation has the form

\[
T=
\begin{bmatrix}
R & p\\
0 & 1
\end{bmatrix},
\]

with

\[
R\in SO(3).
\]

`is_homogeneous_transform()` should verify:

- shape `(4, 4)`;
- validity of the upper-left rotation block through `is_rotation_matrix()`;
- validity of the homogeneous final row;
- real components for fully numerical inputs.

The translation vector does not require an additional geometric constraint beyond reality in the numerical case.

The final row should satisfy

\[
T_{30}=0,\quad
T_{31}=0,\quad
T_{32}=0,\quad
T_{33}=1.
\]

For numerical matrices, the same `tol` should be used for the final row. Therefore tiny floating-point deviations may be accepted consistently with the rotation-block checks.

For symbolic matrices, each final-row condition should use the same ternary zero-classification strategy as rotation validation.

The complete result should combine component statuses conservatively:

```text
if any required condition is False -> False
if all required conditions are True -> True
otherwise -> None
```

In particular, a symbolically indeterminate rotation block should be able to make the complete transform predicate return `None` rather than being collapsed to `False`.

### 3.6 `_validate_homogeneous_transform()`

Operations that require a valid rigid pose should use:

```python
_validate_homogeneous_transform(T, *, tol=1e-9)
```

Its behavior should mirror `_validate_rotation_matrix()`:

```text
True  -> return normalized Matrix(T)
False -> ValueError: T is not a valid homogeneous transformation
None  -> ValueError: validity could not be established
```

The helper should delegate geometric classification to `is_homogeneous_transform()` rather than duplicating its rules.

This validator is intended to support operations such as rigid structured inversion and the canonical pose target accepted by full-pose inverse kinematics.

### 3.7 Selective adoption in existing functions

The new validation infrastructure should not cause every helper in `transformations.py` to perform full geometric validation.

Functions should validate only what their mathematics requires.

Full `SO(3)` validation should be used by:

```text
rot2eul()
rot2axa()
rot2quat()
rot2rotvec()
```

Full homogeneous-transform validation should be used by:

```text
invhtm()
```

because its structured inverse relies on

\[
R^{-1}=R^T.
\]

`invhtm()` should therefore evolve to a tolerance-aware signature such as

```python
invhtm(T, *, tol=1e-9)
```

and should validate the input before computing `R.T` and `-R.T*p`.

The following existing functions should retain structural validation only:

```text
rot2htm()
rt2htm()
htm2rot()
htm2tra()
```

This preserves their role as simple constructors or extractors.

The following functions construct valid transformations by definition and do not require additional geometric validation:

```text
rot()
rotx()
roty()
rotz()
htmrot()
htmtra()
dh()
```

This selective policy keeps the module compact and avoids turning simple structural utilities into unnecessarily strict geometric gates.

### 3.8 Compatibility and deprecation of `is_SO3()`

The existing `moro.util.is_SO3()` should not remain as an independent second implementation.

For Moro 0.5.0:

- `is_rotation_matrix()` in `moro.transformations` becomes the canonical public API;
- `moro.util.is_SO3()` remains temporarily available for import compatibility;
- `is_SO3()` is deprecated;
- `is_SO3()` delegates to `is_rotation_matrix()` rather than preserving separate logic;
- `is_SO3()` preserves its historical boolean contract by collapsing indeterminate symbolic results to `False`.

Conceptually:

```python
def is_SO3(R, tol=1e-9):
    warnings.warn(
        "is_SO3() is deprecated; use "
        "moro.transformations.is_rotation_matrix() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from moro.transformations import is_rotation_matrix

    return is_rotation_matrix(R, tol=tol) is True
```

The import from `moro.transformations` should be local to the wrapper or otherwise arranged so that the current dependency of `transformations.py` on utilities such as `deg2rad()` and `rad2deg()` does not introduce a module-level circular import.

This strategy preserves existing code such as

```python
from moro.util import is_SO3
```

while establishing one source of truth for `SO(3)` validation.

Removal of `is_SO3()` may be considered in a later release according to Moro's compatibility policy.

### 3.9 Validation tests

At minimum, validation tests should cover:

- exact numerical rotation matrices;
- floating-point rotation matrices within tolerance;
- matrices outside tolerance;
- orthogonal matrices with determinant `-1`;
- incorrect shapes;
- non-real numerical entries;
- exact symbolic rotations generated by Moro;
- demonstrably invalid symbolic matrices;
- generic symbolic matrices that should return `None`;
- exact and approximate valid homogeneous transforms;
- invalid homogeneous final rows;
- homogeneous transforms with invalid rotation blocks;
- propagation of `None` from an indeterminate rotation block;
- distinction between predicate behavior and private-validator exceptions;
- deprecation warning and boolean compatibility of `moro.util.is_SO3()`;
- `invhtm()` rejection of matrices that are merely `4 x 4` but not valid rigid transforms.

## 4. Axis-angle refinements

### 4.1 Public API

Moro 0.5.0 will retain the existing axis-angle conversion names:

```python
rot2axa(R, deg=False, tol=1e-9)
axa2rot(k, theta, deg=False)
```

`deg=False` is added to `axa2rot()` so that degree handling is symmetric in both conversion directions and consistent with the rest of the transformations API.

`rot2axa()` returns

```python
(k, theta)
```

where `k` is a three-component SymPy column vector and `theta` is a scalar angle.

When `deg=True`, only the angular quantity is converted; the axis is unchanged.

### 4.2 Axis normalization and zero-axis policy

`axa2rot()` should continue to accept axes that are not already normalized. The input axis is converted to a three-dimensional vector and normalized internally:

\[
k \leftarrow \frac{k}{\lVert k\rVert}.
\]

This preserves the convenient interpretation that any nonzero vector parallel to the desired axis is acceptable.

A demonstrably zero axis is invalid and should raise `ValueError`.

For symbolic axes, however, an indeterminate norm should not be rejected merely because SymPy cannot prove that it is nonzero. The intended policy is:

```text
norm_sq.is_zero is True  -> reject
norm_sq.is_zero is False -> accept
norm_sq.is_zero is None  -> accept symbolically
```

This allows useful symbolic constructions without requiring Moro to prove a nonzero assumption that may be implicit in the user's model.

### 4.3 `axa2rot()` construction

`axa2rot()` should continue to use Rodrigues' rotation formula. For a normalized axis `k`,

\[
R = I + \sin\theta\,[k]_\times
    + (1-\cos\theta)[k]_\times^2.
\]

When `deg=True`, `theta` should be converted to radians before evaluating the formula.

No `tol` argument is required by `axa2rot()` because it is a constructor and does not classify a noisy matrix.

### 4.4 `rot2axa()` validation and principal angle

`rot2axa()` should delegate rotation-matrix precondition checking to the shared validator:

```python
R = _validate_rotation_matrix(R, tol=tol)
```

The current duplicated use of `is_SO3()` and `_is_SO3_numeric_tol()` should therefore be removed once the new validation infrastructure is introduced.

The returned angle should use the principal axis-angle convention

\[
\theta\in[0,\pi].
\]

The basic angle quantity is obtained from

\[
\cos\theta = \frac{\operatorname{tr}(R)-1}{2}.
\]

Numerical values near the theoretical interval limits should be classified using `tol` and clipped when the deviation is within tolerance rather than failing because of small floating-point errors.

### 4.5 Identity rotation

At

\[
\theta=0,
\]

the rotation axis is indeterminate.

Moro will preserve the existing representative convention

\[
k =
\begin{bmatrix}
1\\0\\0
\end{bmatrix}.
\]

Thus the identity rotation has one deterministic public representative, but documentation should make clear that this axis is conventional rather than information recovered from the matrix.

### 4.6 General case

For

\[
0<\theta<\pi,
\]

the axis can be recovered from the antisymmetric part of the rotation matrix:

\[
k=
\frac{1}{2\sin\theta}
\begin{bmatrix}
R_{32}-R_{23}\\
R_{13}-R_{31}\\
R_{21}-R_{12}
\end{bmatrix}.
\]

No additional canonicalization of the axis sign is required in the general case once the principal nonnegative angle has been selected.

### 4.7 Rotation by `pi`

At

\[
\theta=\pi,
\]

the antisymmetric recovery formula is singular because `sin(theta)=0`.

Moro should retain the robust special-case strategy based on

\[
A=\frac{R+I}{2}=kk^T.
\]

The implementation should reconstruct a unit axis by selecting a numerically suitable dominant diagonal component and recovering the remaining components from the corresponding row or column relationships.

At exactly `theta=pi`, both

\[
(k,\pi)
\]

and

\[
(-k,\pi)
\]

represent the same orientation. Moro will not impose an additional sign-canonicalization rule such as requiring the first nonzero axis component to be positive.

Tests and downstream code should therefore rely on rotation reconstruction rather than exact axis-sign equality at this singular representation.

### 4.8 Symbolic handling

For symbolic inputs whose `SO(3)` membership can be established, `rot2axa()` should preserve symbolic expressions whenever possible.

If the identity or `pi` case can be established exactly, the corresponding special branch should be used.

If symbolic reasoning cannot prove that the angle is one of those singular values, the general symbolic expression may be retained rather than introducing aggressive case splitting.

The implementation should avoid excessive symbolic manipulation solely to classify an angle branch.

### 4.9 Shared internal helpers for axis-angle and rotation vectors

The difficult matrix decomposition logic should be factored into a small set of private helpers that can be reused by `rot2axa()` and the future `rot2rotvec()` implementation.

The accepted helper set is:

```python
_rotation_angle_from_matrix(R, tol)
_axis_from_rotation_matrix(R, theta, angle_case, tol)
_axis_at_pi(R, tol)
```

#### `_rotation_angle_from_matrix()`

This helper is responsible for determining how much rotation is represented by `R`.

It should compute

\[
\cos\theta = \frac{\operatorname{tr}(R)-1}{2}
\]

and return conceptually

```python
(theta, angle_case)
```

where `angle_case` is one of:

```text
"identity"
"pi"
"general"
```

Its responsibilities include:

- numerical clipping of `cos(theta)` to `[-1, 1]` when deviations are within `tol`;
- tolerance-aware detection of values near `+1` and `-1`;
- exact/symbolic recognition of identity and `pi` cases when possible;
- construction of the principal general angle in `[0, pi]`.

#### `_axis_at_pi()`

This helper should encapsulate only the numerically delicate `theta=pi` recovery using

\[
(R+I)/2=kk^T.
\]

It returns one valid unit axis and does not promise a canonical sign.

Keeping this logic separate prevents the special `pi` handling from being duplicated across axis-angle and rotation-vector conversions.

#### `_axis_from_rotation_matrix()`

This helper acts as the small axis-recovery dispatcher.

Conceptually:

```python
if angle_case == "identity":
    return Matrix([1, 0, 0])

if angle_case == "pi":
    return _axis_at_pi(R, tol)

return general_axis_formula(R, theta)
```

It answers the complementary question to `_rotation_angle_from_matrix()`: the first helper determines how much the orientation rotates, while this helper determines around which axis it rotates.

No public decomposition object or internal dataclass is needed for 0.5.0.

A lower-level helper such as `_skew_vector_from_rotation_matrix()` should not be introduced unless implementation work reveals meaningful reuse beyond a single formula.

### 4.10 Reuse by rotation vectors

The future `rot2rotvec()` implementation should reuse the same matrix decomposition rather than independently reproducing identity, `pi`, clipping, and general-axis logic.

Conceptually, `rot2axa()` becomes:

```python
R = _validate_rotation_matrix(R, tol=tol)
theta, angle_case = _rotation_angle_from_matrix(R, tol)
k = _axis_from_rotation_matrix(R, theta, angle_case, tol)

if deg:
    theta = rad2deg(theta)

return k, theta
```

while `rot2rotvec()` can use:

```python
R = _validate_rotation_matrix(R, tol=tol)
theta, angle_case = _rotation_angle_from_matrix(R, tol)

if angle_case == "identity":
    return Matrix.zeros(3, 1)

k = _axis_from_rotation_matrix(R, theta, angle_case, tol)
return theta * k
```

The exact rotation-vector public API will be defined in its own design section; this subsection records only the intended internal reuse.

### 4.11 Compatibility with Moro 0.4.0

The following behavior is preserved:

- public names `rot2axa()` and `axa2rot()`;
- `(k, theta)` return structure from `rot2axa()`;
- automatic normalization of a nonzero axis in `axa2rot()`;
- principal angle in `[0, pi]`;
- conventional x-axis representative for the identity rotation;
- special handling of rotations by `pi`.

The following behavior is added or regularized:

- `deg=False` support in `axa2rot()`;
- shared geometric validation through `_validate_rotation_matrix()`;
- shared tolerance policy through `_validate_tol()`;
- reusable internal angle/axis decomposition for rotation vectors.

No axis-sign canonicalization is added for the `pi` case.

### 4.12 Axis-angle tests

At minimum, tests should cover:

- reconstruction for representative general rotations;
- exact identity rotation;
- exact rotations by `pi` about coordinate and non-coordinate axes;
- reconstruction when the returned `pi` axis has either equivalent sign;
- angles close to zero and close to `pi` under numerical tolerance;
- radian and degree modes in both conversion directions;
- acceptance and normalization of non-unit input axes;
- rejection of a demonstrably zero axis;
- acceptance of symbolically indeterminate but potentially nonzero axes;
- rejection of invalid rotation matrices by `rot2axa()`;
- exact symbolic rotations whose `SO(3)` membership can be established;
- principal output angle constrained to `[0, pi]`;
- consistency of shared helper behavior with future `rot2rotvec()` reconstruction tests.

## 5. Quaternions

### 5.1 Scope and public API

Moro 0.5.0 will add a compact quaternion conversion API focused on orientation representation rather than general quaternion algebra:

```python
rot2quat(R, tol=1e-9)
quat2rot(q, *, tol=1e-9)
axa2quat(k, theta, deg=False)
quat2axa(q, deg=False, *, tol=1e-9)
```

No Moro-specific quaternion class will be introduced in 0.5.0.

Direct Euler/quaternion helpers such as `eul2quat()` and `quat2eul()` are intentionally omitted. Users can compose the existing matrix conversions:

```text
Euler -> rotation matrix -> quaternion
quaternion -> rotation matrix -> Euler
```

SLERP, quaternion interpolation, quaternion trajectories, quaternion integration, and a general quaternion algebra API remain out of scope for 0.5.0.

### 5.2 Public representation and ordering

A quaternion is represented publicly as a four-component SymPy column vector using scalar-first ordering:

\[
q=
\begin{bmatrix}
w\\x\\y\\z
\end{bmatrix}.
\]

The scalar-first convention is chosen because it maps directly to the axis-angle relationship

\[
q=
\begin{bmatrix}
\cos(\theta/2)\\
k\sin(\theta/2)
\end{bmatrix}.
\]

Unit quaternions represent orientation, but conversion functions may accept non-unit nonzero inputs and normalize them internally.

### 5.3 Double coverage and canonical sign

Unit quaternions double-cover `SO(3)`:

\[
q\equiv -q.
\]

`rot2quat()` should return a canonical representative whenever the sign is decidable by imposing

\[
w\ge0.
\]

This convention corresponds to the principal orientation angle

\[
\theta\in[0,\pi].
\]

At `w=0`, corresponding to `theta=pi`, no second sign rule will be introduced for the vector part. Both

\[
[0,k]^T
\]

and

\[
[0,-k]^T
\]

remain acceptable representatives.

For symbolic outputs, canonicalization should be applied only when the sign of `w` can be established without introducing artificial `Piecewise` expressions:

```text
w provably negative      -> q = -q
w provably nonnegative   -> keep q
w sign indeterminate     -> keep symbolic expression
```

### 5.4 Quaternion shape and normalization

A private helper should centralize quaternion input handling:

```python
_normalize_quaternion(q, tol)
```

The helper should:

1. convert the input to a four-component SymPy column vector;
2. validate `tol` using the shared tolerance policy;
3. compute

   \[
   \|q\|^2=q^Tq;
   \]

4. reject zero or numerically near-zero quaternions;
5. return

   \[
   q/\|q\|.
   \]

For a numerical quaternion, a norm satisfying

\[
\|q\|\le tol
\]

should be rejected because normalization would strongly amplify numerical noise.

For symbolic inputs, the policy mirrors the axis normalization rules:

```text
norm_sq.is_zero is True  -> reject
norm_sq.is_zero is False -> normalize
norm_sq.is_zero is None  -> normalize symbolically
```

No public `is_unit_quaternion()` predicate is required for 0.5.0.

### 5.5 General vector conversion helper

Rather than introducing an independent `_as_4d_vector()` alongside the existing `_as_3d_vector()`, implementation work may generalize the current helper to something like

```python
_as_vector(v, size, name="vector")
```

with `_as_3d_vector()` retained as a thin convenience wrapper if that improves readability.

Quaternion normalization can then use conceptually:

```python
q = _as_vector(q, 4, name="quaternion")
```

The goal is to avoid duplicating shape-conversion logic merely because quaternions have four components.

### 5.6 `quat2rot()`

`quat2rot()` should normalize the quaternion first and then construct the rotation matrix directly.

For

\[
q=[w,x,y,z]^T,
\]

the matrix is

\[
R=
\begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}.
\]

Because the input is normalized internally, no subsequent `SO(3)` validation of the constructed matrix is required.

The function should satisfy explicitly

\[
\operatorname{quat2rot}(q)=\operatorname{quat2rot}(-q).
\]

### 5.7 `rot2quat()` numerical algorithm

`rot2quat()` should begin with the shared rotation precondition:

```python
R = _validate_rotation_matrix(R, tol=tol)
```

For numerical matrices, the implementation should not rely exclusively on the trace formula

\[
w=\frac12\sqrt{1+\operatorname{tr}(R)},
\]

because the subsequent divisions become poorly conditioned when `w` is close to zero.

Instead, Moro should use a dominant-component algorithm based on the quantities

\[
\begin{aligned}
s_w &= 1+R_{00}+R_{11}+R_{22},\\
s_x &= 1+R_{00}-R_{11}-R_{22},\\
s_y &= 1-R_{00}+R_{11}-R_{22},\\
s_z &= 1-R_{00}-R_{11}+R_{22},
\end{aligned}
\]

which correspond theoretically to

\[
4w^2,\quad4x^2,\quad4y^2,\quad4z^2.
\]

The numerically largest candidate should be recovered first, and the remaining components should then be obtained from the appropriate off-diagonal matrix combinations.

For example, when the scalar component dominates,

\[
w=\frac12\sqrt{s_w},
\]

followed by

\[
x=\frac{R_{21}-R_{12}}{4w},\qquad
y=\frac{R_{02}-R_{20}}{4w},\qquad
z=\frac{R_{10}-R_{01}}{4w}.
\]

Equivalent branch formulas should be used when `x`, `y`, or `z` is the dominant component.

The resulting quaternion should be normalized once more to absorb small numerical drift, then canonicalized according to `w >= 0`.

### 5.8 `rot2quat()` symbolic path

Selecting the numerically largest component is not generally meaningful for symbolic matrices.

For symbolic rotation matrices whose `SO(3)` membership can be established, `rot2quat()` should therefore use the axis-angle decomposition as a symbolic fallback:

\[
(k,\theta)=\operatorname{rot2axa}(R),
\]

followed by

\[
q=
\begin{bmatrix}
\cos(\theta/2)\\
k\sin(\theta/2)
\end{bmatrix}.
\]

This avoids introducing large symbolic `Piecewise` expressions solely to emulate the dominant-component numerical algorithm.

The result should be normalized if useful for simplification/consistency and sign-canonicalized only when the sign of `w` can be established.

### 5.9 Quaternion sign helper

A small private helper may encapsulate the public sign convention:

```python
_canonicalize_quaternion_sign(q)
```

Its intended behavior is:

```text
w < 0 or provably negative  -> -q
w > 0 or provably positive  -> q
w == 0                      -> q
w symbolically indeterminate-> q
```

The helper must not attempt a second vector-part sign convention when `w=0`.

### 5.10 `axa2quat()`

`axa2quat()` should directly implement

\[
q=
\begin{bmatrix}
\cos(\theta/2)\\
k\sin(\theta/2)
\end{bmatrix}
\]

after normalizing the axis using the same policy as `axa2rot()`.

A non-unit nonzero axis is accepted and normalized automatically. A demonstrably zero axis is rejected, while a symbolically indeterminate potentially nonzero axis is accepted.

When `deg=True`, `theta` is converted to radians before evaluating the half-angle expressions.

No `tol` argument is required because `axa2quat()` is a constructor rather than a noisy-data classifier.

### 5.11 `quat2axa()`

`quat2axa()` should normalize the quaternion and apply the canonical sign rule before recovering the principal axis-angle representation.

Write

\[
q=
\begin{bmatrix}
w\\v
\end{bmatrix},
\qquad
v=
\begin{bmatrix}x\\y\\z\end{bmatrix},
\]

with

\[
s=\|v\|.
\]

The angle should be recovered robustly as

\[
\theta=2\operatorname{atan2}(s,w),
\]

rather than relying only on `2*acos(w)`.

After canonicalization with `w >= 0`, the principal result satisfies

\[
\theta\in[0,\pi].
\]

For the general case `s>0`, the axis is

\[
k=\frac{v}{s}.
\]

At the identity quaternion

\[
q=[1,0,0,0]^T,
\]

Moro should return the same conventional axis used by `rot2axa()`:

\[
k=[1,0,0]^T,\qquad\theta=0.
\]

At `theta=pi`, `w=0` and the vector part itself provides the unit axis. Because no second sign convention is imposed, either equivalent axis sign is acceptable.

`quat2axa()` should convert directly rather than routing through `quat2rot()` and `rot2axa()`.

### 5.12 No direct Euler/quaternion API in 0.5.0

Moro 0.5.0 will not add

```text
eul2quat()
quat2eul()
```

The existing conversion graph is sufficient:

```text
Euler --eul2rot--> R --rot2quat--> quaternion
quaternion --quat2rot--> R --rot2eul--> Euler
```

This keeps Euler sequence, intrinsic/extrinsic, singularity, and degree conventions centralized in the existing Euler API instead of duplicating them inside quaternion-specific functions.

### 5.13 Quaternion test philosophy

Quaternion tests should primarily verify represented orientation rather than raw component equality because

\[
q\equiv -q.
\]

For two quaternions, equivalence means

\[
q_1\sim q_2
\iff
q_1=q_2\ \text{or}\ q_1=-q_2,
\]

with tolerance-aware comparison for numerical tests.

Exact component equality should be required only where the public contract deliberately selects a unique representative, such as the identity quaternion or a numerical `rot2quat()` result with positive scalar part away from `w=0`.

### 5.14 Quaternion normalization and validation tests

Tests should cover:

- already unit quaternions;
- non-unit quaternions that normalize to the same orientation;
- list, tuple, and matrix forms of four-component input;
- incorrect vector shape;
- exact zero quaternion rejection;
- numerical near-zero quaternion rejection using `tol`;
- symbolic demonstrably zero quaternion rejection;
- symbolic indeterminate potentially nonzero quaternion acceptance.

### 5.15 Matrix/quaternion conversion tests

For `quat2rot()`, tests should include:

- identity quaternion to identity matrix;
- known quarter-turn/half-turn examples around coordinate axes;
- general-axis examples;
- explicit double-coverage check `quat2rot(q) == quat2rot(-q)`;
- invariance under nonzero scalar multiplication before internal normalization.

For `rot2quat()`, tests should emphasize

```text
R -> q -> R
```

reconstruction, numerical canonicalization `w >= 0`, and robustness near `theta=pi`.

Exact rotations by `pi` should be tested about coordinate axes and at least one general axis such as `[1,1,1]`. At `pi`, tests must not require one specific sign for the vector part.

### 5.16 Axis-angle/quaternion conversion tests

`axa2quat()` tests should cover:

- analytical coordinate-axis examples;
- non-unit input-axis normalization;
- zero-axis rejection;
- radian/degree consistency;
- symbolic half-angle expressions.

`quat2axa()` tests should cover:

- identity representative `[1,0,0,0]^T`;
- a general orientation;
- `theta=pi` with either equivalent axis sign;
- principal output range `[0, pi]`;
- radian/degree consistency;
- reconstruction of the represented orientation.

### 5.17 Round-trip and representative tests

Parameterized tests should cover the main conversion cycles:

```text
R -> q -> R
q -> R -> q
(k, theta) -> q -> (k, theta)
q -> (k, theta) -> q
```

The comparison metric should depend on the representation:

- matrices are compared as orientations/matrices;
- quaternions are compared modulo sign unless canonicalization guarantees a representative;
- axis-angle cycles are compared through reconstructed rotation matrices rather than literal axis equality at singular cases.

Representative numerical angles should include values close to zero, ordinary interior angles, values close to `pi`, and `pi` exactly.

Representative axes should include all coordinate axes and general directions such as `[1,1,1]` and `[1,-2,3]`.

### 5.18 Symbolic quaternion tests

Because Moro is SymPy-first, symbolic tests are required rather than optional.

For example, with real symbolic `theta`,

```python
q = axa2quat([0, 0, 1], theta)
```

should produce an expression equivalent to

\[
[\cos(\theta/2),0,0,\sin(\theta/2)]^T.
\]

`quat2rot(q)` should reconstruct a symbolic z-axis rotation.

For a symbolic rotation matrix generated by Moro, such as `rotx(theta)`, `rot2quat()` need not be tested against one rigid symbolic component form. Instead, the preferred assertion is reconstruction:

```text
rot2quat(R) -> quat2rot(q) -> R
```

with symbolic simplification applied to the matrix difference.

The guiding principle for this block is therefore:

```text
test the represented orientation, not an arbitrary non-unique representation
```

except where the public API explicitly promises a canonical representative.

## 6. Rotation vectors and `SO(3)` logarithmic/exponential maps

### 6.1 Public API and representation

Moro 0.5.0 will expose rotation vectors through:

```python
rot2rotvec(R, tol=1e-9)
rotvec2rot(phi)
```

A rotation vector is

\[
\phi=\theta k,
\]

where `k` is a unit rotation axis and `theta` is the rotation angle. Therefore

\[
\|\phi\|=\theta.
\]

The public representation is a three-component SymPy column vector. Rotation vectors are always interpreted in radians; no `deg` option is introduced because the vector simultaneously encodes axis and angular magnitude.

The implementation is based on the geometry of the exponential and logarithmic maps of `SO(3)`, but public `so3_exp()` and `so3_log()` names are intentionally not added in 0.5.0.

### 6.2 Principal logarithm convention

`rot2rotvec()` returns the principal rotation vector with

\[
\|\phi\|\in[0,\pi].
\]

For the identity rotation,

\[
R=I\quad\Rightarrow\quad\phi=0.
\]

At exactly `theta=pi`, the rotation vectors

\[
\pi k
\]

and

\[
-\pi k
\]

represent the same orientation. Moro will not impose an additional sign-canonicalization rule for this case, matching the axis-angle convention.

### 6.3 `rotvec2rot()` exponential map

`rotvec2rot()` should implement the exponential map directly rather than routing through the public axis-angle API.

Let

\[
\theta=\|\phi\|,
\qquad
\Phi=[\phi]_\times.
\]

Then

\[
R
=
I
+
\frac{\sin\theta}{\theta}\Phi
+
\frac{1-\cos\theta}{\theta^2}\Phi^2.
\]

This is Rodrigues' formula written directly in rotation-vector coordinates and corresponds to

\[
R=\exp([\phi]_\times).
\]

The implementation should use the existing public `skew()` helper to construct `Phi`.

`rotvec2rot()` accepts any finite rotation-vector magnitude. It must not restrict the input to the principal interval `[0, pi]`; periodicity follows naturally from the trigonometric terms.

### 6.4 Exact zero and small-angle numerical behavior

At

\[
\phi=0,
\]

`rotvec2rot()` should return `Matrix.eye(3)` exactly.

For small numerical angles, direct evaluation of

\[
A(\theta)=\frac{\sin\theta}{\theta},
\qquad
B(\theta)=\frac{1-\cos\theta}{\theta^2}
\]

may lose precision. A small-angle numerical branch should therefore use low-order series such as

\[
A(\theta)
\approx
1-\frac{\theta^2}{6}+\frac{\theta^4}{120},
\]

and

\[
B(\theta)
\approx
\frac12-\frac{\theta^2}{24}+\frac{\theta^4}{720}.
\]

The switch between the series and direct formulas is an internal numerical-stability detail. It should use a small private threshold rather than introducing a public `tol` argument to `rotvec2rot()`.

### 6.5 Symbolic `rotvec2rot()` behavior

For a symbolic rotation vector, define

\[
\theta^2=\phi^T\phi.
\]

The policy is:

```text
theta_sq.is_zero is True  -> return identity
theta_sq.is_zero is False -> use closed-form exponential formula
theta_sq.is_zero is None  -> use closed-form exponential formula
```

No automatic `Piecewise` expression should be introduced solely to cover a symbolically indeterminate zero case.

The symbolic path should preserve the explicit Rodrigues/exponential form rather than relying on a generic symbolic matrix exponential such as `exp(skew(phi))`, which may remain unevaluated and would be less useful pedagogically.

### 6.6 `rot2rotvec()` implementation architecture

`rot2rotvec()` should use the same shared matrix decomposition already designed for axis-angle rather than calling `rot2axa()` as a public function or duplicating an independent logarithm implementation.

Conceptually:

```python
R = _validate_rotation_matrix(R, tol=tol)
theta, angle_case = _rotation_angle_from_matrix(R, tol)

if angle_case == "identity":
    return Matrix.zeros(3, 1)

k = _axis_from_rotation_matrix(R, theta, angle_case, tol)
return theta * k
```

This ensures that axis-angle and rotation-vector conversions share exactly the same behavior for:

- numerical clipping;
- identity classification;
- rotations by `pi`;
- symbolic branch handling;
- principal-angle selection.

### 6.7 Near-identity and near-`pi` behavior

For numerical matrices classified as identity within `tol`, `rot2rotvec()` returns the zero vector.

For matrices classified as rotations by `pi` within `tol`, it reuses `_axis_at_pi()` through `_axis_from_rotation_matrix()` and returns

\[
\phi=\pi k.
\]

The general formula

\[
\phi=
\frac{\theta}{2\sin\theta}
\operatorname{vex}(R-R^T)
\]

should not be used as the sole implementation because it becomes ill-conditioned near `theta=pi`.

### 6.8 Symbolic `rot2rotvec()` behavior

For symbolic matrices whose `SO(3)` membership can be established, exact identity and exact `pi` cases should use the corresponding shared branches when they can be proven.

Otherwise, Moro may retain the general symbolic axis/angle expression and return `theta*k` without introducing large case splits or `Piecewise` expressions.

For a symbolic matrix such as `rotx(theta)` with an unconstrained real `theta`, the API should not promise literal output `[theta, 0, 0]^T`, because the principal logarithm depends on the angular branch. Reconstruction is the stronger invariant.

### 6.9 Relationship with `skew()`, `vex()`, and the Lie algebra

The mathematical relationship should be documented explicitly:

\[
[\phi]_\times=\log(R),
\qquad
\phi=\operatorname{vex}(\log R).
\]

However, `rot2rotvec()` is not required to compute a generic matrix logarithm internally. The shared angle/axis decomposition is preferred for numerical robustness and code reuse.

`rotvec2rot()` should use `skew(phi)` directly. `vex()` remains an independent public `so(3)` vector/matrix utility and may be used internally where natural, but it is not an architectural requirement for `rot2rotvec()`.

### 6.10 Non-uniqueness and periodicity

The exponential map is not injective. For a unit axis `k`, rotation vectors whose magnitudes differ by integer multiples of `2*pi` may represent the same orientation.

Therefore:

```text
rotvec2rot(phi)
    accepts non-principal magnitudes

rot2rotvec(R)
    returns the principal representative
    with norm in [0, pi]
```

For example,

\[
\phi=\frac{3\pi}{2}k
\]

and

\[
\phi_p=-\frac{\pi}{2}k
\]

represent the same rotation, while `rot2rotvec(rotvec2rot(phi))` should return a principal equivalent rather than the original non-principal vector.

### 6.11 Rotation-vector tests

Tests should cover `rotvec2rot()` for:

- zero vector;
- coordinate-axis rotations;
- general-axis rotations;
- very small numerical magnitudes;
- values close to `pi`;
- exact `pi`;
- magnitudes greater than `pi`;
- periodicity under equivalent `2*pi` changes;
- symbolic coordinate-axis vectors.

Tests should cover `rot2rotvec()` for:

- reconstruction `R -> phi -> R`;
- principal norm in `[0, pi]`;
- identity;
- near-zero rotations;
- near-`pi` rotations;
- exact `pi` around coordinate and non-coordinate axes;
- invalid rotation-matrix rejection;
- symbolic Moro-generated rotations.

For the round trip

```text
phi -> R -> phi_principal
```

literal vector equality is expected only when the original vector already lies unambiguously in the principal branch. For non-principal vectors, tests should compare reconstructed rotations.

### 6.12 Cross-consistency with axis-angle

For any valid nonzero axis `k` and angle `theta`, tests should verify

\[
\operatorname{rotvec2rot}(\theta k)
=
\operatorname{axa2rot}(k,\theta).
\]

Similarly, for a valid rotation matrix outside ambiguous cases,

\[
\operatorname{rot2rotvec}(R)
=
\theta k
\]

should agree with `rot2axa(R)` up to the accepted sign ambiguity at `theta=pi`.

## 7. `skew()` and `vex()`

### 7.1 Public API

Moro 0.5.0 retains the existing

```python
skew(u)
```

and adds

```python
vex(S, *, tol=1e-9)
```

for the standard vector/matrix correspondence of `so(3)`.

For

\[
u=
\begin{bmatrix}u_x\\u_y\\u_z\end{bmatrix},
\]

`skew(u)` returns

\[
[u]_\times=
\begin{bmatrix}
0 & -u_z & u_y\\
u_z & 0 & -u_x\\
-u_y & u_x & 0
\end{bmatrix}.
\]

Both functions return SymPy matrices.

### 7.2 `skew()` contract

`skew()` accepts any three-component vector convertible to the shared vector representation and returns a `3 x 3` skew-symmetric matrix.

No tolerance or geometric validation is required because `skew()` is a direct linear constructor.

The generalized helper discussed earlier may be reused:

```python
u = _as_vector(u, 3, name="vector")
```

### 7.3 `vex()` formula

For a valid skew-symmetric matrix `S`,

\[
\operatorname{vex}(S)
=
\frac12
\begin{bmatrix}
S_{32}-S_{23}\\
S_{13}-S_{31}\\
S_{21}-S_{12}
\end{bmatrix}.
\]

The difference-based formula is preferred to extracting only three entries because it treats both triangular halves symmetrically and behaves sensibly for accepted numerical matrices containing tiny skew-symmetry errors.

### 7.4 `vex()` validation

`vex()` requires a `3 x 3` skew-symmetric input satisfying

\[
S^T=-S.
\]

For numerical matrices, the condition should be checked element-wise through

\[
|S+S^T|\le tol.
\]

A matrix outside tolerance should raise `ValueError`.

For symbolic matrices, Moro should simplify the entries of

\[
S+S^T
\]

and classify them with `.is_zero`:

```text
all True                -> accept
any False               -> ValueError
otherwise indeterminate -> ValueError
```

The symbolic indeterminate case is rejected because skew symmetry is an operation precondition for `vex()`, not merely an inspection question.

### 7.5 No silent projection and no public skew predicate

Although the extraction formula is equivalent to using the skew part

\[
\frac12(S-S^T),
\]

`vex()` must not silently project an arbitrary matrix onto `so(3)`. The input must first satisfy skew symmetry within tolerance.

No public `is_skew_symmetric()` predicate is planned for 0.5.0. A private validation helper should only be introduced if implementation reveals meaningful reuse beyond `vex()`.

### 7.6 `skew()`/`vex()` tests

Tests should verify the round trips

\[
\operatorname{vex}(\operatorname{skew}(u))=u
\]

and

\[
\operatorname{skew}(\operatorname{vex}(S))=S
\]

for valid exact inputs.

Additional tests should cover:

- numerical matrices within skew-symmetry tolerance;
- matrices outside tolerance;
- incorrect shapes;
- exact symbolic skew matrices;
- demonstrably non-skew symbolic matrices;
- symbolically indeterminate matrices;
- compatibility of `skew(phi)` with `rotvec2rot()`.

## 8. Compatibility and migration

### 8.1 General policy

The transformations work in Moro 0.5.0 is primarily additive and should avoid unnecessary breaking changes.

Existing conventions, public names, and return structures are preserved wherever possible. Stricter validation is introduced only where the mathematics of the operation requires a valid rotation or rigid transform.

### 8.2 Additive public capabilities

The following are additive changes:

- Tait-Bryan support in `eul2rot()` and `rot2eul()`;
- explicit `intrinsic=True/False` handling;
- `deg=False` in `axa2rot()`;
- quaternion conversion functions;
- rotation-vector conversion functions;
- `vex()`;
- `is_rotation_matrix()`;
- `is_homogeneous_transform()`.

Existing ordinary calls remain valid because new optional parameters are added without changing the meaning of prior positional arguments.

### 8.3 Stricter orientation preconditions

The inverse orientation conversions

```text
rot2eul()
rot2axa()
rot2quat()
rot2rotvec()
```

will require established membership in `SO(3)` through the common validator.

Therefore a matrix that merely has shape `3 x 3` but is not a valid rotation matrix may be rejected in 0.5.0 even if older code previously attempted to process it.

This is an intentional correctness improvement.

For symbolic matrices:

```text
validity established   -> accept
invalidity established -> ValueError
validity indeterminate -> ValueError with a distinct message
```

### 8.4 Stricter `invhtm()` precondition

`invhtm()` should evolve to

```python
invhtm(T, *, tol=1e-9)
```

and require a valid rigid homogeneous transformation before using the structured inverse

\[
T^{-1}=
\begin{bmatrix}
R^T & -R^Tp\\
0 & 1
\end{bmatrix}.
\]

Matrices that are merely `4 x 4` but not members of `SE(3)` are therefore rejected.

This is the other notable intentional behavior tightening in 0.5.0.

### 8.5 Structural helpers remain lightweight

The following functions retain structural rather than full geometric validation:

```text
rot2htm()
rt2htm()
htm2rot()
htm2tra()
```

Constructors such as

```text
rot()
rotx()
roty()
rotz()
htmrot()
htmtra()
dh()
```

continue to construct valid objects by definition and do not require redundant validation.

### 8.6 `is_SO3()` deprecation

`moro.util.is_SO3()` remains import-compatible in 0.5.0 but is deprecated in favor of

```python
from moro.transformations import is_rotation_matrix
```

The compatibility wrapper should:

- emit `DeprecationWarning`;
- delegate to `is_rotation_matrix()`;
- preserve the historical strict-boolean contract;
- collapse a symbolic `None` result to `False`.

Removal of `is_SO3()` is deferred to a later release according to Moro's compatibility policy.

No other existing public transformation function requires deprecation for 0.5.0.

### 8.7 Existing return structures and conventions

Existing return structures remain unchanged:

```text
rot2eul() -> list of angle tuples
rot2axa() -> (k, theta)
skew()    -> SymPy Matrix
```

New vector-like representations use column matrices:

```text
rot2quat()    -> Matrix(4, 1)
rot2rotvec()  -> Matrix(3, 1)
vex()         -> Matrix(3, 1)
```

Moro continues to use:

- active rotations;
- column vectors;
- radians by default;
- `seq="zxz"` as the Euler default;
- the existing intrinsic three-angle multiplication convention.

### 8.8 Migration summary

The expected migration surface is:

| Area | Moro 0.4.x | Moro 0.5.0 |
| --- | --- | --- |
| Euler sequences | 6 proper Euler | 12 sequences |
| intrinsic/extrinsic | intrinsic convention only | explicit `intrinsic` option |
| `rot2eul()` validation | limited | requires `SO(3)` |
| axis-angle | existing | preserved + `deg` symmetry |
| quaternions | unavailable | conversion API added |
| rotation vectors | unavailable | conversion API added |
| `skew()` | available | preserved |
| `vex()` | unavailable | added |
| `is_SO3()` | current utility | deprecated compatibility wrapper |
| `is_rotation_matrix()` | unavailable | canonical predicate |
| homogeneous validation | unavailable | public predicate added |
| `invhtm()` | mainly structural precondition | requires valid rigid transform |

The release should document the two intentional behavior tightenings clearly: inverse orientation functions require `SO(3)`, and `invhtm()` requires a valid rigid homogeneous transform.

## 9. Overall test strategy

### 9.1 Testing principles

The 0.5.0 transformations tests should verify both individual conversion families and consistency across equivalent orientation representations.

The primary invariant is:

```text
equivalent representations must reconstruct the same orientation
```

Tests should avoid brittle literal equality when a representation is mathematically non-unique.

### 9.2 Public-contract tests

Tests should verify:

- accepted input forms;
- public signatures and optional-argument semantics;
- SymPy return shapes/types;
- expected `TypeError` versus `ValueError` behavior;
- tolerance validation;
- compatibility of existing positional calls;
- deprecation behavior for `is_SO3()`.

Representative return contracts include:

```text
rot2eul()     -> list[tuple]
rot2axa()     -> (Matrix(3, 1), scalar)
rot2quat()    -> Matrix(4, 1)
rot2rotvec()  -> Matrix(3, 1)
vex()         -> Matrix(3, 1)
```

### 9.3 Representation-specific tests

Each representation retains the detailed test requirements defined in its own section:

- all Euler and Tait-Bryan sequences;
- axis-angle identity/general/`pi` behavior;
- quaternion normalization and double coverage;
- rotation-vector principal-log behavior and exponential periodicity;
- `skew()`/`vex()` round trips;
- numerical and symbolic rotation/transform validation.

### 9.4 Shared singular and near-singular cases

The suite should deliberately exercise numerical values near important singular structures, including scales such as

```text
1e-12
1e-9
1e-6
```

where appropriate.

Important regions include:

\[
\theta\approx0,
\qquad
\theta\approx\pi,
\]

and for Tait-Bryan sequences,

\[
\theta\approx\pm\pi/2.
\]

Tests should focus on finite output, correct reconstruction, stable branch handling, and absence of spurious division-by-zero or `nan` behavior rather than requiring arbitrary literal parameter values.

### 9.5 Shared orientation fixtures

A compact reusable orientation set should be used across conversion families. Useful reference cases include:

```text
identity
Rx(pi/6)
Ry(pi/4)
Rz(pi/2)
general three-angle rotation
general arbitrary-axis rotation
rotation near zero
rotation near pi
exact pi about a non-coordinate axis
```

Euler-specific singular fixtures should be added separately for each relevant sequence family.

Using shared orientation fixtures helps expose convention mismatches between independently implemented conversion paths.

### 9.6 Cross-representation consistency

Starting from a valid rotation matrix `R`, tests should verify reconstruction through every supported orientation representation:

```text
R -> Euler -> R
R -> axis-angle -> R
R -> quaternion -> R
R -> rotation vector -> R
```

Cross-representation compositions should also be exercised where useful, for example:

```text
axis-angle -> quaternion -> R
axis-angle -> rotation vector -> R
quaternion -> axis-angle -> R
rotation vector -> axis-angle -> R
```

Direct public conversion functions are not required for every conceptual arrow; tests may compose the accepted public API.

### 9.7 Representation-aware equality

Assertions should respect mathematical non-uniqueness:

```text
Euler:
    reconstruction preferred over literal angle equality

axis-angle:
    reconstruction preferred over axis-sign equality at pi

quaternion:
    q and -q treated as equivalent unless the public contract fixes a sign

rotation vector:
    reconstruction preferred outside the principal branch and at pi
```

Literal comparisons remain appropriate where the public API intentionally defines a unique representative, such as the identity quaternion or zero rotation vector.

### 9.8 Numerical test helpers

The test suite may centralize small comparison helpers such as:

```python
assert_matrix_close(A, B, tol=...)
assert_quaternion_equivalent(q1, q2, tol=...)
```

and, if useful,

```python
assert_rotation_close(R1, R2, tol=...)
```

These helpers should keep representation-specific equivalence logic out of individual tests.

### 9.9 Symbolic testing strategy

Symbolic tests are a first-class requirement because Moro is SymPy-first.

Representative symbolic objects should be generated with Moro constructors such as:

```text
rotx(theta)
roty(theta)
rotz(theta)
eul2rot(...)
axa2rot(...)
quat2rot(...)
rotvec2rot(...)
```

Inverse operations should be tested by reconstruction whenever exact returned parameter forms depend on symbolic branch assumptions.

A preferred assertion pattern is conceptually:

```python
simplify(R_reconstructed - R) == zeros(3)
```

or equivalent element-wise zero checking.

Generic symbolic matrices should also verify ternary validation behavior:

```text
is_rotation_matrix(generic_R) -> None
```

while operations requiring established membership in `SO(3)` reject that indeterminate input.

### 9.10 Backward-compatibility regression tests

Existing valid 0.4.x behavior should receive explicit regression coverage for at least:

```text
eul2rot()
rot2eul()
rot2axa()
axa2rot()
skew()
invhtm()
rot2htm()
rt2htm()
htm2rot()
htm2tra()
```

Historical valid inputs should preserve their geometric results. Tests should separately record the intentional validation tightenings where formerly accepted invalid geometric inputs are now rejected.

`is_SO3()` requires a compatibility test that confirms both `DeprecationWarning` and a strict boolean return value.

### 9.11 Test tooling scope

`pytest` parametrization is sufficient for the accepted 0.5.0 scope. Property-based testing with an additional dependency such as Hypothesis is not required for this release.

Property-based orientation tests may be considered later if the conversion surface grows substantially.

### 9.12 Acceptance criteria

The transformations implementation should be considered ready for Moro 0.5.0 when:

- all supported representations reconstruct orientations consistently;
- all twelve Euler/Tait-Bryan sequences pass intrinsic/extrinsic tests;
- numerical behavior is stable around zero, `pi`, and Tait-Bryan singularities;
- symbolic Moro-generated rotations remain usable under the documented assumptions;
- invalid and indeterminate geometric preconditions are handled consistently;
- existing valid 0.4.x transformation workflows retain their expected geometric behavior;
- the only intentional compatibility changes are documented and regression-tested.
