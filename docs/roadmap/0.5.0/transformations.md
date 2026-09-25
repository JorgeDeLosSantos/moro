# Transformations scope for Moro 0.5.0

This section records the accepted preliminary scope for the `moro.transformations` audit planned for Moro 0.5.0. It is intended to be incorporated into the main `docs/roadmap/0.5.0.md` roadmap while keeping implementation-level details deferred to the detailed design phase.

## Status

Accepted for Moro 0.5.0. Detailed design complete.

## Objective

Complete and regularize the orientation and rigid-transformation utilities required by the new kinematic capabilities planned for Moro 0.5.0, while preserving the module's compact educational character.

The initial scope focuses on four related areas:

- completing three-angle rotation sequences;
- adding quaternion conversions as an additional orientation representation;
- adding rotation-vector conversions based on the logarithmic/exponential geometry of `SO(3)`;
- making rotation and homogeneous-transform validation explicit and reusable.

The feature should extend the existing `moro/transformations.py` module rather than introducing a separate orientation package.

## Euler and Tait-Bryan sequences

The existing public names

```python
eul2rot(...)
rot2eul(...)
```

will be retained.

The current implementation supports the six proper Euler sequences:

```text
xyx, xzx, yxy, yzy, zxz, zyz
```

Moro 0.5.0 should extend the same API to the six Tait-Bryan sequences:

```text
xyz, xzy, yxz, yzx, zxy, zyx
```

so that the two functions support all twelve standard three-axis rotation sequences.

A separate public function family for Tait-Bryan angles is not planned. The sequence is already an explicit part of the existing API, and keeping one conversion pair avoids unnecessary duplication.

The current convention should remain unchanged:

- active rotations;
- column vectors;
- for `seq="abc"`,

\[
R = R_a(\phi)R_b(\theta)R_c(\psi).
\]

The singularity behavior and representative-angle conventions for Tait-Bryan sequences should be documented as explicitly as they are for the proper Euler cases. Exact formulas, angle ranges, and singular representatives are deferred to detailed design and testing.

## Quaternion representation

Moro 0.5.0 should add quaternions as a compact orientation representation, limited initially to representation and conversion utilities.

The accepted public convention is scalar-first:

\[
q = (w,x,y,z),
\]

corresponding to

\[
q = w + xi + yj + zk.
\]

The preliminary conversion API should include at least:

```python
rot2quat(R)
quat2rot(q)
axa2quat(k, theta)
quat2axa(q)
```

The exact accepted input containers, output type, normalization policy, canonical sign convention, numerical tolerances, and behavior for non-unit input quaternions are deferred to detailed design.

The initial quaternion scope is intentionally limited. Quaternions will not become an alternative primary target format for `solve_pose_ik()` in 0.5.0; users may convert them to a rotation matrix and construct the canonical homogeneous-transform target explicitly.

## Rotation vectors and `SO(3)` logarithmic/exponential maps

Moro 0.5.0 should expose rotation vectors as a practical public representation of the logarithmic and exponential maps of `SO(3)`.

A rotation vector is defined as

\[
\phi = \theta k,
\]

where `k` is a unit rotation axis and `theta` is the corresponding rotation angle. Its norm therefore satisfies

\[
\|\phi\| = \theta.
\]

The accepted public API is:

```python
rot2rotvec(R)
rotvec2rot(phi)
```

rather than exposing separate `so3_log()` and `so3_exp()` functions in the initial release.

This naming is intentionally consistent with the existing conversion API:

```text
rot2eul()      <-> eul2rot()
rot2axa()      <-> axa2rot()
rot2quat()     <-> quat2rot()
rot2rotvec()   <-> rotvec2rot()
```

The implementation should nevertheless be based on the geometry of the logarithmic and exponential maps:

\[
R = \exp([\phi]_\times),
\]

and

\[
\phi = \operatorname{vex}(\log R).
\]

Robust numerical treatment near zero rotation and rotations near `pi` is required. The exact algorithms and thresholds are deferred to detailed design.

`rot2rotvec()` is expected to provide the reusable primitive used by full-pose inverse kinematics to construct the orientation residual, subject to the final left/right relative-rotation convention selected for `solve_pose_ik()`.

## `skew()` and `vex()`

The existing `skew()` helper should gain a public inverse operation:

```python
vex(S)
```

such that

\[
\operatorname{vex}([u]_\times)=u.
\]

Conceptually,

\[
[u]_\times =
\begin{bmatrix}
0 & -u_z & u_y\\
u_z & 0 & -u_x\\
-u_y & u_x & 0
\end{bmatrix}.
\]

`vex()` should return a three-component SymPy column vector and should validate that its input has the appropriate skew-symmetric structure.

The exact behavior for symbolic matrices whose skew symmetry cannot be decided conclusively is deferred to detailed design.

## Axis-angle API symmetry

The existing axis-angle conversion API should be made symmetric with the rest of the module by extending

```python
axa2rot(k, theta)
```

to

```python
axa2rot(k, theta, deg=False)
```

so that angles may be supplied in degrees when explicitly requested, matching the behavior already available in `rot2axa()` and the elementary/Euler rotation functions.

No separate axis-angle representation object is planned.

## Rotation-matrix validation

The transformations module should expose a public predicate

```python
is_rotation_matrix(R, *, tol=1e-9)
```

for determining whether a matrix belongs to `SO(3)`.

For numerical matrices, validation should check at least:

\[
R^T R \approx I
\]

and

\[
\det(R) \approx 1,
\]

in addition to the required `3 x 3` shape.

For symbolic inputs the predicate should use ternary semantics:

- `True`: membership in `SO(3)` can be established;
- `False`: non-membership can be established;
- `None`: symbolic membership cannot be determined conclusively.

This distinction is intentional because inability to prove a symbolic property is not equivalent to proving that the property is false.

For example, a symbolic matrix constructed from elementary rotation functions should normally be recognizable as a rotation matrix after appropriate symbolic simplification, while a fully generic symbolic `3 x 3` matrix may remain indeterminate.

The current `is_SO3()` utility should be reviewed during implementation for compatibility, possible delegation, and whether its public role should be superseded by the more descriptive `is_rotation_matrix()` name. An unnecessary immediate breaking change is not required by the roadmap.

## Homogeneous-transform validation

The module should similarly expose

```python
is_homogeneous_transform(T, *, tol=1e-9)
```

with the same ternary semantics for symbolic inputs.

A valid rigid homogeneous transformation has the form

\[
T =
\begin{bmatrix}
R & p\\
0 & 1
\end{bmatrix},
\]

so validation should cover at least:

- shape `4 x 4`;
- a valid `SO(3)` rotation block;
- a valid homogeneous final row;
- real numerical components where required for numerical validation.

If the rotation block is symbolically indeterminate, the complete homogeneous-transform predicate should also be able to return `None` rather than collapsing the result to `False`.

## Predicates versus operation preconditions

Public `is_*` helpers are intended for inspection and should not raise merely because a matrix is not a valid rotation or homogeneous transform.

Operations that require a valid rotation or pose should use internal validation helpers that enforce the precondition and raise clear exceptions when it is not satisfied.

Conceptually:

```text
is_rotation_matrix()
is_homogeneous_transform()
        -> inspection
        -> True | False | None

internal validators
        -> operation preconditions
        -> accept or raise a clear exception
```

A numerical operation such as full-pose IK must reject an indeterminate symbolic target unless all required symbolic parameters have first been resolved numerically.

## Validation policy for existing transformation helpers

The introduction of stronger validation helpers should not imply that every existing constructor or extractor must perform full geometric validation on every call.

Simple structural utilities such as rotation/translation extraction may continue to perform only the validation necessary for their operation. Full `SO(3)` or homogeneous-transform validation should be applied where geometric validity is semantically required, including orientation conversions and the target accepted by `solve_pose_ik()`.

Moro 0.5.0 should not silently project invalid matrices onto `SO(3)` or `SE(3)`. Within tolerance, numerical inputs may be accepted; otherwise they should be reported as invalid. Explicit projection utilities may be considered separately in a future version.

## Relationship with full-pose inverse kinematics

This transformations scope provides the orientation primitives required by the accepted full-pose IK feature.

In particular:

- `is_homogeneous_transform()` supports validation of the canonical `4 x 4` pose target;
- `rot2rotvec()` provides the intended geometric orientation-error representation;
- `skew()` / `vex()` expose the corresponding `so(3)` vector/matrix relationship;
- robust behavior near zero and `pi` rotations is shared transformation functionality rather than solver-specific mathematics.

The exact relative rotation used by the solver, for example whether the error is constructed from `R_d R^T` or `R^T R_d`, remains a full-pose IK detailed-design decision because it must match the frame convention of the geometric Jacobian.

## Initial examples and tests

The transformations work should eventually include examples covering at least:

- all six proper Euler sequences and all six Tait-Bryan sequences;
- forward/inverse Euler round trips away from singularities;
- representative singular configurations for both sequence families;
- matrix/quaternion round trips;
- axis-angle/quaternion round trips;
- matrix/rotation-vector round trips;
- interpretation of a rotation vector's direction and norm;
- `skew()` / `vex()` round trips;
- numerical and symbolic rotation-matrix validation;
- numerical and symbolic homogeneous-transform validation.

Tests should emphasize reconstruction rather than equality of non-unique orientation parameters. They should also cover numerical tolerance behavior and the important singular cases near zero and `pi` for axis-angle and rotation-vector conversions.

## Explicitly outside the initial 0.5.0 scope

The following capabilities are not part of the accepted initial transformations scope:

- quaternion SLERP;
- quaternion trajectory interpolation;
- quaternion differential equations or angular-velocity integration;
- dual quaternions;
- direct quaternion targets for full-pose IK;
- pose or orientation trajectories;
- `SE(3)` logarithmic/exponential maps as a public API;
- screw-coordinate utilities as a new general subsystem;
- adjoint transformations as part of this feature;
- automatic projection of approximate matrices onto `SO(3)` or `SE(3)`;
- a general orientation-representation class hierarchy.

These capabilities may be considered after the core orientation conversions and full-pose IK conventions are stable.

## Detailed design

The implementation-level contract for this feature is finalized in:

[`docs/design/0.5.0/transformations.md`](../../design/0.5.0/transformations.md)

That document is the normative source for detailed API semantics, validation, numerical policy, result invariants, tests, and implementation guidance. If implementation evidence requires a contract change, update the detailed design explicitly rather than changing behavior silently.
