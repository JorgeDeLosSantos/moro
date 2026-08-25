# Full-pose inverse kinematics

## Status

Accepted as a candidate feature for Moro 0.5.0.

## Objective

Extend numerical IK from position-only solving to complete end-effector pose targets while preserving the established `moro/inverse_kinematics.py` philosophy for validation, initialization, joint limits, stagnation handling, and symbolic parameters.

## Canonical target

The canonical public target is a `4 x 4` homogeneous transformation

\[
T_d=\begin{bmatrix}R_d&p_d\\0&1\end{bmatrix}.
\]

Euler angles, quaternions, and axis-angle are not primary target formats in the initial API; users may convert them explicitly through transformation utilities.

## Preliminary API

```python
solve_pose(
    robot,
    target,
    q0=None,
    *,
    method="lm",
    position_weight=1.0,
    orientation_weight=1.0,
    position_tol=None,
    orientation_tol=None,
    joint_limits=None,
    parameters=None,
    ...,
)
```

`solve_position()` remains part of the public API.

## Pose error

Position error:

\[
e_p=p_d-p(q).
\]

Orientation error should use a geometric rotation-vector/log-map formulation rather than Euler-angle subtraction. Conceptually:

\[
e_R=\operatorname{Log}(R_dR(q)^T)^\vee.
\]

The exact left/right relative-rotation convention is deferred until detailed design so that it matches the geometric-Jacobian frame convention and numerical update rule.

## Weighting and convergence

Use separate scalar weights for position and orientation:

\[
e_w=\begin{bmatrix}w_pe_p\\w_Re_R\end{bmatrix},
\qquad
J_w=\begin{bmatrix}w_pJ_v\\w_RJ_\omega\end{bmatrix}.
\]

No full weighting-matrix API or automatic characteristic-length scaling is planned initially.

Convergence should be checked separately:

\[
\|e_p\|\le\varepsilon_p,
\qquad
\|e_R\|\le\varepsilon_R.
\]

Exact defaults and compatibility with the current `tol` argument are deferred.

## Methods

Levenberg-Marquardt is in scope. The existing Newton/Jacobian method may also be supported where mathematically compatible. CCD remains position-only.

A robot must not be rejected solely because it has fewer than six joints; a lower-DOF manipulator may reach particular poses even if it cannot generically realize arbitrary 6D tasks.

## `PoseIKSolution`

Use a dedicated result type rather than extending `IKSolution` with irrelevant optional fields. It should conceptually expose:

- `q`;
- achieved pose or achieved position/orientation;
- position and orientation errors;
- their norms;
- iteration count;
- method/status information;
- `success`;
- `message`.

Exact fields are deferred.

## Target validation

The target must be validated as a numerical homogeneous transformation: shape, homogeneous bottom row, orthogonal rotation block, and determinant approximately `+1`. Reusable transformation validators should be used. Silent projection onto `SO(3)` is not assumed.

## Relationship with transformations

Full-pose IK depends on robust rotation-vector conversion/log-map behavior near zero and near `pi`, together with homogeneous-transform validation. These mathematics belong in reusable transformation utilities, not as opaque solver-only formulas.

## Relationship with differential kinematics

Use the geometric Jacobian

\[
J=\begin{bmatrix}J_v\\J_\omega\end{bmatrix},
\]

with explicit tests confirming compatibility between orientation-error and Jacobian frame conventions.

## Relationship with trajectory generation

Initial scope is a single pose target only. `solve_pose_trajectory()` and pose trajectory generation are deferred.

## Examples and tests

Emphasize forward-kinematics round trips, position-only versus pose IK, nontrivial orientations, weighting, lower-DOF reachable targets, joint limits, symbolic parameters, target validation, and orientation errors near zero and `pi`.

## Explicitly outside 0.5.0

- Euler/quaternion/axis-angle primary targets;
- CCD full-pose IK;
- arbitrary partial pose subsets;
- analytical full-pose IK;
- null-space objectives;
- collision-aware IK;
- pose-trajectory IK;
- orientation interpolation / SLERP;
- closed-chain IK.

## Deferred detailed-design decisions

Exact SO(3) error convention, numerical algorithms near zero/`pi`, tolerance defaults, weight validation, compatibility with existing solver keywords, exact result fields, supported method names, shared helpers, caching/compilation, and projection policy remain deferred.
