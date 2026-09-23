# Detailed design: full-pose inverse kinematics for Moro 0.5.0

This document records the detailed design for extending Moro's numerical inverse kinematics from position-only solving to complete end-effector pose targets in Moro 0.5.0.

The feature builds on the existing moro.inverse_kinematics implementation and on the 0.5.0 designs for transformations, differential kinematics, and singularity/manipulability analysis.

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

Full-pose IK is an additive extension of the existing numerical IK module.

The existing public position-IK API remains conceptually stable:

~~~python
solve_position_ik(...)
solve_position_trajectory(...)
IKSolution
IKTrajectorySolution
~~~

Moro 0.5.0 adds:

~~~python
solve_pose_ik(...)
PoseIKSolution
~~~

No solve_pose_trajectory() is introduced in this release.

The implementation should reuse existing solver infrastructure wherever the mathematical responsibility is genuinely shared, while keeping the position and pose solving loops readable and explicit.

The solver remains numerical. Symbolic robot models are supported through numerical substitution of model parameters before iteration.

## 2. Public API

The planned public function is:

~~~python
solve_pose_ik(
    robot,
    target,
    q0=None,
    *,
    method="lm",
    position_weight=1.0,
    orientation_weight=1.0,
    position_tol=1e-6,
    orientation_tol=1e-6,
    joint_limits=None,
    parameters=None,
    max_iter=None,
    damping=1.0,
    damping_scale=0.5,
    random_state=None,
    step_tol=1e-12,
    error_change_tol=1e-12,
    stagnation_iterations=5,
)
~~~

The exact defaults may be adjusted during implementation only if required for compatibility or demonstrated numerical behavior, but the public concepts and semantics described below should remain stable.

Supported methods are:

~~~text
"lm"
"newton"
~~~

"lm" is the default and recommended method.

CCD remains position-only. Calling solve_pose_ik(..., method="ccd") must raise a clear ValueError.

## 3. Canonical pose target

The canonical public target is a numerical homogeneous transformation:

\[
T_d=
\begin{bmatrix}
R_d & p_d\\
0 & 1
\end{bmatrix}
\in SE(3).
\]

Euler angles, quaternions, axis-angle pairs, and rotation vectors are not accepted as primary pose-target formats in 0.5.0.

Users should convert those representations explicitly through moro.transformations before calling solve_pose_ik().

## 4. Target validation

The target must be convertible to a numerical \(4\times4\) matrix and satisfy:

\[
T_d.shape=(4,4).
\]

The homogeneous last row must be approximately:

\[
[0,0,0,1].
\]

The rotation block must satisfy, within the established transformation-validation tolerance policy:

\[
R_d^TR_d\approx I,
\qquad
\det(R_d)\approx +1.
\]

All entries must be real and finite.

Validation should reuse the reusable SO(3) / SE(3) utilities designed in moro.transformations.

The IK solver must not silently project an invalid or nearly valid rotation onto SO(3). Projection or correction, if desired, must be an explicit transformation operation performed before IK.

## 5. Numerical versus symbolic target behavior

The robot model may contain symbolic parameters. The target itself must be numerical.

The parameters argument is reserved for resolving symbolic quantities in the robot model, for example:

~~~python
parameters = {
    l1: 1.0,
    l2: 0.8,
}
~~~

It is not the mechanism for resolving symbolic entries inside target.

A symbolic target with unresolved free symbols is invalid input.

## 6. Pose error convention

Let the current end-effector pose be:

\[
T(q)=
\begin{bmatrix}
R(q) & p(q)\\
0 & 1
\end{bmatrix}.
\]

### 6.1 Position error

The position residual is:

\[
e_p = p_d-p(q).
\]

It is expressed in the base frame.

### 6.2 Orientation error

The orientation residual is defined from the base-frame relative rotation:

\[
R_e=R_dR(q)^T.
\]

The orientation error vector is:

\[
\boxed{
e_R=
\operatorname{Log}\left(R_dR(q)^T\right)^\vee
}
\]

or equivalently through the transformation utility:

~~~python
R_error = R_target @ R_current.T
e_R = rot2rotvec(R_error)
~~~

The vector \(e_R\) is expressed in the base frame and represents the principal rotation that takes the current orientation to the desired orientation.

Conceptually:

\[
R_d=
\operatorname{Exp}([e_R]_\times)R(q).
\]

This convention matches Moro's geometric-Jacobian convention, where vector and physical quantities are expressed in the base frame unless otherwise stated.

## 7. Geometric Jacobian compatibility

The full geometric Jacobian is:

\[
J=
\begin{bmatrix}
J_v\\
J_\omega
\end{bmatrix}.
\]

For a small joint increment \(\Delta q\):

\[
R(q+\Delta q)
\approx
\operatorname{Exp}
\left(
[J_\omega\Delta q]_\times
\right)
R(q).
\]

Therefore the orientation update used by the solver is:

\[
J_\omega\Delta q\approx e_R.
\]

The combined local pose system is:

\[
\begin{bmatrix}
J_v\\
J_\omega
\end{bmatrix}
\Delta q
\approx
\begin{bmatrix}
e_p\\
e_R
\end{bmatrix}.
\]

## 8. Scope of the SO(3) linearization

The exact derivative of a finite rotation-vector residual involves an inverse left/right Jacobian of SO(3).

Moro 0.5.0 does not introduce that additional machinery for pose IK.

The solver uses the geometric angular Jacobian directly:

\[
J_R=J_\omega.
\]

This is the intended local linearization.

Near convergence:

\[
e_R\to0,
\]

and the exact SO(3) Jacobian correction tends to identity, so the adopted formulation matches the local behavior required by an iterative solver.

Public SO(3) left/right Jacobian utilities remain outside the scope of this feature.

## 9. Weighted pose residual

Position and orientation have different physical scales.

The solver therefore uses scalar weights:

\[
w_p>0,
\qquad
w_R>0.
\]

The weighted residual is:

\[
e_w=
\begin{bmatrix}
w_pe_p\\
w_Re_R
\end{bmatrix},
\]

and the weighted Jacobian is:

\[
J_w=
\begin{bmatrix}
w_pJ_v\\
w_RJ_\omega
\end{bmatrix}.
\]

Weights must be applied consistently to both residual and Jacobian.

The internal scalar merit function is:

\[
E_w=\|e_w\|_2
=
\sqrt{
w_p^2\|e_p\|_2^2+
w_R^2\|e_R\|_2^2
}.
\]

No full weighting-matrix API is introduced in 0.5.0.

No characteristic-length or automatic translational/rotational normalization is performed.

## 10. Weight validation

position_weight and orientation_weight must be scalar, numeric, real, finite, and strictly positive.

Zero weights are not accepted. Allowing a zero orientation or position weight would implicitly turn solve_pose_ik() into a partial-task solver, which is outside the intended 0.5.0 contract.

Large or highly unbalanced positive weights are permitted. The documentation should note that extreme weights can affect conditioning and solver behavior.

## 11. Convergence semantics

Weights influence optimization behavior, but they do not define whether a pose was reached.

Convergence is checked using independent physical residual criteria:

\[
\|e_p\|_2\le\varepsilon_p,
\]

and:

\[
\|e_R\|_2\le\varepsilon_R.
\]

Therefore:

~~~python
converged = (
    position_error <= position_tol
    and orientation_error <= orientation_tol
)
~~~

There is no single public weighted pose tolerance.

solve_pose_ik() does not expose the position-IK tol argument.

## 12. Tolerance semantics

position_tol is expressed in the linear units used by the robot model.

orientation_tol is expressed in radians because \(\|e_R\|\) is the principal rotation angle.

Both tolerances must be finite, real, scalar, and strictly positive.

## 13. Newton method

For method="newton", the local update solves:

\[
J_w\Delta q\approx e_w.
\]

If \(J_w\) is square, the implementation may first attempt:

\[
J_w\Delta q=e_w
\]

with a direct numerical solve.

If the matrix is singular, or if the system is rectangular, use the Moore-Penrose pseudoinverse:

\[
\boxed{
\Delta q=J_w^\dagger e_w
}
\]

This allows lower-DOF robots, redundant robots, overdetermined local pose equations, and singular or rank-deficient configurations.

No null-space secondary objective is added.

## 14. Levenberg-Marquardt method

For method="lm", the update is:

\[
\boxed{
(J_w^TJ_w+\lambda^2I)\Delta q
=
J_w^Te_w
}
\]

with positive damping:

\[
\lambda>0.
\]

This preserves the numerical philosophy of the current position-IK solver.

The initial damping is controlled by damping. The adaptation factor is controlled by damping_scale. Existing validation and scaling semantics should be retained where possible.

## 15. LM trial acceptance

A trial step is evaluated at:

\[
q_{trial}=q+\Delta q,
\]

after joint-limit enforcement.

Compute the trial weighted merit \(E_{w,trial}\).

The trial is accepted when:

\[
E_{w,trial}<E_w.
\]

On acceptance, damping is reduced according to the existing solver policy. On rejection, damping is increased according to the existing solver policy.

A rejected LM attempt still counts as an algorithm iteration, preserving current position-IK iteration semantics.

## 16. Position and orientation convergence are independent

If position is within tolerance but orientation is not, the solver continues.

If orientation is within tolerance but position is not, the solver continues.

Only simultaneous satisfaction of both criteria yields converged=True.

## 17. Robots with fewer or more than six DOF

A robot must not be rejected solely because:

\[
n<6.
\]

A lower-DOF manipulator may reach particular full poses even when it cannot locally realize arbitrary six-dimensional twists.

Likewise, redundant manipulators with \(n>6\) are valid.

The solver attempts the requested target and convergence is determined only by the final physical residual criteria.

No DOF-count precondition such as robot.dof >= 6 is allowed.

## 18. Singular Jacobians

A singular or rank-deficient Jacobian is not an input error.

Newton may fall back to pseudoinverse solving. LM continues through regularization.

A singular configuration does not automatically imply failure because a particular requested residual may still lie in the locally attainable subspace.

No dedicated singularity exception is raised.

## 19. Initial guess and reproducibility

q0 must contain exactly \(n=robot.dof\) finite real values when supplied.

Partial configurations and dictionary-style configurations are not supported.

If q0=None, the existing random-initialization infrastructure should be reused.

random_state preserves the current reproducibility semantics.

### 19.1 Compatibility with current q0 clipping

The existing position-IK helper clips an explicit q0 to the active joint limits.

For compatibility, solve_pose_ik() should preserve that behavior in 0.5.0.

Therefore an explicit initial guess outside the active limits is normalized by clipping rather than rejected.

## 20. Joint limits

solve_pose_ik() uses exactly the same public joint-limit format and semantics as solve_position_ik().

No pose-specific joint-limit syntax is introduced.

After computing a candidate update:

\[
q+\Delta q,
\]

the trial configuration is clipped to the active bounds before forward-kinematics evaluation.

This keeps all accepted solver states within the configured joint limits.

This policy is simple bound projection, not a constrained optimal IK method.

If a target would require violating limits, the expected outcome is a normal non-converged solution rather than an exception.

## 21. Effective step for stagnation

Joint-limit clipping can reduce or eliminate an attempted update.

The step used for stagnation detection should therefore be the effective post-limit step:

\[
\Delta q_{eff}=q_{trial}-q.
\]

The step-size stagnation condition uses:

\[
\|\Delta q_{eff}\|_2.
\]

This correctly identifies a solver repeatedly pushing against an active bound without actual motion.

## 22. Error-change stagnation

The error-change stagnation criterion uses the weighted merit function:

\[
E_w.
\]

Conceptually:

\[
|E_w^{k+1}-E_w^k|
\le
error\_change\_tol
\]

for the configured number of consecutive stagnation iterations.

This is an internal algorithmic criterion, so dependence on the user-selected weights is intentional.

## 23. Initially satisfied target

If the initial configuration already satisfies both position and orientation tolerances, return immediately with:

~~~text
converged = True
iterations = 0
~~~

No artificial Newton/LM step should be executed.

## 24. Orientation behavior near zero

When:

\[
R_dR^T\approx I,
\]

the orientation residual should approach:

\[
e_R\approx0.
\]

No special solver-level small-angle branch is required beyond the robust rot2rotvec() behavior supplied by moro.transformations.

## 25. Orientation behavior near pi

For an orientation difference near \(\theta=\pi\), the rotation axis has an unavoidable sign ambiguity:

\[
\pi u
\quad\text{and}\quad
-\pi u
\]

represent the same physical rotation.

The solver must accept either valid principal rotation-vector branch.

Tests near \(\pi\) should compare rotation equivalence and/or residual norm rather than require a fixed component-wise sign.

A 180-degree orientation difference is difficult but valid input and must not be rejected solely for that reason.

## 26. Reachable versus unreachable targets

A valid target may be unreachable because of workspace limitations, orientation incompatibility, insufficient task capability, active joint limits, local minima, singularity, or poor initialization.

These conditions do not make the input invalid.

They produce a normal:

~~~python
PoseIKSolution(
    converged=False,
    ...
)
~~~

with final diagnostics whenever those diagnostics can be evaluated.

Input validation errors are reserved for malformed or incomplete problems such as invalid target transformations, invalid options, invalid weights/tolerances, invalid limits, invalid q0 content, or unresolved required model parameters.

## 27. No automatic restart or multi-start

Moro 0.5.0 performs one solve from the prepared initial configuration.

No automatic restart, branch search, or multi-start strategy is added.

Users may explore alternative solution branches by supplying different q0 values or random seeds.

## 28. PoseIKSolution

Pose IK uses a dedicated result type rather than extending IKSolution with pose-specific optional fields.

The planned result is:

~~~python
@dataclass
class PoseIKSolution:
    q: list
    converged: bool
    iterations: int
    position_error: float
    orientation_error: float
    method: str = "lm"
    position_residual: Optional[list] = None
    orientation_residual: Optional[list] = None
    target_pose: Optional[Matrix] = None
    achieved_pose: Optional[Matrix] = None
    message: str = ""
~~~

The public field name converged is retained for consistency with IKSolution and IKTrajectorySolution.

## 29. Pose-result semantics

The vector residuals are:

\[
position\_residual=e_p,
\]

\[
orientation\_residual=e_R.
\]

Their scalar diagnostics are:

\[
position\_error=\|e_p\|_2,
\]

\[
orientation\_error=\|e_R\|_2.
\]

The orientation error is therefore directly interpretable as the remaining principal angular error in radians.

No public combined weighted-error field is required initially.

## 30. Target and achieved pose storage

PoseIKSolution stores both target_pose and achieved_pose as public \(4\times4\) SymPy matrices when available.

This makes a normal result self-contained and allows users to inspect the requested and achieved transformations without manually recomputing FK.

target_pose should always be available for solutions produced by a successfully validated public call.

achieved_pose may be None only when a numerical failure prevents a valid final FK evaluation.

## 31. PoseIKSolution invariants

For a normal numerically evaluable result:

~~~text
len(q) == robot.dof
len(position_residual) == 3
len(orientation_residual) == 3
target_pose.shape == (4, 4)
achieved_pose.shape == (4, 4)
~~~

and:

\[
position\_error=\|position\_residual\|_2,
\]

\[
orientation\_error=\|orientation\_residual\|_2.
\]

Also:

~~~text
iterations >= 0
position_error >= 0
orientation_error >= 0
method in {"newton", "lm"}
~~~

A converged result requires finite joint values, errors, residuals, and achieved pose.

## 32. Numerical-failure result behavior

When a numerical failure prevents a final pose/residual evaluation, the result may use:

~~~text
converged = False
achieved_pose = None
position_residual = None
orientation_residual = None
position_error = inf
orientation_error = inf
~~~

while preserving the validated target_pose and final finite joint state when possible.

This follows the current position-IK philosophy of distinguishing non-convergence from unavailable numerical diagnostics.

## 33. Result representation

PoseIKSolution.__repr__() should remain compact.

Conceptually:

~~~text
PoseIKSolution(
    q=[...],
    Converged,
    method=lm,
    iters=12,
    position_error=2.1e-07,
    orientation_error=4.8e-07
)
~~~

The full target and achieved matrices should not be printed in the compact representation.

## 34. Internal architecture

The refactor should remain moderate and educationally readable.

The intended module structure is conceptually:

~~~text
# result types
IKSolution
IKTrajectorySolution
PoseIKSolution

# shared validation/preparation
_prepare_rng
_validate_stagnation_options
_validate_common_solver_options
_validate_method

# limits / initialization
_prepare_joint_limits
_prepare_initial_guess

# symbolic/numerical preparation
_apply_parameters
_validate_free_symbols
_prepare_position_ik_model
_prepare_pose_ik_model

# state evaluation
_evaluate_position_state
_evaluate_pose_state

# shared Jacobian-based numerical machinery
_compute_ik_step
small LM/stagnation helpers if useful

# result builders
_make_position_solution
_make_pose_solution

# solver implementations
_solve_position_jacobian
_solve_position_ccd
_solve_pose_jacobian

# public API
solve_position_ik
solve_position_trajectory
solve_pose_ik
~~~

Exact private helper names may evolve during implementation.

## 35. Shared validation helpers

The following existing responsibilities should remain shared where practical:

~~~text
_prepare_rng
_prepare_joint_limits
_prepare_initial_guess
_apply_parameters
_validate_free_symbols
_is_finite_array
_as_finite_vector
_validate_stagnation_options
~~~

Solver-option validation may be split so that common scalar options are shared while method availability remains solver-specific.

## 36. Model preparation

Position and pose model preparation should remain explicit rather than hidden behind a generic mode flag.

Conceptually:

~~~python
_prepare_position_ik_model(robot, parameters)
_prepare_pose_ik_model(robot, parameters)
~~~

For position:

\[
p(q)=T(q)_{0:3,3},
\qquad
J_p=J_v.
\]

For pose:

\[
T(q)=robot.T,
\qquad
J=robot.J.
\]

Both preparation paths reuse parameter substitution, free-symbol validation, and lambdify.

## 37. State-evaluation helpers

Position and pose should use separate evaluation helpers.

A pose-state evaluator should produce conceptually:

~~~text
current_pose
position_residual
orientation_residual
position_error
orientation_error
weighted_residual
weighted_error_norm
~~~

The orientation residual calculation should exist in one place to avoid inconsistent relative-rotation conventions across trial, current, and final evaluations.

## 38. Shared Newton/LM step helper

A private helper may compute the Jacobian-based update:

~~~python
_compute_ik_step(
    J,
    residual,
    *,
    method,
    damping,
)
~~~

For Newton:

\[
\Delta q\approx J^\dagger e
\]

with direct solve permitted for a regular square system.

For LM:

\[
(J^TJ+\lambda^2I)\Delta q=J^Te.
\]

This helper need not know whether the residual came from position IK or pose IK.

## 39. Do not over-generalize the full iteration loop

Moro 0.5.0 should not introduce a callback-heavy generic optimizer unless implementation experience demonstrates that it is clearly simpler.

It is acceptable for _solve_position_jacobian and _solve_pose_jacobian to contain parallel explicit loops while reusing small mathematical helpers.

Readability and correspondence with the numerical algorithm are preferred over eliminating every repeated control-flow line.

## 40. CCD remains separate

CCD has a different joint-by-joint geometric update structure and remains isolated from the shared Newton/LM step machinery.

The refactor must not force CCD through an inappropriate Jacobian-step abstraction.

## 41. Position-IK compatibility

The 0.5.0 refactor should preserve the external behavior of solve_position_ik(), solve_position_trajectory(), IKSolution, and IKTrajectorySolution unless a concrete existing bug is identified and intentionally fixed.

Full-pose IK should not require users of position-only IK to change code.

## 42. Messages and outcome categories

Pose IK should reuse the established outcome categories where possible:

~~~text
Converged successfully.
Maximum number of iterations reached.
Solver stagnated because the joint update became too small.
Solver stagnated because the pose error stopped improving.
Numerical failure while evaluating the forward kinematics.
Numerical failure while evaluating the Jacobian.
Numerical failure while computing the joint update.
~~~

Messages should describe the observed solver outcome rather than speculate about geometric causes.

Avoid messages claiming failure solely because the robot has fewer than six DOF or because the Jacobian is singular.

## 43. Test strategy

### 43.1 Pose-error mathematics

Tests should verify:

- zero pose residual when current pose equals target;
- pure position mismatch with zero orientation mismatch;
- pure orientation mismatch with zero position mismatch;
- base-frame sign/convention consistency of the orientation residual;
- small-angle compatibility between \(e_R\) and \(J_\omega\Delta q\);
- orientation residual near zero;
- orientation residual near \(\pi\);
- rotation equivalence rather than fixed axis sign at \(\pi\).

### 43.2 Forward-kinematics round trips

For known configurations:

~~~python
q_target = [...]
T_target = fk(q_target)

solution = solve_pose_ik(
    robot,
    T_target,
    q0=...,
)
~~~

acceptance is based on the achieved pose:

\[
\|p_d-p(q)\|\le position\_tol,
\]

\[
\|\operatorname{Log}(R_dR(q)^T)^\vee\|
\le orientation\_tol.
\]

Tests must not require recovery of the exact original joint vector when multiple IK branches exist.

### 43.3 Method tests

Test both LM and Newton on representative well-posed targets.

No requirement is imposed that both methods converge in the same number of iterations.

Calling pose IK with CCD must fail clearly.

### 43.4 Weighting tests

Cover default weights, position-dominant weighting, orientation-dominant weighting, invalid zero/negative weights, non-finite weights, and non-real weights.

Changing weights may change the solver path, but converged results must still satisfy the same independent position/orientation tolerances.

### 43.5 Tolerance tests

Explicitly cover:

~~~text
position inside tolerance, orientation outside -> not yet converged
orientation inside tolerance, position outside -> not yet converged
both inside tolerance -> converged
~~~

### 43.6 Target validation tests

Cover wrong shape, invalid homogeneous last row, non-orthogonal rotation, determinant not approximately +1, reflection, NaN, Inf, unresolved symbolic target, and valid numerical homogeneous transformations.

No invalid target should be silently projected.

### 43.7 Joint-limit tests

Cover reachable pose within limits, pose blocked by limits, final state always inside limits, explicit q0 clipping compatibility, and repeated pushing against a bound leading to stagnation through the effective post-clip step.

### 43.8 Symbolic-model parameter tests

Cover successful parameter substitution, missing required parameters, multiple model parameters, and preservation of the original symbolic robot model after solving.

### 43.9 DOF-structure tests

Include:

- representative 6-DOF full-pose case;
- redundant robot \(n>6\);
- lower-DOF robot reaching a particular pose;
- lower-DOF robot with an incompatible target;
- at least one model containing a prismatic joint.

The lower-DOF reachable case is an explicit acceptance requirement.

### 43.10 Singular / near-singular behavior

Tests should verify that singularity does not raise merely because rank is deficient, Newton can fall back to pseudoinverse behavior, LM remains numerically defined through regularization, a singular configuration may still converge for a compatible target, and difficult cases may return a valid non-converged result.

### 43.11 Initially satisfied target

Verify:

~~~text
converged = True
iterations = 0
~~~

when q0 already achieves the requested pose within both tolerances.

### 43.12 PoseIKSolution invariants

For normal solutions verify shape and length invariants and numerical consistency between vector residuals and scalar error norms.

For non-converged but numerically valid solves, these diagnostics should still be available.

For genuine numerical failures, optional diagnostics may be None and scalar errors may be infinite.

## 44. Documentation examples

At least four user-facing examples should accompany the feature.

### 44.1 Full-pose FK -> IK round trip

Construct a target from a known configuration and solve it with LM.

Show q, converged, iterations, position_error, orientation_error, and achieved_pose.

### 44.2 Position IK versus pose IK

Use targets with the same Cartesian position but different end-effector orientation requirements.

Demonstrate that position-only IK constrains only position, while pose IK constrains both translation and rotation.

### 44.3 Effect of position/orientation weights

Solve the same target with different positive weight ratios and compare solver behavior.

Clarify that weighting changes the numerical path, not the final convergence definition.

### 44.4 Restricted or underactuated target

Demonstrate either a target blocked by joint limits or a lower-DOF robot facing an incompatible full-pose target.

Show that the solver returns a useful non-converged PoseIKSolution with final pose diagnostics rather than raising solely because the target is unreachable.

## 45. Relationship with moro.transformations

Full-pose IK depends on transformation utilities for homogeneous-transform validation, robust rotation-vector extraction, stable behavior near zero, and valid principal rotation-vector behavior near \(\pi\).

These mathematical operations belong in moro.transformations, not as solver-local duplicate formulas.

## 46. Relationship with differential kinematics

Pose IK uses the same geometric Jacobian convention as moro.differential_kinematics:

\[
J=
\begin{bmatrix}
J_v\\
J_\omega
\end{bmatrix},
\]

with linear and angular rows expressed in the base frame.

The orientation residual convention is chosen specifically to remain compatible with that Jacobian frame convention.

## 47. Explicitly outside Moro 0.5.0

The following remain outside the present full-pose IK scope:

- Euler-angle targets as primary API input;
- quaternion targets as primary API input;
- axis-angle targets as primary API input;
- rotation-vector targets as primary API input;
- CCD full-pose IK;
- partial-pose task subsets;
- full weighting-matrix APIs;
- automatic characteristic-length normalization;
- exact SO(3) residual Jacobian corrections;
- public SO(3) left/right Jacobian utilities added solely for IK;
- analytical full-pose IK;
- null-space secondary objectives;
- manipulability optimization during IK;
- collision-aware IK;
- constrained/QP IK;
- automatic multi-start or branch search;
- pose-trajectory IK;
- orientation interpolation / SLERP as part of IK;
- closed-chain IK.

## 48. Acceptance criteria

The full-pose IK block is ready for implementation when:

1. solve_pose_ik() uses a validated numerical SE(3) target;
2. position IK remains externally compatible;
3. the orientation residual is \(\operatorname{Log}(R_dR^T)^\vee\) in the base frame;
4. the geometric angular Jacobian \(J_\omega\) is used as the local orientation linearization;
5. position and orientation weights are positive scalar values applied consistently to residual and Jacobian;
6. convergence requires independent position and orientation tolerances;
7. LM and Newton are supported, while CCD remains position-only;
8. lower-DOF robots are not rejected by DOF count alone;
9. singular Jacobians are handled numerically rather than treated as automatic failures;
10. joint-limit behavior remains compatible with the current position solver;
11. stagnation step size uses the effective post-limit update;
12. unreachable but valid targets return informative non-converged results;
13. PoseIKSolution preserves target/achieved pose and separate position/orientation diagnostics;
14. shared helpers are reused without hiding the position and pose algorithms behind excessive abstraction;
15. representative regular, weighted, bounded, symbolic-parameter, lower-DOF, singular, near-zero, and near-\(\pi\) orientation cases are covered by tests;
16. no 0.5.0 implementation introduces pose trajectories, partial tasks, null-space objectives, constrained optimization, or automatic restart logic.
