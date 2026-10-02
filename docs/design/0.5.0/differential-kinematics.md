# Detailed design: `moro.differential_kinematics` for Moro 0.5.0

This document records the detailed design decisions for the differential-kinematics work planned for Moro 0.5.0. It complements `docs/roadmap/0.5.0/differential-kinematics.md`, which remains the higher-level scope document.

The goal is to add a compact functional layer for task-space Jacobians, forward differential kinematics, and velocity-level inverse kinematics without duplicating the geometric Jacobian machinery already owned by `Robot` in `moro.core`.

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

The existing geometric model remains in `moro.core`.

`Robot.J` and `Robot.J_point()` are the source of truth for geometric Jacobians. The new module must not recompute Jacobians directly from DH parameters or duplicate the kinematic derivation already implemented in `Robot`.

`moro.differential_kinematics` owns:

- task-component selection;
- symbolic/numerical Jacobian evaluation at a configuration;
- Cartesian velocity propagation;
- velocity-level inverse kinematics;
- velocity-IK diagnostics.

The API remains functional rather than adding stateful methods such as `robot.solve_velocity_ik()`.

The canonical geometric-twist ordering is

\[
(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z).
\]

All task-space vector quantities are expressed in the base frame `{0}` in 0.5.0, consistent with Moro's current frame convention. Tool/body Jacobians and frame-selection arguments remain outside this release.

The implementation should preserve the SymPy-first character of Moro for modeling and direct velocity propagation, while using NumPy numerical linear algebra for the actual velocity-IK solver.

## 2. Public API

The public API is:

```python
task_jacobian(robot, q=None, *, task="twist", parameters=None)
```

```python
cartesian_velocity(
    robot,
    q,
    qd,
    *,
    task="twist",
    parameters=None,
)
```

```python
solve_velocity_ik(
    robot,
    q,
    velocity,
    *,
    task="twist",
    method="pinv",
    damping=None,
    joint_velocity_limits=None,
    parameters=None,
    tol=1e-9,
)
```

The structured solver result is:

```python
@dataclass(frozen=True)
class VelocityIKSolution:
    qd: Matrix
    unconstrained_qd: Matrix
    desired_velocity: Matrix
    achieved_velocity: Matrix
    residual: Matrix
    residual_norm: float
    rank: int
    condition_number: float
    method: str
    success: bool
    limited: bool
    message: str
```

No public inverse-matrix method is planned. The initial solver methods are:

```text
"pinv"
"dls"
```

## 3. Task-space representation

### 3.1 Canonical components

The six named geometric-twist components are:

```text
vx, vy, vz, wx, wy, wz
```

The canonical row mapping is:

```python
_TASK_ROWS = {
    "vx": 0,
    "vy": 1,
    "vz": 2,
    "wx": 3,
    "wy": 4,
    "wz": 5,
}
```

Roll/pitch/yaw terminology must not be used for geometric-Jacobian angular rows.

### 3.2 Presets

The convenience presets are:

```python
_TASK_PRESETS = {
    "linear": ("vx", "vy", "vz"),
    "angular": ("wx", "wy", "wz"),
    "twist": ("vx", "vy", "vz", "wx", "wy", "wz"),
}
```

### 3.3 Explicit task subsets

Explicit task subsets are supported, for example:

```python
task=("vx", "vy")
task=("vx", "vy", "wz")
task=("wz", "vx")
```

The order requested by the user must be preserved exactly. For example,

```python
task=("wz", "vx")
```

produces a two-row Jacobian ordered as

\[
\begin{bmatrix}
J_{\omega_z}\\
J_{v_x}
\end{bmatrix}.
\]

The same order defines the expected ordering of `velocity` in `solve_velocity_ik()`.

Task dimensionality must never be inferred from the length of a velocity vector.

### 3.4 Task normalization

A private helper should normalize task specifications:

```python
_normalize_task(task)
```

Accepted public forms are:

- one of the string presets;
- an explicit sequence of component-name strings.

Matching is case-insensitive.

The normalizer should return a tuple such as:

```python
("vx", "vy", "wz")
```

The following should be rejected:

- unsupported input types;
- unknown preset strings;
- unknown component names;
- empty task sequences;
- duplicate components.

Duplicate components such as

```python
task=("vx", "vx")
```

are invalid because they create an artificial repeated task equation with no useful public interpretation.

Integer row indices are not part of the public API.

## 4. `task_jacobian()`

### 4.1 Role

`task_jacobian()` is a functional selection/evaluation layer over `robot.J`.

Conceptually:

```python
J = robot.J
components = _normalize_task(task)
rows = [_TASK_ROWS[name] for name in components]
J_task = J[rows, :]
```

It must not independently derive the Jacobian from the robot's DH parameters.

### 4.2 Symbolic and evaluated behavior

When

```python
q=None
```

no joint substitution is performed and the selected Jacobian remains symbolic in `robot.qs`.

When `q` is supplied, the entries are substituted in the same order as `robot.qs`.

For example:

```python
task_jacobian(robot, q=[q1_value, q2_value])
```

substitutes

```python
dict(zip(robot.qs, q))
```

into the selected Jacobian.

`parameters` is then applied as a separate symbolic-substitution mapping.

A Jacobian evaluated at `q` may still contain unresolved model parameters. This is acceptable for `task_jacobian()` because it remains an inspection/modeling function rather than a numerical solver.

### 4.3 Return type and shape

The return value is always a SymPy matrix.

For a task of dimension `m` and a robot with `n=robot.dof`,

\[
J_{task}\in\mathbb{R}^{m\times n}.
\]

### 4.4 Robot interface expectations

The function should not require strict `isinstance(robot, Robot)` checking.

The required interface is conceptually:

```text
robot.J
robot.qs
robot.dof
```

This keeps the API flexible and simplifies compatible test doubles or future robot-model abstractions.

Missing required attributes should produce a clear error rather than an opaque attribute traceback where practical.

## 5. Configuration and parameter evaluation

### 5.1 Joint configuration `q`

`q` represents only the robot configuration and is positional with respect to `robot.qs`.

Accepted vector-like forms may include lists, tuples, and SymPy row/column matrices, normalized to a column or flat internal representation as appropriate.

The number of components must be exactly:

\[
\operatorname{len}(q)=\texttt{robot.dof}.
\]

Partial configurations are not supported. Inputs such as

```python
[0.2, None, 0.5]
```

or vectors with fewer/more than `robot.dof` entries are invalid.

Dictionary-form joint configurations are not part of the 0.5.0 public contract.

### 5.2 `parameters`

`parameters` resolves non-joint symbolic quantities, such as link lengths or other model constants.

The intended form is a mapping from SymPy objects to numerical values, for example:

```python
parameters = {
    l1: 1.0,
    l2: 0.8,
}
```

String-key parameter substitution is not part of the initial API.

### 5.3 Evaluation order

The conceptual order is:

```text
symbolic expression
    -> substitute q
    -> substitute parameters
    -> inspect unresolved symbols
    -> validate real/finiteness when numerical evaluation is required
```

This order should be shared across the module.

### 5.4 Numeric meaning in a SymPy-first library

"Numerical" does not necessarily mean floating-point.

Exact SymPy values such as

```text
Integer
Rational
pi
sqrt(2)
```

are acceptable when no free symbols remain.

`cartesian_velocity()` may therefore preserve exact SymPy arithmetic.

`solve_velocity_ik()`, however, eventually converts evaluated quantities to finite real floating-point arrays for NumPy linear algebra.

## 6. `cartesian_velocity()`

### 6.1 Mathematical definition

For the selected task,

\[
\dot x_{task}=J_{task}(q)\dot q.
\]

`q` and `qd` are required.

### 6.2 Joint-velocity vector

`qd` must contain exactly `robot.dof` components and is interpreted in the same joint order as `robot.qs`.

A scalar shortcut is not planned, even for a one-DOF robot; users should provide a one-component vector.

Symbolic entries in `qd` are allowed only if `parameters` resolves them before numerical completeness is checked.

### 6.3 Return type

The result is always a SymPy column matrix of dimension equal to the selected task size.

For example:

```python
task=("vx", "wz")
```

returns

\[
\begin{bmatrix}
v_x\\
\omega_z
\end{bmatrix}.
\]

No structured result object is needed for forward velocity propagation.

### 6.4 Numeric completeness

After evaluating `q`, `qd`, and `parameters`, no unresolved symbols may remain in the final velocity result.

This allows exact symbolic numbers while rejecting genuinely unresolved symbolic models.

### 6.5 Joint limits

`cartesian_velocity()` does not inspect or enforce joint-velocity limits. It propagates the velocity supplied by the user exactly as a kinematic operation.

## 7. `solve_velocity_ik()`

### 7.1 Problem definition

Given

\[
J\equiv J_{task}(q)\in\mathbb{R}^{m\times n}
\]

and a desired task velocity

\[
\dot x_d\in\mathbb{R}^{m},
\]

the solver computes a joint velocity

\[
\dot q\in\mathbb{R}^{n}
\]

using either the Moore-Penrose pseudoinverse or damped least squares.

### 7.2 Desired velocity

`velocity` must have exactly the number of components implied by the normalized task.

Its ordering follows the task ordering exactly.

For example:

```python
task=("wz", "vx")
velocity=[omega_z_desired, vx_desired]
```

is valid and unambiguous.

Task dimensionality is never inferred from `velocity` length.

### 7.3 Numerical solver boundary

The symbolic model is fully evaluated before numerical solving.

After substitution, both the selected Jacobian and desired velocity must be:

- free of unresolved symbols;
- real;
- finite.

They are then converted to NumPy floating-point arrays.

The solver proper operates only on finite real arrays.

If numerical linear algebra produces non-finite values, the function should raise a clear numerical failure rather than returning an apparently valid `VelocityIKSolution` containing `nan` or `inf` joint velocities.

## 8. Pseudoinverse method

### 8.1 Method name

```python
method="pinv"
```

selects the Moore-Penrose pseudoinverse solution:

\[
\dot q=J^\dagger\dot x_d.
\]

### 8.2 SVD implementation

The implementation should use a single economy-size SVD:

\[
J=U\Sigma V^T.
\]

Conceptually:

```python
U, s, Vt = np.linalg.svd(J, full_matrices=False)
```

The same singular values should support:

- pseudoinverse construction;
- rank calculation;
- condition-number calculation.

This prevents slightly different default tolerances from separate NumPy routines from creating inconsistent diagnostics.

### 8.3 Rank threshold

The internal numerical threshold should follow a standard scale-sensitive rule such as

\[
\tau=\max(m,n)\,\epsilon\,\sigma_{max},
\]

where `epsilon` is machine precision for the floating-point dtype.

The numerical rank is

\[
\operatorname{rank}(J)=\#\{\sigma_i>\tau\}.
\]

This threshold is internal and distinct from the public task-residual `tol`.

### 8.4 Pseudoinverse singular values

The Moore-Penrose reciprocal rule is

\[
\sigma_i^\dagger=
\begin{cases}
1/\sigma_i, & \sigma_i>\tau,\\
0, & \sigma_i\le\tau.
\end{cases}
\]

The resulting solution is the minimum-norm solution in underdetermined/redundant problems and the least-squares solution in overdetermined problems.

## 9. Damped least squares

### 9.1 Method name and damping

```python
method="dls"
```

selects damped least squares.

The conceptual formula is

\[
J_\lambda^\dagger
=
J^T(JJ^T+\lambda^2I)^{-1}.
\]

`damping` must be supplied explicitly and satisfy

\[
\lambda>0.
\]

No implicit damping default is selected by Moro.

If `method="pinv"`, supplying `damping` is considered an invalid argument combination rather than being silently ignored.

### 9.2 SVD-based implementation

Although the public/theoretical documentation may present the classical DLS expression, the implementation should preferably reuse the existing SVD:

\[
J_\lambda^\dagger
=
V\,\operatorname{diag}\left(
\frac{\sigma_i}{\sigma_i^2+\lambda^2}
\right)U^T.
\]

This avoids explicitly forming and inverting `JJ.T + lambda**2 I`, reuses the already-computed singular values, and makes the effect of damping transparent.

### 9.3 Diagnostic meaning

`rank` and `condition_number` always describe the original task Jacobian `J`, not the damped/regularized operator.

## 10. Rank and condition number

### 10.1 Rank

The returned `rank` is the SVD-based numerical rank of `J_task` at the supplied configuration.

It is not the rank of the full six-row `robot.J` unless the selected task is the full twist.

### 10.2 Condition number

The condition number is based on singular values:

\[
\kappa(J)=\frac{\sigma_{max}}{\sigma_{min}}
\]

when the Jacobian has full rank relative to its minimum dimension.

If

\[
\operatorname{rank}(J)<\min(m,n),
\]

then

```python
condition_number = inf
```

The zero matrix therefore has

```text
rank = 0
condition_number = inf
```

rather than an undefined/NaN diagnostic.

### 10.3 No automatic success/failure interpretation

A high or infinite condition number does not by itself make `success=False`.

Conditioning describes the geometry/numerical sensitivity of the Jacobian; success describes whether the requested velocity was actually achieved within tolerance.

No arbitrary condition-number warning threshold is planned for 0.5.0.

## 11. Residual and success semantics

### 11.1 Achieved velocity

After all solving and any velocity limiting,

\[
\dot x_a=J\dot q.
\]

### 11.2 Residual convention

The residual is defined as

\[
r=\dot x_d-\dot x_a.
\]

That is, it represents the part of the requested task velocity that remains unachieved.

The residual norm is

\[
\|r\|_2.
\]

### 11.3 Success criterion

`success` means:

> the desired task velocity was achieved within the requested task-space residual tolerance.

Formally,

\[
\texttt{success}
\iff
\|r\|_2\le\texttt{tol}.
\]

The public `tol` therefore controls task-achievement semantics only. It should not be reused as the SVD rank threshold.

### 11.4 Tolerance validation

`tol` must be:

- scalar;
- numeric;
- real;
- finite;
- strictly positive.

Invalid types should raise `TypeError`; non-positive numerical values should raise `ValueError`.

## 12. Redundancy, overdetermined tasks, and singularity

### 12.1 Redundant systems

When

\[
m<n,
\]

the pseudoinverse returns the minimum-norm joint velocity.

No null-space secondary objective is included in 0.5.0.

Terms of the form

\[
(I-J^\dagger J)z
\]

remain outside scope.

### 12.2 Overdetermined systems

When

\[
m>n,
\]

the pseudoinverse solves the least-squares problem

\[
\min_{\dot q}\|J\dot q-\dot x_d\|_2.
\]

Overdetermination is not itself an error. A desired velocity that lies in the image of `J` may still be achieved exactly.

### 12.3 Rank deficiency

Rank deficiency is not treated as an exception for either `pinv` or `dls`.

A rank-deficient Jacobian may still achieve a requested velocity that lies within its attainable task-space subspace.

Therefore

```text
rank deficient != automatic failure
```

and `success` remains residual-based.

### 12.4 Near singularity

A Jacobian may remain full rank while having a very small minimum singular value and therefore a large condition number.

`pinv` may then produce large joint velocities even if the task residual is very small.

Moro should report this through `qd` and `condition_number`, not force `success=False` or emit an arbitrary warning.

The DLS method provides the regularized alternative.

### 12.5 Degenerate zero Jacobian

For

\[
J=0,
\]

the minimum-norm pseudoinverse solution is

\[
\dot q=0.
\]

If

\[
\dot x_d=0,
\]

then

```text
rank = 0
condition_number = inf
residual = 0
success = True
```

If the requested velocity is nonzero, the residual equals the desired velocity and `success=False`.

## 13. Joint-velocity limits

### 13.1 Scope

Moro 0.5.0 treats joint-velocity limits as explicit post-solution saturation/clipping.

Constrained least-squares, QP-based solving, active-set redistribution, and re-solving after saturation are outside scope.

### 13.2 Accepted forms

Two public forms are supported.

Symmetric limits:

```python
joint_velocity_limits = [vmax1, vmax2, ..., vmaxn]
```

interpreted as

\[
-v_{max,i}\le\dot q_i\le v_{max,i}.
\]

Asymmetric limits:

```python
joint_velocity_limits = [
    (vmin1, vmax1),
    (vmin2, vmax2),
    ...,
]
```

The number of entries must be exactly `robot.dof`.

For symmetric limits each maximum must be real, finite, and strictly positive.

For asymmetric limits each bound must be real and finite and satisfy

\[
v_{min}<v_{max}.
\]

`None` entries inside the limit sequence are not supported in the initial API. Unlimited solving is represented by

```python
joint_velocity_limits=None
```

### 13.3 Internal normalization

A helper such as

```python
_normalize_joint_velocity_limits(limits, dof)
```

should normalize both accepted forms to explicit `(lower, upper)` pairs.

### 13.4 Saturation semantics

Let

\[
\dot q^\star
\]

be the unconstrained solver result. The returned velocity is obtained component-wise as

\[
\dot q_i=
\operatorname{clip}
(\dot q_i^\star,v_{min,i},v_{max,i}).
\]

The solver does not redistribute the clipped velocity through the remaining joints.

### 13.5 Diagnostics after saturation

`unconstrained_qd` stores the pre-clipping solution.

`qd` stores the actual returned solution.

`limited` is `True` when at least one component was modified by clipping.

All task-space diagnostics must be recomputed from the final returned `qd`:

\[
\dot x_a=J\dot q,
\qquad
r=\dot x_d-\dot x_a.
\]

Therefore joint-velocity saturation may change `success`.

`rank` and `condition_number` do not change after clipping because they are properties of the task Jacobian, not of the selected output velocity.

### 13.6 Units

Moro does not enforce units.

Velocity limits use the natural unit of each generalized coordinate:

- revolute joint -> angular rate;
- prismatic joint -> linear rate.

Users remain responsible for consistent units throughout the robot model.

## 14. Method, damping, and input validation

### 14.1 `method`

`method` must be a string and is matched case-insensitively.

Accepted values are:

```text
pinv
dls
```

Unknown methods raise `ValueError`.

### 14.2 `damping`

For `dls`, `damping` is required and must be a finite real scalar satisfying

\[
\lambda>0.
\]

For `pinv`, `damping` must be `None`; supplying a damping value is treated as an invalid argument combination.

### 14.3 Numerical finiteness

Before entering NumPy linear algebra, numerical quantities used by the solver must reject:

```text
nan
+inf
-inf
complex values
```

This applies to:

- evaluated `q`;
- evaluated desired velocity;
- evaluated Jacobian entries;
- damping;
- joint-velocity limits.

## 15. `VelocityIKSolution` invariants

For a robot with `n` DOF and a selected task of size `m`, every returned solution must satisfy:

```text
qd.shape == (n, 1)
unconstrained_qd.shape == (n, 1)
desired_velocity.shape == (m, 1)
achieved_velocity.shape == (m, 1)
residual.shape == (m, 1)
```

and

\[
\texttt{residual}
=
\texttt{desired\_velocity}
-
\texttt{achieved\_velocity}.
\]

Also:

\[
\texttt{residual\_norm}\ge0,
\]

\[
0\le\texttt{rank}\le\min(m,n).
\]

`method` must be normalized to one of:

```text
pinv
dls
```

`success` must reflect the public residual criterion used in the solve call.

`limited` must reflect whether clipping changed at least one joint-velocity component.

If no joint-velocity limits are supplied:

```text
qd == unconstrained_qd
limited is False
```

## 16. Result messages

Messages should be derived from solver state rather than assembled ad hoc throughout the implementation.

A private helper such as

```python
_build_velocity_ik_message(success, limited)
```

may cover the four principal combinations:

```text
success=True, limited=False
success=False, limited=False
success=True, limited=True
success=False, limited=True
```

Example meanings are:

```text
Velocity task achieved within tolerance.
Velocity task could not be achieved within tolerance.
Joint velocity limits were applied; task achieved within tolerance.
Joint velocity limits were applied; task residual exceeds tolerance.
```

Rank deficiency or a high condition number should not by themselves redefine the message as failure.

## 17. Internal architecture

### 17.1 Module boundary

`moro.core` remains responsible for:

```text
Robot.J
Robot.J_point()
Robot.T
Robot.T_i0()
Robot.R_i0()
robot.qs
robot.dof
```

`moro.differential_kinematics` is responsible for:

```text
task selection
configuration/parameter evaluation
forward velocity mapping
numerical velocity IK
diagnostics
joint-velocity saturation
```

No additional solver state or cache is added to `Robot`.

### 17.2 Task helpers

Expected private helpers/constants include:

```text
_TASK_PRESETS
_TASK_ROWS
_normalize_task()
```

### 17.3 Vector helpers

The module will require compact vector normalization for:

- `q`;
- `qd`;
- task-space velocity.

A local helper such as

```python
_as_vector(value, size, *, name)
```

is acceptable.

Private helpers from `transformations.py` should not be imported across module boundaries merely to reuse implementation details.

If identical validation logic later becomes common across several modules, it may be promoted to a dedicated internal validation module, but 0.5.0 should not introduce that abstraction preemptively.

### 17.4 Evaluation helpers

The implementation may use helpers conceptually equivalent to:

```text
_substitute_configuration()
_substitute_parameters()
_require_numeric_real()
_to_numpy_matrix()
_to_numpy_vector()
```

The intent is to keep the symbolic-to-numerical boundary explicit rather than hide several behaviors behind one highly configurable evaluator.

### 17.5 Solver helpers

Expected solver-specific helpers may include:

```text
_normalize_velocity_ik_method()
_svd_diagnostics()
_solve_pinv()
_solve_dls()
_normalize_joint_velocity_limits()
_apply_joint_velocity_limits()
_build_velocity_ik_message()
```

Exact helper names may evolve during implementation, but responsibilities should remain separated.

### 17.6 No additional caching

`Robot.J` already uses the `Robot` kinematics cache.

`differential_kinematics.py` should not add a second cache for task Jacobians in 0.5.0 because substitution depends on configuration, task, and parameter mappings and the likely gain does not justify cache invalidation complexity.

### 17.7 Dependencies

The initial implementation should require no numerical dependency beyond NumPy in addition to SymPy and Moro itself.

SciPy is not required for the 0.5.0 differential-kinematics scope.

## 18. Conceptual `solve_velocity_ik()` flow

The public solver should conceptually follow:

```python
components = _normalize_task(task)

J_sym = task_jacobian(
    robot,
    q=q,
    task=components,
    parameters=parameters,
)

velocity_sym = normalize_and_evaluate_velocity(...)

J = _to_numpy_matrix(J_sym)
v = _to_numpy_vector(velocity_sym)

U, s, Vt = np.linalg.svd(J, full_matrices=False)
rank, condition_number, threshold = diagnostics_from_svd(...)

if method == "pinv":
    qd_unconstrained = solve_pinv_from_svd(...)
else:
    qd_unconstrained = solve_dls_from_svd(...)

qd, limited = apply_joint_velocity_limits(...)

achieved = J @ qd
residual = v - achieved
residual_norm = norm(residual)
success = residual_norm <= tol

return VelocityIKSolution(...)
```

This pseudocode records the intended data flow rather than prescribing exact implementation syntax.

## 19. Test strategy

### 19.1 `task_jacobian()` tests

Tests should cover:

- `linear`, `angular`, and `twist` presets;
- explicit task subsets;
- preservation of explicit task ordering;
- case-insensitive component names;
- invalid presets;
- invalid component names;
- empty task sequences;
- duplicate components;
- configuration dimension errors;
- symbolic `q=None` behavior;
- substitution of `q`;
- substitution of `parameters`;
- partially symbolic results remaining valid for this function.

### 19.2 `cartesian_velocity()` tests

Tests should verify directly

\[
\dot x=J_{task}(q)\dot q.
\]

At minimum include:

- planar 2R analytical velocity;
- reduced planar tasks;
- full-twist propagation where relevant;
- at least one robot containing a prismatic joint;
- exact SymPy numerical values;
- unresolved-symbol rejection;
- dimension validation for `q` and `qd`.

### 19.3 Pseudoinverse tests

The pseudoinverse solver should be tested for:

- square regular systems;
- redundant systems (`m<n`);
- overdetermined systems (`m>n`);
- rank-deficient systems;
- one-dimensional tasks.

For square nonsingular systems, the solution may be checked against the ordinary inverse as a test oracle even though no public inverse method exists.

For redundant systems, tests should verify the minimum-norm behavior with constructed null-space alternatives.

For overdetermined systems, tests should verify the least-squares normal condition

\[
J^T(J\dot q-\dot x_d)\approx0.
\]

### 19.4 DLS tests

Tests should cover:

- positive damping validation;
- missing damping rejection;
- damping supplied with `pinv` rejection;
- near-singular examples;
- reduced joint-velocity magnitudes relative to an ill-conditioned pseudoinverse example;
- increased regularization behavior for a controlled example with larger damping;
- residual/diagnostic consistency.

### 19.5 Joint-velocity-limit tests

Tests should cover:

- symmetric limits;
- asymmetric limits;
- wrong number of limit entries;
- invalid bound ordering;
- invalid non-finite bounds;
- no-limit behavior;
- `unconstrained_qd` preservation;
- `limited` state;
- recomputation of `achieved_velocity`, residual, and success from the clipped `qd`.

### 19.6 Degenerate and singular tests

Explicitly test:

- zero Jacobian with zero requested velocity;
- zero Jacobian with nonzero requested velocity;
- rank-deficient Jacobian with an achievable desired velocity;
- rank-deficient Jacobian with an unachievable desired velocity;
- `condition_number == inf` when numerical rank is deficient;
- success independent of rank when residual is within tolerance.

### 19.7 `VelocityIKSolution` invariants

Every solver test should be able to verify:

```text
residual == desired_velocity - achieved_velocity
residual_norm == norm(residual)
success == (residual_norm <= tol)
```

with numerical tolerance as appropriate.

Shape invariants for every returned vector should also be checked.

## 20. Examples and acceptance scenarios

At least four user-facing examples should accompany the feature.

### 20.1 Planar 2R forward differential kinematics

Demonstrate:

- symbolic `task_jacobian()`;
- reduced `("vx", "vy")` task;
- numerical `cartesian_velocity()`;
- comparison with analytical planar equations.

### 20.2 Planar 2R near singularity

Demonstrate the same desired task velocity using:

```text
pinv
dls
```

and compare:

- joint-velocity magnitudes;
- residuals;
- condition number.

The example should illustrate that DLS trades exactness for bounded/regularized joint motion near a singularity.

### 20.3 Redundant task

Use a manipulator/task combination with

\[
m<n
\]

and demonstrate that the pseudoinverse returns the minimum-norm solution without adding secondary/null-space objectives.

### 20.4 Joint-velocity saturation

Demonstrate:

```text
unconstrained_qd
qd
limited
achieved_velocity
residual
success
```

before and after explicit limits are applied.

## 21. Compatibility and interaction with other 0.5.0 features

This feature is additive and should not change existing `Robot.J` or `Robot.J_point()` behavior.

It establishes the numerical/task-space layer that later 0.5.0 work can reuse conceptually:

- singularity and manipulability analysis can use task-selected Jacobians and the same geometric-twist ordering;
- numerical inverse kinematics can reuse task-selection conventions and Jacobian diagnostics where appropriate;
- trajectory features may later build resolved-rate workflows on top of `cartesian_velocity()` / `solve_velocity_ik()`, but resolved-rate controllers are outside the present scope.

The differential-kinematics solver should not itself absorb higher-level singularity/manipulability policy. It reports rank and condition number only as diagnostics.

## 22. Explicitly outside Moro 0.5.0

The following remain outside this feature:

- null-space optimization and secondary objectives;
- joint-limit or obstacle avoidance through null-space gradients;
- QP/constrained least-squares velocity IK;
- active-set redistribution after joint-velocity clipping;
- resolved-rate trajectory controllers;
- Cartesian trajectory tracking;
- acceleration-level differential kinematics;
- `Jdot`;
- dynamic control;
- body/tool-frame Jacobians;
- public frame-selection arguments for the Jacobian;
- motion planning.

## 23. Acceptance criteria

The differential-kinematics block is ready for implementation when the following are satisfied:

1. task selection is explicit, ordered, and independent of velocity-vector length;
2. `task_jacobian()` reuses `Robot.J` rather than duplicating geometric Jacobian derivation;
3. symbolic Jacobian inspection remains available;
4. forward velocity propagation preserves SymPy exactness when possible;
5. velocity IK crosses explicitly into finite real NumPy arrays;
6. `pinv` and DLS share one SVD-based diagnostic path;
7. rank deficiency is handled as a solvable numerical condition rather than an exception;
8. success is defined only by final task residual;
9. post-solution joint-velocity clipping is observable and reflected in all final task diagnostics;
10. `VelocityIKSolution` satisfies the documented invariants;
11. representative regular, redundant, overdetermined, singular, near-singular, prismatic, and limited cases are covered by tests;
12. no 0.5.0 implementation introduces null-space control, constrained optimization, `Jdot`, acceleration-level kinematics, or additional Jacobian-frame conventions.
