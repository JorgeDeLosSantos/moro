# Detailed design: singularities and manipulability for Moro 0.5.0

This document records the detailed design for the singularity and manipulability analysis planned for Moro 0.5.0.

The feature remains part of `moro.differential_kinematics` and builds directly on the task-space Jacobian, evaluation, and SVD infrastructure defined in `docs/design/0.5.0/differential-kinematics.md`.

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

The analysis is numerical and task dependent.

All metrics are evaluated from the selected task Jacobian

\[
J_{task}(q)\in\mathbb{R}^{m\times n}.
\]

The task conventions are exactly those defined for differential kinematics:

```text
vx, vy, vz, wx, wy, wz
```

with the presets:

```text
linear
angular
twist
```

and ordered explicit subsets such as:

```python
("vx", "vy")
("vx", "vy", "wz")
("wz", "vx")
```

The feature must reuse the same numerical Jacobian evaluation path and SVD logic as velocity-level inverse kinematics. Separate rank or singular-value policies must not be introduced.

The analysis functions diagnose the local kinematic properties of a configuration. They do not modify the robot model, change configuration state, solve symbolic singularity conditions, or perform global workspace analysis.

## 2. Public API

The public API is:

```python
singular_values(
    robot,
    q,
    *,
    task="twist",
    parameters=None,
)
```

```python
jacobian_rank(
    robot,
    q,
    *,
    task="twist",
    parameters=None,
    tol=None,
)
```

```python
condition_number(
    robot,
    q,
    *,
    task="twist",
    parameters=None,
)
```

```python
is_singular(
    robot,
    q,
    *,
    task="twist",
    parameters=None,
    tol=None,
)
```

```python
manipulability(
    robot,
    q,
    *,
    task,
    parameters=None,
)
```

`task` is intentionally required for `manipulability()`.

No public aggregate `JacobianAnalysis` result object is planned for 0.5.0.

## 3. Shared numerical foundation

All analysis is based on the economy-size SVD

\[
J=U\Sigma V^T.
\]

Conceptually:

```python
U, s, Vt = np.linalg.svd(J, full_matrices=False)
```

where

\[
s=(\sigma_1,\sigma_2,\ldots,\sigma_r),
\qquad
r=\min(m,n),
\]

and

\[
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r\ge0.
\]

The same SVD infrastructure must be reused by:

- `singular_values()`;
- `jacobian_rank()`;
- `condition_number()`;
- `is_singular()`;
- `manipulability()`;
- `solve_velocity_ik()`.

This avoids inconsistent classification caused by independent default tolerances or multiple numerical paths.

## 4. Numerical Jacobian evaluation

These functions are numerical even though the robot model may be symbolic.

The conceptual evaluation path is:

```text
robot.J
    -> task selection
    -> substitute q
    -> substitute parameters
    -> reject unresolved symbols
    -> reject non-real/non-finite values
    -> convert to NumPy
    -> SVD analysis
```

`q` must contain exactly `robot.dof` entries in the order of `robot.qs`.

`parameters` resolves additional symbolic model quantities such as link lengths.

Before NumPy analysis, the selected task Jacobian must be fully numerical, real, and finite.

Values containing

```text
nan
+inf
-inf
complex entries
```

must be rejected.

## 5. Shared internal architecture

The detailed implementation may use two private levels.

First, a numerical evaluator such as:

```python
_evaluate_task_jacobian_numeric(
    robot,
    q,
    *,
    task,
    parameters=None,
)
```

which owns symbolic substitution, validation, and conversion to NumPy.

Second, an SVD helper such as:

```python
_svd_analysis(J_num, *, rank_tol=None)
```

which owns:

- SVD computation;
- singular values;
- rank threshold;
- numerical rank;
- condition number.

A private dataclass or similarly structured internal result is acceptable if it simplifies reuse within one public call.

The exact private names may evolve, but there should be one shared numerical policy.

No cross-call cache is required in 0.5.0.

## 6. Rank threshold

### 6.1 Automatic threshold

When the caller does not provide an explicit tolerance,

```python
tol=None
```

rank classification should use the same scale-aware policy adopted by velocity IK:

\[
\tau=\max(m,n)\,\epsilon\,\sigma_{max},
\]

where `epsilon` is machine precision for the numerical dtype.

The numerical rank is then

\[
\operatorname{rank}(J)
=
\#\{\sigma_i>\tau\}.
\]

### 6.2 Explicit rank tolerance

When `tol` is supplied to `jacobian_rank()` or `is_singular()`, it is interpreted as an absolute singular-value threshold:

\[
\tau=tol.
\]

A value contributes to rank only when

\[
\sigma_i>tol.
\]

This gives the user direct control over effective rank classification.

### 6.3 Tolerance validation

An explicit `tol` must be:

- scalar;
- numeric;
- real;
- finite;
- strictly positive.

`tol=0` is not accepted because exact zero classification is not a robust numerical rank policy.

## 7. `singular_values()`

### 7.1 Meaning

`singular_values()` exposes the compact singular-value spectrum of the selected task Jacobian.

For

\[
J\in\mathbb{R}^{m\times n},
\]

it returns exactly

\[
\min(m,n)
\]

singular values.

No additional zeros are appended to match either `m` or `n`.

### 7.2 Return type

The public return value is a SymPy column matrix:

```python
Matrix([
    sigma_1,
    sigma_2,
    ...,
    sigma_r,
])
```

The entries are numerical floating-point values obtained from NumPy SVD.

The order is nonincreasing.

### 7.3 Validation

Returned values must be finite and nonnegative.

Unexpected non-finite SVD output should be treated as a numerical failure rather than propagated silently.

No tolerance argument is exposed because this function returns the raw SVD data rather than a rank classification.

## 8. `jacobian_rank()`

`jacobian_rank()` returns an `int`.

Its rank is always the numerical rank of the selected task Jacobian, not necessarily the full six-row geometric Jacobian.

The result satisfies:

\[
0\le\operatorname{rank}(J)\le\min(m,n).
\]

The function supports `tol=None` for automatic classification and an explicit positive threshold for user-controlled classification.

## 9. `is_singular()`

### 9.1 Definition

For the selected task, a configuration is classified as singular when

\[
\operatorname{rank}(J)<\min(m,n).
\]

Therefore:

```python
is_singular(...)
```

is conceptually equivalent to comparing `jacobian_rank()` against the maximum attainable rank for the matrix shape under the same tolerance policy.

### 9.2 Task dependence

Singularity is always task dependent.

A configuration may be singular for

```python
task="twist"
```

and nonsingular for

```python
task=("vx", "vy")
```

because the corresponding Jacobians have different rows and potentially different rank.

This behavior is intentional and should be demonstrated explicitly in examples and tests.

### 9.3 Return type

The result is always a Python `bool`.

Because this API is numerical, unresolved symbolic models should raise during evaluation rather than return an indeterminate third state such as `None`.

### 9.4 Singularity is not an exception

A singular Jacobian is a valid result of analysis. `is_singular()` must not raise merely because rank is deficient.

## 10. `condition_number()`

### 10.1 Definition

For a numerically full-rank selected Jacobian,

\[
\kappa(J)=\frac{\sigma_{max}}{\sigma_{min}}.
\]

Full rank here means

\[
\operatorname{rank}(J)=\min(m,n)
\]

under the automatic SVD threshold.

### 10.2 Rank-deficient behavior

If

\[
\operatorname{rank}(J)<\min(m,n),
\]

then

```python
condition_number = inf
```

rather than raising or returning `nan`.

The zero Jacobian therefore produces:

```text
rank = 0
condition_number = inf
```

### 10.3 Rectangular Jacobians

Condition number is defined from the compact singular-value spectrum.

For `m<n`, a full-row-rank Jacobian has finite condition number.

For `m>n`, a full-column-rank Jacobian also has finite condition number.

The relevant maximum rank is always

\[
\min(m,n).
\]

### 10.4 No public `tol`

`condition_number()` does not expose a public rank tolerance.

It uses the automatic internal threshold.

This keeps condition number as a standard numerical diagnostic while `jacobian_rank(..., tol=...)` and `is_singular(..., tol=...)` remain the explicit policy-controlled classification functions.

As a result, it is possible for

```python
is_singular(..., tol=1e-4)
```

to return `True` while

```python
condition_number(...)
```

returns a large but finite value under the automatic threshold.

This is acceptable and should be documented.

## 11. One-dimensional tasks

Tasks with

\[
m=1
\]

require no special algorithmic branch.

For a nonzero one-row Jacobian there is one singular value

\[
\sigma_1.
\]

If it exceeds the active rank threshold:

\[
rank=1,
\qquad
is\_singular=False,
\]

and

\[
\kappa=\frac{\sigma_1}{\sigma_1}=1.
\]

This is mathematically correct because a one-dimensional task has no internal anisotropy.

The absolute local capability is instead reflected by `singular_values()` and `manipulability()`.

If the only singular value falls below the threshold:

```text
rank = 0
is_singular = True
condition_number = inf
```

## 12. `manipulability()`

### 12.1 Metric

The initial metric is Yoshikawa velocity manipulability:

\[
w(q)=\sqrt{\det(JJ^T)}.
\]

### 12.2 Required task

Unlike the other analysis functions, `task` has no default and must be supplied explicitly.

This is intentional because manipulability depends strongly on the selected task.

For example:

```python
manipulability(robot, q, task=("vx", "vy"))
```

measures a planar translational capability, while

```python
manipulability(robot, q, task="angular")
```

measures angular capability.

A full `"twist"` task is mathematically valid but mixes translational and angular scales.

### 12.3 SVD implementation

The implementation should not form a determinant explicitly when the singular values are already available.

For `m<=n`,

\[
w=\prod_{i=1}^{m}\sigma_i.
\]

This is equivalent to

\[
\sqrt{\det(JJ^T)}
\]

for a full-row-rank task Jacobian and is numerically more direct.

### 12.4 Rank-deficient behavior

If `m<=n` and the Jacobian is rank deficient under the automatic threshold, return exactly:

```python
0.0
```

rather than a tiny roundoff-dependent positive value.

Therefore, in automatic mode:

\[
is\_singular=True
\Rightarrow
w=0
\]

for tasks with `m<=n`.

### 12.5 Tasks with `m>n`

When

\[
m>n,
\]

`JJ^T` is necessarily singular and the `m`-dimensional Yoshikawa volume is zero.

Therefore:

```python
manipulability(...) == 0.0
```

for such tasks.

This does not imply that `is_singular()` must be `True`.

A matrix with `m>n` may have full column rank:

\[
rank=n=\min(m,n),
\]

which makes `is_singular()` return `False` under the adopted rank-loss definition even though the `m`-dimensional manipulability volume is zero.

This distinction must be documented explicitly.

### 12.6 Return type

`manipulability()` returns a Python `float`.

No public tolerance argument is exposed in 0.5.0.

The automatic SVD threshold is used consistently for numerical rank loss.

## 13. Units and scale

Manipulability is sensitive to scaling.

For tasks that mix linear and angular Jacobian rows, the numerical value depends on the relative units/scales of translation and rotation.

For robots with mixed revolute and prismatic joints, column scaling likewise reflects the chosen generalized-coordinate units.

Moro 0.5.0 does not attempt to normalize these effects automatically.

The library should document that:

- reduced translational tasks are often easier to interpret physically;
- angular tasks have different units from translational tasks;
- full-twist manipulability mixes scales;
- no characteristic length or automatic weighting is applied.

This is a property of the chosen metric, not an implementation error.

## 14. Relationship between rank, conditioning, and manipulability

These metrics answer related but distinct questions.

### 14.1 Rank / singularity

Rank asks whether directions of local motion capability have been lost relative to the maximum rank allowed by matrix shape.

### 14.2 Condition number

Condition number measures anisotropy/sensitivity:

\[
\kappa=\frac{\sigma_{max}}{\sigma_{min}}.
\]

A large value indicates that some task directions are much harder to generate than others.

### 14.3 Manipulability

Manipulability measures a velocity-volume scale through the product of singular values.

A small manipulability can result from small absolute singular values even when the condition number is near one.

### 14.4 Important non-equivalences

The implementation and documentation must not imply:

```text
condition number large <=> singular
manipulability zero <=> is_singular
```

The first fails for near-singular but still full-rank matrices.

The second fails for `m>n`, where the Yoshikawa `m`-dimensional volume is necessarily zero even for full column rank.

## 15. Degenerate zero Jacobian

For

\[
J=0,
\]

the expected analysis is:

```text
singular_values = zeros(min(m, n))
rank = 0
condition_number = inf
is_singular = True
manipulability = 0.0
```

for all nonempty tasks where `min(m,n)>0`.

## 16. Public return types

The public return types are intentionally simple:

```text
singular_values()   -> sympy.Matrix
jacobian_rank()     -> int
condition_number()  -> float
is_singular()       -> bool
manipulability()    -> float
```

No public analysis dataclass is introduced in 0.5.0.

This keeps the API compact and functional.

A combined analysis object can be considered later if repeated SVD work across several user calls becomes a demonstrated problem.

## 17. Interaction with velocity IK

The singularity/manipulability feature and velocity IK share the same numerical Jacobian and SVD policy.

Within one `solve_velocity_ik()` call, the solver should compute SVD only once and derive:

- rank;
- condition number;
- pseudoinverse or DLS operator;
- solver diagnostics.

The public analysis functions may independently recompute SVD when called separately.

No global/cross-call caching is planned.

This design prioritizes consistency and simplicity over premature caching complexity.

## 18. Test strategy

### 18.1 Singular-value tests

Cover:

- descending order;
- length `min(m,n)`;
- nonnegative values;
- SymPy column-matrix return type;
- regular square Jacobians;
- rectangular Jacobians;
- reduced tasks;
- numerical evaluation of symbolic robot parameters;
- rejection of unresolved symbols;
- rejection of non-finite or complex numerical data.

### 18.2 Rank and singularity tests

Cover:

- regular configurations;
- exact singularities;
- near-singular configurations;
- automatic threshold;
- explicit `tol` changing classification;
- invalid tolerance values;
- one-dimensional tasks;
- `m<n` Jacobians;
- `m>n` Jacobians;
- zero Jacobian;
- task-dependent singularity classification.

An important acceptance scenario is a single configuration where:

```python
is_singular(robot, q, task="twist")
```

and

```python
is_singular(robot, q, task=("vx", "vy"))
```

produce different valid results.

### 18.3 Condition-number tests

Cover:

- well-conditioned matrices;
- large but finite condition number near singularity;
- exact singularity -> `inf`;
- zero Jacobian -> `inf`;
- regular one-dimensional task -> `1.0`;
- full-rank rectangular matrices -> finite result;
- growth of condition number when approaching a known singularity.

### 18.4 Manipulability tests

The central analytical test should use a planar 2R manipulator with task

```python
("vx", "vy")
```

for which

\[
w=|l_1 l_2\sin q_2|.
\]

Tests should include:

\[
q_2=0
\Rightarrow
w=0,
\]

and

\[
q_2=\frac{\pi}{2}
\Rightarrow
w=l_1l_2.
\]

Also cover:

- one-dimensional tasks;
- rank-deficient tasks;
- `m>n` -> `0.0`;
- revolute/prismatic combinations;
- symbolic parameters resolved numerically.

### 18.5 Cross-metric invariants

Under automatic thresholding:

\[
rank<\min(m,n)
\Rightarrow
is\_singular=True.
\]

Also:

\[
is\_singular(...,tol=None)=True
\Rightarrow
condition\_number=\infty.
\]

For `m<=n`:

\[
rank< m
\Rightarrow
manipulability=0.
\]

No stronger global equivalence between the metrics should be asserted.

## 19. Documentation examples

At least four user-facing examples should accompany the implementation.

### 19.1 Planar 2R singularity analysis

Evaluate several configurations and display:

```text
singular values
rank
condition number
is_singular
```

The example should include a regular configuration and a fully extended/folded singular posture.

### 19.2 Planar 2R manipulability

Sweep `q2` for task

```python
("vx", "vy")
```

and compare numerical results against

\[
|l_1l_2\sin q_2|.
\]

### 19.3 Task-dependent singularity

Compare the same configuration under a full or larger task and under a reduced task.

The goal is to show that singularity is a property of the selected local task mapping, not an absolute label attached to a robot configuration independently of task.

### 19.4 Condition number versus manipulability

Show a controlled case where these metrics convey different information.

A one-dimensional task is particularly useful because a regular 1D task has

\[
\kappa=1
\]

while its manipulability equals the absolute singular-value magnitude and can be arbitrarily small.

## 20. Relationship with workspace

Workspace sampling remains decoupled from singularity/manipulability analysis.

Workspace result objects should not automatically store singularity labels, condition numbers, or manipulability values.

If workspace sampling stores configurations, users can evaluate these local metrics later for selected samples.

This avoids coupling global sampling APIs to a particular local metric or task definition.

## 21. Explicitly outside Moro 0.5.0

The following are outside the present scope:

- symbolic singularity equations;
- symbolic solving/classification of singular configurations;
- manipulability ellipsoids;
- singular vectors as a dedicated public API;
- isotropy indices as separate public functions;
- characteristic-length normalization;
- automatic translational/rotational weighting;
- manipulability gradients;
- manipulability optimization;
- null-space manipulability maximization;
- force manipulability;
- dynamic manipulability;
- global manipulability maps;
- automatic attachment of metrics to workspace samples;
- cross-call SVD caching.

## 22. Acceptance criteria

The singularities/manipulability block is ready for implementation when:

1. all analysis uses the selected task Jacobian and existing task conventions;
2. all public functions share one numerical Jacobian/SVD policy with velocity IK;
3. automatic rank uses a scale-aware SVD threshold;
4. `jacobian_rank()` and `is_singular()` optionally accept an explicit positive absolute singular-value threshold;
5. `singular_values()` exposes the compact descending singular-value spectrum;
6. `condition_number()` returns `inf` for automatically rank-deficient Jacobians and remains finite for full-rank rectangular Jacobians;
7. `is_singular()` diagnoses rank loss rather than treating singularity as an exception;
8. `manipulability()` requires an explicit task and uses the Yoshikawa velocity metric;
9. manipulability is computed from singular values rather than an explicit determinant;
10. `m>n` manipulability returns `0.0` without forcing `is_singular=True`;
11. scale/unit limitations are documented explicitly;
12. representative regular, singular, near-singular, reduced-task, rectangular, 1D, revolute/prismatic, and symbolic-parameter cases are covered by tests;
13. no 0.5.0 implementation introduces symbolic singularity solving, manipulability ellipsoids, gradients, optimization, force/dynamic manipulability, or automatic normalization.
