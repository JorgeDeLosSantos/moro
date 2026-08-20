# Inverse Kinematics

`moro` provides numerical solvers for **position inverse kinematics**.

Given a desired Cartesian position for the end-effector,

[
p_d =
\begin{bmatrix}
x_d \
y_d \
z_d
\end{bmatrix},
]

the inverse-kinematics solver searches for a joint configuration (q) such that the end-effector position is sufficiently close to the target.

The current interface focuses on position only. Full-pose inverse kinematics, including end-effector orientation constraints, is not currently supported.

This section focuses on practical use of the inverse-kinematics API. For the mathematical background of the numerical methods, see **Theory → Inverse Kinematics**.

## Solving a position target

The main function is:

```python
from moro.inverse_kinematics import solve_position_ik
```

Consider a planar 2R robot:

```python
from moro import Robot
from moro.abc import q1, q2

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (1.0, 0, 0, q2, "r"),
)
```

A Cartesian target can be defined as:

```python
target = [1.5, 0.5, 0.0]
```

Then solve the inverse-kinematics problem:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
)
```

The function returns an `IKSolution` object rather than only the final joint vector.

A basic convergence check is:

```python
if solution.converged:
    print(solution.q)
else:
    print(solution.message)
```

The target must contain exactly three finite numeric values:

```text
[x, y, z]
```

and the convergence tolerance is expressed in the same linear units used by the robot geometry and target position.

## Providing numerical model parameters

A robot can contain symbolic geometric parameters.

For example:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Because inverse kinematics is solved numerically, all non-joint symbolic quantities involved in the forward kinematics must have numerical values.

Use the `parameters` argument:

```python
solution = solve_position_ik(
    robot,
    [1.5, 0.5, 0.0],
    q0=[0.1, 0.1],
    parameters={
        l1: 1.0,
        l2: 1.0,
    },
)
```

These substitutions are applied only to the expressions used by the solver.

The original symbolic robot model is not modified:

```python
robot.T
```

remains symbolic after the IK computation.

If unresolved non-joint symbols remain after applying `parameters`, `moro` raises an error instead of attempting to evaluate an incomplete numerical model.

This allows the same symbolic `Robot` instance to be reused with different geometric parameter values.

## Choosing a solver

Three numerical methods are currently available:

```text
"lm"
"newton"
"ccd"
```

The solver is selected with:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    method="lm",
)
```

### Levenberg-Marquardt

Levenberg-Marquardt is the default method:

```python
method="lm"
```

It uses the linear part of the geometric Jacobian together with a damping term.

In simplified form, the joint update is:

[
\Delta q
========

\left(
J^T J + \lambda^2 I
\right)^{-1}
J^T e,
]

where:

[
e = p_d-p(q).
]

The damping parameter is adjusted during the solution process according to whether a proposed step improves the position error.

The initial damping value can be changed with:

```python
damping=1.0
```

and the scaling factor with:

```python
damping_scale=0.5
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    method="lm",
    damping=0.5,
    damping_scale=0.5,
)
```

Levenberg-Marquardt is generally a useful first choice for numerical position IK because damping improves behavior near poorly conditioned configurations.

### Newton-Raphson

The Newton method can be selected with:

```python
method="newton"
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    method="newton",
)
```

The method uses the end-effector position Jacobian and computes an update based on its inverse or pseudoinverse.

Conceptually:

[
\Delta q
========

J^\dagger e.
]

Newton-type methods can converge rapidly when the initial guess is appropriate, but their behavior may be more sensitive to singularities, poor initial guesses, or joint constraints.

### Cyclic Coordinate Descent

CCD can be selected with:

```python
method="ccd"
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    method="ccd",
)
```

CCD does not construct or invert a Jacobian matrix.

Instead, it updates one joint at a time, starting from the joint closest to the end-effector and moving toward the base.

For revolute joints, the algorithm rotates the corresponding joint to move the end-effector toward the target in the plane perpendicular to the joint axis.

For prismatic joints, the update moves the joint along its translation axis.

A single CCD iteration corresponds to one complete sweep through all robot joints from joint `n` back to joint `1`.

CCD can be useful when a Jacobian-free iterative approach is preferred, although its convergence is generally linear.

## Initial guess and joint limits

Inverse kinematics is generally not unique, and numerical methods depend on their starting configuration.

The initial guess can be supplied with:

```python
q0=[...]
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.2, -0.1],
)
```

The number of values in `q0` must match:

```python
robot.dof
```

### Random initialization

If `q0=None`, `moro` generates a random initial configuration inside the active joint limits:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=None,
)
```

A reproducible random initial guess can be obtained with:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=None,
    random_state=42,
)
```

Using the same integer seed produces the same initial random configuration without changing NumPy's global random state.

For repeatable examples and tests, providing an explicit `q0` is usually preferable.

### Joint limits

By default, the solver uses:

```python
robot.joint_limits
```

For example:

```python
robot.joint_limits = [
    (-1.5, 1.5),
    (-2.0, 2.0),
]
```

These limits are then used automatically:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
)
```

Limits can also be overridden for a specific solve:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    joint_limits=[
        (-1.0, 1.0),
        (-1.5, 1.5),
    ],
)
```

The temporary limits do not modify the `Robot` object.

Every joint update is clipped to the active limits.

If an explicitly provided `q0` lies outside them, the initial configuration is clipped before the iterative process begins.

Because the default limits stored by `Robot` are convenience values rather than physical constraints, actual robot limits should be defined before solving constrained IK problems.

## Convergence and stagnation

The solver is considered converged when:

[
\left|
p_d-p(q)
\right|
<
\mathrm{tol}.
]

The default tolerance is:

```python
tol=1e-6
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    tol=1e-8,
)
```

### Maximum iterations

The maximum number of iterations can be controlled with:

```python
max_iter=...
```

If it is not specified, the defaults are:

```text
Newton-Raphson       100
Levenberg-Marquardt  100
CCD                   500
```

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
    method="ccd",
    max_iter=1000,
)
```

### Stagnation detection

A numerical method can stop making useful progress before reaching the maximum number of iterations.

`moro` detects two forms of stagnation.

The first uses:

```python
step_tol
```

to detect joint updates that have become too small.

The second uses:

```python
error_change_tol
```

to detect cases where the position error stops improving.

The number of consecutive stalled iterations required before termination is controlled by:

```python
stagnation_iterations
```

The defaults are:

```python
step_tol=1e-12
error_change_tol=1e-12
stagnation_iterations=5
```

These options normally do not need to be changed for basic use, but they can be useful when diagnosing difficult IK problems.

## Inspecting an IK solution

`solve_position_ik()` returns an `IKSolution`.

The main fields are:

```python
solution.q
solution.converged
solution.iterations
solution.error
solution.method
solution.residual
solution.message
```

### Joint configuration

The final solver state is:

```python
solution.q
```

For a two-joint robot this may look like:

```text
[0.98, -1.32]
```

### Convergence status

Always inspect:

```python
solution.converged
```

before treating the returned joint vector as a valid IK solution.

For example:

```python
if solution.converged:
    q_solution = solution.q
```

### Final error

The final Cartesian error norm is:

```python
solution.error
```

and corresponds to:

[
\left|
p_d-p(q)
\right|.
]

### Residual

The complete Cartesian residual is available through:

```python
solution.residual
```

and is defined as:

[
r =
p_d-p(q).
]

When available, it contains three values:

```text
[rx, ry, rz]
```

and:

[
\text{solution.error}
=====================

|\text{solution.residual}|.
]

If a numerical failure prevents a finite residual from being evaluated, `residual` can be `None` and `error` is reported as infinity.

### Iterations

The number of completed global solver steps is:

```python
solution.iterations
```

For Newton and LM, one iteration corresponds to one attempted global joint update.

For CCD, one iteration corresponds to one complete sweep through all joints.

If the initial guess already satisfies the requested tolerance:

```python
solution.iterations == 0
```

### Outcome message

A short description of the solver termination condition is stored in:

```python
solution.message
```

Possible outcomes include successful convergence, maximum iterations, stagnation, or a numerical failure.

This makes the following pattern useful:

```python
if solution.converged:
    print("Solution:", solution.q)
else:
    print("IK failed:", solution.message)
```

## Solving a position trajectory

`moro` can also solve a sequence of Cartesian position targets.

Use:

```python
from moro.inverse_kinematics import solve_position_trajectory
```

Consider:

```python
targets = [
    [1.5, 0.2, 0.0],
    [1.4, 0.4, 0.0],
    [1.2, 0.6, 0.0],
]
```

A trajectory can be solved with:

```python
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=[0.1, 0.1],
)
```

Unlike `solve_position_ik()`, the trajectory function requires an explicit `q0`.

The target sequence must contain one 3D position per row:

```text
(m, 3)
```

A single three-element vector is not accepted by this function; use `solve_position_ik()` for one target.

### Sequential initialization

Targets are processed in order.

The initial configuration is used for the first target:

```text
target 0
    ↑
   q0
```

If that target converges, its solution is reused as the initial guess for the next target:

```text
target 0 → q0
              ↓
target 1 → q solution from target 0
              ↓
target 2 → q solution from target 1
              ↓
...
```

This usually improves local continuity and convergence when neighboring Cartesian targets are close to each other.

For symbolic models, parameters are passed in the same way as for a single IK problem:

```python
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=[0.1, 0.1],
    parameters={
        l1: 1.0,
        l2: 1.0,
    },
)
```

The same solver options can also be used:

```python
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=[0.1, 0.1],
    method="lm",
    tol=1e-8,
)
```

## Inspecting a trajectory solution

`solve_position_trajectory()` returns an `IKTrajectorySolution`.

The main fields are:

```python
trajectory.solutions
trajectory.converged
trajectory.failed_index
trajectory.message
```

It also provides convenience properties:

```python
trajectory.qs
trajectory.errors
trajectory.iterations
```

### Individual solutions

Each processed target produces an `IKSolution` stored in:

```python
trajectory.solutions
```

For example:

```python
for solution in trajectory.solutions:
    print(solution.q, solution.error)
```

### Joint configurations

All processed joint configurations are available directly through:

```python
trajectory.qs
```

For example:

```python
[
    [q11, q12],
    [q21, q22],
    [q31, q32],
]
```

These configurations can be used directly by workflows such as robot animation.

### Errors and iteration counts

The per-target final errors are:

```python
trajectory.errors
```

and the iteration counts are:

```python
trajectory.iterations
```

### Failed target

If every target converges:

```python
trajectory.converged
# True

trajectory.failed_index
# None
```

If a target fails, the trajectory stops at that point.

The index of the failed target is:

```python
trajectory.failed_index
```

and the corresponding failed `IKSolution` is included as the last element in:

```python
trajectory.solutions
```

For example:

```python
if not trajectory.converged:
    i = trajectory.failed_index
    print("Failed target:", i)
    print(trajectory.solutions[i].message)
```

## Handling unsuccessful solutions

Numerical inverse kinematics is not guaranteed to converge.

A failure can occur because of:

* an unreachable target;
* restrictive joint limits;
* an unsuitable initial guess;
* a singular or poorly conditioned configuration;
* solver stagnation;
* unresolved symbolic parameters;
* invalid numerical inputs;
* numerical failures during forward-kinematics or Jacobian evaluation.

A returned `IKSolution` with:

```python
solution.converged == False
```

should therefore not be interpreted as a successful solution simply because `solution.q` contains finite joint values.

The recommended pattern is:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
)

if solution.converged:
    print("q =", solution.q)
else:
    print(solution.message)
    print("Final error:", solution.error)
```

For difficult problems, useful actions include:

* trying a different initial guess;
* checking whether the target is inside the workspace;
* checking physical joint limits;
* increasing `max_iter`;
* trying another solver;
* verifying that all geometric parameters have numerical values.

Changing tolerances should be done with care. A looser tolerance may report convergence farther from the requested target.

## Validating a solution with forward kinematics

An inverse-kinematics solution can be checked independently using forward kinematics.

Suppose:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.1, 0.1],
)
```

If the solution converged, build a substitution dictionary:

```python
values = dict(zip(robot.qs, solution.q))
```

Then evaluate the end-effector position:

```python
p = robot.T[:3, 3].subs(values)
```

For a symbolic robot, also include its geometric parameters:

```python
values.update({
    l1: 1.0,
    l2: 1.0,
})

p = robot.T[:3, 3].subs(values).evalf()
```

The resulting position should be close to the requested target.

This is also a useful way to verify the meaning of:

```python
solution.residual
```

which represents:

[
p_d-p(q).
]

## A worked example

Consider a symbolic planar 2R robot:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2
from moro.inverse_kinematics import solve_position_ik

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Set physical joint limits:

```python
from sympy import pi

robot.joint_limits = [
    (-pi, pi),
    (-pi, pi),
]
```

Define a target:

```python
target = [1.2, 0.8, 0.0]
```

and numerical geometric parameters:

```python
parameters = {
    l1: 1.0,
    l2: 1.0,
}
```

Now solve the problem using Levenberg-Marquardt:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.2, 0.2],
    parameters=parameters,
    method="lm",
    tol=1e-8,
)
```

Inspect the result:

```python
print(solution.converged)
print(solution.q)
print(solution.error)
print(solution.message)
```

If convergence was successful, validate the joint configuration:

```python
values = dict(zip(robot.qs, solution.q))
values.update(parameters)

p_solution = robot.T[:3, 3].subs(values).evalf()
p_solution
```

The evaluated position should agree with:

```python
target
```

within the requested tolerance.

The same robot can now be used to solve several nearby targets:

```python
from moro.inverse_kinematics import solve_position_trajectory

targets = [
    [1.2, 0.8, 0.0],
    [1.1, 0.9, 0.0],
    [1.0, 1.0, 0.0],
    [0.9, 1.1, 0.0],
]
```

```python
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=solution.q,
    parameters=parameters,
    method="lm",
    tol=1e-8,
)
```

Check the global result:

```python
trajectory.converged
```

and inspect the joint sequence:

```python
trajectory.qs
```

These configurations can then be passed to the visualization tools to animate the resulting robot motion.

## Notes and limitations

The inverse-kinematics capabilities in the current version of `moro` are intentionally focused on numerical position IK for serial manipulators.

Keep the following points in mind:

* only Cartesian position targets ([x,y,z]) are currently supported;
* orientation constraints are not part of the IK objective;
* the solver may return different valid configurations for the same target;
* the result can depend strongly on the initial guess;
* joint limits are enforced during the numerical updates;
* symbolic non-joint parameters must be assigned through `parameters`;
* Levenberg-Marquardt is the default solver;
* Newton and LM use the linear part of the geometric Jacobian;
* CCD is Jacobian-free and supports both revolute and prismatic joints;
* a non-converged result should always be inspected through `converged`, `error`, and `message`;
* `solve_position_trajectory()` solves an existing sequence of Cartesian targets but does not generate that trajectory.

In particular, trajectory IK does not currently perform:

* Cartesian interpolation;
* timing or velocity assignment;
* trajectory smoothing;
* collision avoidance;
* motion planning;
* optimization over IK branches;
* guaranteed global branch continuity.

Reusing each converged configuration as the next initial guess often produces locally continuous results, but this is not a formal guarantee of globally continuous joint motion.

## See also

* **Forward Kinematics** — evaluate and validate the Cartesian position produced by a joint configuration.
* **Jacobians** — compute the geometric Jacobian used by Newton and Levenberg-Marquardt solvers.
* **Robot Modeling** — configure joint types and joint limits.
* **Visualization** — plot and animate configurations obtained from inverse kinematics.
* **Theory → Inverse Kinematics** — mathematical background for the numerical IK methods.
* **Theory → Differential Kinematics** — Jacobians and velocity relationships used by Jacobian-based IK.
* **API Reference → Inverse Kinematics** — complete signatures and result-object definitions for `solve_position_ik()`, `solve_position_trajectory()`, `IKSolution`, and `IKTrajectorySolution`.
