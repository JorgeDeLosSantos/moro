# Inverse kinematics

Inverse kinematics determines the joint configuration required to place a robot at a desired Cartesian target.

For a serial manipulator with configuration

$$
\vec q=
\begin{bmatrix}
q_1 &
q_2 &
\cdots &
q_n
\end{bmatrix}^{T},
$$

forward kinematics provides the Cartesian position of a point $P$ attached to the robot:

$$
\boxed{
\vec r_P^{\,0}=f_P(\vec q).
}
$$

The inverse problem asks for a configuration $\vec q$ that produces a desired position

$$
\vec r_d^{\,0}.
$$

The conventions used in this section follow [Mathematical notation and conventions](notation.md), [Forward kinematics](forward-kinematics.md), and [Differential kinematics](differential-kinematics.md).

## Position inverse kinematics in Moro

The general inverse-kinematics problem may involve both position and orientation:

$$
T(\vec q)=T_d.
$$

Moro currently focuses on **numerical position inverse kinematics**.

The problem solved is therefore

$$
\boxed{
\text{Given }\vec r_d^{\,0},
\text{ find }\vec q\in\mathcal Q
\text{ such that }
\vec r_P^{\,0}(\vec q)\approx\vec r_d^{\,0}.
}
$$

For the current public position IK solver, $P$ corresponds to the origin of the terminal frame.

Thus,

$$
\boxed{
P=O_n.
}
$$

Orientation constraints are not currently part of the position IK problem solved by Moro.

```{important}
Moro's current inverse-kinematics solver addresses Cartesian **position**, not full pose.

The orientation of the terminal frame is therefore not constrained by the desired target.
```

## Position error

At iteration $k$, let

$$
\vec q_k
$$

be the current joint configuration.

The Cartesian residual is defined as

$$
\boxed{
\vec e_k
=
\vec r_d^{\,0}
-
\vec r_{O_n}^{\,0}(\vec q_k).
}
$$

Its Euclidean norm is

$$
\boxed{
e_k
=
\|\vec e_k\|_2.
}
$$

A numerical solution is considered converged when

$$
\boxed{
e_k<\texttt{tol}.
}
$$

The tolerance is expressed in the same linear units used for the robot geometry and the target position.

## Multiple inverse-kinematics solutions

Unlike forward kinematics, inverse kinematics does not generally define a unique mapping from Cartesian space to joint space.

A target may have

$$
\boxed{
0,\quad 1,\quad \text{multiple, or infinitely many solutions}.
}
$$

For example, a planar 2R manipulator can often reach the same Cartesian point using two different configurations, commonly described as *elbow-up* and *elbow-down*.

Therefore,

$$
\vec r_d
\not\Rightarrow
\text{a unique }\vec q.
$$

This non-uniqueness is one of the reasons why numerical IK depends on the initial configuration used by the solver.

## Numerical inverse kinematics

Moro provides three numerical methods:

- Newton-type iteration using the Jacobian pseudoinverse,
- Levenberg--Marquardt,
- Cyclic Coordinate Descent (CCD).

The first two methods use the linear part of the geometric Jacobian,

$$
\boxed{
J_p
=
J[:3,:].
}
$$

For position IK,

$$
\vec v_{O_n}^{\,0}
=
J_p(\vec q)\dot{\vec q}.
$$

For a small joint displacement,

$$
\Delta\vec r
\approx
J_p(\vec q)\Delta\vec q.
$$

The numerical methods use this local relationship to construct joint updates that reduce the position error.

## Initial configuration

Iterative IK requires an initial configuration

$$
\boxed{
\vec q_0.
}
$$

The solver then generates a sequence

$$
\vec q_0,\vec q_1,\vec q_2,\ldots
$$

with the goal of reducing

$$
\|\vec e(\vec q)\|.
$$

The initial configuration can influence both:

- whether the solver converges,
- which inverse-kinematics solution is found.

If multiple joint configurations produce the same Cartesian position, different initial configurations may lead to different solution branches.

```{note}
The initial guess is an algorithmic starting point.

It is not a preferred posture or a secondary optimization objective. The solver is free to move away from $\vec q_0$ while reducing the Cartesian error.
```

### Automatic initialization

If no initial configuration is provided, Moro generates one randomly within the joint limits.

A reproducible initialization can be obtained through `random_state`.

Integer seeds use a local NumPy random generator and do not modify NumPy's global random state.

## Joint limits

If joint limits are defined,

$$
q_i^{\min}\leq q_i\leq q_i^{\max},
$$

the admissible configuration space becomes

$$
\boxed{
\mathcal Q_{\mathrm{adm}}
=
\left\{
\vec q:
q_i^{\min}\leq q_i\leq q_i^{\max}
\right\}.
}
$$

A Cartesian target may therefore be reachable by the unconstrained robot model while being unreachable within the allowed joint ranges.

Moro enforces limits by clipping joint updates:

$$
\boxed{
q_i
\leftarrow
\min
\left(
q_i^{\max},
\max(q_i^{\min},q_i)
\right).
}
$$

For Jacobian-based methods, clipping is applied to the complete trial configuration.

For CCD, clipping is applied immediately after each individual joint update.

If a user-provided $\vec q_0$ lies outside the joint limits, Moro clips it to the admissible range before the iterative process begins.

```{note}
Joint limits are constraints, not optimization objectives.

The current solver does not explicitly attempt to stay near the middle of the joint range or maximize the distance from joint limits.
```

Clipping can modify the update originally proposed by the numerical method. Consequently, a solver may stagnate at the boundary of the admissible configuration space even while the Cartesian error remains above the requested tolerance.

## Newton method

For `method="newton"`, Moro computes a joint update from

$$
J_p(\vec q_k)\Delta\vec q_k
\approx
\vec e_k.
$$

In the general case, the update is obtained using the Moore--Penrose pseudoinverse:

$$
\boxed{
\Delta\vec q_k
=
J_p^\dagger(\vec q_k)\vec e_k.
}
$$

Therefore,

$$
\boxed{
\vec q_{k+1}
=
\vec q_k
+
J_p^\dagger(\vec q_k)\vec e_k.
}
$$

When the robot has three degrees of freedom, Moro first attempts to solve the square linear system directly:

$$
J_p\Delta\vec q=\vec e.
$$

If that system is singular, the implementation falls back to the pseudoinverse. For other Jacobian shapes, the pseudoinverse is used directly.

The resulting trial configuration is then projected onto the joint limits.

Newton-type updates can converge rapidly near a suitable solution, but their behavior depends on the local Jacobian and the initial configuration.

Near singularities or poorly conditioned configurations, the pseudoinverse may generate large or unstable joint updates.

## Levenberg--Marquardt

Levenberg--Marquardt is the default position IK method in Moro.

For `method="lm"`, the update is

$$
\boxed{
\Delta\vec q_k
=
\left(
J_p^TJ_p+\lambda_k^2I
\right)^{-1}
J_p^T\vec e_k.
}
$$

The trial configuration is

$$
\vec q_{\mathrm{trial}}
=
\vec q_k+\Delta\vec q_k,
$$

followed by projection onto the joint limits.

The regularization term

$$
\lambda_k^2I
$$

limits excessively large joint updates when the Jacobian is ill-conditioned or close to singular.

The update may also be interpreted as the solution of the regularized least-squares problem

$$
\boxed{
\min_{\Delta\vec q}
\left(
\|J_p\Delta\vec q-\vec e\|_2^2
+
\lambda^2\|\Delta\vec q\|_2^2
\right).
}
$$

The first term attempts to reduce Cartesian error, while the second penalizes large joint changes.

### Adaptive damping

Moro adjusts the damping parameter during the iteration.

Let

$$
0<s<1
$$

be `damping_scale`.

If the trial configuration improves the error,

$$
e_{\mathrm{trial}}<e_k,
$$

the step is accepted and

$$
\boxed{
\lambda_{k+1}
=
s\lambda_k.
}
$$

If the trial configuration does not improve the error, the step is rejected and

$$
\boxed{
\lambda_{k+1}
=
\frac{\lambda_k}{s}.
}
$$

Thus, successful steps make the method progressively less damped, while unsuccessful steps increase regularization.

A rejected LM trial still counts as one algorithm iteration.

## Cyclic Coordinate Descent

Cyclic Coordinate Descent, or CCD, solves inverse kinematics without explicitly using a Jacobian matrix.

Instead, it updates one joint at a time, starting from the terminal joint and moving toward the base:

$$
\boxed{
n,n-1,\ldots,1.
}
$$

One complete sweep through all joints is considered one CCD iteration in Moro.

## CCD for revolute joints

Consider revolute joint $i$.

Its axis is

$$
\hat z_{i-1}^{\,0},
$$

and its origin is

$$
\vec r_{O_{i-1}}^{\,0}.
$$

Let the current end-effector position be

$$
\vec r_E^{\,0}
$$

and the desired target be

$$
\vec r_d^{\,0}.
$$

Construct the vectors

$$
\vec r_{ie}
=
\vec r_E^{\,0}
-
\vec r_{O_{i-1}}^{\,0},
$$

and

$$
\vec r_{it}
=
\vec r_d^{\,0}
-
\vec r_{O_{i-1}}^{\,0}.
$$

A revolute joint can only rotate these vectors around its own axis.

Therefore, both vectors are projected onto the plane perpendicular to

$$
\hat z_{i-1}^{\,0}.
$$

The projections are

$$
\boxed{
\vec u_{ie}
=
\vec r_{ie}
-
(\vec r_{ie}\cdot\hat z_{i-1}^{\,0})
\hat z_{i-1}^{\,0},
}
$$

and

$$
\boxed{
\vec u_{it}
=
\vec r_{it}
-
(\vec r_{it}\cdot\hat z_{i-1}^{\,0})
\hat z_{i-1}^{\,0}.
}
$$

After normalization,

$$
\hat u_{ie}
=
\frac{\vec u_{ie}}{\|\vec u_{ie}\|},
\qquad
\hat u_{it}
=
\frac{\vec u_{it}}{\|\vec u_{it}\|}.
$$

The signed angular update is obtained from

$$
\cos\Delta q_i
=
\hat u_{ie}\cdot\hat u_{it},
$$

and

$$
\sin\Delta q_i
=
\hat z_{i-1}^{\,0}
\cdot
\left(
\hat u_{ie}\times\hat u_{it}
\right).
$$

Thus,

$$
\boxed{
\Delta q_i
=
\operatorname{atan2}
\left(
\sin\Delta q_i,
\cos\Delta q_i
\right).
}
$$

The joint is updated according to

$$
q_i
\leftarrow
q_i+\Delta q_i,
$$

and then clipped to its joint limits.

Geometrically, the update attempts to rotate the end-effector direction toward the target direction within the plane of motion available to that joint.

If either projected vector is too small to define a reliable direction, Moro skips that individual angular update.

## CCD for prismatic joints

For a prismatic joint, motion is restricted to translation along its axis.

The current position residual is

$$
\vec e
=
\vec r_d^{\,0}
-
\vec r_E^{\,0}.
$$

The required joint displacement is obtained by projecting this residual onto the joint axis:

$$
\boxed{
\Delta q_i
=
\vec e\cdot\hat z_{i-1}^{\,0}.
}
$$

The update is therefore

$$
q_i
\leftarrow
q_i+\Delta q_i,
$$

followed by clipping to the corresponding joint limits.

Moro recomputes the current end-effector position before the prismatic update so that the residual reflects changes already made by other joints during the same CCD sweep.

## Convergence and termination

Solver termination does not necessarily imply convergence.

The main success condition is

$$
\boxed{
\|\vec e_k\|_2<\texttt{tol}.
}
$$

A solve may also terminate because of:

- maximum iterations,
- joint-step stagnation,
- error-improvement stagnation,
- numerical failure.

Therefore,

$$
\boxed{
\text{termination}\neq\text{convergence}.
}
$$

## Maximum iterations

If the tolerance has not been reached after the allowed number of global iterations,

$$
k=\texttt{max\_iter},
$$

the solver terminates without convergence.

The default maximum is method dependent:

- 100 iterations for Newton and LM,
- 500 sweeps for CCD.

Reaching the iteration limit does not prove that the target is unreachable. It only indicates that the selected method did not converge within the available iteration budget.

## Stagnation by joint step

The effective joint update is measured after applying joint limits.

For Newton and LM,

$$
\boxed{
s_k
=
\|\vec q_{k+1}-\vec q_k\|_2.
}
$$

For CCD, it is measured between the configurations before and after a complete sweep.

If

$$
s_k\leq\texttt{step\_tol}
$$

while the Cartesian error remains above the desired tolerance, a stagnation counter is increased.

The solver terminates when this condition persists for

$$
\texttt{stagnation\_iterations}
$$

consecutive iterations or sweeps.

## Stagnation by lack of error improvement

Moro also tracks the change in Cartesian error:

$$
\boxed{
\Delta e_k
=
e_{k-1}-e_k.
}
$$

If

$$
\Delta e_k
\leq
\texttt{error\_change\_tol}
$$

for the required number of consecutive iterations, the solver terminates because the error is no longer improving sufficiently.

This condition can occur near singularities, joint-limit boundaries, unreachable targets, or unfavorable local configurations.

## Numerical failures

Moro checks that quantities produced during numerical IK remain finite.

Controlled numerical failure results may be generated when finite values cannot be obtained while evaluating:

- forward kinematics,
- the position Jacobian,
- the joint update,
- CCD geometric quantities.

Invalid user inputs, unresolved symbolic parameters, and failures to construct valid numerical functions are treated separately as input or setup errors.

## Position IK result

The result of solving one Cartesian position target is represented by

```python
IKSolution
```

which contains:

- `q`,
- `converged`,
- `iterations`,
- `error`,
- `method`,
- `residual`,
- `message`.

### Final configuration

`q` contains the final valid joint configuration reached by the solver.

This field is available even when the solver does not converge.

### Convergence status

`converged` indicates whether the requested position tolerance was achieved.

For a valid converged result,

$$
\boxed{
\texttt{converged=True}
\Rightarrow
\vec q
\text{ and }
\texttt{error}
\text{ are finite}.
}
$$

This invariant is enforced by `IKSolution`.

### Residual and error

When available,

$$
\boxed{
\texttt{residual}
=
\vec r_d^{\,0}
-
\vec r_{O_n}^{\,0}(\vec q).
}
$$

The reported scalar error satisfies

$$
\boxed{
\texttt{error}
=
\|\texttt{residual}\|_2.
}
$$

A finite residual must contain exactly three components.

If no finite residual can be obtained after a numerical failure,

```text
residual = None
error = np.inf
```

is used instead.

### Iteration count

For Newton and LM, `iterations` counts attempted global updates.

For LM, rejected trial steps still count.

For CCD, `iterations` counts complete joint sweeps.

If the initial configuration already satisfies the requested tolerance,

```text
iterations = 0
```

is returned.

### Outcome message

`message` contains a short description of the termination cause, such as:

- successful convergence,
- maximum iterations,
- step stagnation,
- error stagnation,
- numerical failure.

## Solving a sequence of position targets

Moro also provides sequential position IK through

```python
solve_position_trajectory(...)
```

for a sequence

$$
\vec r_{d,0},
\vec r_{d,1},
\ldots,
\vec r_{d,m-1}.
$$

The first target uses the user-provided initial configuration

$$
\vec q_0.
$$

After a target converges, its solution is reused as the initial configuration for the next target:

$$
\boxed{
\vec q_k^\ast
\rightarrow
\vec q_{0,k+1}.
}
$$

This strategy is commonly called a *warm start*.

It can reduce the number of iterations and often encourages local continuity between neighboring IK solutions.

```{important}
Sequential seeding does not guarantee global branch continuity.

`solve_position_trajectory` does not perform branch optimization, smoothing, timing, interpolation, or global trajectory planning.
```

The trajectory solver stops at the first target that does not converge.

## Trajectory IK result

A sequence of IK solves is represented by

```python
IKTrajectorySolution
```

which contains:

- `solutions`,
- `converged`,
- `failed_index`,
- `message`.

### Successful trajectory

A converged trajectory requires every individual solve to have converged:

$$
\boxed{
\texttt{trajectory.converged=True}
\Rightarrow
\forall i,\;
\texttt{solutions[i].converged=True}.
}
$$

In this case,

```text
failed_index = None
```

is required.

### Failed trajectory

If target $k$ is the first one that does not converge, processing stops and

$$
\boxed{
\texttt{failed\_index}=k.
}
$$

The failed `IKSolution` is retained as the final processed result.

All previous solutions are guaranteed to be converged:

$$
\boxed{
\forall i<k,
\quad
\texttt{solutions[i].converged=True},
}
$$

while

$$
\boxed{
\texttt{solutions[k].converged=False}.
}
$$

Therefore, a failed trajectory result has the structure

$$
[
\text{success},
\ldots,
\text{success},
\text{failure}
].
$$

Targets after the failing one are not processed.

### Convenience properties

`IKTrajectorySolution` provides convenient access to

```python
trajectory.qs
trajectory.errors
trajectory.iterations
```

which return the per-target joint configurations, final error norms, and iteration counts, respectively.

## Inverse kinematics in Moro

A single Cartesian target can be solved using

```python
from moro.inverse_kinematics import solve_position_ik

solution = solve_position_ik(
    robot,
    target_position,
    q0=initial_guess,
)
```

The available methods are

```python
method="newton"
method="lm"
method="ccd"
```

with Levenberg--Marquardt used by default.

For symbolic robot models containing geometric parameters, numerical values can be supplied through

```python
parameters={...}
```

These substitutions are applied locally to the inverse-kinematics expressions and do not modify the robot model or its cached symbolic expressions.

## Example: planar 2R manipulator

Consider a planar 2R robot with link lengths $a_1$ and $a_2$.

Its forward position is

$$
x
=
a_1\cos q_1
+
a_2\cos(q_1+q_2),
$$

$$
y
=
a_1\sin q_1
+
a_2\sin(q_1+q_2).
$$

For a desired Cartesian position

$$
\vec r_d=
\begin{bmatrix}
x_d\\
y_d\\
0
\end{bmatrix},
$$

the inverse-kinematics problem is

$$
\boxed{
\begin{bmatrix}
a_1\cos q_1+a_2\cos(q_1+q_2)\\
a_1\sin q_1+a_2\sin(q_1+q_2)\\
0
\end{bmatrix}
\approx
\begin{bmatrix}
x_d\\
y_d\\
0
\end{bmatrix}.
}
$$

Using Moro:

```python
import moro as mr
from moro.abc import l1, l2, q1, q2
from moro.inverse_kinematics import solve_position_ik

robot = mr.Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)

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

The resulting object can be inspected through

```python
solution.q
solution.converged
solution.error
solution.residual
solution.iterations
solution.message
```

Different initial guesses may lead to different valid joint configurations for the same Cartesian target.

## Example: sequence of targets

A sequence of Cartesian targets can be solved as

```python
from moro.inverse_kinematics import solve_position_trajectory

targets = [
    [1.5, 0.2, 0.0],
    [1.4, 0.4, 0.0],
    [1.2, 0.6, 0.0],
]

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

The sequence of joint configurations is available through

```python
trajectory.qs
```

and can be used, for example, as input to a robot animation workflow.

## Scope and limitations

The current inverse-kinematics implementation in Moro provides numerical position IK for serial manipulators.

It supports:

- revolute and prismatic joints,
- joint limits,
- user-defined or reproducible random initialization,
- Newton-type pseudoinverse updates,
- adaptive Levenberg--Marquardt,
- Cyclic Coordinate Descent,
- stagnation detection,
- sequential position targets,
- controlled numerical-failure results.

The current implementation does not provide:

- full-pose inverse kinematics,
- orientation-error constraints,
- analytical closed-form IK,
- collision avoidance,
- global IK branch optimization,
- trajectory interpolation,
- trajectory timing,
- joint-space smoothing,
- global motion planning.

## Summary of conventions

| Concept | Moro convention |
|---|---|
| IK scope | Position inverse kinematics |
| Target | $\vec r_d^{\,0}\in\mathbb R^3$ |
| Current position | $\vec r_{O_n}^{\,0}(\vec q)$ |
| Residual | $\vec e=\vec r_d^{\,0}-\vec r_{O_n}^{\,0}$ |
| Error | $\|\vec e\|_2$ |
| Convergence | $\|\vec e\|_2<\texttt{tol}$ |
| Jacobian used for IK | $J_p=J[:3,:]$ |
| Newton update | $J_p^\dagger\vec e$ |
| LM update | $(J_p^TJ_p+\lambda^2I)^{-1}J_p^T\vec e$ |
| LM damping | Adaptive |
| CCD joint order | $n,n-1,\ldots,1$ |
| Joint limits | Enforced by clipping |
| Missing `q0` | Random initialization within limits |
| Random reproducibility | `random_state` |
| Newton/LM iteration | One attempted global update |
| CCD iteration | One complete joint sweep |
| Trajectory initialization | Previous converged solution |
| Trajectory failure | Stop at first non-converged target |
| Full pose IK | Not currently supported |

Inverse kinematics complements forward and differential kinematics by solving the local numerical search problem in the opposite direction: from a desired Cartesian position to an admissible joint configuration.