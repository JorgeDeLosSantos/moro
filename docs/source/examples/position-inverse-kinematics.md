# Position Inverse Kinematics

This example shows how to solve a Cartesian position inverse-kinematics problem with `moro`.

We will use the anthropomorphic RRR manipulator introduced in the previous example and follow a complete numerical workflow:

1. define the robot model;
2. choose a reachable Cartesian target;
3. solve the inverse-kinematics problem;
4. inspect the solver result;
5. validate the solution using forward kinematics;
6. compare the available numerical methods;
7. visualize the recovered robot configuration.

The goal is to show how the symbolic robot model can be reused directly for numerical inverse kinematics.

## Problem

Consider the anthropomorphic RRR manipulator defined by the Denavit-Hartenberg table:

| Link | \(a_i\) | \(\alpha_i\) | \(d_i\) | \(\theta_i\) | Joint |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | \(0\) | \(\pi/2\) | \(d_1\) | \(q_1\) | revolute |
| 2 | \(l_2\) | \(0\) | \(0\) | \(q_2\) | revolute |
| 3 | \(l_3\) | \(0\) | \(0\) | \(q_3\) | revolute |

We want to determine the joint coordinates

\[
q =
\begin{bmatrix}
q_1 & q_2 & q_3
\end{bmatrix}^{T}
\]

that place the end-effector at a prescribed Cartesian position

\[
p_d =
\begin{bmatrix}
x_d & y_d & z_d
\end{bmatrix}^{T}.
\]

In this example, only end-effector position is considered. Orientation is not part of the inverse-kinematics objective.

## Robot model

First import the required libraries and symbolic variables:

```python
import numpy as np
import sympy as sp

from moro import Robot
from moro.abc import q1, q2, q3, d1, l2, l3
from moro.inverse_kinematics import solve_position_ik
```

Define the robot:

```python
robot = Robot(
    (0, sp.pi / 2, d1, q1, "r"),
    (l2, 0, 0, q2, "r"),
    (l3, 0, 0, q3, "r"),
)
```

We will use the numerical geometric parameters:

\[
d_1=1.0,
\qquad
l_2=1.2,
\qquad
l_3=0.9.
\]

Store them in a dictionary:

```python
parameters = {
    d1: 1.0,
    l2: 1.2,
    l3: 0.9,
}
```

These parameters are passed separately to the inverse-kinematics solver because the robot model itself remains symbolic.

## Defining a reachable target

A convenient way to construct a test problem is to generate the Cartesian target from a known joint configuration.

This guarantees that the target belongs to the robot workspace.

Choose the reference configuration:

\[
q_\text{ref} =
\begin{bmatrix}
30^\circ &
-20^\circ &
35^\circ
\end{bmatrix}^{T}.
\]

In radians:

```python
q_ref = [
    np.deg2rad(30.0),
    np.deg2rad(-20.0),
    np.deg2rad(35.0),
]
```

To compute the corresponding end-effector position, create the substitution dictionary:

```python
reference_values = {
    **parameters,
    **dict(zip(robot.qs, q_ref)),
}
```

Extract the Cartesian position from the forward kinematics:

```python
target_expr = robot.T[:3, 3]

target = np.asarray(
    target_expr.subs(reference_values),
    dtype=float,
).reshape(3)

target
```

This numerical vector will be used as the desired Cartesian position.

Although the target was generated from a known configuration for validation purposes, the inverse-kinematics solver does not use `q_ref`.

Its task is simply to find a joint configuration that reaches the same Cartesian point.

## Solving the inverse kinematics problem

The main position IK interface is:

```python
solve_position_ik(...)
```

We provide:

- the robot;
- the Cartesian target;
- an initial joint estimate;
- the numerical geometric parameters.

For example:

```python
solution = solve_position_ik(
    robot,
    target,
    q0=[0.3, -0.1, 0.2],
    parameters=parameters,
    method="lm",
    tol=1e-9,
)
```

Here:

```text
method="lm"
```

selects the Levenberg-Marquardt solver.

The result is an `IKSolution` object containing both the joint solution and information about the numerical process.

## Inspecting the solution

Display the complete result:

```python
solution
```

The most important quantity is the joint vector:

```python
solution.q
```

The convergence state is available through:

```python
solution.converged
```

and the final Cartesian error through:

```python
solution.error
```

The number of iterations used by the solver is:

```python
solution.iterations
```

Additional diagnostic information is available with:

```python
solution.residual
solution.message
solution.method
```

A typical inspection may therefore look like:

```python
print("Converged:", solution.converged)
print("Joint solution:", solution.q)
print("Iterations:", solution.iterations)
print("Error:", solution.error)
print("Residual:", solution.residual)
print("Message:", solution.message)
```

The returned joint vector is numerical and follows the same ordering as:

```python
robot.qs
```

For this robot:

```text
[q1, q2, q3]
```

## Validating with forward kinematics

An inverse-kinematics solution should be checked by substituting the recovered joint values back into the forward-kinematics model.

Build the numerical substitution dictionary:

```python
solution_values = {
    **parameters,
    **dict(zip(robot.qs, solution.q)),
}
```

Evaluate the end-effector position:

```python
p_solution = np.asarray(
    robot.T[:3, 3].subs(solution_values),
    dtype=float,
).reshape(3)

p_solution
```

Now compare it with the desired target:

```python
target
```

The validation error is:

```python
validation_error = np.linalg.norm(
    target - p_solution
)

validation_error
```

For a converged solution, this quantity should be consistent with the solver tolerance.

This validation step is useful because it checks the result independently through the forward-kinematics model.

## Multiple inverse-kinematics solutions

Serial manipulators may admit more than one joint configuration for the same Cartesian target.

The solution obtained numerically can therefore depend on:

- the initial estimate;
- the selected solver;
- joint limits;
- the geometry of the mechanism.

For example, changing the initial estimate may lead to another valid solution:

```python
solution_2 = solve_position_ik(
    robot,
    target,
    q0=[1.0, 0.5, -0.5],
    parameters=parameters,
    method="lm",
    tol=1e-9,
)

solution_2.q
```

Both solutions can be valid if they place the end-effector sufficiently close to the desired Cartesian point.

They can be compared using forward kinematics:

```python
values_2 = {
    **parameters,
    **dict(zip(robot.qs, solution_2.q)),
}

p_solution_2 = np.asarray(
    robot.T[:3, 3].subs(values_2),
    dtype=float,
).reshape(3)

np.linalg.norm(target - p_solution_2)
```

This illustrates why the initial estimate is an important part of numerical inverse kinematics.

## Comparing IK methods

`moro` currently provides three position IK methods:

```text
"lm"
"newton"
"ccd"
```

We can solve the same problem with each method.

### Levenberg-Marquardt

```python
sol_lm = solve_position_ik(
    robot,
    target,
    q0=[0.3, -0.1, 0.2],
    parameters=parameters,
    method="lm",
    tol=1e-8,
)
```

### Newton-Raphson

```python
sol_newton = solve_position_ik(
    robot,
    target,
    q0=[0.3, -0.1, 0.2],
    parameters=parameters,
    method="newton",
    tol=1e-8,
)
```

### Cyclic Coordinate Descent

```python
sol_ccd = solve_position_ik(
    robot,
    target,
    q0=[0.3, -0.1, 0.2],
    parameters=parameters,
    method="ccd",
    tol=1e-8,
    max_iter=600,
)
```

The results can be summarized with:

```python
solutions = [
    sol_lm,
    sol_newton,
    sol_ccd,
]

for sol in solutions:
    print(
        f"{sol.method:>6s} | "
        f"converged={sol.converged} | "
        f"iterations={sol.iterations} | "
        f"error={sol.error:.3e}"
    )
```

The methods solve the same position problem but use different numerical strategies.

The exact number of iterations and final joint coordinates may differ.

For most applications, the solver should therefore be judged primarily by:

- whether it converges;
- whether the final Cartesian error is acceptable;
- whether the resulting joint configuration satisfies the desired constraints.

## Joint limits

Inverse kinematics can also be solved under joint limits.

For example:

```python
joint_limits = [
    (-np.pi, np.pi),
    (-np.pi / 2, np.pi / 2),
    (-np.pi / 2, np.pi / 2),
]
```

Pass them to the solver:

```python
limited_solution = solve_position_ik(
    robot,
    target,
    q0=[0.3, -0.1, 0.2],
    parameters=parameters,
    joint_limits=joint_limits,
    method="lm",
)
```

The solver keeps the joint coordinates inside the specified intervals.

Joint limits are particularly useful when several mathematical solutions exist but only some correspond to physically admissible robot configurations.

## Random initialization

If `q0` is omitted, the solver can generate an initial configuration automatically.

For reproducible results, provide:

```python
random_state=...
```

For example:

```python
random_solution = solve_position_ik(
    robot,
    target,
    parameters=parameters,
    random_state=42,
)
```

Calling the solver again with the same seed gives the same random initialization:

```python
random_solution_2 = solve_position_ik(
    robot,
    target,
    parameters=parameters,
    random_state=42,
)
```

This is useful when numerical experiments need to be reproducible.

## Visualizing the solution

The recovered joint configuration can be visualized using `RobotVisualizer`.

Import the visualizer:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
```

Construct the substitution dictionary from the IK solution:

```python
solution_values = {
    **parameters,
    **dict(zip(robot.qs, solution.q)),
}
```

### Matplotlib

```python
fig, ax = viz.plot(
    solution_values,
    backend="matplotlib",
)
```

### Three.js

For interactive inspection:

```python
viz.plot(
    solution_values,
    backend="threejs",
)
```

The Three.js viewer makes it easy to inspect the spatial configuration from multiple viewpoints.

## Comparing the reference and recovered configurations

Because the target was generated from `q_ref`, we can also compare the original configuration with the one recovered by IK:

```python
print("Reference:", q_ref)
print("Recovered:", solution.q)
```

These vectors do not necessarily have to be identical.

Inverse kinematics solves the Cartesian condition:

\[
p(q)=p_d,
\]

not the condition:

\[
q=q_\text{ref}.
\]

If several joint configurations produce the same end-effector position, the numerical solver may converge to any one of them depending on its initial state and numerical method.

The relevant validation criterion is therefore the Cartesian error, not necessarily the difference between `q_ref` and `solution.q`.

## Handling unsuccessful solutions

A numerical IK computation may fail to converge.

The result should therefore be checked before using the returned joint vector:

```python
if solution.converged:
    print("IK solution found.")
else:
    print(solution.message)
```

A failure may occur because:

- the target is outside the reachable workspace;
- the initial estimate is unfavorable;
- joint limits prevent the target from being reached;
- the iteration limit is too small;
- the numerical method stagnates.

The `IKSolution` object preserves diagnostic information even when convergence is not achieved.

## Discussion

This example illustrates how `moro` combines symbolic robot modeling with numerical inverse kinematics.

The robot is first defined symbolically:

```python
robot = Robot(...)
```

Its forward kinematics remain available through:

```python
robot.T
```

while the same model is passed directly to:

```python
solve_position_ik(...)
```

The complete workflow is:

```text
define symbolic robot
        ↓
assign numerical geometry
        ↓
define Cartesian target
        ↓
choose initial joint estimate
        ↓
solve numerical IK
        ↓
inspect IKSolution
        ↓
validate with forward kinematics
        ↓
visualize the solution
```

One important distinction is that `solve_position_ik()` solves only the Cartesian position condition:

\[
p(q)=p_d.
\]

It does not currently impose a desired end-effector orientation.

The solution is also local in nature. Numerical inverse-kinematics algorithms do not generally guarantee that a particular branch of the solution space will be found.

The initial estimate, joint limits, and numerical method can all influence the final result.

These characteristics become even more important when solving a sequence of Cartesian targets.

Instead of solving each target independently, `moro` provides `solve_position_trajectory()`, which uses the previous solution as the initial estimate for the next target.

That workflow is explored in the next example.

## See also

- **Anthropomorphic RRR Manipulator** — symbolic forward and differential kinematics of the robot used here.
- **Cartesian Trajectory with Inverse Kinematics** — solving a sequence of Cartesian position targets.
- **User Guide → Inverse Kinematics** — complete description of solver options and result objects.
- **User Guide → Forward Kinematics** — validating IK solutions through the robot model.
- **User Guide → Visualization** — rendering the recovered joint configurations.
- **Theory → Inverse Kinematics** — mathematical background and numerical methods.