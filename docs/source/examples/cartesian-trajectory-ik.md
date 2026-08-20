# Cartesian Trajectory with Inverse Kinematics

This example shows how to solve a sequence of Cartesian position targets with `moro` and convert the resulting joint configurations into a robot animation.

We will:

1. define a planar 2R robot;
2. generate a Cartesian path;
3. sample the path into discrete target positions;
4. solve the inverse kinematics sequentially;
5. inspect the trajectory result;
6. validate the Cartesian tracking error;
7. convert the joint trajectory into visualization configurations;
8. animate the robot and display the end-effector path.

The key idea is that `solve_position_trajectory()` does not generate a Cartesian trajectory. It receives a sequence of target positions already defined by the user and solves inverse kinematics for them in order.

## Problem

Consider a planar two-link manipulator with revolute joints:

\[
q_1,\; q_2,
\]

and link lengths:

\[
l_1,\; l_2.
\]

Using the classical Denavit-Hartenberg convention, the robot is described by:

| Link | \(a_i\) | \(\alpha_i\) | \(d_i\) | \(\theta_i\) | Joint |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | \(l_1\) | \(0\) | \(0\) | \(q_1\) | revolute |
| 2 | \(l_2\) | \(0\) | \(0\) | \(q_2\) | revolute |

We want the end-effector to follow a sequence of Cartesian positions

\[
p_d^{(0)},\;
p_d^{(1)},\;
\dots,\;
p_d^{(N-1)}.
\]

For each target, inverse kinematics must determine a corresponding joint configuration

\[
q^{(k)} =
\begin{bmatrix}
q_1^{(k)} \\
q_2^{(k)}
\end{bmatrix}.
\]

The complete sequence of joint configurations can then be interpreted as a discrete robot motion.

## Robot model

First import the required objects:

```python
import numpy as np

from moro import Robot
from moro.abc import q1, q2, l1, l2
from moro.inverse_kinematics import solve_position_trajectory
```

Define the planar 2R robot:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

We will use:

\[
l_1=1.0,
\qquad
l_2=1.0.
\]

Store the geometry in a parameter dictionary:

```python
parameters = {
    l1: 1.0,
    l2: 1.0,
}
```

Because the robot model remains symbolic, the same geometry dictionary will also be useful later for visualization.

## Defining the Cartesian path

For this example, consider a smooth curve in the robot plane.

Let

\[
s\in[0,1]
\]

be a path parameter and define:

\[
x(s)=1.0+0.3\cos(2\pi s),
\]

\[
y(s)=0.8+0.3\sin(2\pi s),
\]

\[
z(s)=0.
\]

This describes a small circular path centered at:

\[
(1.0,\;0.8).
\]

Sample the curve numerically:

```python
s = np.linspace(0.0, 1.0, 60)

targets = [
    [
        1.0 + 0.3 * np.cos(2 * np.pi * value),
        0.8 + 0.3 * np.sin(2 * np.pi * value),
        0.0,
    ]
    for value in s
]
```

The resulting object is a sequence of Cartesian position targets:

```python
len(targets)
```

which returns:

```text
60
```

Each target has the form:

```text
[x, y, z]
```

The trajectory solver will process these targets in the same order.

## Choosing the initial configuration

Unlike a single position IK problem, trajectory IK requires an initial joint estimate.

For example:

```python
q0 = [0.5, 0.5]
```

This estimate is used for the first Cartesian target.

After the first target is solved, the resulting joint configuration is used as the initial estimate for the next target.

This warm-start strategy is repeated throughout the trajectory:

```text
q0
 ↓
target 0 → q(0)
             ↓
target 1 → q(1)
             ↓
target 2 → q(2)
             ↓
            ...
```

This usually helps preserve continuity between neighboring IK solutions.

## Solving the IK trajectory

Use:

```python
solve_position_trajectory(...)
```

to solve the complete sequence:

```python
trajectory = solve_position_trajectory(
    robot,
    targets,
    q0=q0,
    parameters=parameters,
    method="lm",
    tol=1e-8,
)
```

The solver attempts to find one joint configuration for each Cartesian target.

If all targets are solved successfully:

```python
trajectory.converged
```

returns:

```text
True
```

## Inspecting the trajectory solution

The returned trajectory object contains information about the complete numerical process.

Check the global convergence state:

```python
trajectory.converged
```

The message associated with the trajectory is:

```python
trajectory.message
```

If the solver fails at one target, the corresponding position in the sequence is available through:

```python
trajectory.failed_index
```

For a completely successful trajectory:

```text
failed_index = None
```

The computed joint configurations are stored in:

```python
trajectory.qs
```

The number of returned configurations should match the number of solved Cartesian targets.

For example:

```python
len(trajectory.qs)
```

The final Cartesian errors for each target are available through:

```python
trajectory.errors
```

and the number of iterations required at each step through:

```python
trajectory.iterations
```

A simple summary can therefore be printed with:

```python
print("Converged:", trajectory.converged)
print("Message:", trajectory.message)
print("Failed index:", trajectory.failed_index)
print("Solved configurations:", len(trajectory.qs))
```

## Inspecting the joint trajectory

The joint trajectory is a sequence of vectors:

```python
trajectory.qs
```

For a planar 2R robot, each one contains:

```text
[q1, q2]
```

The first configuration can be inspected with:

```python
trajectory.qs[0]
```

and the final one with:

```python
trajectory.qs[-1]
```

If needed, the joint trajectories can be separated into individual arrays:

```python
q_values = np.asarray(trajectory.qs, dtype=float)

q1_values = q_values[:, 0]
q2_values = q_values[:, 1]
```

These arrays may be useful for plotting the resulting motion in joint space.

For example:

```python
import matplotlib.pyplot as plt

plt.plot(s[:len(q1_values)], q1_values, label="q1")
plt.plot(s[:len(q2_values)], q2_values, label="q2")

plt.xlabel("Path parameter")
plt.ylabel("Joint angle [rad]")
plt.legend()
plt.show()
```

The Cartesian path and the resulting joint trajectory are different representations of the same motion.

## Validating the trajectory

As with a single IK solution, the computed joint configurations can be checked using forward kinematics.

Create a helper function:

```python
def end_effector_position(robot, q, parameters):
    values = {
        **parameters,
        **dict(zip(robot.qs, q)),
    }

    return np.asarray(
        robot.T[:3, 3].subs(values),
        dtype=float,
    ).reshape(3)
```

Now evaluate the end-effector position for every solved configuration:

```python
computed_positions = np.array([
    end_effector_position(
        robot,
        q,
        parameters,
    )
    for q in trajectory.qs
])
```

Convert the corresponding desired targets to an array:

```python
desired_positions = np.asarray(
    targets[:len(trajectory.qs)],
    dtype=float,
)
```

The Cartesian error at each point is:

```python
validation_errors = np.linalg.norm(
    desired_positions - computed_positions,
    axis=1,
)
```

Inspect the largest error:

```python
validation_errors.max()
```

For a successfully converged trajectory, the errors should be consistent with the requested tolerance.

The desired and computed Cartesian paths can also be compared visually:

```python
plt.plot(
    desired_positions[:, 0],
    desired_positions[:, 1],
    "--",
    label="Desired",
)

plt.plot(
    computed_positions[:, 0],
    computed_positions[:, 1],
    label="Computed",
)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.show()
```

This provides an independent validation of the inverse-kinematics result.

## Preparing the animation

`RobotVisualizer.animate()` expects a sequence of substitution dictionaries rather than a sequence of joint vectors.

Therefore, the trajectory must first be converted.

Import the visualization tools:

```python
from moro.visualization import (
    RobotVisualizer,
    VisualizationStyle,
)
```

Create the visualizer:

```python
viz = RobotVisualizer(robot)
```

Convert each joint vector into a complete substitution dictionary:

```python
configurations = [
    {
        **parameters,
        **dict(zip(robot.qs, q)),
    }
    for q in trajectory.qs
]
```

Each element now contains both:

- the fixed geometric parameters;
- the corresponding joint coordinates.

For example:

```python
configurations[0]
```

has the conceptual structure:

```text
{
    l1: 1.0,
    l2: 1.0,
    q1: ...,
    q2: ...,
}
```

This is the format expected by the visualization module.

## Visualizing the robot motion

### Matplotlib

A Matplotlib animation can be created with:

```python
animation = viz.animate(
    configurations,
    backend="matplotlib",
    interval=80,
)
```

The returned object is a Matplotlib `FuncAnimation`.

Keep a reference to it until the animation has been displayed or saved.

### Three.js

For an interactive notebook animation:

```python
viz.animate(
    configurations,
    backend="threejs",
)
```

The viewer provides controls for:

- Play and Pause;
- frame selection;
- Front view;
- Top view;
- Isometric view;
- orthographic projection;
- perspective projection;
- free orbit navigation.

This makes it possible to inspect both the complete motion and individual IK solutions.

## Showing the end-effector path

The Cartesian trajectory can also be displayed directly inside the robot animation.

Create a visualization style with:

```python
style = VisualizationStyle(
    show_trajectory=True,
)
```

Then:

```python
viz.animate(
    configurations,
    backend="threejs",
    style=style,
)
```

The displayed trajectory is extracted from the sequence of end-effector positions associated with the animation frames.

This provides a direct visual connection between:

```text
Cartesian targets
        ↓
IK solutions
        ↓
robot configurations
        ↓
end-effector path
```

## Tracing the path progressively

By default, the trajectory is shown in `"full"` mode.

This means that the complete Cartesian path is visible throughout the animation.

To display only the part already traversed by the robot, use:

```python
style = VisualizationStyle(
    show_trajectory=True,
    trajectory_mode="trace",
)
```

Then animate again:

```python
viz.animate(
    configurations,
    backend="threejs",
    style=style,
)
```

In `"trace"` mode, the trajectory grows progressively as the robot advances through the sequence.

The same option can also be used with the Matplotlib backend.

## Handling an unsuccessful trajectory

Trajectory IK may fail before every target has been solved.

The result should therefore be checked before assuming that the complete path is available:

```python
if trajectory.converged:
    print("Complete trajectory solved.")
else:
    print(
        "Trajectory failed at target:",
        trajectory.failed_index,
    )
    print(trajectory.message)
```

When a failure occurs, the trajectory object still contains the configurations successfully computed before the failing target.

For example:

```python
len(trajectory.qs)
```

indicates how many configurations were recovered.

A failure may occur because:

- one target lies outside the workspace;
- joint limits make a target unreachable;
- the solver stagnates;
- the number of iterations is insufficient;
- the local solution branch becomes difficult to follow.

The Cartesian path should therefore be designed with the robot workspace and joint constraints in mind.

## Changing the numerical method

The same trajectory can be solved with the other position IK methods.

For example:

```python
trajectory_newton = solve_position_trajectory(
    robot,
    targets,
    q0=q0,
    parameters=parameters,
    method="newton",
    tol=1e-8,
)
```

or:

```python
trajectory_ccd = solve_position_trajectory(
    robot,
    targets,
    q0=q0,
    parameters=parameters,
    method="ccd",
    tol=1e-8,
    max_iter=500,
)
```

The choice of method can affect:

- convergence;
- iterations per target;
- numerical robustness;
- the joint-space branch followed by the solution.

For a trajectory problem, continuity between neighboring configurations can be just as important as solving each target independently.

## Discussion

This example introduces an important distinction between Cartesian trajectory definition and inverse-kinematics trajectory solving.

The Cartesian targets are created independently:

```python
targets = [...]
```

and are then passed to:

```python
solve_position_trajectory(...)
```

The function does not interpolate, smooth, or plan the Cartesian path.

Instead, it solves a sequence of position IK problems.

The first problem starts from:

```python
q0
```

and each subsequent problem starts from the previous solution.

The workflow is therefore:

```text
define Cartesian path
        ↓
sample Cartesian targets
        ↓
provide initial configuration
        ↓
solve first IK target
        ↓
use solution as next initial guess
        ↓
repeat for remaining targets
        ↓
obtain joint trajectory
        ↓
validate with forward kinematics
        ↓
animate robot motion
```

This warm-start strategy helps the solver follow a locally continuous branch of the inverse-kinematics solution.

However, it does not guarantee:

- globally smooth joint motion;
- optimal joint trajectories;
- velocity or acceleration continuity;
- collision avoidance;
- Cartesian interpolation;
- time parameterization;
- global branch consistency.

The result should therefore be interpreted as a sequence of IK solutions associated with an ordered sequence of Cartesian targets.

More advanced trajectory generation would require additional planning or trajectory-generation tools.

## See also

- **Planar 2R Manipulator** — forward kinematics, Jacobian, and visualization of the robot used here.
- **Position Inverse Kinematics** — solving and validating a single Cartesian target.
- **User Guide → Inverse Kinematics** — solver parameters and trajectory result objects.
- **User Guide → Forward Kinematics** — validating Cartesian positions.
- **User Guide → Visualization** — animation and end-effector trajectory options.
- **Theory → Inverse Kinematics** — mathematical background and numerical methods.