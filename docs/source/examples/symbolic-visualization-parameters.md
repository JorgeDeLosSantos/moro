# Symbolic DH Parameters in Robot Visualization

This notebook-ready Moro 0.7.0 example demonstrates **API-001**: use a
numerical joint vector for the robot configuration and pass constant
symbolic DH parameters separately. The same workflow works in Jupyter
and Google Colab, with **Three.js** as the default rendering backend.

These examples use the `parameters=` visualization API introduced in
Moro 0.7.0. No browser-specific JavaScript or extra visualization
packages are needed in notebook cells.

## 1. Build a symbolic RR model

```python
import numpy as np

from IPython.display import display
from moro import Robot
from moro.abc import l1, l2, q1, q2
from moro.visualization import RobotVisualizer

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
viz = RobotVisualizer(robot)
parameters = {l1: 1.0, l2: 0.8}

print(robot.qs)
# [q_1(t), q_2(t)]
```

Here, `robot.qs` fixes the order of the numerical joint-vector entries.
The geometric constants stay in `parameters`; the symbolic `Robot`
is **not mutated or numerically bound**.

## 2. Display a single configuration

```python
configuration = np.array([0.5, -0.3])

# Interactive Three.js viewer
display(viz.plot(
    configuration=configuration,
    parameters=parameters,
))
```

To inspect the same robot with Matplotlib:

```python
import matplotlib.pyplot as plt

fig, ax = viz.plot(
    configuration,
    backend="matplotlib",
    parameters=parameters,
)
plt.show()
```

The backend is still the second positional argument if desired;
`parameters` is always keyword-only.

## 3. Animate nonuniform physical time

Define a few joint samples with deliberately nonuniform timestamps:

```python
q_history = np.array([
    [0.00,  0.00],
    [0.20, -0.10],
    [0.50, -0.40],
    [0.80, -0.60],
    [1.00, -0.20],
])
sample_times = np.array([0.0, 0.1, 0.7, 0.8, 1.5])

assert q_history.shape == (sample_times.size, robot.dof)

display(viz.animate(
    configurations=q_history,
    times=sample_times,
    parameters=parameters,
    playback_speed=1.0,
))
```

The Three.js player follows the supplied physical timeline in
seconds; no interpolation between unsupplied joint samples occurs.
For the same configurations, compare the **legacy fixed-interval** mode:

```python
display(viz.animate(
    q_history,
    parameters=parameters,
    interval=300,
))
```

The legacy mode displays every sample with the same frame interval.
For a longer explanation of playback, see
[Physical-Time Animation Playback](physical-time-playback.md).

## 4. Existing complete symbolic dictionaries remain valid

The original API pattern is still supported:

```python
full_values = {
    l1: 1.0,
    l2: 0.8,
    q1: 0.5,
    q2: -0.3,
}
display(viz.plot(full_values))
```

If you provide a **partial** configuration mapping, share the fixed
parameters explicitly:

```python
display(viz.plot(
    {q1: 0.5, q2: -0.3},
    parameters=parameters,
))
```

Avoid repeating the same symbol in both dictionaries: Moro rejects
overlapping keys rather than using implicit precedence.

## 5. DH expressions and mixed R/P joints

The `parameters` keys identify individual **SymPy symbols**, including
symbols inside derived DH expressions. A DH entry such as `L/2` needs
a value for `L`, not a substitution mapping keyed by `L/2`.

```python
import sympy as sp

L, h = sp.symbols("L h", real=True)

rp_robot = Robot(
    (L / 2, 0, h, q1, "r"),
    (0, 0, q2, 0, "p"),
)
rp_viz = RobotVisualizer(rp_robot)

display(rp_viz.plot(
    [0.3, 0.4],
    parameters={L: 2.0, h: 0.2},
))
```

Here, `q2` represents prismatic displacement rather than rotation.
Parameters associated **only** with the dynamic model (masses, inertia,
gravity) are not required for kinematic visualization.

## 6. Input validation and compatibility

- `configuration` is a numerical vector of length `robot.dof`, or a
  SymPy-keyed mapping. `configurations` accepts a nonempty series
  of configurations, including a numerical array `(N, dof)`.
- Missing DH constants or joint coordinates, unknown keys in
  `parameters`, overlapping substitutions, and nonfinite or complex
  values raise explicit errors.
- `parameters` is **constant** across an animation; no symbolic
  object or input array is modified.
- `plot(num_vals=...)` and `animate(num_vals_list=...)` are **not
  supported** starting with Moro 0.7.0. Use the new names
  `configuration=` and `configurations=`; **positional** calls do
  not change.
- For a composite joint expression such as `q1 + theta0` in a
  DH joint field, the vector-to-coordinate correspondence is
  ambiguous with the current `Robot.qs` representation. Use a complete
  symbolic mapping rather than a numerical joint vector.
- `backend="matplotlib"` accepts `interval` but **not** `times`
  physical-time playback.

### Suggested notebook checks

1. Vary only `configuration` while keeping `parameters` fixed;
   check that the link lengths remain unchanged.
2. Compare the Three.js and Matplotlib plots of the same RR pose.
3. Compare the nonuniform physical-timestamp animation with the
   fixed-interval version.
4. Change `L` in the RP example; observe how its geometry changes
   without modifying either joint coordinate.

For full API conventions, see the
[Visualization User Guide](../user-guide/visualization.md)
and [0.7.0 release notes](../release-notes/0.7.0.md).
