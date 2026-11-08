# Trapezoidal Joint-Space Trajectories: R and RR Examples

This notebook-style example illustrates the rest-to-rest joint-space
`joint_trapezoidal_trajectory()` generator introduced in Moro 0.7.0.
We first examine a single revolute joint, then synchronize two joints,
and finally compare the trapezoidal motion with quintic interpolation.
The mathematical results require only NumPy; the plots use Matplotlib.

The essential distinction is between **velocity/acceleration limits**
(trapezoidal motion) and **boundary conditions**
(cubic or quintic interpolation). The generator does not plan around
obstacles or check actuator torques.

## 1. A one-joint R motion

Move one revolute joint from \(-0.3\) to \(0.9\) rad with a maximum speed of
\(0.6\) rad/s and a maximum acceleration magnitude of \(1.0\) rad/s²:

```python
import numpy as np
import matplotlib.pyplot as plt

from moro.trajectory import (
    joint_trajectory,
    joint_trapezoidal_trajectory,
)

q0, qf = -0.3, 0.9

motion = joint_trapezoidal_trajectory(
    q0, qf, v_max=0.6, a_max=1.0, samples=261,
)
info = motion.profile

print(info.kind)
print(f"Duration: {motion.duration:.2f} s")
print(f"Acceleration: {info.t_accel:.2f} s")
print(f"Cruise: {info.t_cruise:.2f} s")
print(f"Deceleration: {info.t_decel:.2f} s")
```

Expected values (within floating-point precision):

```text
trapezoidal
Duration: 2.60 s
Acceleration: 0.60 s
Cruise: 1.40 s
Deceleration: 0.60 s
```

The arrays `motion.t`, `motion.q`, `motion.qd` and `motion.qdd`
have shapes `(261,)`, `(261, 1)`, `(261, 1)` and `(261, 1)`.
For a 1-DOF joint, select column 0 when plotting.

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
quantities = [
    (motion.q[:, 0], r"$q$ [rad]"),
    (motion.qd[:, 0], r"$\dot q$ [rad/s]"),
    (motion.qdd[:, 0], r"$\ddot q$ [rad/s²]"),
]

phase_changes = [
    motion.t[0] + info.t_accel,
    motion.t[0] + info.t_accel + info.t_cruise,
]

for ax, (values, label) in zip(axes, quantities):
    ax.plot(motion.t, values)
    for instant in phase_changes:
        ax.axvline(instant, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(label)
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
plt.show()
```

The speed is continuous, but acceleration jumps at the phase changes;
this trajectory is **C1, not C2**. The generator reports the acceleration
of the right-hand phase at an interior transition and of the last phase
at the final sample. No jerk bound is enforced.

## 2. Trapezoidal versus triangular motion

A sufficiently short move cannot reach the prescribed maximum speed.
The generator then selects a triangular velocity profile, without a
constant-speed phase:

```python
triangular = joint_trapezoidal_trajectory(
    0.0, 1.0, v_max=2.0, a_max=1.0, samples=101,
)

print(triangular.profile.kind)       # "triangular"
print(triangular.profile.t_cruise)   # 0.0
print(triangular.duration)           # 2.0
```

The public `method` field is always `"trapezoidal"` for this family;
the actual regime is available as `motion.profile.kind`.

## 3. A synchronized two-joint RR motion

For a planar RR manipulator, suppose that

\[
\mathbf q_0 =
\begin{bmatrix} 0 \\ 0 \end{bmatrix},
\quad
\mathbf q_f =
\begin{bmatrix} 1.2 \\ -0.8 \end{bmatrix}.
\]

Use independent joint limits (radians, seconds):

```python
q0_rr = np.array([0.0, 0.0])
qf_rr = np.array([1.2, -0.8])
v_max = np.array([0.6, 0.4])
a_max = np.array([1.0, 0.8])

rr = joint_trapezoidal_trajectory(
    q0_rr, qf_rr,
    v_max=v_max, a_max=a_max,
    samples=261,
)

print(rr.profile.kind, f"{rr.duration:.2f} s")
# trapezoidal 2.60 s
```

Both joints follow the *same* normalized progress `s(t)`:

\[
\mathbf q(t) = \mathbf q_0 +
s(t)(\mathbf q_f-\mathbf q_0).
\]

Therefore the joint velocities and accelerations share the same phase
boundaries. For this example, `q_2(t) = -2q_1(t)/3`.

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for j in range(2):
    axes[0].plot(rr.t, rr.q[:, j], label=f"q{j + 1}")
    axes[1].plot(rr.t, rr.qd[:, j], label=f"qd{j + 1}")
    axes[2].plot(rr.t, rr.qdd[:, j], label=f"qdd{j + 1}")

for ax, label in zip(
    axes, ["Position [rad]", "Velocity [rad/s]", "Acceleration [rad/s²]"]
):
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
plt.show()
```

### Verify the limits analytically

The limits apply throughout the piecewise profile, not only at the
sampled instants. Since `s(t)` has a piecewise-linear speed and
piecewise-constant acceleration, the actual global maxima of each joint
are obtained directly from the profile metadata:

```python
displacement = np.abs(qf_rr - q0_rr)

maximum_joint_speeds = displacement * rr.profile.peak_path_speed
maximum_joint_accelerations = displacement * rr.profile.path_acceleration

assert np.all(maximum_joint_speeds <= v_max + 1e-12)
assert np.all(maximum_joint_accelerations <= a_max + 1e-12)
```

The reported `peak_path_speed` has units s⁻¹ and
`path_acceleration` has units s⁻²; they refer to the **dimensionless**
normalized progress, not to any specific joint.

### Optional visualization in Jupyter/Colab

Moro's visualizer can animate the joint configurations. If the robot
contains symbolic geometric parameters, supply their numerical values in
each configuration, as the current visualizer API requires:

```python
from IPython.display import display
from moro import Robot
from moro.abc import q1, q2, l1, l2
from moro.visualization import RobotVisualizer

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)

configurations = [
    {l1: 1.0, l2: 0.8, q1: float(q[0]), q2: float(q[1])}
    for q in rr.q
]

viewer = RobotVisualizer(robot).animate(
    configurations,
    backend="threejs",
    times=rr.t,
    playback_speed=1.0,
)
display(viewer)
```

**Playback note:** T-06 uses the physical timestamps returned by this
generator, so there is no need to manually calculate a millisecond
`interval`. The player follows a monotonic browser clock but does
not provide hard real-time guarantees or interpolate configurations.
For a nonuniform multi-waypoint example and interactive checks,
see [Physical-Time Animation Playback](physical-time-playback.md).

## 4. Compare against a quintic trajectory

Generate a rest-to-rest quintic polynomial with **the same endpoints
and duration** as the RR profile. This is a meaningful comparison of
boundary-condition interpolation versus a motion profile with explicit
speed and acceleration limits:

```python
quintic = joint_trajectory(
    q0_rr, qf_rr, rr.t, method="quintic",
)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

quantities = [
    ("q", "Position [rad]"),
    ("qd", "Velocity [rad/s]"),
    ("qdd", "Acceleration [rad/s²]"),
]

for ax, (attribute, label) in zip(axes, quantities):
    ax.plot(
        rr.t, getattr(rr, attribute)[:, 0],
        label="Trapezoidal",
    )
    ax.plot(
        quintic.t, getattr(quintic, attribute)[:, 0],
        "--", label="Quintic",
    )
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.tight_layout()
plt.show()
```

The quintic has zero velocity **and acceleration** at the endpoints.
The trapezoidal profile has zero velocity at the endpoints but
piecewise-constant, potentially discontinuous acceleration.

**Do not infer that the quintic respects the trapezoidal generator's
`v_max` and `a_max`.** Its constraints are the chosen boundary
conditions, not those speed and acceleration limits. To compare maximum
values, inspect its velocity/acceleration curves, or perform a separate
continuous-time extremum analysis.

## 5. Prescribe a longer duration or a stationary hold

A requested duration greater than the minimum uniformly scales the time:

```python
slower = joint_trapezoidal_trajectory(
    q0_rr, qf_rr,
    v_max=v_max, a_max=a_max,
    duration=2 * rr.duration,
    samples=rr.samples,
)

np.testing.assert_allclose(slower.q, rr.q, atol=1e-12)
np.testing.assert_allclose(slower.qd, rr.qd / 2, atol=1e-12)
np.testing.assert_allclose(slower.qdd, rr.qdd / 4, atol=1e-12)
```

Identical endpoints with an explicit positive duration produce a hold:

```python
hold = joint_trapezoidal_trajectory(
    [0.2, -0.1], [0.2, -0.1],
    v_max=v_max, a_max=a_max,
    duration=2.0,
)
assert hold.profile.kind == "stationary"
assert np.all(hold.qd == 0.0)
```

The optional `profile` metadata describes **how the original motion was
generated**. It is immutable, but the result arrays are mutable; editing
those arrays does not update the metadata.

## Suggested student experiments

1. Reduce one joint's `v_max` or `a_max` and identify which coordinate
   determines the shared progress limit.
2. Choose a short displacement and examine when `profile.kind` changes
   from `"trapezoidal"` to `"triangular"`.
3. Compare position, velocity and acceleration for a quintic and a
   trapezoidal profile at equal duration. Explain their endpoint behavior.
4. Double the requested duration and check the velocity/acceleration
   scaling by factors `1/2` and `1/4`.
