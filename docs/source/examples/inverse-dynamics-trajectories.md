# Inverse Dynamics of Joint Trajectories: Quintic vs. Trapezoidal

This notebook-ready example explores **API-002** in Moro 0.7.0.
We generate two rest-to-rest motions of a planar
RR manipulator with **identical joint-space endpoints and duration**,
then compare the generalized torques required by the ideal dynamics.

This is **not** an optimization or actuator feasibility test. A
trapezoidal velocity profile has acceleration jumps and is not jerk
limited; a quintic rest-to-rest polynomial has zero acceleration at
the endpoints but does not automatically satisfy velocity,
acceleration or torque limits.

## 1. Define the physical RR robot

The link geometry, center-of-mass locations, masses, inertias and gravity
must be defined before numerical dynamics can be evaluated.

~~~python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display

from moro import Robot
from moro.abc import q1, q2
from moro.dynamics import inverse_dynamics, inverse_dynamics_trajectory
from moro.trajectory import joint_trajectory, joint_trapezoidal_trajectory

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
)

robot.masses = [1.2, 0.8]  # kg
robot.cm_positions = [
    (-0.5, 0, 0),           # m, expressed in the link-1 frame
    (-0.4, 0, 0),           # m, expressed in the link-2 frame
]
robot.inertia_tensors = [
    sp.diag(0, 0, 0.08),    # kg m^2
    sp.diag(0, 0, 0.04),
]
robot.gravity = (0, -9.81, 0)  # m/s^2 in the base frame
~~~

The model is planar and both joints are revolute, so the computed
generalized efforts are **torques in N·m**.

## 2. Generate two motions of the same duration

First generate a synchronized trapezoidal profile with explicit joint
velocity and acceleration limits. Then generate a quintic trajectory at
the **same sample times**, removing duration and sampling-grid differences.

~~~python
q0 = np.array([0.0, 0.0])
qf = np.array([1.2, -0.8])

trapezoidal = joint_trapezoidal_trajectory(
    q0, qf,
    v_max=[0.6, 0.4],       # rad/s
    a_max=[1.0, 0.8],       # rad/s^2
    samples=101,
)

quintic = joint_trajectory(
    q0, qf,
    trapezoidal.t,
    method="quintic",
)

assert trapezoidal.samples == quintic.samples
assert np.array_equal(trapezoidal.t, quintic.t)
assert np.allclose(trapezoidal.q[[0, -1]], quintic.q[[0, -1]])

print(f"Duration: {trapezoidal.duration:.2f} s")
print(f"Profile: {trapezoidal.profile.kind}")
~~~

The profiles follow the same straight path in joint space, but their
velocity and acceleration histories differ. The limits used to
**construct the trapezoidal trajectory** do not constrain the quintic.

## 3. Compute the required torque histories

Both **JointTrajectory** objects expose the time-major histories
**q**, **qd**, and **qdd**, which is the contract accepted by
**inverse_dynamics_trajectory()**. The symbolic mass, Coriolis and
gravity expressions are prepared **once per call** and evaluated at
every sample.

~~~python
tau_trapezoidal = inverse_dynamics_trajectory(robot, trapezoidal)
tau_quintic = inverse_dynamics_trajectory(robot, quintic)

assert tau_trapezoidal.shape == (trapezoidal.samples, robot.dof)
assert tau_quintic.shape == (quintic.samples, robot.dof)

print("Peak |torque|, trapezoidal [N m]:",
      np.max(np.abs(tau_trapezoidal), axis=0))
print("Peak |torque|, quintic [N m]:",
      np.max(np.abs(tau_quintic), axis=0))
~~~

These are the efforts required to execute **each prescribed motion
under the ideal model**. They are not simulated applied torques,
and the calculation does not check motor ratings, power, friction,
external contact loads, or joint limits.

## 4. Compare position, velocity, acceleration and torque

~~~python
fig, axes = plt.subplots(4, 2, sharex="col", figsize=(11, 10))

for joint in range(robot.dof):
    for row, (name, units) in enumerate([
        ("q", "rad"),
        ("qd", "rad/s"),
        ("qdd", "rad/s²"),
    ]):
        axes[row, joint].plot(
            trapezoidal.t, getattr(trapezoidal, name)[:, joint],
            label="Trapezoidal",
        )
        axes[row, joint].plot(
            quintic.t, getattr(quintic, name)[:, joint],
            label="Quintic", linestyle="--",
        )
        axes[row, joint].set_ylabel(f"{name} [{units}]")

    axes[3, joint].plot(
        trapezoidal.t, tau_trapezoidal[:, joint],
        label="Trapezoidal",
    )
    axes[3, joint].plot(
        quintic.t, tau_quintic[:, joint],
        label="Quintic", linestyle="--",
    )
    axes[3, joint].set_ylabel("Torque [N·m]")
    axes[3, joint].set_xlabel("Time [s]")
    axes[0, joint].set_title(f"Joint {joint + 1}")

for ax in axes.flat:
    ax.grid(True)
    ax.legend()

fig.tight_layout()
plt.show()
~~~

Observe the beginning and end of each motion, and the acceleration
changes at the trapezoidal phase boundaries. A nonzero torque at rest
may also be required just to balance **gravity**.

Do not interpret peak values as complete actuator sizing: a realistic
design may require reflected motor inertia, friction, external loads,
torque-speed curves, thermal limits and safety factors.

## 5. Verify consistency with single-state inverse dynamics

At any selected sample, the new trajectory operation must reproduce
the existing **inverse_dynamics()** result:

~~~python
index = trapezoidal.samples // 2

reference_tau = inverse_dynamics(
    robot,
    trapezoidal.q[index],
    trapezoidal.qd[index],
    trapezoidal.qdd[index],
)

np.testing.assert_allclose(
    tau_trapezoidal[index], reference_tau,
    rtol=1e-10, atol=1e-10,
)
~~~

This establishes mathematical equivalence with the single-state API,
not an independent experimental validation of the physical model.

## 6. Optional: visualize both motions in a notebook

**RobotVisualizer.animate()** accepts the position history directly
with its corresponding physical timestamps:

~~~python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
display(viz.animate(trapezoidal.q, times=trapezoidal.t))
display(viz.animate(quintic.q, times=quintic.t))
~~~

The dynamics module does not depend on visualization. Changing
timestamp values while **holding q, qd and qdd fixed** cannot change
the resulting efforts; the derivatives already encode time dependence.

## Further exercises

1. Change the duration and distinguish inertial effects from gravity.
2. Turn gravity off and recompute the effort histories.
3. Increase the link masses or inertias and compare peak torques.
4. Reverse the second joint's displacement with the same duration.

For details, see [Dynamics](../user-guide/dynamics.md),
[Trajectory Generation](../user-guide/trajectories.md) and
[0.7.0 release notes](../release-notes/0.7.0.md).
