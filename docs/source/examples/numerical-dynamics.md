# Numerical Dynamics and Simulation

This example connects Moro's symbolic robot model with numerical inverse dynamics, forward dynamics and simulation.

## 1. Inverse and forward dynamics

Assume a fully defined planar 2R robot named `robot`.

```python
import numpy as np

from moro.dynamics import forward_dynamics, inverse_dynamics

q = np.array([0.3, -0.4])
qd = np.array([0.5, -0.2])
qdd_ref = np.array([0.7, -0.6])

tau = inverse_dynamics(
    robot,
    q,
    qd,
    qdd_ref,
)

qdd_check = forward_dynamics(
    robot,
    q,
    qd,
    tau,
)
```

The two operations answer complementary questions:

- inverse dynamics: what generalized force is required for a prescribed acceleration?;
- forward dynamics: what acceleration results from a prescribed generalized force?.

For a consistent model, `qdd_check` should agree numerically with `qdd_ref`.

## 2. Free motion under gravity

```python
from moro.dynamics import simulate

solution = simulate(
    robot,
    (0.0, 5.0),
    q0=[0.4, -0.2],
    qd0=[0.0, 0.0],
    tau=None,
    t_eval=np.linspace(0.0, 5.0, 501),
)
```

`tau=None` means zero applied generalized force. Gravity remains part of the robot model, so the system may accelerate from rest.

Inspect:

```python
solution.q
solution.qd
solution.qdd
solution.success
solution.message
```

## 3. Time-varying generalized force

A callable can depend on time and state:

```python
def tau(t, q, qd):
    return np.array([
        2.0 * np.sin(3.0 * t),
        0.0,
    ])

solution = simulate(
    robot,
    (0.0, 3.0),
    q0=[0.0, 0.0],
    qd0=[0.0, 0.0],
    tau=tau,
    t_eval=np.linspace(0.0, 3.0, 301),
)
```

Moro passes `q` and `qd` to the callable as one-dimensional NumPy arrays.

## 4. Simple feedback law

The callable interface can also represent a lightweight feedback law:

```python
Kp = np.diag([8.0, 6.0])
Kd = np.diag([3.0, 2.5])
q_ref = np.array([0.5, -0.3])

def control(t, q, qd):
    return -Kp @ (q - q_ref) - Kd @ qd
```

This is user-defined generalized-force logic, not a dedicated controller framework.

## 5. Prescribed trajectory to required generalized force

Trajectory generation and dynamics remain separate:

```python
from moro.trajectory import joint_trajectory

traj = joint_trajectory(
    [0.0, 0.0],
    [0.6, -0.4],
    np.linspace(0.0, 2.0, 101),
    method="quintic",
)

tau_history = np.array([
    inverse_dynamics(robot, q, qd, qdd)
    for q, qd, qdd in zip(
        traj.q,
        traj.qd,
        traj.qdd,
    )
])
```

The result has shape `(traj.samples, robot.dof)`.

## 6. Visualization

`DynamicsSolution.q` follows the same time-major `(N, dof)` convention used by joint trajectories:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
viz.animate(solution.q)
```

The animation backend controls playback interval independently from the physical time vector `solution.t`.
