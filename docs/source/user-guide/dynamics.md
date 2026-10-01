# Dynamics

Moro separates **symbolic model construction** from **numerical dynamics**.

The `Robot` class remains the symbolic layer. It constructs energies and the standard manipulator terms

$$
M(q),\qquad C(q,\dot q),\qquad G(q).
$$

The `moro.dynamics` module evaluates those symbolic expressions numerically and integrates the equations of motion.

## Defining the physical model

A robot must first define its kinematic structure and the physical quantities required by the dynamic model.

```python
import sympy as sp

from moro import Robot
from moro.abc import q1, q2

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.8, 0, 0, q2, "r"),
)

robot.masses = [1.2, 0.8]
robot.cm_positions = [
    (-0.5, 0, 0),
    (-0.4, 0, 0),
]
robot.inertia_tensors = [
    sp.diag(0, 0, 0.08),
    sp.diag(0, 0, 0.04),
]
robot.gravity = (0, -9.81, 0)
```

For numerical dynamics, joint coordinates must be time-dependent SymPy quantities. The generalized coordinates from `moro.abc` satisfy this requirement.

## Symbolic dynamics

The main symbolic quantities remain available on `Robot`:

```python
K = robot.kinetic_energy()
P = robot.potential_energy()
L = robot.lagrangian()
M = robot.inertia_matrix()
C = robot.coriolis_matrix()
G = robot.gravity_vector()
```

### Euler-Lagrange equations

Moro 0.5.0 exposes the per-joint Euler-Lagrange equations explicitly through:

```python
equations = robot.euler_lagrange_equations()
```

Each equation has the form

$$
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot q_i}
\right)
-
\frac{\partial L}{\partial q_i}
=
\tau_i.
$$

### Matrix dynamic model

The standard manipulator equation is now returned by:

```python
model = robot.dynamic_model()
```

with

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau.
$$

:::{important}
This is a breaking API change in Moro 0.5.0. In 0.4.x, `dynamic_model()` returned the Euler-Lagrange equation list. Use `euler_lagrange_equations()` for that representation in 0.5.0.
:::

`dynamic_model_matrix_form()` remains temporarily available as a deprecated alias of `dynamic_model()`.

## Numerical inverse dynamics

Inverse dynamics answers:

> What generalized force is required to produce a prescribed acceleration?

Use:

```python
from moro.dynamics import inverse_dynamics

tau = inverse_dynamics(
    robot,
    q=[0.3, -0.4],
    qd=[0.5, -0.2],
    qdd=[0.7, -0.6],
)
```

The calculation is

$$
\tau
=
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q).
$$

The result is a one-dimensional NumPy array with shape `(robot.dof,)`.

For a 1-DOF robot, scalar state values are accepted as a convenience. Scalar broadcasting is intentionally rejected for multi-DOF models.

## Numerical forward dynamics

Forward dynamics answers:

> What acceleration results from the applied generalized force?

```python
from moro.dynamics import forward_dynamics

qdd = forward_dynamics(
    robot,
    q=[0.3, -0.4],
    qd=[0.5, -0.2],
    tau=tau,
)
```

Internally Moro solves

$$
M(q)\ddot q
=
\tau-C(q,\dot q)\dot q-G(q)
$$

using a numerical linear solve. Moro does not form `inv(M)` and does not silently fall back to a pseudoinverse when the mass matrix is singular.

## Symbolic model parameters

Fixed symbolic quantities can be supplied through `parameters`:

```python
params = {
    m1: 2.0,
    m2: 1.5,
    g: 9.81,
}

tau = inverse_dynamics(
    robot,
    q,
    qd,
    qdd,
    parameters=params,
)
```

`parameters` is intended for fixed model quantities such as geometry, mass, inertia and gravity constants. Joint state values are supplied separately through `q`, `qd` and `qdd`.

The original symbolic robot is not mutated.

## State derivative

The state convention is

$$
x=
\begin{bmatrix}
q\\
\dot q
\end{bmatrix}.
$$

Use:

```python
from moro.dynamics import state_derivative

xd = state_derivative(
    robot,
    t=0.5,
    state=[q1_value, q2_value, qd1_value, qd2_value],
    tau=[1.0, 0.0],
)
```

The returned vector is

$$
\dot x=
\begin{bmatrix}
\dot q\\
\ddot q
\end{bmatrix}.
$$

## Generalized-force inputs

`state_derivative()` and `simulate()` accept three generalized-force forms.

### Zero applied force

```python
tau=None
```

means

$$
\tau=0.
$$

### Constant generalized force

```python
tau=[1.0, -0.5]
```

is applied throughout the simulation.

### Callable generalized force

```python
def tau(t, q, qd):
    return np.array([
        2.0 * np.sin(t),
        0.0,
    ])
```

The callable receives `q` and `qd` as one-dimensional NumPy arrays.

This also permits simple user-defined feedback laws:

```python
def control(t, q, qd):
    return -Kp @ (q - q_ref) - Kd @ qd
```

This callable interface is not a dedicated controller framework.

## Time-domain simulation

Use `simulate()` to integrate

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau(t,q,\dot q).
$$

```python
import numpy as np
from moro.dynamics import simulate

t_eval = np.linspace(0.0, 5.0, 501)

solution = simulate(
    robot,
    (0.0, 5.0),
    q0=[0.3, -0.2],
    qd0=[0.0, 0.0],
    tau=None,
    t_eval=t_eval,
)
```

Moro uses `scipy.integrate.solve_ivp`; the default solver is `RK45`.

Optional `rtol`, `atol` and `max_step` values are forwarded only when explicitly supplied.

`t_eval` selects output times. It does not force the internal integration step size.

## DynamicsSolution

A successful or partially successful simulation returns a `DynamicsSolution`:

```python
solution.t
solution.q
solution.qd
solution.qdd
solution.success
solution.message
solution.method
```

The numerical arrays are time-major:

```text
t.shape   == (N,)
q.shape   == (N, dof)
qd.shape  == (N, dof)
qdd.shape == (N, dof)
```

Derived properties are:

```python
solution.samples
solution.duration
solution.dof
```

`qdd` is reconstructed from the forward dynamic model at every returned state. It is not obtained by finite-differencing velocity samples.

If SciPy returns a normal integration failure, Moro preserves the valid partial trajectory and sets `solution.success = False`. Invalid model evaluation, malformed force inputs and singular mass matrices remain exceptions.

## Gravity-driven motion

A simple free-motion simulation uses `tau=None`:

```python
solution = simulate(
    robot,
    (0.0, 2.0),
    q0=[0.3, -0.2],
    qd0=[0.0, 0.0],
    tau=None,
    t_eval=np.linspace(0.0, 2.0, 201),
)
```

If gravity generates a nonzero generalized force, the robot will accelerate from rest.

## From a prescribed trajectory to required generalized force

The trajectory and dynamics modules intentionally remain separate.

```python
from moro.trajectory import joint_trajectory
from moro.dynamics import inverse_dynamics

traj = joint_trajectory(
    q0,
    qf,
    t,
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

This computes the generalized-force history required by the prescribed joint trajectory without introducing a trajectory-wide inverse-dynamics API.

## Visualization

Simulation output follows the same `(N, dof)` convention as joint trajectories, so it can be animated directly:

```python
from moro.visualization import RobotVisualizer

viz = RobotVisualizer(robot)
viz.animate(solution.q)
```

The dynamics module does not depend on visualization.

The animation interval is still controlled by the visualization backend; nonuniform physical timing in `solution.t` is not reproduced automatically.

## Joint limits and physical scope

Numerical dynamics does **not** enforce `robot.joint_limits`.

A simulated state may cross a configured limit because clipping is not a physically valid model of a mechanical stop.

Moro 0.5.0 does not introduce contact dynamics, impacts, friction, actuator dynamics, torque saturation, constrained dynamics or a controller framework.
