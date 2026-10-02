# Dynamic Model of a Planar 2R Manipulator

This example develops the symbolic dynamic model of a planar two-link manipulator and connects it with the numerical dynamics API introduced in Moro 0.5.0.

We will:

1. define the robot and its physical parameters;
2. inspect kinetic and potential energy;
3. obtain $M(q)$, $C(q,\dot q)$, and $G(q)$;
4. obtain both Euler-Lagrange and matrix-form equations;
5. evaluate inverse dynamics numerically.

## Robot model

For dynamic modeling, use time-dependent generalized coordinates. The variables in `moro.abc` are suitable for this purpose.

```python
import sympy as sp

from moro import Robot
from moro.abc import t, q1, q2, m1, m2, l1, l2, lc1, lc2, g

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Assign masses, center-of-mass positions, inertia tensors, and gravity:

```python
I1, I2 = sp.symbols("I1 I2", positive=True)

robot.masses = [m1, m2]
robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
robot.inertia_tensors = [
    sp.diag(0, 0, I1),
    sp.diag(0, 0, I2),
]
robot.gravity = (0, -g, 0)
```

The sign of each center-of-mass coordinate depends on the selected link-frame convention. With the frames used here, each center of mass lies along the negative local $x$ direction from the link-frame origin.

## Energy expressions

The link kinetic energies are:

```python
K1 = robot.link_kinetic_energy(1)
K2 = robot.link_kinetic_energy(2)
```

and the total kinetic energy is:

```python
K = robot.kinetic_energy()
```

For link $i$,

$$
K_i=
\frac12 m_i v_{C_i}^T v_{C_i}
+
\frac12
\omega_i^T
R_i^0 I_{C_i}^{i}(R_i^0)^T
\omega_i.
$$

The total potential energy is:

```python
P = robot.potential_energy()
```

and the Lagrangian is:

```python
L = robot.lagrangian()
```

with

$$
\mathcal L=K-P.
$$

## Inertia matrix

The symbolic inertia matrix is:

```python
M = robot.inertia_matrix()
```

For this 2-DOF robot,

$$
M(q)\in\mathbb R^{2\times2}.
$$

It is constructed from the translational and rotational center-of-mass Jacobians of the links.

## Coriolis and centrifugal terms

The Coriolis matrix is:

```python
C = robot.coriolis_matrix()
```

Construct the generalized-velocity vector:

```python
qd = sp.Matrix([
    q1.diff(t),
    q2.diff(t),
])
```

The complete velocity-dependent contribution is then:

```python
velocity_terms = C * qd
```

Different valid Coriolis-matrix conventions may exist; the physically relevant quantity is the generalized-force product $C(q,\dot q)\dot q$.

## Gravity vector

The generalized gravity vector is:

```python
G = robot.gravity_vector()
```

with

$$
G(q)=\nabla_q P(q).
$$

## Euler-Lagrange equations

Moro 0.5.0 exposes the per-joint Euler-Lagrange equations through:

```python
equations = robot.euler_lagrange_equations()
```

The result contains one SymPy equation per generalized coordinate:

```python
for equation in equations:
    sp.pprint(equation)
```

Each equation has the form

$$
\frac{d}{dt}
\left(
\frac{\partial\mathcal L}{\partial\dot q_i}
\right)
-
\frac{\partial\mathcal L}{\partial q_i}
=
\tau_i.
$$

## Matrix dynamic model

The standard manipulator equation is returned directly by:

```python
matrix_equation = robot.dynamic_model()
matrix_equation
```

It has the form

$$
\boxed{
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau
}.
$$

:::{important}
In Moro 0.4.x, `dynamic_model()` returned the per-joint Euler-Lagrange equations. Starting with Moro 0.5.0, use `euler_lagrange_equations()` for that representation.
:::

The old helper

```python
robot.dynamic_model_matrix_form()
```

remains temporarily as a deprecated compatibility alias of `dynamic_model()`.

## Explicit torque expression

Define the acceleration vector:

```python
qdd = sp.Matrix([
    q1.diff(t, 2),
    q2.diff(t, 2),
])
```

The symbolic generalized-force expression is:

```python
tau_expr = M * qdd + C * qd + G
```

This is the inverse-dynamics relationship:

$$
\tau=M(q)\ddot q+C(q,\dot q)\dot q+G(q).
$$

## Numerical symbolic substitution

For example, use:

```python
values = {
    l1: 1.0,
    l2: 0.8,
    lc1: 0.5,
    lc2: 0.4,
    m1: 2.0,
    m2: 1.5,
    I1: 0.15,
    I2: 0.08,
    g: 9.81,
    q1: sp.pi / 6,
    q2: -sp.pi / 9,
    q1.diff(t): 0.4,
    q2.diff(t): -0.2,
    q1.diff(t, 2): 0.5,
    q2.diff(t, 2): 0.1,
}
```

Then:

```python
M_num = M.subs(values).evalf()
C_num = C.subs(values).evalf()
G_num = G.subs(values).evalf()
tau_num = tau_expr.subs(values).evalf()
```

This workflow is useful when the symbolic expressions themselves are part of the analysis.

## Numerical inverse dynamics API

Moro 0.5.0 also provides a dedicated numerical layer:

```python
from moro.dynamics import inverse_dynamics

parameters = {
    l1: 1.0,
    l2: 0.8,
    lc1: 0.5,
    lc2: 0.4,
    m1: 2.0,
    m2: 1.5,
    I1: 0.15,
    I2: 0.08,
    g: 9.81,
}

tau = inverse_dynamics(
    robot,
    q=[float(sp.pi / 6), float(-sp.pi / 9)],
    qd=[0.4, -0.2],
    qdd=[0.5, 0.1],
    parameters=parameters,
)
```

The result is a NumPy array with shape `(2,)`.

The original symbolic `Robot` is not modified by the numerical parameter substitution.

## Forward dynamics consistency

The same generalized force can be passed to forward dynamics:

```python
from moro.dynamics import forward_dynamics

qdd_check = forward_dynamics(
    robot,
    q=[float(sp.pi / 6), float(-sp.pi / 9)],
    qd=[0.4, -0.2],
    tau=tau,
    parameters=parameters,
)
```

Within numerical tolerance, `qdd_check` should reproduce the prescribed acceleration `[0.5, 0.1]`.

This round trip illustrates the relation between symbolic model construction and Moro's numerical dynamics layer.

## Summary

For a serial manipulator with physical parameters assigned, the main symbolic dynamics workflow is:

```text
Robot
  |
  +--> kinetic_energy()
  +--> potential_energy()
  +--> lagrangian()
  +--> inertia_matrix()      -> M(q)
  +--> coriolis_matrix()     -> C(q, qd)
  +--> gravity_vector()      -> G(q)
  +--> euler_lagrange_equations()
  +--> dynamic_model()       -> matrix manipulator equation
```

For numerical evaluation and simulation, use the functions in `moro.dynamics`.
