# Dynamics

Moro models serial-manipulator dynamics with the standard rigid-body manipulator equation

$$
M(q)\ddot q + C(q,\dot q)\dot q + G(q)=\tau.
$$

The `Robot` class constructs the symbolic terms. The `moro.dynamics` module evaluates them numerically and integrates the resulting ordinary differential equation.

## Generalized coordinates

For an $n$-DOF manipulator,

$$
q=\begin{bmatrix}q_1&\cdots&q_n\end{bmatrix}^T,
$$

with generalized velocity and acceleration

$$
\dot q=\frac{dq}{dt},
\qquad
\ddot q=\frac{d^2q}{dt^2}.
$$

The same notation applies to revolute and prismatic joints; only the physical units differ.

## Kinetic energy

For link $i$, with mass $m_i$, center-of-mass linear velocity $v_{C_i}$, angular velocity $\omega_i$, orientation $R_i^0$, and inertia tensor $I_{C_i}^{i}$ expressed at the center of mass,

$$
K_i=
\frac12m_i v_{C_i}^Tv_{C_i}
+
\frac12\omega_i^T
R_i^0 I_{C_i}^{i}(R_i^0)^T
\omega_i.
$$

The total kinetic energy is

$$
K=\sum_i K_i.
$$

Using center-of-mass Jacobians,

$$
v_{C_i}=J_{v_i}\dot q,
\qquad
\omega_i=J_{\omega_i}\dot q,
$$

which leads to

$$
K=\frac12\dot q^T M(q)\dot q.
$$

Moro constructs

$$
M(q)=\sum_i
\left[
m_iJ_{v_i}^TJ_{v_i}
+
J_{\omega_i}^T R_i^0 I_{C_i}^{i}(R_i^0)^T J_{\omega_i}
\right].
$$

## Potential energy and gravity

With gravity vector $g$ expressed in the base frame and center-of-mass position $r_{C_i}$,

$$
P_i=-m_i g^T r_{C_i}.
$$

The total potential energy is

$$
P=\sum_i P_i.
$$

The generalized gravity vector is

$$
G(q)=\nabla_q P(q).
$$

## Euler-Lagrange equations

Define the Lagrangian

$$
\mathcal L(q,\dot q)=K(q,\dot q)-P(q).
$$

For each generalized coordinate,

$$
\frac{d}{dt}
\left(
\frac{\partial\mathcal L}{\partial\dot q_i}
\right)
-
\frac{\partial\mathcal L}{\partial q_i}
=\tau_i.
$$

In Moro 0.5.0 these equations are returned by

```python
robot.euler_lagrange_equations()
```

## Coriolis matrix

Moro forms the velocity-dependent terms through Christoffel symbols of the first kind,

$$
c_{ijk}=\frac12
\left(
\frac{\partial M_{ij}}{\partial q_k}
+
\frac{\partial M_{ik}}{\partial q_j}
-
\frac{\partial M_{jk}}{\partial q_i}
\right).
$$

Then

$$
C_{ij}(q,\dot q)=\sum_k c_{ijk}\dot q_k.
$$

Therefore the velocity-dependent generalized-force term is

$$
C(q,\dot q)\dot q.
$$

## Matrix dynamic model

Collecting inertia, Coriolis/centrifugal and gravity effects gives

$$
\boxed{
M(q)\ddot q+C(q,\dot q)\dot q+G(q)=\tau
}.
$$

In Moro 0.5.0 this symbolic matrix equation is returned by

```python
robot.dynamic_model()
```

`dynamic_model_matrix_form()` remains temporarily as a deprecated alias.

## Inverse dynamics

If $q$, $\dot q$ and $\ddot q$ are prescribed, the required generalized force is

$$
\boxed{
\tau=M(q)\ddot q+C(q,\dot q)\dot q+G(q)
}.
$$

Numerically:

```python
inverse_dynamics(robot, q, qd, qdd)
```

## Forward dynamics

If $q$, $\dot q$ and $\tau$ are known, forward dynamics solves

$$
M(q)\ddot q=	au-C(q,\dot q)\dot q-G(q).
$$

Thus

$$
\boxed{
\ddot q=M(q)^{-1}
\left[
\tau-C(q,\dot q)\dot q-G(q)
\right]
}.
$$

The equation above is mathematical notation. Numerically Moro uses a linear solve,

```python
np.linalg.solve(M, rhs)
```

rather than explicitly forming $M^{-1}$.

A singular mass matrix is treated as a dynamic-model failure; no pseudoinverse fallback is used.

## State-space form for integration

Define

$$
x=
\begin{bmatrix}
q\\
\dot q
\end{bmatrix}.
$$

Then

$$
\dot x=
\begin{bmatrix}
\dot q\\
M(q)^{-1}
\left[
\tau(t,q,\dot q)-C(q,\dot q)\dot q-G(q)
\right]
\end{bmatrix}.
$$

This first-order system is the quantity integrated by `scipy.integrate.solve_ivp` in `moro.dynamics.simulate()`.

## Numerical model preparation

Moro does not derive a second, independent numerical dynamic model. Instead it reuses symbolic

$$
M(q),\qquad C(q,\dot q),\qquad G(q),
$$

applies fixed parameter substitutions, validates unresolved symbols and creates numerical callables with SymPy `lambdify`.

For simulation, these numerical callables are prepared once and reused inside the ODE right-hand side.

## Applied generalized force

Simulation supports

$$
\tau=0,
$$

a constant generalized-force vector, or a callable

```python
tau(t, q, qd)
```

which permits time-dependent inputs and simple feedback laws without introducing a controller framework.

## Acceleration reconstruction

The ODE solver returns sampled $q(t)$ and $\dot q(t)$. Moro reconstructs

$$
\ddot q(t_k)
$$

at every returned sample by evaluating forward dynamics at the same $(t_k,q_k,\dot q_k)$ state.

It does not estimate acceleration by finite differences.

## Energy in conservative motion

For a system with no applied generalized force, no dissipation and no contacts, the mechanical energy

$$
E(t)=K(t)+P(t)
$$

is conserved analytically.

A numerical integrator such as RK45 may show small energy drift; conservation should therefore be interpreted within the requested numerical tolerances rather than as exact floating-point equality.

## Physical scope

The 0.5.0 numerical model is unconstrained rigid-body joint dynamics. It does not include friction, contacts, impacts, actuator dynamics, torque saturation, joint-stop physics, closed-chain constraints or collision response.

Configured joint limits are therefore not automatically enforced during forward simulation.
