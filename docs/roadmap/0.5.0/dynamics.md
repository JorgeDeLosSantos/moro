# Direct dynamics and numerical simulation

## Status

Accepted as a candidate feature for Moro 0.5.0.

## Objective

Extend Moro from symbolic dynamic-model construction to numerical inverse/forward dynamics and time-domain simulation.

The architectural split is:

```text
Robot / core
├── kinetic_energy()
├── potential_energy()
├── lagrangian()
├── inertia_matrix()
├── coriolis_matrix()
├── gravity_vector()
├── euler_lagrange_equations()   [planned rename]
└── dynamic_model()              [planned new meaning]

moro.dynamics
├── inverse_dynamics()
├── forward_dynamics()
├── state_derivative()
├── simulate()
└── DynamicsSolution
```

Symbolic model construction/inspection remains on `Robot`; `moro.dynamics` is the numerical layer.

## Symbolic API rename

Planned 0.5.0 meaning:

```text
CURRENT                         PLANNED
dynamic_model()             -> euler_lagrange_equations()
dynamic_model_matrix_form() -> dynamic_model()
```

`euler_lagrange_equations()` returns one Euler-Lagrange equation per joint. `dynamic_model()` returns the standard matrix form

\[
M(q)\ddot q+C(q,\dot q)\dot q+G(q)=\tau.
\]

Backward-compatibility/deprecation policy is deferred; a temporary deprecation period is preferred.

## SciPy

SciPy becomes a required dependency. Numerical integration will use `scipy.integrate.solve_ivp`.

## Preliminary public API

```python
inverse_dynamics(robot, q, qd, qdd, *, parameters=None)
```

Evaluates

\[
\tau=M(q)\ddot q+C(q,\dot q)\dot q+G(q).
\]

Returns the generalized-force vector directly.

```python
forward_dynamics(robot, q, qd, tau, *, parameters=None)
```

Evaluates

\[
\ddot q=M(q)^{-1}[\tau-C(q,\dot q)\dot q-G(q)].
\]

Implementation must use a numerical linear solve, not explicit matrix inversion. Returns acceleration directly.

```python
state_derivative(robot, t, state, tau=None, *, parameters=None)
```

For `state=[q; qd]`, returns `[qd; qdd]`. This is a public numerical operation, not a symbolic state-space constructor.

```python
simulate(
    robot,
    t_span,
    q0,
    qd0=None,
    *,
    tau=None,
    parameters=None,
    t_eval=None,
    method="RK45",
    rtol=None,
    atol=None,
    max_step=None,
)
```

If `qd0` is omitted, zero initial velocity is assumed.

## Generalized-force input

For `state_derivative()` and `simulate()`, `tau` may be:

- `None` -> zero generalized force;
- a constant numerical vector;
- a callable `tau(t, q, qd)`.

This enables time-varying inputs and simple user-defined feedback laws without introducing a controller framework.

## `DynamicsSolution`

`simulate()` returns a structured result with at least:

- `t`;
- `q`;
- `qd`;
- `qdd`;
- `success`;
- `message`.

Preferred shapes are time-major `(N, dof)` for `q`, `qd`, and `qdd`. `qdd` may be reconstructed from returned states through forward dynamics.

## Performance and symbolic parameters

Symbolic robot parameters are supported through `parameters`. Repeated `.subs()` inside every ODE RHS evaluation should be avoided. The expected direction is to build symbolic `M`, `C`, `G`, substitute fixed parameters once, and create reusable numerical callables, e.g. through `sympy.lambdify`.

## Joint limits

The initial simulator does not physically enforce joint limits. Clipping positions/velocities is not a valid model of stops or impacts, so simulation remains unconstrained and this limitation should be documented.

## Visualization relationship

The dynamics module should not depend on visualization. Result shapes should nevertheless make `visualizer.animate(solution.q)` natural.

## Examples and tests

Cover inverse/forward dynamics consistency, gravity motion, constant and time-varying torque, a simple PD-like callable, planar 2R simulation, symbolic parameters, `t_eval`, result consistency, and visualization reuse.

## Explicitly outside 0.5.0

- symbolic state-space construction;
- friction models;
- external Cartesian wrenches;
- contacts, collision response, impacts;
- physical joint-limit enforcement;
- actuator models and torque saturation;
- dedicated controller classes / built-in computed torque;
- constrained/closed-chain dynamics;
- advanced event handling;
- multibody contact simulation.

## Deferred detailed-design decisions

Validation/messages, deprecation mechanics, exact `DynamicsSolution` invariants, accepted input shapes/dtypes, ODE tolerance defaults/options, failed-integration policy, unresolved-symbol handling, caching/compilation location, `qdd` reconstruction, and visualization interoperability remain deferred.
