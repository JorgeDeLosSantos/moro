# API-002 — Inverse dynamics of sampled joint motion

> **Status:** B1 implementation integrated in PR #55; B2 numerical
> regression coverage integrated in PR #56. B3 adds user documentation,
> an executable RR quintic/trapezoidal teaching example and a notebook
> smoke test. Moro 0.7.0 as a whole remains unreleased.

## Purpose

Provide a small numerical bridge from sampled robot motion to the
generalized actuator efforts required by the *ideal* manipulator model:

```python
from moro.dynamics import inverse_dynamics_trajectory

tau_history = inverse_dynamics_trajectory(
    robot, motion, parameters=params,
)
```

For every sample `i`, compute

```text
tau[i] = M(q[i]) @ qdd[i] + C(q[i], qd[i]) @ qd[i] + G(q[i])
```

using precisely the same model as `inverse_dynamics()`, without its
per-call symbolic preparation overhead.

## Input and output contract

- `motion` uses structural compatibility: it must have `q`, `qd`
  and `qdd` attributes; no inheritance from a particular trajectory
  type is required.
- Each attribute must be numeric, real, finite and have shape
  `(N, robot.dof)`. All three histories must have identical shapes.
- `N >= 1`. A single sample is allowed as a two-dimensional history
  of shape `(1, dof)`. No scalar broadcasting or automatic finite
  differences are performed.
- `motion.t` is optional. If present and non-`None`, it must be a
  one-dimensional finite array with exactly `N` values, strictly
  increasing for `N > 1`. The evaluation does *not* use the timestamps;
  the provided velocities and accelerations already encode timing.
- `parameters` follows the existing `inverse_dynamics()` symbolic
  substitution semantics. The robot and caller-owned inputs are not
  modified.
- Return an independent NumPy `float` array of shape `(N, robot.dof)`.
  Revolute coordinates produce torques and prismatic coordinates produce
  forces, in their respective generalized units.
- Invalid inputs raise `ValueError` naming the offending history.
  A per-sample numerical evaluation error identifies the zero-based
  sample index.

The function supports `moro.trajectory.JointTrajectory`,
`moro.dynamics.DynamicsSolution`, and externally supplied compatible
objects, but does not import either trajectory type.

## Implementation

1. Validate and defensively copy the arrays with the existing private
   numerical dynamics normalization utilities, before compiling M/C/G.
2. Prepare the symbolic-to-numerical model once using
   `_prepare_numerical_model(robot, parameters=parameters)`.
3. Reuse `_inverse_dynamics_prepared(model, q, qd, qdd)` for each sample.
4. Accumulate a time-major `(N, robot.dof)` array without storing a
   new solution dataclass.

This keeps the existing `inverse_dynamics()` single-state API intact,
and avoids a new public compiled-model type, mandatory vectorization,
changes to `moro.trajectory`, or an implicit time-differentiation API.

## Physical interpretation and future extensions

The first increment returns the ideal generalized efforts from M/C/G.
Friction, externally applied wrenches, contact forces, actuator models
and torque-limit feasibility are **not** included.

When these contributions are designed later, they must use consistent
force and sign conventions in `inverse_dynamics()`,
`inverse_dynamics_trajectory()`, `forward_dynamics()`, and
`simulate()`. In particular, externally applied spatial wrenches
should be converted by the appropriate point Jacobian and reference
frame, rather than treated as arbitrary scalar joint loads. This is a
separate design step, not part of B1–B3.

## Planned validation increments

- **B1:** Public function, input validation, one preparation per call,
  basic R/P and RR comparisons, and a failure diagnostic.
- **B2:** Independent one-axis analytical rotor check; R/RR/RP
  samplewise inverse-dynamics and forward-dynamics round trips;
  symbolic geometry/mass/inertia/gravity parameter substitution without
  mutation; missing-parameter errors; one-sample and partial
  DynamicsSolution compatibility; trapezoidal JointTrajectory support;
  invariance to valid timestamp changes; malformed state/time rejection
  before compilation; finite-input overflow diagnostics naming the
  sample index; and spies asserting one model preparation, three
  symbolic lambdify compilations and one evaluation per sample.
  These checks deliberately do not benchmark elapsed wall time.
- **B3:** Update the Dynamics User Guide and the numerical-dynamics
  example to use the new history API, document input/output and future
  loading scope, and add [a standalone RR comparison](../../source/examples/inverse-dynamics-trajectories.md)
  of quintic and trapezoidal joint trajectories with matching endpoints
  and duration. Execute all example code cells in a browser-free test;
  interactive Jupyter/Colab rendering remains a manual acceptance check.
  Add the feature to [unreleased development notes](../../source/release-notes/0.7.0.md).
