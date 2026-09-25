# Detailed design: numerical dynamics and simulation for Moro 0.5.0

This document records the detailed design for the dynamics changes planned for Moro 0.5.0.

The feature extends Moro from symbolic dynamic-model construction to numerical:

- inverse dynamics;
- forward dynamics;
- state-derivative evaluation;
- time-domain simulation.

The central architectural decision is to preserve \`Robot\` as the symbolic modeling layer while introducing a numerical layer in:

\`\`\`text
moro/dynamics.py
\`\`\`

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

The dynamics feature should:

- preserve the existing symbolic robotics model;
- make numerical inverse and forward dynamics easy to use;
- use SciPy for numerical integration;
- avoid repeated symbolic substitution inside ODE evaluations;
- preserve compact educational APIs;
- avoid unnecessary controller, actuator, or simulation-framework abstractions;
- keep numerical results compatible with NumPy and visualization workflows.

## 2. Architectural split

The intended separation is:

\`\`\`text
Robot / core.py
    symbolic model construction and inspection
    ├── kinetic_energy()
    ├── potential_energy()
    ├── lagrangian()
    ├── inertia_matrix()
    ├── coriolis_matrix()
    ├── gravity_vector()
    ├── euler_lagrange_equations()
    └── dynamic_model()

moro/dynamics.py
    numerical evaluation and simulation
    ├── inverse_dynamics()
    ├── forward_dynamics()
    ├── state_derivative()
    ├── simulate()
    └── DynamicsSolution
\`\`\`

Conceptually:

\[
\boxed{
\texttt{Robot}
=
\text{symbolic model}
}
\]

and:

\[
\boxed{
\texttt{moro.dynamics}
=
\text{numerical evaluation and time evolution}
}
\]

## 3. Existing symbolic dynamic model

Moro already provides symbolic construction of:

\[
K(q,\dot q),
\]

\[
P(q),
\]

\[
\mathcal L(q,\dot q)=K-P,
\]

\[
M(q),
\]

\[
C(q,\dot q),
\]

and:

\[
G(q).
\]

The numerical layer must reuse these symbolic expressions rather than re-derive the model independently.

## 4. Symbolic API naming change

The planned final 0.5.0 symbolic API is:

\`\`\`text
euler_lagrange_equations()
dynamic_model()
\`\`\`

with the meanings described below.

### 4.1 Euler-Lagrange equations

\`\`\`python
robot.euler_lagrange_equations()
\`\`\`

returns one symbolic equation per joint:

\[
\frac{d}{dt}
\left(
\frac{\partial \mathcal L}{\partial \dot q_i}
\right)
-
\frac{\partial \mathcal L}{\partial q_i}
=
\tau_i.
\]

The return value remains a list of SymPy equation objects.

### 4.2 Matrix dynamic model

\`\`\`python
robot.dynamic_model()
\`\`\`

returns the standard manipulator equation:

\[
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau.
\]

This corresponds to the current meaning of:

\`\`\`python
robot.dynamic_model_matrix_form()
\`\`\`

## 5. Backward-compatibility policy

The current 0.4.x API uses:

\`\`\`text
dynamic_model()
    -> Euler-Lagrange equation list

dynamic_model_matrix_form()
    -> matrix equation
\`\`\`

The 0.5.0 target meaning is:

\`\`\`text
euler_lagrange_equations()
    -> Euler-Lagrange equation list

dynamic_model()
    -> matrix equation
\`\`\`

Because \`dynamic_model()\` changes meaning, a conventional same-name deprecation period is not practical.

Moro 0.5.0 should therefore document this as a breaking API change.

The old:

\`\`\`python
dynamic_model_matrix_form()
\`\`\`

may remain temporarily as a deprecated alias of:

\`\`\`python
dynamic_model()
\`\`\`

because those two names can safely return the same value.

## 6. No public DynamicModel class in 0.5.0

A public wrapper such as:

\`\`\`python
DynamicModel(robot, parameters)
\`\`\`

is not introduced initially.

The public API remains function-oriented.

An internal numerical representation may be used to hold compiled callables for:

\[
M(q),
\qquad
C(q,\dot q),
\qquad
G(q).
\]

This internal representation is an implementation detail.

## 7. SciPy dependency

SciPy becomes a required dependency for Moro 0.5.0.

Time integration uses:

\`\`\`python
scipy.integrate.solve_ivp
\`\`\`

Moro does not implement its own Runge-Kutta integrator.

## 8. Numerical dynamics API

The planned public functions are:

\`\`\`python
inverse_dynamics(
    robot,
    q,
    qd,
    qdd,
    *,
    parameters=None,
)
\`\`\`

\`\`\`python
forward_dynamics(
    robot,
    q,
    qd,
    tau,
    *,
    parameters=None,
)
\`\`\`

\`\`\`python
state_derivative(
    robot,
    t,
    state,
    tau=None,
    *,
    parameters=None,
)
\`\`\`

and:

\`\`\`python
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
\`\`\`

## 9. Numerical vector convention

For a robot with:

\[
n=robot.dof,
\]

numerical generalized-coordinate vectors use one-dimensional NumPy arrays:

\[
q,\dot q,\ddot q,\tau\in\mathbb R^n.
\]

Therefore public numerical return shapes are:

\`\`\`text
(dof,)
\`\`\`

not column matrices.

For 1 DOF, scalar input is accepted as a convenience and normalized to:

\`\`\`text
(1,)
\`\`\`

The output remains a one-dimensional vector.

For multi-DOF robots, scalar broadcasting is not allowed.

## 10. Numerical vector validation

Inputs such as:

\`\`\`text
q
qd
qdd
tau
\`\`\`

must contain exactly \`robot.dof\` components.

Values must be:

- numerical;
- real;
- finite.

The standalone inverse/forward operations evaluate one state per call.

Batch inputs such as:

\`\`\`text
(N, dof)
\`\`\`

are outside the initial 0.5.0 API.

## 11. Model parameters versus state

The \`parameters\` argument resolves fixed symbolic model quantities such as:

- link lengths;
- masses;
- center-of-mass parameters;
- inertia parameters;
- gravity magnitudes;
- other symbolic physical constants.

The instantaneous state is supplied separately through:

\`\`\`text
q
qd
qdd
\`\`\`

Therefore:

\[
parameters
\rightarrow
\text{fixed model quantities}
\]

while:

\[
q,\dot q,\ddot q
\rightarrow
\text{instantaneous state}
\]

The \`parameters\` mapping should not be used as an alternate mechanism for supplying numerical joint states.

## 12. Dynamic joint variables

Numerical dynamics in Moro 0.5.0 requires joint variables that are genuinely time-dependent symbolic quantities.

For example:

\[
q_i=q_i(t).
\]

This matches the variables provided by \`moro.abc\` and SymPy \`dynamicsymbols\`.

This requirement exists because the current symbolic Coriolis construction uses:

\[
\dot q_i = \frac{dq_i}{dt}.
\]

If a joint coordinate is represented by a static SymPy symbol, differentiation with respect to time produces zero and may silently remove valid velocity-dependent terms.

The numerical dynamics API should therefore reject incompatible static joint-variable models rather than continue with physically incorrect Coriolis terms.

## 13. Numerical model preparation

The numerical layer obtains:

\`\`\`python
M_sym = robot.inertia_matrix()
C_sym = robot.coriolis_matrix()
G_sym = robot.gravity_vector()
\`\`\`

Then it:

1. applies fixed parameter substitutions;
2. validates unresolved symbols;
3. creates numerical callables;
4. evaluates those callables with NumPy values.

Conceptually:

\[
(M,C,G)_{\text{SymPy}}
\rightarrow
\text{substitute parameters}
\rightarrow
\text{validate}
\rightarrow
\text{lambdify}
\rightarrow
(M,C,G)_{\text{NumPy}}.
\]

## 14. Allowed free state symbols

After fixed model parameters are applied, the symbolic expressions may retain only state quantities.

For \(M\) and \(G\), the remaining dynamic variables should correspond to:

\[
q_1,\ldots,q_n.
\]

For \(C\), they may correspond to:

\[
q_1,\ldots,q_n,
\dot q_1,\ldots,\dot q_n.
\]

Any unresolved physical/model parameter causes validation failure before numerical evaluation.

## 15. Numerical compilation

The expected direction is equivalent to:

\`\`\`python
M_func = lambdify(
    qs,
    M_sym,
    modules="numpy",
)
\`\`\`

\`\`\`python
C_func = lambdify(
    (*qs, *qds),
    C_sym,
    modules="numpy",
)
\`\`\`

and:

\`\`\`python
G_func = lambdify(
    qs,
    G_sym,
    modules="numpy",
)
\`\`\`

If direct lambdification of time-dependent SymPy expressions is inconvenient, internal temporary symbols may be introduced for:

\[
q_i(t)
\]

and:

\[
\dot q_i(t).
\]

This remains an implementation detail.

## 16. Internal compiled model

A private object or equivalent structure may hold compiled numerical callables.

Conceptually:

\`\`\`python
@dataclass
class _NumericalDynamicsModel:
    dof: int
    mass_matrix: Callable
    coriolis_matrix: Callable
    gravity_vector: Callable
\`\`\`

The exact private representation is not part of the public API.

## 17. No global numerical cache in 0.5.0

Standalone calls may initially prepare the numerical model each time.

No public compiled-model object or global cache is introduced.

However, \`simulate()\` must prepare and reuse the numerical model once per simulation.

This avoids repeated symbolic operations inside the ODE RHS.

## 18. inverse_dynamics()

The inverse dynamics equation is:

\[
\tau
=
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q).
\]

The public function:

\`\`\`python
tau = inverse_dynamics(
    robot,
    q,
    qd,
    qdd,
    parameters=params,
)
\`\`\`

returns a NumPy vector:

\`\`\`text
tau.shape == (robot.dof,)
\`\`\`

No result dataclass is used.

## 19. inverse_dynamics() implementation

Conceptually:

\`\`\`python
M = model.mass_matrix(q)
C = model.coriolis_matrix(q, qd)
G = model.gravity_vector(q)

tau = M @ qdd + C @ qd + G
\`\`\`

The final result must be real and finite.

## 20. forward_dynamics()

Forward dynamics solves:

\[
M(q)\ddot q
=
\tau
-
C(q,\dot q)\dot q
-
G(q).
\]

Define:

\[
b
=
\tau
-
C(q,\dot q)\dot q
-
G(q).
\]

Then solve:

\[
M(q)\ddot q=b.
\]

The public function returns:

\`\`\`text
qdd.shape == (robot.dof,)
\`\`\`

## 21. Linear solve instead of inverse

Forward dynamics must use:

\`\`\`python
np.linalg.solve(M, b)
\`\`\`

or an equivalent numerical linear solve.

It must not compute:

\`\`\`python
np.linalg.inv(M) @ b
\`\`\`

The solve operation is both numerically preferable and closer to the actual mathematical problem.

## 22. Singular mass matrix

If the numerical mass matrix is singular, forward dynamics must fail.

No automatic pseudoinverse is used.

A singular mass matrix means the standard equation:

\[
M(q)\ddot q=b
\]

does not define a unique acceleration through the intended formulation.

A \`numpy.linalg.LinAlgError\` may be propagated or re-raised with more useful context, for example:

\`\`\`text
Mass matrix is singular at the requested state.
\`\`\`

## 23. No automatic condition-number policy

The 0.5.0 API does not introduce a public condition-number threshold for \(M\).

A poorly conditioned but solvable mass matrix is initially passed to \`np.linalg.solve\`.

Diagnostics for near-singular dynamic models may be considered later if real use cases justify them.

## 24. No routine physical validation of M

Forward dynamics should validate structural numerical requirements such as:

\`\`\`text
M.shape == (n, n)
all entries finite
\`\`\`

It should not perform a full symmetry/eigenvalue/positive-definiteness test on every evaluation.

Such checks are not part of the numerical runtime contract.

## 25. Joint types and units

The numerical API is identical for revolute and prismatic joints.

For revolute joints:

\`\`\`text
q   -> angle
qd  -> angular velocity
qdd -> angular acceleration
tau -> torque
\`\`\`

For prismatic joints:

\`\`\`text
q   -> linear displacement
qd  -> linear velocity
qdd -> linear acceleration
tau -> generalized force
\`\`\`

Moro does not impose a unit system.

Dimensional consistency remains the user's responsibility.

## 26. Joint limits

\`inverse_dynamics()\`, \`forward_dynamics()\`, and later simulation do not enforce:

\`\`\`python
robot.joint_limits
\`\`\`

A state outside configured joint limits may still be mathematically evaluated.

Joint stops, impacts, and constrained dynamics are separate physical models and are outside 0.5.0.

## 27. state_derivative()

The public function is:

\`\`\`python
state_derivative(
    robot,
    t,
    state,
    tau=None,
    *,
    parameters=None,
)
\`\`\`

For:

\[
state
=
\begin{bmatrix}
q\\
\dot q
\end{bmatrix},
\]

it returns:

\[
\dot{state}
=
\begin{bmatrix}
\dot q\\
\ddot q
\end{bmatrix}.
\]

## 28. State representation

For an \(n\)-DOF robot:

\`\`\`text
state.shape == (2*n,)
\`\`\`

with fixed ordering:

\`\`\`text
[q1, ..., qn, qd1, ..., qdn]
\`\`\`

The state must be real and finite.

No scalar special case is used for \`state\`, even for one DOF.

No batch state matrices are accepted.

## 29. Time argument in state_derivative()

The \`t\` argument is a scalar finite real number.

It is always part of the public signature even when the applied generalized force is time-independent.

This matches the standard ODE callback form used by SciPy.

## 30. Generalized-force input

For \`state_derivative()\` and \`simulate()\`, \`tau\` supports exactly three forms:

\`\`\`text
None
constant numerical vector
callable tau(t, q, qd)
\`\`\`

No controller classes or additional force abstractions are introduced.

## 31. tau=None

\`\`\`python
tau=None
\`\`\`

means zero applied generalized force:

\[
\tau=0.
\]

This supports natural free-motion and gravity-driven simulation.

## 32. Constant generalized force

A numerical vector means a constant input for the full evaluation/simulation.

For example:

\`\`\`python
tau=[1.0, -0.5]
\`\`\`

for a 2-DOF robot.

The vector is validated using the same numerical-vector rules as forward dynamics.

For 1 DOF, a scalar is accepted and normalized to shape \((1,)\).

For multiple DOFs, scalar broadcasting is rejected.

## 33. Callable generalized force

A user-defined callable has the public contract:

\`\`\`python
tau(t, q, qd)
\`\`\`

where:

- \`t\` is a scalar float;
- \`q\` is a NumPy array with shape \((n,)\);
- \`qd\` is a NumPy array with shape \((n,)\).

The callable returns a generalized-force vector with shape \((n,)\).

For 1 DOF, scalar return is accepted.

## 34. Simple feedback laws

The callable interface intentionally supports simple user-defined feedback, for example:

\`\`\`python
def control(t, q, qd):
    return -Kp @ (q - q_ref) - Kd @ qd
\`\`\`

This does not imply a dedicated controller framework.

Built-in controller classes remain outside 0.5.0.

## 35. parameters are not passed to tau

The callable signature remains:

\`\`\`python
tau(t, q, qd)
\`\`\`

and does not receive \`parameters\`.

Users may use ordinary Python closures for controller constants.

This keeps \`parameters\` dedicated to symbolic robot-model quantities.

## 36. Callable validation policy

The public API validates that a supplied force input is callable when expected.

It should not invoke user callables artificially before the first real dynamics evaluation.

The returned generalized-force vector is validated when the callable is actually evaluated.

## 37. User callable exceptions

Exceptions raised internally by user-defined \`tau\` callables should not be indiscriminately replaced by generic Moro exceptions.

The original traceback should remain available.

Moro should only add validation errors when the callable's returned value violates the generalized-force contract.

## 38. Private generalized-force normalization

Simulation may prepare a private uniform callable:

\`\`\`python
tau_func(t, q, qd) -> np.ndarray
\`\`\`

from any of the three public forms.

Conceptually:

\`\`\`text
None
    -> zero-returning callable

constant vector
    -> constant-returning callable

user callable
    -> validated wrapper
\`\`\`

This avoids branching on input type during every ODE evaluation.

## 39. Defensive copying of constant tau

A constant input vector should be copied during preparation.

Subsequent mutation of the caller's original array must not unexpectedly alter an in-progress numerical simulation.

## 40. Public versus private state derivative

The public:

\`\`\`python
state_derivative(...)
\`\`\`

may prepare the numerical model for a standalone call.

However, \`simulate()\` must not repeatedly call that public wrapper if doing so would rebuild/lambdify the model every time.

Instead, both should share a private core that receives an already prepared numerical dynamics model.

## 41. Lightweight RHS validation

The public state-derivative function performs full input validation.

Inside \`solve_ivp\`, the internal RHS may use lighter repeated checks while still rejecting non-finite states.

The integrator already preserves the expected state dimension.

## 42. simulate()

The simulation interface is:

\`\`\`python
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
\`\`\`

It numerically integrates:

\[
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau(t,q,\dot q).
\]

## 43. Initial conditions

For \(n\) DOFs:

\[
q_0,\dot q_0\in\mathbb R^n.
\]

If:

\`\`\`python
qd0=None
\`\`\`

then:

\[
\dot q_0=0.
\]

Scalar initial values are accepted only for 1-DOF robots.

## 44. Initial ODE state

The initial solver state is:

\[
x_0=
\begin{bmatrix}
q_0\\
\dot q_0
\end{bmatrix}.
\]

Conceptually:

\`\`\`python
initial_state = np.concatenate((q0, qd0))
\`\`\`

with:

\`\`\`text
initial_state.shape == (2*dof,)
\`\`\`

## 45. t_span

\`t_span\` contains exactly:

\`\`\`python
(t0, tf)
\`\`\`

with finite real values and:

\[
t_f>t_0.
\]

Backward-time integration is not exposed in Moro 0.5.0.

The initial time is not required to be zero.

## 46. t_eval

If provided, \`t_eval\` must be:

- one-dimensional;
- real;
- finite;
- strictly increasing;
- fully contained within \([t_0,t_f]\).

It may be nonuniform.

If \`t_eval=None\`, Moro returns the time points chosen by \`solve_ivp\`.

## 47. t_eval does not control integration steps

Documentation must clarify that:

\`\`\`python
t_eval=...
\`\`\`

selects output times.

It does not directly force the internal ODE step size.

Internal stepping remains controlled by SciPy's solver.

## 48. solve_ivp method

The default is:

\`\`\`python
method="RK45"
\`\`\`

Moro does not create its own solver-method enum.

The method value is passed to SciPy.

Documentation may mention common methods such as:

\`\`\`text
RK45
RK23
DOP853
Radau
BDF
LSODA
\`\`\`

but SciPy remains the authority over accepted methods.

## 49. Solver tolerances and max_step

Optional arguments:

\`\`\`text
rtol
atol
max_step
\`\`\`

are passed to \`solve_ivp\` only when explicitly provided.

If they are \`None\`, Moro should not pass explicit replacement values.

This preserves SciPy's own defaults.

## 50. Tolerance scope

For Moro 0.5.0, \`rtol\` and \`atol\` are kept conceptually simple.

Scalar positive finite tolerances are sufficient for the initial public API.

Per-state-component tolerance arrays are not required initially.

\`max_step\`, when provided, must be positive and finite.

## 51. Simulation preparation order

A typical simulation preparation sequence is:

\`\`\`text
validate t_span / q0 / qd0 / t_eval
              ↓
prepare symbolic M, C, G
              ↓
apply parameters
              ↓
validate unresolved symbols
              ↓
lambdify once
              ↓
prepare tau_func once
              ↓
build initial state
              ↓
solve_ivp(...)
\`\`\`

No repeated symbolic substitution or lambdification occurs inside the RHS.

## 52. DynamicsSolution

The planned simulation result is:

\`\`\`python
@dataclass
class DynamicsSolution:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    success: bool
    message: str
    method: str

    @property
    def samples(self) -> int:
        ...

    @property
    def duration(self) -> float:
        ...

    @property
    def dof(self) -> int:
        ...
\`\`\`

## 53. DynamicsSolution shapes

For \(N\) returned time samples:

\`\`\`text
t.shape   == (N,)
q.shape   == (N, dof)
qd.shape  == (N, dof)
qdd.shape == (N, dof)
\`\`\`

The public representation is time-major.

This differs intentionally from SciPy's internal state layout:

\`\`\`text
solve_ivp.y.shape == (2*dof, N)
\`\`\`

## 54. Result conversion

Conceptually:

\`\`\`python
states = result.y.T

q = states[:, :n]
qd = states[:, n:]
\`\`\`

This aligns simulation output with the conventions used by trajectory and visualization.

## 55. Derived DynamicsSolution properties

The result provides:

\`\`\`python
@property
def samples(self):
    return self.t.size
\`\`\`

\`\`\`python
@property
def duration(self):
    return self.t[-1] - self.t[0]
\`\`\`

and:

\`\`\`python
@property
def dof(self):
    return self.q.shape[1]
\`\`\`

These values are not stored redundantly.

## 56. Compact DynamicsSolution representation

The default representation should remain compact.

For example:

\`\`\`text
DynamicsSolution(
    samples=501,
    dof=2,
    duration=5.0,
    success=True,
    method='RK45'
)
\`\`\`

Large arrays should not be included in the default representation.

## 57. qdd reconstruction

SciPy integrates and returns:

\[
q(t),\dot q(t).
\]

Moro should reconstruct:

\[
\ddot q(t)
\]

for every returned time sample using the same numerical dynamic model.

For each sample \(k\):

\[
\ddot q_k
=
M(q_k)^{-1}
\left[
\tau(t_k,q_k,\dot q_k)
-
C(q_k,\dot q_k)\dot q_k
-
G(q_k)
\right].
\]

## 58. No finite-difference acceleration

\`DynamicsSolution.qdd\` must not be obtained by finite-differencing \`qd\`.

It is evaluated directly from the forward dynamic model.

This preserves physical consistency and avoids numerical differentiation noise.

## 59. Why qdd is reconstructed after integration

\`solve_ivp\` does not return RHS acceleration evaluations aligned with every final output sample.

Recomputing \`qdd\` after integration guarantees that:

\`\`\`text
t[k]
q[k]
qd[k]
qdd[k]
\`\`\`

all refer to the same returned state.

## 60. Solver failure policy

If \`solve_ivp\` returns:

\`\`\`python
success=False
\`\`\`

Moro should return a:

\`\`\`python
DynamicsSolution(
    success=False,
    message=result.message,
    ...
)
\`\`\`

containing the valid partial solution returned by the integrator.

A normal integration failure is therefore represented in the result rather than automatically raised as an exception.

## 61. Model/RHS errors remain exceptions

Invalid model or runtime conditions such as:

- malformed numerical input;
- unresolved parameters;
- invalid callable output;
- singular mass matrix;
- non-finite numerical dynamics;

remain exceptions.

These are distinct from an integrator simply failing to reach the requested final time.

## 62. Validity of failed DynamicsSolution

Even when:

\`\`\`python
success is False
\`\`\`

the arrays stored in \`DynamicsSolution\` must remain structurally valid:

- finite;
- consistently shaped;
- matching sample counts;
- strictly increasing time values.

A failed solution is partial, not malformed.

## 63. qdd reconstruction after solver failure

If SciPy returns a valid partial state history, Moro should reconstruct \`qdd\` for those returned samples.

If acceleration reconstruction itself fails because the dynamic model cannot be evaluated at one of those states, the operation should raise rather than silently insert invalid acceleration values.

## 64. No full OdeResult exposure

\`DynamicsSolution\` does not initially expose all SciPy metadata such as:

\`\`\`text
nfev
njev
nlu
status
dense solution
events
\`\`\`

The purpose is a compact robotics result, not a mirror of \`OdeResult\`.

Additional solver diagnostics may be added later if justified.

## 65. No advanced solve_ivp features in 0.5.0

The public \`simulate()\` API does not expose:

- events;
- dense output;
- vectorized RHS mode;
- user-supplied state Jacobians;
- event-based joint-limit stops.

Advanced users can use Moro's numerical dynamics primitives directly with SciPy if needed.

## 66. Joint-limit policy during simulation

Simulation does not enforce:

\`\`\`python
robot.joint_limits
\`\`\`

Crossing a configured joint limit does not:

- stop integration;
- clip position;
- clip velocity;
- generate an impact;
- raise an error solely because of the limit.

Clipping is not a physically valid model of a mechanical stop.

## 67. Visualization interoperability

The dynamics module does not depend on visualization.

However:

\`\`\`text
solution.q.shape == (N, dof)
\`\`\`

is intentionally compatible with the planned numerical configuration support in \`RobotVisualizer.animate()\`.

Thus:

\`\`\`python
viz.animate(solution.q)
\`\`\`

should become a natural workflow.

The physical time vector \`solution.t\` is not automatically mapped to nonuniform animation timing in 0.5.0.

## 68. Validation tests for inverse and forward dynamics

Tests should cover:

- 1-DOF scalar inputs;
- multi-DOF vector inputs;
- wrong vector lengths;
- rejected scalar broadcasting for multi-DOF robots;
- NaN;
- Inf;
- complex input;
- unresolved parameters;
- incompatible static joint variables;
- finite output shape \((dof,)\).

## 69. Inverse-forward consistency

A central acceptance test is:

\`\`\`python
tau = inverse_dynamics(
    robot,
    q,
    qd,
    qdd_ref,
    parameters=params,
)

qdd = forward_dynamics(
    robot,
    q,
    qd,
    tau,
    parameters=params,
)
\`\`\`

with:

\[
qdd
\approx
qdd_{ref}.
\]

This should be tested for representative:

- revolute;
- prismatic;
- mixed R/P;
- planar multi-DOF models.

## 70. Gravity-only consistency

For:

\[
\dot q=0,
\qquad
\ddot q=0,
\]

inverse dynamics should satisfy:

\[
\tau=G(q).
\]

The numerical result should agree with independent numerical evaluation of:

\`\`\`python
robot.gravity_vector()
\`\`\`

## 71. Inertia-only consistency

With:

\[
G=0,
\qquad
\dot q=0,
\]

inverse dynamics should satisfy:

\[
\tau=M(q)\ddot q.
\]

This provides a transparent independent check of inertia terms.

## 72. Coriolis-only consistency

For a model and state with nonzero velocity-dependent effects, and with:

\[
G=0,
\qquad
\ddot q=0,
\]

inverse dynamics should satisfy:

\[
\tau=C(q,\dot q)\dot q.
\]

A nontrivial state should be chosen so the expected term is not identically zero.

## 73. Singular mass-matrix test

At least one deliberately degenerate model should demonstrate that singular \(M\) causes forward dynamics to fail.

Tests must protect against accidental future fallback to a pseudoinverse.

## 74. state_derivative tests

For:

\[
state=[q;\dot q],
\]

tests should verify:

\[
\dot{state}_{1:n}=\dot q,
\]

and:

\[
\dot{state}_{n+1:2n}
=
forward\_dynamics(q,\dot q,\tau).
\]

Also verify:

\`\`\`python
tau=None
\`\`\`

is numerically equivalent to an explicit zero vector.

## 75. Generalized-force callable tests

Tests should include:

- time-dependent force;
- state-dependent force;
- feedback-like force;
- valid scalar return for 1 DOF;
- invalid scalar return for multi-DOF;
- wrong returned vector length;
- NaN/Inf return values.

## 76. User-callable exception test

A callable that raises an intentional exception should preserve that failure clearly.

Moro should not hide user-code errors behind a generic simulation exception.

## 77. Simulation initial-condition tests

Tests should verify:

\`\`\`python
qd0=None
\`\`\`

is equivalent to an explicit zero velocity vector.

Also test scalar 1-DOF initial states and rejection of scalar broadcasting for multiple DOFs.

## 78. t_span tests

Tests should cover:

- valid nonzero initial time;
- invalid equal endpoints;
- invalid backward interval;
- non-finite endpoints;
- malformed interval length.

## 79. t_eval tests

Tests should cover:

- regular uniform grids;
- nonuniform strictly increasing grids;
- repeated values;
- decreasing values;
- values outside \`t_span\`;
- NaN/Inf;
- multidimensional inputs.

For successful integration with explicit \`t_eval\`:

\`\`\`python
np.allclose(solution.t, t_eval)
\`\`\`

should hold.

## 80. DynamicsSolution invariant tests

Tests should verify:

\`\`\`text
t.shape == (N,)
q.shape == (N, dof)
qd.shape == (N, dof)
qdd.shape == (N, dof)
\`\`\`

as well as:

\`\`\`python
solution.samples == N
solution.dof == dof
solution.duration == solution.t[-1] - solution.t[0]
\`\`\`

and all stored numerical arrays are finite.

## 81. qdd reconstruction consistency

For selected returned samples \(k\), recompute:

\`\`\`python
qdd_ref = forward_dynamics(
    robot,
    solution.q[k],
    solution.qd[k],
    tau_k,
    parameters=params,
)
\`\`\`

and verify:

\[
solution.qdd[k]
\approx
qdd_{ref}.
\]

## 82. Analytical 1-DOF integration case

A simple 1-DOF model with:

\[
M=\text{constant},
\qquad
C=0,
\qquad
G=0,
\]

and constant applied generalized force:

\[
\tau=\tau_0
\]

produces:

\[
\ddot q=a=\frac{\tau_0}{M}.
\]

The exact solution is:

\[
\dot q(t)
=
\dot q_0+a(t-t_0),
\]

\[
q(t)
=
q_0+\dot q_0(t-t_0)
+\frac12a(t-t_0)^2.
\]

This should be a primary end-to-end simulation regression test.

## 83. Gravity-driven motion test

A simple model with:

\[
G(q_0)\neq0
\]

and:

\`\`\`python
tau=None
\`\`\`

should demonstrate motion from rest or another appropriate initial state.

At minimum:

\[
\ddot q(t_0)
=
forward\_dynamics(q_0,\dot q_0,0).
\]

The system should not remain artificially stationary when gravity produces a nonzero generalized force.

## 84. Energy consistency test

For a simple conservative system with:

- no applied generalized force;
- no friction;
- no contacts;

the mechanical energy:

\[
E(t)=K(t)+P(t)
\]

should remain approximately constant within expected numerical integration error.

The test should use a reasonable relative tolerance rather than exact equality.

RK45 is not symplectic, so small energy drift is numerically acceptable.

## 85. PD-like feedback example/test

A simple law such as:

\`\`\`python
def control(t, q, qd):
    return -Kp @ (q - q_ref) - Kd @ qd
\`\`\`

can be used to demonstrate state-feedback input.

A lightweight test may verify that the final configuration is closer to the reference than the initial configuration for a carefully selected stable example.

This is not intended as a general controller-stability test.

## 86. Solver-failure policy test

The behavior:

\`\`\`text
solve_ivp success=False
    -> DynamicsSolution(success=False)
\`\`\`

should preferably be tested through a controlled mocked/monkeypatched SciPy result rather than relying on a fragile physical model that happens to fail in a particular SciPy version.

This tests Moro's result-handling policy directly.

## 87. Solver-option forwarding tests

Tests should verify that:

\`\`\`text
rtol=None
atol=None
max_step=None
\`\`\`

are not forwarded unnecessarily as explicit values.

Explicit positive values should be passed through to SciPy.

## 88. Symbolic-parameter tests

Tests should include robot models with symbolic:

- geometry;
- masses;
- inertia parameters;
- center-of-mass parameters;
- gravity magnitude.

A complete \`parameters\` mapping should support:

\`\`\`python
inverse_dynamics(...)
forward_dynamics(...)
simulate(...)
\`\`\`

without mutating the original symbolic robot model.

## 89. Incomplete dynamic model tests

If required robot physical properties are missing, existing clear \`Robot\` validation errors should remain visible where practical.

The numerical layer should not replace useful messages such as missing masses or inertia tensors with a generic preparation failure.

## 90. Joint-limit non-enforcement test

At least one test should confirm that simulation is not clipped or stopped merely because a numerical state crosses configured joint limits.

This protects the explicit 0.5.0 physical-model boundary.

## 91. Visualization integration test

If \`RobotVisualizer.animate()\` gains numerical joint-matrix support, an integration test should demonstrate:

\`\`\`python
solution = simulate(...)
viz.animate(solution.q)
\`\`\`

without requiring the dynamics module to import visualization.

## 92. Documentation example: inverse and forward dynamics

A planar 2R example should demonstrate:

\`\`\`python
tau = inverse_dynamics(
    robot,
    q,
    qd,
    qdd,
    parameters=params,
)

qdd_check = forward_dynamics(
    robot,
    q,
    qd,
    tau,
    parameters=params,
)
\`\`\`

This illustrates the complementary questions:

- what generalized force produces a prescribed motion?;
- what acceleration results from an applied generalized force?

## 93. Documentation example: free motion under gravity

A second example should simulate:

\`\`\`python
solution = simulate(
    robot,
    (0.0, 5.0),
    q0=[...],
    tau=None,
    parameters=params,
    t_eval=np.linspace(0.0, 5.0, 501),
)
\`\`\`

and inspect or plot:

\[
q(t),
\qquad
\dot q(t).
\]

## 94. Documentation example: time-varying input

For example:

\`\`\`python
def tau(t, q, qd):
    return np.array([
        A * np.sin(w * t),
        0.0,
    ])
\`\`\`

This demonstrates that applied generalized force can vary in time.

## 95. Documentation example: simple feedback

A minimal feedback example may use:

\`\`\`python
def control(t, q, qd):
    return -Kp @ (q - q_ref) - Kd @ qd
\`\`\`

Documentation must state clearly that this is a user-defined force law, not a new Moro controller framework.

## 96. Trajectory-to-inverse-dynamics example

A useful advanced example connects the new trajectory and dynamics modules:

\`\`\`python
traj = joint_trajectory(
    q0,
    qf,
    t,
    method="quintic",
)

tau = np.array([
    inverse_dynamics(
        robot,
        q,
        qd,
        qdd,
        parameters=params,
    )
    for q, qd, qdd in zip(
        traj.q,
        traj.qd,
        traj.qdd,
    )
])
\`\`\`

This computes the generalized-force history required to follow a prescribed joint trajectory.

No vectorized trajectory-wide inverse-dynamics API is introduced in 0.5.0.

## 97. Explicitly outside Moro 0.5.0

The following remain outside scope:

- symbolic state-space model construction;
- friction models;
- external Cartesian wrenches;
- contact dynamics;
- collision response;
- impacts;
- physical joint-stop models;
- actuator dynamics;
- torque saturation;
- dedicated controller classes;
- built-in computed-torque control;
- constrained dynamics;
- closed-chain dynamics;
- advanced event handling;
- multibody contact simulation;
- automatic model regularization;
- pseudoinverse forward dynamics;
- automatic dynamic feasibility planning;
- trajectory optimization;
- time-optimal control.

## 98. Acceptance criteria

The dynamics block is ready for implementation when:

1. symbolic dynamic-model construction remains on \`Robot\`;
2. numerical evaluation/simulation lives in \`moro.dynamics\`;
3. \`euler_lagrange_equations()\` returns the per-joint Euler-Lagrange equations;
4. \`dynamic_model()\` returns the symbolic matrix manipulator equation;
5. \`dynamic_model_matrix_form()\` may remain temporarily as a deprecated alias;
6. the \`dynamic_model()\` meaning change is documented as a breaking 0.5.0 API change;
7. numerical dynamics requires time-dependent joint variables compatible with the symbolic Coriolis model;
8. symbolic \(M\), \(C\), and \(G\) are reused rather than independently reconstructed;
9. fixed symbolic parameters are applied before numerical evaluation;
10. unresolved physical/model symbols fail clearly;
11. numerical generalized-coordinate vectors use shape \((dof,)\);
12. scalar numerical inputs are accepted only for 1 DOF;
13. \`inverse_dynamics()\` returns generalized force directly;
14. \`forward_dynamics()\` returns generalized acceleration directly;
15. forward dynamics uses a numerical linear solve rather than explicit matrix inversion;
16. singular mass matrices fail without pseudoinverse fallback;
17. \`state_derivative()\` uses state ordering \([q;\dot q]\);
18. \`tau\` supports only \`None\`, a constant vector, or a callable \`tau(t,q,qd)\`;
19. user-defined torque callables receive NumPy 1D state arrays;
20. user callable errors are not hidden by generic exceptions;
21. \`simulate()\` uses \`scipy.integrate.solve_ivp\`;
22. SciPy becomes a required dependency;
23. \`qd0=None\` means zero initial velocity;
24. \`t_span\` is a finite forward-time interval;
25. \`t_eval\` may be nonuniform but must be strictly increasing and contained in \`t_span\`;
26. the numerical dynamics model is prepared once per simulation;
27. \`DynamicsSolution\` stores time-major \`q\`, \`qd\`, and \`qdd\`;
28. \`DynamicsSolution\` exposes \`success\`, \`message\`, and \`method\`;
29. \`samples\`, \`duration\`, and \`dof\` are derived properties;
30. \`qdd\` is reconstructed from forward dynamics rather than finite differences;
31. normal \`solve_ivp\` failure returns \`DynamicsSolution(success=False)\`;
32. malformed input, invalid model evaluation, and singular dynamics remain exceptions;
33. simulation does not physically enforce joint limits;
34. no contact, impact, friction, actuator, or controller framework is introduced;
35. inverse and forward dynamics are numerically consistent;
36. simple analytical cases agree with known solutions;
37. conservative examples exhibit approximate energy conservation within expected numerical integration error;
38. symbolic parameters are evaluated without mutating the original robot model;
39. result shapes are compatible with numerical visualization workflows;
40. no 0.5.0 implementation introduces advanced event handling, constrained dynamics, automatic torque saturation, or dynamic trajectory optimization.
