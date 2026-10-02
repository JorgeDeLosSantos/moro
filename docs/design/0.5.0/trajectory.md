# Detailed design: trajectory generation for Moro 0.5.0

This document records the detailed design for numerical point-to-point trajectory generation in Moro 0.5.0.

The feature provides compact educational trajectory generation in:

- joint space;
- Cartesian position space.

It is intentionally independent of robot-specific inverse kinematics, joint limits, dynamics, collision checking, and visualization.

The implementation is expected in:

\`\`\`text
moro/trajectory.py
\`\`\`

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

The trajectory feature should be:

- numerical;
- compact;
- educational;
- explicit about time;
- mathematically transparent;
- independent of \`Robot\`;
- compatible with NumPy-based downstream workflows.

The API retains linear, cubic, and quintic interpolation because the progression between them is useful for teaching.

## 2. Public API

The planned joint-space function is:

\`\`\`python
joint_trajectory(
    q0,
    qf,
    t,
    *,
    method="quintic",
    qd0=None,
    qdf=None,
    qdd0=None,
    qddf=None,
)
\`\`\`

The planned Cartesian-position function is:

\`\`\`python
position_trajectory(
    p0,
    pf,
    t,
    *,
    method="quintic",
    v0=None,
    vf=None,
    a0=None,
    af=None,
)
\`\`\`

The default interpolation method is \`"quintic"\`.

## 3. Supported methods

The accepted method names are:

\`\`\`text
linear
cubic
quintic
\`\`\`

String methods may be normalized to lowercase before validation.

No aliases such as \`"lin"\`, \`"poly3"\`, or \`"poly5"\` are introduced.

## 4. Independence from Robot

Neither public trajectory function inspects or requires:

- robot joint types;
- robot joint limits;
- DH parameters;
- inverse kinematics;
- dynamics;
- collision information;
- visualization state.

Trajectory generation is a standalone mathematical operation.

A joint trajectory is simply a time-parameterized vector in joint-coordinate space.

A Cartesian position trajectory is simply a time-parameterized vector in \(\mathbb{R}^3\).

## 5. Path versus trajectory

Documentation must distinguish a geometric path from a trajectory.

A path describes geometry only.

A trajectory adds time parametrization:

\[
x=x(t).
\]

Two trajectories may traverse the same geometric path with different timing.

No separate public \`Path\` abstraction is introduced in 0.5.0.

## 6. JointTrajectory result type

The planned data container is:

\`\`\`python
@dataclass
class JointTrajectory:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
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

The array shapes are:

\[
t\in\mathbb{R}^{N},
\]

\[
q,\dot q,\ddot q\in\mathbb{R}^{N\times n}.
\]

Therefore:

\`\`\`text
t.shape   == (N,)
q.shape   == (N, dof)
qd.shape  == (N, dof)
qdd.shape == (N, dof)
\`\`\`

## 7. PositionTrajectory result type

The planned Cartesian result is:

\`\`\`python
@dataclass
class PositionTrajectory:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    method: str

    @property
    def samples(self) -> int:
        ...

    @property
    def duration(self) -> float:
        ...
\`\`\`

The public Cartesian representation is always:

\[
p,v,a\in\mathbb{R}^{N\times3}.
\]

Therefore:

\`\`\`text
t.shape == (N,)
p.shape == (N, 3)
v.shape == (N, 3)
a.shape == (N, 3)
\`\`\`

This shape remains \((N,3)\) even for planar motion.

## 8. Derived properties

For both trajectory result types:

\`\`\`python
@property
def samples(self):
    return self.t.size
\`\`\`

and:

\`\`\`python
@property
def duration(self):
    return self.t[-1] - self.t[0]
\`\`\`

For \`JointTrajectory\` only:

\`\`\`python
@property
def dof(self):
    return self.q.shape[1]
\`\`\`

These quantities are derived rather than stored redundantly.

## 9. No additional convenience API in 0.5.0

The result objects do not initially add:

- slicing semantics;
- \`__len__\`;
- \`dt\`;
- uniform-time detection;
- start/end aliases;
- plural property aliases;
- automatic resampling.

Users can work directly with the public arrays.

## 10. Defensive copying and mutability

Trajectory dataclasses should normalize inputs to independent NumPy float arrays.

The objects protect themselves from subsequent mutation of arrays supplied to their constructors.

The internal arrays remain mutable in 0.5.0.

The dataclasses are not deeply immutable value objects.

## 11. Result invariants

For both result types:

- \`t\` must be a one-dimensional finite real array;
- \`len(t) >= 2\`;
- \`t\` must be strictly increasing;
- all trajectory arrays must have exactly \(N\) rows;
- all stored values must be real and finite;
- \`method\` must be a supported normalized method name.

For \`JointTrajectory\`:

\`\`\`text
q.ndim == 2
qd.ndim == 2
qdd.ndim == 2
q.shape == qd.shape == qdd.shape
q.shape[1] >= 1
\`\`\`

For \`PositionTrajectory\`:

\`\`\`text
p.shape == v.shape == a.shape == (N, 3)
\`\`\`

## 12. Compact representations

The result objects should have concise notebook-friendly representations.

For example:

\`\`\`text
JointTrajectory(samples=101, dof=3, duration=2.0, method='quintic')
\`\`\`

and:

\`\`\`text
PositionTrajectory(samples=101, duration=2.0, method='quintic')
\`\`\`

The full arrays should not appear in the default representation.

## 13. Joint-space input normalization

\`q0\` and \`qf\` represent one joint vector each.

For one degree of freedom, scalar inputs are accepted:

\`\`\`python
joint_trajectory(
    0.0,
    1.0,
    t,
)
\`\`\`

Internally these are normalized to one-component vectors.

The output still uses:

\`\`\`text
(N, 1)
\`\`\`

rather than collapsing to \`(N,)\`.

For multiple degrees of freedom, \`q0\` and \`qf\` must have the same number of components.

All values must be real and finite.

## 14. Joint derivative boundary conditions

Provided joint derivative conditions:

\`\`\`text
qd0
qdf
qdd0
qddf
\`\`\`

must match the dimensionality of \`q0\`.

For a 1-DOF trajectory, scalar derivative conditions are accepted and normalized to one-component vectors.

For \(n>1\), scalar broadcasting is not allowed.

For example:

\`\`\`python
q0 = [0.0, 0.0]
qf = [1.0, 1.0]
qd0 = 0.5
\`\`\`

is invalid.

A scalar is not implicitly interpreted as the same boundary velocity for every joint.

## 15. Cartesian input normalization

\`position_trajectory()\` uses exactly three Cartesian components.

Therefore:

\[
p_0,p_f\in\mathbb{R}^3.
\]

Likewise, provided:

\[
v_0,v_f,a_0,a_f
\]

must each contain exactly three real finite components.

Two-component planar inputs are not automatically promoted to \(z=0\).

Planar motion should be written explicitly, for example:

\`\`\`python
p0 = [0.0, 0.0, 0.0]
pf = [1.0, 0.5, 0.0]
\`\`\`

## 16. Explicit time vector

Both trajectory functions require an explicit numerical time vector:

\`\`\`python
t = np.linspace(0.0, 2.0, 101)
\`\`\`

The initial API does not add alternative combinations such as:

\`\`\`text
duration=
tf=
samples=
dt=
\`\`\`

The caller controls the evaluation times explicitly.

## 17. Time-vector validation

The time array must satisfy:

- one-dimensional;
- numerical;
- real;
- finite;
- at least two samples;
- strictly increasing.

Mathematically:

\[
t_{k+1}>t_k
\]

for every valid index \(k\).

Therefore repeated or decreasing times are invalid.

## 18. Arbitrary time origin

The trajectory does not require:

\[
t_0=0.
\]

For example:

\`\`\`python
t = np.linspace(5.0, 7.0, 101)
\`\`\`

is valid.

The duration is:

\[
T=t_f-t_0.
\]

Only elapsed time matters to the polynomial interpolation.

## 19. Nonuniform sample times

The time vector need not be uniformly spaced.

For example:

\`\`\`python
t = [0.0, 0.1, 0.4, 1.0]
\`\`\`

is valid.

The trajectory polynomial is evaluated directly at each supplied time.

No single public \`dt\` property is defined.

## 20. Normalized time

The internal polynomial formulation uses:

\[
\tau=
\frac{t-t_0}{T},
\qquad
T=t_f-t_0>0.
\]

Thus:

\[
0\le\tau\le1.
\]

Using normalized time avoids unnecessary dependence of polynomial coefficients on the absolute time origin.

## 21. Physical-time derivatives

Public velocities and accelerations are derivatives with respect to physical time \(t\), not normalized time \(\tau\).

Therefore:

\[
\frac{d}{dt}
=
\frac{1}{T}
\frac{d}{d\tau},
\]

and:

\[
\frac{d^2}{dt^2}
=
\frac{1}{T^2}
\frac{d^2}{d\tau^2}.
\]

Correct handling of these scaling factors is a core numerical requirement.

## 22. Linear interpolation

For linear interpolation:

\[
x(\tau)
=
x_0+(x_f-x_0)\tau.
\]

Therefore:

\[
\dot x(t)
=
\frac{x_f-x_0}{T},
\]

and:

\[
\ddot x(t)=0.
\]

The formulation is applied componentwise to vector-valued \(x\).

## 23. Linear boundary-condition policy

The linear method uses endpoint positions only.

No explicit velocity or acceleration conditions are accepted.

Therefore, all of the following must remain unspecified:

\`\`\`text
v0
vf
a0
af
\`\`\`

or, in the joint API:

\`\`\`text
qd0
qdf
qdd0
qddf
\`\`\`

An explicitly supplied zero is still rejected.

This prevents the API from implying that a condition was enforced when the linear polynomial cannot generally satisfy it.

## 24. Linear educational interpretation

A linear segment has constant velocity inside the interval.

If connected to rest before or after the interval, the velocity is generally discontinuous at the endpoints.

This behavior should be documented explicitly because it motivates higher-order interpolation.

## 25. Cubic interpolation

For cubic interpolation:

\[
x(\tau)
=
c_0+c_1\tau+c_2\tau^2+c_3\tau^3.
\]

The endpoint conditions are:

\[
x(0)=x_0,
\qquad
x(1)=x_f,
\]

\[
\dot x(t_0)=v_0,
\qquad
\dot x(t_f)=v_f.
\]

The normalized-time coefficients are:

\[
c_0=x_0,
\]

\[
c_1=Tv_0,
\]

\[
c_2=
3(x_f-x_0)-T(2v_0+v_f),
\]

\[
c_3=
-2(x_f-x_0)+T(v_0+v_f).
\]

## 26. Cubic derivatives

The physical-time velocity is:

\[
\dot x
=
\frac{1}{T}
\left(
c_1+2c_2\tau+3c_3\tau^2
\right).
\]

The physical-time acceleration is:

\[
\ddot x
=
\frac{1}{T^2}
\left(
2c_2+6c_3\tau
\right).
\]

## 27. Cubic boundary-condition policy

Endpoint velocities are optional.

If omitted independently, each defaults to zero.

For example:

\`\`\`text
v0=None -> zero vector
vf=None -> zero vector
\`\`\`

The same applies to \`qd0\` and \`qdf\`.

Endpoint accelerations are unsupported by cubic interpolation and must not be supplied, even explicitly as zero.

## 28. Quintic interpolation

For quintic interpolation:

\[
x(\tau)
=
c_0+c_1\tau+c_2\tau^2+
c_3\tau^3+c_4\tau^4+c_5\tau^5.
\]

The endpoint conditions are:

\[
x(0)=x_0,
\qquad
x(1)=x_f,
\]

\[
\dot x(t_0)=v_0,
\qquad
\dot x(t_f)=v_f,
\]

\[
\ddot x(t_0)=a_0,
\qquad
\ddot x(t_f)=a_f.
\]

## 29. Quintic coefficients

Let:

\[
\Delta x=x_f-x_0.
\]

Then:

\[
c_0=x_0,
\]

\[
c_1=Tv_0,
\]

\[
c_2=\frac{T^2}{2}a_0,
\]

\[
c_3=
10\Delta x
-6Tv_0
-4Tv_f
-\frac{3}{2}T^2a_0
+\frac{1}{2}T^2a_f,
\]

\[
c_4=
-15\Delta x
+8Tv_0
+7Tv_f
+\frac{3}{2}T^2a_0
-T^2a_f,
\]

\[
c_5=
6\Delta x
-3Tv_0
-3Tv_f
-\frac{1}{2}T^2a_0
+\frac{1}{2}T^2a_f.
\]

## 30. Quintic derivatives

The physical-time velocity is:

\[
\dot x=
\frac{1}{T}
\left(
c_1+
2c_2\tau+
3c_3\tau^2+
4c_4\tau^3+
5c_5\tau^4
\right).
\]

The physical-time acceleration is:

\[
\ddot x=
\frac{1}{T^2}
\left(
2c_2+
6c_3\tau+
12c_4\tau^2+
20c_5\tau^3
\right).
\]

## 31. Quintic boundary-condition policy

Endpoint velocities and accelerations are optional.

Any omitted condition defaults independently to a zero vector.

Thus the default quintic trajectory corresponds to:

\[
v_0=v_f=0,
\]

\[
a_0=a_f=0.
\]

This makes quintic interpolation a convenient default for smooth point-to-point motion.

## 32. Classical time-scaling profiles

With zero derivative boundary conditions, the three standard normalized profiles are:

### Linear

\[
s(\tau)=\tau.
\]

### Cubic

\[
s(\tau)=3\tau^2-2\tau^3.
\]

### Quintic

\[
s(\tau)=10\tau^3-15\tau^4+6\tau^5.
\]

Then:

\[
x(t)=x_0+s(t)(x_f-x_0).
\]

These profiles should be emphasized in the theory/documentation because they provide a clear educational progression:

\`\`\`text
linear  -> position
cubic   -> position + velocity
quintic -> position + velocity + acceleration
\`\`\`

## 33. Shared polynomial core

Joint and Cartesian trajectory generation should share a private mathematical core.

Conceptually:

\`\`\`python
_polynomial_trajectory(
    x0,
    xf,
    t,
    *,
    method,
    v0=None,
    vf=None,
    a0=None,
    af=None,
)
\`\`\`

returns:

\`\`\`python
x, xd, xdd
\`\`\`

The public wrappers handle domain-specific naming and dimensional validation.

## 34. Vectorized coefficient evaluation

Polynomial coefficients are vectors:

\[
c_i\in\mathbb{R}^{d}.
\]

Normalized time can be reshaped as:

\`\`\`python
tau = tau[:, None]
\`\`\`

with:

\`\`\`text
tau.shape == (N, 1)
\`\`\`

while each coefficient has shape:

\`\`\`text
(d,)
\`\`\`

NumPy broadcasting then produces:

\`\`\`text
(N, d)
\`\`\`

without separate Python loops per coordinate.

## 35. Closed-form coefficients

The implementation should use the explicit closed-form coefficients above.

It should not solve a linear system numerically for every trajectory call.

The derivation from endpoint constraints may still be presented in documentation.

Closed-form evaluation is:

- simpler;
- faster;
- easier to test;
- more predictable.

## 36. Numerical rather than symbolic runtime

The runtime implementation should use NumPy.

It should not construct SymPy polynomials, differentiate them symbolically, and substitute every time a trajectory is generated.

The feature is numerical by design.

Symbolic derivations belong in the theory documentation, not the runtime path.

## 37. No endpoint patching

The implementation should not manually overwrite generated endpoint samples with values such as:

\`\`\`python
q[0] = q0
q[-1] = qf
\`\`\`

to hide numerical or coefficient errors.

The analytical formulas should satisfy the endpoint conditions directly.

Tests should use suitable numerical tolerances.

## 38. Cartesian geometric interpretation

With zero endpoint velocity and acceleration conditions:

\[
p(t)=p_0+s(t)(p_f-p_0),
\]

so the Cartesian path is a line segment.

However, nonzero derivative conditions need not remain parallel to:

\[
p_f-p_0.
\]

Therefore a general cubic or quintic Cartesian position trajectory may be curved.

Documentation must not imply that \`position_trajectory()\` always produces a straight line.

## 39. Method-specific validation order

Unsupported explicitly supplied boundary conditions must be rejected before default substitution.

For linear:

\`\`\`text
v0, vf, a0, af must all be None
\`\`\`

For cubic:

\`\`\`text
a0 and af must be None
\`\`\`

Then omitted velocity conditions are replaced by zeros.

For quintic, omitted velocity and acceleration conditions are all replaced by zeros.

This preserves the distinction between:

- not specifying a condition;
- explicitly specifying an unsupported condition equal to zero.

## 40. Integration with inverse kinematics

The intended Cartesian workflow is explicit:

\`\`\`python
cart = position_trajectory(
    p0,
    pf,
    t,
    method="quintic",
)

ik = solve_position_trajectory(
    robot,
    cart.p,
    q0=q_initial,
)
\`\`\`

The trajectory generator creates Cartesian positions.

The IK solver resolves those positions into robot joint configurations.

These remain separate operations.

## 41. No PositionTrajectory-aware IK overload

\`solve_position_trajectory()\` should not require or special-case \`PositionTrajectory\`.

The user explicitly passes:

\`\`\`python
cart.p
\`\`\`

This avoids coupling \`inverse_kinematics.py\` to \`trajectory.py\`.

It also makes clear that current position IK uses only the target positions, not \`cart.v\` or \`cart.a\`.

## 42. Time and IK trajectory results

\`IKTrajectorySolution\` is not extended with time metadata as part of this trajectory feature.

When combining:

\`\`\`python
cart = position_trajectory(...)
ik = solve_position_trajectory(robot, cart.p, ...)
\`\`\`

the time vector remains available as:

\`\`\`python
cart.t
\`\`\`

while joint IK solutions remain in the IK result.

A future robot-aware trajectory abstraction may combine these concerns if justified.

## 43. Visualization integration goal

The numerical representation:

\`\`\`text
JointTrajectory.q -> (N, dof)
\`\`\`

should be directly usable in visualization workflows.

The existing visualization API currently expects substitution dictionaries.

Moro 0.5.0 should consider a small compatibility improvement so that \`RobotVisualizer\` can accept numerical joint vectors directly.

## 44. Proposed RobotVisualizer.plot extension

Preserve the existing mapping form:

\`\`\`python
viz.plot({
    q1: 0.3,
    q2: -0.2,
})
\`\`\`

and additionally support:

\`\`\`python
viz.plot([
    0.3,
    -0.2,
])
\`\`\`

For numerical vector input:

\[
len(q)=robot.dof.
\]

The vector is mapped by joint order:

\[
q_i \leftrightarrow robot.qs[i].
\]

No missing-coordinate filling is allowed.

## 45. Proposed RobotVisualizer.animate extension

Preserve the existing sequence-of-mappings form:

\`\`\`python
viz.animate([
    {q1: 0.0, q2: 0.0},
    {q1: 0.1, q2: 0.2},
])
\`\`\`

and additionally support numerical joint configurations:

\`\`\`python
viz.animate([
    [0.0, 0.0],
    [0.1, 0.2],
])
\`\`\`

or:

\`\`\`python
viz.animate(joint_traj.q)
\`\`\`

with:

\`\`\`text
joint_traj.q.shape == (N, robot.dof)
\`\`\`

This also enables:

\`\`\`python
viz.animate(ik_traj.qs)
\`\`\`

without manual conversion to substitution dictionaries.

## 46. No visualization dependency on JointTrajectory

The visualization extension should be structural rather than type-based.

\`RobotVisualizer\` should not import or inspect \`JointTrajectory\`.

Instead it should normalize:

- mappings;
- one-dimensional numerical joint vectors;
- sequences/two-dimensional arrays of numerical joint vectors.

This keeps:

\`\`\`text
visualization
\`\`\`

independent from:

\`\`\`text
trajectory
\`\`\`

while still making their array conventions compatible.

## 47. Visualization configuration helpers

A small internal helper may normalize one configuration:

\`\`\`python
_normalize_configuration(robot, values)
\`\`\`

and a companion helper may normalize sequences:

\`\`\`python
_normalize_configurations(robot, values)
\`\`\`

The exact helper names are implementation details.

Existing mapping semantics must remain backward compatible.

## 48. Animation timing scope

\`RobotVisualizer.animate()\` should not automatically consume \`JointTrajectory.t\` in 0.5.0.

Trajectory time vectors may be nonuniform, while current animation backends use a uniform interval-style API.

Therefore:

\`\`\`python
viz.animate(
    traj.q,
    interval=50,
)
\`\`\`

visualizes the configuration sequence but does not claim to reproduce the exact physical timing represented by \`traj.t\`.

Variable-time animation support is deferred.

## 49. PositionTrajectory visualization scope

No dedicated:

\`\`\`python
plot_position_trajectory(...)
\`\`\`

helper is introduced in 0.5.0.

Users can plot the public arrays directly with Matplotlib when needed.

The trajectory module remains independent from visualization.

## 50. Relationship among trajectory-like objects

The following remain distinct concepts:

### PositionTrajectory

A prescribed Cartesian time trajectory:

\`\`\`text
t, p, v, a
\`\`\`

### JointTrajectory

A prescribed joint-space time trajectory:

\`\`\`text
t, q, qd, qdd
\`\`\`

### IKTrajectorySolution

A sequential inverse-kinematics solver result.

No shared public \`Trajectory\` base class is introduced.

These objects have related names but different semantics and responsibilities.

## 51. Time validation tests

Tests should cover:

- empty time vectors;
- a single time sample;
- repeated times;
- decreasing times;
- NaN and Inf;
- arbitrary nonzero time origins;
- nonuniform spacing;
- valid two-sample trajectories;
- very short but positive durations.

No arbitrary minimum duration is imposed.

## 52. Joint input validation tests

Tests should cover:

- valid 1-DOF scalar endpoints;
- resulting \((N,1)\) output;
- valid multi-DOF vectors;
- mismatched endpoint dimensions;
- empty joint vectors;
- non-finite endpoint values;
- complex values;
- valid scalar derivative conditions for 1 DOF;
- rejection of scalar derivative broadcasting for multi-DOF trajectories;
- mismatched derivative vector dimensions.

## 53. Cartesian validation tests

Tests should cover:

- valid three-component positions;
- rejection of two-component planar inputs;
- rejection of scalar Cartesian inputs;
- non-finite Cartesian data;
- malformed velocity/acceleration boundary vectors;
- exact preservation of \((N,3)\) result shapes.

## 54. Method validation tests

Tests should cover:

- \`linear\`;
- \`cubic\`;
- \`quintic\`;
- accepted capitalization normalization if implemented;
- invalid method names.

Boundary-condition support must also be tested explicitly.

## 55. Linear mathematical tests

For linear interpolation, verify:

\[
x(t_0)=x_0,
\]

\[
x(t_f)=x_f,
\]

\[
\dot x(t)
=
\frac{x_f-x_0}{T},
\]

for every sample, and:

\[
\ddot x(t)=0.
\]

Explicit derivative boundary conditions must be rejected.

## 56. Cubic mathematical tests

Verify:

\[
x(t_0)=x_0,
\qquad
x(t_f)=x_f,
\]

\[
\dot x(t_0)=v_0,
\qquad
\dot x(t_f)=v_f.
\]

Also verify the analytically expected acceleration profile.

Tests should include:

- both velocities omitted;
- only initial velocity supplied;
- only final velocity supplied;
- both velocities supplied;
- rejection of acceleration boundary conditions.

## 57. Quintic mathematical tests

Verify:

\[
x(t_0)=x_0,
\qquad
x(t_f)=x_f,
\]

\[
\dot x(t_0)=v_0,
\qquad
\dot x(t_f)=v_f,
\]

\[
\ddot x(t_0)=a_0,
\qquad
\ddot x(t_f)=a_f.
\]

Tests should include:

- all derivative conditions omitted;
- nonzero velocities;
- nonzero accelerations;
- partially omitted boundary conditions;
- multi-component vector trajectories.

## 58. Classical profile regression tests

For:

\[
x_0=0,
\qquad
x_f=1,
\qquad
T=1,
\]

compare generated profiles against:

\[
s_L(\tau)=\tau,
\]

\[
s_C(\tau)=3\tau^2-2\tau^3,
\]

\[
s_Q(\tau)=10\tau^3-15\tau^4+6\tau^5.
\]

These provide compact analytical regression tests.

## 59. Time-scaling tests

If the duration is scaled while the same normalized-time samples are used:

\[
T\rightarrow \alpha T,
\]

the position profile at equal \(\tau\) should remain unchanged.

Velocities should scale as:

\[
\dot x\propto\frac{1}{T},
\]

and accelerations as:

\[
\ddot x\propto\frac{1}{T^2}.
\]

This explicitly protects the physical-time derivative scaling.

## 60. Time-origin invariance tests

Two time vectors with equal duration and equal normalized sampling, for example:

\`\`\`python
t1 = np.linspace(0.0, 2.0, 101)
t2 = np.linspace(10.0, 12.0, 101)
\`\`\`

should generate identical:

\`\`\`text
x
xd
xdd
\`\`\`

for identical endpoint conditions.

This protects against accidental use of absolute time rather than elapsed normalized time.

## 61. Multi-DOF tests

Tests should verify:

- independent interpolation of each coordinate;
- preserved \((N,dof)\) shapes;
- coordinates with identical start/end values remain constant when boundary derivatives are compatible;
- mixed positive and negative coordinate motions;
- nonzero endpoint velocities/accelerations on selected coordinates only.

## 62. Cartesian geometry tests

With zero derivative boundary conditions, generated Cartesian positions should lie on the line segment connecting:

\[
p_0
\]

and:

\[
p_f.
\]

With transverse nonzero boundary velocity, a curved Cartesian path is valid and should not be rejected.

This test protects the intended vector-polynomial semantics.

## 63. Dataclass tests

Direct result-object tests should cover:

- valid shapes;
- invalid time shape;
- invalid data shapes;
- mismatched sample counts;
- non-finite values;
- invalid methods;
- defensive copying;
- \`samples\`;
- \`duration\`;
- \`dof\` for joint trajectories;
- compact \`repr\`.

## 64. IK integration test

A representative flow should verify:

\`\`\`python
cart = position_trajectory(...)
ik = solve_position_trajectory(
    robot,
    cart.p,
    q0=...,
)
\`\`\`

If every IK target converges:

\`\`\`python
len(ik.qs) == cart.samples
\`\`\`

No assertions are made about \`cart.v\` or \`cart.a\` because current position IK does not consume them.

## 65. Visualization integration tests

If the numerical configuration extension is implemented, tests should preserve existing mapping behavior and add:

\`\`\`python
viz.plot(q_vector)
\`\`\`

and:

\`\`\`python
viz.animate(q_matrix)
\`\`\`

Tests should verify:

- exact DOF length validation;
- finite numeric values;
- existing dict-based calls remain unchanged;
- \`viz.animate(joint_traj.q)\` works;
- \`viz.animate(ik_traj.qs)\` works.

Physical timing against \`traj.t\` is explicitly not part of these tests.

## 66. Documentation examples

At least four user-facing examples are recommended.

### 66.1 Linear, cubic, and quintic comparison

Use the same 1-DOF point-to-point motion and compare:

\[
q(t),
\qquad
\dot q(t),
\qquad
\ddot q(t).
\]

This is the primary teaching example.

### 66.2 Multi-joint quintic trajectory

Demonstrate a multi-DOF trajectory with zero or selected nonzero derivative conditions.

### 66.3 Cartesian trajectory plus inverse kinematics

Generate:

\`\`\`python
cart = position_trajectory(...)
\`\`\`

then pass:

\`\`\`python
cart.p
\`\`\`

to \`solve_position_trajectory()\`.

This demonstrates the separation between trajectory generation and robot-specific IK.

### 66.4 Joint trajectory plus visualization

If the visualization extension is implemented:

\`\`\`python
traj = joint_trajectory(...)
viz.animate(traj.q)
\`\`\`

This demonstrates direct reuse of the time-major numerical convention.

## 67. Explicitly outside Moro 0.5.0

The following remain outside the trajectory scope:

- multiple waypoints;
- piecewise trajectories;
- splines;
- segment blending;
- trapezoidal velocity profiles;
- jerk-limited S-curves;
- automatic time allocation;
- time scaling from velocity limits;
- time scaling from acceleration limits;
- synchronized physical-limit planning;
- joint-limit enforcement;
- robot-aware trajectory validation;
- collision-aware planning;
- obstacle avoidance;
- pose trajectories;
- orientation interpolation;
- SLERP;
- online trajectory generation;
- time-optimal generation;
- dynamic feasibility checks;
- torque-constrained planning;
- direct variable-time animation support.

## 68. Acceptance criteria

The trajectory block is ready for implementation when:

1. \`joint_trajectory()\` generates numerical time-major \`q\`, \`qd\`, and \`qdd\` arrays;
2. \`position_trajectory()\` generates numerical time-major \`p\`, \`v\`, and \`a\` arrays;
3. Cartesian outputs always have three components;
4. 1-DOF joint inputs may be scalar while outputs remain \((N,1)\);
5. multi-DOF scalar broadcasting is rejected;
6. the explicit time vector is finite, one-dimensional, strictly increasing, and may be nonuniform;
7. the absolute time origin does not affect the physical profile;
8. public derivatives are with respect to physical time;
9. linear interpolation supports endpoint positions only;
10. cubic interpolation supports endpoint positions and velocities;
11. quintic interpolation supports endpoint positions, velocities, and accelerations;
12. omitted supported derivative conditions default independently to zero;
13. explicitly supplied unsupported derivative conditions are rejected;
14. the closed-form normalized-time polynomial coefficients satisfy all endpoint conditions;
15. joint and Cartesian generation share a common private polynomial core;
16. runtime generation is NumPy-based rather than symbolic;
17. no endpoint patching hides coefficient errors;
18. \`JointTrajectory\` exposes \`t\`, \`q\`, \`qd\`, \`qdd\`, \`method\`, \`samples\`, \`duration\`, and \`dof\`;
19. \`PositionTrajectory\` exposes \`t\`, \`p\`, \`v\`, \`a\`, \`method\`, \`samples\`, and \`duration\`;
20. trajectory generation remains independent of \`Robot\`;
21. Cartesian trajectory reuse with IK is explicit through \`cart.p\`;
22. no special \`PositionTrajectory\` overload is required in IK;
23. visualization may accept raw numerical joint vectors/matrices without importing trajectory types;
24. existing dict-based visualization remains backward compatible;
25. animation does not claim to reproduce nonuniform physical timing automatically;
26. no 0.5.0 implementation introduces waypoints, splines, trapezoidal/S-curve profiles, automatic timing, pose interpolation, collision-aware planning, or dynamic feasibility planning.
