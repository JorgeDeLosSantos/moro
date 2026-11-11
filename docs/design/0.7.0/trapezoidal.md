# Moro 0.7.0 — Detailed design: synchronized trapezoidal motion (T-04)

> **Status:** Implemented in B1 (#45) and B2 (#46); B3 examples and regression stabilization included in the current increment.  
> **Scope:** Point-to-point, rest-to-rest, synchronized joint-space motion with acceleration/cruise/deceleration phases and prescribed or minimum duration.  
> **Planning reference:** [0.7.0 inventory](../../roadmap/0.7.0/index.md); [multi-waypoint polynomial design](trajectory.md) (T-01–T-03, merged through PR #43).

## 1. Purpose and boundaries

T-04 adds a robot-independent, teaching-friendly motion generator. It complements rather than extends the existing polynomial `joint_trajectory()` and `joint_waypoint_trajectory()` interpolation methods. A shared scalar progress variable makes all joints start and end together and traces a straight line in joint-coordinate space.

**In scope:** synchronized joint-space motion; automatically selected triangular or trapezoidal speed profile; nonzero, signed coordinate changes; fixed positive joint speed/acceleration limits; minimum feasible duration within this profile family; explicit longer durations via uniform time scaling; stationary holds with explicit duration; sampled position, velocity and acceleration; optional typed profile information.

**Not in scope:** independent asynchronous joint profiles, multi-waypoint trapezoidal blending, nonzero initial/final velocities, jerk limits/S-curves, arbitrary path retiming, trajectory feasibility checks for other generators (T-05), Cartesian pose interpolation, actuator/dynamic constraints, robot joint-limit or collision checks, nonuniform visualizer playback (T-06), and public continuous-time profile evaluation.

The library does not query `Robot` or inspect physical joint units; every input coordinate and its limits must use consistent units supplied by the caller.

## 2. Public API proposal

```python
def joint_trapezoidal_trajectory(
    q0,
    qf,
    *,
    v_max,
    a_max,
    duration=None,
    t0=0.0,
    samples=101,
) -> JointTrajectory:
    ...
```

- `q0` and `qf` accept scalars for one DOF and 1D vectors for multiple DOF, using the existing joint endpoint validator; dimensions must match. No coordinate broadcasting.
- `v_max` and `a_max` are strictly positive, finite real **scalars** (broadcast intentionally) or vectors of shape `(dof,)`. No matrices or bools; numeric units apply individually to each joint. In mixed revolute/prismatic robots, scalar limits carry different physical units per coordinate, so explicit vectors are preferable.
- `duration=None` computes minimum duration within the defined synchronized path and symmetric acceleration profile family. Otherwise require a finite strictly positive duration no smaller than the minimum. Any longer duration uniformly scales physical time rather than reoptimizing the phase allocation.
- `t0` is a finite real scalar; `samples` is an integer >= 2, excluding bool. The time array is uniformly spaced from `t0` to `t0 + duration` and preserves both exact endpoints. The generated timeline must be finite and strictly increasing; reject nonrepresentable times/grids.
- Initial and final velocities are always zero. Acceleration is piecewise constant with jumps at phase changes.
- Return the existing `JointTrajectory` type: `t: (N,)`, `q/qd/qdd: (N,dof)`, with `method="trapezoidal"` irrespective of actual triangular/trapezoidal/stationary profile kind.
- Explicit `sample_times` is not part of initial T-04. It can be added later with a well-defined conflict rule for `samples`.

A quick example:

```python
motion = joint_trapezoidal_trajectory(
    [0.0, 0.0], [1.2, -0.8],
    v_max=[0.6, 0.4],
    a_max=[1.0, 0.8],
    duration=5.2,  # optional; minimum is 2.6 s
    samples=101,
)
assert motion.q.shape == (101, 2)
assert motion.profile.kind == "trapezoidal"
```

## 3. Shared normalized progress and limits

For non-stationary motion, let

```text
delta_q = qf - q0
q(t)    = q0 + s(t) * delta_q
qd(t)   = sd(t) * delta_q
qdd(t)  = sdd(t) * delta_q
```

with `s(t0)=0`, `s(tf)=1`, `sd(t0)=sd(tf)=0` and nonnegative `sd`. For coordinates with `delta_q[i] != 0`, derive scalar progress limits:

```text
v_bar = min_i(v_max[i] / abs(delta_q[i]))
a_bar = min_i(a_max[i] / abs(delta_q[i]))
```

Ignore zero-displacement coordinates when computing the minima; they remain constant. The scalar progress variable is dimensionless, so its derivatives have units 1/time and 1/time^2.

With a unit progress distance `s(tf)-s(t0)=1` and symmetric peak acceleration, the unconstrained triangular peak is `sqrt(a_bar)`. Thus:

```text
v_peak = min(v_bar, sqrt(a_bar))
t_accel = v_peak / a_bar
t_cruise = (1 - v_peak**2 / a_bar) / v_peak
t_decel = t_accel
T_min = 2*t_accel + t_cruise
```

For numerical stability, equivalent formulas are allowed, provided phase durations and all outputs are finite and nonnegative. Clamp *roundoff-sized* negative cruise time to zero, not physically invalid inputs. When `t_cruise=0` within a justified numerical tolerance, the profile is **triangular**; otherwise it is **trapezoidal**.

`T_min` is the minimum duration **for this rest-to-rest straight joint path and symmetric bounded-acceleration profile**, not a claim of global robot time optimality or dynamic feasibility. The resulting scalar profile respects all prescribed joint coordinate speed and acceleration bounds in continuous time, including between samples.

## 4. Explicit duration via uniform time scaling

When `duration=T >= T_min`, let `alpha=T/T_min`. Start from the minimum-duration profile `s_min(tau)` defined for `tau in [0,T_min]`:

```text
s(t)   = s_min((t - t0)/alpha)
sd(t)  = sd_min((t - t0)/alpha) / alpha
sdd(t) = sdd_min((t - t0)/alpha) / alpha**2
```

All phase durations scale by `alpha`; actual peak speed is `v_peak/alpha` and actual progress acceleration magnitude is `a_bar/alpha**2`.

This *does not* generally use the maximum available acceleration once duration exceeds the minimum; that is an intentional clarity/simplicity tradeoff. Reject `duration<T_min` with a useful error identifying the infeasibility. Use roundoff-aware comparisons at equality without silently accepting a meaningfully shorter requested duration.

## 5. Analytic phase evaluation and conventions

Let `tau=t-t0` and effective phase parameters `A>0`, `V=A*t_accel`, `t_a=t_accel`, `t_c=t_cruise` and `T=duration`.

| Local phase | Progress `s(t)` | `sd(t)` | `sdd(t)` |
| --- | --- | --- | --- |
| Accelerate, `0 <= tau < t_a` | `A*tau²/2` | `A*tau` | `+A` |
| Cruise, `t_a <= tau < t_a+t_c` | `A*t_a²/2 + V*(tau-t_a)` | `V` | `0` |
| Decelerate, `t_a+t_c <= tau <= T` | `1 - A*(T-tau)²/2` | `A*(T-tau)` | `-A` |

For triangular profiles (`t_c=0`), the central phase has zero duration; at the peak, choose the deceleration phase. Use numerically stable evaluation near endpoints and enforce exact endpoint positions/zero endpoint velocities where justified.

At interior transition times, reported acceleration belongs to the **right-hand** phase. At the final time it belongs to the final phase (**left-hand limit**). At the initial time it belongs to the first phase. Position and velocity are continuous (C1); acceleration jumps at transitions (not C2), and the stored value at a discontinuity is a one-sided convention, not a two-sided derivative.

## 6. Stationary holds

When `q0 == qf` in all coordinates:

- Without an explicit `duration`: reject, since there is no unique positive duration for a rest-to-rest stationary move.
- With a finite positive `duration`: return constant `q`, zero `qd` and `qdd` over the requested time grid, with `method="trapezoidal"` and `profile.kind="stationary"`.
- The stationary metadata convention is: `t_accel=0`, `t_decel=0`, `t_cruise=duration` (the field denotes a hold, not motion at cruise speed), `peak_path_speed=0`, `path_acceleration=0`, and `minimum_duration=None`.

Input limits remain strictly positive and finite even for holds so the validation contract does not depend on whether the displacement happens to vanish.

## 7. Result metadata and compatibility

Use **one result type** `JointTrajectory` and optional, typed, immutable profile metadata rather than a second return tuple or a specialized trajectory subclass.

Candidate public metadata:

```python
@dataclass(frozen=True)
class TrapezoidalProfileInfo:
    kind: str  # "triangular", "trapezoidal", "stationary"
    t_accel: float
    t_cruise: float
    t_decel: float
    duration: float
    minimum_duration: float | None
    peak_path_speed: float
    path_acceleration: float
```

All phase durations and speeds are **effective values after duration scaling**. `peak_path_speed` and `path_acceleration` describe the normalized scalar progress `s`, **not individual joint values**. `minimum_duration` is the unscaled minimum for nonstationary motion, and `None` for a stationary hold. The invariant `t_accel + t_cruise + t_decel == duration` holds to floating-point tolerance.

Add a field to `JointTrajectory`:

```python
profile: TrapezoidalProfileInfo | None = field(
    default=None,
    kw_only=True,
    repr=False,
    compare=False,
)
```

- Preserve existing five positional constructor arguments, time-major numerical arrays, public properties and current custom `repr`.
- Polynomial results continue with `profile=None`.
- The trapezoidal generator populates the field; an attached profile must be of the expected type and compatible with `method="trapezoidal"`. The result remains fundamentally a sampled mutable array container; metadata describes *original generation* and does not update if the user edits its arrays.
- This changes `dataclasses.fields()` and `dataclasses.asdict()` output; document and test that structural change explicitly.
- No generalized metadata registry, new trajectory hierarchy or public continuous-time `profile.evaluate()` in T-04. If a second specialized profile requires metadata in the future, generalize only then.

The current `_normalize_method()` validates only polynomial interpolation names, and must remain unchanged for polynomial generators. Introduce a separate result-label validation path, e.g.:

```python
_POLYNOMIAL_METHODS = {"linear", "cubic", "quintic"}
_TRAJECTORY_METHODS = _POLYNOMIAL_METHODS | {"trapezoidal"}
```

Use the appropriate validator in `JointTrajectory` so `joint_trajectory(..., method="trapezoidal")` and the waypoint polynomial functions still reject unsupported interpolation. `PositionTrajectory` need not be extended by T-04.

## 8. Implementation organization

Keep `moro.trajectory` independent of robot modeling, visualization and dynamics. Separate a **private scalar phase resolver** from a **private analytic scalar evaluator**, allowing tests against exact phase transitions and later internal reuse without making a continuous-time profile public.

Recommended sequence:

| PR | Scope | Exit criterion |
| --- | --- | --- |
| **B1 — Scalar core** | Private resolver/evaluator, duration scaling, triangular/trapezoidal/stationary contracts | Analytic endpoint/phase/limits tests, degeneracy and floating-point cases; no public changes |
| **B2 — Joint API + metadata** | New public generator, normalized joint limits, `JointTrajectory` optional profile, method-validation split | Sync multi-DOF and 1-DOF tests, stationary movement, compatibility; no existing polynomial regression |
| **B3 — Documentation and stabilization** | Teaching examples (R/RR), comparative plots vs quintic, API docs/migration, complete regression | CI green on supported Python versions, docs build, release scope review |

The design-only PR #44 was merged before B1. The private scalar core (PR #45) and public joint-space API with metadata (PR #46) are integrated. B3 adds a [teaching example](../../source/examples/trapezoidal-trajectories.md), theory updates and additional public-result regression tests. Continue to require full CI and review before integration.

## 9. Required tests and acceptance

- **Scalar phase math:** both shape regimes, exact phase boundaries, known numerical cases, C1 and one-sided acceleration convention; nonnegative effective phase times.
- **Continuous-time bounds:** analytic extrema and phase magnitudes `|qd[i]|<=v_max[i]` and `|qdd[i]|<=a_max[i]`, including joints with negative and zero displacements, not merely sampled checks.
- **Duration scaling:** exact geometry and endpoints; when `duration` doubles, velocities halve, accelerations quarter, phase durations double; reject too-short duration.
- **Multijoint synchronization:** shared `s(t)`, signed joint direction, joint arrays and derivative shapes, one-DOF scalar/vector equivalence.
- **Stationary:** explicit hold allowed, otherwise rejected; metadata kind and zero kinematic derivatives.
- **Validation:** mismatched shapes, ragged/complex/nonfinite inputs, zero/negative/boolean limit values, scalar-vs-vector limits, too few samples, nonfinite/indistinguishable time grids, overflow in computed limits or durations.
- **Compatibility:** existing five-argument `JointTrajectory` constructors; polynomial method guards; no change to `joint_trajectory()`, `joint_waypoint_trajectory()` and `PositionTrajectory`; expected new `asdict`/`fields` behavior.
- **Docs:** show normalized-progress interpretation, mixed R/P units, duration policy, discontinuous acceleration, and explicitly differentiate prescribed vs effective limits.

## 10. Design decisions adopted and deferred

**Adopted:** synchronized joint-space straight-path profile; rest-to-rest boundaries; triangular/trapezoidal automatic selection; independently supplied coordinate speed and acceleration bounds; optional prescribed duration via uniform time scaling; explicit stationary holds; one-sided acceleration at phase boundaries; shared `JointTrajectory` with optional `TrapezoidalProfileInfo`.

**Deferred:** full continuous profile object, additional `profile` types, explicit sample times, nonzero endpoint velocity, trapezoidal waypoint blending, arbitrary trajectory retiming and global kinematic-limit analyzer (T-05). T-06 time-aware animation remains a distinct visualizer feature.
