# Moro 0.7.0 — Detailed design: multi-waypoint trajectories (T-01–T-03)

> **Status:** Detailed design for review; no implementation in this change.
>
> **Scope:** T-01 joint/Cartesian waypoints, T-02 piecewise polynomial interpolation, T-03 continuity at segment boundaries, and cubic shape-preserving derivative estimation.
>
> **Planning reference:** [0.7.0 inventory](../../roadmap/0.7.0/index.md), proposed in [PR #38](https://github.com/JorgeDeLosSantos/moro/pull/38).
>
> **Implementation base:** Moro 0.6.1 (`moro/trajectory.py`). The design decisions below were discussed and accepted as an initial contract, subject to review of this document before PR-A1.

## 1. Goals and non-goals

Extend the existing robot-independent point-to-point functions to time-parameterized trajectories through multiple waypoints. Retain educational transparency: each segment is a known linear, cubic Hermite, or quintic Hermite polynomial; continuity follows from explicit shared endpoint conditions.

Goals:

- Preserve `joint_trajectory()`, `position_trajectory()`, `JointTrajectory` and `PositionTrajectory` and their public semantics.
- Accept ordered joint or Cartesian-position waypoints with explicit arrival times.
- Evaluate at caller-supplied time samples or a fixed number of samples per segment.
- Support prescribed endpoint derivatives and a simple default of zero derivatives.
- Support an explicit *shape-preserving* cubic option using local PCHIP-style interior slopes.
- Verify continuity from the actual polynomials, not just sampled arrays.
- Preserve time-major NumPy outputs for IK, visualization and later inverse dynamics.

Non-goals for T-01–T-03:

- Time allocation or optimization, velocity/acceleration feasibility limits (T-05).
- Trapezoidal profiles (T-04), physical-time visualization playback (T-06).
- Pose/orientation interpolation, SLERP, joint-limit enforcement, collision checking or robot-aware planning.
- C2 global cubic splines, automatic quintic derivative estimation, PCHIP's usual nonzero endpoint policy, online replanning or a public `Path` / `Segment` hierarchy.
- Persistent compiled trajectory evaluators or new public dependencies.

The trajectory module must not import `moro.core`, IK, dynamics, or visualization. SciPy already exists as a package dependency, but a new dependency is not required for T-01–T-03.

## 2. Public API: additional entry points

Keep the old functions unchanged:

```python
joint_trajectory(q0, qf, t, *, method="quintic",
                 qd0=None, qdf=None, qdd0=None, qddf=None)

position_trajectory(p0, pf, t, *, method="quintic",
                    v0=None, vf=None, a0=None, af=None)
```

Add two standalone functions:

```python
def joint_waypoint_trajectory(
    waypoints,
    waypoint_times,
    *,
    method="quintic",
    derivative_policy="zero",
    velocities=None,
    accelerations=None,
    sample_times=None,
    samples_per_segment=None,
) -> JointTrajectory:
    ...

def position_waypoint_trajectory(
    waypoints,
    waypoint_times,
    *,
    method="quintic",
    derivative_policy="zero",
    velocities=None,
    accelerations=None,
    sample_times=None,
    samples_per_segment=None,
) -> PositionTrajectory:
    ...
```

Argument families:

- **Motion specification:** `waypoints`, `waypoint_times`.
- **Polynomial and boundary specification:** `method`, `derivative_policy`, `velocities`, `accelerations`.
- **Sampling:** `sample_times` OR `samples_per_segment`.

The default interpolation method remains `"quintic"`. New functions return existing result classes, with no added required fields. Old positional/keyword calls and output shapes must remain valid.

## 3. Waypoint convention and validation

Let `K` be the waypoint count (`K >= 2`), `d` the coordinate dimension, and `N` the sample count (`N >= 2`).

| Input | Accepted shape | Canonical shape |
| --- | --- | --- |
| Joint waypoints (multi-DOF) | `(K, dof)` | `(K, dof)` |
| Joint waypoints (1-DOF) | `(K,)` or `(K, 1)` | `(K, 1)` |
| Cartesian-position waypoints | `(K, 3)` | `(K, 3)` |

Only three-component Cartesian positions are valid, including planar trajectories with explicit `z=0`. No implicit 2D-to-3D promotion. Robot joint count is not checked here; joint coordinate order is that supplied by the caller.

All arrays must be real, finite, non-empty in each coordinate dimension, convertible to `float`, and copied defensively. Reject ragged, complex, nonnumeric and incorrectly ranked arrays with informative `ValueError`. Do not broadcast a single waypoint or joint coordinate across rows.

Repeated consecutive positions **are allowed** when the times differ. With zero endpoint derivatives such a segment is stationary; prescribed nonzero derivatives may instead produce an excursion returning to the same position. Do not silently label all repeated waypoints as dwell phases.

## 4. Waypoint times and physical derivatives

`waypoint_times` is a one-dimensional array of length `K`:

```python
waypoint_times = [0.0, 1.0, 2.5, 4.0]
```

Require finite real values and strictly increasing order. Segment `i` has true boundaries `[t_i, t_(i+1)]` and positive duration `h_i = t_(i+1) - t_i`. Time origin need not be zero. Do not accept segment durations as an alternative public argument in this iteration; users can construct waypoint times via cumulative sums.

All velocities and accelerations are derivatives with respect to **physical time** (`t`), not normalized segment time. For local coordinate `u=(t-t_i)/h_i`:

- `dx/dt = (1/h_i) dx/du`
- `d²x/dt² = (1/h_i²) d²x/du²`

The generator must not infer a segment's duration from the first/last evaluation sample. A segment may have no sample precisely at either of its true boundaries.

## 5. Sampling contract

`sample_times` and `samples_per_segment` are mutually exclusive when both are non-`None`.

### 5.1. Automatic samples

If neither argument is supplied, resolve to `samples_per_segment=51`.

`samples_per_segment` must be an integer `S >= 2`, rejecting booleans and nonintegral values. For each of `K-1` segments use `S` uniformly spaced timestamps, including the segment endpoints; concatenate without duplicating interior joins.

```text
N = (K - 1) * (S - 1) + 1
```

Automatic sample grids **include each waypoint time exactly once**. Spacing can vary across segments because segment durations may differ.

### 5.2. Explicit samples

`sample_times` must be a finite strictly increasing one-dimensional vector with `N >= 2`. It must start and end at the first and last `waypoint_times` values, respectively, and stay inside the full interval. The initial contract requires equality of the normalized endpoint values; do not silently clamp or extend a caller-supplied vector.

Interior waypoint times need **not** occur in `sample_times`. Return precisely the caller's vector, in its original order, without inserting missing interior waypoint times. The underlying piecewise polynomial still interpolates every waypoint.

Reject both sampling controls being supplied, an invalid count, nonfinite values, reversed or repeated sample times, or endpoints outside the defined motion. An extremely short duration that makes an automatically generated float grid non-strictly increasing must fail with a useful error instead of yielding duplicate times.

### 5.3. Segment ownership at joins

Each interior waypoint time is assigned to the segment **to its right**, and the final endpoint to the final segment (a right-continuous convention). A `searchsorted`-style selection is sufficient; avoid repeated or skipped output rows.

When a derivative is mathematically discontinuous:

- For interior joins, store the **right-hand** derivative.
- For the final endpoint, store the **left-hand** derivative.

Examples: piecewise-linear velocity can jump; cubic-Hermite acceleration can jump. Their stored values at joins are one-sided conventions, **not** a claim that a classical two-sided derivative exists there.

Linear interpolation reports zero *within-segment* acceleration, but its velocity jumps at corners; it does not model impulsive acceleration or physically feasible instantaneous velocity changes.

## 6. Derivative policies and boundary conditions

A single velocity vector `v_i` (and, where relevant, acceleration vector `a_i`) belongs to each waypoint, and the adjacent segments use the **same** vector. This is the mechanism for achieving polynomial continuity.

Allowed `method` values stay `"linear"`, `"cubic"` and `"quintic"` (case-normalized), and `derivative_policy` accepts `"zero"` and `"shape_preserving"` (case-normalized).

### 6.1. Policy `zero` (default)

- If `velocities` is omitted, initialize `(K,d)` zeros.
- If `accelerations` is omitted for quintic, initialize `(K,d)` zeros.
- If derivative matrices are supplied, use them exactly after validation.
- No implicit broadcasting or automatic inference for partially supplied matrices.
- With all default derivatives, the robot comes to instantaneous rest at every waypoint; **no positive dwell duration** is implied.

For 1-DOF, derivative arrays `(K,)` or `(K,1)` are accepted and normalized to `(K,1)`. For other joint dimensions, derivatives must have shape `(K,dof)`; Cartesian derivatives always have shape `(K,3)`. Scalar broadcasting is forbidden, even for 1-DOF across all waypoints.

A cubic with prescribed velocities is C1. A quintic with prescribed velocities and accelerations is C2. Prescribed derivatives may cause overshoot, high acceleration or other physically undesirable behavior: no feasibility guarantee is implied.

### 6.2. Policy `shape_preserving`

- Only legal with `method="cubic"`.
- `velocities` and `accelerations` must both be omitted; even explicitly passing a matrix of zeros is rejected.
- Estimate **interior** waypoint velocities independently per coordinate with PCHIP-style slopes (see `7).
- Set **both endpoint velocities to zero**, unlike standard PCHIP endpoint slope extrapolation.
- The resulting cubic Hermite trajectory is C1, and is monotonic within each segment for monotone coordinate data, without coordinate-wise segment overshoot.
- C2 continuity, monotonicity of arbitrary Cartesian-path geometry, and velocity/acceleration limits are **not** guaranteed.

This is intentionally named a *PCHIP-style interior slope policy with resting endpoints*, not a claim of exact equivalence to a standard `PchipInterpolator` across the whole interval.

### 6.3. Valid combinations

| Method | Policy | Explicit velocities | Explicit accelerations | Guaranteed continuity |
| --- | --- | --- | --- | --- |
| `linear` | `zero` | Reject | Reject | C0 |
| `cubic` | `zero` | Accept | Reject | C1 |
| `cubic` | `shape_preserving` | Reject | Reject | C1 |
| `quintic` | `zero` | Accept | Accept | C2 |
| `quintic` | `shape_preserving` | Reject | Reject | Not supported |

`derivative_policy="zero"` with `method="linear"` means only that the default argument is accepted; it does **not** impose zero velocity in the linear segments.

Incompatible policies, wrong derivative dimensions, nonfinite derivatives and attempts to pass acceleration boundaries to a cubic must raise `ValueError` with the conflicting argument(s) identified.

## 7. Shape-preserving interior slopes

For one coordinate (apply independently to every column), define:

```text
h[i] = t[i+1] - t[i]
d[i] = (x[i+1] - x[i]) / h[i]
```

For interior waypoint `i`:

- If `d[i-1] == 0` or `d[i] == 0` or the slopes have opposite signs, choose `v[i] = 0`.
- Otherwise use the weighted harmonic mean:

```text
w1 = 2*h[i] + h[i-1]
w2 = h[i] + 2*h[i-1]
v[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
```

Set `v[0]=v[K-1]=0`. All arithmetic and generated slopes must be checked for finite values. Use a numerically stable equivalent where extreme magnitudes could overflow intermediate products or sums.

For `K=2` there are no interior derivatives, so the result coincides with the resting-endpoint cubic point-to-point behavior. Flat coordinates and flat segments must remain flat. The method intentionally sets velocity to zero at a per-coordinate change of direction.

This estimator may be implemented directly (transparent formulas and no extra dependency), with SciPy's `PchipInterpolator` used only as a reference for **interior** slopes in tests. Do **not** use SciPy's standard endpoint slope policy unchanged, because it differs from the resting-endpoint contract.

The no-overshoot guarantee belongs only to this automatic cubic policy. It must not be advertised for arbitrary prescribed derivatives or quintic segments.

## 8. Piecewise polynomial evaluation

Define a polynomial `P_i(u)` for each interval `[t_i, t_(i+1)]` and evaluate position, first derivative and second derivative in physical time.

- **Linear:** endpoint positions alone, piecewise-constant velocity, zero second derivative within segments.
- **Cubic Hermite:** endpoint positions and shared endpoint velocities, one cubic per segment, C1 across joins.
- **Quintic Hermite:** endpoint positions, shared velocities and accelerations, one quintic per segment, C2 across joins.

Do not implement general global splines in this step. Derivative values at joins are shared scalar/vector inputs; no separate left/right user controls are exposed.

### Internal refactoring direction (private names are illustrative)

Keep public functions thin. Candidate helpers:

| Helper | Purpose |
| --- | --- |
| `_prepare_waypoints()` | Validate joint/Cartesian waypoints and normalize dimensions |
| `_prepare_waypoint_times()` | Validate K strictly increasing finite times |
| `_prepare_sample_times()` | Resolve explicit versus per-segment sampling |
| `_normalize_derivative_policy()` | Parse `zero` / `shape_preserving` |
| `_prepare_waypoint_derivatives()` | Validate matrices and policy-method combinations |
| `_estimate_shape_preserving_velocities()` | Compute interior slopes, zero endpoints |
| `_evaluate_polynomial_segment()` | Evaluate one segment at arbitrary samples using true `t0,tf` |
| `_piecewise_polynomial_trajectory()` | Assign samples to intervals, evaluate and assemble arrays |

Keep these helpers private; the exact decomposition can be adjusted during PR-A1/A2 to avoid duplication. Avoid a public `Waypoint` class, an inheritance hierarchy or per-segment result classes until a concrete later capability requires them.

`_polynomial_trajectory()` in 0.6.1 derives duration from `t[0]` and `t[-1]`. Refactor to share the mathematical kernel with a private evaluator whose **true segment times are explicit**. Existing point-to-point wrappers should then delegate to that evaluator without changing their observable behavior.

## 9. Result types, compatibility and integration

Return the unchanged types:

| Result | Time-major arrays | Other fields |
| --- | --- | --- |
| `JointTrajectory` | `t:(N,)`, `q/qd/qdd:(N,dof)` | `method`, `samples`, `duration`, `dof` |
| `PositionTrajectory` | `t:(N,)`, `p/v/a:(N,3)` | `method`, `samples`, `duration` |

Maintain defensive copying and validation in the existing dataclasses. No new required metadata fields in this increment; a sampled result cannot necessarily reconstruct the exact piecewise polynomial between samples.

For two waypoints and matching samples/boundary conditions, the new polynomial functions should numerically match `joint_trajectory()` and `position_trajectory()` to floating-point tolerances.

Downstream interoperability is structural: the joint arrays match numerical `DynamicsSolution` conventions; `traj.q` remains valid for existing visualizer input; `cart.p` remains valid for `solve_position_trajectory()`. The new functions do **not** perform IK or dynamics and do **not** make animations reproduce nonuniform physical time automatically.

The code and documentation must not silently change argument names, array orientation, return types, default visualization backends, frame conventions or old `method` behavior.

## 10. Required tests and acceptance criteria

Build on `tests/test_trajectory.py`, possibly with a dedicated multipoint test file if test organization warrants it.

### 10.1. Regression and equivalence

- Keep all existing point-to-point tests passing.
- For `K=2`, compare old vs new numerical position, velocity, acceleration for linear/cubic/quintic, including nonzero supported boundary derivatives, arbitrary time origins and nonuniform evaluation times.
- Confirm output shapes and copying semantics for both result classes.
- Confirm `shape_preserving` with `K=2` matches the resting-endpoint cubic case.

### 10.2. Geometry, timing and evaluation

- Test 1-DOF vector input and `(K,1)` equivalence; multi-DOF and Cartesian `(K,3)`.
- Test 3+ waypoints, repeated positions, constant coordinates and planar Cartesian data with explicit z.
- Verify interpolation at **every waypoint time analytically**, even when omitted from `sample_times`.
- Verify `N=(K-1)(S-1)+1` for default and explicit `samples_per_segment`.
- Verify all automatically sampled interior waypoint times occur once.
- Verify an explicit `sample_times` array is returned unmodified, with nonuniform spacing and possibly absent interior waypoints.
- Verify right-hand derivative convention at interior knots and left-hand at final knot.
- Verify time-origin translation invariance.
- Verify duration scaling with appropriate scaling of prescribed derivatives (`v -> v/alpha`, `a -> a/alpha²`).

### 10.3. Mathematical continuity

At **every interior waypoint**, directly evaluate both neighboring segment polynomials at the shared time:

- Linear: left/right positions coincide (C0); do not claim C1.
- Cubic with zero/prescribed/shape-preserving slopes: left/right position and velocity coincide (C1); acceleration need not coincide.
- Quintic with zero/prescribed derivatives: left/right position, velocity and acceleration coincide (C2).
- Do not infer continuity solely from differences of potentially sparse sampled outputs.

### 10.4. Shape-preserving behavior

- Monotone increasing/decreasing per-coordinate waypoints produce no segment overshoot.
- Direction reversals force zero slope for the reversing coordinate.
- Constant/plateau coordinates remain constant on the corresponding segments.
- Unequal segment durations are handled correctly.
- Compare interior slopes with a trusted PCHIP reference, but verify our resting-endpoint rule separately.
- Test separately for joint and Cartesian data, including coordinates that change direction at different waypoints.

### 10.5. Validation and numerical failure

- Reject `K<2`, wrong dimensions, ragged/complex/nonnumeric/nonfinite waypoint data.
- Reject missing/mismatched waypoint times, repeated/decreasing/nonfinite times.
- Reject sample times outside motion, missing required endpoints, repeated/nonfinite values, and simultaneous sampling controls.
- Reject noninteger/boolean samples-per-segment and `S<2`.
- Reject invalid method/policy names and incompatible method/policy/derivative combinations.
- Reject badly shaped derivative matrices, scalar broadcasting, and nonfinite derivatives.
- Reject nonfinite computed polynomial values, derivatives or generated slopes.

In tests of numerical equality use justified tolerances scaled to expected quantities. Do not use plotting appearance as the only mathematical acceptance criterion.

## 11. PR-sized implementation plan

Work on feature branches based on an up-to-date `develop`. PRs should be independently reviewable and integrate sequentially, with full regression and relevant documentation before each merge.

| PR | Scope | Definition of done |
| --- | --- | --- |
| **A1 — Core polynomial refactor** | Separate true segment boundaries from evaluation sample times; reuse current coefficients for point-to-point calls | Existing API and tests remain unchanged; new partial-segment evaluation tests pass |
| **A2 — Waypoint inputs + linear** | Public functions, waypoint/time normalization, explicit/per-segment sampling, knot ownership; start with linear | Validated time-major outputs, corner derivative convention, shape and edge-case tests |
| **A3 — Cubic/quintic piecewise** | `zero` and prescribed shared waypoint derivatives, C1/C2 | Analytic join continuity, K=2 equivalence, full boundary-validation tests |
| **A4 — Shape-preserving cubic** | PCHIP-style interior slopes with endpoints at rest | C1, no coordinate overshoot, reversal/plateau/nonuniform-time tests |

Proposed branch names:

```text
feature/0.7.0-trajectory-core
feature/0.7.0-waypoints
feature/0.7.0-piecewise-polynomials
feature/0.7.0-shape-preserving
```

Each PR should include relevant user-guide/API updates, examples suitable for Jupyter/Colab, regression tests and migration notes if necessary. A1 is deliberately **internal-only**. Do not begin A1 before reviewing this design.

## 12. Explicit deferred work and review checklist

Defer trapezoidal velocity profiles (T-04), basic kinematic limits/time allocation (T-05), nonuniform physical-time animation playback (T-06), global cubic C2 splines, quintic auto-estimation and public piecewise-polynomial re-evaluation until separately designed.

Before authorizing PR-A1, confirm:

- [ ] Public API signatures and the no-breaking-change requirement.
- [ ] Waypoint/time validation including repeated positions.
- [ ] Explicit/automatic sampling and right-hand join semantics.
- [ ] Method-policy compatibility and derivative matrix conventions.
- [ ] PCHIP-style interior slopes with zero endpoint velocities.
- [ ] The four-PR sequence and acceptance tests.

This document records **design**, not delivered behavior. The 0.7.0 roadmap and any implementation docs must not claim these new functions exist until their code and tests are merged.
