# Moro 0.7.1 — D2 numerical correction decision

> **Status:** Proposed implementation policy after D1; implementation and empirical evaluation pending.
> **Baseline:** D1 PR #72 and the original Moro 0.7.0 solver.
> **Decision boundary:** Keep 0.7.1 a maintenance release, with reproducible evidence for every changed numerical behavior.

## Findings that drive the design

The LM implementations in `_solve_newton_or_lm` and `_solve_pose_jacobian` both:

1. Reject an LM trial when its Cartesian-position or weighted-pose error does not strictly decrease.
2. Increase `lambda` by `1/damping_scale` on rejection.
3. Retain `q` and count the rejected trial as one algorithm iteration.
4. Report step and/or error stagnation after `stagnation_iterations` consecutive rejected trials, even though each one tries a different damping magnitude.

The RR and RRR regression witnesses fail at five attempts with `lambda=1,2,4,8,16`, while `lambda=32` accepts the first trial, after which the solver converges. D1 CI has confirmed these as strict expected failures, while the rest of the suite passes.

Changing `stagnation_iterations` from five to ten is not a sufficient correction. Nor should a rejected LM step be treated as an accepted step of zero size for stagnation accounting.

## D2-A: Handling LM rejection and stagnation

### Chosen direction: separate concepts, keep the public API

Maintain three internal concerns:

- **Total attempts**: increments on every attempted update; never exceed existing `max_iter`. Retains the public meaning of `iterations`.
- **Consecutive rejected LM trials**: resets when an improving trial is accepted. Rejection increases damping but *does not* increment the accepted-step stagnation counters.
- **Accepted-step stagnation**: a step that was accepted, but whose effective joint movement or error reduction falls under the configured thresholds, counts toward `stagnation_iterations`. For Newton and CCD, preserve their current semantics.

Use a **finite internal rejection budget** (candidate: 12 or `max(12, stagnation_iterations)`, subject to targeted tests); if exceeded, return nonconvergence with a truthful reason rather than hanging or overstating success. The `max_iter` budget remains an unconditional upper bound.

Do not create a new public argument for the rejection budget in 0.7.1. Do not silently reset `max_iter` or hide rejected attempts in `iterations`.

### Edge cases and diagnostics

- If joint-limit clipping produces no feasible movement for all repeated trials, the solver must terminate within a bounded number of attempts. Consider whether the existing step-stagnation message or error-stagnation message is most truthful; prioritize backward-compatible result shape and avoid misleading claims.
- If `lambda` would become nonfinite or mathematically invalid, preserve a numerical failure status.
- Near convergence, strict improvement can become dominated by floating-point precision; the rejection budget must prevent endless retries.
- An accepted update with a tiny effective step and large error must still count as possible *true* stagnation.
- Explicitly test `stagnation_iterations=1` and `max_iter` less than the rejection budget; neither should bypass finite termination.
- Apply the same rejection-accounting rule to `solve_pose_ik`, while keeping its **independent position and orientation convergence criteria**.

### Alternatives rejected for the first patch

1. **Increase `stagnation_iterations` globally:** masks the early-stop symptom and changes Newton/CCD behavior without evidence.
2. **Disable all stagnation checks for LM:** causes poor diagnostics and unnecessary work for infeasible cases.
3. **Count only accepted attempts toward `max_iter`:** changes documented iteration semantics and risks an unbounded number of attempted updates.
4. **Introduce a new trust-region optimizer:** unnecessary complexity for this bug-fix cycle.

## D2-B: Unit scale and damping

The existing position LM update is

```text
dq = solve(J.T @ J + lambda**2 * I, J.T @ error)
```

When link lengths and the target are scaled by factor `s`, revolute-joint position Jacobian columns and the position residual scale by `s`. A fixed numeric `lambda` does not, so the ratio of the regularization term to `J.T @ J` changes. The scale dependence is real even when convergence thresholds are scaled consistently.

**Selected incremental policy:**
- First land and evaluate D2-A with default `damping=1.0` and `stagnation_iterations=5` unchanged.
- Measure the original RR and RRR, scaled counterparts, and fixed-seed success classifications **after D2-A**. Compare convergence rates, total attempts and terminal diagnostics; do not require identical iteration counts.
- Implement a scale-adaptation change in a separate, conditional PR only if the remaining scale dependence is important enough and the compatibility impacts are acceptable.

Evaluate at least these candidates for the second PR:
- A bounded LM damping warm-up/adaptation based on observed trial outcomes, preserving the numeric meaning of user-supplied `damping`.
- A Jacobian-derived characteristic damping scale with explicitly specified behavior for custom `damping`. The latter changes existing numerical semantics if applied unconditionally.
- A more comprehensive change of variable scaling for mixed R/P joints or weighted pose IK; defer to a future minor release if it requires user-facing choices or unit assumptions.

Do **not** simply divide a mixed-joint Jacobian by a characteristic link length: revolute and prismatic columns transform differently when linear units change. Position and orientation rows in pose IK similarly cannot be scaled identically. The `position_weight` / `orientation_weight` API already defines a weighted objective that must be respected.

**Tolerance conventions:** position IK `tol` and `error_change_tol` have linear-position units, `step_tol` has joint-coordinate units (mixed R/P steps have different units), and pose IK `error_change_tol` measures a weighted mixed residual norm. Explicit numerical equivalence across unit systems therefore requires care; no implicit automatic tolerance normalization is approved for 0.7.1.

## D3 implementation handoff and acceptance

**IK-071-A** (implement first):
- D2-A rejection-versus-stagnation accounting in both position and pose LM.
- Maintain existing `IKSolution` / `PoseIKSolution` signatures, explicit initial damping, convergence tolerances and `max_iter` semantics.
- Convert the two strict xfail RR/RRR witnesses to ordinary passing tests once behavior is corrected.
- Protect fixed prismatic limits, unreachable targets, `stagnation_iterations=1`, `max_iter`, near singularities, mixed joints, Newton and CCD.
- Confirm CI across Python 3.11–3.14 plus packaging and documentation.

**IK-071-B** (conditional; not automatically approved by IK-071-A):
- Write down comparative results after A on mm/m and mixed-joint cases.
- Choose the *smallest* justified scale-robustness change with no undocumented API semantic shifts.
- If no patch-compatible modification is satisfactory, explicitly defer general unit invariance rather than introducing a hidden numerical contract change.

## Release boundaries

Do not change the release version until the candidate passes full verification. Do not alter default `stagnation_iterations` or make multi-start implicit. Any new termination reason must use the existing result `message` field without changing the type of the result. A true guarantee of convergence or invariance under arbitrary units/weighting is **not** a release acceptance requirement.

## References

- [0.7.1 plan](index.md)
- [D1 diagnostic evidence](d1-diagnostics.md)
- [Inverse-kinematics theory](../../source/theory/inverse-kinematics.md)
