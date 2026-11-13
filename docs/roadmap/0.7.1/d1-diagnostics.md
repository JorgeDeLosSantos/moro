# Moro 0.7.1 — D1 baseline diagnostics

> **Status:** Initial source-level analysis and independent numerical reproduction complete. **Moro CI execution of the new regression tests is pending.** No production solver changes in this PR.

## Current code behavior

In `moro/inverse_kinematics.py`, `_solve_newton_or_lm` computes a trial LM step with `J.T @ J + lambda**2 * I`. If the trial does not reduce the error, the implementation rejects it (leaves `q` unchanged), multiplies `lambda` by `1/damping_scale` and still increments both stagnation counters because the effective step and improvement were zero.

Thus `stagnation_iterations=5` can trigger after five **damping adjustments** without any accepted update. In `_solve_pose_jacobian`, related step/error counting occurs as well. Actual pose-target experiments remain to be added before applying a shared fix.

## Reproduction supplied by the user

| Case | `q0` | `stagnation_iterations=5` | `stagnation_iterations=6` |
| --- | --- | --- | --- |
| RR, lengths 200/200, target [200, 200, 0] | [0.1, 0.1] | 5 attempts, error ~240.2383, nonconverged | 10 attempts, error ~3.618925e-8, converged |
| RRR, DH [(0, pi/2, 200), (200, 0, 0), (200, 0, 0)], target [200, 200, 200] | [0.5, 0.5, 0.5] | 5 attempts, error ~276.1885, nonconverged | 14 attempts, error ~3.115992e-8, converged |

An **independent NumPy/SymPy reconstruction of the same DH forward kinematics, Jacobian and LM acceptance/rejection rules** reproduced those outcomes (up to floating-point rounding). In both 200-length cases, damping trials with `lambda=[1, 2, 4, 8, 16]` are rejected and the first accepted trial occurs at `lambda=32`. This external reconstruction is not a substitute for executing Moro's test suite.

## Scale comparison in the independent reproduction

Geometric parameters and targets were scaled by 0.001, from 200 mm to 0.2 m. Position tolerance was changed from `1e-6` to `1e-9`, and `error_change_tol` from `1e-12` to `1e-15`. These are physically equivalent *absolute* Cartesian thresholds. Angular `q0` and `step_tol` were not rescaled.

| Case | Length representation | Stagnation 5 | Stagnation 6 |
| --- | --- | --- | --- |
| RR | 200 | Fails at 5, error ~240.2383 | Converges in 10 |
| RR | 0.2 | Converges in 9 | Converges in 9 |
| RRR | 200 | Fails at 5, error ~276.1885 | Converges in 14 |
| RRR | 0.2 | Converges in 9 | Converges in 9 |

This comparison confirms **non-invariant default behavior** of the reconstructed LM method under length-unit changes. Its regularizer remains numerically `lambda**2 * I` while `J.T @ J` changes quadratically with length scale. It does **not** show that always increasing the damping is the correct fix, especially for prismatic or mixed-joint manipulators.

### Seeded exploratory results (independent reconstruction only)

With 30 identical integer seeds `0..29` per case and the default `q0=None` random-limit sampling:

| Case | Stalls | 200-length success / 30 | 0.2-length success / 30 |
| --- | --- | --- | --- |
| RR | 5 | 14 | 22 |
| RR | 6 | 16 | 22 |
| RR | 10 | 26 | 22 |
| RRR | 5 | 4 | 16 |
| RRR | 6 | 4 | 16 |
| RRR | 10 | 7 | 16 |

Sample sizes are exploratory. These numbers must not be turned into CI success-rate requirements, and they are not guaranteed rates. The first purpose of `scripts/diagnose_ik_071.py` is to run the same comparisons through **actual Moro**.

## Added verification artifacts

- `tests/test_ik_071_regressions.py`: two strict xfail witnesses for known default LM failures (one RR, one RRR); tests verifying convergence when stalled threshold is relaxed; tests verifying reachable scaled targets. Strict xfail is temporary; after implementing a fix, replace it with a normal success assertion.
- `scripts/diagnose_ik_071.py`: calls the actual public `solve_position_ik`, records attempted damping and proposed step sizes through a temporary internal observer, checks fixed-start results, and reports seeded random-start outcomes. Diagnostic only; it does not modify the solver or add public logging.

## Remaining D1 gates

1. Execute added tests against Moro in CI and inspect actual stdout from diagnostic script.
2. Expand controlled cases to truly unreachable targets, imposed joint limits, singular/near-singular starts, and at least one prismatic/mixed-joint case.
3. Examine `solve_pose_ik` separately: translation and orientation have different dimensions and tolerance semantics; no naive position-only normalization should be applied to the weighted pose Jacobian.
4. Produce a D2 decision record comparing a bounded damping-rejection budget, effective stagnation checks on accepted steps, and scale-aware alternatives.
5. Avoid any public API or default change before D2 approval.

## Constraints

- Preserve public function signatures and result-object fields if possible.
- Do not equate a rejected trial with a physically impossible target.
- Maintain controlled termination for genuinely blocked problems.
- `stagnation_iterations=10` alone is not an accepted fix.


## Follow-up D1 coverage (PR #72)

The initial regression commit passed GitHub CI, including Python 3.11–3.14, distribution checks, and documentation. Python 3.11 reported `1639 passed, 2 skipped, 2 xfailed`. The two strict expected failures were the reported default-LM RR and RRR witnesses.

An additional commit extends deterministic coverage to:
- a target unreachable because a prismatic joint is limited to a shorter travel, at both length scales;
- a reachable mixed R/P robot with scaled linear geometry **and** explicitly scaled prismatic limits;
- seeded random-start reproducibility;
- a near-singular reachable position-IK control;
- a reachable full-pose IK control with separate angular and translational accuracy checks.

These are **regression protections**, not claims of a global success rate. Their new CI status must be confirmed before declaring D1 complete.

### D1-to-D2 preliminary design implications

1. Keep the LM damping-adjustment budget independent from an accepted-step stagnation measure. A rejected trial signals an inadequate proposed damping value, not proof that the configuration is stuck.
2. Keep a finite bound on consecutive rejected trials and on total iterations; preserve meaningful messages if damping adaptation cannot produce improvement.
3. Compare normalization policies against mixed R/P systems. Scaling position alone changes only translational Jacobian rows for revolute joints, while prismatic columns retain dimensionless translational derivatives; blindly normalizing a common `J` can change the relative joint-step penalty.
4. Pose-IK weights and orientation residuals require their own examination; pose IK should not inherit a position-only normalization uncritically.
5. Default `damping=1` semantics must be reviewed explicitly before deciding whether a scale-normalized policy is acceptable in a patch release.

The above are **design considerations**, not yet approved numerical changes.
