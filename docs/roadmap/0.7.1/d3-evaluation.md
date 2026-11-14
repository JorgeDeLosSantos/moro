# Moro 0.7.1 — IK-071-A evaluation and IK-071-B gate

> **Status:** Preliminary exploratory comparison. Regression tests against Moro are part of PR #75; results of its CI must be checked before declaring this evaluation complete.
> **No further solver code change** is proposed in this document.

## Baseline

IK-071-A is merged (PR #74). It separates rejected LM damping trials from accepted-step stagnation, retains `max_iter` as an unconditional attempt limit and keeps `damping=1.0` and `stagnation_iterations=5` unchanged.

For the original reachable RR and RRR problems, an independent reconstruction of the LM numerical loop predicts the corrected algorithm converges with the **default stagnation threshold**, using physically equivalent tolerances:

| Manipulator | Length 200 (mm) | Length 0.2 (m) |
| --- | --- | --- |
| RR, q0=[0.1, 0.1] | 10 attempted steps | 9 attempted steps |
| RRR, q0=[0.5, 0.5, 0.5] | 14 attempted steps | 9 attempted steps |

**Experimental artifact:** `tests/test_ik_071_regressions.py` now asserts these cases at both length scales in actual Moro, and separately protects finite nonconvergence for unreachable or fixed-joint-limit cases.

## Seeded-start sensitivity: independent reproduction (not library CI)

To inspect the remaining scale dependence rather than promising general invariance, the same mathematical update/rejection rules and forward kinematics were independently reproduced for 150 integer seeds `0..149`, each using `np.random.default_rng(seed).uniform(-pi, pi, n)` to generate a *single* start, with the same initial `damping=1.0`. Linear tolerances and targets were scaled together; joint angular limits were unchanged. The baseline and fixed methods used the same seeds and case definitions.

| Manipulator | Scale | Pre-fix successes / 150 | After IK-071-A successes / 150 |
| --- | --- | ---: | ---: |
| RR | 200 | 54 | 111 |
| RR | 0.2 | 113 | 113 |
| RRR | 200 | 35 | 63 |
| RRR | 0.2 | 99 | 99 |

These are **results of the independent numerical reconstruction, not runs of Moro's installed package**, and are exploratory statistics, not general probability estimates or CI acceptance thresholds. They show that termination correction greatly improves large-scale cases without demonstrating full unit invariance.

## Why an unconditional IK-071-B change is not yet warranted

The position LM regularization is `lambda**2 * I`; a linear-unit conversion scales position derivatives of revolute joints but not the prismatic position Jacobian columns in the same manner. Even in R-only robots the same numeric damping causes scale-dependent trajectories.

Simple Jacobian-norm damping or column-normalized Marquardt regularization might improve unit sensitivity, but each changes the meaning of user-supplied `damping` and potentially the selected IK branch, particularly for mixed R/P and full-pose IK. Under 0.7.1 patch constraints, those changes should not be applied without a separate proven compatibility policy.

**Recommendation pending PR #75 CI:** accept IK-071-A as the minimal critical fix; preserve `damping` semantics; consider IK-071-B only with a defined mathematical invariance criterion, comparative fixtures and explicit compatibility analysis. If appropriate, defer a broader dimensionless solver policy to a minor version rather than silently redefining a public argument in a patch.

## Decision checkpoints

1. Confirm PR #75 across supported Python versions (including paired mm/m tests).
2. If all pass, merge the scale regression suite to `develop`.
3. Decide explicitly whether 0.7.1 should include further damping scaling; do not assume the fix guarantees random-start success.
4. Run release-specific regression, documentation, distribution, and version checks before publishing.

## Related

- [D1 baseline](d1-diagnostics.md)
- [D2 design](d2-design.md)
- [0.7.1 plan](index.md)
