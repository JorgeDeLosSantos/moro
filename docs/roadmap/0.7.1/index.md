# Moro 0.7.1 — Inverse-kinematics robustness maintenance

> **Status:** Planning proposal; no numerical changes implemented yet.
> **Baseline:** Moro 0.7.0; `master` and `develop` were identical at `79f34b4` when planning began (2026-11-12).
> **Theme:** Fix premature numerical IK termination and investigate dependence on geometric scale while preserving the public API.
> **Release boundary:** Patch release; no new solvers, multi-start, pose trajectories, or unrelated changes.

## Motivation and observed issue

In `solve_position_ik()`, LM may reject several consecutive trial steps while increasing its damping parameter, but each rejected trial also increments the step- and error-stagnation counters. Five rejected attempts can therefore terminate the solve just before a smaller damped step would be accepted.

Two reported reproductions (default `method="lm"`, `damping=1`, `damping_scale=0.5`):

| Manipulator | Target | Initial q | Default stalls | With `stagnation_iterations=6` |
| --- | --- | --- | --- | --- |
| RR: `(200,0,0,q1)`, `(200,0,0,q2)` | `[200,200,0]` | `[0.1,0.1]` | Fails after 5 attempts (error ~240.2383) | Converges after 10 attempts (error ~3.62e-8) |
| RRR: `(0,pi/2,200,q1)`, `(200,0,0,q2)`, `(200,0,0,q3)` | `[200,200,200]` | `[0.5,0.5,0.5]` | Fails after 5 attempts (error ~276.1885) | Converges after 14 attempts (error ~3.12e-8) |

These are user-reported observations, **not yet independently verified by the regression suite**.

The current LM step is `(J.T @ J + lambda**2 * I)^-1 @ J.T @ e`. An unchanged numerical `lambda` is not generally scale invariant under conversions of Cartesian length units. `tol` and `error_change_tol` also have length units for position IK. Physical-unit-equivalent experiments must scale those tolerances consistently.

Relevant implementation: `moro/inverse_kinematics.py` (`_solve_newton_or_lm`, `_solve_pose_jacobian`, `_compute_ik_step`).

## Scope

**In:**
- Document and reproduce scale-sensitive LM behavior, rejected steps, joint-limit clipping, and stagnation outcomes.
- Improve termination logic so adapting LM damping is not confused with demonstrated stagnation.
- Evaluate scale-aware LM damping or normalization; adopt only if evidence supports it, while accounting for position-vs-orientation dimensions and mixed revolute/prismatic joints.
- Regression-check `solve_position_ik()`, `solve_position_trajectory()`, and `solve_pose_ik()`; keep Newton and CCD behavior stable.
- Maintain clear diagnostics, deterministic tests, and user-facing documentation.

**Out:**
- New IK algorithms, mandatory multi-start, guaranteed convergence or global branch continuity, optimization, new trajectory APIs, general solver architecture redesign, workspace or dynamics features.
- Automatic increase of `stagnation_iterations` without empirical evidence.

## Work sequence and gates

### D1 — Diagnose and establish baselines

1. Reproduce reported RR and RRR with the existing public API and capture exact outcomes at `stagnation_iterations=5,6,10`.
2. Make scale-paired models (e.g. link lengths `0.2` m and `200` mm) and scale **all Cartesian targets and absolute position/error-change tolerances** consistently; leave joint quantities in radians. Record whether results differ in success, count, and residual *in physical units*.
3. Run seeded `q0=None` cases and fixed initial configurations; compare random-start success rates using identical seeds; no nondeterministic assertion thresholds.
4. Instrument rejected vs accepted LM steps, damping values, proposed and clipped step norms, residual improvement, and termination cause in test-only instrumentation or a standalone diagnostic script, without changing public result classes.
5. Cover reachable/unreachable, tight joint limits, singular/near-singular, revolute/prismatic, and 1-DOF examples; inspect relevant pose-IK behavior.

**Exit:** reproducible evidence and a concise technical decision record naming causes, not merely symptoms. Do not change defaults or solver internals during D1.

### D2 — Select correction

Evaluate independently:
- **Stagnation:** separate rejected LM trials from accepted near-zero-progress steps; explicitly bound repeated unsuccessful damping adjustments so truly unproductive solves terminate.
- **Scale:** compare Jacobian-based damping scale, dimensionless position normalization, or controlled alternatives. Check absolute vs relative error thresholds.
- **Compatibility:** preserve `damping`'s explicit semantics if reasonably possible; if normalization changes its meaning, document and evaluate patch-version suitability.
- **Pose IK:** separately account for the dimensional difference between translational and rotational residuals/Jacobian rows.

**Exit:** written algorithm choice, edge-case handling, and precise acceptance metrics before implementation.

### D3 — Implement in small PRs

- **IK-071-A:** stagnation/adaptation correction plus focused tests.
- **IK-071-B (conditional):** scale-robust LM policy plus tests, only if D1/D2 justify it.

No public argument renames or new result class hierarchy. Changes to convergence, iteration counts, and the returned valid branch are possible, but must be explained and tested.

### D4 — Integration and verification

- Test all IK methods and position/pose trajectory entry points, as well as symbolic parameters and joint-limit clipping.
- Validate equivalent m/mm models with physically equivalent tolerances.
- Ensure genuinely unreachable or constrained cases still terminate with accurate diagnostics.
- Run full CI across the supported Python versions; docs and packaging smoke checks.

**Exit:** no unexplained regressions; evidence supporting any changed defaults and documented behavior.

### D5 — Release 0.7.1

- Update `CHANGELOG.md`, user guide/theory notes as needed, release notes, `pyproject.toml` and `moro/version.py` only when preparing the release candidate.
- Test documentation build, package wheel/sdist and clean installations.
- Merge reviewed candidate to `master`; create an unprefixed `0.7.1` tag only after acceptance.
- Explicit maintainer authorization is required before publishing a stable GitHub Release, which triggers PyPI Trusted Publishing.
- Verify PyPI installation and synchronize `develop` after release.

Follow [maintainer release procedure](../../.github/RELEASING.md).

## Acceptance principles

- The two reported reachable examples converge with default public options if a targeted correction can achieve that without sacrificing safety or consistency.
- Equivalent physical setups in different units have comparable outcome classifications after equivalent tolerance scaling; **identical iteration counts are not required**.
- A rejected LM attempt is not automatically proof of true stagnation.
- Infeasible targets and hard joint constraints still produce finite, bounded termination with useful messages.
- Regression tests remain deterministic and target causes, not magic values.
- Default `stagnation_iterations=5` stays unchanged unless D1/D2 explicitly justify altering it.
- Public signatures, result structures, and central mathematical conventions remain stable wherever possible.

## References

- [Moro 0.7.0 release plan](../0.7.0/index.md)
- [Inverse-kinematics theory](../../source/theory/inverse-kinematics.md)
- [Inverse-kinematics user guide](../../source/user-guide/inverse-kinematics.md)
- [Release runbook](../../.github/RELEASING.md)
