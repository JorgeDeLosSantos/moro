# Moro 0.7.1 — D4 stabilization gate

> **Status:** D4 completed; PR #76 merged after CI approval. Release-candidate validation remains a separate D5 gate.
> **Baseline:** `develop` after IK-071-A (#74) and scaled regression evaluation (#75).
> **Scope:** no numerical algorithm redesign or changed public solver options.

## Verification evidence

- D1 (PR #72) established the original RR/RRR default-LM failures and feasibility controls, initially with two strict xfails.
- IK-071-A (PR #74) separated rejected LM damping adjustments from accepted-step stagnation, while retaining the `max_iter` attempt budget and existing public defaults.
- Scale evaluation (PR #75) passed Python 3.11–3.14, distribution smoke tests and docs build, with **1659 passed, 2 skipped** reported by the Python 3.11 test job. This established that the reported RR/RRR targets are solved both in metres and millimetres when linear tolerances are scaled consistently.
- D4 PR #76 updated theory/user guidance and regression descriptions, and added trajectory/pose IK coverage. Its CI passed Python 3.11–3.14, documentation, and distribution checks; Python 3.11 reported **1662 passed, 2 skipped**. The PR was merged into `develop` as `be8d5d8`.

## D4 acceptance checklist

- [x] No changes to public argument names, defaults, return type shapes or `iterations` accounting.
- [x] Original default-LM RR/RRR regressions promoted to ordinary passing tests.
- [x] Tests cover feasible mm/m targets and unreachable or constrained cases.
- [x] Explicit guards for `max_iter` and `stagnation_iterations=1`.
- [x] Control tests for mixed R/P, near-singular start, seeded initial configurations and reachable full pose.
- [x] The scale-invariance limitation is explicitly documented; full normalization remains out of the accepted patch.
- [x] New D4 trajectory/pose tests and doc changes pass Python 3.11–3.14 and documentation CI.
- [x] Review D4 CI and integrate PR #76.
- [ ] Validate the 0.7.1 release candidate packaging, Sphinx build and clean wheel/sdist installation under D5.

## Important numerical limits

This maintenance patch does **not** guarantee convergence from all initial joint configurations or strict invariance under changes of length units. `damping` retains its original numerical meaning; full unit-aware scaling of mixed revolute/prismatic and weighted full-pose objectives remains a separate future design consideration.

## Handoff to D5

Once PR #76 passes and is integrated, prepare only versioned release metadata (`0.7.1` in `pyproject.toml` and `moro/version.py`), `CHANGELOG.md`, and release notes in a separate candidate PR. Run CI, Sphinx and wheel/sdist smoke tests. Do **not** tag or publish without explicit authorization.
