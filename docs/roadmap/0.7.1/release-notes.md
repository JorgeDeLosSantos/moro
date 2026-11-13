# Moro 0.7.1 — Release notes (candidate)

Moro 0.7.1 is a targeted maintenance release improving the reliability of
numerical inverse kinematics with Levenberg–Marquardt (LM). No new user-facing
features or public API changes are introduced.

## Highlights

- **Fix premature LM stagnation:** successive trial updates rejected during
  damping adjustment are no longer treated automatically as accepted-step
  stagnation. This applies to both position and full-pose inverse kinematics.
- **Preserve safe termination:** LM still honors `max_iter`, counts rejected
  attempts in `IKSolution.iterations` and `PoseIKSolution.iterations`, detects
  effectively blocked joint motion and limits consecutive rejected attempts.
- **Reproducible regression protection:** tests cover the originally failing
  RR and RRR manipulators, metre/millimetre representations, joint limits,
  unreachable targets, mixed revolute/prismatic joints, near-singular
  configurations, pose IK and position-trajectory orchestration.
- **Clearer documentation:** explains how damping adaptation relates to
  stagnation and why changes in length units may affect the solver path.

## Compatibility

The signatures of `solve_position_ik()`, `solve_pose_ik()` and
`solve_position_trajectory()` are unchanged. The default options remain
`method="lm"`, `damping=1.0` and `stagnation_iterations=5`.

Because this version fixes the termination criteria, a call that stopped
prematurely in 0.7.0 can now converge or return a different iteration count.

**Limitations:** the correction does not guarantee convergence from every
initial configuration or full numerical invariance under a change of units.
When switching metres to millimetres, scale the absolute Cartesian
convergence/error-change tolerances accordingly. The semantics of user-supplied
`damping` are unchanged; a broader unit-aware LM redesign is deferred.

## Installation (after publication)

```bash
python -m pip install --upgrade "moro==0.7.1"
python -c "import moro; print(moro.__version__)"
```

## Release validation checklist

- [ ] Candidate PR CI passes Python 3.11–3.14.
- [ ] Documentation builds cleanly.
- [ ] Wheel and source distribution build/install smoke checks pass.
- [ ] `pyproject.toml`, `moro/version.py` and tag version match.
- [ ] Reviewed candidate is merged to `master`.
- [ ] Maintainer explicitly approves creating tag and publishing stable GitHub Release.
- [ ] PyPI publication and clean installation are verified after release.

Publication is triggered by **publishing the stable GitHub Release**,
not by merging this candidate PR or pushing a tag.
