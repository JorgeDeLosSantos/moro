# Moro 0.6.0 — Release hardening (V6)

**Status:** V6 completed and integrated into `develop` via PR #29.
**Release status:** V7 is preparing package metadata for 0.6.0 on
`release/0.6.0`; no final tag or PyPI publication has occurred.

## Scope and review policy

V1–V5-C2 (visualization scaling, DH geometry, joint representations,
framing, camera interactions and Three.js default) have completed their
functional development and notebook validation. V6 must stabilize and
verify their behavior, not introduce new visualization features or
redesign the public API.

## Cross-cutting contracts

| Contract | Expected 0.6.0 behavior |
| --- | --- |
| `RobotVisualizer.plot(values)` | Three.js `IPython.display.HTML` |
| `RobotVisualizer.animate(configurations)` | Three.js `IPython.display.HTML` |
| Explicit `backend="matplotlib"` | Matplotlib figure/axes or `FuncAnimation` |
| `VisualizationStyle(joint_representation="auto")` | Three.js solid R/P for complete joint metadata; otherwise simple |
| Matplotlib with default style | Spherical markers |
| Legacy `SceneData` without joint metadata | Simple marker fallback |
| `interval` | Positive finite milliseconds in both animation backends |
| Camera navigation | Standard touchpad orbit, Right preset and orientation-safe Fit View |
| Robot length units | Scaling covariant under m/mm changes |
| Three.js delivery | HTML depends on browser CDN availability |

The Python API's default return type changes in 0.6.0. This is a
deliberate breaking change documented in the User Guide and CHANGELOG.

## V6 checks and implementation

- [x] Merge V5-C2 to `develop` via PR #28; GitHub CI green.
- [x] Review direct dependency metadata. `ipython` is required by the
  default HTML backend and present in `pyproject.toml`.
- [x] Update the installed-wheel CI smoke test to check the **actual**
  default static and animated Three.js entry points and verify that
  HTML/JS package resources are installed.
- [x] Avoid hard-coded 0.5.0 assertions in tests that must survive a
  future 0.6.0 version bump. Version still must match package metadata.
- [x] Document notebook/CDN requirements, explicit offline Matplotlib
  behavior, and development branch installation.
- [x] Verify Python 3.11–3.14, distribution smoke and warnings-as-errors
  Sphinx on the V6 PR.
- [x] Review clean installed-wheel smoke test evidence and release gates.
- [x] Merge V6 into `develop` after CI success (merge commit
  `2fb54230c2576d532217e6905aaf1f1981acec58`).

## Known limitations deliberately **not** addressed in V6

- Three.js template loads CDN assets; fully offline interactive browser
  rendering is not promised.
- Matplotlib does not draw kinematic solid cylinder/prism joints;
  `schematic` remains its explicit kinematic option.
- Fitted animation bounds cover the sampled configurations, not a
  mathematically continuous swept volume between frames.
- Legacy `moro.plotting` utilities remain untouched: eliminating public
  or importable compatibility code is not part of a release-hardening PR.
- The browser animation `interval` controls playback frames, not
  physically sampled trajectory timestamps.

## V7 — Separate release preparation (in progress)

1. Review and date the 0.6.0 section of `CHANGELOG.md`, including the
   breaking default-backend behavior and migration instructions.
2. Update `pyproject.toml` and `moro/version.py` together from
   `0.5.0` to `0.6.0`.
3. Verify no tests, workflows, documentation or metadata still require
   the previous version number.
4. Build wheel and sdist; install from the built artifacts in clean
   environments, recheck package resources and notebook output.
5. Run the complete supported Python matrix and Sphinx.
6. Integrate the release branch through the usual review process and
   publish the final release/tag to PyPI only after explicit approval.

The release itself remains a separate user-approved operation.
