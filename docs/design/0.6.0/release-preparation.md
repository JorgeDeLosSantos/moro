# Moro 0.6.0 — Release preparation (V7)

**State:** candidate under review on `release/0.6.0`.
**Publishing:** NOT YET published to PyPI, no tag or GitHub Release created.

## Scope

This release promotes the completed 0.6.0 visualization enhancements
to the stable `master` line after the functional work and V6 hardening
have been integrated into `develop`.

The source version metadata in the release branch is set to `0.6.0`;
this alone is not a publication. The `CHANGELOG.md` records the release
content under `[0.6.0] - Pending release`. Replace that placeholder
with the **actual** release date immediately before tagging.

## Main changes

- Evaluated DH segments and joint axes and more faithful robot geometry
  with preserved frame/segment topologies.
- Scale-covariant visualization layout and unified camera framing,
  including stable bounds across sampled animation frames.
- Schematic and solid R/P joint representations, with separate colors;
  Three.js uses solid joints by default when metadata is available.
- Touchpad-friendly orbit controls, unambiguous Right view and Fit View
  without animation or orientation reset.
- Three.js is the default for `RobotVisualizer.plot()` and
  `RobotVisualizer.animate()`. Matplotlib remains an explicit choice.
- Configurable `interval` for Three.js playback and direct `ipython`
  dependency.

## Breaking change and migration

In earlier releases, `RobotVisualizer.plot(values)` returned Matplotlib
`(fig, ax)` and `RobotVisualizer.animate(configurations)` returned a
`FuncAnimation`. In 0.6.0 they return `IPython.display.HTML` using
Three.js unless the caller supplies an explicit backend.

To keep Matplotlib return types:

```python
fig, ax = viz.plot(values, backend="matplotlib")
animation = viz.animate(configurations, backend="matplotlib")
```

Notebook users can use the default HTML view directly; the browser must
be able to fetch external Three.js/OrbitControls resources from the CDN.
For offline contexts or standard scripts use Matplotlib explicitly.

## Release gates

- [x] Functional V1–V5-C2 and V6 integrated into `develop`.
- [x] User-tested interactive static and animated Three.js visuals.
- [x] Set `pyproject.toml` and `moro/version.py` to `0.6.0`.
- [x] Review and consolidate feature, compatibility, and breaking
  migration notes in `CHANGELOG.md`.
- [ ] V7 PR: Python 3.11, 3.12, 3.13, 3.14 test matrix green.
- [ ] V7 PR: warnings-as-errors Sphinx build green.
- [ ] V7 PR: `python -m build` produces wheel and sdist;
  `python -m twine check dist/*` validates metadata.
- [ ] V7 PR: clean wheel and sdist installations both pass runtime
  import and Three.js resource smoke tests **outside** the source tree.
- [ ] Review remaining version-sensitive files and release
  documentation after the final CI run.
- [ ] Obtain explicit approval before merging the release PR into
  `master`, creating a version tag or uploading to PyPI.
- [ ] Set the true release date in the changelog before tagging.
- [ ] After approval, create a tag `v0.6.0` on the final merged
  release commit and publish the reviewed artifacts.
- [ ] Confirm PyPI package metadata and installation by version.
- [ ] Synchronize `develop` from `master` after the release.

## Release verification

GitHub Actions `Tests` includes the complete supported Python
matrix and a distribution job that builds the wheel and sdist, checks
metadata using Twine, installs each archive in a fresh virtual
environment and verifies default Three.js plotting/animation and
packaged HTML/JS resources.

`Build and deploy documentation` checks Sphinx with warnings treated
as errors. Its deployment step is skipped on PRs; site deployment only
occurs when changes reach `master`.

## Not in scope

No new kinematics, simulation or rendering capabilities; no refactor of
historical `moro.plotting` functions; no bundling of CDN assets; no
automatic switching of backend for terminal environments; no
background or automated publishing of a release.

The release should preserve the user-validated interactive behavior.
