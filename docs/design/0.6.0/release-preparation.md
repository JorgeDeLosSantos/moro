# Moro 0.6.0 — Release preparation (V7)

**State:** 0.6.0 released; `develop` synchronized with `master` at publication.
**Publishing:** GitHub Release `0.6.0` published and PyPI Trusted Publishing workflow completed successfully.

## Scope

This release promotes the completed 0.6.0 visualization enhancements
to the stable `master` line after the functional work and V6 hardening
have been integrated into `develop`.

The source version metadata on `master` is `0.6.0` and the `CHANGELOG.md`
release date is `2026-11-01`. The annotated `0.6.0` tag points to commit
`8cc7d2e`, and the GitHub Release was published on 2026-11-02 UTC.
Publication through PyPI Trusted Publishing was triggered separately.

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
- [x] V7 PR: Python 3.11, 3.12, 3.13, 3.14 test matrix green.
- [x] V7 PR: warnings-as-errors Sphinx build green.
- [x] V7 PR: `python -m build` produces wheel and sdist;
  `python -m twine check dist/*` validates metadata.
- [x] V7 PR: clean wheel and sdist installations both pass runtime
  import and Three.js resource smoke tests **outside** the source tree.
- [x] Review remaining version-sensitive files and release
  documentation after the final CI run.
- [x] Release candidate merged into `master` following explicit approval.
- [x] Register the `publish.yml` Trusted Publisher on the existing PyPI
  project and configure GitHub environment `pypi`.
- [x] Set the release date in the changelog before tagging.
- [x] After publisher configuration, create an annotated tag `0.6.0`
  on the final merged release commit, then explicitly publish the
  GitHub Release to trigger the Trusted Publishing workflow.
- [ ] Confirm PyPI package metadata and installation by version.
- [x] Synchronize `develop` from `master` after PR #30 integration.
- [x] Confirm the GitHub Actions build and PyPI publishing jobs succeed.

## Release verification

GitHub Actions `Tests` includes the complete supported Python
matrix and a distribution job that builds the wheel and sdist, checks
metadata using Twine, installs each archive in a fresh virtual
environment and verifies default Three.js plotting/animation and
packaged HTML/JS resources.

`Build and deploy documentation` checks Sphinx with warnings treated
as errors. Its deployment step is skipped on PRs; site deployment only
occurs when changes reach `master`.

The [0.6.0 GitHub Release](https://github.com/JorgeDeLosSantos/moro/releases/tag/0.6.0)
and the [publishing workflow](https://github.com/JorgeDeLosSantos/moro/actions/runs/41395593575)
confirm successful package build and publication to PyPI. Independent
installation verification from PyPI itself remains unchecked.

## Not in scope

No new kinematics, simulation or rendering capabilities; no refactor of
historical `moro.plotting` functions; no bundling of CDN assets; no
automatic switching of backend for terminal environments. Publishing
is deliberately triggered by a maintainer's GitHub Release action,
not automatically by merging or tagging alone.

The release should preserve the user-validated interactive behavior.

## Automated publishing extension

The independent `publish.yml` workflow successfully published 0.6.0
using PyPI Trusted Publishing and the `pypi` GitHub environment.
This first live stable-release publication also validated the OIDC
publisher handshake beyond pull-request CI. See
[Publishing releases to PyPI](../../source/development/publishing.md)
for the reusable procedure, including unprefixed tags such as `0.6.0`.
