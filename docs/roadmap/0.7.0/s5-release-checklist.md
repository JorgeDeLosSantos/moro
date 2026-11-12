# S5 — Moro 0.7.0 release-candidate checklist

> **Status:** **Completed.** Moro 0.7.0 was published on 2026-11-11
> (Mexico City local date); the stable GitHub Release, distribution build,
> and PyPI Trusted Publishing workflow all succeeded.
> This checklist is retained as a historical release record.

## Candidate metadata

- [x] Version set to `0.7.0` in `pyproject.toml`.
- [x] Version set to `0.7.0` in `moro/version.py`.
- [x] `CHANGELOG.md` contains a distinct `0.7.0` entry; **the date is recorded in `CHANGELOG.md`**.
- [x] Release notes include breaking changes, migration paths, new
      capabilities and explicitly deferred features.
- [x] Existing PyPI Trusted Publishing workflow and the unprefixed
      stable-tag convention are retained.
- [x] Wheel and sdist CI smoke tests now exercise the 0.7.0 workspace
      template and installed waypoint, section and playback APIs.

## Release-candidate validation gates

- [x] The S5 PR CI passes on Python 3.11–3.14.
- [x] Documentation build passes on the candidate.
- [x] `python -m build` and `python -m twine check dist/*` pass.
- [x] Wheel **and** sdist independently install in clean environments,
      import `moro`, and report `0.7.0`.
- [x] Both installed distributions include the Three.js viewer,
      animation, workspace and common JavaScript resources.
- [x] Installed distribution smoke tests exercise trajectory,
      physical-time animation, workspace and section viewers.
- [x] Final review finds no version inconsistencies or regressions.
- [x] The changelog and release-note publication wording is finalized
      after the release date is decided.

## Maintainer publication sequence (completed for 0.7.0)

1. Review the candidate diff, CI and packaging results.
2. Merge the accepted candidate into `develop`, then prepare the
   reviewed `master` merge according to project policy.
3. Verify `master` contains the release candidate and CI passes.
4. Use an unprefixed **annotated tag** `0.7.0` on the approved
   `master` commit (never on an unreviewed development commit).
5. Publish the stable **GitHub Release** based on that tag. This is
   the action that triggers the `Publish to PyPI` workflow, not
   merely pushing a tag.
6. Confirm the Trusted Publishing job completes successfully,
   then check PyPI and fresh `pip install "moro==0.7.0"`.
7. Synchronize `develop` with the released `master`.

Follow the authoritative maintainer runbook
[`.github/RELEASING.md`](../../../.github/RELEASING.md)
for detailed tag and PyPI instructions.

**Scope constraint:** This release introduces no T-05 general
kinematic-limit feasibility engine, friction/external-load APIs or
new algorithms beyond the frozen S1 scope.
