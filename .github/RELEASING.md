# Releasing Moro to PyPI

Maintainer runbook for configuring and publishing official Moro releases.
This procedure is not required for ordinary contributions or library usage.

Moro uses GitHub Actions **Trusted Publishing (OIDC)** to upload wheel and
source distributions to the existing [moro PyPI project](https://pypi.org/project/moro/).
No PyPI API token or password is stored in GitHub.

Publication is **not** triggered by merging a pull request or pushing a
Git tag. A maintainer must explicitly **publish a stable GitHub Release**.
The workflow is defined in [`.github/workflows/publish.yml`](workflows/publish.yml).

## One-time configuration

### 1. Configure the GitHub environment

In [the GitHub repository](https://github.com/JorgeDeLosSantos/moro),
navigate to **Settings → Environments → New environment** and create
an environment named exactly `pypi`.

Where supported, configure:

- required reviewers so publication needs a manual approval;
- deployment branch/tag restrictions permitting release tags that follow
  the unprefixed version convention, e.g. `0.6.0`.

Reviewers and deployment protections are repository settings, **not**
defined by this Markdown file. Ensure that the environment is configured
before creating a release.

### 2. Register the Trusted Publisher on PyPI

Sign in with the maintainer account, open the existing project settings
at <https://pypi.org/manage/project/moro/settings/publishing/> and add a
**GitHub Actions** trusted publisher using the **exact** values:

| Field | Value |
| --- | --- |
| PyPI project | `moro` (existing project) |
| GitHub owner | `JorgeDeLosSantos` |
| GitHub repository | `moro` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

These values must match the GitHub workflow and environment. This
registration belongs to the existing `moro` project, not the page
for creating a new project. No GitHub `PYPI_TOKEN` secret is necessary.

See the [official PyPI instructions for existing projects](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [using Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Release procedure

1. Prepare the release candidate with matching versions in
   `pyproject.toml` and `moro/version.py`, an updated `CHANGELOG.md`,
   and the release notes. Run the CI suite, documentation build, and
   wheel and sdist smoke tests. Merge the reviewed candidate into `master`.
2. Confirm that the released version and notes are committed on `master`
   and that the required checks passed **before** creating the tag.
3. Create an **unprefixed**, preferably annotated Git tag on the
   reviewed commit of `master`:

   ```bash
   git switch master
   git pull --ff-only origin master
   # Replace X.Y.Z with the version being published.
   git tag -a X.Y.Z -m "Moro X.Y.Z"
   git push origin X.Y.Z
   ```

4. On GitHub, go to **Releases → Draft a new release**. Select the existing
   tag `X.Y.Z` and use the title `Moro X.Y.Z`. Review the release notes.
5. Click **Publish release** (not `Save draft`). This triggers the
   `Publish to PyPI` workflow. A prerelease is intentionally skipped.
6. If the `pypi` environment requires approval, approve the deployment.
   The publishing job uses OIDC short-lived credentials; no token is
   entered by the maintainer.
7. Check GitHub Actions and confirm that the project page shows the new
   version on <https://pypi.org/project/moro/>. Verify installation in
   a new environment:

   ```bash
   python -m pip install "moro==X.Y.Z"
   python -c "import moro; print(moro.__version__)"
   ```

8. Synchronize `develop` with the released `master` after publication.

Replace `X.Y.Z` with the intended release version throughout this
procedure; the commands are examples, not literal version identifiers.

## Safeguards and limitations

- Pushing a tag alone does not upload anything to PyPI; publishing a
  **non-prerelease GitHub Release** is the explicit trigger.
- The build job checks that the tag equals the package version in both
  version files, uses the `MAJOR.MINOR.PATCH` convention without `v`,
  and verifies the tagged commit belongs to the history of `master`.
- The workflow builds and validates both wheel and sdist in an
  unprivileged job, then transfers the validated artifacts to a separate
  publish job. Only the publish job receives `id-token: write`.
- The GitHub environment and PyPI publisher registration must exist for
  uploading to succeed. The workflow's presence alone does not set up
  those account-level settings.
- PyPI versions are immutable; do not try to replace the files of an
  already published version. If a correction is needed, release a new
  version (for example `0.6.1`).
- An end-to-end OIDC upload cannot be validated by pull-request CI alone.
  Moro 0.6.0 successfully exercised the live publishing workflow;
  subsequent releases must still be checked in GitHub Actions and PyPI.

Historical tags may contain a `v` prefix. From `0.5.0` onward, Moro
uses **unprefixed release tags** (`0.5.0`, `0.6.0`, and so on). Do not
rewrite historical tags.
