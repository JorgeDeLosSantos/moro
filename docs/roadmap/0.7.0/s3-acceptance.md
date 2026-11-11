# S3 — Moro 0.7.0 integration and visual acceptance

> **Status:** CI acceptance in progress; interactive checks require a human in
> Jupyter/Colab and are **not** proved by Python/Node unit tests.

The S3 integration tests cover joint multi-waypoint motion, irregular
physical timestamps, Three.js animation payloads, ideal inverse dynamics
and forward-dynamics equivalence, plus sampled workspace sections in
both rendering backends. They do **not** claim live browser/WebGL coverage.

## Automated checks

Run on the S3 branch before the release candidate:

```bash
python -m pytest tests/test_s3_integration_acceptance.py -q
python -m pytest -q
python -m build
```

GitHub Actions additionally runs Python 3.11–3.14, distribution
install checks and the Sphinx documentation build. The release
candidate must repeat those checks after the stabilization changes.

## Manual Jupyter/Colab acceptance

Use the current `develop` branch **after the S3 PR is integrated**,
or the current S3 branch while reviewing it:

```python
import numpy as np
from IPython.display import display
from moro import Robot
from moro.abc import q1, q2
from moro.workspace import sample_workspace, section_workspace
from moro.visualization import plot_workspace, plot_workspace_section

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.0, 0, q2, 0, "p"),
)
robot.joint_limits = [(-np.pi, np.pi), (0.0, 0.5)]
workspace = sample_workspace(
    robot, method="grid", grid_shape=(60, 11),
)
section = section_workspace(
    workspace, plane="xy", level=0.25, thickness=0.0,
)
empty = section_workspace(
    workspace, plane="yz", level=4.0, thickness=0.0,
)

display(plot_workspace(workspace, projection="3d", show_grid=True))
display(plot_workspace_section(section, show_grid=True))
display(plot_workspace_section(section, show_grid=False))
display(plot_workspace_section(empty, show_grid=True))
```

Confirm each item explicitly:

- [ ] 3D cloud grid appears in XY and does not distort the cylinder.
- [ ] XY section grid is parallel to the section, at the selected Z level.
- [ ] Section cloud begins in orthographic Top view and retains circular proportions.
- [ ] `show_grid=False` hides the grid, while keeping axes and points.
- [ ] Empty section gives a readable no-points status and a valid camera,
      including Fit View and optional grid.
- [ ] YZ and XZ sections are correctly oriented when selected; the camera
      follows the plane (Right / Front).
- [ ] Rotate, pan, zoom, change camera mode and Fit View without console
      errors; check both Jupyter and Colab where available.
- [ ] Multi-waypoint physical-time animation still plays/pause/seeks
      correctly and remains independent of other viewers in the notebook.

**Recording policy:** these boxes must only be checked after actual
human inspection. A green GitHub CI run does not establish their status.
File issues and fix regressions before marking S3 complete.

## Exit criteria

- Cross-module tests and documentation build are green.
- Wheel and sdist install smoke checks complete.
- Manual visual checks are confirmed or explicitly recorded as release
  blockers (never silently marked passed).
- No scope expansion to T-05, friction/external wrenches, or new
  workspace algorithms.
