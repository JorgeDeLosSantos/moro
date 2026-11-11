# Cartesian Workspace Sections

This example uses a mixed revolute–prismatic (RP) robot to compare a
sampled 3D position workspace with an approximate Cartesian section.
A section filters existing samples; it does not compute an exact
intersection of the reachable set with a mathematical plane.

## 1. Define the robot and sample its workspace

```python
import numpy as np
import matplotlib.pyplot as plt

from moro import Robot
from moro.abc import q1, q2
from moro.workspace import sample_workspace, section_workspace
from moro.visualization import plot_workspace, plot_workspace_section

robot = Robot(
    (1.0, 0, 0, q1, "r"),
    (0.0, 0, q2, 0, "p"),
)
robot.joint_limits = [
    (-np.pi, np.pi),
    (0.0, 0.5),
]

workspace = sample_workspace(
    robot, method="grid", grid_shape=(80, 21),
)
assert workspace.samples == 1680
```

The grid ensures known joint-space coverage. It remains a discrete
point cloud, not a continuous filled region.

## 2. Plot the full workspace

```python
fig, ax = plot_workspace(workspace, projection="3d", backend="matplotlib")
plt.show()
```

Here Matplotlib is selected explicitly to compare with the 2D section figures. For the interactive Three.js viewer (the default in 0.7.0), use `plot_workspace(workspace, projection="3d")` instead.

The full workspace is a 3D cylinder-like point cloud sampled at
finite positions of both joints.

## 3. Filter the XY section at Z = 0.25

```python
section = section_workspace(
    workspace,
    plane="xy",
    level=0.25,
    thickness=0.0,
)

print("Samples:", section.samples)
print("Original indices:", section.indices[:8])
assert section.samples == 80

fig, ax = plot_workspace_section(section, backend="matplotlib", marker_size=12, alpha=0.7)
ax.set_title("Sampled XY section at Z = 0.25")
plt.show()
```

The result contains original 3D positions and corresponding robot
configurations. Only the plotted coordinates are projected onto XY.
The same original zero-based row order is retained.

## 4. Compare with a sparse or empty section

```python
empty = section_workspace(
    workspace, plane="xy", level=0.251, thickness=0.0,
)
assert empty.samples == 0

fig, ax = plot_workspace_section(empty, backend="matplotlib")
ax.set_title("No points selected")
plt.show()
```

The empty section is a valid result. To capture more discrete samples,
choose an *explicitly* larger thickness. No automatic resampling,
interpolation or thickness expansion occurs.

## 5. Change the section plane

```python
yz_section = section_workspace(
    workspace, plane="yz", level=1.0, thickness=0.06,
)

fig, ax = plot_workspace_section(yz_section, backend="matplotlib")
ax.set_title("Approximate YZ section near X = 1")
plt.show()
```

## 6. Interactive sections and an optional grid

Three.js is now the default for both workspace and section plots:

```python
from IPython.display import display

display(plot_workspace_section(section))
display(plot_workspace_section(yz_section, show_grid=True))
display(plot_workspace_section(empty))
```

The first viewer starts in an orthographic view perpendicular to the
selected plane, preserves the nominal section level and displays only
the points selected by `section_workspace`. The optional grid follows
that plane. For an empty section, the viewer shows a valid reference
frame and a clear no-points status. The source positions, configurations
and original row indices remain unchanged. To obtain ordinary Matplotlib
axes, use `backend="matplotlib"` explicitly.

The section is approximate because it includes original points within
a slab of the selected thickness; it is not a boundary calculation.
