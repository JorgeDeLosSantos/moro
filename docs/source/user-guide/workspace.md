# Workspace Sampling

Moro provides numerical sampling of the reachable end-effector position workspace of a serial manipulator.

The feature approximates

$$
\mathcal W = \{p(q)\mid q\in\mathcal Q\}
$$

by drawing joint configurations from a finite joint-space domain and evaluating forward kinematics.

The result is a sampled point cloud, not an exact workspace boundary.

## Basic usage

```python
from moro.workspace import sample_workspace

workspace = sample_workspace(
    robot,
    samples=5000,
    seed=1,
)
```

The result contains:

```python
workspace.points
workspace.configurations
workspace.joint_limits
workspace.seed
workspace.samples
workspace.dof
workspace.bounds
```

The public array shapes are:

```text
workspace.points.shape == (N, 3)
workspace.configurations.shape == (N, robot.dof)
```

Planar robots still use three Cartesian coordinates.

## Joint-space domain

By default, sampling uses:

```python
robot.joint_limits
```

You may override them for a single call:

```python
workspace = sample_workspace(
    robot,
    samples=2000,
    joint_limits=[
        (-1.0, 1.0),
        (-0.5, 0.5),
    ],
)
```

Explicit limits completely replace the robot limits for that sampling call and do not mutate the robot.

Every interval must be finite and strictly ordered:

$$
q_{i,\min}<q_{i,\max}.
$$

Degenerate intervals are rejected.

Default robot limits are modeling conveniences and may not represent the physical range of a particular mechanism, especially for prismatic joints.

## Sampling distribution

With the default `method='random'`, each joint is sampled independently and uniformly over its effective interval:

$$
q_i\sim\mathcal U(q_{i,\min},q_{i,\max}).
$$

This means sampling is uniform in joint space, not in Cartesian space.

A dense Cartesian region does not imply a larger geometric portion of the true workspace; the forward-kinematics mapping can distort the point density strongly.

## Alternative joint-space sampling methods (0.7.0 development)

The default remains independent uniform pseudo-random sampling with an effective
count of 1,000 when `samples` is omitted. A specific method can be selected:

```python
sobol_workspace = sample_workspace(
    robot, samples=1024, method="sobol", seed=42,
)
lhs_workspace = sample_workspace(
    robot, samples=1000, method="latin_hypercube", seed=42,
)
grid_workspace = sample_workspace(
    robot, method="grid", grid_shape=(40, 50),
)
```

`random` is the original NumPy sampler and retains its seeded sequence.
`sobol` uses scrambled SciPy Sobol points. Powers of two (e.g., 1024)
preserve the usual Sobol balance property; other positive counts take a
prefix of the next power-of-two block, and should not be interpreted as
having the same balance guarantees. `latin_hypercube` uses scrambled
SciPy Latin-hypercube sampling. All three use `samples` as the **total**
number of configurations; their coverage is defined in joint space, not
Cartesian space.

For `grid`, `grid_shape` gives **one count per joint** (with product
equal to the total number of configurations). With 2 or more divisions,
both endpoints of the joint range are included; a single division uses
the interval midpoint. The grid is deterministic and requires
`seed=None`; omit `samples` instead of supplying a conflicting count.
The last joint varies fastest in row order. A safety limit of 1,000,000
samples protects against accidental combinatorial explosion.

The resulting dataset includes optional descriptive metadata
`workspace.method` and `workspace.grid_shape`; directly constructed
`Workspace` instances may leave these unspecified.

## Reproducibility

Sampling uses a local NumPy random generator:

```python
workspace = sample_workspace(
    robot,
    samples=1000,
    seed=42,
)
```

The same model, parameters, limits, sample count, and seed reproduce the same sampled configurations within the same relevant numerical environment.

The global NumPy random state is not used.

## Symbolic geometric parameters

Workspace sampling can evaluate robots with symbolic geometric parameters:

```python
workspace = sample_workspace(
    robot,
    samples=1000,
    parameters={
        l1: 1.0,
        l2: 0.8,
    },
    seed=3,
)
```

Parameter substitution is applied to a local forward-kinematics expression. The symbolic robot model is not modified.

Any unresolved model parameter causes a clear validation error before sampling begins.

## Point/configuration correspondence

Rows correspond one-to-one:

$$
workspace.points[k]
=
p(workspace.configurations[k]).
$$

Samples are never reordered, deduplicated, or silently removed.

If one forward-kinematics evaluation produces invalid numerical data, the entire sampling call fails instead of returning a smaller dataset.

## Cartesian sample bounds

The property:

```python
workspace.bounds
```

returns the observed sample bounds:

```python
(
    (xmin, xmax),
    (ymin, ymax),
    (zmin, zmax),
)
```

These are point-cloud bounds only. They are not an exact analytical workspace boundary.

## Cartesian plane sections (0.7.0 development)

A projection keeps every sample and discards one plotted coordinate. A **section**
instead selects only points sufficiently close to a specified Cartesian plane.
The numerical operation is independent of the robot model and all plotting backends:

```python
from moro.workspace import section_workspace

section = section_workspace(
    workspace,
    plane="xy",
    level=0.3,
    thickness=0.02,
)
```

Here the selected points satisfy `abs(z - 0.3) <= 0.01`. Other options
are `plane="xz"` (filter Y) and `plane="yz"` (filter X).
The full `thickness` is nonnegative and expressed in the same linear units
as the workspace; a value of zero matches only exactly equal coordinates.

The returned `WorkspaceSection` contains `plane`, `level`,
`thickness`, `indices`, `points`, `configurations`,
`source_samples`, `samples`, and `dof`. Its points retain their
**original 3D coordinates**, with exact row correspondence to configurations.
`indices` are the original zero-based row positions in the source workspace,
in ascending order. A section containing no samples is **valid** and has
point shape `(0, 3)` and configuration shape `(0, dof)`.

A section never reconstructs an exact reachable-set intersection,
interpolates missing data, increases its thickness automatically, or
reevaluates forward kinematics. It is simply an observed point-cloud slab.
Visualize the filtered point cloud with the default Three.js viewer:

```python
from moro.visualization import plot_workspace_section

viewer = plot_workspace_section(section)
viewer_with_grid = plot_workspace_section(section, show_grid=True)
```

The section is shown in the selected XY/XZ/YZ plane at its **nominal
level** (e.g. Z = 0.3 for XY), while the original 3D points remain
unchanged. The viewer starts with an orthographic camera perpendicular
to the plane. Empty sections render a valid reference frame and a
"No points in section" status.

For Matplotlib figures and axes, select its backend explicitly:

```python
fig, ax = plot_workspace_section(
    section, backend="matplotlib", marker_size=6, alpha=0.5,
)
```

Both backends accept `show_grid=False` by default. When enabled,
the Three.js grid lies in the selected projection plane (XY for a
full 3D workspace), and Matplotlib enables its axis grid. The grid
is visual guidance only; it does not modify sampling, interpolation or
the physical interpretation of the section.

No viewer resamples, interpolates, recovers a workspace frontier or
changes any `WorkspaceSection` arrays.

## Visualization

Use the default **Three.js** interactive point-cloud viewer:

```python
from moro.visualization import plot_workspace

viewer = plot_workspace(workspace)
```

The result is `IPython.display.HTML`, as for `RobotVisualizer`. The
viewer provides matching Front / Right / Top / Isometric camera presets,
orthographic / perspective cameras, orbit/pan/zoom controls and Fit View.
Point clouds are rendered using one Three.js position buffer, not a
separate object per sample.

For static Matplotlib plotting and access to ordinary figure axes, select
the backend explicitly:

```python
fig, ax = plot_workspace(workspace, backend="matplotlib")
```

Supported projections are:

```text
auto
xy
xz
yz
3d
```

`projection="auto"` selects a base-axis-aligned 2D projection only when one Cartesian coordinate is approximately constant. Otherwise it uses a 3D view.

For example:

```python
viewer = plot_workspace(
    workspace,
    projection="xy",
    marker_size=6,
    alpha=0.4,
)
```

The Three.js viewer honors the requested projection by flattening only
the displayed normal coordinate: XY begins with Top view, XZ with Front
view and YZ with Right view. Original `workspace.points` remain
unchanged. `projection="3d"` preserves the full cloud and begins in
Isometric view.

**Moro 0.7.0 migration:** `plot_workspace(workspace)` changes from
Matplotlib's `(fig, ax)` result to HTML. Old notebooks that unpack
figures must use `backend="matplotlib"`. With Three.js,
`width`/`height` specify viewer dimensions in pixels; Matplotlib-only
`ax` and `figsize` are rejected rather than silently ignored. The
inverse applies to Matplotlib: Three.js-only dimension arguments are
rejected. `marker_size` denotes Matplotlib scatter area in point-squared
units, but fixed pixel size in Three.js; visual appearance therefore
differs between backends.

`plot_workspace_section(section)` also defaults to Three.js. For
Matplotlib axes, use `plot_workspace_section(section, backend="matplotlib")`.
Both plotting functions accept `show_grid=False` (the default), with
`show_grid=True` enabling an optional visual grid.

## What workspace sampling does not compute

Workspace sampling does not attempt to compute:

- exact boundaries;
- convex hulls;
- areas or volumes;
- dextrous or orientation workspace;
- manipulability maps;
- singularity maps;
- collision-aware workspace;
- adaptive sampling or guaranteed uniform Cartesian coverage.

The stored configurations make later post-processing possible without expanding the scope of the sampler itself.
