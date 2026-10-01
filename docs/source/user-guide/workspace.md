# Workspace Sampling

Moro 0.5.0 provides numerical sampling of the reachable end-effector position workspace of a serial manipulator.

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

Degenerate intervals are intentionally rejected in 0.5.0.

Default robot limits are modeling conveniences and may not represent the physical range of a particular mechanism, especially for prismatic joints.

## Sampling distribution

Each joint is sampled independently and uniformly over its effective interval:

$$
q_i\sim\mathcal U(q_{i,\min},q_{i,\max}).
$$

This means sampling is uniform in joint space, not in Cartesian space.

A dense Cartesian region does not imply a larger geometric portion of the true workspace; the forward-kinematics mapping can distort the point density strongly.

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

## Visualization

Use:

```python
from moro.visualization import plot_workspace

fig, ax = plot_workspace(workspace)
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
fig, ax = plot_workspace(
    workspace,
    projection="xy",
    marker_size=6,
    alpha=0.4,
)
```

Workspace visualization is Matplotlib-only in 0.5.0.

## What workspace sampling does not compute

The 0.5.0 feature does not attempt to compute:

- exact boundaries;
- convex hulls;
- areas or volumes;
- dextrous or orientation workspace;
- manipulability maps;
- singularity maps;
- collision-aware workspace;
- adaptive or quasi-random sampling.

The stored configurations make later post-processing possible without expanding the scope of the sampler itself.
