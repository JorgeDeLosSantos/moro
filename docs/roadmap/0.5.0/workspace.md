# Workspace

## Status

Accepted as a candidate feature for Moro 0.5.0.

## Objective

Approximate the sampled reachable end-effector workspace under finite joint limits:

\[
\mathcal W=\{p(q)\mid q\in\mathcal Q\}.
\]

No general analytical workspace solution or exact boundary is planned.

The implementation should live in `moro/workspace.py`.

## Sampling strategy

Initial 0.5.0 support is uniform random sampling in joint space only. A `seed` argument provides reproducibility. Grid/adaptive sampling is deferred because of scaling and scope.

Revolute, prismatic, and mixed-joint robots should be supported.

## Preliminary public API

```python
sample_workspace(
    robot,
    *,
    samples=1000,
    joint_limits=None,
    parameters=None,
    seed=None,
)
```

By default, use limits defined by the robot model. Explicit finite overrides are allowed.

## Finite joint limits

Finite intervals are required. If any joint remains unbounded and no finite override is supplied, sampling must fail with a clear validation error. The workspace is always relative to a specified finite joint-space domain.

## `Workspace`

The result should represent sampled workspace data rather than a solver outcome and expose at least:

- `points`;
- `configurations`;
- `joint_limits`;
- `samples`;
- `bounds`.

`points` are always stored as `(N, 3)`, including planar robots. `configurations` retain the corresponding joint samples, enabling later manipulability/singularity analysis without resampling.

`bounds` may expose axis-aligned Cartesian limits such as

```python
((xmin, xmax), (ymin, ymax), (zmin, zmax))
```

without implying an exact workspace boundary.

## Symbolic and numerical behavior

Robot geometry may contain symbolic parameters resolved through `parameters`. Remaining unresolved symbols must cause a clear failure. Repeated `.subs()` per sample should be avoided; a compiled/lambdified numerical FK callable is the expected performance direction.

## Visualization

Workspace visualization belongs to the visualization subsystem, not as `.plot()` on `Workspace`. The exact entry point is deferred. Planar workspaces may be displayed in 2D while storage remains 3D.

## Examples and tests

Cover planar and spatial robots, revolute/prismatic joints, explicit/model-defined limits, symbolic parameters, deterministic sampling via `seed`, point/configuration consistency, shapes, and visualization.

## Explicitly outside 0.5.0

- analytical workspace computation;
- exact boundary reconstruction;
- grid/adaptive sampling;
- area/volume estimation;
- convex hull as workspace representation;
- dextrous/orientation/pose/constant-orientation workspace;
- obstacles/collision checking;
- automatic manipulability or singularity maps;
- arbitrary intermediate-link workspace sampling.

## Deferred detailed-design decisions

Exact dataclass invariants, validation/messages, precedence of explicit versus robot limits, numerical array types beyond agreed shapes, sampling helpers, numerical evaluation/caching, visualization API, and planar-display thresholds remain deferred.
