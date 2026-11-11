# Moro 0.7.0 — API-003/D1: Workspace sampling and data contracts

> **Status:** Design proposal for review; not an accepted public API or a code change.
> **Baseline:** `develop` after API-002/B3 (PR #57).
> **Tracks:** W-03 sampling, W-02 Cartesian sections; W-01 rendering compatibility only.

## 1. Objective and boundaries

Extend the existing sampled position workspace without turning `Workspace` into a solver, a graphical object, or an exact reachable-set description.

This proposal deliberately preserves:

- `moro.workspace` for numerical evaluation and data; `moro.visualization.workspace` for plotting.
- `Workspace.points.shape == (N, 3)` and `Workspace.configurations.shape == (N, dof)`.
- The same ordering and one-to-one relationship between configuration rows and FK points.
- Finite, strictly ordered joint limits and model-parameter substitutions without mutating `Robot`.
- Compilation of numerical position FK **once per sampling call**, not once per point.
- No point filtering, deduplication, reordering or silent failed-FK sample removal.
- Current `sample_workspace(robot, samples=..., seed=...)` behavior and its seeded random sequences.
- Existing Matplotlib `plot_workspace()` functionality through an **explicit** `backend='matplotlib'` selection; the implicit backend and return type are intentionally allowed to change in 0.7.0.

A sampled cloud is **not** the analytical workspace boundary, and its point density is not a uniform Cartesian-space measure.

## 2. Current implementation

`moro/workspace.py` exposes `Workspace` and `sample_workspace()`. The latter validates `samples`, `seed`, complete effective joint limits and model parameters, prepares `lambdify` for `robot.T[:3, 3]`, generates independent uniform joint samples with a local `numpy.random.default_rng`, then evaluates FK per row.

`Workspace` stores independent real arrays, normalized limits and a seed; it derives `samples`, `dof`, and Cartesian sample `bounds`. Direct construction is supported.

`moro/visualization/workspace.py` supplies a Matplotlib-only `plot_workspace()`; `projection='auto'` may choose a base-axis-aligned XY/XZ/YZ projection, otherwise 3D. A projection retains *all* points; it is not a plane section.

## 3. Proposed sampling API

```python
sample_workspace(
    robot,
    *,
    samples=None,
    joint_limits=None,
    parameters=None,
    seed=None,
    method="random",
    grid_shape=None,
)
```

The current *effective* default remains **1,000 random samples**. The proposal changes the Python signature default `samples=1000` to `samples=None` solely to distinguish omitted sample count from an explicit count when `method='grid'`. Existing calls and results for random sampling must remain unchanged. Review this signature adjustment before implementation.

Supported methods are exactly `"random"`, `"sobol"`, `"latin_hypercube"`, and `"grid"` (case normalization to be documented; invalid types/values fail with `ValueError`).

| Method | Required cardinality argument | Number of returned rows | Seed |
| --- | --- | --- | --- |
| random | `samples` (default 1000) | exactly `samples` | optional, as today |
| sobol | `samples` (default 1000) | exactly `samples` | optional scrambled sequence |
| latin_hypercube | `samples` (default 1000) | exactly `samples` | optional scrambled sequence |
| grid | `grid_shape=(n1, ..., ndof)` | exactly product of `grid_shape` | must be `None` |

For non-grid methods `grid_shape` is forbidden. For grid `samples` must be omitted/`None`, and `grid_shape` is required. Never silently ignore contradictory arguments.

**Why explicit `grid_shape`?** A tensor-product grid cannot generally contain exactly an arbitrary `N` while maintaining comparable axis resolution. Do not reinterpret `samples=N` as `N` *per joint* (which creates `N**dof` samples), choose axis counts secretly, or truncate an asymmetric grid.

For grid, each axis count is a positive non-Boolean integer. Counts >=2 include both joint limits with `linspace`; a count of 1 samples that joint's midpoint. Use a deterministic Cartesian-product order (e.g. `meshgrid(indexing='ij')`, last joint changing fastest), documented and tested. Calculate total rows and expected array size before allocating; agree a reasonable safety policy as part of W-03 implementation rather than silently generating enormous arrays.

**Algorithm semantics:**
- `random`: retain the existing local NumPy generator and identical sampling flow to preserve seeded result compatibility.
- `sobol`: `scipy.stats.qmc.Sobol`, scrambled with optional `seed`, scaled from unit cube to joint limits. For arbitrary positive N, generate the next power-of-two block with `random_base2` and keep the first N rows; explain that full Sobol balance guarantees apply to power-of-two sample counts.
- `latin_hypercube`: `scipy.stats.qmc.LatinHypercube`, scrambled with optional `seed`, scaled to limits.
- `grid`: deterministic tensor product; no stochastic state.

`scipy` is already a project dependency, so no new package is necessary. Reproducibility is guaranteed only for compatible numerical library environments, not across arbitrary future dependency versions.

All methods generate `(N, dof)` configurations **first**, then reuse the current single FK preparation and per-row evaluation/error diagnostics.

## 4. Workspace data and provenance

Keep the existing positional constructor contract:

```python
Workspace(points, configurations, joint_limits, seed=None)
```

Propose **optional appended metadata** `method: str | None = None` and `grid_shape: tuple[int, ...] | None = None` (review field naming). `sample_workspace()` supplies these attributes. Directly constructed datasets may leave them unspecified; a direct constructor cannot prove how the arrays were generated.

Do not change the meaning of `samples`, `dof`, `bounds`, `points`, `configurations`, `joint_limits`, or existing `seed` and `repr`. In particular, sample `bounds` remain observed min/max, not exact reachability bounds. Retain input defensive copying.

Provenance fields are *descriptive*, not used by plotting or FK and not a reason to add a sampling class hierarchy.

## 5. Cartesian sections: numerical contract for W-02

A **projection** maps every 3D point to two coordinates, e.g. `(x, y, z) -> (x, y)`. A **section** filters points by proximity to a plane before projection.

Proposed independent numerical function:

```python
section_workspace(
    workspace,
    *,
    plane="xy",  # "xy", "xz", "yz"
    level,        # normal coordinate in the robot base frame
    thickness,    # full slab thickness, in the same linear units
)
```

Selection is inclusive:

```text
xy: |z - level| <= thickness / 2
xz: |y - level| <= thickness / 2
yz: |x - level| <= thickness / 2
```

`level` must be finite real. `thickness` must be finite and nonnegative. Zero thickness allows exact matching sampled coordinates but may return zero rows. No implicit tolerance increase or automatic resampling if a section is empty.

Return an independent `WorkspaceSection` dataset with:

- `plane`, `level`, `thickness`;
- `indices: (K,)`: original zero-based row indices, ascending;
- `points: (K, 3)`: original 3D Cartesian coordinates, unprojected;
- `configurations: (K, dof)`: corresponding original joint configurations;
- `source_samples` and a derived `samples == K` (optional convenience).

An empty section (`K=0`) is valid. **Do not** model a section as a `Workspace`, whose current invariant rejects zero samples. Select without reordering; retain duplicate points if distinct configurations map there. The section should not require a `Robot` or reevaluate FK. Selected data should be independent copies to avoid accidental mutation coupling.

The numerical selection API should not import Matplotlib, Three.js or IPython. Dedicated plotting behavior for `WorkspaceSection` can be designed in W-02 rendering without prematurely overloading `Workspace` semantics.

The slab is an *approximate filtered subset*, not the exact intersection of the continuous reachable set with a plane.

## 6. Three.js compatibility and visualization direction for W-01

**Approved W-01 direction:** In Moro 0.7.0, `plot_workspace(workspace)` will default to **Three.js**, consistent with `RobotVisualizer`. This is an **intentional breaking change** from the Matplotlib-only 0.6.x behavior; no automatic legacy fallback or deprecated alias is required. Matplotlib remains available by explicitly passing `backend='matplotlib'`. D1 contains no graphical implementation.

**Proposed public usage after W-01:**

```python
viewer = plot_workspace(workspace)                       # Three.js (default)
viewer = plot_workspace(workspace, backend="threejs")     # explicit Three.js
fig, ax = plot_workspace(workspace, backend="matplotlib") # old Matplotlib tuple
```

Backend-specific return types follow Moro's existing visualizer pattern: Three.js returns `IPython.display.HTML`, while Matplotlib returns `(fig, ax)`. Calls that previously used `fig, ax = plot_workspace(workspace)` **must migrate** by supplying `backend='matplotlib'`. Document this prominently in the 0.7.0 migration notes, examples and API reference.

Keep common numerical `Workspace` and `WorkspaceSection` data free of viewer state. Three.js should use a point-cloud buffer rather than creating one mesh per sample, preserve geometric aspect, and share camera convention and Fit View behavior with robot visualization. Do not treat low-discrepancy point clouds as exact boundaries or shaded surfaces.

Explicit 2D projections (`xy`, `xz`, `yz`) must remain understandable with both backends. The precise Three.js behavior (orthographic camera orientation vs planar rendering), `projection='auto'` behavior and viewer-specific options will be defined and tested during W-01 implementation. Matplotlib-only options such as `ax` and `figsize` must not be silently ignored by Three.js; specify backend-sensitive validation. Do not overload D1 with final visual styling choices.

## 7. Validation and acceptance plan

Before merging W-03 numerical changes, test:

1. Old random calls (including omitted `samples` and seeded cases) preserve prior results and exact shapes.
2. All four methods respect finite joint limits for R, P, mixed and symbolic-geometry robots.
3. Repeatability for each seeded stochastic method, independence from global NumPy RNG and no model mutation.
4. For a set of sampled rows, FK of `configurations[k]` matches `points[k]`.
5. Exact row count; small N, N=1, Sobol power-of-two and non-power-of-two.
6. Grid shape, deterministic order, endpoints/midpoint and high-DOF explosion protection.
7. Invalid method, shape, sample/seed combinations and non-finite evaluations fail early and meaningfully.
8. Existing `Workspace` constructor tests remain valid; existing Matplotlib plotting tests are migrated to explicit `backend='matplotlib'` in the W-01 implementation.

Before merging W-02 numerical section changes, test:

1. Inclusive XY/XZ/YZ filters and slab boundary cases.
2. `indices`, 3D points and configurations retain exact original-row correspondence.
3. Empty results, thickness zero, single point, duplicates, and very sparse clouds.
4. Invalid plane, level, thickness; source data remains unmodified.
5. Section data does not assume a viewer or a robot instance.

## 8. Proposed PR-sized increments

- **D1 (this document):** design review only; no public behavior changes.
- **D2 (W-03):** add sampling methods/provenance plus numerical tests and guide updates.
- **D3 (W-02):** add section computation/result type plus numerical tests.
- **D4 (W-02 visualization):** Matplotlib section display and notebook example.
- **D5 (W-01):** Three.js point-cloud viewer; switch the public `plot_workspace` default to Three.js, preserve explicit Matplotlib support, add return-type and projection/parameter validation tests and 0.7.0 migration documentation; manual Jupyter/Colab visual validation separately.

No advanced adaptive sampling, manipulability mapping, convex/nonconvex frontier reconstruction, 3D area/volume estimates, exact plane sections, or robot–environment interaction in this scope.

## 9. Review points

The main design choices that should be reviewed before code:

1. Accept `samples=None` in the signature (effective 1000 remains unchanged) to make `grid_shape` explicit and unambiguous.
2. Keep optional `method` and `grid_shape` provenance on `Workspace` rather than a separate metadata object.
3. Return independent, possibly empty `WorkspaceSection` with original indices and 3D source points.
4. **Resolved:** switch `plot_workspace()` to Three.js by default in 0.7.0, consistent with `RobotVisualizer`; keep explicit Matplotlib with its `(fig, ax)` return. This deliberately breaks old implicit-backend calls. Remaining review concerns: exact Three.js projection semantics and backend-specific arguments.
