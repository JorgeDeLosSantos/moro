# Moro 0.6.0 — Visualization design

**Status:** V1–V5-B1 integrated into `develop`; V5-B2 schematic symbols in review; V5-C pending.

**Current integration branch:** `feat/0.6.0-joint-symbols` (from `develop`).

## Purpose and boundaries

Moro currently evaluates DH frame origins and draws a straight connection between successive origins. For spatial manipulators, this connection may not show the separate translations described by the classic Denavit–Hartenberg convention.

**V1** adds numerically evaluated DH translation segments to scene data, independent of any rendering backend. It does **not** introduce CAD geometry, change the robot model, infer the physical positions of actuators, or change what either backend draws. It must preserve the existing `RobotVisualizer` API, including its current default backend.

Later increments will address characteristic length, shared visualization layout, framing, rendering, and eventual adoption of Three.js as the default. Those decisions are *not* part of V1.

## 1. Geometry and notation

For DH row `i`, Moro uses the classic convention:

\[
T_i^{i-1} = R_z(\theta_i)\,T_z(d_i)\,T_x(a_i)\,R_x(\alpha_i).
\]

Let `F_i` be the evaluated frame `i`, `p_i` its origin in the base/world frame, `z_{i-1}` the positive Z direction of `F_{i-1}`, and `x_i` the positive X direction of `F_i`. The intermediate point is

\[
p_{d_i} = p_{i-1} + d_i z_{i-1}.
\]

The expected position identity is

\[
p_i = p_{d_i} + a_i x_i.
\]

The rotation `R_x(\alpha_i)` does not change the X direction, so using `F_i.x` for the `a_i` translation is consistent with the classic DH convention. All returned positions are expressed in the base/world frame, not in local joint coordinates.

The two evaluated segments for row `i` are, in **fixed order**:

1. `kind="d"`: from `p_{i-1}` to `p_{d_i}`;
2. `kind="a"`: from `p_{d_i}` to `p_i`.

The endpoints carry signed direction: negative DH parameters reverse the corresponding segment, while the Euclidean segment length is its absolute parameter value. The two segments do **not** imply a physical link or an actuator placement.

## 2. Numerical scene data contract

Place the following dataclasses in `moro/visualization/data.py`.

~~~python
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

@dataclass(slots=True)
class DHSegmentData:
    """One numerically evaluated DH translation contribution."""
    row_index: int
    kind: Literal["d", "a"]
    start: np.ndarray
    end: np.ndarray

@dataclass(slots=True)
class SceneData:
    """Numerically evaluated data for one robot configuration."""
    joints: list[np.ndarray]
    frames: list[FrameData]
    dimension: float
    dh_segments: list[DHSegmentData] = field(default_factory=list)
~~~

Field contracts:

| Field | Meaning |
|---|---|
| `row_index` | 1-based DH row index; in `[1, robot.dof]` |
| `kind` | One of `"d"` or `"a"` |
| `start` / `end` | Independent finite real NumPy arrays with shape `(3,)`, in world coordinates |
| `dh_segments` | Ordered flat list `[d1, a1, ..., dn, an]`; length `2 * robot.dof` for evaluated robots |

Keep both zero-length segments when a row has `d_i = 0` or `a_i = 0`. This preserves one-to-one correspondence with DH rows and simplifies later layout calculations. Renderers may choose to omit zero-length primitives when rendering is introduced in a later increment.

These data objects remain lightweight containers: they are not full geometric entities or validation frameworks. `evaluate_robot()` is responsible for producing values that meet the contract. A manually constructed `SceneData` remains valid without supplying `dh_segments`; an independently instantiated `DHSegmentData` is expected to satisfy the documented contract.

Make `DHSegmentData` importable from `moro.visualization` by exporting it in `moro/visualization/__init__.py`.

### Compatibility policy

- Preserve `SceneData(joints, frames, dimension)`, including positional construction and named construction.
- Keep `dimension` and its existing calculation unchanged in V1; its redesign belongs to V2.
- Do not rename or reinterpret `joints` now: the existing field stores DH frame origins (including base), not necessarily physical joint locations.
- Do not change `FrameData`, backend defaults, serialized Three.js payloads, camera logic, or rendered appearance in V1.
- Adding `dh_segments` must not break existing tests that build `SceneData` manually.

## 3. Evaluation algorithm and responsibilities

Continue to evaluate all frame transforms once through the existing `evaluate_robot(robot, num_vals)` path. After the `frames` list has been built, call the private helper:

~~~python
def _build_dh_segments(
    robot: Robot,
    num_vals: dict,
    frames: list[FrameData],
) -> list[DHSegmentData]:
    ...
~~~

For each `i = 1, ..., robot.dof`:

1. Read the original row `(a_i, alpha_i, d_i, theta_i)` from `robot.dh_parameters[i - 1]`.
2. Evaluate `a_i` and `d_i` to finite real scalars by applying `num_vals`. Use a consistent scalar conversion path for plain Python numbers, NumPy scalars, and SymPy numbers/expressions.
3. Take `p_prev = frames[i-1].position` and `z_prev = frames[i-1].z`.
4. Calculate `p_mid = p_prev + d_i * z_prev`.
5. Take `p_end = frames[i].position` and append `DHSegmentData(i, "d", p_prev, p_mid)` followed by `DHSegmentData(i, "a", p_mid, p_end)`.
6. Make independent float-array copies for every stored endpoint; do not expose views of `FrameData.T`.

Only the signed `d_i` scalar is needed to locate `p_mid`; `a_i` is still evaluated to require a well-defined finite DH translation and to support the mathematical invariant in tests. The final endpoint should be taken from the evaluated frame to avoid unnecessary independent transform evaluation.

**No extra symbolic matrix multiplication** is required. Do not re-evaluate the complete DH transformation inside the helper.

### Validation and errors

- Require finite real results for each evaluated `a_i` and `d_i`, not unresolved symbols, complex results, `nan`, or `inf`.
- Raise a contextual `ValueError` naming the DH row and parameter when scalar evaluation fails.
- Preserve the existing error behavior for invalid or nonnumerical full frame transforms unless a directly related V1 regression necessitates a focused correction.
- Do not add a global magnitude threshold or a fixed geometric minimum in V1.
- Do not apply rounding to force geometrically small values to zero. Approximate floating-point equality belongs in tests.

The internal helper may use a small private scalar-evaluation helper if it reduces duplication. Do not create a public validation or geometry subsystem for V1.

### Substitution details

The same mapping `num_vals` used to evaluate `robot.T_i0(i)` must be applied to DH scalars. Plain numeric constants should work without requiring an explicit `.subs()` method on the original value (e.g., use `sympy.sympify(value).subs(num_vals)`).

If a DH parameter cannot be resolved to a finite real number, the evaluation fails explicitly rather than silently omitting its geometric contribution.

## 4. Mathematical invariants and tests

Add focused tests in `tests/test_visualization.py` or in a dedicated V1 test module without rewriting unrelated visualization tests.

For each row, up to floating-point tolerance:

\[
p_{d_i} - p_{i-1} = d_i z_{i-1},\quad
p_i - p_{d_i} = a_i x_i.
\]

Also verify:

- Exactly two segments per DH row and stable `d`-then-`a` ordering;
- Indexes start at one and match the rows;
- Endpoint shape `(3,)`, finite values, world-coordinate interpretation, and no shared mutable endpoint views;
- Segment lengths `|d_i|` and `|a_i|`;
- Continuity: the `d` endpoint equals the `a` start; the `a` endpoint equals the next row's `d` start;
- Zero lengths are retained as explicit degenerate segments;
- Signed negative parameter orientation;
- `SceneData` legacy three-argument construction and default empty `dh_segments`;
- Existing serialization and visualization behavior remain unchanged during V1.

**Representative robot cases:**

1. Planar RR, with `d_i = 0` and nonzero `a_i`.
2. Spatial RRR with nonzero twist (`alpha_i`), nonzero `a_i`, and nonzero `d_i`.
3. RRP or SCARA with a prismatic `d_i` that varies by configuration.
4. Fully or partly folded configurations, including coincident DH origins.
5. Symbolic link/offset parameters resolved numerically through `num_vals`.
6. Negative offsets and lengths; zero/degenerate configurations.
7. Unresolved, nonreal, or nonfinite DH translation scalar inputs produce contextual errors.

Numerical comparisons should use an appropriate absolute/relative tolerance; the tests must not require bitwise equality between independently evaluated symbolic/numeric expressions.

## 5. Increment boundaries and acceptance criteria

**This PR should contain:**

- `moro/visualization/data.py`: `DHSegmentData` and optional `SceneData.dh_segments` field.
- `moro/visualization/evaluation.py`: evaluation helper and scene population.
- `moro/visualization/__init__.py`: public export.
- Focused tests and relevant API documentation.

**Leave for V2/V3/V4/V5:**

- Characteristic length `L_c` and layout resolution;
- Kinematic and visual bounding boxes;
- Stable per-animation sizes and camera framing;
- Matplotlib/Three.js rendering of DH segments and JavaScript payload changes;
- Default backend change to Three.js;
- Advanced joint semantics, joint labels, link body geometry, CAD representations.

**V1 is accepted when** the new segment data are numerically and geometrically correct, the representative tests pass, and existing scene creation, backends, animation, and public calls retain their current behavior. Follow the normal review/CI process before merging into `develop`.

## 6. Follow-on design note

V2 will revisit the proposed scale `L_c = sum_i(|a_i|+|d_i|)`, its degeneracy rules, and stability across animation sequences. V3 will design `VisualizationLayout` / `resolve_layout()`. These formulas and APIs are **not approved as final** by this V1 document.


## 7. V2 — Scale metrics (implemented, backend adoption deferred)

### Distinct concepts

- **Characteristic length** is a geometric, unit-covariant measure of the evaluated DH chain.
- **Reference length** is a single scale chosen across a plot or animation to keep visual dimensions stable.
- **Spatial bounds** describe the region occupied by frames, joints, DH segments, and optional trajectories; they belong to V3–V4, not V2.

### Geometry metric

For evaluated DH segments `s`, in fixed `d_i, a_i` order:

\[
L_c = \sum_s \lVert s.end - s.start \rVert_2
    = \sum_{i=1}^n(|d_i| + |a_i|).
\]

`SceneData.characteristic_length` is a read-only computed property:

- Returns a nonnegative finite `float` when `dh_segments` is not empty.
- Returns exactly `0.0` for available but entirely degenerate geometry.
- Returns `None` if geometric segments are **unavailable**, including scenes manually constructed with the legacy constructor and default empty `dh_segments`.
- It is not cached: recomputation honors subsequent changes to mutable segment endpoints.
- Invalid, nonfinite segment geometry raises `ValueError`, rather than introducing a floor or silently ignoring data.

This metric is intentionally **not** the physical reach, swept volume, or the sum of actual solid link lengths. It describes the evaluated DH translations and remains meaningful for folded configurations whose end-effector position is close to the base.

### Global animation reference

```python
from moro.visualization import resolve_reference_length

reference = resolve_reference_length(scenes)
```

`resolve_reference_length(scenes: Iterable[SceneData]) -> float | None` returns the maximum available `characteristic_length` in the provided iterable. Scenes without evaluated DH geometry (`None`) are skipped:

- If at least one scene has geometry, the result is its largest nonnegative length, including `0.0` when all have zero length.
- If no scene has geometry, return `None`.
- The iterable is consumed once, permitting a generator.
- A finite result is mandatory.

For a single evaluated scene, the chosen reference is that scene's characteristic length. For a **precomputed** animation, the reference stays constant for every frame. A caller must not recompute a separate reference per frame during rendering.

### Degenerate and legacy scenes

`0.0` and `None` must not be confused. V2 adds no unit-specific floor and does not alter the legacy `SceneData.dimension` calculation.

V3 will decide how to resolve an explicit user `reference_length`, an automatic reference, and an eventual **visual-only** fallback for fully degenerate or legacy scenes. This fallback must not change the geometrical quantity `L_c`.

### Scope and compatibility

- V2 adds `SceneData.characteristic_length` and the reusable `resolve_reference_length` helper in `moro/visualization/scaling.py`, exported from `moro.visualization`.
- No camera, scene bounding boxes, backend defaults, Matplotlib marker sizes, Three.js cylinder radii, `SceneData.dimension`, or JavaScript payloads change in V2.
- Dedicated regression tests cover proportional unit conversion, folding, prismatic motion, empty/legacy scenes, negative lengths, mutation, and nonfinite coordinates.


## 8. V3-A — Backend-independent layout (implemented)

`moro.visualization.layout` introduces `VisualizationLayout` (frozen, slotted) and `resolve_layout(scenes, *, reference_length=None)`.

The layout contains the following **world-space lengths**, in the robot's input linear units: `reference_length`, `frame_length`, `joint_radius`, `base_radius`, and `link_radius`. Initial ratios to the resolved reference length are `1/6`, `0.020`, `0.030`, and `0.015`, respectively. These ratios are internal provisional defaults, not a backend's screen-space drawing sizes.

Reference selection precedence:

1. Explicit finite positive `reference_length` (a nonpositive, nonfinite or nonnumeric value raises `ValueError`).
2. Positive maximum evaluated DH characteristic length across the scenes, via V2 `resolve_reference_length`.
3. Positive finite maximum legacy `SceneData.dimension` when DH geometry does not provide a usable positive length.
4. Visual-only fallback `1.0` when no positive length is available.

The resolver requires a nonempty sequence and computes the reference once for all its configurations; the frozen result is suitable for reuse across animation frames. `SceneData.characteristic_length` and `SceneData.dimension` are not changed by resolution. Fallback is **not** interpreted as a physical size.

Backend adaptation, especially Matplotlib's points-squared markers versus Three.js world-space sphere radii, remains in **V3-B**. Explicit `VisualizationStyle` overrides, link linewidth semantics, serialization, geometry rendering, and camera framing are deliberately unchanged in V3-A. Subsequent work should avoid recalculating nominal scale independently in JavaScript.

V3-A acceptance requires scale homogeneity for nonfallback inputs, constant animation layouts, correct precedence and invalid input handling, and no regressions in existing backend behavior.


## 9. V3-B — Backend layout adapters

V3-B **consumes** the world-space `VisualizationLayout` built in V3-A without changing the DH evaluator or introducing camera/bounding-box logic.

### Three.js

Both `ThreeJSBackend.render(scene)` and `ThreeJSBackend.animate(scenes)` calculate a layout in Python when one is not supplied. Animation resolves a single layout from the entire sequence. The resulting JSON includes one top-level `layout` with `reference_length`, `frame_length`, `joint_radius`, `base_radius`, and `link_radius`. The legacy `_scene_to_payload` result is unchanged.

`threejs_common.js::resolveStyle(style, sceneScale, layout)` uses the Python layout values **for automatic sizes**:

- Automatic axis length: `layout.frame_length`.
- Automatic joint/base sphere radii: `layout.joint_radius` and `layout.base_radius`.
- Automatic link cylinder radius: `layout.link_radius * style.link_linewidth / 3`.

Explicit `frame_scale`, `joint_size`, and `base_size` retain their previous meaning as world-space sizes; explicit `link_linewidth` continues to multiply the base link radius. JavaScript retains its legacy sizing fallback only when no usable layout is serialized (for compatibility with older hand-built template payloads).

### Matplotlib

`MatplotlibBackend.render()` resolves one layout for a static scene, while `MatplotlibBackend.animate()` resolves it **once for all animation frames** and passes the same object through each redraw.

- Automatic `frame_scale` uses `layout.frame_length` as the `ax.quiver(length=...)` world length.
- Automatic joint and base marker areas use the respective world-space layout radii, converted into Matplotlib `scatter(s=...)` points-squared. The conversion uses the shorter axes-viewport dimension and a **nominal** visible span of three reference lengths.
- Explicit `joint_size`/`base_size` preserve the historical `s=size*100` in points-squared; explicit `frame_scale` preserves world units. `link_linewidth` remains in Matplotlib points.

The nominal span is an **interim display adapter**, not an exact perspective projection. Exact world-to-screen sizing under zoom/resizing, camera framing, consistent world scale for very small robots, and full bounding boxes belong to V4. Therefore V3-B does not promise identical pixel size under arbitrary camera zoom or viewport changes.

Both backends accept an optional `layout: VisualizationLayout` parameter to support explicit user-selected scales from `resolve_layout(..., reference_length=...)`. Existing calls remain valid.

### Non-goals

V3-B intentionally does **not** change:

- Existing `SceneData.dimension` or geometry payload fields;
- The `RobotVisualizer` default backend;
- Joint/link geometry placement (DH-segment drawing belongs to a later increment);
- Matplotlib axis limits or Three.js camera/grid/world-axis sizing;
- Trajectory rendering or colors.

### Regression expectations

Test unit conversion proportionality, explicit override compatibility, shared animation layout, Three.js JSON serialization, existing visualization behavior, and the Python 3.11–3.14 CI matrix. V3-B should be merged only after all required checks pass.

## 10. V4-A — Backend-independent scene bounds and framing

### Purpose

V4-A introduces `moro.visualization.framing` and exports `SceneBounds`, `SceneFraming`, `compute_scene_bounds`, `union_bounds`, and `resolve_framing`. No renderer, camera, axis limits, JavaScript payload, or `SceneData.dimension` changes in V4-A.

### Geometry contract

`SceneBounds(minimum, maximum)` is a frozen, slotted dataclass of three-coordinate **finite world-space** tuples. Array-like constructor arguments are normalized to independent tuples, and each minimum coordinate must not exceed its corresponding maximum. The `center` and `size` properties return three-element tuples. The center is computed without directly adding large endpoint coordinates.

`compute_scene_bounds(scene, *, layout=None, style=None)` includes the following **visible** primitives:

- If `style.show_links`, the evaluated DH segment endpoints, expanded conservatively by `layout.link_radius * style.link_linewidth / 3`. Zero-length DH segments are skipped. If the scene has no DH segments, include nonzero legacy joint-to-joint segments instead.
- If `style.show_joints`, joint centers expanded by nominal `layout.joint_radius`, or nominal `layout.base_radius` for the first joint. Exclude the base center/radius when `show_base=False`.
- If `style.show_frames`, each frame origin and three axis tips calculated from the numerical frame directions with explicit `style.frame_scale`, otherwise `layout.frame_length`.
- If `style.show_trajectory`, the end-effector position given by the final numerical frame of each scene. For a precomputed animation, the union naturally covers its complete **sampled** trajectory, regardless of whether the line is displayed in `full` or `trace` mode.

Do **not** use the legacy `SceneData.dimension` as an additional geometric bound. Do **not** include the adaptive grid, global world axes, hidden geometry, or points that have no visual representation.

Radii are conservative backend-independent world-space approximations; explicit Matplotlib point-squared marker sizes cannot be exactly bounded in world coordinates. Exact screen-space overscan and zoom-dependent behavior remain V4-B concerns.

When no visible primitives are available, use a degenerate bounds point at the world origin. Its positive viewing volume is resolved by the *visual-only* fallback described below.

`union_bounds(bounds)` computes the coordinate-wise union of a nonempty sequence of such boxes. Empty sequences and invalid positions are rejected with `ValueError`.

### Framing contract

`SceneFraming(bounds, center, half_extent, fit_radius)` is immutable and contains a cubic half-extent (for Matplotlib), and a bounding-sphere radius (for projection-independent Three.js camera fitting), in world units. Its values do **not** include specific camera FOVs, aspect ratios, orientation, near/far planes, grids, or axes.

```python
layout = resolve_layout(scenes)
framing = resolve_framing(scenes, layout=layout, style=style)
```

`resolve_framing` computes a **single** global union across all supplied evaluated configurations. The default margin is 10%, and the minimum visual half-extent factor is 10% of `layout.reference_length`. Write `s` for the box size and `L` for the reference length. The provisional formulas are:

```text
h = (1 + margin) * max(s_x/2, s_y/2, s_z/2, min_extent_factor * L)
R = (1 + margin) * max(hypot(s_x/2, s_y/2, s_z/2), min_extent_factor * L)
```

Here `h` is `half_extent` and `R` is `fit_radius`. Neither result enforces a fixed minimum in meters, millimeters, or any other physical unit. Positive finite results are required. User-supplied margin and minimum factor must be finite and nonnegative.

This provides scale covariance for properly evaluated DH geometry and stable framing over sampled animations. It is **not** a continuous swept-volume bound between samples. The caller may provide an existing `VisualizationLayout` to guarantee that size and framing refer to the same global animation reference.

### V4-B boundary

V4-B will apply the framing to Matplotlib's three axis limits and aspect ratio, and to Three.js perspective/orthographic camera target, visible height/distance, clipping planes, reset and view switches. The current V3-B Matplotlib marker-area adapter still uses a nominal view span; V4-B should tie it to the actual fitted view. Any exact screen-pixel padding needed for user-supplied marker sizes belongs in the backend adapters, not in geometric AABBs.

## 11. V4-B — Backend framing adapters

### Purpose

Apply the Python-resolved `SceneFraming` from V4-A consistently to Matplotlib and Three.js, without changing `evaluate_robot()`, the legacy `SceneData.dimension`, the robot API, the default backend, or the geometry of rendered links.

All rendered scenes are framed from their **visible evaluated geometry**. Animations share one global `VisualizationLayout` and `SceneFraming` across all sampled frames; neither visual size nor camera target is recomputed per frame. The framing describes samples, **not** a guaranteed continuous swept volume between samples.

### Matplotlib

`MatplotlibBackend.render` and `MatplotlibBackend.animate` accept optional `framing: SceneFraming` after the existing `layout` argument. When omitted, they call `resolve_framing(..., layout=layout, style=style)`; animations resolve it **once** before constructing the `FuncAnimation`.

The common `_apply_framing(ax, framing)` sets symmetric numerical limits about `framing.center` along X, Y and Z, each with `framing.half_extent`, and `ax.set_box_aspect((1,1,1))`. The animation initializer and per-frame redraw reapply the same limits after clearing. `frame_scale` retains its V3-B world-space meaning.

Automatic Matplotlib point-squared scatter-marker sizes now use the **actual common visible span** `2 * framing.half_extent`, instead of the provisional V3-B nominal length. Explicit user `joint_size`/`base_size` overrides still map to `s=size*100` in points². Since Matplotlib's markers are display-space primitives, their projected size remains approximate under perspective, zoom or viewport resizing.

### Three.js

`ThreeJSBackend.render`/`animate` likewise accept optional `framing: SceneFraming` and compute it after `layout` when omitted. A single top-level `framing` dictionary is serialized along with `layout`:

- `minimum` and `maximum`: world-space AABB corners;
- `center`: target point for `OrbitControls` and all preset views;
- `half_extent`: cubic half-extent available to backends;
- `fit_radius`: conservatively enclosing sphere radius including margin.

Individual scene payloads from `_scene_to_payload` are **unchanged**.

`threejs_common.js::resolveCameraFit(framing, aspect, fovDegrees)` computes the projection-specific fit. Let `R` be the Python-resolved `fit_radius`, `a` the viewport width/height aspect ratio, and `phi_v` the vertical perspective FOV (50°). Then:

```text
phi_h = 2 * atan(tan(phi_v / 2) * a)
phi_min = min(phi_h, phi_v)
distance = R / sin(phi_min / 2)
ortho_half_height = R / min(1, a)
ortho_half_width = a * ortho_half_height
near = 0.1 * (distance - R)
far = distance + 4 * R
```

The orthographic camera uses symmetrical front/back clipping planes `[-far, far]`. These choices are provisional, conservative and **unit covariant** (m and mm); no absolute floor such as 0.01, 1 or 1000 is introduced in the new path. The old `createCameras` path remains available only for legacy JavaScript callers without a framing payload.

Both the static and animation templates use `framing.center` as the initial and reset view target. The camera positions for front, top and isometric views use the fitted distance; top view uses the valid camera `up=[0,1,0]` so the world remains Z-up. Grid sizing remains centered at the world origin and uses bounds for a sufficiently large footprint without contributing to the camera fitting volume. Lighting follows the fitted center to work with offset robots.

Camera type switching retains the same control target; the synchronized projection path is shared by static and animation views. The reset action restores the original fitted projection, zoom and isometric orientation, preserving the currently selected camera type.

### Validation and limitations

Tests cover offset geometry, a common world-scale framing per animation, consistency across m/mm, explicit framing and style overrides, Matplotlib equal-axis limits and stable frames, and serialized Three.js payloads. The V4-B JavaScript helper must also be verified at representative wide and tall aspect ratios.

The visible-geometry bounds of V4-A are conservative approximations. Exact projected marker padding, complex viewport resizing, non-Cartesian geometries, and future continuous swept-volume bounds are not included here. Review in actual Jupyter/Colab views before treating appearance as finalized. V5 will address remaining display polish and the default backend.

## 12. V5-A — Render evaluated classic-DH geometry

### Objective and scope

V5-A closes the gap between V1's evaluated DH geometry and the actual
rendered link paths. The numerical evaluator remains the **single authority**
for the ordered translation contributions `(d_i, a_i)`; each endpoint is in
base/world coordinates. Neither a CAD solid nor an actuator location is
inferred from these segments.

V5-A does **not** change the joint markers, robot kinematic model,
`VisualizationStyle` fields, camera framing, `SceneData.dimension`,
`RobotVisualizer` API, or the default backend. V5-B handles joint
symbols; V5-C handles the default backend switch and release cleanup.

### Serialization

`threejs_backend._scene_to_payload(scene)` preserves its original
`{"joints", "frames", "dimension"}` keys for legacy scenes that have no DH
segments. For scenes with evaluated DH geometry, append a
`"dh_segments"` array containing, in their original order:

```json
{"row_index": 1, "kind": "d", "start": [0, 0, 0], "end": [0, 0, 1]}
```

Each entry contains the 1-based DH row index, `"d"` or `"a"` contribution,
and two 3D numerical endpoints. This array retains **zero-length** segments:
a collapsed prismatic offset must not alter animation mesh topology or
change the identity of a DH row.

The same `_scene_to_payload` helper is used for a static view and for each
sample of an animation. V3's global layout and V4's global framing payloads
are unchanged.

### Matplotlib

When `scene.dh_segments` is available and `show_links=True`, render one
`Line3D` for each nonzero evaluated contribution, directly between its
start and end. Zero-length segments remain in the numerical scene, but do
not add empty plotted lines. `show_links=False` suppresses them.

For manually constructed legacy scenes without DH segments, preserve the
existing direct joint-to-joint polyline. Axis labels, joints, frames,
trajectory, global animation framing, and style overrides are unchanged.
Redrawing each animation frame uses that frame's evaluated DH segments.

### Three.js

Static scenes create one cylinder from each **nonzero** evaluated DH
segment and use the Python-supplied endpoints directly. An exact zero
length produces no cylinder, regardless of the robot's units.

Animations allocate **one persistent mesh per evaluated DH segment**, even
for zero-length contributions. The update step reads each frame's
endpoints and updates the mesh's position, direction, length and visibility.
A degenerate segment is hidden, rather than removed; it becomes visible
again when a prismatic displacement increases.

The animation topology validator rejects changes in the existence, count,
ordering, `row_index`, or `kind` of DH segments between frames (in
addition to the existing joint/frame-count checks). Geometric segment
lengths may change freely.

When `dh_segments` is unavailable, retain the previous legacy geometry
fallback, including the existing corner reconstruction. This calculation
must not be used for new evaluated scenes. The former 1e-6 **absolute
world-unit** cutoff for static and animated cylinders is removed; positive,
finite link lengths are never rejected merely for being small.

### Acceptance criteria

1. RR planar robots render the two nonzero `a` contributions without
   spurious `d=0` geometry.
2. Spatial robots with nonzero positive or negative DH translations
   render exactly the ordered `d` and `a` segments rather than one
   diagonal between origins.
3. RP/RRP animations preserve mesh topology as a prismatic `d`
   contribution crosses zero; a hidden mesh must reappear.
4. Realizing an otherwise identical robot in m or mm changes link sizes
   proportionally but not visibility.
5. Legacy scenes with no evaluated DH geometry continue to work without
   a new required field.
6. The Python 3.11–3.14 test matrix, packaging and Sphinx documentation
   builds remain green. Confirm actual appearance in Jupyter/Colab before
   integrating when possible.

## 13. V5-B1 — Evaluated kinematic joint metadata

### Geometry and semantics

Classic DH joint `i` acts **about** (revolute) or **along** (prismatic)
the positive `z_(i-1)` direction through the previous evaluated frame
origin `p_(i-1)`, in base/world coordinates:

```text
A_i(lambda) = p_(i-1) + lambda * z_(i-1),  i = 1,...,robot.dof
```

This is a *kinematic motion line*, not a physical bearing, motor, guide,
actuator housing or point that must be attached to the moving link. In
particular, when `a_i != 0`, the current frame origin `p_i` need not lie
on the joint axis. The existing `SceneData.joints` field continues to
store the `n+1` frame origins, including the base. Never reinterpret
those origins as the `n` joint axes.

### Data contracts

`moro.visualization.data.JointData` is a slotted dataclass with fields:

| Field | Type | Contract |
| --- | --- | --- |
| `index` | int | 1-based joint/DH row index, positive; reject booleans and floats |
| `kind` | Literal["r", "p"] | Exact lowercase robot joint type |
| `origin` | numpy float[3] | Independent copy of `p_(i-1)`, finite world-space position |
| `axis` | numpy float[3] | Independent normalized copy of `z_(i-1)`, finite nonzero direction |

The constructor validates both vector shapes, finite values and the
nonzero axis norm, and normalizes the supplied axis to unit length.
There is no physical-unit threshold for a direction's magnitude.
`JointData` represents a **line and kind**, not the visual size or
position of a particular three-dimensional drawing.

`SceneData` appends the optional field
`joint_data: list[JointData] = field(default_factory=list)` **after**
`dh_segments`. This preserves existing positional constructors with
three or four arguments and keeps the list independent for each scene.
The `joints` list is unchanged.

`evaluate_robot` builds the ordered `n` joint entries using
`robot.joint_types` and the `n+1` already evaluated frames, with
`index=i`, `origin=frames[i-1].position` and
`axis=frames[i-1].z`. It does **not** recalculate symbolic DH transforms.
The axis and origin arrays do not alias frame matrices or DH segments.
For animations, each scene evaluates current world-space joint axes
independently, without changing the joint order or types.

### Renderers and forward compatibility

**V5-B1 does not change the appearance or rendering payload.** Neither
Matplotlib nor Three.js consumes `joint_data` yet; the Three.js
`_scene_to_payload()` keys remain unchanged, including for evaluated
scenes with both DH segments and joint metadata. Existing sphere markers
continue to represent frame origins, not kinematic joint symbols.

**V5-B2** will serialize `joint_data`, design the actual R/P symbols,
orient them about/along the motion axes, and integrate them with the
shared layout and framing. Proposed style choice:
`joint_representation="simple" | "kinematic"`, retaining `"simple"` as
the initial default. Exact graphic anchor, symbol dimension ratios,
base overlap handling and camera-space appearance remain V5-B2 design
decisions; they are *not* inferred in V5-B1.

### Acceptance criteria

- Evaluate RR, RP and spatial RRP with nonzero signed DH translations and twists;
- Verify axes originate at the **previous** frame, not the next;
- Verify joint kind and one-based order match `robot.joint_types`;
- Verify finite, unit-length and independently copied vectors;
- Reject zero/invalid axis directions and malformed index or kind;
- Verify prismatic movement changes its DH `d_i` displacement without
  changing the corresponding axis when prior joints are held fixed;
- Preserve old `SceneData` construction and renderer payload behavior;
- Pass Python 3.11–3.14, distribution and Sphinx CI before integration.

## 14. V5-B2 — Shared schematic revolute and prismatic joint symbols

### User-facing style (opt-in)

`VisualizationStyle(joint_representation="simple", joint_scale=1.0)`
preserves the existing frame-origin sphere markers by default. The
`"kinematic"` option replaces frame-origin joint markers with schematic
motion symbols at `JointData.origin` oriented with `JointData.axis`.
Both options honor `show_joints`. `show_frames` and `show_links`
remain independent. A base sphere is kept at `SceneData.joints[0]`
only when `show_base=True`, independently of the symbol for joint 1.

`joint_scale` is a finite, strictly positive, dimensionless multiplier
for the new symbols. It does not change the size of the legacy spheres,
the nominal link radius, or the frame axes. For the first iteration it
defaults to 1, and the initial default representation remains
`"simple"` to avoid surprising callers.

A caller who asks for `kinematic` with visible joints must provide
`SceneData.joint_data` (normally obtained from `evaluate_robot`).
No joint types or axes are fabricated for legacy hand-built scenes;
an informative `ValueError` is raised. With `show_joints=False`,
neither metadata nor symbols are required.

### Geometry and kinematic meaning

`moro.visualization.joint_symbols.build_joint_symbols(scene, layout, style)`
builds JSON-compatible **world-space polylines** in Python. Matplotlib
uses the points directly; Three.js receives exactly the same points,
without recomputing a motion axis, numerical DH transform or geometry.

For joint i, the symbol is anchored at `p_(i-1)`, oriented along the
unit `z_(i-1)` axis, and rotated in the transverse plane using the
previous frame's X direction for a stable basis. For scenes without a
previous frame the fallback basis chooses the least-aligned world axis.
The transverse pair is right-handed, with
`u × v = z_(i-1)` (approximately, subject to floating-point error).

- **Revolute R:** a positive-direction circular arc in the plane normal
  to the joint axis plus a short arrowhead. Its nominal radius is
  `2.8 * layout.joint_radius * joint_scale`.
- **Prismatic P:** two axial guide lines, a centered rectangle symbol,
  and an axial arrow. Dimensions are fixed ratios of the same radius.
  The representation describes permitted translation only; the
  indicated slider is **not** driven along the evaluated `d_i` segment
  and does not assert physical actuator geometry.

All dimensions scale with `VisualizationLayout`. They have no
absolute minimum in meters, millimeters, or screen units. Shapes remain
numerically well defined for zero evaluated DH translations, since the
symbols depend on `JointData` and layout, not on nonzero `d_i`.

### Rendering and animations

`compute_scene_bounds` includes every symbol polyline vertex while
kinematic symbols are visible. If `show_base` is enabled, it includes
the independent base sphere radius. Hidden joint symbols make no
contribution to framing.

Matplotlib draws each polyline as a Line3D, retaining the same
`joint_color`; with `simple` it keeps the current scatter markers
unchanged. Each animation frame is redrawn with the shared sequence
layout and framing.

Three.js serializes `joint_symbols` only when visible in `kinematic`
mode, at the scene level for static views or the frame level for
animations. Every symbol has `index`, `kind`, and ordered `paths`.
`simple` payloads keep their historic geometry. On animation
initialization, one THREE.Line is allocated for each symbol path. Each
sample updates the existing BufferGeometry position attribute and marks
it dirty, with no per-frame geometry reconstruction. The topology check
rejects count, kind, index, path-count and vertex-count changes across
samples. These line symbols do not use Three.js mesh transparency or
physically model a prismatic housing.

### Acceptance

1. Both simple and kinematic modes render and animate without changing
   the robot model, DH segments, or V4 camera framing contract.
2. RR, RP and spatial RRP demonstrate the correct axes even when frame
   i is offset from the physical motion line.
3. Kinematic primitives rotate/translate with the **preceding** DH frame
   and retain stable vertex counts across samples.
4. Models expressed in m or mm create proportionally scaled polylines
   and the same appearance under camera framing.
5. Hiding joints or base works independently; legacy scenes without
   joint metadata work in simple mode.
6. CI passes on Python 3.11–3.14, packaging and Sphinx.
   The user reviews the visual appearance before merging.

### V5-B2 experimental Three.js joint appearance comparison

After the first schematic symbols were validated visually, V5-B2 includes
two **experimental Three.js-only** appearances for comparison before
choosing the long-term default:

- `joint_appearance="schematic"` (default): existing R/P line symbols,
  including their sense-of-motion arrows.
- `joint_appearance="solid"`: a cylinder on the R motion axis or a
  rectangular prism aligned with the P axis, with no overlaid line symbols.
- `joint_appearance="hybrid"`: a semitransparent cylinder/prism together
  with the existing R/P line symbols.

This option is appended **after** existing style fields to avoid breaking
positional `VisualizationStyle` construction. It has an effect only when
`joint_representation="kinematic"` and `show_joints=True`. Matplotlib
supports `schematic`; attempting either 3D appearance in Matplotlib raises
a contextual `ValueError` rather than silently showing something else.

`joint_solids.build_joint_solids(scene, layout, style)` resolves world-space
solid descriptors from the existing `JointData` and `VisualizationLayout`.
The solid center is always `JointData.origin` and its main longitudinal
axis is `JointData.axis`. The cylinder radius and length and rectangular
prism width, depth and length are positive proportional multiples of
`layout.joint_radius * joint_scale`. Their sizes do not depend on the
prismatic extension. Three.js constructs `CylinderGeometry` or
`BoxGeometry` with their long axis along local Y. The orientation
uses the motion axis for local Y and the preceding DH frame X for local X,
so the prism faces and schematic guides have a consistent roll. Static
and animated views
share the same descriptors. Animation meshes persist and only the world
position/quaternion are updated per frame. A stable joint `index`/`kind`
order is checked before updates.

When solid geometry is visible, `compute_scene_bounds` includes a
conservative world-space enclosing sphere around each solid, so no camera
clipping occurs for tilted R/P axes. For the hybrid appearance it also
includes the existing line paths. The original base sphere remains
independently controlled by `show_base`.

This experiment does **not** imply mechanical CAD geometry or authorize
switching the library's default joint representation/backend. The
comparative notebook recipe is in
`docs/design/0.6.0/joint-appearance-experiment.md`. The result should be
evaluated visually on RR, RP and spatial RRP before selecting a preferred
appearance or implementing volumetric Matplotlib primitives.
