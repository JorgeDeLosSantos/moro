# Moro 0.6.0 — Visualization design

**Status:** V1, V2 and V3-A integrated into `develop`; V3-B implementation in review; V4–V5 pending.

**Current integration branch:** `feat/0.6.0-layout-backends` (from `develop`).

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
