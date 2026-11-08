# Moro 0.7.0 — Detailed design: physical-time animation playback (T-06)

> **Status:** Design approved and merged in PR #48; C1 contract merged in PR #49; C2 Three.js playback implemented in the current increment, pending review.  
> **Scope:** Use explicit trajectory timestamps to reproduce sampled robot configurations with their original, possibly nonuniform timing in Three.js, while preserving the existing fixed-interval mode and documenting Matplotlib limitations.  
> **Planning reference:** [0.7.0 inventory](../../roadmap/0.7.0/index.md).  
> **Implementation base:** After T-04/B3 (PR #47), with current `RobotVisualizer.animate()` and backend animation APIs.

## 1. Motivation and scope

Trajectory generators return time-major numerical data, including potentially
nonuniform physical times from `JointTrajectory.t` and `PositionTrajectory.t`.
The visualizer currently ignores trajectory times: Three.js advances using
`setInterval` with a fixed millisecond interval, and Matplotlib uses
`FuncAnimation(interval=...)`. Supplying a sampled joint matrix does not
implicitly transfer the trajectory's physical time axis.

**T-06 must make timestamped playback explicit.** The visualization layer
reproduces supplied samples, but does not derive new configurations or
interpolate between them; mathematical motion generation remains separate.

**Committed first target:** Three.js, currently the default backend. For
Matplotlib, preserve existing `interval` functionality and explicitly reject
the unsupported new physical-time mode rather than silently misrepresent
timestamps. Matplotlib time-aware support is an independent future candidate
if a reliable cross-backend/export contract can be demonstrated.

Not in scope: direct input of a `JointTrajectory` object, resampling, per-frame
pose interpolation, trajectories of full Cartesian pose, dynamic simulation,
real-time control guarantees, new actuator constraints, or a general animation
scheduler library.

## 2. Public API and compatibility

Keep the existing `RobotVisualizer.animate(configurations, backend="threejs",
**kwargs)` call form and its defaults. Add optional keyword-only settings:

```python
viz.animate(configurations, times=None, playback_speed=1.0, backend="threejs")
```

Illustrative usage:

```python
# Existing legacy frame mode remains unchanged:
viz.animate(traj.q, interval=50)

# Physical-time mode (seconds, possibly nonuniform):
viz.animate(traj.q, times=traj.t, playback_speed=1.0)

# Parameters for a symbolic robot are handled through the existing mapping
# interface until API-001 independently introduces 'parameters':
configurations = [
    {**parameters, **dict(zip(robot.qs, q))}
    for q in traj.q
]
viz.animate(configurations, times=traj.t)
```

`times` is separate from the configuration sequence; direct
`viz.animate(traj)` is **not** added by T-06. This avoids coupling the
visualization layer to a specific trajectory result class and leaves any
result-object interoperability proposal to API-001 or subsequent work.

Keep `backend` as the existing optional second positional argument,
`num_vals_list` as the existing positional/keyword parameter, and
`**kwargs` for style/layout/etc. The future API-001 parameter renames and
symbolic `parameters` support are **not** prerequisites for T-06.

### 2.1. Mode selection

| Invocation | Mode | Expected result |
| --- | --- | --- |
| `animate(configurations)` | Legacy uniform interval, default 100 ms | Existing behavior |
| `animate(configurations, interval=50)` | Legacy uniform interval | Existing behavior |
| `animate(configurations, times=t)` | Physical-time mode, default speed 1.0× | Timestamped playback on Three.js |
| `animate(configurations, times=t, playback_speed=2)` | Physical-time mode, 2× speed | Timestamped playback on Three.js |
| `animate(configurations, times=t, interval=100)` | Invalid ambiguous timing controls | Raise `ValueError` even when 100 ms |
| `animate(configurations, playback_speed=2)` | Invalid without timestamps | Raise `ValueError` |
| `animate(configurations, backend="matplotlib", times=t)` | Unsupported physical-time backend | Raise explanatory `ValueError` |

An explicit `interval` keyword must be distinguishable from its implicit
default. Prefer checking keyword presence in the public orchestrator, or
an internal sentinel/default-`None` in a backend, rather than treating
`interval=100` as if the caller omitted it.

`playback_speed` is a finite positive real scalar (excluding booleans),
is dimensionless, and only applies when `times` is supplied. The explicit
default `1.0` is accepted in legacy mode; a non-default value without
`times` must not be silently ignored.

Return type remains `IPython.display.HTML` for Three.js and
`matplotlib.animation.FuncAnimation` for legacy Matplotlib. Existing
plotting, styling, framing, scene-data, and joint-configuration contracts
remain unchanged.

## 3. Timestamp validation and normalization (Python)

For `N` configurations, `times` must be a one-dimensional, finite, real
numeric sequence of shape `(N,)`, strictly increasing and containing at
least **two** samples. Invalid inputs raise explanatory `ValueError`:

- Mismatch between number of frames and timestamps.
- Scalar/matrix/ragged, empty, complex, nonnumeric, boolean or nonfinite data.
- Duplicate or decreasing times, including values that become
  indistinguishable under float conversion.
- A single frame in physical-time mode (legacy animation may still use one).
- Nonfinite, zero, negative or boolean playback speed.
- Conflicting `interval` and `times` arguments.

Nonzero and negative initial timestamps are valid: compute relative
`u_i=t_i-t_0` so `u_0=0` and `u_(N-1)>0`. These are physical
**seconds**, whereas legacy `interval` remains milliseconds. Preserve
the source sequence without mutation, and validate normalized timestamps
are finite and strictly increasing before JSON serialization; reject
nonrepresentable differences or scales.

The browser payload carries one relative timestamp per frame plus
`playback_speed`. Do not implicitly normalize sample spacing to uniform.
Do not pass the original `t0` to a browser clock or use wall-clock time.
The displayed elapsed time can use relative seconds (`0..u_(N-1)`),
while original `t` stays available to the Python caller.

## 4. Three.js playback engine

### 4.1. Playback state

Time-aware playback has an elapsed physical-time cursor `u`, a monotonic
browser clock anchor (via `performance.now()`), a playback speed factor,
and a running/paused/finished state. `requestAnimationFrame` drives
updates when available; it is **not** used as the physical clock itself.

When running, the target elapsed physical time is

```text
u_target = u_anchor + (performance.now() - wall_anchor_ms) / 1000
                      * playback_speed
u_target = min(u_target, u_final)
```

The displayed frame index is the largest `i` such that
`relative_times[i] <= u_target` (upper-bound/binary-search style).
On arrival at `u_final`, explicitly show the last frame and stop.
No interpolation or generation of intermediate robot configurations.

When rendering falls behind (e.g., browser tab throttling, expensive
scenes), it is acceptable to **skip visual frames** to catch up to the
physical-time clock; never accumulate successive timer delays and
call that faithful temporal playback. This is a best-effort visual
reproduction, **not** a hard real-time guarantee.

### 4.2. User interactions

- **Play from the beginning:** start from the first timestamp.
- **Pause:** capture the current elapsed physical time and retain the
  corresponding frame; do not reset.
- **Resume:** re-anchor the monotonic clock from the saved elapsed time.
- **Seek with frame slider:** snap to that frame's timestamp. If playing,
  re-anchor so playback continues from the sought timestamp; if paused,
  stay paused.
- **Reaching final time:** show final configuration exactly and stop.
  Pressing Play after completion restarts from the first frame.
- **Single-frame legacy mode:** retain existing behavior, but not
  supported for physical-time playback.
- **Camera changes, orbit, Fit View:** never modify playback time,
  speed, frame selection or pause/resume state.

The existing slider can continue selecting integer **frame indices**.
Add a compact visible relative-time indicator (elapsed/total seconds),
and display speed as a label (e.g. "1.0×"); no interactive speed picker
or new transport mode is required in this increment.

### 4.3. Legacy mode isolation

Without `times`, preserve current `setInterval`-driven playback
(including current cycling/looping behavior), slider behavior, and
100 ms default interval. Do not change legacy speed, loop semantics,
camera controls, or generated HTML return type as a side effect of T-06.
The new clock-based path is selected **only** for explicit `times`.

The animation template should avoid maintaining two competing live
timers. If both modes share helper functions, document whether a user
control is resetting the clock anchor, and test transitions within the
chosen mode.

## 5. Matplotlib support policy

For T-06 initial completion:

- `viz.animate(configurations, backend="matplotlib", interval=...)`
  remains fully supported as before.
- `viz.animate(configurations, times=..., backend="matplotlib")`
  raises a targeted `ValueError` such as: "Physical-time playback with
  times= is currently supported only by the Three.js backend; use
  backend='threejs' or omit times and specify a fixed interval."
- Documentation must not suggest that `interval` can reproduce arbitrary
  nonuniform `times`.
- Do not silently choose mean/first `dt`, approximate nonuniform times
  by fixed intervals, or claim physical-time correctness for
  `FuncAnimation`.

Future optional Matplotlib support may study changing timer intervals
or explicit resampling, including the differences between notebook,
interactive GUI and exported formats. Its feasibility does not block
Three.js T-06 completion.

## 6. Compatibility with API-001 and downstream results

T-06 acts on ordinary numerical matrices or substitution-dictionary
sequences already accepted by `RobotVisualizer.animate()`. In the
future, API-001 may rename the public configuration argument and add a
`parameters` dictionary for static symbolic model substitutions.
The T-06 options `times` and `playback_speed` are independent
keyword-only controls, so API-001 can proceed without changing their
meaning. For now, supply symbolic parameters using the existing
substitution-dictionary workflow.

`JointTrajectory`, `DynamicsSolution` or IK results may later expose
matching time and configuration arrays. Consuming these via
`viz.animate(motion.q, times=motion.t)` requires no class inheritance,
new shared base class or imports of `moro.trajectory` inside the
visualizer. Do not infer a time vector when a raw matrix is supplied.

## 7. Tests and acceptance gates

### Python validation

- Backward compatibility for current `animate(configurations)`,
  `interval`, mapping and NumPy matrix forms, and both backend result
  types.
- Irregular timestamp grid (e.g. `[0, 0.1, 0.7, 0.8, 1.5]`),
  shifted/negative origins, valid speed factors, and defensive copying.
- Rejection of mismatched lengths, repeated/nonmonotone/nonfinite/boolean
  times, single-frame temporal mode, and non-representable offsets.
- Rejection of explicit `times+interval` (including `interval=100`),
  non-default `playback_speed` without `times`, and Matplotlib `times`.
- Verify that legacy mode serializes **no** time-aware flag that changes
  behavior, and that time-aware HTML contains normalized timestamps and
  speed without source data mutation.

### Browser behavior

- At elapsed `u=0.4` for times `[0, 0.1, 0.7, 0.8, 1.5]`,
  frame 1 must be selected; at `u=0.7`, frame 2; the final frame
  must be visible at completion.
- Pause/resume maintains elapsed physical time instead of restarting;
  seek while playing and paused snaps to sample times.
- Slow/fast speed factors, last-frame completion, Play-after-finish
  restart, background throttling and skipped frames, camera changes
  during playback, and two independent animation widgets in a notebook.
- No interpolation between supplied configurations and no changes
  to joint appearance/topology or end-effector trace semantics.

### Verification strategy

Unit tests should validate Python timestamp normalization and serialized
payloads. Where feasible, factor a small pure JavaScript
timestamp-to-index helper for deterministic browser-side tests.
CI HTML/string assertions alone do **not** prove smooth or correct
browser behavior; supplement with interactive Jupyter/Colab manual checks
of pause, resume, seeking and nonuniform timing. Keep visual regression
requirements proportionate to repository infrastructure.

## 8. C1 safe-increment policy

C1 added the public keyword-only input contract, private validation
helpers, compatibility tests and explicit backend error paths, without
activating Three.js physical-time playback until C2.

C2 removes the provisional Three.js `NotImplementedError` guard,
serializes validated relative timestamps and speed in the HTML payload,
and selects frames using `performance.now()` in the existing
`requestAnimationFrame` loop. The legacy `setInterval` path is retained
only when `times` is absent. Pausing preserves elapsed time; seeking
reanchors the running clock; the final configuration remains displayed
at completion. The built-in frame slider does not interpolate joint data.

Matplotlib `times=` continues to raise a targeted `ValueError` in
C1/C2/C3. Existing uniform `interval` calls remain usable and unchanged
throughout. C3 will include notebook smoke checks and final examples.

## 9. Work packages

| Increment | Scope | Exit criterion |
| --- | --- | --- |
| **C1 — API and time contract** | Optional `times`/`playback_speed`, validation, mode selection, explicit Matplotlib rejection, compatibility tests | Nonuniform input and conflicts tested without altering existing behavior |
| **C2 — Three.js timestamped playback** | Relative time payload, monotonic clock, frame selection, terminal handling, pause/resume/seek, time UI | Python serialization tests and browser-manual tests for variable sample spacing |
| **C3 — Stabilization and docs** | Regression tests, examples from multi-waypoint and trapezoidal motion, user/API docs, Matplotlib limitation | Full supported Python CI, Sphinx build, notebook smoke tests, clear capability matrix |

C3 does **not** require implementing physical-time Matplotlib playback.
A separate future issue or scoped increment may address Matplotlib
only after its semantics have been proven reliable.

## 10. Proposed decisions

**Adopted for design:** Three.js first; Matplotlib retains legacy
uniform-interval support and documents explicit physical-time
limitations; no silent fallback; new controls opt-in; no interpolation
or model-layer coupling.

**Proposed detailed semantics for review:** physical-time mode stops
on the final frame, Play-after-finish restarts, legacy mode continues
looping, frame slider seeks to sample timestamps, playback uses a
monotonic clock, and API uses explicit `times` with optional
`playback_speed`. No direct `JointTrajectory` input in T-06.
