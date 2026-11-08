# Moro 0.7.0 — Preliminary feature inventory

> **Status:** Candidate inventory, **not frozen release scope**.  
> **Consolidated:** 2026-11-03.  
> **Planning base:** the 2026-10-28 rolling Moro roadmap and API discussions held during use of Moro 0.6.1.  
> **Purpose:** decide which features to design and implement in 0.7.0 without prematurely committing to every candidate.

## Release direction

**Working theme:** Advanced Trajectories & Numerical Exploration.

**Trajectory progress (2026-11-07):** T-01–T-03 (multi-waypoint
interpolation and C0/C1/C2 continuity) are implemented in `develop` through
PRs #40–#43. T-04 was designed in PR #44 and completed through
B1/B2/B3 (PRs #45–#47), including
[R/RR teaching examples](../../source/examples/trapezoidal-trajectories.md)
and analytic limit regressions. T-05 remains a separate P1 candidate.
T-06 has an [approved Three.js-first design](../../design/0.7.0/physical-time-playback.md);
C1 stages input validation and error contracts, without activating
physical-time playback until C2.

The primary goal inherited from the rolling roadmap is to extend `moro.trajectory` from point-to-point interpolation to **multi-waypoint trajectories and motion profiles**. Complementary improvements emerged from practical use: more consistent visualization parameters, reusable inverse dynamics over sampled motion, and improved sampled-workspace exploration.

Preserve the project principles:

- educational clarity and inspectable mathematics first;
- simple notebook workflows on Windows, macOS, Linux, Jupyter, and Colab;
- `Robot` as the symbolic model, with numerical algorithms in focused modules;
- interoperability through stable NumPy shapes, rather than unnecessary coupling or new class hierarchies;
- explicit conventions, useful diagnostics, regression tests, and reproducible examples;
- avoid speculative abstractions and measure performance before deeper optimization.

## Planning status and priority definitions

| Level | Meaning |
| --- | --- |
| **P0 — release core** | Proposed identity of 0.7.0. Design first; still requires detailed acceptance and sizing. |
| **P1 — complementary candidates** | Valuable work for 0.7.0 if scoped and independently testable; may move to a later version. |
| **P2 — future radar** | Preserve the idea, but do not assume implementation in 0.7.0. |

**Important:** P0 means *strong planning preference*, not approval of all API signatures, algorithms, or implementation details. The only explicitly approved API candidate from the current discussions is API-001; API-002 and workspace items remain candidates.

## P0 — Primary track: advanced trajectories

The following item IDs are local to this roadmap.

| ID | Candidate | Intent and preliminary scope |
| --- | --- | --- |
| **T-01** | Multi-waypoint joint and Cartesian-position trajectories | Prescribe ordered waypoints, associated with a clearly defined time convention. |
| **T-02** | Piecewise polynomial interpolation | Extend existing `linear`, `cubic`, `quintic` capabilities to multiple segments where mathematically appropriate. |
| **T-03** | Continuity across segments | Specify and validate position/velocity/acceleration continuity (`C0`, `C1`, `C2`) as appropriate to interpolation method and boundary conditions. |
| **T-04** | Trapezoidal velocity profile | [Designed](../../design/0.7.0/trapezoidal.md) and implemented (B1/B2): synchronized rest-to-rest joint motion, triangular/trapezoidal phases, optional prescribed duration and typed profile metadata; B3 adds examples and validation. |
| **T-06** | Physical-time playback | [Designed](../../design/0.7.0/physical-time-playback.md): C1 contracts/guards are staged; actual Three.js timestamped playback (including pause/resume/seek) follows in C2, with Matplotlib explicitly limited to fixed `interval`. |
| **M-01** | API cleanup and compatibility | Complete intended deprecation cleanup and document migrations. |

**T-05 — Basic kinematic limits** is P1 for now: velocity/acceleration constraints can rapidly expand into time allocation and feasibility policy. The limited subset needs design before it is treated as release core.

### Open design questions for T-01–T-06

1. Extend existing `joint_trajectory()` / `position_trajectory()` or add explicit multi-waypoint functions? Preserve simple point-to-point entry points.
2. Define input shape and meaning for waypoints, times, boundary derivatives, and intermediate derivatives.
3. Distinguish geometric paths from time-parameterized trajectories; specify when `C0/C1/C2` refers to continuity in physical time.
4. Decide how interpolation modes behave at joins and whether a requested set of constraints is feasible.
5. Establish a trapezoidal-profile API without promising advanced jerk-limited or time-optimal planning.
6. Decide whether visualizers accept a trajectory object and/or a separate time array, preserving existing array-based animation calls and backend return types.
7. Preserve public output conventions: `t: (N,)`, `q/qd/qdd: (N,dof)`, `p/v/a: (N,3)`.
8. Ensure diagnostics and didactic examples expose the effect of methods, timing, and boundary conditions.

### M-01: previously scheduled 0.7.0 compatibility work

Release notes for 0.6.1 identify these as intended 0.7.0 cleanup, subject to final import-usage and migration review:

- remove deprecated `Robot.dynamic_model_matrix_form()` alias;
- remove deprecated `is_SO3()`, `is_SE3()`, `isrot()`, and `ishtm()` predicates;
- review removal of the legacy `moro.plotting` module, which was explicitly retained through 0.6.1;
- evaluate named-argument compatibility for `RobotVisualizer.plot()` and `RobotVisualizer.animate()` as part of API-001.

No silent changes to mathematics, frames, array shapes, or default backends. Document removals and alternatives clearly.

## P1 — Complementary API improvements

### API-001 — Explicit visualizer parameters (approved for 0.7.0 planning)

Improve consistency between numerical robotics operations and visualizer entry points:

```python
viz.plot(configuration, parameters=params)
viz.animate(configurations, parameters=params)
```

- Add optional keyword-only `parameters` for constant symbolic model substitutions.
- Rename `num_vals` → `configuration` and `num_vals_list` → `configurations` at the public level, with a documented compatibility decision for existing named calls.
- Continue to support existing full substitution dictionaries for convenience and migration.
- Reject ambiguous duplicate substitutions rather than silently selecting precedence.
- Validate missing/nonfinite inputs; preserve the symbolic robot and source dictionaries.
- Review whether the public `evaluate_robot()` helper should also accept `parameters`.
- **Not included:** compiled FK optimization or persistent parameter binding to `Robot` / `RobotVisualizer`.

### API-002 — Inverse dynamics for sampled joint motion (candidate)

Proposed public operation:

```python
tau_history = inverse_dynamics_trajectory(
    robot,
    motion,
    parameters=params,
)
```

- Accept compatible `q`, `qd`, `qdd` histories with shapes `(N,dof)`, including `JointTrajectory`, `DynamicsSolution`, and structurally compatible objects.
- Return a NumPy history `(N,dof)`, with generalized forces applicable to revolute and prismatic joints.
- Prepare the numerical `M/C/G` model **once per call**, then reuse the existing prepared inverse-dynamics implementation for samples.
- Keep `moro.trajectory` independent of `moro.dynamics`; do not add dynamic methods to `JointTrajectory` or require a new shared base class.
- Keep `inverse_dynamics()` single-state for now instead of silently overloading it with batch semantics.
- Clarify any time-vector validation without treating `t` as mathematically necessary for evaluating `M(q)qdd + C(q,qd)qd + G(q)`.
- **Not included:** RNEA-specific logic, public compiled model objects, mandatory vectorization, actuator-limit feasibility, or a result dataclass.

## P1 — Workspace exploration track (MORO-API-003)

Keep `moro.workspace` (sampling/data) independent of `moro.visualization.workspace` (rendering). A sampled point cloud is **not** an exact reachable-space boundary. Retain point/configuration one-to-one correspondence.

| ID | Candidate | Initial direction |
| --- | --- | --- |
| **W-01** | Three.js workspace backend | Extend `plot_workspace()` to an interactive point-cloud viewer with camera conventions and fit behavior coherent with `RobotVisualizer`, retaining explicit Matplotlib support. |
| **W-02** | Cartesian plane sections | Introduce Python-side approximate slices near `xy/xz/yz` planes at an explicit level and thickness, suitable for 2D plots or Three.js exploration. |
| **W-03** | Alternative joint-space samplers | Add method selection, initially considering `random` (current default), `sobol`, `latin_hypercube`, and `grid`, with seed and sample-count contracts documented. |

Specific concerns:

- **W-01:** use efficient point-cloud buffers; scale consistently across units, support planar and spatial workspaces, and avoid mandatory JavaScript for numerical analysis.
- **W-02:** distinguish *orthogonal projection* from *section*. A section selected by `|z-z0| <= thickness/2` (analogously for other planes) is a **thick sample filter**, not a mathematically exact plane–workspace intersection. Report selected point count and avoid silently expanding thickness when sparse.
- **W-03:** low-discrepancy samples improve **joint-space** coverage; they do not ensure uniform Cartesian coverage or accurate frontier recovery. Explicitly define how Sobol handles non-power-of-two sample counts; avoid `grid` combinatorial explosions for high DOF.
- Keep reusable section data in Python so both Matplotlib and Three.js can render it; do not make the numerical APIs depend on browser interactions.
- Decide the `plot_workspace` default backend after reviewing backward compatibility; adding Three.js need not automatically switch the default in the same step.

## P2 — Preserve but do not commit to 0.7.0

| ID | Future idea | Why deferred |
| --- | --- | --- |
| **W-04** | Interactive point selection and configuration recovery | Requires browser-to-model interaction and a policy for accessing the robot instance. |
| **W-05** | Manipulability, condition-number, singularity, or joint-coordinate coloring | Needs optional metric evaluation and coherent color/scale semantics. |
| **W-06** | Nonconvex boundaries, area/volume estimates | Requires careful geometric/uncertainty contracts; convex hull can contain unreachable regions. |
| **W-07** | Adaptive workspace sampling and larger-scale optimization | Needs refinement objectives and benchmarks, not just a new sampling option. |
| **D-NE** | Recursive Newton–Euler (RNEA) | Currently planned for **0.10.0** in the rolling roadmap; no decision to advance it to 0.7.0. |
| **Later horizon** | Pose trajectories/SLERP, acceleration-level differential kinematics, control, URDF/ROS 2, predefined manipulators | Retain the separate roadmap horizon; do not expand 0.7.0 by implication. |

The rolling roadmap currently places **pose trajectories/SLERP in 0.8.0**, **second-order differential kinematics in 0.9.0**, and **RNEA/reusable numerical dynamics architecture in 0.10.0**. These placements are revisable, not guarantees.

## Suggested work packages

| Package | Contents | Proposed outcome |
| --- | --- | --- |
| **A — Trajectory design and implementation** | T-01, T-02, T-03, T-04; design limits T-05 separately | Transparent multi-point trajectories and one additional velocity profile. |
| **B — Cross-module interoperability** | API-001, API-002, T-06 | Direct notebook workflows from generated motion to visualization and inverse dynamics. |
| **C — Workspace numerical/visual exploration** | W-01, W-02, W-03 | Interactive sampled workspace, optional sections, and selectable sampling methods. |
| **D — Stabilization and release** | M-01, examples, tests, docs, migration notes, benchmarks when justified | Consistent and supportable 0.7.0 release. |

Packages are **planning groupings, not a committed implementation schedule**. Package C is independently movable to a later release if scope or reliability requires it.

## Cross-module contracts and architectural guardrails

- `moro.trajectory` remains robot-independent and does not import dynamics or visualization.
- Dynamic analysis consumes compatible sampled state arrays; avoid forcing inheritance between `JointTrajectory` and `DynamicsSolution`.
- Numerical workspace sampling does not depend on Three.js.
- Prefer module-level public functions and modest result objects over early general frameworks.
- Do not promise nonuniform physical playback until actual time semantics have been defined and tested.
- Support both fixed model parameters and changing joint states explicitly, without mutating `Robot`.
- Preserve 1-DOF, mixed R/P, degenerate/planar/3D, and symbolic-parameter regression cases.

## Proposed acceptance and scope-freeze gates

Before moving from inventory to committed scope:

1. Specify public APIs, input/output shapes, exceptions, and backward compatibility for T-01–T-06.
2. Estimate and test the numerical cost of each P1 candidate against the intended release size.
3. Prioritize the smallest coherent workspace subset rather than assuming all W-* items land together.
4. Make a deliberate decision on API removals and migration guidance.
5. Write measurable acceptance cases: single-/multi-DOF trajectories; continuity and limits; nonuniform time playback; numerical inverse dynamics equivalence; 2D/3D workspace sampling, slices and backends.
6. Identify at least one notebook-style teaching example per accepted functional area.
7. Split accepted candidates into detailed design documents and testable PR-sized increments; document anything postponed explicitly.

**Definition of done for accepted features:** validated numerical behavior and edge cases, regression tests, useful error messages, docs/API reference, examples, compatibility notes, and successful full tests/builds. A proof of concept alone is insufficient.

## Planning references

- Moro rolling roadmap (Obsidian, 2026-10-28): `30_Engineering/Software/moro/Roadmap.md`.
- Moro vision/principles (Obsidian): `30_Engineering/Software/moro/Vision.md`.
- Previous roadmap pattern: [0.5.0 release scope](../0.5.0/index.md).
- [Moro 0.6.1 release notes](../../source/release-notes/0.6.1.md), including planned 0.7.0 cleanup.
- Discussions consolidated on 2026-11-03: visualizer parameters (API-001); inverse dynamics of trajectories (API-002); workspace sampling and graphics (W-01–W-07).
