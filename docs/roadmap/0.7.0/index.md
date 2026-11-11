# Moro 0.7.0 — Frozen feature scope and stabilization plan

> **Status:** Scope frozen (S1), M-01 cleanup merged (S2), integration and Colab checks accepted (S3); S4 documentation consolidation underway. Moro 0.7.0 remains **unreleased**.  
> **Baseline:** `develop` after PR #62 (merge `0c1b11b`), with no outstanding feature PRs when S1 began.  
> **Goal:** stabilize, document and release the accepted functionality without adding new features.  
> **Previous planning:** the 2026-11-03 candidate inventory and the rolling Moro roadmap are preserved in the design discussions and the historical notes below.

## Release direction

**Working theme:** Advanced Trajectories & Numerical Exploration.

**Current status:** The functional scope described below is implemented in
`develop`. The core trajectory changes (T-01–T-04) were integrated via
PRs #40–#47; physical-time playback (T-06) via PRs #49–#51;
visualization API (API-001) via PRs #52–#54; inverse dynamics over
trajectories (API-002) via PRs #55–#57; and workspace work
(API-003/D1–D5) via PRs #58–#62. **Implementation complete does not
mean release-ready:** compatibility cleanup, final cross-module
validation, docs/migration review, and version/release checks remain.

The primary goal inherited from the rolling roadmap is to extend `moro.trajectory` from point-to-point interpolation to **multi-waypoint trajectories and motion profiles**. Complementary improvements emerged from practical use: more consistent visualization parameters, reusable inverse dynamics over sampled motion, and improved sampled-workspace exploration.

Preserve the project principles:

- educational clarity and inspectable mathematics first;
- simple notebook workflows on Windows, macOS, Linux, Jupyter, and Colab;
- `Robot` as the symbolic model, with numerical algorithms in focused modules;
- interoperability through stable NumPy shapes, rather than unnecessary coupling or new class hierarchies;
- explicit conventions, useful diagnostics, regression tests, and reproducible examples;
- avoid speculative abstractions and measure performance before deeper optimization.

## S1 — Accepted scope and release boundary (frozen 2026-11-11)

| Track | Accepted functionality | State in `develop` |
| --- | --- | --- |
| **Trajectories** | T-01 multi-waypoint joint/Cartesian-position motion; T-02 piecewise polynomials; T-03 join continuity; T-04 rest-to-rest triangular/trapezoidal profiles | Implemented; S3 integration regressions passed |
| **Playback** | T-06 Three.js physical-time playback with timestamps and speed control | Implemented; S3 integration regressions passed |
| **API-001** | Separate joint `configuration(s)` from constant model `parameters` in `RobotVisualizer`; direct keyword rename | Implemented; migration documented in S4 |
| **API-002** | `inverse_dynamics_trajectory()` for compatible sampled joint motion | Implemented; S3 forward/inverse dynamics checks passed |
| **API-003** | W-03 random/Sobol/LHS/grid sampling; W-02 numerical Cartesian sections; W-01 Three.js workspace/section views, Matplotlib alternative, optional grids | Implemented; Colab visual validation reported as correct in S3 |
| **M-01** | Remove scheduled 0.7.0 deprecated interfaces and update migration/tests | Integrated (PR #64); S3 regressions passed |

**Scope freeze:** No further feature development is planned for 0.7.0.
Only bug fixes, compatibility cleanup already scheduled in M-01,
regressions, integration examples, documentation, packaging and release
work are in scope. If validation reveals a material defect, fix and
retest it within the release candidate; do not use it to introduce
unrelated features.

**Explicitly deferred:** T-05 (basic velocity/acceleration-limit
constraints) is **not part of Moro 0.7.0**. Reconsider it as a separately
designed feature in a **future version not yet assigned**; do not assume
0.8.0 automatically. This avoids silently committing to time-allocation,
limit enforcement or trajectory-feasibility policies. The P2 ideas below,
including friction/external forces and new workspace metrics, also
remain outside 0.7.0.

**Release gate:** The version stays unreleased until S2–S5 below pass.
Passing the functional PR CI is necessary but not sufficient for a
release candidate.

## Historical planning priorities and design context

| Level | Meaning |
| --- | --- |
| **P0 — release core** | Original candidate ranking; T-01–T-04 and T-06 are now accepted and implemented. |
| **P1 — complementary candidates** | Original ranking; API-001/002/003 were accepted, while T-05 was explicitly deferred. |
| **P2 — future radar** | Preserved ideas outside the frozen 0.7.0 scope. |

These labels are historical planning classifications, **not** a current to-do list. The S1 accepted-scope table above is authoritative.

## Accepted primary track: advanced trajectories

The following item IDs are local to this roadmap.

| ID | Accepted item | Implemented intent |
| --- | --- | --- |
| **T-01** | Multi-waypoint joint and Cartesian-position trajectories | Prescribe ordered waypoints, associated with a clearly defined time convention. |
| **T-02** | Piecewise polynomial interpolation | Extend existing `linear`, `cubic`, `quintic` capabilities to multiple segments where mathematically appropriate. |
| **T-03** | Continuity across segments | Specify and validate position/velocity/acceleration continuity (`C0`, `C1`, `C2`) as appropriate to interpolation method and boundary conditions. |
| **T-04** | Trapezoidal velocity profile | [Designed](../../design/0.7.0/trapezoidal.md) and implemented (B1/B2): synchronized rest-to-rest joint motion, triangular/trapezoidal phases, optional prescribed duration and typed profile metadata; B3 (PR #47) added examples and validation. |
| **T-06** | Physical-time playback | **Completed** in PRs #49–#51: timestamped Three.js playback, controls, validated examples, backend limitations and visual Colab checks. |


**T-05 — Basic kinematic limits is deferred beyond 0.7.0.** Velocity/acceleration constraints need a separate design of time allocation and feasibility policy. No release version has been assigned.

### Historical design questions for T-01–T-06

These questions guided the already integrated PRs; they are **not open gates** for 0.7.0. Consult the implemented API and corresponding design documents when reviewing regression cases.

1. Extend existing `joint_trajectory()` / `position_trajectory()` or add explicit multi-waypoint functions? Preserve simple point-to-point entry points.
2. Define input shape and meaning for waypoints, times, boundary derivatives, and intermediate derivatives.
3. Distinguish geometric paths from time-parameterized trajectories; specify when `C0/C1/C2` refers to continuity in physical time.
4. Decide how interpolation modes behave at joins and whether a requested set of constraints is feasible.
5. Establish a trapezoidal-profile API without promising advanced jerk-limited or time-optimal planning.
6. Decide whether visualizers accept a trajectory object and/or a separate time array, preserving existing array-based animation calls and backend return types.
7. Preserve public output conventions: `t: (N,)`, `q/qd/qdd: (N,dof)`, `p/v/a: (N,3)`.
8. Ensure diagnostics and didactic examples expose the effect of methods, timing, and boundary conditions.

### M-01: completed compatibility cleanup (S2, PR #64)

Release notes for 0.6.1 scheduled this cleanup for 0.7.0. S2 audited imports, tests and replacements and completed the following removals:

- removed deprecated `Robot.dynamic_model_matrix_form()` alias;
- removed deprecated `is_SO3()`, `is_SE3()`, `isrot()`, and `ishtm()` predicates;
- removed the legacy `moro.plotting` module, retained through 0.6.1;
- **API-001 decision implemented:** directly rename the first named arguments to `configuration` and `configurations`; do not retain `num_vals`/`num_vals_list` aliases. Positional calls remain valid. See the [migration preview](../../source/release-notes/0.7.0.md).

The removals and replacement paths are recorded in the [0.7.0 migration notes](../../source/release-notes/0.7.0.md); no mathematical conventions, frame semantics, or established 0.7.0 backend defaults changed in S2.

## Accepted cross-module API improvements

### API-001 — Explicit visualizer parameters (PRs #52–#54 integrated)

The integrated change aligns numerical robotics operations and visualizer entry points:

```python
viz.plot(configuration, parameters=params)
viz.animate(configurations, parameters=params)
```

- Add optional keyword-only `parameters` for constant symbolic model substitutions.
- Renamed `num_vals` → `configuration` and `num_vals_list` → `configurations` directly; no legacy named aliases. Existing positional calls are preserved.
- Continue to support existing full substitution dictionaries for convenience and migration.
- Reject ambiguous duplicate substitutions rather than silently selecting precedence.
- Validate missing/nonfinite inputs; preserve the symbolic robot and source dictionaries.
- Keep `evaluate_robot(robot, substitutions)` unchanged as a lower-level evaluation interface.
- **Not included:** compiled FK optimization or persistent parameter binding to `Robot` / `RobotVisualizer`.

### API-002 — Inverse dynamics for sampled joint motion (PRs #55–#57 integrated)

Implemented public operation:

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

**Implementation:** B1 (PR #55) added the function and its input
contract. B2 (PR #56) validated R/RR/RP, symbolic parameters,
single-state and forward-dynamics equivalence, and one-time model
preparation. B3 (PR #57) adds the
[RR quintic/trapezoidal notebook example](../../source/examples/inverse-dynamics-trajectories.md),
the [Dynamics User Guide](../../source/user-guide/dynamics.md),
and the [unreleased 0.7.0 feature notes](../../source/release-notes/0.7.0.md).
Friction and externally applied forces are deferred to a separately
designed extension shared with forward dynamics and simulation.

## Accepted workspace exploration (MORO-API-003; PRs #58–#62 integrated)

Keep `moro.workspace` (sampling/data) independent of `moro.visualization.workspace` (rendering). A sampled point cloud is **not** an exact reachable-space boundary. Retain point/configuration one-to-one correspondence.

| ID | Feature | Integrated behavior |
| --- | --- | --- |
| **W-01** | Three.js workspace backend | D5 (PR #62): `plot_workspace()` and `plot_workspace_section()` default to Three.js; explicit `backend="matplotlib"` preserves `(fig, ax)`. Optional `show_grid=False`; shared camera presets/Fit View. |
| **W-02** | Cartesian plane sections | D3–D4 (PRs #60–#61): `section_workspace()` selects observed points near XY/XZ/YZ planes, preserves 3D data/configuration rows; visualizations support Matplotlib and Three.js. |
| **W-03** | Alternative joint-space samplers | D2 (PR #59): `random` (default), `sobol`, `latin_hypercube`, and `grid` with explicit seed/count/shape rules and tests. |

Specific concerns:

- **W-01:** uses one Three.js point-cloud position buffer, axis-aligned display projections, shared camera helpers and explicit Matplotlib support. The user validated the core camera/preset/navigation behavior in Jupyter/Colab before the final section/grid refinements; the user confirmed successful Colab visual checks for S3. Numerical analysis remains browser-independent.
- **W-02:** distinguish *orthogonal projection* from *section*. A section selected by `|z-z0| <= thickness/2` (analogously for other planes) is a **thick sample filter**, not a mathematically exact plane–workspace intersection. Report selected point count and avoid silently expanding thickness when sparse.
- **W-03:** low-discrepancy samples improve **joint-space** coverage; they do not ensure uniform Cartesian coverage or accurate frontier recovery. Explicitly define how Sobol handles non-power-of-two sample counts; avoid `grid` combinatorial explosions for high DOF.
- Keep reusable section data in Python so both Matplotlib and Three.js can render it; do not make the numerical APIs depend on browser interactions.
- **Decision for 0.7.0:** `plot_workspace` defaults to Three.js (an intentional breaking change); explicit Matplotlib remains available and old `(fig, ax)` consumers must request `backend='matplotlib'`.

## Deferred features — outside Moro 0.7.0

| ID | Future idea | Why deferred |
| --- | --- | --- |
| **T-05** | Basic joint velocity/acceleration limits | Explicitly deferred by the S1 decision; requires a separately scoped feasibility and time-allocation design. No destination release assigned. |
| **W-04** | Interactive point selection and configuration recovery | Requires browser-to-model interaction and a policy for accessing the robot instance. |
| **W-05** | Manipulability, condition-number, singularity, or joint-coordinate coloring | Needs optional metric evaluation and coherent color/scale semantics. |
| **W-06** | Nonconvex boundaries, area/volume estimates | Requires careful geometric/uncertainty contracts; convex hull can contain unreachable regions. |
| **W-07** | Adaptive workspace sampling and larger-scale optimization | Needs refinement objectives and benchmarks, not just a new sampling option. |
| **Friction / external forces** | Dynamic friction and externally applied wrenches in inverse/forward dynamics | Deferred to a separately designed extension with consistent interfaces across dynamics and simulation. |
| **D-NE** | Recursive Newton–Euler (RNEA) | Currently planned for **0.10.0** in the rolling roadmap; no decision to advance it to 0.7.0. |
| **Later horizon** | Pose trajectories/SLERP, acceleration-level differential kinematics, control, URDF/ROS 2, predefined manipulators | Retain the separate roadmap horizon; do not expand 0.7.0 by implication. |

The rolling roadmap currently places **pose trajectories/SLERP in 0.8.0**, **second-order differential kinematics in 0.9.0**, and **RNEA/reusable numerical dynamics architecture in 0.10.0**. These placements are revisable, not guarantees.

## Work packages and present status

| Package | Contents | Proposed outcome |
| --- | --- | --- |
| **A — Trajectory design and implementation** | T-01–T-04 | Implemented; T-05 is deferred. |
| **B — Cross-module interoperability** | API-001, API-002, T-06 | Implemented and accepted in S3. |
| **C — Workspace numerical/visual exploration** | W-01–W-03 | Implemented; Colab visual acceptance reported in S3. |
| **D — Stabilization and release** | S1–S5, including M-01 | **Current work package**; 0.7.0 remains unreleased. |

Packages A–C are now implemented. Package D defines the required exit criteria; optional features cannot be added under the name of stabilization.

## Cross-module contracts and architectural guardrails

- `moro.trajectory` remains robot-independent and does not import dynamics or visualization.
- Dynamic analysis consumes compatible sampled state arrays; avoid forcing inheritance between `JointTrajectory` and `DynamicsSolution`.
- Numerical workspace sampling does not depend on Three.js.
- Prefer module-level public functions and modest result objects over early general frameworks.
- Physical-time playback remains Three.js-only and follows validated timestamp semantics; do not silently approximate irregular timing in Matplotlib.
- Support both fixed model parameters and changing joint states explicitly, without mutating `Robot`.
- Preserve 1-DOF, mixed R/P, degenerate/planar/3D, and symbolic-parameter regression cases.

S3 is complete: CI passed and the user confirmed successful Colab visualization. The [S3 integration and visual acceptance record](s3-acceptance.md) distinguishes automated verification from the reported manual checks.

## S1–S5 — Stabilization and release gates

| Stage | Work | Exit criteria |
| --- | --- | --- |
| **S1 — Scope freeze** | Publish this accepted inventory, explicit deferrals, release sequence and compatibility boundary. | Reviewed PR merged into `develop`; no ambiguity about T-05 or feature additions. |
| **S2 — M-01 cleanup** | Audit internal/public uses of deprecated dynamic-model alias, transformation predicates and `moro.plotting`; remove scheduled obsolete APIs and update tests/docs. | Replacements documented; public imports and package tests pass; no unintended numerical changes. |
| **S3 — Integration and acceptance** | Run full Python 3.11–3.14 tests/builds, installed wheel/sdist smoke tests, complete notebook workflows for trajectories → Three.js playback → inverse dynamics, and confirm workspace/section/grid visual behavior in Jupyter/Colab. | Known regressions resolved; required visual checks explicitly confirmed or recorded as a release blocker. |
| **S4 — Documentation and migration** | Consolidate `CHANGELOG.md`, release notes, API guide and complete migration table (renamed visualization keywords, removed deprecated interfaces, Three.js workspace/section defaults). | Examples execute; docs build; no misleading "planned" wording for implemented features. |
| **S5 — Release preparation** | Synchronize `pyproject.toml` and `moro/version.py` to `0.7.0`, validate packages, prepare reviewed release candidate, then follow maintainer runbook. | CI and docs green on release candidate; release notes finalized and explicit maintainer authorization before `master` merge, tag, GitHub Release or PyPI publication. |

**Definition of done:** accepted features behave as documented; numerical
and cross-module regressions pass; backend and installation contracts
are tested; migration notes are sufficient for 0.6.x users; and the
release checklist is complete. Manual browser checks should not be
misrepresented as automatically proven by CI.

**No automatic publication:** freezing scope and merging documentation
do not create a tag, GitHub Release or PyPI distribution. Release
publication remains a separate explicit maintainer action.

## Planning references

- Moro rolling roadmap (Obsidian, 2026-10-28): `30_Engineering/Software/moro/Roadmap.md`.
- Moro vision/principles (Obsidian): `30_Engineering/Software/moro/Vision.md`.
- Previous roadmap pattern: [0.5.0 release scope](../0.5.0/index.md).
- [Moro 0.6.1 release notes](../../source/release-notes/0.6.1.md), including planned 0.7.0 cleanup.
- Discussions consolidated on 2026-11-03: visualizer parameters (API-001); inverse dynamics of trajectories (API-002); workspace sampling and graphics (W-01–W-07).
