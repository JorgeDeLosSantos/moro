# Moro 0.5.0 roadmap

This directory records the accepted preliminary scope and design decisions for Moro 0.5.0. The documents define what belongs in the release and the main public-API direction; implementation-level details remain intentionally deferred to later design work.

## Release direction

Moro 0.5.0 is planned as a coherent expansion from symbolic kinematics/dynamics and position inverse kinematics toward a broader educational workflow for motion, local kinematic capability, full-pose inverse kinematics, trajectory generation, workspace exploration, and numerical simulation of serial manipulators.

The accepted feature areas are:

- [Differential kinematics](differential-kinematics.md)
- [Singularities and manipulability](singularities-manipulability.md)
- [Full-pose inverse kinematics](inverse-kinematics.md)
- [Transformations and orientation representations](transformations.md)
- [Trajectory generation](trajectory.md)
- [Direct dynamics and numerical simulation](dynamics.md)
- [Workspace sampling](workspace.md)

## Dependency graph

The main implementation dependencies are intentionally limited:

```text
transformations
      |
      +------> full-pose IK

core Jacobian
      |
      v
differential kinematics
      |
      v
singularities / manipulability

symbolic dynamics on Robot
      |
      v
numerical inverse / forward dynamics
      |
      v
state derivative
      |
      v
simulation

trajectory generation          Robot + finite joint limits
      |                                  |
      v                                  v
position IK / visualization          workspace
```

These arrows represent implementation dependencies or especially strong reuse relationships, not mandatory package coupling.

Some relationships are deliberately weaker:

- full-pose IK and differential kinematics should share the same geometric-Jacobian conventions, but full-pose IK does not require the differential-kinematics module to be complete;
- trajectory generation is mathematically independent from `Robot` and may later feed position IK or visualization;
- workspace sampling depends on robot forward kinematics and finite joint limits, but not on manipulability analysis;
- dynamics results should be easy to visualize without making `moro.dynamics` depend directly on the visualization subsystem.

## Recommended implementation sequence

The preferred implementation order for Moro 0.5.0 is the following.

### 1. Transformations and orientation foundations

Complete the `transformations.py` audit first:

- Tait-Bryan support in `eul2rot()` / `rot2eul()`;
- quaternion conversions with scalar-first convention;
- rotation-vector conversions;
- `vex()`;
- rotation and homogeneous-transform validation;
- small API symmetry improvements such as `axa2rot(..., deg=False)`.

This increment establishes the orientation primitives and validation rules required by full-pose IK while remaining independently useful and testable.

### 2. Differential kinematics

Implement the task-space conventions and main velocity-level API:

- `task_jacobian()`;
- `cartesian_velocity()`;
- `solve_velocity_ik()`;
- pseudoinverse and damped least squares;
- `VelocityIKSolution`;
- optional joint-velocity limits.

This creates the reusable task-Jacobian layer needed by the next increment.

### 3. Singularities and manipulability

Build the numerical analysis API on the task-Jacobian conventions introduced in the previous increment:

- singular values;
- numerical Jacobian rank;
- singularity detection;
- condition number;
- Yoshikawa manipulability.

Keeping this immediately after differential kinematics minimizes duplicated numerical-Jacobian logic and allows singular/near-singular behavior to be tested together with velocity IK.

### 4. Full-pose inverse kinematics

Extend `inverse_kinematics.py` with `solve_pose()` and `PoseIKSolution` after the rotation-vector and HTM-validation primitives are stable.

The main integration point to verify is mathematical compatibility between the chosen orientation-error convention and the geometric Jacobian used by the solver.

This increment should preserve position-only IK and keep CCD position-only.

### 5. Trajectory generation

Introduce `moro/trajectory.py` with independent point-to-point trajectory generation:

- linear;
- cubic;
- quintic;
- `JointTrajectory`;
- `PositionTrajectory`.

This block has few dependencies and can be implemented without changing robot modeling. Its placement after the kinematic work makes it straightforward to demonstrate reuse with existing position-trajectory IK and visualization.

### 6. Numerical dynamics and simulation

Implement the numerical dynamics layer after reviewing the symbolic dynamics API and its planned naming changes:

1. apply the accepted `dynamic_model()` / `euler_lagrange_equations()` API transition;
2. add SciPy as a required dependency;
3. implement `inverse_dynamics()`;
4. implement `forward_dynamics()`;
5. implement `state_derivative()`;
6. implement `simulate()` and `DynamicsSolution`.

Numerical compilation/lambdification should be addressed early in this increment so that ODE integration does not rely on repeated symbolic substitution.

### 7. Workspace sampling

Implement the sampled reachable-workspace feature after the central kinematics/dynamics work:

- finite joint-limit validation;
- random joint-space sampling;
- reproducible `seed` behavior;
- efficient numerical FK evaluation;
- `Workspace` result object;
- visualization integration or helper.

Workspace remains intentionally independent from manipulability/singularity maps.

### 8. Integration, documentation, and release hardening

After all accepted functional increments are implemented:

- review public exports and module-level API consistency;
- complete deprecations and compatibility notes;
- ensure cross-module array conventions are consistent;
- add and refine documentation and teaching examples;
- verify visualization interoperability;
- update the changelog;
- run the complete test suite and documentation build;
- perform release-oriented regression testing against 0.4.0 behavior.

## Why this order

The sequence is intended to minimize rework rather than imply equal priority among features.

The first four increments form a coherent kinematic chain:

```text
orientation primitives
        -> task Jacobians
        -> local capability analysis
        -> full-pose IK
```

Trajectory generation is mostly independent and can then consume already-stable kinematic results. Numerical dynamics is a larger architectural increment that is easier to isolate after the kinematic API settles. Workspace is the most independent accepted feature and is therefore deliberately placed near the end.

Parallel development remains possible where dependencies do not overlap. In particular, trajectory generation and workspace could be implemented independently of most full-pose IK or dynamics work, but the sequence above is preferred for a single main development stream.

## Compatibility and deprecation policy

Moro 0.5.0 is allowed to make a small number of explicit pre-1.0 API corrections, but silent semantic changes should be avoided except where the change is intentional, documented, and judged preferable to carrying a known naming problem forward.

### Dynamic-model API

The accepted 0.5.0 behavior is:

```text
Robot.euler_lagrange_equations()
    -> Euler-Lagrange equations

Robot.dynamic_model()
    -> standard matrix-form dynamic model

Robot.dynamic_model_matrix_form()
    -> deprecated alias of Robot.dynamic_model()
```

This means `Robot.dynamic_model()` changes meaning relative to Moro 0.4.x. The change is intentionally accepted as a breaking change for 0.5.0 because retaining or deprecating the same name before reusing it with a different meaning would create a more confusing transition.

Release notes and the changelog must state clearly that users who relied on the 0.4.x behavior of `dynamic_model()` should migrate to `euler_lagrange_equations()`.

### Rotation and homogeneous-transform predicates

The preferred public API becomes:

```python
is_rotation_matrix(R, *, tol=1e-9)
is_homogeneous_transform(T, *, tol=1e-9)
```

These new predicates belong conceptually in `moro.transformations` and use the accepted symbolic ternary semantics `True | False | None`.

The existing names

```text
is_SO3()
is_SE3()
isrot()
ishtm()
```

are considered redundant once the descriptive predicates exist. They should be deprecated in 0.5.0 and removed in a later release.

During the deprecation period, legacy wrappers should preserve their historical Boolean contract rather than exposing the new ternary behavior directly. An indeterminate symbolic result should therefore remain `False` through the legacy wrappers.

### Root-package exports

Existing root-level imports should remain compatible where practical, but new feature APIs do not need to be re-exported automatically from `moro`.

Preferred usage for new capabilities is module-oriented, for example:

```python
from moro.transformations import rot2quat, rot2rotvec
from moro.inverse_kinematics import solve_pose
from moro.differential_kinematics import solve_velocity_ik
from moro.trajectory import joint_trajectory
from moro.dynamics import simulate
```

This keeps the package root compact and avoids accumulating every public helper in `moro.__init__`.

### SciPy dependency

SciPy becomes a required dependency in Moro 0.5.0 because numerical simulation is part of the main library rather than an optional extra. This installation-footprint change should be documented in the changelog and release notes.

## Scope discipline

The accepted 0.5.0 scope intentionally excludes several natural follow-on capabilities so that the release remains focused. In particular, the release does not aim to introduce general motion planning, collision-aware IK, null-space secondary objectives, pose trajectories, SLERP, screw-theory/SE(3) utilities as a general subsystem, constrained/contact dynamics, controller classes, advanced trajectory profiles, or exact analytical workspace computation.

If schedule pressure requires prioritization, workspace is the most independent feature and could be deferred without disrupting the central kinematics/dynamics narrative. Without a fixed release deadline, it remains part of the accepted candidate scope.

## Planning status

The feature scope, implementation order, and compatibility/deprecation policy are now defined. The next planning steps are:

1. define completion criteria for each feature increment;
2. begin detailed design and implementation in the agreed sequence.

Detailed numerical tolerances, dataclass invariants, private helper structure, caching strategies, and exact exception messages remain deferred to detailed design.
