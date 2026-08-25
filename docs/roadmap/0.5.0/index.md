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

## Cross-feature relationships

The main dependencies are conceptual rather than hard package coupling:

```text
transformations
      ↓
full-pose IK
      ↓
differential kinematics
      ↓
singularities / manipulability

trajectory generation
      ↓
position IK / visualization

symbolic dynamics
      ↓
forward dynamics
      ↓
simulation

Robot + joint limits
      ↓
workspace
```

The modules should remain separable where possible. In particular, trajectory generation should not require `Robot`, the dynamics module should not depend directly on visualization, and workspace sampling should not automatically compute manipulability or singularity metrics.

## Scope discipline

The accepted 0.5.0 scope intentionally excludes several natural follow-on capabilities so that the release remains focused. In particular, the release does not aim to introduce general motion planning, collision-aware IK, null-space secondary objectives, pose trajectories, SLERP, screw-theory/SE(3) utilities as a general subsystem, constrained/contact dynamics, controller classes, advanced trajectory profiles, or exact analytical workspace computation.

If schedule pressure requires prioritization, workspace is the most independent feature and could be deferred without disrupting the central kinematics/dynamics narrative. Without a fixed release deadline, it remains part of the accepted candidate scope.

## Planning status

The feature scope is considered mature enough to move next into dependency ordering, implementation sequencing, compatibility/deprecation review, and per-feature completion criteria. Detailed numerical tolerances, dataclass invariants, private helper structure, caching strategies, and exact exception messages remain deferred to detailed design.
