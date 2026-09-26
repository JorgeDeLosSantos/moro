# Moro 0.5.0 detailed design

This directory contains the implementation-level design contracts for the accepted Moro 0.5.0 feature set.

The roadmap in \`docs/roadmap/0.5.0/\` defines release scope, ordering, compatibility policy, and high-level completion criteria. The documents in this directory refine that roadmap into concrete public APIs, numerical conventions, invariants, validation behavior, internal architecture, tests, and examples.

## Status

Detailed design is complete for all accepted functional increments.

Implementation should now proceed against these documents. Scope or contract changes discovered during implementation should be recorded explicitly rather than introduced silently.

## Detailed-design documents

| Increment | Feature | Detailed design | Status |
|---|---|---|---|
| 0.5-A | Transformations and orientation foundations | [transformations.md](transformations.md) | Implemented |
| 0.5-B | Differential kinematics | [differential-kinematics.md](differential-kinematics.md) | Complete |
| 0.5-C | Singularities and manipulability | [singularities-manipulability.md](singularities-manipulability.md) | Complete |
| 0.5-D | Full-pose inverse kinematics | [inverse-kinematics.md](inverse-kinematics.md) | Complete |
| 0.5-E | Trajectory generation | [trajectory.md](trajectory.md) | Complete |
| 0.5-F | Numerical dynamics and simulation | [dynamics.md](dynamics.md) | Complete |
| 0.5-G | Workspace sampling | [workspace.md](workspace.md) | Complete |
| 0.5-H | Integration, documentation, and release hardening | Cross-cutting | Begins after functional increments |

## Implementation sequence

The preferred single-stream implementation order is:

\`\`\`text
0.5-A  transformations
   ↓
0.5-B  differential kinematics
   ↓
0.5-C  singularities / manipulability
   ↓
0.5-D  full-pose IK
   ↓
0.5-E  trajectory
   ↓
0.5-F  dynamics
   ↓
0.5-G  workspace
   ↓
0.5-H  integration / release hardening
\`\`\`

This order minimizes rework:

- full-pose IK depends on stable orientation primitives;
- singularity/manipulability analysis reuses the differential-kinematics numerical Jacobian/SVD conventions;
- trajectory is mathematically independent but benefits from stable IK and visualization conventions;
- numerical dynamics is a larger isolated architectural increment;
- workspace is comparatively independent and can remain near the end;
- final integration verifies conventions across all modules.

## Cross-cutting contracts

Several decisions span more than one feature and should remain consistent during implementation.

### Numerical array orientation

Time-series outputs use time-major arrays:

\`\`\`text
(N, dof)
\`\`\`

for joint trajectories and dynamic solutions.

Cartesian sampled/trajectory positions use:

\`\`\`text
(N, 3)
\`\`\`

including planar cases.

### Geometric twist ordering

Differential-kinematics and pose-IK work uses:

\`\`\`text
(vx, vy, vz, wx, wy, wz)
\`\`\`

as the canonical geometric-twist ordering.

### Symbolic versus numerical responsibilities

\`Robot\` remains the main symbolic model.

New numerical layers evaluate or simulate that model without replacing its symbolic construction:

- differential kinematics preserves symbolic Jacobian inspection;
- numerical IK evaluates symbolic robot geometry through supplied parameters;
- dynamics compiles symbolic \(M\), \(C\), and \(G\) for numerical use;
- workspace compiles symbolic forward kinematics for sampling.

### Visualization interoperability

Trajectory, dynamics, and workspace do not depend on visualization.

Visualization should instead accept their stable numerical representations where appropriate.

In particular, the accepted integration direction is to extend robot visualization so that numerical joint vectors/matrices may be supplied directly, while preserving the existing dictionary-based API.

### Root-package policy

New 0.5.0 functionality is primarily module-oriented.

Examples:

\`\`\`python
from moro.transformations import rot2rotvec
from moro.differential_kinematics import solve_velocity_ik
from moro.inverse_kinematics import solve_pose_ik
from moro.trajectory import joint_trajectory
from moro.dynamics import simulate
from moro.workspace import sample_workspace
\`\`\`

New helpers do not need automatic root-package re-export unless there is a specific compatibility or usability reason.

## Accepted compatibility changes

The most important intentional 0.5.0 compatibility changes are:

- \`Robot.dynamic_model()\` changes from per-joint Euler-Lagrange equations to the standard matrix-form dynamic model;
- the former behavior moves to \`Robot.euler_lagrange_equations()\`;
- \`Robot.dynamic_model_matrix_form()\` may remain temporarily as a deprecated alias;
- descriptive rotation/HTM predicates supersede legacy transformation predicate names while preserving legacy Boolean behavior during deprecation;
- SciPy becomes a required dependency because numerical dynamics and simulation are part of the main package.

These changes must be explicit in tests, documentation, CHANGELOG, and release notes.

## Implementation discipline

For each increment:

1. implement the accepted public contract;
2. add focused unit and regression tests;
3. add/update API and educational documentation;
4. run the complete test suite;
5. keep excluded functionality out of the increment;
6. integrate only after the increment is internally coherent.

If implementation evidence reveals that a detailed-design decision is impractical or mathematically incorrect, update the corresponding design document before normalizing the new behavior into the codebase.

## Planning state

The Moro 0.5.0 functional design phase is closed.

Implementation is underway. **0.5-A: transformations and orientation foundations** is complete; the next active increment is **0.5-B: differential kinematics**.
