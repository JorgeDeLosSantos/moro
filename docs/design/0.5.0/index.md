# Moro 0.5.0 detailed design

This directory contains the implementation-level design contracts for the accepted Moro 0.5.0 feature set.

The roadmap in `docs/roadmap/0.5.0/` defines release scope, ordering, compatibility policy, and high-level completion criteria. The documents in this directory refine that roadmap into concrete public APIs, numerical conventions, invariants, validation behavior, internal architecture, tests, and examples.

## Status

Detailed design and implementation are complete for all accepted Moro 0.5.0 increments.

The 0.5.x development line has completed integration, documentation, compatibility review, packaging smoke tests, and release hardening. The next phase is final release preparation.

## Detailed-design documents

| Increment | Feature | Detailed design | Status |
|---|---|---|---|
| 0.5-A | Transformations and orientation foundations | [transformations.md](transformations.md) | Implemented |
| 0.5-B | Differential kinematics | [differential-kinematics.md](differential-kinematics.md) | Implemented |
| 0.5-C | Singularities and manipulability | [singularities-manipulability.md](singularities-manipulability.md) | Implemented |
| 0.5-D | Full-pose inverse kinematics | [inverse-kinematics.md](inverse-kinematics.md) | Implemented |
| 0.5-E | Trajectory generation | [trajectory.md](trajectory.md) | Implemented |
| 0.5-F | Numerical dynamics and simulation | [dynamics.md](dynamics.md) | Implemented |
| 0.5-G | Workspace sampling | [workspace.md](workspace.md) | Implemented |
| 0.5-H | Integration, documentation, and release hardening | Cross-cutting | Implemented |

## Implementation sequence

The completed implementation order is:

```text
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
```

This order minimized rework:

- full-pose IK depended on stable orientation primitives;
- singularity/manipulability analysis reused the differential-kinematics numerical Jacobian/SVD conventions;
- trajectory remained mathematically independent while benefiting from stable IK and visualization conventions;
- numerical dynamics remained an isolated architectural increment;
- workspace stayed comparatively independent;
- final integration verified conventions across all modules.

## Cross-cutting contracts

Several decisions span more than one feature and remain release contracts for 0.5.0.

### Numerical array orientation

Time-series outputs use time-major arrays:

```text
(N, dof)
```

for joint trajectories and dynamic solutions.

Cartesian sampled/trajectory positions use:

```text
(N, 3)
```

including planar cases.

### Geometric twist ordering

Differential-kinematics and pose-IK work uses:

```text
(vx, vy, vz, wx, wy, wz)
```

as the canonical geometric-twist ordering.

### Symbolic versus numerical responsibilities

`Robot` remains the main symbolic model.

Numerical layers evaluate or simulate that model without replacing its symbolic construction:

- differential kinematics preserves symbolic Jacobian inspection;
- numerical IK evaluates symbolic robot geometry through supplied parameters;
- dynamics compiles symbolic \(M\), \(C\), and \(G\) for numerical use;
- workspace compiles symbolic forward kinematics for sampling.

### Visualization interoperability

Trajectory, dynamics, and workspace do not depend on visualization.

Visualization accepts their stable numerical representations where appropriate, including numerical joint vectors and time-major joint matrices, while preserving the existing dictionary-based API.

### Root-package policy

New 0.5.0 functionality is primarily module-oriented.

Examples:

```python
from moro.transformations import rot2rotvec
from moro.differential_kinematics import solve_velocity_ik
from moro.inverse_kinematics import solve_pose_ik
from moro.trajectory import joint_trajectory
from moro.dynamics import simulate
from moro.workspace import sample_workspace
```

New helpers are not automatically re-exported from the package root.

## Accepted compatibility changes

The most important intentional 0.5.0 compatibility changes are:

- `Robot.dynamic_model()` changes from per-joint Euler-Lagrange equations to the standard matrix-form dynamic model;
- the former behavior moves to `Robot.euler_lagrange_equations()`;
- `Robot.dynamic_model_matrix_form()` remains temporarily as a deprecated alias;
- descriptive rotation/HTM predicates supersede legacy transformation predicate names while preserving legacy Boolean behavior during deprecation;
- NumPy and SciPy are direct runtime dependencies because numerical analysis and simulation are first-class package capabilities.

These changes are explicit in tests, documentation, CHANGELOG, and release notes.

## Release-hardening evidence

0.5-H completed the cross-cutting definition of done:

- public imports and `__all__` declarations reviewed;
- no unintended root-package API expansion;
- accepted deprecations covered by regression tests;
- `dynamic_model()` breaking change documented prominently;
- runtime dependencies synchronized with packaging metadata;
- time-major and Cartesian shape conventions covered across modules;
- representative 0.4.x workflows retained through regression tests;
- wheel and sdist build successfully;
- clean wheel installation and packaged-resource smoke test pass;
- Sphinx builds successfully with warnings treated as errors;
- complete test suite passes on Python 3.11, 3.12, 3.13, and 3.14.

Final hardening validation recorded **872 passed, 2 skipped**.

## Planning state

All accepted Moro 0.5.0 implementation increments **0.5-A through 0.5-H are complete**.

The development line remains versioned as:

```text
0.5.0.dev0
```

until the dedicated release-preparation step updates release metadata, dates the changelog, revalidates the distribution, integrates the release line to `master`, and creates the final release tag.
