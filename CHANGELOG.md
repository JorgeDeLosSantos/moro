# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.0] - 2026-10-01

### Added

* Expanded orientation support with all accepted proper-Euler and Tait-Bryan sequences, quaternion conversions, rotation-vector conversions, `vex()`, and descriptive rotation/HTM validation predicates.
* New differential-kinematics API with `task_jacobian()`, `cartesian_velocity()`, `solve_velocity_ik()`, and `VelocityIKSolution`.
* New singularity and local-capability helpers: `singular_values()`, `jacobian_rank()`, `condition_number()`, `is_singular()`, and `manipulability()`.
* Full-pose inverse kinematics through `solve_pose_ik()` and `PoseIKSolution`, while preserving the existing position-only IK workflows.
* New `moro.trajectory` module with linear, cubic, and quintic point-to-point trajectories, `JointTrajectory`, and `PositionTrajectory`.
* Numerical joint-vector and joint-matrix input support in `RobotVisualizer.plot()` and `RobotVisualizer.animate()`.
* New `moro.dynamics` module with `inverse_dynamics()`, `forward_dynamics()`, `state_derivative()`, `simulate()`, and `DynamicsSolution`.
* Numerical forward dynamics based on `numpy.linalg.solve()` and time-domain simulation based on `scipy.integrate.solve_ivp`.
* Support for zero, constant, time-varying, and state-feedback generalized-force inputs through the `tau` simulation interface.
* New `moro.workspace` module with `Workspace` and reproducible sampled reachable-workspace analysis through `sample_workspace()`.
* Matplotlib workspace visualization through `plot_workspace()` with automatic axis-aligned planar detection and explicit `xy`, `xz`, `yz`, and `3d` projections.
* New user-guide, theory, API-reference, and worked-example documentation for differential kinematics, singularity/manipulability analysis, full-pose IK, trajectory generation, numerical dynamics, and workspace sampling.

### Changed

* **Breaking:** `Robot.dynamic_model()` now returns the symbolic matrix manipulator equation `M(q) qdd + C(q, qd) qd + G(q) = tau`.
* The previous per-joint Euler-Lagrange result of `dynamic_model()` is now provided by `Robot.euler_lagrange_equations()`.
* `Robot.dynamic_model_matrix_form()` remains temporarily available as a deprecated alias of `dynamic_model()`.
* Legacy transformation predicates `is_SO3()`, `is_SE3()`, `isrot()`, and `ishtm()` are deprecated in favor of `is_rotation_matrix()` and `is_homogeneous_transform()` while preserving their historical Boolean behavior.
* New time-series APIs use time-major arrays with shape `(N, dof)`; Cartesian sampled/trajectory positions use shape `(N, 3)` even for planar cases.
* `RobotVisualizer` preserves mapping-based configuration input and additionally accepts numerical joint vectors/matrices ordered by `robot.qs`.
* Python support for Moro 0.5.x is Python 3.11 and newer.
* NumPy and SciPy are explicit required dependencies for the numerical APIs introduced in 0.5.0.

### Compatibility notes

* Users who relied on the Moro 0.4.x meaning of `Robot.dynamic_model()` should migrate to `Robot.euler_lagrange_equations()`.
* `Robot.dynamic_model_matrix_form()` is provided only as a temporary compatibility alias and emits `DeprecationWarning`.
* The legacy transformation predicates remain callable during the deprecation period and continue returning plain `True`/`False`; the new descriptive predicates may return `None` when symbolic membership is indeterminate.
* New 0.5.0 feature APIs are intentionally module-oriented and are not automatically re-exported from the package root.

## [0.4.0] - 2026-08-20

### Added

* New `inverse_kinematics` module for numerical position inverse kinematics.
* `solve_position_ik()` with Levenberg-Marquardt, Newton-Raphson, and Cyclic Coordinate Descent (CCD) methods.
* `IKSolution` result object with convergence status, iteration count, final error, residual, solver method, and diagnostic message.
* `solve_position_trajectory()` for sequential position inverse kinematics along user-defined Cartesian target sequences.
* `IKTrajectorySolution` for inspecting per-target solutions, errors, iterations, and failed-target information.
* Support for symbolic geometric parameters in inverse-kinematics problems through the `parameters` argument.
* Joint-limit handling, reproducible random initialization, and stagnation detection for numerical IK solvers.
* Interactive Three.js backend for robot visualization in Jupyter notebooks.
* Interactive robot animations using Three.js.
* Configurable visualization styles through the new `VisualizationStyle` class.
* End-effector trajectory visualization for robot animations.
* Predefined camera views (front, top, and isometric) in the Three.js viewer.
* Support for orthographic and perspective camera projections.
* `Robot.model_summary()` to inspect the modeling state and distinguish explicitly set parameters from automatically generated defaults.
* Public homogeneous transformation helpers: `rot2htm()`, `rt2htm()`, `htm2rot()`, `htm2tra()`, and `invhtm()`.
* Expanded automated test coverage for inverse kinematics, transformations, visualization, and robot-model validation.
* New structured documentation including Overview, Installation, Quick Start, User Guide, Examples, API Reference, Theory, and Development sections.
* New worked examples covering planar and spatial manipulators, numerical inverse kinematics, Cartesian IK trajectories, and symbolic dynamics.
* Contributor documentation and naming conventions for library development.

### Changed

* Refactored the visualization package into multiple modules with clearer responsibilities.
* Unified visualization configuration across Matplotlib and Three.js through `VisualizationStyle`.
* Improved Three.js animation performance by updating existing scene objects instead of recreating them every frame.
* Simplified the visualization API around `RobotVisualizer.plot()` and `RobotVisualizer.animate()`.
* HTML visualization templates are now packaged as library resources and loaded through `importlib.resources`.
* Improved validation of robot DH rows, joint types, joint limits, center-of-mass positions, inertia tensors, and gravity vectors.
* Joint types are now normalized case-insensitively to `"r"` or `"p"`.
* Documented the default-values policy for dynamic model parameters.
* `inertia_tensors = None` now generates symbolic diagonal inertia tensors as an explicit modeling convenience.
* Improved caching and invalidation of kinematic and dynamic quantities when model parameters change.
* `T_ij(i, j)` now uses the analytic inverse of homogeneous transformations when `i < j`, avoiding a general symbolic matrix inverse.
* Velocity-dependent dynamic methods now warn when static joint symbols are used instead of time-dependent generalized coordinates.
* Euler-angle conversion utilities now consistently support the six proper Euler sequences: `xyx`, `xzx`, `yxy`, `yzy`, `zxz`, and `zyz`.
* Improved validation and singular-case handling in Euler-angle and axis-angle conversion utilities.
* `axa2rot()` now uses Rodrigues' rotation formula.
* `skew()` and `axa2rot()` now accept consistent 3D vector input formats.
* `htmrot()` now reuses the public `rot2htm()` helper.
* Reorganized the Sphinx documentation into narrative MyST pages and focused RST API-reference pages.
* Documentation builds now use `docs/source` as the source tree and `docs/build/html` as generated output.

### Fixed

* Fixed frame orientation rendering in the Three.js backend.
* Fixed animation scaling so robot geometry remains visually consistent across frames.
* Fixed packaging of visualization templates for installations from GitHub and PyPI.
* Improved synchronization between robot data and rendered scenes during animations.
* Fixed stale cached center-of-mass positions and associated Jacobians after changing `cm_positions`.
* Invalid joint types now raise a clear `ValueError` instead of being silently interpreted incorrectly.
* Fixed initialization and storage behavior of `qis_range`.
* `cm_positions` no longer mutates caller-provided containers and now accepts tuples correctly.
* `axa2rot()` now rejects the zero vector with a clear `ValueError`.
* `axa2rot()` and `skew()` now reject invalid vector dimensions with descriptive errors.
* Improved inverse-kinematics handling of numerical failures, stagnation, joint-limit clipping, and result consistency.

---

## [0.3.0] - 2026-03-09

### Added

* Initial public release of Moro.
* Symbolic robot modeling based on Denavit–Hartenberg parameters.
* Forward kinematics computation.
* Robot visualization using Matplotlib.
* Support for numerical evaluation of symbolic robot models.
