# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

* Interactive Three.js backend for robot visualization in Jupyter notebooks.
* Interactive robot animations using Three.js.
* Configurable visualization styles through the new `VisualizationStyle` class.
* End-effector trajectory visualization for robot animations.
* Predefined camera views (front, top, and isometric) in the Three.js viewer.
* Support for orthographic and perspective camera projections.
* Shared JavaScript infrastructure for Three.js templates.
* Comprehensive automated test suite for the visualization module.
* `Robot.model_summary()` to inspect the modeling state and distinguish explicitly-set parameters from auto-generated defaults.

### Changed

* Refactored the visualization package into multiple modules with clear responsibilities.
* Unified visualization configuration across Matplotlib and Three.js backends through `VisualizationStyle`.
* Improved Three.js animation performance by updating existing scene objects instead of recreating them every frame.
* Simplified the visualization API by removing notebook-specific rendering methods in favor of a unified interface.
* HTML templates are now packaged as library resources and loaded through `importlib.resources`.
* Documented the default-values policy: intrinsic link parameters (`masses`, `inertia_tensors`) may auto-provide symbolic placeholders, while problem/environment configuration (`cm_positions`, `gravity`) always requires explicit values.
* `inertia_tensors = None` now auto-generates diagonal symbolic tensors instead of raising an obscure TypeError; the helper was renamed to the internal `_generate_diagonal_inertia_tensors()` (stores the tensors directly, with no return value).

### Fixed

* Fixed frame orientation rendering in the Three.js backend.
* Fixed animation scaling to remain consistent across all frames.
* Fixed packaging of visualization templates for installation from GitHub and PyPI.
* Improved synchronization between robot data and rendered scene during animations.
* Fixed a stale cache for `r_cm` (and the `J_cm`/`Jv_cm`/`Jw_cm` family) when `cm_positions` changes: the kinematics cache is now invalidated too.
* `joint_type` is now validated and case-insensitively normalized to `"r"/"p"`; invalid values raise a clear `ValueError` instead of silently treating the joint as prismatic.

---

## [0.3.0] - 2026-07-XX

### Added

* Initial public release of Moro.
* Symbolic robot modeling based on Denavit–Hartenberg parameters.
* Forward kinematics computation.
* Robot visualization using Matplotlib.
* Support for numerical evaluation of symbolic robot models.
