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

### Changed

* Refactored the visualization package into multiple modules with clear responsibilities.
* Unified visualization configuration across Matplotlib and Three.js backends through `VisualizationStyle`.
* Improved Three.js animation performance by updating existing scene objects instead of recreating them every frame.
* Simplified the visualization API by removing notebook-specific rendering methods in favor of a unified interface.
* HTML templates are now packaged as library resources and loaded through `importlib.resources`.

### Fixed

* Fixed frame orientation rendering in the Three.js backend.
* Fixed animation scaling to remain consistent across all frames.
* Fixed packaging of visualization templates for installation from GitHub and PyPI.
* Improved synchronization between robot data and rendered scene during animations.

---

## [0.3.0] - 2026-07-XX

### Added

* Initial public release of Moro.
* Symbolic robot modeling based on Denavit–Hartenberg parameters.
* Forward kinematics computation.
* Robot visualization using Matplotlib.
* Support for numerical evaluation of symbolic robot models.
