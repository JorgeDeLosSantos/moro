Trajectory Generation
=====================

The :mod:`moro.trajectory` module provides numerical point-to-point
and multi-waypoint trajectories in joint and Cartesian position spaces.
Multi-waypoint interpolation supports linear, cubic Hermite, and quintic
Hermite methods, as well as a PCHIP-style shape-preserving cubic
interior-velocity policy with resting endpoints. Synchronized rest-to-rest
joint motion also supports trapezoidal/triangular velocity profiles with
optional immutable phase metadata.

.. automodule:: moro.trajectory
   :members:
   :undoc-members:
   :show-inheritance:
