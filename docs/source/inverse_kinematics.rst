The :code:`inverse_kinematics` module
-------------------------------------

This module provides numerical solvers for the position inverse kinematics
problem of serial robots. Three methods are available: Newton-Raphson,
Levenberg-Marquardt (default), and Cyclic Coordinate Descent (CCD).


Overview
^^^^^^^^

The functions in this module solve **position** inverse kinematics: they try to
find a vector of joint variables that places the end-effector origin at a target
Cartesian position ``[x, y, z]``. They do not solve orientation constraints.

Because the solvers are numerical and local, convergence depends on the initial
guess, joint limits, robot geometry and target reachability. Always inspect the
returned result instead of assuming success.


Available methods
^^^^^^^^^^^^^^^^^

``method="lm"``
    Levenberg-Marquardt. This is the default method and is usually the most
    robust first choice for position IK.

``method="newton"``
    Newton-Raphson update based on the position Jacobian. It can converge fast
    near a solution but may be less robust far from one.

``method="ccd"``
    Cyclic Coordinate Descent. This method updates one joint at a time and can
    be useful for redundant or difficult initial configurations.


Solving one target
^^^^^^^^^^^^^^^^^^

Use :func:`moro.inverse_kinematics.solve_position_ik` for a single target. Any
geometric symbols that are not joint variables must be substituted with numeric
values through ``parameters``.

.. code-block:: python

    import sympy as sp
    import moro as mr
    from moro.abc import l1, l2, q1, q2
    from moro.inverse_kinematics import solve_position_ik

    robot = mr.Robot(
        (l1, 0, 0, q1, "r"),
        (l2, 0, 0, q2, "r"),
    )

    solution = solve_position_ik(
        robot,
        target_position=[1.0, 1.0, 0.0],
        q0=[0.1, 0.1],
        parameters={l1: 1.0, l2: 1.0},
        joint_limits=[(-sp.pi, sp.pi), (-sp.pi, sp.pi)],
        method="lm",
    )

    if solution.converged:
        print("Joint solution:", solution.q)
    else:
        print("IK failed:", solution.message)


Joint limits
^^^^^^^^^^^^

Joint limits are lists of ``(lower, upper)`` pairs. They can be passed directly
to the solver:

.. code-block:: python

    joint_limits = [(-sp.pi, sp.pi), (-sp.pi / 2, sp.pi / 2)]

    solution = solve_position_ik(
        robot,
        [1.0, 0.5, 0.0],
        q0=[0.0, 0.0],
        parameters={l1: 1.0, l2: 1.0},
        joint_limits=joint_limits,
    )

If ``joint_limits`` is ``None``, the solver uses ``robot.joint_limits``.


Understanding solver results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`moro.inverse_kinematics.IKSolution` stores the outcome of a single IK
solve:

.. list-table:: IKSolution fields
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``q``
     - Final joint variables.
   * - ``converged``
     - ``True`` when the requested tolerance was reached.
   * - ``iterations``
     - Number of completed iterations.
   * - ``error``
     - Final position error norm.
   * - ``method``
     - Method used by the solver.
   * - ``residual``
     - Final vector ``target_position - current_position`` when available.
   * - ``message``
     - Human-readable status message.


Solving a trajectory
^^^^^^^^^^^^^^^^^^^^

Use :func:`moro.inverse_kinematics.solve_position_trajectory` for a sequence of
position targets. The solution of each target is reused as the seed for the next
target.

.. code-block:: python

    from moro.inverse_kinematics import solve_position_trajectory

    targets = [
        [1.5, 0.2, 0.0],
        [1.4, 0.4, 0.0],
        [1.2, 0.6, 0.0],
    ]

    trajectory = solve_position_trajectory(
        robot,
        targets,
        q0=[0.1, 0.1],
        parameters={l1: 1.0, l2: 1.0},
    )

    if trajectory.converged:
        print(trajectory.qs)
    else:
        print("Failed target index:", trajectory.failed_index)

:class:`moro.inverse_kinematics.IKTrajectorySolution` exposes convenience
properties such as ``qs``, ``errors`` and ``iterations``. The returned joint
sequence can be used as input for visualization or animation.


Limitations
^^^^^^^^^^^

The trajectory helper is a sequential IK orchestration layer. It does not
perform interpolation, timing, smoothing, full-pose orientation IK, branch
optimization or global trajectory planning.


API reference
^^^^^^^^^^^^^

.. automodule:: moro.inverse_kinematics
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance: