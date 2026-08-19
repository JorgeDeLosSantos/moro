Examples
--------

This page collects small, executable examples for the most common tasks in
Moro. Longer examples are available as Jupyter notebooks at the end of this
page.


Forward kinematics
^^^^^^^^^^^^^^^^^^

Create a planar two-revolute-joint robot and compute the end-effector
homogeneous transformation matrix:

.. code-block:: python

    import moro as mr
    from moro.abc import l1, l2, q1, q2

    robot = mr.Robot(
        (l1, 0, 0, q1, "r"),
        (l2, 0, 0, q2, "r"),
    )

    T = robot.T
    T_numeric = T.subs({l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.0})


Jacobian matrix
^^^^^^^^^^^^^^^

The geometric Jacobian of the end-effector is available through
:attr:`moro.core.Robot.J`:

.. code-block:: python

    from moro.util import pprint

    J = robot.J
    pprint(J)

Use :meth:`moro.core.Robot.J_point` to compute the geometric Jacobian of a
point attached to a specific link.


Inverse kinematics
^^^^^^^^^^^^^^^^^^

The numerical inverse-kinematics helpers solve position IK. Geometric
parameters must be substituted with numeric values:

.. code-block:: python

    import sympy as sp
    from moro.inverse_kinematics import solve_position_ik

    solution = solve_position_ik(
        robot,
        target_position=[1.0, 1.0, 0.0],
        q0=[0.2, 0.2],
        parameters={l1: 1.0, l2: 1.0},
        joint_limits=[(-sp.pi, sp.pi), (-sp.pi, sp.pi)],
    )

    if solution.converged:
        print(solution.q)
    else:
        print(solution.message)


Dynamic model
^^^^^^^^^^^^^

Dynamic computations require masses, center-of-mass positions, inertia tensors
and gravity. Intrinsic link parameters such as masses and inertia tensors can be
auto-generated as symbolic placeholders, but problem-specific data such as
center-of-mass positions and gravity should be set explicitly.

.. code-block:: python

    from moro.abc import m1, m2, lc1, lc2, g

    robot.masses = [m1, m2]
    robot.inertia_tensors = None  # auto-generate diagonal symbolic tensors
    robot.cm_positions = [[-lc1, 0, 0], [-lc2, 0, 0]]
    robot.gravity = [0, -g, 0]

    M, C, G = robot.dynamic_model_matrix_form()


Visualization
^^^^^^^^^^^^^

Use :class:`moro.visualization.RobotVisualizer` to render a configuration with
Matplotlib or Three.js:

.. code-block:: python

    from moro.visualization import RobotVisualizer, VisualizationStyle

    style = VisualizationStyle(show_frames=True, show_links=True)
    visualizer = RobotVisualizer(robot)

    fig, ax = visualizer.plot(
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5},
        style=style,
    )

In a Jupyter notebook, use ``backend="threejs"`` for an interactive view:

.. code-block:: python

    html = visualizer.plot(
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5},
        backend="threejs",
        style=style,
    )


Transformations
^^^^^^^^^^^^^^^

The :mod:`moro.transformations` module provides helpers for rotation matrices
and homogeneous transformations:

.. code-block:: python

    from sympy import pi
    from moro import rotx, htmtra, htmrot, invhtm

    R = rotx(pi / 2)
    H = htmtra(1, 2, 3) * htmrot(pi / 2, axis="z")
    H_inverse = invhtm(H)


Jupyter notebooks
^^^^^^^^^^^^^^^^^

For longer examples, check out the Jupyter notebooks available in the
repository:

* `Forward kinematics <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Forward%20kinematics.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Forward%20kinematics.ipynb>`__)
* `Inverse kinematics <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Inverse%20kinematics.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Inverse%20kinematics.ipynb>`__)
* `Jacobian matrix <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Jacobian%20matrix.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Jacobian%20matrix.ipynb>`__)
* `Dynamic model <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Dynamic%20model.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Dynamic%20model.ipynb>`__)
* `Solve position trajectory <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Solve%20position%20trajectory.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Solve%20position%20trajectory.ipynb>`__)
* `Visualization <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Visualization.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Visualization.ipynb>`__)
* `Transformations <https://github.com/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Transformations.ipynb>`__
  (`Open in Colab <https://colab.research.google.com/github/JorgeDeLosSantos/moro/blob/main/examples/nbooks/Transformations.ipynb>`__)