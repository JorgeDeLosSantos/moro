The :code:`visualization` module
--------------------------------

This module provides tools for visualizing serial robots. It supports two
backends: Matplotlib for static plots and Three.js for interactive
visualizations and animations in Jupyter notebooks.


Overview
^^^^^^^^

The high-level entry point is :class:`moro.visualization.RobotVisualizer`. It
takes a :class:`moro.core.Robot`, evaluates it at one or more numerical
configurations and dispatches the rendering to one of the available backends.


Quick example
^^^^^^^^^^^^^

.. code-block:: python

    import moro as mr
    from moro.abc import l1, l2, q1, q2
    from moro.visualization import RobotVisualizer

    robot = mr.Robot(
        (l1, 0, 0, q1, "r"),
        (l2, 0, 0, q2, "r"),
    )

    visualizer = RobotVisualizer(robot)

    fig, ax = visualizer.plot(
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5},
    )


Backends
^^^^^^^^

Matplotlib
~~~~~~~~~~

The Matplotlib backend returns a ``(fig, ax)`` pair for static plots:

.. code-block:: python

    fig, ax = visualizer.plot(
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5},
        backend="matplotlib",
        figsize=(8, 6),
        view_init=(30, 45),
    )

It can also create Matplotlib animations from a list of configurations:

.. code-block:: python

    configurations = [
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.0},
        {l1: 1.0, l2: 1.0, q1: 0.2, q2: 0.4},
        {l1: 1.0, l2: 1.0, q1: 0.4, q2: 0.8},
    ]

    animation = visualizer.animate(configurations, interval=100)

Keep a reference to the returned animation until it is displayed or saved.


Three.js
~~~~~~~~

The Three.js backend returns an ``IPython.display.HTML`` object and is intended
for Jupyter notebooks:

.. code-block:: python

    html = visualizer.plot(
        {l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5},
        backend="threejs",
        width=800,
        height=600,
    )

Animations are available through the same high-level method:

.. code-block:: python

    html_animation = visualizer.animate(
        configurations,
        backend="threejs",
        width=800,
        height=600,
    )


VisualizationStyle
^^^^^^^^^^^^^^^^^^

Use :class:`moro.visualization.VisualizationStyle` to configure the appearance
in a backend-independent way.

.. code-block:: python

    from moro.visualization import VisualizationStyle

    style = VisualizationStyle(
        show_frames=True,
        show_links=True,
        show_joints=True,
        show_grid=False,
        show_trajectory=True,
        trajectory_mode="trace",
    )

    animation = visualizer.animate(configurations, style=style)

Important style options include:

.. list-table:: Visualization style options
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``show_frames``
     - Draw coordinate frames at each joint.
   * - ``show_links``
     - Draw links connecting consecutive joints.
   * - ``show_joints``
     - Draw joint markers.
   * - ``show_base``
     - Highlight the base joint when joints are shown.
   * - ``show_grid``
     - Draw the reference grid.
   * - ``link_color``, ``joint_color``, ``base_color``
     - Configure colors.
   * - ``frame_scale``
     - Length of coordinate-frame axes. If ``None``, it is derived from the scene size.
   * - ``show_trajectory``
     - Draw the end-effector trajectory in animations.
   * - ``trajectory_mode``
     - ``"full"`` for the complete path or ``"trace"`` up to the current frame.


Lower-level utilities
^^^^^^^^^^^^^^^^^^^^^

For advanced use, :func:`moro.visualization.evaluate_robot` converts a robot and
a numeric configuration into a :class:`moro.visualization.SceneData` object.
Backends can render this scene data directly.


API reference
^^^^^^^^^^^^^

.. automodule:: moro.visualization
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance: