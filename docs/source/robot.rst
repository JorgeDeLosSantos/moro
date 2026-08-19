Robot modeling
--------------

The :class:`moro.core.Robot` class is the central object in Moro. It represents
a serial robot using Denavit-Hartenberg parameters and exposes symbolic tools
for forward kinematics, differential kinematics and dynamics.


Creating a robot
^^^^^^^^^^^^^^^^

A robot is created by passing one row per joint/link. Each row contains the
Denavit-Hartenberg parameters in the order
``(a, alpha, d, theta)``. A fifth element can be used to specify the joint type:

.. code-block:: python

    from moro import Robot
    from moro.abc import l1, l2, q1, q2

    robot = Robot(
        (l1, 0, 0, q1, "r"),
        (l2, 0, 0, q2, "r"),
    )

Joint types are:

* ``"r"`` for revolute joints;
* ``"p"`` for prismatic joints.

If the joint type is omitted, Moro assumes a revolute joint.


Revolute and prismatic joints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a revolute joint, the joint variable is the ``theta`` parameter. For a
prismatic joint, the joint variable is the ``d`` parameter.

.. code-block:: python

    from moro.abc import d1, q2, l2

    rp_robot = Robot(
        (0, 0, d1, 0, "p"),
        (l2, 0, 0, q2, "r"),
    )

    print(rp_robot.joint_types)  # ['p', 'r']
    print(rp_robot.dof)          # 2


Forward kinematics
^^^^^^^^^^^^^^^^^^

The end-effector homogeneous transformation matrix is available through
:attr:`moro.core.Robot.T`:

.. code-block:: python

    T = robot.T

You can also access intermediate transformations. Link indices are one-based:

.. code-block:: python

    T_1_0 = robot.T_i0(1)    # frame {1} with respect to frame {0}
    T_2_0 = robot.T_i0(2)    # frame {2} with respect to frame {0}
    T_2_1 = robot.T_ij(2, 1) # frame {2} with respect to frame {1}


Jacobian matrix
^^^^^^^^^^^^^^^

The geometric Jacobian of the end-effector is available as:

.. code-block:: python

    J = robot.J

Use :meth:`moro.core.Robot.J_point` for the Jacobian of a point attached to a
specific link:

.. code-block:: python

    J_tip = robot.J_point([0, 0, 0], robot.dof)


Numerical substitution
^^^^^^^^^^^^^^^^^^^^^^

Moro uses SymPy expressions. Numeric values are commonly introduced with
``subs``:

.. code-block:: python

    T_numeric = robot.T.subs({l1: 1.0, l2: 1.0, q1: 0.0, q2: 0.5})


Joint limits
^^^^^^^^^^^^

Joint limits are represented as a list of ``(lower, upper)`` pairs, one pair per
degree of freedom:

.. code-block:: python

    from sympy import pi

    robot.joint_limits = [(-pi, pi), (-pi, pi)]

The inverse-kinematics functions use ``robot.joint_limits`` when their own
``joint_limits`` argument is not provided.


Dynamic parameters
^^^^^^^^^^^^^^^^^^

The dynamic model uses masses, inertia tensors, center-of-mass positions and
gravity:

.. code-block:: python

    from moro.abc import m1, m2, lc1, lc2, g

    robot.masses = [m1, m2]
    robot.inertia_tensors = None
    robot.cm_positions = [[-lc1, 0, 0], [-lc2, 0, 0]]
    robot.gravity = [0, -g, 0]

Intrinsic link quantities such as masses and inertia tensors can be provided
explicitly or auto-generated as symbolic placeholders. Problem/environment
quantities such as center-of-mass positions and gravity should be set
explicitly because Moro cannot infer them from DH parameters alone.

Use :meth:`moro.core.Robot.model_summary` to inspect the current modeling state:

.. code-block:: python

    print(robot.model_summary())


Dynamic model
^^^^^^^^^^^^^

The matrix form of the equations of motion can be obtained with:

.. code-block:: python

    M, C, G = robot.dynamic_model_matrix_form()

where ``M`` is the inertia matrix, ``C`` is the Coriolis/centrifugal matrix and
``G`` is the gravity vector. The symbolic Euler-Lagrange equations are available
through :meth:`moro.core.Robot.dynamic_model`.


API reference
^^^^^^^^^^^^^

See :doc:`core` for the full API reference of :class:`moro.core.Robot`.