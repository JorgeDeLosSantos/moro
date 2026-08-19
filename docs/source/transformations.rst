The :code:`transformations` module
----------------------------------

The :mod:`moro.transformations` module provides symbolic helpers for common
robotics transformations: rotation matrices in :math:`SO(3)`, homogeneous
transformation matrices in :math:`SE(3)`, Denavit-Hartenberg transformations,
Euler angles, axis-angle representation and skew-symmetric matrices.


Summary
^^^^^^^

.. list-table:: Transformation helpers
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`moro.transformations.rot`
     - Generic rotation matrix around the ``"x"``, ``"y"`` or ``"z"`` axis.
   * - :func:`moro.transformations.rotx`
     - Rotation matrix around the x-axis.
   * - :func:`moro.transformations.roty`
     - Rotation matrix around the y-axis.
   * - :func:`moro.transformations.rotz`
     - Rotation matrix around the z-axis.
   * - :func:`moro.transformations.dh`
     - Homogeneous transformation matrix from Denavit-Hartenberg parameters.
   * - :func:`moro.transformations.htmtra`
     - Homogeneous transformation matrix for a pure translation.
   * - :func:`moro.transformations.htmrot`
     - Homogeneous transformation matrix for a pure rotation.
   * - :func:`moro.transformations.rot2htm`
     - Embed a 3x3 rotation matrix into a 4x4 homogeneous matrix.
   * - :func:`moro.transformations.rt2htm`
     - Build a homogeneous matrix from a rotation matrix and translation vector.
   * - :func:`moro.transformations.htm2rot`
     - Extract the rotation block from a homogeneous matrix.
   * - :func:`moro.transformations.htm2tra`
     - Extract the translation vector from a homogeneous matrix.
   * - :func:`moro.transformations.invhtm`
     - Compute the structured inverse of a homogeneous matrix.
   * - :func:`moro.transformations.eul2rot`
     - Convert Euler angles to a rotation matrix.
   * - :func:`moro.transformations.rot2eul`
     - Convert a rotation matrix to Euler-angle solutions.
   * - :func:`moro.transformations.axa2rot`
     - Convert an axis-angle representation to a rotation matrix.
   * - :func:`moro.transformations.rot2axa`
     - Convert a rotation matrix to axis-angle representation.
   * - :func:`moro.transformations.skew`
     - Build a skew-symmetric matrix from a 3D vector.


Basic rotations
^^^^^^^^^^^^^^^

.. code-block:: python

    from sympy import pi
    from moro import rotx, roty, rotz

    Rx = rotx(pi / 2)
    Ry = roty(pi / 2)
    Rz = rotz(pi / 2)

Angles are interpreted as radians by default. Pass ``deg=True`` when using
degrees:

.. code-block:: python

    R = rotz(90, deg=True)


Homogeneous transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Translations and rotations can be composed by multiplying homogeneous matrices:

.. code-block:: python

    from sympy import pi
    from moro import htmtra, htmrot, invhtm, htm2rot, htm2tra

    H = htmtra(1, 2, 3) * htmrot(pi / 2, axis="z")

    R = htm2rot(H)
    p = htm2tra(H)
    H_inverse = invhtm(H)


Denavit-Hartenberg matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`moro.transformations.dh` to build the homogeneous transformation
associated with one Denavit-Hartenberg row:

.. code-block:: python

    from moro import dh
    from moro.abc import a1, alpha1, d1, theta1

    A_1_0 = dh(a1, alpha1, d1, theta1)


API reference
^^^^^^^^^^^^^

.. automodule:: moro.transformations
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance: