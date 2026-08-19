Getting started
---------------

In this section we will review how "moro" can be used to address some common exercises in robot kinematics.


Forward kinematics for RR manipulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the figure is shown a RR manipulator with references frames and its DH parameters table. Now, the goal is to calculate the forward kinematics using :code:`moro`, how this is done?

.. image:: https://raw.githubusercontent.com/JorgeDeLosSantos/moro/main/examples/nbooks/img/rr_robot_dh.svg
	:width: 400

Well, the next lines of code can do this task:

.. code-block:: python
	
	>>> from moro import Robot
	>>> from moro.abc import l1, l2, q1, q2
	>>> rr = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
	>>> T = rr.T
	>>> print(T)
	Matrix([[cos(q_1(t) + q_2(t)), -sin(q_1(t) + q_2(t)), 0, l_1*cos(q_1(t)) + l_2*cos(q_1(t) + q_2(t))], [sin(q_1(t) + q_2(t)), cos(q_1(t) + q_2(t)), 0, l_1*sin(q_1(t)) + l_2*sin(q_1(t) + q_2(t))], [0, 0, 1, 0], [0, 0, 0, 1]])

In :code:`T` is saved the :math:`T_2^0` matrix calculated. What about the above code?

* First line import the :code:`Robot` class.
* Second line imports the symbolic variables :code:`l1`, :code:`l2` (link lengths) and :code:`q1`, :code:`q2` (joint variables).
* Third line create a :code:`Robot` object using the DH parameters of the RR manipulator. The DH parameters are passed as tuples in the following order: :math:`(a_i, \alpha_i, d_i, \theta_i, \text{joint\_type})`, where the joint type is ``"r"`` for revolute or ``"p"`` for prismatic.
* In the fourth line the :code:`T` attribute from :code:`rr` object is accessed and saved in `T` variable. 
* The fifth line print the result.

As you can see, the matrix print in console is not so practical when symbolic variables are used. Alternatively, you can use the :code:`pprint` function and to obtain better results: 

.. code-block:: python

	>>> from moro.util import pprint
	>>> pprint(T)
	⎡cos(q₁(t) + q₂(t))  -sin(q₁(t) + q₂(t))  0  l₁⋅cos(q₁(t)) + l₂⋅cos(q₁(t) + q₂(t))⎤
	⎢                                                                                  ⎥
	⎢sin(q₁(t) + q₂(t))  cos(q₁(t) + q₂(t))   0  l₁⋅sin(q₁(t)) + l₂⋅sin(q₁(t) + q₂(t))⎥
	⎢                                                                                  ⎥
	⎢         0                    0           1                    0                   ⎥
	⎢                                                                                  ⎥
	⎣         0                    0           0                    1                   ⎦

For best results (in printing aspects) **we encourage you to use Jupyter Notebooks**.

If you want to replace symbolic variables by numeric values, then you can use :code:`subs` method:

.. code-block:: python

	>>> T.subs({l1:100,l2:100,q1:0,q2:0})
	⎡1  0  0  200⎤
	⎢            ⎥
	⎢0  1  0   0 ⎥
	⎢            ⎥
	⎢0  0  1   0 ⎥
	⎢            ⎥
	⎣0  0  0   1 ⎦




Calculating geometric jacobian for RR manipulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	>>> rr = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
	>>> J = rr.J
	>>> pprint(J)
	⎡-l₁⋅sin(q₁(t)) - l₂⋅sin(q₁(t) + q₂(t))  -l₂⋅sin(q₁(t) + q₂(t))⎤
	⎢                                                                 ⎥
	⎢l₁⋅cos(q₁(t)) + l₂⋅cos(q₁(t) + q₂(t))   l₂⋅cos(q₁(t) + q₂(t))  ⎥
	⎢                                                                 ⎥
	⎢                 0                            0                 ⎥
	⎢                                                                 ⎥
	⎢                 0                            0                 ⎥
	⎢                                                                 ⎥
	⎢                 0                            0                 ⎥
	⎢                                                                 ⎥
	⎣                 1                            1                 ⎦