Getting started
---------------

In this section we will review how "moro" can be used to address some common exercises in robot kinematics.


Forward kinematics for RR manipulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the figure is shown a RR manipulator with references frames and its DH parameters table. Now, the goal is to calculate the forward kinematics using :code:`moro`, how this is done?

.. image:: https://raw.githubusercontent.com/numython-rd/moro/9bfbb6ec0b8162b726c0f0ff7be1b84a02a5bca8/examples/nbook/es/img/rr_robot_dh.svg
	:width: 400

Well, the next lines of code can do this task:

.. code-block:: python
	
	>>>from moro import *
	>>> rr = Robot((l1,0,0,t1),(l2,0,0,t2))
	>>> T = rr.T
	>>> print(T)
	Matrix([[cos(theta_1 + theta_2), -sin(theta_1 + theta_2), 0, l_1*cos(theta_1) + l_2*cos(theta_1 + theta_2)], [sin(theta_1 + theta_2), cos(theta_1 + theta_2), 0, l_1*sin(theta_1) + l_2*sin(theta_1 + theta_2)], [0, 0, 1, 0], [0, 0, 0, 1]])

In :code:`T` is saved the :math:`T_2^0` matrix calculated. What about the above code?

* First line import the library
* Second line create a :code:`Robot` object using the DH parameters of the RR manipulator. The DH parameters are passed as tuples in the following order: :math:`(a_i, \alpha_i, d_i, \theta_i)`
* In the third line the :code:`T` attribute from :code:`rr` object is accessed and saved in `T` variable. 
* The fourth line print the result.

As you can see, the matrix print in console is not so practical when symbolic variables are used. Alternatively, you can use the :code:`pprint` function and to obtain better results: 

.. code-block:: python

	>>> pprint(T)
	⎡cos(θ₁ + θ₂)  -sin(θ₁ + θ₂)  0  l₁⋅cos(θ₁) + l₂⋅cos(θ₁ + θ₂)⎤
	⎢                                                            ⎥
	⎢sin(θ₁ + θ₂)  cos(θ₁ + θ₂)   0  l₁⋅sin(θ₁) + l₂⋅sin(θ₁ + θ₂)⎥
	⎢                                                            ⎥
	⎢     0              0        1               0              ⎥
	⎢                                                            ⎥
	⎣     0              0        0               1              ⎦

For best results (in printing aspects) **we encourage you to use Jupyter Notebooks**.

If you want to replace symbolic variables by numeric values, then you can use :code:`subs` method:

.. code-block:: python

	>>> T.subs({l1:100,l2:100,t1:0,t2:0})
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

	>>> rr = Robot((l1,0,0,t1), (l2,0,0,t2))
	>>> J = rr.J
	>>> pprint(J)
	⎡-l₁⋅sin(θ₁) - l₂⋅sin(θ₁ + θ₂)  -l₂⋅sin(θ₁ + θ₂)⎤
	⎢                                               ⎥
	⎢l₁⋅cos(θ₁) + l₂⋅cos(θ₁ + θ₂)   l₂⋅cos(θ₁ + θ₂) ⎥
	⎢                                               ⎥
	⎢              0                       0        ⎥
	⎢                                               ⎥
	⎢              0                       0        ⎥
	⎢                                               ⎥
	⎢              0                       0        ⎥
	⎢                                               ⎥
	⎣              1                       1        ⎦




