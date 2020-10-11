Getting started
---------------

:code:`Robot` is the main class of moro, this class define a serial-robot using the Denavit-Hartenberg parameters. 

.. image:: https://raw.githubusercontent.com/numython-rd/moro/9bfbb6ec0b8162b726c0f0ff7be1b84a02a5bca8/examples/nbook/es/img/rr_robot_dh.svg
	:width: 300

.. code-block:: language

	rr = Robot((l1,0,0,t1), (l2,0,0,t2))