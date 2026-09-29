# Singularity and manipulability analysis

This example collects four complementary local analyses using the same SVD-based task-Jacobian policy as Moro's velocity inverse kinematics.

## 1. Planar 2R singularity analysis

```python
from sympy import pi

from moro import Robot
from moro.abc import l1, l2, q1, q2
from moro.differential_kinematics import (
    singular_values,
    jacobian_rank,
    condition_number,
    is_singular,
    manipulability,
)

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)

parameters = {l1: 1.0, l2: 0.8}
task = ("vx", "vy")
```

Compare a regular posture with a fully extended one:

```python
regular_q = [0, pi / 2]
singular_q = [0, 0]

for q in (regular_q, singular_q):
    print(singular_values(robot, q, task=task, parameters=parameters))
    print(jacobian_rank(robot, q, task=task, parameters=parameters))
    print(condition_number(robot, q, task=task, parameters=parameters))
    print(is_singular(robot, q, task=task, parameters=parameters))
```

At the fully extended posture, the planar position Jacobian loses one independent direction, so its rank drops and its condition number becomes infinite under the automatic threshold.

## 2. Planar 2R manipulability

For the same planar position task, the analytical Yoshikawa metric is

$$
w=|l_1l_2\sin q_2|.
$$

A small sweep can compare the numerical result with this expression:

```python
from math import sin

for q2_value in [0.0, 0.25, 0.5, 1.0, float(pi / 2)]:
    numerical = manipulability(
        robot,
        [0.0, q2_value],
        task=task,
        parameters=parameters,
    )
    analytical = abs(parameters[l1] * parameters[l2] * sin(q2_value))
    print(q2_value, numerical, analytical)
```

At $q_2=0$, manipulability is exactly zero. At $q_2=\pi/2$, it reaches $l_1l_2$.

## 3. Singularity depends on the selected task

Consider a planar 3R robot:

```python
from moro.abc import q3

robot3 = Robot(
    (1, 0, 0, q1, "r"),
    (1, 0, 0, q2, "r"),
    (1, 0, 0, q3, "r"),
)

q = [0.2, 0.8, -0.4]

print(is_singular(robot3, q, task=("vx", "vy")))
print(is_singular(robot3, q, task=("vx", "vy", "vz")))
```

The planar translational task `("vx", "vy")` can have full row rank two, while adding the unattainable `vz` component creates a three-row task whose rank remains two. The same configuration is therefore regular for the reduced task and singular for the larger task.

## 4. Condition number and manipulability measure different things

For any regular one-dimensional task the condition number is one because there is only one singular value. Its absolute magnitude can still be very small.

Using the planar 2R robot near a posture where the $x$-velocity row is small:

```python
q = [0.0, 1e-3]
task_1d = ("vx",)

print(condition_number(robot, q, task=task_1d, parameters=parameters))
print(manipulability(robot, q, task=task_1d, parameters=parameters))
```

The condition number is one while manipulability is small. This illustrates that condition number measures anisotropy, whereas manipulability also reflects absolute local velocity scale.

## Interpretation cautions

Manipulability depends on the chosen task and units. A task that mixes translational and angular rows also mixes their scales. Moro 0.5.0 does not introduce a characteristic length or automatic weighting, so the task should be chosen explicitly according to the quantity being studied.
