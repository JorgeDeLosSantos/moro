# Dynamic Model of a Planar 2R Manipulator

This example shows how to build the symbolic dynamic model of a planar two-link manipulator with `moro`.

We will:

1. define the robot kinematically;
2. assign the physical parameters of each link;
3. inspect the dynamic model configuration;
4. compute kinetic and potential energies;
5. obtain the inertia matrix;
6. compute Coriolis, centrifugal, and gravity terms;
7. generate the Euler-Lagrange equations of motion;
8. obtain the standard matrix form;
9. evaluate the model numerically.

The objective is to show how the symbolic kinematic model of a robot becomes a dynamic model once masses, centers of mass, inertia tensors, and gravity are specified.

## Problem

Consider a planar two-link manipulator with two revolute joints:

\[
q_1,\; q_2,
\]

and link lengths:

\[
l_1,\; l_2.
\]

The robot moves in the \(xy\)-plane and both joints rotate about axes parallel to \(z\).

Its Denavit-Hartenberg table is:

| Link | \(a_i\) | \(\alpha_i\) | \(d_i\) | \(\theta_i\) | Joint |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | \(l_1\) | \(0\) | \(0\) | \(q_1\) | revolute |
| 2 | \(l_2\) | \(0\) | \(0\) | \(q_2\) | revolute |

We want to obtain the equations of motion in the standard form

\[
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau,
\]

where

\[
q=
\begin{bmatrix}
q_1\\
q_2
\end{bmatrix},
\qquad
\tau=
\begin{bmatrix}
\tau_1\\
\tau_2
\end{bmatrix}.
\]

## Robot model

For dynamic modeling, the generalized coordinates must be time-dependent.

The symbols provided by `moro.abc`, such as `q1` and `q2`, are suitable for this purpose.

Import the required objects:

```python
import sympy as sp

from moro import Robot
from moro.abc import (
    t,
    q1,
    q2,
    m1,
    m2,
    l1,
    l2,
    lc1,
    lc2,
    g,
)
```

Create the robot:

```python
robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The robot initially contains only the kinematic description.

To construct a dynamic model, we must also provide the physical properties of the links.

## Dynamic parameters

The dynamic model requires:

- link masses;
- center-of-mass positions;
- inertia tensors;
- gravity.

These quantities are added to the existing `Robot` object.

### Link masses

Assign one mass to each link:

```python
robot.masses = [
    m1,
    m2,
]
```

The masses can be inspected with:

```python
robot.masses
```

or individually with:

```python
robot.m(1)
robot.m(2)
```

### Center-of-mass positions

The center of mass of each link is specified relative to its corresponding link frame.

For this example, assume that both centers of mass lie along the local \(x\)-axis.

Using positive symbolic distances:

\[
l_{c1},\; l_{c2},
\]

we define:

```python
robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
```

The sign depends on the selected frame convention.

With the DH frames used here, the centers of mass are placed along the negative local \(x\)-direction from the corresponding frame origins.

The center-of-mass positions expressed in the base frame can be obtained with:

```python
robot.r_cm(1)
robot.r_cm(2)
```

### Inertia tensors

Each inertia tensor must be defined about the corresponding link center of mass and aligned with that link frame.

For a planar model, only rotation about the local \(z\)-axis contributes to the rotational kinetic energy.

Define symbolic moments of inertia:

```python
I1, I2 = sp.symbols(
    "I1 I2",
    positive=True,
)
```

and assign:

```python
robot.inertia_tensors = [
    sp.diag(0, 0, I1),
    sp.diag(0, 0, I2),
]
```

The tensors can be inspected with:

```python
robot.inertia_tensors
```

or individually with:

```python
robot.I_cm(1)
robot.I_cm(2)
```

For this planar example, the unused \(x\)- and \(y\)-axis inertia components are set to zero for simplicity because only the \(z\)-axis terms contribute to the rotational motion considered here.

In a general three-dimensional model, a complete \(3\times3\) inertia tensor should be supplied.

### Gravity

Gravity is defined in the base frame.

Assume that the \(y\)-axis points upward, so gravity acts in the negative \(y\)-direction:

```python
robot.gravity = (
    0,
    -g,
    0,
)
```

The gravity vector can be inspected with:

```python
robot.gravity
```

The robot now contains all parameters required for the complete dynamic model.

## Inspecting the dynamic model

The current model configuration can be summarized with:

```python
print(robot.model_summary())
```

The summary reports whether quantities such as:

- joint limits;
- masses;
- inertia tensors;
- center-of-mass positions;
- gravity;

were explicitly assigned or are still undefined.

This is particularly useful before requesting a dynamic quantity because different computations require different subsets of the physical parameters.

For the complete equations of motion, all four dynamic properties used above must be defined.

## Center-of-mass kinematics

Before computing the energies, we can inspect the kinematics of the link centers of mass.

Their positions in the base frame are:

```python
r_cm1 = robot.r_cm(1)
r_cm2 = robot.r_cm(2)
```

The corresponding linear velocities are:

```python
v_cm1 = robot.v_cm(1)
v_cm2 = robot.v_cm(2)
```

The link angular velocities are:

```python
w1 = robot.w(1)
w2 = robot.w(2)
```

These quantities are computed from the same symbolic robot model used for forward kinematics.

They form the basis of the kinetic-energy calculation.

## Kinetic energy

The kinetic energy of each link contains translational and rotational contributions:

\[
K_i
=
\frac{1}{2}m_i v_{C_i}^{T}v_{C_i}
+
\frac{1}{2}
\omega_i^{T}
R_i^0 I_{C_i}^{i}(R_i^0)^T
\omega_i.
\]

The kinetic energy of link 1 is:

```python
K1 = robot.link_kinetic_energy(1)
K1
```

and for link 2:

```python
K2 = robot.link_kinetic_energy(2)
K2
```

The total kinetic energy is:

```python
K = robot.kinetic_energy()
K
```

The resulting expression depends on:

\[
q_1,\; q_2,\; \dot q_1,\; \dot q_2
\]

and on the mass, geometry, and inertia parameters.

## Potential energy

The gravitational potential energy of each link can be obtained with:

```python
P1 = robot.link_potential_energy(1)
P2 = robot.link_potential_energy(2)
```

The total potential energy is:

```python
P = robot.potential_energy()
P
```

The gravitational potential used by `moro` is based on the center-of-mass position and the gravity vector expressed in the base frame.

The Lagrangian is then:

\[
\mathcal{L}=K-P.
\]

It can be computed directly with:

```python
L = robot.lagrangian()
L
```

## Inertia matrix

The inertia matrix is obtained with:

```python
M = robot.inertia_matrix()
M
```

For a two-degree-of-freedom manipulator:

\[
M(q)
\in
\mathbb{R}^{2\times2}.
\]

The inertia matrix relates the generalized accelerations to the inertial part of the joint torques.

It is constructed from the translational and rotational Jacobians of the link centers of mass:

\[
M(q)
=
\sum_{i=1}^{2}
\left[
m_i J_{v_i}^{T}J_{v_i}
+
J_{\omega_i}^{T}
R_i^0 I_{C_i}^{i}(R_i^0)^T
J_{\omega_i}
\right].
\]

The matrix is symbolic and can be reused for different configurations and physical parameter values.

## Coriolis and centrifugal terms

The Coriolis matrix is obtained with:

```python
C = robot.coriolis_matrix()
C
```

For this robot:

\[
C(q,\dot q)
\in
\mathbb{R}^{2\times2}.
\]

The velocity-dependent generalized-force contribution is:

\[
C(q,\dot q)\dot q.
\]

Construct the joint-velocity vector:

```python
qd = sp.Matrix([
    q1.diff(t),
    q2.diff(t),
])
```

Then:

```python
velocity_terms = C * qd
velocity_terms
```

The matrix `C` is computed from the inertia matrix through the Christoffel symbols.

Different conventions can produce different valid Coriolis matrices, provided that the product

\[
C(q,\dot q)\dot q
\]

represents the same velocity-dependent generalized forces.

## Gravity vector

The gravity torque vector is:

```python
G = robot.gravity_vector()
G
```

For the planar RR robot:

\[
G(q)
\in
\mathbb{R}^{2}.
\]

The gravity vector is obtained from the gradient of the potential energy with respect to the generalized coordinates.

Conceptually:

\[
G(q)
=
\frac{\partial P}{\partial q}.
\]

It represents the generalized torques required to balance the gravitational contribution at a given configuration when velocity and acceleration are zero.

## Equations of motion

The complete Euler-Lagrange equations can be generated with:

```python
equations = robot.dynamic_model()
```

The result is a list containing one SymPy equation per generalized coordinate.

For example:

```python
equations[0]
```

contains the equation associated with \(q_1\), and:

```python
equations[1]
```

contains the equation associated with \(q_2\).

The equations have the form:

\[
\frac{d}{dt}
\left(
\frac{\partial\mathcal{L}}
{\partial\dot q_i}
\right)
-
\frac{\partial\mathcal{L}}
{\partial q_i}
=
\tau_i.
\]

They can be displayed together with:

```python
for equation in equations:
    sp.pprint(equation)
```

This representation is useful when studying the Euler-Lagrange formulation directly.

## Matrix form

The same dynamic model can also be written in the standard manipulator form:

\[
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau.
\]

Use:

```python
matrix_equation = robot.dynamic_model_matrix_form()
matrix_equation
```

This returns a symbolic SymPy equation containing the complete matrix model.

The generalized acceleration vector is:

```python
qdd = sp.Matrix([
    q1.diff(t, 2),
    q2.diff(t, 2),
])
```

and the generalized torque vector can be written conceptually as:

\[
\tau=
\begin{bmatrix}
\tau_1\\
\tau_2
\end{bmatrix}.
\]

The two interfaces:

```python
robot.dynamic_model()
robot.dynamic_model_matrix_form()
```

represent the same underlying dynamic model.

The first emphasizes the Euler-Lagrange equations individually, while the second exposes the standard matrix structure commonly used in robotics.

## Computing the torque expression

The left-hand side of the matrix equation can also be constructed explicitly:

```python
tau_expr = (
    M * qdd
    + C * qd
    + G
)
```

This symbolic vector represents the generalized torques required for a prescribed:

- joint position;
- joint velocity;
- joint acceleration.

It is therefore convenient for inverse-dynamics-style evaluation.

## Numerical evaluation

The symbolic dynamic model can be evaluated by substituting numerical values.

Assume:

\[
l_1=1.0,
\qquad
l_2=0.8,
\]

\[
l_{c1}=0.5,
\qquad
l_{c2}=0.4,
\]

\[
m_1=2.0,
\qquad
m_2=1.5,
\]

\[
I_1=0.15,
\qquad
I_2=0.08,
\]

and:

\[
g=9.81.
\]

Consider the state:

\[
q_1=30^\circ,
\qquad
q_2=-20^\circ,
\]

\[
\dot q_1=0.4,
\qquad
\dot q_2=-0.2,
\]

with accelerations:

\[
\ddot q_1=0.5,
\qquad
\ddot q_2=0.1.
\]

Create the substitution dictionary:

```python
values = {
    l1: 1.0,
    l2: 0.8,
    lc1: 0.5,
    lc2: 0.4,
    m1: 2.0,
    m2: 1.5,
    I1: 0.15,
    I2: 0.08,
    g: 9.81,

    q1: sp.pi / 6,
    q2: -sp.pi / 9,

    q1.diff(t): 0.4,
    q2.diff(t): -0.2,

    q1.diff(t, 2): 0.5,
    q2.diff(t, 2): 0.1,
}
```

Evaluate the inertia matrix:

```python
M_num = M.subs(values).evalf()
M_num
```

Evaluate the Coriolis matrix:

```python
C_num = C.subs(values).evalf()
C_num
```

Evaluate the gravity vector:

```python
G_num = G.subs(values).evalf()
G_num
```

The required generalized torques are:

```python
tau_num = tau_expr.subs(values).evalf()
tau_num
```

This result corresponds to the torque vector required to produce the prescribed instantaneous motion according to the symbolic dynamic model.

## Evaluating individual contributions

It is often useful to inspect each term separately.

### Inertial contribution

```python
inertial_torque = (
    M * qdd
).subs(values).evalf()

inertial_torque
```

### Coriolis and centrifugal contribution

```python
velocity_torque = (
    C * qd
).subs(values).evalf()

velocity_torque
```

### Gravity contribution

```python
gravity_torque = G.subs(values).evalf()

gravity_torque
```

The total torque is the sum:

```python
total_torque = (
    inertial_torque
    + velocity_torque
    + gravity_torque
)

total_torque
```

which should agree with:

```python
tau_num
```

This decomposition is useful for understanding the physical origin of the generalized forces.

## Reusing the symbolic model

Once the symbolic matrices have been computed:

```python
M
C
G
```

they can be evaluated repeatedly without rebuilding the robot.

For example, a different configuration can be analyzed simply by changing the substitution dictionary.

This is especially useful for:

- comparing configurations;
- generating symbolic expressions for teaching;
- evaluating inverse dynamics at several operating points;
- inspecting how inertial or gravitational terms vary with joint position.

The symbolic model therefore acts as a reusable analytical representation of the robot dynamics.

## Discussion

This example illustrates the complete transition from a kinematic robot model to a symbolic dynamic model.

The process is:

```text
define robot kinematics
        ↓
assign masses
        ↓
assign centers of mass
        ↓
assign inertia tensors
        ↓
define gravity
        ↓
compute energies
        ↓
obtain M(q), C(q,q̇), G(q)
        ↓
generate equations of motion
        ↓
evaluate numerically
```

The kinematic structure alone is not sufficient to determine the dynamics.

The dynamic model additionally requires physical information about each link.

Once these quantities are defined, `moro` can obtain:

```python
robot.kinetic_energy()
robot.potential_energy()
robot.lagrangian()

robot.inertia_matrix()
robot.coriolis_matrix()
robot.gravity_vector()

robot.dynamic_model()
robot.dynamic_model_matrix_form()
```

The matrix form

\[
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau
\]

can be evaluated for prescribed joint positions, velocities, and accelerations to determine the required generalized torques.

This is an inverse-dynamics-style use of the model.

The current dynamic capabilities of `moro` do not integrate the equations of motion forward in time.

In other words, `moro` does not currently solve a problem of the form:

\[
\tau(t),\;
q(0),\;
\dot q(0)
\quad\longrightarrow\quad
q(t).
\]

Forward dynamics and numerical trajectory integration would require solving the differential equations of motion over time and are outside the current scope of the library.

## See also

- **Planar 2R Manipulator** — kinematic analysis of the same type of robot.
- **User Guide → Dynamics** — defining dynamic parameters and using the dynamics API.
- **User Guide → Jacobians** — center-of-mass Jacobians used in the inertia matrix.
- **User Guide → Forward Kinematics** — frame transformations used in link kinematics.
- **Theory → Dynamics** — derivation of kinetic energy, potential energy, Euler-Lagrange equations, and matrix-form dynamics.