# Dynamics

`moro` provides symbolic tools for defining and analyzing the dynamic model of serial robotic manipulators.

Once the kinematic structure of a `Robot` has been created, additional physical parameters can be assigned to describe the mass distribution of its links and the gravity field. These quantities can then be used to compute energies, the manipulator inertia matrix, Coriolis terms, gravity torques, and equations of motion.

This section focuses on practical use of the dynamics API. For the mathematical derivation of the expressions used by `moro`, see **Theory → Dynamics**.

## Defining the dynamic model

Consider a planar 2R manipulator:

```python
from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

The DH parameters define the robot kinematics, but they are not sufficient to construct its dynamic model.

For dynamic calculations, the following quantities may also be required:

```python
robot.masses
robot.cm_positions
robot.inertia_tensors
robot.gravity
```

These describe:

* the mass of each link;
* the position of each link center of mass;
* the inertia tensor of each link;
* the gravity acceleration expressed in the base frame.

The required subset depends on the quantity being computed. For example, the inertia matrix requires masses, center-of-mass positions, and inertia tensors, while potential energy also requires gravity.

For dynamic analyses, the joint variables should be time-dependent symbols. The variables provided by `moro.abc`, such as `q1` and `q2`, are suitable for this purpose.

## Link masses

The mass of each link is assigned through:

```python
robot.masses = [
    m1,
    m2,
]
```

where the number of values must match the number of robot links.

Masses may be numerical:

```python
robot.masses = [
    2.0,
    1.5,
]
```

or symbolic:

```python
from sympy import symbols

m1, m2 = symbols("m1 m2", positive=True)

robot.masses = [
    m1,
    m2,
]
```

An individual link mass can be accessed with:

```python
robot.m(1)
robot.m(2)
```

If symbolic placeholder masses are desired, they can also be generated automatically with:

```python
robot.masses = None
```

which creates one symbolic mass for each link.

These automatically generated quantities are convenience symbols rather than assumed physical values.

## Center-of-mass positions

The center of mass of each link is specified relative to the corresponding link frame `{i}`.

Use:

```python
robot.cm_positions = [
    (x1, y1, z1),
    (x2, y2, z2),
]
```

For example, consider a planar model where each link frame is located at its distal end and each center of mass lies along the negative local $x_i$-axis:

```python
from sympy import symbols

lc1, lc2 = symbols("lc1 lc2", positive=True)

robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
```

Each entry must contain exactly three components.

The stored center-of-mass position of link `i`, expressed in its local frame, is used internally to compute quantities such as its position in the base frame:

```python
rG1 = robot.r_cm(1)
rG2 = robot.r_cm(2)
```

The resulting vectors are expressed with respect to frame `{0}`.

The corresponding linear velocities are available through:

```python
vG1 = robot.v_cm(1)
vG2 = robot.v_cm(2)
```

Because these velocities are obtained by differentiation with respect to time, time-dependent joint variables should be used.

## Inertia tensors

The inertia tensor of each link is assigned with:

```python
robot.inertia_tensors = [
    I1,
    I2,
]
```

Each tensor must be a $3\times3$ matrix.

The tensor of link `i` is defined with respect to a frame:

* located at the center of mass of the link;
* oriented in the same way as frame `{i}`.

For example:

```python
from sympy import Matrix, symbols

Ixx1, Iyy1, Izz1 = symbols("Ixx1 Iyy1 Izz1")
Ixx2, Iyy2, Izz2 = symbols("Ixx2 Iyy2 Izz2")

I1 = Matrix([
    [Ixx1, 0, 0],
    [0, Iyy1, 0],
    [0, 0, Izz1],
])

I2 = Matrix([
    [Ixx2, 0, 0],
    [0, Iyy2, 0],
    [0, 0, Izz2],
])

robot.inertia_tensors = [
    I1,
    I2,
]
```

The stored tensor can be retrieved with:

```python
robot.I_cm(1)
```

To express the same tensor with axes aligned with the base frame, use:

```python
robot.I_cm0(1)
```

which applies the corresponding link rotation:

$$
I_{C_i}^{0}
=
R_i^0
I_{C_i}^{i}
(R_i^0)^T.
$$

### Automatically generated diagonal tensors

If:

```python
robot.inertia_tensors = None
```

is used, `moro` creates symbolic diagonal inertia tensors automatically.

Conceptually:

$$
I_i =
\begin{bmatrix}
I_{x_i x_i} & 0 & 0 \\
0 & I_{y_i y_i} & 0 \\
0 & 0 & I_{z_i z_i}
\end{bmatrix}.
$$

This assumes zero products of inertia.

It is therefore a modeling convenience, not a universal physical assumption. Explicit tensors should be provided when the link geometry does not justify diagonal inertia tensors in the selected center-of-mass frame.

## Gravity

The gravity acceleration is specified with respect to the robot base frame:

```python
robot.gravity = (gx, gy, gz)
```

For example, if gravity acts along the negative $y$-axis:

```python
from sympy import symbols

g = symbols("g", positive=True)

robot.gravity = (0, -g, 0)
```

A numerical value can also be used:

```python
robot.gravity = (0, -9.81, 0)
```

The current gravity vector can be inspected with:

```python
robot.gravity
```

The choice of direction must be consistent with the orientation of the robot base frame.

## Inspecting the model state

Before computing dynamic quantities, it is often useful to check which physical properties have already been defined.

Use:

```python
print(robot.model_summary())
```

The summary reports the state of quantities such as:

* joint limits;
* masses;
* inertia tensors;
* center-of-mass positions;
* gravity.

For example, a quantity may appear as:

```text
explicit
```

when it was provided directly by the user,

```text
assumed (...)
```

when `moro` generated a symbolic placeholder based on a documented assumption, or:

```text
NOT SET
```

when the required information has not yet been defined.

This is particularly useful before requesting a dynamic quantity that depends on several model properties.

## Center-of-mass kinematics

Several dynamic computations depend on the motion of each link center of mass.

The position of the center of mass of link `i` in the base frame is:

```python
robot.r_cm(i)
```

Its linear velocity is:

```python
robot.v_cm(i)
```

and the angular velocity of the link is:

```python
robot.w(i)
```

The center-of-mass Jacobians are also available:

```python
robot.J_cm_i(i)
robot.Jv_cm_i(i)
robot.Jw_cm_i(i)
```

These quantities are the same interfaces introduced in **Jacobians**, but they become especially useful when constructing kinetic-energy and inertia expressions.

## Kinetic and potential energy

`moro` can compute both per-link and total system energies.

### Link kinetic energy

Use:

```python
K1 = robot.link_kinetic_energy(1)
```

For link $i$, the kinetic energy includes both translational and rotational contributions:

$$
K_i
=
\frac{1}{2}
m_i
v_{G_i}^T
v_{G_i}
+
\frac{1}{2}
\omega_i^T
R_i^0 I_{C_i}^{i}(R_i^0)^T
\omega_i.
$$

The total kinetic energy is:

```python
K = robot.kinetic_energy()
```

which corresponds to:

$$
K = \sum_{i=1}^{n} K_i.
$$

### Link potential energy

The gravitational potential energy of link `i` is:

```python
P1 = robot.link_potential_energy(1)
```

and is computed as:

$$
P_i
=
-m_i g^T r_{G_i}.
$$

The total potential energy is:

```python
P = robot.potential_energy()
```

with:

$$
P = \sum_{i=1}^{n} P_i.
$$

### Lagrangian

The system Lagrangian is available directly:

```python
L = robot.lagrangian()
```

and is defined as:

$$
L = K - P.
$$

All these quantities are returned symbolically.

## The inertia matrix

The manipulator inertia matrix is computed with:

```python
M = robot.inertia_matrix()
```

For a robot with $n$ degrees of freedom, the result is an $n\times n$ symbolic matrix.

`moro` constructs it from the translational and rotational kinetic-energy contributions of all links:

$$
M(q)
=
\sum_{i=1}^{n}
\left[
m_i
J_{v_i}^{T}
J_{v_i}
+
J_{\omega_i}^{T}
R_i^0
I_{C_i}^{i}
(R_i^0)^T
J_{\omega_i}
\right].
$$

Before calling `inertia_matrix()`, the following must be defined:

```text
masses
cm_positions
inertia_tensors
```

For example:

```python
M = robot.inertia_matrix()
```

can then be simplified or inspected using normal SymPy operations:

```python
from sympy import simplify

M = simplify(M)
```

For larger symbolic mechanisms, explicit simplification can become computationally expensive.

## Coriolis and gravity terms

The remaining terms in the standard manipulator equation can also be computed directly.

### Coriolis matrix

Use:

```python
C = robot.coriolis_matrix()
```

which returns:

$$
C(q,\dot q).
$$

The implementation constructs this matrix from the Christoffel symbols of the first kind.

The resulting matrix satisfies the usual form:

$$
C(q,\dot q)\dot q.
$$

Since this quantity depends on joint velocities, the joint variables should be time dependent.

### Gravity vector

Use:

```python
G = robot.gravity_vector()
```

which returns the generalized gravity-force vector:

$$
G(q)
=
\nabla P(q).
$$

The result has one component per degree of freedom.

Because `gravity_vector()` is obtained from the potential energy, masses, center-of-mass locations, and gravity must already be defined.

## Equations of motion

`moro` provides two convenient representations of the robot equations of motion.

### Euler-Lagrange equations

Use:

```python
equations = robot.dynamic_model()
```

This returns one equation per joint:

$$
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot q_i}
\right)
-
\frac{\partial L}{\partial q_i}
=
\tau_i.
$$

For example:

```python
eq1 = equations[0]
eq2 = equations[1]
```

The equations are returned as SymPy equation objects.

### Matrix form

The compact manipulator equation is available through:

```python
model = robot.dynamic_model_matrix_form()
```

which represents:

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau.
$$

This form is often convenient when inspecting the structure of the dynamic model or comparing it with the standard robotics notation.

The two interfaces represent the same underlying model from different viewpoints: `dynamic_model()` exposes the Euler-Lagrange equations individually, while `dynamic_model_matrix_form()` organizes the dynamics into the usual $M$, $C$, and $G$ terms.

## Evaluating symbolic dynamics

Dynamic quantities returned by `moro` are symbolic SymPy expressions and matrices.

Consider a model containing symbolic physical parameters:

```python
M = robot.inertia_matrix()
C = robot.coriolis_matrix()
G = robot.gravity_vector()
```

Numerical values can be introduced with a substitution dictionary.

For example:

```python
values = {
    l1: 1.0,
    l2: 0.8,
    lc1: 0.5,
    lc2: 0.4,
    m1: 2.0,
    m2: 1.5,
    g: 9.81,
}
```

Joint configurations can be added as well:

```python
values.update({
    q1: 0.5,
    q2: 0.8,
})
```

Then:

```python
M_num = M.subs(values).evalf()
G_num = G.subs(values).evalf()
```

Velocity-dependent quantities require values for the derivatives of the joint variables.

For example:

```python
values.update({
    q1.diff(): 0.2,
    q2.diff(): -0.1,
})
```

and acceleration-dependent expressions can similarly use:

```python
values.update({
    q1.diff().diff(): 0.5,
    q2.diff().diff(): 0.3,
})
```

The same symbolic model can therefore be evaluated at multiple states without reconstructing the robot.

## A worked example

Consider a planar 2R robot with symbolic geometry and dynamic parameters:

```python
from sympy import diag, symbols

from moro import Robot
from moro.abc import q1, q2, l1, l2

robot = Robot(
    (l1, 0, 0, q1, "r"),
    (l2, 0, 0, q2, "r"),
)
```

Define masses and center-of-mass locations:

```python
m1, m2 = symbols("m1 m2", positive=True)
lc1, lc2 = symbols("lc1 lc2", positive=True)

robot.masses = [m1, m2]

robot.cm_positions = [
    (-lc1, 0, 0),
    (-lc2, 0, 0),
]
```

For a planar mechanism, suppose only the $z$-axis moments of inertia are relevant:

```python
Iz1, Iz2 = symbols("Iz1 Iz2", positive=True)

robot.inertia_tensors = [
    diag(0, 0, Iz1),
    diag(0, 0, Iz2),
]
```

Define gravity along the negative $y$-direction:

```python
g = symbols("g", positive=True)

robot.gravity = (0, -g, 0)
```

Check the model:

```python
print(robot.model_summary())
```

Now compute the main dynamic quantities:

```python
M = robot.inertia_matrix()
C = robot.coriolis_matrix()
G = robot.gravity_vector()
```

The total energies are:

```python
K = robot.kinetic_energy()
P = robot.potential_energy()
L = robot.lagrangian()
```

The Euler-Lagrange equations can be generated with:

```python
equations = robot.dynamic_model()
```

or the compact matrix representation with:

```python
matrix_model = robot.dynamic_model_matrix_form()
```

To evaluate the model numerically, define:

```python
values = {
    l1: 1.0,
    l2: 0.8,
    lc1: 0.5,
    lc2: 0.4,
    m1: 2.0,
    m2: 1.5,
    Iz1: 0.15,
    Iz2: 0.08,
    g: 9.81,
    q1: 0.4,
    q2: -0.2,
    q1.diff(): 0.3,
    q2.diff(): -0.1,
}
```

Then:

```python
M_num = M.subs(values).evalf()
C_num = C.subs(values).evalf()
G_num = G.subs(values).evalf()
```

This workflow keeps the derivation symbolic while allowing the resulting model to be evaluated for specific physical parameters and robot states.

## Notes and limitations

The dynamics tools in the current version of `moro` are primarily intended for symbolic modeling and analysis.

Keep the following points in mind:

* dynamic properties are defined on the same `Robot` object used for kinematics;
* masses, center-of-mass locations, inertia tensors, and gravity are not inferred automatically from DH geometry;
* automatically generated masses or diagonal inertia tensors are symbolic conveniences and should not be interpreted as measured physical properties;
* center-of-mass positions are expressed in their corresponding link frames;
* inertia tensors are defined at the link center of mass and aligned with the corresponding `{i}` frame;
* gravity is expressed in the base frame;
* dynamic results are symbolic SymPy expressions;
* velocity-dependent calculations require time-dependent joint variables;
* `moro.abc.q1`, `q2`, and related variables are appropriate for dynamic analyses;
* using static SymPy symbols for joints in velocity-dependent operations can produce warnings and incorrect derivative terms.

The current implementation provides symbolic equations of motion and inverse-dynamics-style model evaluation, but it does not yet provide forward-dynamics integration.

In particular, `moro` does not currently integrate:

$$
M(q)\ddot q
+
C(q,\dot q)\dot q
+
G(q)
=
\tau
$$

over time to obtain a trajectory $q(t)$ from applied torques and initial conditions.

Numerical simulation, control, contact dynamics, collision forces, and physics-engine integration are also outside the current scope.

## See also

* **Robot Modeling** — define the kinematic structure of a serial manipulator.
* **Forward Kinematics** — compute the frame transformations used in dynamic calculations.
* **Jacobians** — compute center-of-mass linear and angular Jacobians.
* **Theory → Dynamics** — mathematical derivation of energies, inertia matrices, Coriolis terms, gravity terms, and Euler-Lagrange equations.
* **API Reference → Robot** — complete reference for dynamic properties and methods.