# Dynamics

Robot dynamics describes the relationship between joint motion and the generalized forces required to produce that motion.

For a serial manipulator with configuration

$$
\vec q=
\begin{bmatrix}
q_1 &
q_2 &
\cdots &
q_n
\end{bmatrix}^{T},
$$

the corresponding velocity and acceleration vectors are

$$
\dot{\vec q}
=
\begin{bmatrix}
\dot q_1 &
\dot q_2 &
\cdots &
\dot q_n
\end{bmatrix}^{T},
$$

and

$$
\ddot{\vec q}
=
\begin{bmatrix}
\ddot q_1 &
\ddot q_2 &
\cdots &
\ddot q_n
\end{bmatrix}^{T}.
$$

The conventions used in this section follow [Mathematical notation and conventions](notation.md), [Forward kinematics](forward-kinematics.md), and [Differential kinematics](differential-kinematics.md).

## Scope of dynamics in Moro

Moro currently focuses on **symbolic dynamic modeling** of serial manipulators.

The main dynamic relation is

$$
\boxed{
M(\vec q)\ddot{\vec q}
+
C(\vec q,\dot{\vec q})\dot{\vec q}
+
G(\vec q)
=
\vec\tau.
}
$$

Here,

- $M(\vec q)$ is the inertia matrix,
- $C(\vec q,\dot{\vec q})$ is the Coriolis matrix,
- $G(\vec q)$ is the gravity generalized-force vector,
- $\vec\tau$ contains the generalized forces applied at the joints.

For joint $i$,

$$
\tau_i
$$

represents a torque for a revolute joint and a linear force for a prismatic joint.

Moro currently provides two routes for obtaining the equations of motion:

1. the Euler--Lagrange formulation,
2. direct construction of the matrix form.

These routes represent the same physical model.

```{important} id="scope-dynamics"
Moro currently derives symbolic equations of motion and supports inverse-dynamics modeling.

It does not yet integrate the equations of motion in time to obtain a numerical trajectory from applied forces and initial conditions.
```

Therefore, Moro currently supports the conceptual mapping

$$
\boxed{
(\vec q,\dot{\vec q},\ddot{\vec q})
\longrightarrow
\vec\tau,
}
$$

but not a complete simulation workflow of the form

$$
(\vec q_0,\dot{\vec q}_0,\vec\tau(t))
\longrightarrow
\vec q(t).
$$

## Physical parameters of a link

For each link $i$, the dynamic model uses the following physical quantities:

$$
\boxed{
m_i,\qquad
\vec r_{C_i}^{\,i},\qquad
I_{C_i}^{\,i}.
}
$$

The complete robot model also requires the gravity acceleration vector

$$
\boxed{
\vec g^{\,0}.
}
$$

Here, $C_i$ denotes the center of mass of link $i$.

## Link mass

The mass of link $i$ is denoted by

$$
\boxed{
m_i.
}
$$

Mass is a scalar quantity and does not depend on the choice of reference frame.

In Moro, individual masses are available through

```python
robot.m(i)
```

and collectively through

```python
robot.masses
```

.

## Center of mass

Moro defines the center-of-mass position of link $i$ relative to its own link frame $\{i\}$.

Thus,

$$
\boxed{
\vec r_{C_i}^{\,i}
}
$$

represents the vector from $O_i$ to $C_i$, expressed in frame $\{i\}$.

Using the more explicit notation introduced previously,

$$
\vec r_{C_i}^{\,i}
\equiv
\vec r_{C_i/O_i}^{\,i}.
$$

The corresponding position expressed in the base frame is

$$
\boxed{
\vec r_{C_i}^{\,0}
=
R_i^0\vec r_{C_i}^{\,i}
+
\vec r_{O_i}^{\,0}.
}
$$

Equivalently, using homogeneous coordinates,

$$
\tilde r_{C_i}^{\,0}
=
T_i^0
\tilde r_{C_i}^{\,i}.
$$

In Moro, this quantity is obtained through

```python
robot.r_cm(i)
```

.

The local vector

$$
\vec r_{C_i}^{\,i}
$$

is a constant physical parameter of the link, while

$$
\vec r_{C_i}^{\,0}(\vec q)
$$

depends on the current robot configuration.

## Inertia tensor

The inertia tensor of link $i$ is defined about its center of mass.

Moro assumes that the tensor supplied through `inertia_tensors` is expressed with respect to a frame:

- located at $C_i$,
- aligned with the link frame $\{i\}$.

We denote this tensor by

$$
\boxed{
I_{C_i}^{\,i}.
}
$$

The superscript indicates the orientation of the axes used to express the tensor components.

Conceptually, the same inertia tensor expressed along axes parallel to the base frame is

$$
\boxed{
I_{C_i}^{\,0}
=
R_i^0
I_{C_i}^{\,i}
(R_i^0)^T.
}
$$

In Moro,

```python
robot.I_cm(i)
```

returns $I_{C_i}^{\,i}$, while

```python
robot.I_cm0(i)
```

returns the same tensor expressed in the base-frame orientation.

## Gravity vector

Moro represents gravitational acceleration by

$$
\boxed{
\vec g^{\,0},
}
$$

expressed in the base frame.

For example, if the $z_0$ axis points vertically upward,

$$
\vec g^{\,0}
=
\begin{bmatrix}
0\\
0\\
-g
\end{bmatrix},
\qquad g>0.
$$

Moro does not assume a fixed gravity direction relative to the base frame. The user supplies the appropriate vector through

```python
robot.gravity
```

.

## Kinetic energy of a link

The kinetic energy of link $i$ is the sum of translational and rotational contributions:

$$
\boxed{
K_i
=
K_{T,i}
+
K_{R,i}.
}
$$

The translational term is

$$
\boxed{
K_{T,i}
=
\frac12
m_i
(\vec v_{C_i}^{\,0})^T
\vec v_{C_i}^{\,0}.
}
$$

The rotational term is

$$
\boxed{
K_{R,i}
=
\frac12
(\vec\omega_i^{\,0})^T
R_i^0
I_{C_i}^{\,i}
(R_i^0)^T
\vec\omega_i^{\,0}.
}
$$

Therefore,

$$
\boxed{
K_i
=
\frac12
m_i
(\vec v_{C_i}^{\,0})^T
\vec v_{C_i}^{\,0}
+
\frac12
(\vec\omega_i^{\,0})^T
R_i^0
I_{C_i}^{\,i}
(R_i^0)^T
\vec\omega_i^{\,0}.
}
$$

This is the form used directly by Moro in `link_kinetic_energy(i)`. The intermediate tensor $I_{C_i}^{\,0}$ is useful conceptually, but Moro evaluates the rotated tensor directly as

$$
R_i^0I_{C_i}^{\,i}(R_i^0)^T.
$$



## Kinetic energy and Jacobians

From differential kinematics,

$$
\vec v_{C_i}^{\,0}
=
J_{v,C_i}\dot{\vec q},
$$

and

$$
\vec\omega_i^{\,0}
=
J_{\omega,i}\dot{\vec q}.
$$

Substituting these expressions into the kinetic energy gives

$$
K_i
=
\frac12
\dot{\vec q}^{\,T}
\left[
m_iJ_{v,C_i}^TJ_{v,C_i}
+
J_{\omega,i}^T
R_i^0I_{C_i}^{\,i}(R_i^0)^T
J_{\omega,i}
\right]
\dot{\vec q}.
$$

The total kinetic energy is

$$
\boxed{
K
=
\sum_{i=1}^{n}K_i.
}
$$

In Moro, these quantities are available through

```python
robot.link_kinetic_energy(i)
robot.kinetic_energy()
```

.

## Potential energy

The gravitational potential energy of link $i$ is

$$
\boxed{
P_i
=
-m_i
(\vec g^{\,0})^T
\vec r_{C_i}^{\,0}.
}
$$

The total potential energy is therefore

$$
\boxed{
P(\vec q)
=
-\sum_{i=1}^{n}
m_i
(\vec g^{\,0})^T
\vec r_{C_i}^{\,0}(\vec q).
}
$$

This convention is used directly by Moro in

```python
robot.link_potential_energy(i)
robot.potential_energy()
```

.

For example, if

$$
\vec g^{\,0}
=
\begin{bmatrix}
0\\
0\\
-g
\end{bmatrix},
$$

then

$$
P_i
=
m_i g z_{C_i}.
$$

## Lagrangian

The Lagrangian of the manipulator is defined as

$$
\boxed{
\mathcal L
=
K-P.
}
$$

In general,

$$
\mathcal L
=
\mathcal L(\vec q,\dot{\vec q}).
$$

Moro provides this quantity through

```python
robot.lagrangian()
```

.

## Euler--Lagrange equations

For each generalized coordinate $q_i$, the equation of motion is

$$
\boxed{
\frac{d}{dt}
\left(
\frac{\partial\mathcal L}
{\partial\dot q_i}
\right)
-
\frac{\partial\mathcal L}
{\partial q_i}
=
\tau_i.
}
$$

Collectively, the $n$ equations describe the inverse dynamic model of the robot.

Moro constructs these equations through

```python
robot.dynamic_model()
```

.

The result is a list of symbolic equations, one for each joint.

## Time-dependent joint variables

Dynamic analysis requires joint variables that depend on time:

$$
\boxed{
q_i=q_i(t).
}
$$

Only then are

$$
\dot q_i
=
\frac{dq_i}{dt}
$$

and

$$
\ddot q_i
=
\frac{d^2q_i}{dt^2}
$$

meaningful symbolic quantities.

```{important} id="dynamic-symbols"
For dynamic analyses, joint variables should be defined as time-dependent symbolic functions.

Static SymPy symbols may be appropriate for purely kinematic calculations, but velocity- and acceleration-dependent expressions may become incomplete or incorrect when static symbols are used.
```

Moro emits warnings in velocity-dependent operations when static joint variables are detected. This applies, for example, to angular velocity, the Coriolis matrix, and the Euler--Lagrange dynamic model.

Geometric and physical parameters such as

$$
a_i,\qquad
m_i,\qquad
I_{xx},
$$

may remain symbolic constants while

$$
q_i=q_i(t).
$$

## Matrix form of the dynamic model

The Euler--Lagrange equations can be reorganized as

$$
\boxed{
M(\vec q)\ddot{\vec q}
+
C(\vec q,\dot{\vec q})\dot{\vec q}
+
G(\vec q)
=
\vec\tau.
}
$$

This form separates the dynamic model into:

- inertial effects,
- Coriolis and centrifugal effects,
- gravitational effects.

Moro can construct these quantities directly, without first expanding the complete Euler--Lagrange equations.

## Inertia matrix

The kinetic energy can be written in quadratic form as

$$
\boxed{
K
=
\frac12
\dot{\vec q}^{\,T}
M(\vec q)
\dot{\vec q}.
}
$$

Using the link Jacobians, the inertia matrix is

$$
\boxed{
M(\vec q)
=
\sum_{i=1}^{n}
\left[
m_iJ_{v,C_i}^TJ_{v,C_i}
+
J_{\omega,i}^T
R_i^0
I_{C_i}^{\,i}
(R_i^0)^T
J_{\omega,i}
\right].
}
$$

This is the expression implemented by

```python
robot.inertia_matrix()
```

.

The first term in each summand represents translational inertia, while the second represents rotational inertia.

For a physically valid rigid-body model, the inertia matrix has the theoretical properties

$$
\boxed{
M(\vec q)=M(\vec q)^T
}
$$

and, for nonzero $\vec x$,

$$
\boxed{
\vec x^TM(\vec q)\vec x>0.
}
$$

These properties characterize the usual symmetric positive-definite structure of the joint-space inertia matrix.

## Christoffel symbols

Moro constructs the Coriolis matrix using Christoffel symbols of the first kind.

The convention used is

$$
\boxed{
c_{ijk}
=
\frac12
\left(
\frac{\partial M_{ij}}{\partial q_k}
+
\frac{\partial M_{ik}}{\partial q_j}
-
\frac{\partial M_{jk}}{\partial q_i}
\right).
}
$$

In the API,

```python
robot.christoffel_symbols(i, j, k, M)
```

returns $c_{ijk}$ for a supplied inertia matrix $M$.

## Coriolis matrix

The elements of the Coriolis matrix are defined by

$$
\boxed{
C_{ij}
=
\sum_{k=1}^{n}
c_{ijk}\dot q_k.
}
$$

Therefore,

$$
\boxed{
C
=
C(\vec q,\dot{\vec q}).
}
$$

The term that enters the equations of motion is

$$
\boxed{
C(\vec q,\dot{\vec q})
\dot{\vec q}.
}
$$

This vector contains the Coriolis and centrifugal effects associated with joint velocities.

Moro computes the matrix through

```python
robot.coriolis_matrix()
```

.

```{note} id="coriolis-nonunique"
The matrix representation $C(\vec q,\dot{\vec q})$ is not unique in a completely abstract sense.

Moro uses the Christoffel-based convention defined above. When comparing formulations from different sources, the physically relevant quantity is the resulting velocity-dependent term in the equations of motion.
```

## Gravity generalized-force vector

The gravity term is obtained from the gradient of the potential energy:

$$
\boxed{
G(\vec q)
=
\nabla_{\vec q}P(\vec q).
}
$$

In component form,

$$
\boxed{
G_i(\vec q)
=
\frac{\partial P}{\partial q_i}.
}
$$

Moro provides this vector through

```python
robot.gravity_vector()
```

.

If

$$
\dot{\vec q}=0
$$

and

$$
\ddot{\vec q}=0,
$$

the dynamic equation reduces to

$$
\boxed{
\vec\tau
=
G(\vec q).
}
$$

Thus, $G(\vec q)$ represents the generalized forces required to balance gravity in static equilibrium.

## Complete matrix formulation

After computing $M$, $C$, and $G$, Moro assembles

$$
\boxed{
M(\vec q)\ddot{\vec q}
+
C(\vec q,\dot{\vec q})\dot{\vec q}
+
G(\vec q)
=
\vec\tau.
}
$$

through

```python
robot.dynamic_model_matrix_form()
```

.

The vectors are

$$
\dot{\vec q}
=
\begin{bmatrix}
\dot q_1\\
\vdots\\
\dot q_n
\end{bmatrix},
$$

$$
\ddot{\vec q}
=
\begin{bmatrix}
\ddot q_1\\
\vdots\\
\ddot q_n
\end{bmatrix},
$$

and

$$
\vec\tau
=
\begin{bmatrix}
\tau_1\\
\vdots\\
\tau_n
\end{bmatrix}.
$$

## Energy and matrix formulations

Moro provides two routes for deriving the equations of motion.

### Euler--Lagrange route

The energy-based route is

$$
K,P
\longrightarrow
\mathcal L=K-P
$$

followed by

$$
\boxed{
\frac{d}{dt}
\left(
\frac{\partial\mathcal L}{\partial\dot q_i}
\right)
-
\frac{\partial\mathcal L}{\partial q_i}
=
\tau_i.
}
$$

In Moro, the relevant methods are

```python
robot.kinetic_energy()
robot.potential_energy()
robot.lagrangian()
robot.dynamic_model()
```

.

### Matrix route

The direct matrix route constructs

$$
M(\vec q),
$$

$$
C(\vec q,\dot{\vec q}),
$$

and

$$
G(\vec q)
$$

independently, and then assembles

$$
M\ddot q+C\dot q+G=\tau.
$$

The relevant methods are

```python
robot.inertia_matrix()
robot.coriolis_matrix()
robot.gravity_vector()
robot.dynamic_model_matrix_form()
```

.

These are not different physical models.

They are two algebraic routes to the same equations of motion.

## Dynamic parameter assumptions

Not all dynamic parameters are treated identically by Moro.

### Symbolic masses

If masses are initialized through the setter with `None`, Moro generates symbolic values

$$
m_1,m_2,\ldots,m_n.
$$

These are symbolic modeling parameters rather than identified physical properties.

### Diagonal symbolic inertia tensors

If inertia tensors are initialized with `None`, Moro generates diagonal symbolic tensors of the form

$$
\boxed{
I_{C_i}^{\,i}
=
\begin{bmatrix}
I_{x_ix_i} & 0 & 0\\
0 & I_{y_iy_i} & 0\\
0 & 0 & I_{z_iz_i}
\end{bmatrix}.
}
$$

This assumes zero products of inertia.

```{important} id="inertia-assumption"
A diagonal inertia tensor is a modeling assumption, not a general property of every rigid body.

When products of inertia are relevant, the complete $3\times3$ tensor should be supplied explicitly.
```

### Center-of-mass positions

Center-of-mass positions are not generated automatically.

They must be supplied when required by the requested dynamic quantity.

### Gravity

The gravity vector is also not generated automatically.

It must be defined when potential energy or gravity-dependent quantities are required.

## Model state

Moro distinguishes between quantities that are:

- explicitly defined,
- generated under a documented assumption,
- not set.

The current model state can be summarized through

```python
robot.model_summary()
```

.

This distinction is useful because symbolic placeholders or default assumptions should not be interpreted as experimentally identified physical parameters.

## Dynamic quantities in Moro

The main theory-to-API correspondence is:

| Mathematical quantity | Moro API |
|---|---|
| $m_i$ | `m(i)` |
| $\vec r_{C_i}^{\,0}$ | `r_cm(i)` |
| $I_{C_i}^{\,i}$ | `I_cm(i)` |
| $I_{C_i}^{\,0}$ | `I_cm0(i)` |
| $\vec v_{C_i}^{\,0}$ | `v_cm(i)` |
| $\vec\omega_i^{\,0}$ | `w(i)` |
| $J_{v,C_i}$ | `Jv_cm_i(i)` |
| $J_{\omega,i}$ | `Jw_cm_i(i)` |
| $K_i$ | `link_kinetic_energy(i)` |
| $P_i$ | `link_potential_energy(i)` |
| $K$ | `kinetic_energy()` |
| $P$ | `potential_energy()` |
| $\mathcal L$ | `lagrangian()` |
| Euler--Lagrange equations | `dynamic_model()` |
| $M(\vec q)$ | `inertia_matrix()` |
| $c_{ijk}$ | `christoffel_symbols(i,j,k,M)` |
| $C(\vec q,\dot q)$ | `coriolis_matrix()` |
| $G(\vec q)$ | `gravity_vector()` |
| $M\ddot q+C\dot q+G=\tau$ | `dynamic_model_matrix_form()` |

## Example workflow

A typical symbolic dynamic-modeling workflow is:

```python
import moro as mr

robot = mr.Robot(...)

robot.masses = [...]
robot.cm_positions = [...]
robot.inertia_tensors = [...]
robot.gravity = [...]

M = robot.inertia_matrix()
C = robot.coriolis_matrix()
G = robot.gravity_vector()

model = robot.dynamic_model_matrix_form()
```

Alternatively, the same model can be derived through the energy formulation:

```python
K = robot.kinetic_energy()
P = robot.potential_energy()
L = robot.lagrangian()

equations = robot.dynamic_model()
```

The two workflows provide different symbolic routes to the same physical equations of motion.

## Scope and limitations

The current dynamic-modeling capabilities of Moro include:

- link masses,
- center-of-mass locations,
- full or diagonal inertia tensors,
- arbitrary gravity direction expressed in the base frame,
- translational and rotational kinetic energy,
- gravitational potential energy,
- Lagrangian formulation,
- Euler--Lagrange equations,
- joint-space inertia matrix,
- Christoffel symbols,
- Coriolis matrix,
- gravity generalized-force vector,
- inverse dynamic equations in matrix form.

The current implementation does not provide:

- numerical forward-dynamics integration,
- numerical simulation of $\vec q(t)$ from applied forces,
- contact dynamics,
- collision forces,
- friction models,
- actuator dynamics,
- external wrench handling in the dynamic equations.

## Summary of conventions

| Concept | Moro convention |
|---|---|
| Dynamic coordinates | $q_i=q_i(t)$ |
| Generalized force | $\tau_i$ |
| Revolute generalized force | Torque |
| Prismatic generalized force | Linear force |
| CoM local position | $\vec r_{C_i}^{\,i}$ |
| CoM base position | $\vec r_{C_i}^{\,0}$ |
| Local inertia tensor | $I_{C_i}^{\,i}$ |
| Base-oriented inertia tensor | $R_i^0I_{C_i}^{\,i}(R_i^0)^T$ |
| Gravity | $\vec g^{\,0}$ expressed in base frame |
| Link kinetic energy | Translational + rotational |
| Potential energy | $-m_i(\vec g^{\,0})^T\vec r_{C_i}^{\,0}$ |
| Lagrangian | $\mathcal L=K-P$ |
| Gravity vector | $G=\nabla_qP$ |
| Inertia matrix | Jacobian-based link sum |
| Coriolis convention | Christoffel symbols of the first kind |
| Dynamic model | $M\ddot q+C\dot q+G=\tau$ |
| Current scope | Symbolic modeling / inverse dynamics |
| Forward time integration | Not currently provided |

Dynamics completes the progression from geometry and motion to generalized forces. Forward kinematics determines where the robot is, differential kinematics relates joint rates to Cartesian velocities, and the dynamic model determines the forces required to produce a prescribed joint motion.