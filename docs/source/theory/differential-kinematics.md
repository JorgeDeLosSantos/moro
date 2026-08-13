# Differential kinematics

Differential kinematics describes the relationship between joint velocities and the instantaneous linear and angular velocities of points and links in a robot manipulator.

For a robot with configuration

$$
\vec q=
\begin{bmatrix}
q_1 &
q_2 &
\cdots &
q_n
\end{bmatrix}^{T},
$$

the joint-velocity vector is

$$
\dot{\vec q}=
\begin{bmatrix}
\dot q_1 &
\dot q_2 &
\cdots &
\dot q_n
\end{bmatrix}^{T}.
$$

The conventions used in this section follow [Mathematical notation and conventions](notation.md), [Rotations](rotations.md), [Homogeneous transformations](homogeneous-transformations.md), and [Forward kinematics](forward-kinematics.md).

## Geometric Jacobian

Moro uses the **geometric Jacobian** as the main differential-kinematics representation.

For the terminal frame, the geometric Jacobian satisfies

$$
\boxed{
\begin{bmatrix}
\vec v_{O_n}^{\,0}\\
\vec\omega_n^{\,0}
\end{bmatrix}
=
J(\vec q)\dot{\vec q}.
}
$$

Here,

- $\vec v_{O_n}^{\,0}$ is the linear velocity of the terminal origin,
- $\vec\omega_n^{\,0}$ is the angular velocity of the terminal link,
- both quantities are expressed in the base frame $\{0\}$.

For an $n$-degree-of-freedom robot,

$$
\boxed{
J(\vec q)\in\mathbb R^{6\times n}.
}
$$

The Jacobian is divided into linear and angular blocks:

$$
\boxed{
J=
\begin{bmatrix}
J_v\\
J_\omega
\end{bmatrix},
}
$$

where

$$
J_v\in\mathbb R^{3\times n}
$$

and

$$
J_\omega\in\mathbb R^{3\times n}.
$$

Therefore,

$$
\vec v_{O_n}^{\,0}
=
J_v(\vec q)\dot{\vec q},
$$

and

$$
\vec\omega_n^{\,0}
=
J_\omega(\vec q)\dot{\vec q}.
$$

## Geometric and analytical Jacobians

The geometric Jacobian should not be confused with an analytical Jacobian.

The geometric Jacobian maps joint rates directly to linear and angular velocity:

$$
\begin{bmatrix}
\vec v\\
\vec\omega
\end{bmatrix}
=
J_G(\vec q)\dot{\vec q}.
$$

An analytical Jacobian instead maps joint rates to the time derivative of a chosen pose parametrization. For example,

$$
\begin{bmatrix}
\dot{\vec p}\\
\dot{\boldsymbol\eta}
\end{bmatrix}
=
J_A(\vec q)\dot{\vec q},
$$

where $\boldsymbol\eta$ may represent Euler angles or another minimal orientation parametrization.

```{important} id="y1v91h"
Angular velocity $\vec\omega$ is not, in general, equal to the time derivative of a vector of Euler angles.

For this reason, the geometric and analytical Jacobians are different objects.
```

Moro's main Jacobian interface refers to the **geometric Jacobian**.

## Jacobian of an arbitrary point

Let $P$ be a point rigidly attached to link $k$.

Its position relative to frame $\{k\}$ is

$$
\vec r_P^{\,k}.
$$

The position of the same point with respect to the base frame is

$$
\tilde r_P^{\,0}
=
T_k^0
\tilde r_P^{\,k}.
$$

The geometric Jacobian associated with this point is written as

$$
\boxed{
J_P=
\begin{bmatrix}
J_{v,P}\\
J_{\omega,k}
\end{bmatrix}.
}
$$

It satisfies

$$
\boxed{
\begin{bmatrix}
\vec v_P^{\,0}\\
\vec\omega_k^{\,0}
\end{bmatrix}
=
J_P(\vec q)\dot{\vec q}.
}
$$

The upper block describes the linear velocity of the point $P$.

The lower block describes the angular velocity of the rigid link $k$ to which the point is attached.

```{note} id="psx3da"
A geometric point does not possess an independent angular velocity. The angular block of $J_P$ refers to the angular velocity of the rigid link containing the point.
```

## Column-by-column construction

The geometric Jacobian can be constructed one column at a time using the joint axes and frame-origin positions obtained from forward kinematics.

Under the classic Denavit--Hartenberg convention used by Moro,

$$
\boxed{
\text{joint }i
\text{ acts about or along }z_{i-1}.
}
$$

Therefore, the $i$-th Jacobian column depends on

$$
\vec z_{i-1}^{\,0}
$$

and

$$
\vec r_{O_{i-1}}^{\,0}.
$$

### Revolute joint

For a revolute joint $i$, the linear contribution to the velocity of a point $P$ is

$$
\boxed{
J_{v,P,i}
=
\vec z_{i-1}^{\,0}
\times
\left(
\vec r_P^{\,0}
-
\vec r_{O_{i-1}}^{\,0}
\right).
}
$$

The angular contribution is

$$
\boxed{
J_{\omega,P,i}
=
\vec z_{i-1}^{\,0}.
}
$$

Thus,

$$
\boxed{
J_{P,i}
=
\begin{bmatrix}
\vec z_{i-1}^{\,0}
\times
\left(
\vec r_P^{\,0}
-
\vec r_{O_{i-1}}^{\,0}
\right)
\\[2mm]
\vec z_{i-1}^{\,0}
\end{bmatrix}.
}
$$

### Prismatic joint

For a prismatic joint $i$, the linear contribution is

$$
\boxed{
J_{v,P,i}
=
\vec z_{i-1}^{\,0},
}
$$

while the angular contribution is zero:

$$
\boxed{
J_{\omega,P,i}
=
\vec 0.
}
$$

Therefore,

$$
\boxed{
J_{P,i}
=
\begin{bmatrix}
\vec z_{i-1}^{\,0}\\
\vec 0
\end{bmatrix}.
}
$$

## Joints that do not affect the point

If point $P$ is attached to link $k$, joints located after that link do not affect either the position of $P$ or the orientation of link $k$.

Therefore,

$$
\boxed{
i>k
\quad\Rightarrow\quad
J_{P,i}
=
\vec 0_{6\times1}.
}
$$

The complete point Jacobian may consequently be written as

$$
J_{P,i}
=
\begin{cases}
\begin{bmatrix}
\vec z_{i-1}^{\,0}\times
\left(
\vec r_P^{\,0}
-
\vec r_{O_{i-1}}^{\,0}
\right)
\\
\vec z_{i-1}^{\,0}
\end{bmatrix},
&
\text{revolute},\ i\leq k,
\\[6mm]
\begin{bmatrix}
\vec z_{i-1}^{\,0}\\
\vec 0
\end{bmatrix},
&
\text{prismatic},\ i\leq k,
\\[6mm]
\vec 0,
&
i>k.
\end{cases}
$$

## End-effector Jacobian

The terminal geometric Jacobian is a special case of the point Jacobian.

Taking

$$
P=O_n
$$

and

$$
k=n,
$$

gives

$$
\boxed{
J=J_{O_n}.
}
$$

Thus, the end-effector Jacobian uses the same geometric construction as the Jacobian of any other point attached to the robot.

## Linear Jacobian from forward kinematics

The linear block of the Jacobian can also be obtained by differentiating the forward-position map.

If

$$
\vec r_P^{\,0}
=
f_P(\vec q),
$$

then

$$
\vec v_P^{\,0}
=
\frac{d}{dt}\vec r_P^{\,0}.
$$

Applying the chain rule,

$$
\vec v_P^{\,0}
=
\frac{\partial\vec r_P^{\,0}}
{\partial\vec q}
\dot{\vec q}.
$$

Therefore,

$$
\boxed{
J_{v,P}
=
\frac{\partial\vec r_P^{\,0}}
{\partial\vec q}.
}
$$

This provides an alternative method for obtaining the linear Jacobian and is also useful for checking the geometric construction.

## Angular velocity and rotation matrices

The angular block of the geometric Jacobian should not be obtained by simply differentiating an orientation parametrization.

For a rotation matrix

$$
R_i^0,
$$

whose angular velocity is expressed in the base frame,

$$
\vec\omega_i^{\,0},
$$

the rotation-matrix derivative satisfies

$$
\boxed{
\dot R_i^0
=
[\vec\omega_i^{\,0}]_\times R_i^0.
}
$$

Equivalently,

$$
\boxed{
[\vec\omega_i^{\,0}]_\times
=
\dot R_i^0(R_i^0)^T.
}
$$

This relationship shows why angular velocity is a geometric quantity distinct from the derivatives of Euler angles or other orientation coordinates.

In practice, for serial manipulators the angular Jacobian is naturally constructed from the joint axes.

## Differential motion

For a small joint displacement

$$
\delta\vec q,
$$

the corresponding first-order Cartesian displacement is approximately

$$
\delta\vec x
\approx
J(\vec q)\delta\vec q.
$$

Similarly, in velocity form,

$$
\dot{\vec x}
=
J(\vec q)\dot{\vec q},
$$

where $\dot{\vec x}$ represents the geometric velocity composed of linear and angular components.

This local linear approximation is one of the main reasons why the Jacobian plays a central role in robot motion analysis and numerical inverse kinematics.

## Jacobian rank

The rank of the Jacobian indicates the number of independent instantaneous Cartesian velocity directions that the robot can generate.

For

$$
J\in\mathbb R^{6\times n},
$$

the rank satisfies

$$
\operatorname{rank}(J)
\leq
\min(6,n).
$$

The maximum attainable rank depends on the robot structure and on the task being considered.

A configuration is singular when the relevant Jacobian loses rank relative to its maximum attainable value:

$$
\boxed{
\operatorname{rank}J(\vec q)
<
\operatorname{rank}_{\max}J.
}
$$

This definition applies whether or not the Jacobian is square.

```{important} id="g1r24w"
The condition $\det J=0$ can only be used directly when the relevant Jacobian is square. Rank provides the more general definition of a kinematic singularity.
```

## Physical meaning of singularities

At a singular configuration, the manipulator loses the ability to generate one or more independent instantaneous Cartesian motions.

For some desired velocity

$$
\vec V_d,
$$

there may be no joint velocity satisfying

$$
J\dot{\vec q}
=
\vec V_d.
$$

Near a singularity, generating motion in certain Cartesian directions may also require very large joint velocities.

These effects are important in:

- velocity control,
- trajectory planning,
- numerical inverse kinematics,
- manipulability analysis.

## Task-dependent singularities

The relevant Jacobian depends on the task.

For a full spatial pose task, the geometric Jacobian

$$
J\in\mathbb R^{6\times n}
$$

may be the appropriate object.

For a position-only task, the relevant relationship is

$$
\vec v_P^{\,0}
=
J_{v,P}\dot{\vec q},
$$

and singularity analysis may instead focus on

$$
J_{v,P}\in\mathbb R^{3\times n}.
$$

Thus, a configuration may be singular with respect to one task while retaining useful motion capabilities for another.

## Null space

Joint velocities satisfying

$$
\boxed{
J(\vec q)\dot{\vec q}=0
}
$$

belong to the null space of the Jacobian.

Such joint motions produce zero instantaneous velocity for the task represented by $J$.

For redundant manipulators, nonzero null-space motions may exist even away from singular configurations.

These motions can later be exploited for secondary objectives, although null-space control is outside the scope of this section.

## Example: planar 2R manipulator

Consider the planar 2R manipulator introduced previously.

Its terminal position is

$$
\vec r_{O_2}^{\,0}
=
\begin{bmatrix}
a_1\cos q_1+a_2\cos(q_1+q_2)\\
a_1\sin q_1+a_2\sin(q_1+q_2)\\
0
\end{bmatrix}.
$$

For planar position analysis, only the $x$ and $y$ components are needed:

$$
\vec p=
\begin{bmatrix}
x\\
y
\end{bmatrix}.
$$

The linear Jacobian is obtained by differentiation:

$$
J_v
=
\frac{\partial(x,y)}
{\partial(q_1,q_2)}.
$$

Therefore,

$$
\boxed{
J_v
=
\begin{bmatrix}
-a_1\sin q_1-a_2\sin(q_1+q_2)
&
-a_2\sin(q_1+q_2)
\\
a_1\cos q_1+a_2\cos(q_1+q_2)
&
a_2\cos(q_1+q_2)
\end{bmatrix}.
}
$$

Thus,

$$
\begin{bmatrix}
\dot x\\
\dot y
\end{bmatrix}
=
J_v
\begin{bmatrix}
\dot q_1\\
\dot q_2
\end{bmatrix}.
$$

The same result can be obtained from the geometric construction using the two joint axes.

### Singular configurations

For this $2\times2$ position Jacobian,

$$
\det J_v
=
a_1a_2\sin q_2.
$$

The Jacobian loses rank when

$$
\sin q_2=0,
$$

that is,

$$
\boxed{
q_2=0
\quad\text{or}\quad
q_2=\pi
\pmod{2\pi}.
}
$$

At these configurations, the two links are collinear.

The manipulator loses one independent instantaneous direction of motion in the plane.

This illustrates the direct relationship between the algebraic rank of the Jacobian and the physical geometry of the robot.

## Differential kinematics in Moro

The `Robot` class exposes the main geometric Jacobian quantities directly.

### End-effector Jacobian

The complete geometric Jacobian of the end effector is available through

```python
robot.J
```

Mathematically,

$$
\boxed{
\texttt{robot.J}
\longleftrightarrow
J_{O_n}.
}
$$

Internally, this is the Jacobian of the origin of the final frame.

### Jacobian of an arbitrary point

For a point $P$ attached to link $i$, Moro provides

```python
robot.J_point(point, i)
```

where `point` contains the local coordinates

$$
\vec r_P^{\,i}.
$$

The returned matrix is the complete geometric Jacobian

$$
\boxed{
J_P
=
\begin{bmatrix}
J_{v,P}\\
J_{\omega,i}
\end{bmatrix}.
}
$$

The point coordinates are transformed internally to the base frame before constructing the Jacobian columns.

### Center-of-mass Jacobians

For the center of mass $C_i$ of link $i$, Moro provides

```python
robot.J_cm_i(i)
```

which corresponds to

$$
J_{C_i}
=
\begin{bmatrix}
J_{v,C_i}\\
J_{\omega,i}
\end{bmatrix}.
$$

The linear and angular blocks are also available separately:

```python
robot.Jv_cm_i(i)
robot.Jw_cm_i(i)
```

corresponding to

$$
J_{v,C_i}
$$

and

$$
J_{\omega,i},
$$

respectively.

These quantities are particularly useful in the dynamic formulation of serial manipulators.

## Related kinematic quantities in Moro

The geometric Jacobian construction relies directly on quantities already available through forward kinematics:

| Mathematical quantity | Moro API |
|---|---|
| $J_{O_n}$ | `J` |
| $J_P$ | `J_point(point, i)` |
| $J_{C_i}$ | `J_cm_i(i)` |
| $J_{v,C_i}$ | `Jv_cm_i(i)` |
| $J_{\omega,i}$ at $C_i$ | `Jw_cm_i(i)` |
| $\vec z_i^{\,0}$ | `z(i)` |
| $\vec r_{O_i}^{\,0}$ | `r_o(i)` |
| $T_i^0$ | `T_i0(i)` |

The use of `z(i)` and `r_o(i)` directly reflects the geometric construction

$$
\vec z_{i-1}^{\,0}
\times
\left(
\vec r_P^{\,0}
-
\vec r_{O_{i-1}}^{\,0}
\right)
$$

for revolute joints.

## Summary of conventions

The main differential-kinematics conventions used by Moro are:

| Concept | Convention |
|---|---|
| Main Jacobian type | Geometric Jacobian |
| Jacobian size | $6\times n$ |
| Structure | $J=[J_v^T\ J_\omega^T]^T$ |
| Velocity relation | $[\vec v^T\ \vec\omega^T]^T=J\dot{\vec q}$ |
| Expression frame | Base frame $\{0\}$ |
| Revolute linear column | $\vec z_{i-1}^{\,0}\times(\vec r_P^{\,0}-\vec r_{O_{i-1}}^{\,0})$ |
| Revolute angular column | $\vec z_{i-1}^{\,0}$ |
| Prismatic linear column | $\vec z_{i-1}^{\,0}$ |
| Prismatic angular column | $\vec 0$ |
| Joint after link $k$ | Zero column for a point on link $k$ |
| Linear Jacobian | $J_{v,P}=\partial\vec r_P^{\,0}/\partial\vec q$ |
| End-effector Jacobian | `Robot.J` |
| Arbitrary-point Jacobian | `Robot.J_point(point, i)` |
| Singular configuration | Loss of Jacobian rank |

Differential kinematics provides the local relationship between joint motion and Cartesian motion and forms the mathematical basis for singularity analysis, velocity control, and Jacobian-based inverse kinematics.