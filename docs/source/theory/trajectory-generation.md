# Trajectory Generation

A trajectory is a time-parameterized motion:

$$
x=x(t).
$$

Moro 0.5.0 provides compact numerical point-to-point trajectories using polynomial interpolation.

## Normalized time

Let

$$
T=t_f-t_0>0,
$$

and define

$$
\tau=\frac{t-t_0}{T}.
$$

Then

$$
0\le \tau\le1.
$$

Using normalized time makes the polynomial coefficients independent of the absolute time origin.

Physical-time derivatives satisfy

$$
\frac{d}{dt}
=
\frac{1}{T}\frac{d}{d\tau},
$$

and

$$
\frac{d^2}{dt^2}
=
\frac{1}{T^2}\frac{d^2}{d\tau^2}.
$$

## Linear interpolation

For endpoint positions $x_0$ and $x_f$,

$$
x(\tau)=x_0+(x_f-x_0)\tau.
$$

Therefore

$$
\dot x=\frac{x_f-x_0}{T},
$$

and

$$
\ddot x=0.
$$

## Cubic interpolation

For

$$
x(\tau)=c_0+c_1\tau+c_2\tau^2+c_3\tau^3,
$$

with endpoint velocities $v_0$ and $v_f$,

$$
c_0=x_0,
$$

$$
c_1=Tv_0,
$$

$$
c_2=3(x_f-x_0)-T(2v_0+v_f),
$$

$$
c_3=-2(x_f-x_0)+T(v_0+v_f).
$$

Velocity and acceleration are

$$
\dot x=
\frac{1}{T}
\left(
c_1+2c_2\tau+3c_3\tau^2
\right),
$$

$$
\ddot x=
\frac{1}{T^2}
\left(
2c_2+6c_3\tau
\right).
$$

## Quintic interpolation

For

$$
x(\tau)=
c_0+c_1\tau+c_2\tau^2+c_3\tau^3+c_4\tau^4+c_5\tau^5,
$$

with endpoint velocities $v_0,v_f$ and accelerations $a_0,a_f$,

$$
c_0=x_0,
\qquad
c_1=Tv_0,
\qquad
c_2=\frac{T^2}{2}a_0.
$$

Let

$$
\Delta x=x_f-x_0.
$$

Then

$$
c_3=
10\Delta x
-6Tv_0
-4Tv_f
-\frac{3}{2}T^2a_0
+\frac{1}{2}T^2a_f,
$$

$$
c_4=
-15\Delta x
+8Tv_0
+7Tv_f
+\frac{3}{2}T^2a_0
-T^2a_f,
$$

$$
c_5=
6\Delta x
-3Tv_0
-3Tv_f
-\frac{1}{2}T^2a_0
+\frac{1}{2}T^2a_f.
$$

## Classical zero-derivative profiles

For $x_0=0$, $x_f=1$, and zero supported derivative conditions:

### Linear

$$
s(\tau)=\tau.
$$

### Cubic

$$
s(\tau)=3\tau^2-2\tau^3.
$$

### Quintic

$$
s(\tau)=10\tau^3-15\tau^4+6\tau^5.
$$

This gives the teaching progression

```text
linear  -> position
cubic   -> position + velocity
quintic -> position + velocity + acceleration
```

## Vector trajectories

The same formulas apply componentwise to vector-valued coordinates.

For joint space,

$$
q,\dot q,\ddot q\in\mathbb R^{N\times n}.
$$

For Cartesian position space,

$$
p,v,a\in\mathbb R^{N\times3}.
$$

With zero derivative boundary conditions, a Cartesian position trajectory lies on the line segment between $p_0$ and $p_f$.

Nonzero derivative conditions can produce a curved path because the polynomial acts independently on each Cartesian component.

## Separation from robot algorithms

Trajectory generation is independent of robot kinematics.

A Cartesian trajectory may later be resolved using sequential inverse kinematics:

```python
cart = position_trajectory(...)
ik = solve_position_trajectory(robot, cart.p, q0=...)
```

Similarly, a joint-space trajectory can be passed directly to visualization as a time-major matrix.

No robot constraints or dynamic feasibility conditions are enforced by the trajectory generator itself.
