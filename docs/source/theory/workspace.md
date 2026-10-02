# Sampled Workspace Analysis

For a serial manipulator with joint coordinates

$$
q\in\mathcal Q,
$$

the reachable end-effector position workspace is

$$
\mathcal W
=
\{p(q)\mid q\in\mathcal Q\}.
$$

In Moro 0.5.0, this set is approximated numerically through random sampling.

## Joint-space sampling

Let the robot have $n$ degrees of freedom and finite limits

$$
q_{i,\min}<q_i<q_{i,\max}.
$$

The sampling domain is the Cartesian product

$$
\mathcal Q
=
[q_{1,\min},q_{1,\max}]
\times\cdots\times
[q_{n,\min},q_{n,\max}].
$$

Each sampled configuration is drawn independently from the product-uniform distribution:

$$
q^{(k)}\sim\mathcal U(\mathcal Q).
$$

The corresponding Cartesian point is obtained through forward kinematics:

$$
p^{(k)}=p(q^{(k)}).
$$

Moro stores both arrays in matching row order:

$$
Q\in\mathbb R^{N\times n},
\qquad
P\in\mathbb R^{N\times3}.
$$

Thus

$$
P_k=p(Q_k)
$$

for every sampled row.

## Uniform joint-space density is not uniform Cartesian density

Uniform sampling of $q$ does not imply uniform density in Cartesian coordinates.

The nonlinear map

$$
q\mapsto p(q)
$$

may stretch some regions and compress others.

Therefore point density should not be interpreted as a direct measure of workspace area or volume.

## Finite domains

A sampled workspace is meaningful only after specifying the finite joint-space domain being explored.

The same robot geometry can produce different sampled workspaces when its allowed joint ranges change.

This is especially important for prismatic joints and mechanically restricted revolute joints.

## Approximation rather than exact boundary reconstruction

Random sampling provides evidence of reachability at the observed points, but does not recover the exact analytical boundary of $\mathcal W$.

Observed sample bounds

$$
(x_{\min},x_{\max}),
\quad
(y_{\min},y_{\max}),
\quad
(z_{\min},z_{\max})
$$

are therefore bounds of the sampled cloud rather than guaranteed exact workspace extrema.

Increasing the sample count can improve coverage but does not change the fundamental approximate nature of the method.

## Revolute and prismatic coordinates

The same sampling rule is applied to both joint types:

$$
q_i\sim\mathcal U(q_{i,\min},q_{i,\max}).
$$

For revolute joints, the coordinate is angular; for prismatic joints, it is linear.

No periodicity correction or duplicate-angle removal is performed. The sampler respects the numerical interval requested by the user.

## Reproducibility

A seeded pseudo-random generator allows two equivalent sampling calls to reproduce the same sampled configurations in the same relevant numerical environment.

This is useful for:

- teaching examples;
- regression tests;
- comparing different post-processing metrics on the same configuration set.

## Symbolic-to-numerical workflow

The robot remains symbolic.

For workspace sampling, Moro:

1. extracts the symbolic end-effector position;
2. applies any supplied geometric parameters;
3. verifies that no unresolved model parameters remain;
4. compiles a numerical forward-kinematics callable once;
5. evaluates that callable for the sampled configurations.

This preserves the symbolic model while avoiding repeated symbolic substitution inside the sampling loop.

## Relationship to later analysis

Because each Cartesian point retains its generating joint configuration, a sampled workspace can later support user-driven analyses such as:

- manipulability evaluation;
- singularity checks;
- filtering by configuration;
- branch comparison.

Moro 0.5.0 does not compute these automatically. Workspace sampling remains focused on positional reachability.
