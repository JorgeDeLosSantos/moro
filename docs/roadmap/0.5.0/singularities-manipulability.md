# Singularities and manipulability

## Status

Accepted as a candidate feature for Moro 0.5.0.

## Objective

Provide numerical analysis of local kinematic capability through the selected task Jacobian, reusing the same task-space conventions as differential kinematics.

The initial implementation should remain in `moro/differential_kinematics.py` rather than creating a separate module.

## Preliminary public API

```python
singular_values(robot, q, *, task="twist", parameters=None)
jacobian_rank(robot, q, *, task="twist", parameters=None, tol=None)
condition_number(robot, q, *, task="twist", parameters=None)
is_singular(robot, q, *, task="twist", parameters=None, tol=None)
manipulability(robot, q, *, task, parameters=None)
```

The analysis should be based primarily on SVD:

\[
J_{\mathrm{task}}=U\Sigma V^T.
\]

## Singularity and rank

For a selected task, a configuration is singular when

\[
\operatorname{rank}(J_{\mathrm{task}})<\min(m,n).
\]

Singularity is task dependent, so reduced tasks and full twist tasks may classify the same configuration differently. `is_singular()` should diagnose rank loss rather than raising because the Jacobian is singular.

A scale-aware numerical tolerance should be used when `tol=None`; exact policy is deferred.

## Condition number

Conditioning should be derived from singular values, conceptually

\[
\kappa(J)=\frac{\sigma_{\max}}{\sigma_{\min}}.
\]

Effectively singular Jacobians should return `inf` rather than raising.

## Manipulability

The initial scalar metric is Yoshikawa velocity manipulability:

\[
w(q)=\sqrt{\det(J_{\mathrm{task}}J_{\mathrm{task}}^T)}.
\]

`task` is intentionally required explicitly for `manipulability()` because the value depends strongly on the selected task and full-twist measures mix translational and angular scales. Automatic weighting or characteristic-length normalization is not planned.

For task dimension greater than available joint dimension, the measure is naturally zero because the task Jacobian is rank deficient.

## Relationship with workspace

Workspace sampling should not automatically attach manipulability or singularity labels. Stored sampled configurations allow these metrics to be evaluated later without coupling the APIs.

## Symbolic and numerical behavior

The initial analysis API is numerical, with symbolic robot parameters resolved through `parameters`. General symbolic singularity-condition solving is outside scope.

## Examples and tests

Cover rank loss, singular/nonsingular configurations, condition-number growth, singular values, planar Yoshikawa manipulability, reduced tasks, revolute/prismatic joints, and symbolic parameters.

## Explicitly outside 0.5.0

- symbolic singularity conditions/classification;
- manipulability ellipsoids;
- isotropy indices as separate API;
- gradients and optimization;
- null-space manipulability maximization;
- automatic workspace maps;
- dynamic or force manipulability;
- automatic translational/rotational normalization.

## Deferred detailed-design decisions

Rank tolerance, `is_singular()` tolerance relationship, treatment of tiny singular values, roundoff in the Yoshikawa determinant, validation/messages, shared helpers with velocity IK, caching, and exact singular-value return policy remain deferred.
