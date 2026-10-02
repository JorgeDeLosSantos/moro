# Detailed design: workspace sampling for Moro 0.5.0

This document records the detailed design for sampled end-effector workspace analysis in Moro 0.5.0.

The feature approximates the reachable Cartesian workspace under a finite joint-space domain:

\[
\mathcal W = \{p(q)\mid q\in\mathcal Q\}.
\]

The 0.5.0 implementation is intentionally sampling-based. It does not attempt analytical workspace computation, exact boundary reconstruction, area/volume estimation, or dextrous workspace analysis.

The computational feature lives in:

\`\`\`text
moro/workspace.py
\`\`\`

Visualization belongs to the visualization subsystem.

## Status

Detailed design complete for the 0.5.0 scope described below.

## 1. Design principles

The workspace feature should be:

- numerical;
- simple;
- reproducible;
- compatible with revolute, prismatic, and mixed serial manipulators;
- explicit about the finite joint-space domain being sampled;
- transparent about the difference between a sampled point cloud and an exact workspace boundary.

The central object is a sampled dataset linking joint configurations to corresponding end-effector positions.

## 2. Public sampling API

The planned public function is:

\`\`\`python
sample_workspace(
    robot,
    *,
    samples=1000,
    joint_limits=None,
    parameters=None,
    seed=None,
)
\`\`\`

The return value is a \`Workspace\` object.

Uniform random sampling in joint space is the only sampling strategy planned for 0.5.0.

Grid, adaptive, quasi-random, and boundary-directed sampling are outside scope.

## 3. Workspace model

Let the robot have:

\[
n = robot.dof.
\]

The sampled joint configurations are stored as:

\[
Q=
\begin{bmatrix}
q^{(1)T}\\
q^{(2)T}\\
\vdots\\
q^{(N)T}
\end{bmatrix}
\in\mathbb{R}^{N\times n}.
\]

The corresponding end-effector positions are stored as:

\[
P=
\begin{bmatrix}
p(q^{(1)})^T\\
p(q^{(2)})^T\\
\vdots\\
p(q^{(N)})^T
\end{bmatrix}
\in\mathbb{R}^{N\times3}.
\]

The core correspondence is:

\[
\boxed{
P_k = p(Q_k)
}
\]

for every sample index \(k\).

This one-to-one correspondence must be preserved exactly.

## 4. Workspace result type

The planned data container is:

\`\`\`python
@dataclass
class Workspace:
    points: np.ndarray
    configurations: np.ndarray
    joint_limits: tuple
    seed: Optional[int] = None

    @property
    def samples(self) -> int:
        ...

    @property
    def dof(self) -> int:
        ...

    @property
    def bounds(self):
        ...
\`\`\`

\`Workspace\` represents sampled kinematic data, not a solver outcome and not an exact geometric boundary model.

## 5. Public numerical representation

NumPy arrays are the public representation for sampled data.

This is intentional because workspace datasets may contain thousands or tens of thousands of samples and should support efficient:

- slicing;
- plotting;
- numerical analysis;
- downstream evaluation;
- vectorized min/max operations.

The public shapes are fixed as:

\`\`\`text
points.shape == (N, 3)
configurations.shape == (N, robot.dof)
\`\`\`

Planar robots still store three Cartesian coordinates.

No \`(N, 2)\` special case is introduced.

## 6. Workspace invariants

A valid \`Workspace\` must satisfy:

\[
N\ge1,
\qquad
n\ge1.
\]

The point array must satisfy:

\`\`\`text
points.ndim == 2
points.shape == (N, 3)
all point values are finite and real
\`\`\`

The configuration array must satisfy:

\`\`\`text
configurations.ndim == 2
configurations.shape == (N, n)
all configuration values are finite and real
\`\`\`

Also:

\[
points.shape[0]
=
configurations.shape[0].
\]

The stored joint limits must satisfy:

\`\`\`text
len(joint_limits) == n
\`\`\`

and each joint configuration must lie within the stored effective limits.

The dataclass does not validate that each point is actually the forward-kinematics result of its configuration because that would require access to the robot and would be unnecessarily expensive.

That relationship is guaranteed by \`sample_workspace()\`.

## 7. Defensive copying and mutability

Input arrays should be converted to independent NumPy float arrays during \`Workspace\` construction.

Conceptually:

\`\`\`python
self.points = np.array(points, dtype=float, copy=True)
self.configurations = np.array(
    configurations,
    dtype=float,
    copy=True,
)
\`\`\`

This protects the workspace from later mutation of the caller's original arrays.

The arrays inside \`Workspace\` remain mutable in 0.5.0.

The object is not intended to be a deeply immutable value object.

## 8. Derived property: samples

The number of samples is derived from the data:

\`\`\`python
@property
def samples(self):
    return self.points.shape[0]
\`\`\`

It is not stored redundantly.

By invariant:

\[
workspace.samples
=
workspace.configurations.shape[0].
\]

## 9. Derived property: dof

The number of degrees of freedom represented by the dataset is derived from:

\`\`\`python
@property
def dof(self):
    return self.configurations.shape[1]
\`\`\`

It is not stored redundantly.

## 10. Derived property: bounds

\`Workspace.bounds\` returns the axis-aligned Cartesian bounds observed in the sampled point cloud:

\`\`\`python
(
    (xmin, xmax),
    (ymin, ymax),
    (zmin, zmax),
)
\`\`\`

Conceptually:

\[
x_{\min}=\min_k P_{k,x},
\qquad
x_{\max}=\max_k P_{k,x},
\]

with analogous definitions for \(y\) and \(z\).

These are sample bounds only.

They do not imply an exact workspace boundary.

The bounds are computed dynamically rather than cached because the public arrays remain mutable.

Planar datasets may legitimately produce a degenerate Cartesian interval such as:

\`\`\`python
(0.0, 0.0)
\`\`\`

for one axis.

## 11. Stored joint limits

\`Workspace.joint_limits\` stores the effective joint limits actually used to generate the samples.

The normalized public form is:

\`\`\`python
(
    (lower_1, upper_1),
    (lower_2, upper_2),
    ...
)
\`\`\`

with Python floats.

This allows the dataset to retain the joint-space domain from which it was generated, including when explicit overrides were used.

## 12. Seed metadata

\`Workspace.seed\` stores the seed used to generate the sample set when known.

It may be:

\`\`\`text
None
\`\`\`

or an integer that is not a Boolean value.

The result does not store the entire RNG state.

The seed is metadata only; \`Workspace\` does not attempt to verify that its arrays were actually generated from that seed.

## 13. Joint-limit precedence

The precedence rule is:

\[
\boxed{
sample\_workspace(...,\ joint\_limits=...)
>
robot.joint\_limits
}
\]

If:

\`\`\`python
joint_limits=None
\`\`\`

the function uses:

\`\`\`python
robot.joint_limits
\`\`\`

If explicit limits are supplied, they completely replace the robot limits for that sampling call.

The robot object must not be modified.

## 14. No partial joint-limit overrides

The 0.5.0 API does not support partial overrides such as:

\`\`\`python
[
    None,
    (-1.0, 1.0),
    None,
]
\`\`\`

The caller supplies either:

- no override; or
- a complete list/tuple of limits for all joints.

This keeps the precedence rule and result provenance simple.

## 15. Joint-limit format

Joint limits use the same conceptual format already used throughout Moro:

\`\`\`python
[
    (q1_min, q1_max),
    (q2_min, q2_max),
    ...
]
\`\`\`

The number of limit pairs must equal:

\[
robot.dof.
\]

Each pair must contain exactly two scalar values.

## 16. Finite joint-space domain

Workspace sampling requires a finite domain.

Each joint interval must satisfy:

\[
-\infty
<
q_{i,\min}
<
q_{i,\max}
<
+\infty.
\]

Therefore the following are invalid:

\`\`\`text
(-inf, inf)
(0, inf)
(nan, 1)
(1, 1)
(2, 1)
\`\`\`

All bounds must be real and finite.

## 17. Degenerate joint intervals

Intervals satisfying:

\[
q_{min}=q_{max}
\]

are not accepted in 0.5.0.

Although such an interval could mathematically represent a fixed coordinate, the robot still declares the joint as an active DOF and the resulting sampling semantics would be ambiguous.

If support for intentionally frozen joints is needed later, it should be introduced explicitly.

## 18. Revolute and prismatic joints

The sampler does not use different probability distributions for revolute and prismatic joints.

For every joint:

\[
q_i
\sim
\mathcal U(q_{i,\min},q_{i,\max}).
\]

Joint type determines the physical interpretation of the coordinate and its units, not the sampling rule.

This applies uniformly to revolute, prismatic, and mixed robots.

## 19. Revolute periodicity

Workspace sampling does not add special angular wrapping behavior.

For example:

\[
[-\pi,\pi]
\]

is treated as the requested numerical interval.

Likewise, a user-defined interval such as:

\[
[-2\pi,2\pi]
\]

is accepted even though it may contain physically repeated revolute configurations.

The sampler respects the requested joint-space domain rather than attempting to remove periodic redundancy.

## 20. Robot default limits

Current Moro robot models provide finite default joint limits.

These defaults may therefore be used directly by:

\`\`\`python
sample_workspace(robot)
\`\`\`

The workspace implementation does not inspect private state such as whether joint limits were explicitly assigned.

However, documentation should make clear that default limits are generic modeling conveniences and may not represent the physical range of a specific robot.

This is particularly important for prismatic joints.

## 21. Sampling distribution

The joint-space domain is:

\[
\mathcal Q
=
[q_{1,\min},q_{1,\max}]
\times
\cdots
\times
[q_{n,\min},q_{n,\max}].
\]

Each sampled configuration is drawn independently from the product-uniform distribution:

\[
q^{(k)}
\sim
\mathcal U(\mathcal Q).
\]

The intended NumPy direction is:

\`\`\`python
rng = np.random.default_rng(seed)

Q = rng.uniform(
    low=lower_bounds,
    high=upper_bounds,
    size=(samples, robot.dof),
)
\`\`\`

The upper endpoint follows NumPy's standard continuous-uniform behavior and is not forced into the sample.

## 22. Uniform in joint space, not Cartesian space

The point cloud is generated by uniform sampling in joint coordinates.

It is not uniformly distributed in Cartesian space.

A high density of points in one Cartesian region does not imply that region has greater geometric workspace measure.

The forward-kinematics mapping may strongly distort joint-space density.

This distinction should be explicitly documented.

## 23. Sampling strategy scope

Only independent uniform random sampling is included in 0.5.0.

The following remain outside scope:

- regular grids;
- adaptive sampling;
- Latin hypercube sampling;
- Sobol sampling;
- Halton sampling;
- direct boundary search;
- importance sampling.

These can be added later without changing the basic \`Workspace\` data model.

## 24. Sample-count validation

\`samples\` must be:

- an integer;
- not a Boolean;
- strictly positive.

Therefore values such as:

\`\`\`text
0
-10
3.5
True
\`\`\`

are invalid.

No arbitrary public maximum number of samples is imposed.

\`samples=1\` is valid.

## 25. Reproducibility

Sampling uses a local NumPy generator created from:

\`\`\`python
np.random.default_rng(seed)
\`\`\`

Two calls with the same:

- robot model;
- parameters;
- joint limits;
- sample count;
- seed;

should reproduce the same configurations and corresponding points within the same relevant numerical environment.

The implementation must not depend on NumPy's global random state.

No guarantee of bit-for-bit compatibility across all future NumPy versions is required.

## 26. Symbolic robot geometry

Robot geometry may contain symbolic parameters.

The intended preparation pattern mirrors existing numerical IK infrastructure:

1. extract the symbolic end-effector position;
2. apply \`parameters\`;
3. verify that only joint symbols remain free;
4. compile a numerical callable with \`lambdify\`;
5. evaluate the sampled configurations numerically.

The symbolic position expression is:

\[
p(q)=robot.T[:3,3].
\]

## 27. Parameters

\`parameters\` resolves symbolic robot-model quantities, for example:

\`\`\`python
parameters = {
    l1: 0.5,
    l2: 0.3,
}
\`\`\`

The substitution is applied to a local expression.

The original robot symbolic model must remain unchanged.

After substitution, the only remaining free symbols allowed in the position expression are the robot joint variables.

Any unresolved geometric or model parameter must cause a clear validation failure before sampling begins.

## 28. Numerical FK preparation

The end-effector position callable should be prepared once per \`sample_workspace()\` call:

\`\`\`python
qs = tuple(robot.qs)

position_sym = robot.T[:3, 3]
position_sym = _apply_parameters(
    position_sym,
    parameters,
)

_validate_free_symbols(
    [position_sym],
    qs,
)

position_func = lambdify(
    qs,
    position_sym,
    modules="numpy",
)
\`\`\`

No repeated symbolic substitution or lambdification should occur per sample.

## 29. Initial evaluation strategy

The 0.5.0 implementation should initially evaluate the compiled FK one sampled configuration at a time.

Conceptually:

\`\`\`python
points = np.empty((samples, 3), dtype=float)

for k, q in enumerate(configurations):
    points[k] = _evaluate_workspace_position(
        position_func,
        q,
    )
\`\`\`

This deliberately prioritizes predictable output normalization over aggressive vectorization.

Some lambdified expressions may return scalars for constant coordinates and arrays for variable coordinates when evaluated in bulk, which complicates robust broadcasting.

If profiling later demonstrates a meaningful bottleneck, vectorized FK evaluation may be introduced internally without changing the public API.

## 30. Position evaluation normalization

Each FK evaluation must produce exactly three real finite Cartesian components.

The numerical result should be normalized conceptually as:

\`\`\`python
p = np.asarray(
    position_func(*q),
    dtype=float,
).reshape(-1)
\`\`\`

and then validated to ensure:

\`\`\`text
p.size == 3
all values finite
all values real
\`\`\`

The stored result is always:

\`\`\`text
points[k].shape == (3,)
\`\`\`

No \`(3,1)\` entries are retained inside the dataset.

## 31. Numerical failures

If any sampled configuration produces:

- NaN;
- Inf;
- a complex-valued Cartesian position;
- an unexpected output shape;
- another unrecoverable FK numerical error;

the entire \`sample_workspace()\` call should fail.

Samples must not be silently discarded or replaced.

This preserves the invariant:

\[
Workspace.samples
=
samples\ requested.
\]

The error should identify the failed sample index where practical.

## 32. No sample deduplication or reordering

The output arrays preserve sampling order.

No sorting, filtering, or deduplication is performed.

If:

\[
q^{(i)}\neq q^{(j)}
\]

but:

\[
p(q^{(i)})=p(q^{(j)}),
\]

both rows are retained.

This is intentional because the dataset represents sampled configurations and their image under forward kinematics, not merely a mathematical set of unique Cartesian points.

## 33. No compiled-FK cache in 0.5.0

The numerical FK callable is prepared once per sampling call.

No cross-call caching is added in 0.5.0.

Caching would require stable model identity, parameter-key semantics, and invalidation behavior that are not justified for the initial feature.

## 34. Internal structure

The module may use small private helpers such as:

\`\`\`python
_prepare_workspace_joint_limits(...)
_prepare_workspace_position_model(...)
_evaluate_workspace_position(...)
\`\`\`

The exact helper names may evolve during implementation.

Private validation helpers from unrelated modules should not be imported merely to avoid a few duplicated lines.

If joint-limit or parameter validation later becomes broadly shared across multiple modules, a common internal utility layer may be introduced separately.

## 35. Public visualization API

Workspace visualization belongs to the visualization subsystem.

The planned public entry point is:

\`\`\`python
from moro.visualization import plot_workspace
\`\`\`

with:

\`\`\`python
plot_workspace(
    workspace,
    *,
    projection="auto",
    ax=None,
    figsize=(8, 6),
    marker_size=8,
    alpha=0.5,
)
\`\`\`

The function returns:

\`\`\`python
fig, ax
\`\`\`

consistent with Moro's Matplotlib visualization behavior.

## 36. Visualization backend scope

Workspace visualization uses Matplotlib only in 0.5.0.

No \`backend=\` selector is added while there is only one supported backend.

Three.js workspace visualization is deferred.

This keeps the initial feature focused on sampled kinematic data rather than frontend infrastructure.

## 37. Separation from RobotVisualizer

\`RobotVisualizer\` remains responsible for rendering robot configurations through evaluated scene data.

Workspace visualization is not added as:

\`\`\`python
RobotVisualizer.plot_workspace(...)
\`\`\`

because a \`Workspace\` is an independent point-cloud dataset and does not require a robot instance for plotting.

Likewise, no:

\`\`\`python
workspace.plot()
\`\`\`

method is added.

Computation/data and visualization remain separated.

## 38. Projection modes

\`plot_workspace()\` supports:

\`\`\`text
"auto"
"xy"
"xz"
"yz"
"3d"
\`\`\`

Explicit 2D projections select the corresponding Cartesian coordinates.

\`"3d"\` always creates a spatial point-cloud view.

## 39. Automatic planar detection

For:

\`\`\`python
projection="auto"
\`\`\`

the function may select a 2D projection only when one Cartesian coordinate is approximately constant across the sampled point cloud.

Let:

\[
s_x=x_{\max}-x_{\min},
\]

\[
s_y=y_{\max}-y_{\min},
\]

\[
s_z=z_{\max}-z_{\min}.
\]

A coordinate may be treated as constant using a small internal absolute/relative tolerance based on the largest observed span.

The exact internal tolerance may use values such as:

\`\`\`text
rtol = 1e-9
atol = 1e-12
\`\`\`

but those thresholds are not exposed publicly in 0.5.0.

If no axis is approximately constant, \`"auto"\` selects \`"3d"\`.

## 40. No arbitrary-plane detection

Automatic projection does not use PCA, SVD, or arbitrary fitted planes.

Even if a workspace lies on a rotated plane, \`"auto"\` remains 3D unless one base-frame coordinate is approximately constant.

This preserves direct interpretation in base-frame Cartesian coordinates.

## 41. Degenerate lower-dimensional samples

If more than one Cartesian axis is approximately constant, \`"auto"\` may choose a deterministic 2D projection containing the varying axis.

No separate 1D plotting mode is introduced.

The workspace plotting API does not attempt to classify the intrinsic dimension formally.

## 42. Point-cloud visualization only

Sampled workspace data is plotted using scatter points.

The plotting function must not:

- connect samples with lines;
- triangulate the point set;
- compute a convex hull;
- estimate a boundary;
- fill an area;
- construct a surface.

Random sample order has no geometric trajectory meaning.

The visualization should therefore communicate only the observed sampled point cloud.

## 43. 2D rendering

For explicit or automatic 2D projections, the point cloud is drawn with \`ax.scatter()\`.

Axis labels must correspond to the selected base-frame coordinates:

\`\`\`text
xy -> X, Y
xz -> X, Z
yz -> Y, Z
\`\`\`

The plot should use equal geometric aspect:

\`\`\`python
ax.set_aspect("equal", adjustable="box")
\`\`\`

so that Cartesian geometry is not visually distorted.

## 44. 3D rendering

For \`projection="3d"\`, create or use a Matplotlib 3D axis and draw the samples with \`ax.scatter()\`.

The visualization should use equal Cartesian scaling as closely as practical.

A suitable strategy is to compute the sample bounds, determine the Cartesian center:

\[
c_i=\frac{x_{i,\min}+x_{i,\max}}{2},
\]

and a common half-extent:

\[
h=
\frac{
\max(s_x,s_y,s_z)
}{2}.
\]

Then apply the same extent to all three axes around the corresponding center.

## 45. Visualization style scope

The existing \`VisualizationStyle\` class is not extended with workspace-specific fields in 0.5.0.

Its current concepts are robot-scene oriented:

- frames;
- links;
- joints;
- base;
- trajectory.

Workspace visualization has only a few simple visual parameters, so those remain direct arguments to \`plot_workspace()\`.

A dedicated \`WorkspaceStyle\` class may be introduced later if the feature grows.

## 46. Workspace plot options

The initial public options are limited to:

\`\`\`text
projection
ax
figsize
marker_size
alpha
\`\`\`

No semantic color mapping such as:

\`\`\`python
color_by="manipulability"
\`\`\`

is added in 0.5.0.

No robot configuration is overlaid automatically.

No title is imposed by default.

## 47. Existing-axis behavior

If \`ax=None\`, \`plot_workspace()\` creates the appropriate Matplotlib axis.

If the user supplies an axis, it must be compatible with the requested projection.

Clear incompatible cases, such as using a 2D axis with \`projection="3d"\`, should raise a \`ValueError\`.

The function returns the associated figure and axis.

## 48. Validation tests

Tests should cover:

- positive integer \`samples\`;
- rejection of zero/negative/non-integral/Boolean sample counts;
- \`seed=None\`;
- integer seeds;
- invalid Boolean/non-integral seeds;
- correct number of joint-limit pairs;
- finite real joint bounds;
- rejection of reversed bounds;
- rejection of degenerate bounds;
- complete explicit override precedence;
- no mutation of \`robot.joint_limits\`.

## 49. Sampling tests

Sampling tests should verify:

- output configuration shape \((N,n)\);
- every sampled configuration lies within its effective limits;
- same seed and same inputs reproduce the same configurations;
- corresponding points are reproducible;
- different sampling calls do not depend on NumPy global RNG state;
- \`samples=1\` is valid;
- no hidden filtering changes the requested sample count.

Deep statistical testing of uniformity is not required because it would make tests unnecessarily fragile.

A lightweight sanity test may confirm that sufficiently many samples span the requested interval rather than collapsing to a repeated value.

## 50. Point/configuration consistency tests

For selected rows \(k\), tests should recompute forward kinematics independently and verify:

\[
workspace.points[k]
\approx
p(workspace.configurations[k]).
\]

This explicitly protects the 1:1 correspondence contract.

Tests must not assume sampled Cartesian points are unique.

## 51. Workspace dataclass tests

Direct \`Workspace\` construction tests should cover:

- valid \((N,3)\) point arrays;
- invalid point shapes;
- valid \((N,n)\) configuration arrays;
- mismatched sample counts;
- non-finite point/configuration values;
- invalid joint-limit length;
- invalid joint-limit intervals;
- configurations outside stored joint limits;
- defensive copying of input arrays;
- derived \`samples\`;
- derived \`dof\`;
- correct dynamic \`bounds\`.

## 52. Symbolic-parameter tests

Tests should include:

- symbolic robot geometry successfully resolved through \`parameters\`;
- missing required model parameters;
- multiple symbolic parameters;
- preservation of the original symbolic robot model after sampling.

Unresolved parameters must fail before sample generation or FK iteration.

## 53. Robot-structure coverage

Representative tests should include:

- planar revolute robot;
- spatial revolute robot;
- purely prismatic robot;
- mixed revolute/prismatic robot;
- symbolic-parameter model;
- explicit joint-limit override.

The same sampling semantics apply to all joint types.

## 54. Planar 2R sanity test

A planar 2R robot is the preferred introductory acceptance case.

Expected properties include:

\[
z\approx0
\]

for every sampled point.

The radial distance should not exceed the obvious geometric maximum:

\[
r\lesssim l_1+l_2
\]

within numerical tolerance.

The test should not attempt to prove or reconstruct the exact analytical workspace boundary.

## 55. Prismatic sanity test

A simple prismatic model should demonstrate that Cartesian displacement follows the sampled prismatic range as expected.

This provides a transparent check that the same API works for non-revolute joints.

## 56. Numerical-failure tests

Tests should verify that a single invalid FK evaluation aborts the workspace computation rather than producing a smaller dataset or inserting invalid points.

Failure cases may include:

- NaN output;
- Inf output;
- complex output;
- malformed evaluated position shape.

The requested sample count must never silently change.

## 57. Visualization tests

Visualization tests should avoid pixel/snapshot comparisons.

They should verify:

- \`projection="auto"\` chooses XY when Z is approximately constant;
- analogous behavior for XZ and YZ;
- genuinely spatial data uses 3D;
- explicit \`"xy"\`, \`"xz"\`, \`"yz"\`, and \`"3d"\` modes work;
- returned \`fig, ax\` are valid;
- the plotted sample count matches \`workspace.samples\`;
- coordinate labels are correct;
- equal 2D aspect is applied;
- 3D axes are created when required;
- compatible existing axes can be reused;
- invalid projection values raise \`ValueError\`;
- clearly incompatible supplied axes raise \`ValueError\`.

## 58. Documentation examples

At least three user-facing examples should accompany the feature.

### 58.1 Planar 2R workspace

Demonstrate:

1. robot creation;
2. joint limits;
3. workspace sampling;
4. inspection of \`samples\`, \`dof\`, and \`bounds\`;
5. automatic 2D visualization.

This is the primary introductory example.

### 58.2 Spatial or mixed-joint workspace

Use a spatial or revolute/prismatic manipulator to demonstrate that the same API naturally produces a 3D point cloud.

### 58.3 Effect of joint limits

Sample the same robot with two different finite joint-space domains.

Show that:

\[
\mathcal W
\]

depends on the chosen domain:

\[
\mathcal Q.
\]

The example should emphasize that workspace is not determined by link geometry alone.

## 59. Relationship with later analysis

Storing \`Workspace.configurations\` intentionally enables future or user-driven post-processing such as:

- manipulability evaluation;
- singularity inspection;
- coloring by another metric;
- filtering configurations;
- studying multiple joint-space branches.

However, \`sample_workspace()\` does not automatically compute any of those quantities.

The workspace feature remains focused on sampled position reachability.

## 60. Explicitly outside Moro 0.5.0

The following are outside the present workspace scope:

- analytical workspace computation;
- exact boundary reconstruction;
- regular grid sampling;
- adaptive sampling;
- quasi-random sampling;
- area estimation;
- volume estimation;
- convex-hull workspace representation;
- alpha shapes;
- surface reconstruction;
- dextrous workspace;
- orientation workspace;
- pose workspace;
- constant-orientation workspace;
- obstacles;
- collision checking;
- automatic manipulability maps;
- automatic singularity maps;
- arbitrary intermediate-link workspace sampling;
- automatic robot overlays;
- Three.js workspace rendering;
- arbitrary-plane projection detection.

## 61. Acceptance criteria

The workspace block is ready for implementation when:

1. \`sample_workspace()\` samples exactly \(N\) configurations from a finite joint-space domain;
2. sampling is independent and uniform in joint coordinates;
3. the same seed and inputs reproduce the same dataset in the same numerical environment;
4. explicit joint limits fully override robot limits for that call without modifying the robot;
5. all effective joint intervals are finite and strictly ordered;
6. revolute, prismatic, and mixed joints use the same sampling semantics;
7. symbolic model parameters are resolved before numerical sampling;
8. FK is compiled once per call rather than symbolically reevaluated per sample;
9. every stored point corresponds exactly to the configuration at the same row index;
10. no sample is silently discarded, reordered, or deduplicated;
11. any unrecoverable FK numerical failure aborts the full sampling call;
12. \`Workspace.points\` has shape \((N,3)\);
13. \`Workspace.configurations\` has shape \((N,n)\);
14. \`samples\`, \`dof\`, and \`bounds\` are derived properties;
15. \`joint_limits\` stores the normalized effective domain used for sampling;
16. \`plot_workspace()\` visualizes only the sampled point cloud;
17. automatic 2D plotting is limited to base-axis-aligned planar data;
18. genuinely spatial data is shown in 3D;
19. the initial visualization backend is Matplotlib only;
20. no 0.5.0 implementation introduces exact boundaries, hulls, volumes, manipulability maps, pose workspace, collision handling, adaptive sampling, or Three.js workspace rendering.
