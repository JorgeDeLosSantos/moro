# API-001 — Visualizer configuration/parameter normalization

> **Status:** A1 private helpers merged through PR #52; A2 public
> visualizer integration merged through PR #53. A3 documents the new API,
> provides notebook-ready RR/RP examples and drafts the 0.7.0 migration
> notes. Moro 0.7.0 as a whole remains unreleased.

## Public API integrated in A2

Distinguish **joint configurations** from **constant DH model parameters**:

```python
viz.plot([0.5, -0.3], parameters={l1: 1.0, l2: 0.8})
viz.animate(traj.q, times=traj.t, parameters={l1: 1.0, l2: 0.8})
```

The first positional arguments are to be renamed directly:
- `RobotVisualizer.plot(num_vals, ...)` → `plot(configuration, ...)`.
- `RobotVisualizer.animate(num_vals_list, ...)` → `animate(configurations, ...)`.

No legacy keyword aliases, sentinels, or deprecation warnings will be added.
The first positional argument and the positional `backend` argument will remain
compatible. Named calls using the old names are an intentional 0.7.0
compatibility break and must be listed in the release migration notes.

A2 updates these public methods directly, with keyword-only parameters and without alias compatibility. The implementation retains positional backend arguments and the existing Three.js/Matplotlib return types.

## Input model

- A configuration is a numerical vector of shape `(robot.dof,)` or a SymPy-keyed
  substitution mapping (including legacy full mappings).
- An animation consists of a nonempty sequence of configurations, optionally
  a numerical matrix with shape `(N, robot.dof)`. A1 normalizes fixed parameters
  once per animation and returns independent mapping objects per frame.
- `parameters` is an optional mapping from atomic SymPy `Symbol` keys
  to finite real scalar values; SymPy numeric constants such as `pi / 2` are
  acceptable values. These keys must belong to the kinematic DH table,
  not the dynamic mass, inertia or gravity model.
- A DH entry such as `L / 2` requires the symbol `L`, not the
  expression `L / 2` as a parameter key. Substituting a composite
  expression directly may leave the expanded symbolic transformation
  unchanged.
- Any overlap between a configuration mapping and `parameters` is an
  error, even when the values agree; there is no precedence rule.
- No caller-provided mapping, array, or `Robot` state is modified.
- Unexpected keys in **explicit** `parameters` raise `ValueError`. Legacy full
  configuration mappings may include unused symbolic keys for compatibility.

## Recognizing the symbolic DH parameters

`moro.abc.q1` is an applied dynamicsymbol `q_1(t)`. Its
`free_symbols` contains `t`, **not** the joint coordinate.
Before extracting DH parameter symbols, replace recognized joint atoms with
temporary `Dummy` instances. The remaining free symbols (other than
the reserved explicit time symbol) are constant DH parameter candidates.

An expression like `L * cos(q1(t))` therefore requires `L` but not
`t`. Unresolved functions of time are reported separately; the
visualizer does not silently interpret them as constant model parameters.

## Composite joint expressions

Moro currently stores the entire DH entry as `robot.qs[i]`. For
`theta_i = q_i + theta0`, this entry is not an independent atomic
coordinate. A1 **rejects numerical joint vectors** for such a model
instead of substituting a number for the entire expression.

A full symbolic configuration mapping remains possible. For composite
expressions containing recognizable dynamicsymbols, fixed nonjoint symbols
such as `theta0` may also be supplied in `parameters`.
When a composite joint entry consists only of ordinary Symbols,
the coordinate/offset split is ambiguous: the user must supply a complete
mapping without separate `parameters`. More ambitious joint-offset
modeling is out of scope for API-001.

## Internal helpers and boundaries

A1 introduces the following private functions in
`moro.visualization.substitutions`:

| Helper | Responsibility |
| --- | --- |
| `_get_visualization_parameters(robot)` | Identify constant DH symbols independently of dynamicsymbol time dependencies |
| `_normalize_visualization_parameters(robot, parameters)` | Check mapping keys, numerical scalar values and unknown parameters |
| `_normalize_visualization_substitutions(robot, configuration, parameters=None)` | Validate one configuration, combine without overlap and check DH entries |
| `_normalize_visualization_configurations(robot, configurations, parameters=None)` | Normalize animation frames and reuse the validated fixed parameters |

These helpers return ordinary substitution dictionaries suitable for the
existing `evaluate_robot(robot, substitutions)` contract. A2 connects
`RobotVisualizer.plot()` and `animate()` to this normalization layer;
the backends continue receiving evaluated `SceneData`, with unchanged
time handling. The old visualizer-local normalization helpers were removed.

## Acceptance checks

A1: R/RR and mixed R/P; symbolic and numerical DH parameters; numeric
vectors and full mappings; SymPy numeric constants; missing, duplicated,
unknown, complex, nonfinite and boolean values; composite DH expressions;
dynamic coordinates versus explicit time; independent animation mappings;
legacy mapping flexibility and nonmutation.

A2: updated plot/animate signatures and removed the legacy normalizers from
the public orchestrator. Added regression coverage of named/positional calls,
both rendering backends, physical-time playback, constant DH geometry, legacy
full mappings, input nonmutation and the deliberate keyword rename.

A3: refreshed the Visualization User Guide and Quick Start, updated
trapezoidal/physical-time examples, added an independent symbolic RR/RP
notebook example and its executable-code-block smoke test, and drafted
[unreleased 0.7.0 migration notes](../../source/release-notes/0.7.0.md).
Interactive Colab rendering of the new symbolic-parameter example remains
a manual acceptance step before final closure.
