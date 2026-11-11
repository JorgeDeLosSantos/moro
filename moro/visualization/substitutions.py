"""Private normalization of robot configurations and fixed DH parameters.

A1 prepares validated substitution dictionaries for the visualization layer.
RobotVisualizer and evaluate_robot retain their existing public contracts until
the A2 integration increment.
"""

from collections.abc import Mapping
import math

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from moro.abc import t
from moro.core import Robot


_DH_NAMES = ("a", "alpha", "d", "theta")


def _joint_atoms(robot: Robot) -> frozenset:
    """Find atomic coordinates appearing in the joint expressions.

    A composite DH joint expression may contain an applied dynamicsymbol,
    such as q1(t) + theta0. Its constituent dynamicsymbol is recognizable,
    although the composite expression itself cannot be assigned from a
    numerical joint vector without changing the meaning of Robot.qs.
    """
    joints = set()
    for coordinate in robot.qs:
        expression = sp.sympify(coordinate)
        if isinstance(expression, (sp.Symbol, AppliedUndef)):
            joints.add(expression)
        else:
            joints.update(expression.atoms(AppliedUndef))
    return frozenset(joints)


def _get_visualization_parameters(robot: Robot) -> frozenset:
    """Return constant symbols used by the original kinematic DH table.

    Replace recognized joint atoms with Dummy placeholders BEFORE obtaining
    free_symbols. In particular, q1(t).free_symbols is {t}, not {q1(t)}.
    Parameters used only for mass, inertia or gravity are intentionally absent.

    For a composite joint expression made exclusively of ordinary Symbols,
    the joint/offset split is ambiguous. Explicit parameters for such a model
    are rejected by _normalize_visualization_parameters; complete mapping
    substitutions are still supported.
    """
    if not isinstance(robot, Robot):
        raise TypeError("Expected a Robot instance.")

    joint_atoms = _joint_atoms(robot)
    replacements = {
        joint: sp.Dummy(f"_moro_joint_{i}")
        for i, joint in enumerate(sorted(joint_atoms, key=sp.default_sort_key))
    }
    dummies = set(replacements.values())
    parameters = set()

    for row in robot.dh_parameters:
        for value in row:
            expression = sp.sympify(value).xreplace(replacements)
            parameters.update(expression.free_symbols - dummies)

    # Time is never a constant model parameter. It only appears implicitly
    # inside recognized dynamicsymbols, or explicitly in unsupported geometry.
    parameters.discard(t)
    return frozenset(parameters)


def _finite_real(value, *, name: str) -> float:
    """Accept real numeric scalars, including SymPy constants such as pi/2."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a finite real scalar, not {value!r}.")
    try:
        result = float(sp.sympify(value))
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


def _normalize_visualization_parameters(robot: Robot, parameters=None) -> dict:
    """Validate optional fixed DH substitutions, excluding joint coordinates."""
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping of SymPy Symbols to numbers.")

    # There is no safe way to infer which Symbol means the coordinate in
    # theta=q+theta0 when both q and theta0 are ordinary Symbols.
    # Such models remain usable with a complete substitution mapping.
    if parameters:
        for coordinate in robot.qs:
            expression = sp.sympify(coordinate)
            if not isinstance(expression, (sp.Symbol, AppliedUndef)):
                if not expression.atoms(AppliedUndef):
                    raise ValueError(
                        "Cannot separate parameters from a composite joint "
                        f"expression {expression}. Use a complete configuration "
                        "mapping without parameters."
                    )

    allowed = _get_visualization_parameters(robot)
    normalized = {}
    for key, value in parameters.items():
        if not isinstance(key, sp.Symbol):
            raise ValueError(
                "parameters keys must be atomic SymPy Symbols, not "
                f"{key!r}."
            )
        if key not in allowed:
            raise ValueError(f"Unknown model parameter: {key}")
        normalized[key] = _finite_real(value, name=f"Parameter {key}")
    return normalized


def _normalize_joint_configuration(robot: Robot, configuration) -> dict:
    """Accept a full symbolic mapping or a numerical vector of robot.dof."""
    if isinstance(configuration, Mapping):
        result = {}
        for key, value in configuration.items():
            if not isinstance(key, sp.Basic):
                raise ValueError(
                    "Configuration mappings must use SymPy expressions as keys."
                )
            result[key] = _finite_real(value, name=f"Configuration value for {key}")
        return result

    qs = [sp.sympify(coordinate) for coordinate in robot.qs]
    if (
        len(set(qs)) != robot.dof
        or not all(isinstance(q, (sp.Symbol, AppliedUndef)) for q in qs)
    ):
        raise ValueError(
            "Numerical joint vectors require distinct atomic coordinates "
            "in robot.qs. For composite joint expressions, supply a "
            "complete substitution mapping instead."
        )

    try:
        raw = np.asarray(configuration)
    except Exception as exc:
        raise ValueError(
            "A numerical configuration must be a finite vector of robot.dof values."
        ) from exc
    if np.iscomplexobj(raw) or raw.dtype.kind in ("b", "U", "S"):
        raise ValueError("A numerical configuration must contain real numbers.")
    if any(
        isinstance(value, (bool, np.bool_, str, bytes))
        for value in np.asarray(configuration, dtype=object).flat
    ):
        raise ValueError("A numerical configuration must contain real numbers.")
    try:
        values = np.asarray(configuration, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "A numerical configuration must contain finite real numbers."
        ) from exc
    if values.ndim != 1 or values.size != robot.dof:
        raise ValueError(
            f"A numerical configuration must have shape ({robot.dof},)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("A numerical configuration must be finite.")
    return dict(zip(qs, values.tolist()))


def _validate_dh_substitutions(robot: Robot, substitutions: dict) -> None:
    """Check DH rows directly, including legacy expression-key substitutions.

    Constant parameters are required as atomic Symbol keys, ensuring that
    substitution also resolves the expanded homogeneous transformations.
    The exact composite joint expression may still be substituted by a
    legacy mapping key.
    """
    joint_atoms = _joint_atoms(robot)
    replacements = {
        joint: sp.Dummy(f"_moro_joint_{i}")
        for i, joint in enumerate(sorted(joint_atoms, key=sp.default_sort_key))
    }
    dummies = set(replacements.values())
    joint_fields = ("theta", "d")

    for row_index, row in enumerate(robot.dh_parameters, start=1):
        joint_expression = sp.sympify(robot.qs[row_index - 1])
        joint_field = joint_fields[robot.joint_types[row_index - 1] == "p"]
        for name, source in zip(_DH_NAMES, row):
            expression = sp.sympify(source)

            # The original scene evaluator substitutes against symbolic
            # transformations, not the literal DH-table expressions.
            # Substituting a fragment (e.g. L/2) without assigning L can
            # appear valid in the table but leave symbolic entries in T_i0.
            # Require atomic DH parameters, except when an entire composite
            # joint expression is supplied as a legacy mapping key.
            entire_joint_substituted = (
                name == joint_field
                and expression == joint_expression
                and expression in substitutions
            )
            if not entire_joint_substituted:
                required = (
                    expression.xreplace(replacements).free_symbols
                    - dummies - {t}
                )
                missing = required - set(substitutions)
                if missing:
                    labels = ", ".join(map(str, sorted(
                        missing, key=sp.default_sort_key
                    )))
                    raise ValueError(f"Missing model parameters: {labels}")

            try:
                evaluated = expression.subs(substitutions)
            except (TypeError, ValueError, AttributeError, sp.SympifyError) as exc:
                raise ValueError(
                    f"Could not evaluate DH row {row_index} parameter '{name}'."
                ) from exc

            missing_joints = sorted(
                (joint for joint in joint_atoms if evaluated.has(joint)),
                key=sp.default_sort_key,
            )
            if missing_joints:
                labels = ", ".join(map(str, missing_joints))
                raise ValueError(f"Missing joint coordinates: {labels}")

            # Any unrecognized time-dependent function remains an unresolved
            # input. Do not mistake its free time symbol for a DH constant.
            unknown_functions = sorted(
                evaluated.atoms(AppliedUndef) - joint_atoms,
                key=sp.default_sort_key,
            )
            if unknown_functions:
                labels = ", ".join(map(str, unknown_functions))
                raise ValueError(
                    f"Unresolved time-dependent DH expressions: {labels}"
                )

            missing_symbols = sorted(
                evaluated.free_symbols - {t} -
                {q for q in joint_atoms if isinstance(q, sp.Symbol)},
                key=sp.default_sort_key,
            )
            if missing_symbols:
                labels = ", ".join(map(str, missing_symbols))
                raise ValueError(f"Missing model parameters: {labels}")

            if evaluated.has(t):
                raise ValueError(
                    f"DH row {row_index} parameter '{name}' depends on "
                    "explicit time; a fixed numerical configuration is required."
                )

            try:
                result = float(evaluated)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"DH row {row_index} parameter '{name}' must evaluate "
                    "to a finite real number."
                ) from exc
            if not math.isfinite(result):
                raise ValueError(
                    f"DH row {row_index} parameter '{name}' must evaluate "
                    "to a finite real number."
                )


def _combine_visualization_substitutions(
    robot: Robot, configuration, fixed_parameters: dict
) -> dict:
    joint_values = _normalize_joint_configuration(robot, configuration)
    overlap = set(joint_values) & set(fixed_parameters)
    if overlap:
        labels = ", ".join(map(str, sorted(overlap, key=sp.default_sort_key)))
        raise ValueError(
            f"Duplicate substitutions in configuration and parameters: {labels}"
        )
    result = {**fixed_parameters, **joint_values}
    _validate_dh_substitutions(robot, result)
    return result


def _normalize_visualization_substitutions(
    robot: Robot, configuration, parameters=None
) -> dict:
    """Return evaluated, disjoint DH substitutions for one configuration."""
    fixed = _normalize_visualization_parameters(robot, parameters)
    return _combine_visualization_substitutions(robot, configuration, fixed)


def _normalize_visualization_configurations(
    robot: Robot, configurations, parameters=None
) -> list[dict]:
    """Normalize an animation, validating fixed parameters only once."""
    fixed = _normalize_visualization_parameters(robot, parameters)
    if isinstance(configurations, Mapping):
        raise ValueError("configurations must be a sequence, not a mapping.")
    try:
        frames = list(configurations)
    except TypeError as exc:
        raise ValueError("configurations must be an iterable of configurations.") from exc
    if not frames:
        raise ValueError("configurations must contain at least one frame.")
    normalized = []
    for index, configuration in enumerate(frames):
        try:
            normalized.append(
                _combine_visualization_substitutions(robot, configuration, fixed)
            )
        except ValueError as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc
    return normalized
