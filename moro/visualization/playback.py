"""Private validation of explicit physical-time animation metadata.

T-06/C1 establishes the Python input contract. The Three.js browser-clock
playback engine is deliberately deferred to C2, so this module does not
replace frame timing or interpolate configurations.
"""

import math
from numbers import Real

import numpy as np


def _normalize_playback_speed(playback_speed):
    """Return a finite positive dimensionless playback speed."""
    if (
        isinstance(playback_speed, (bool, np.bool_))
        or not isinstance(playback_speed, Real)
    ):
        raise ValueError("playback_speed must be a finite positive real scalar.")
    try:
        speed = float(playback_speed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "playback_speed must be a finite positive real scalar."
        ) from exc
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("playback_speed must be a finite positive real scalar.")
    return speed


def _normalize_playback_times(times, frame_count):
    """Validate timestamps in seconds and return a fresh relative-time array.

    Values must be finite, real, strictly increasing 1D samples, one per
    scene. Absolute time origins may be negative. Subtraction must remain
    finite and distinguishable in float64; the original input is untouched.
    """
    try:
        raw = np.asarray(times)
    except Exception as exc:
        raise ValueError("times must be a 1D vector of real numbers.") from exc

    if raw.ndim != 1:
        raise ValueError("times must be a one-dimensional vector.")
    if raw.size < 2:
        raise ValueError("times must contain at least two samples.")
    if raw.size != frame_count:
        raise ValueError(
            f"times must contain one timestamp per configuration "
            f"({frame_count} expected, got {raw.size})."
        )
    # Inspect the original scalar types to avoid NumPy coercing bools,
    # mixed bool/numeric sequences, or numeric-looking strings to floats.
    try:
        original = np.asarray(times, dtype=object)
    except Exception as exc:
        raise ValueError("times must contain real numeric values.") from exc
    if not all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in original
    ):
        raise ValueError("times must contain real numeric values, not booleans.")

    try:
        timestamps = np.array(times, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must contain finite real numeric values.") from exc
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("times must contain only finite values.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("times must be strictly increasing.")

    with np.errstate(over="ignore", invalid="ignore"):
        relative = timestamps - timestamps[0]
    if (
        not np.all(np.isfinite(relative))
        or relative[0] != 0.0
        or relative[-1] <= 0.0
        or np.any(np.diff(relative) <= 0.0)
    ):
        raise ValueError(
            "times produce nonrepresentable relative timestamps; "
            "use a smaller absolute time origin or rescale time."
        )
    return relative
