"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import sympy as sp
from sympy import sin, cos, atan2, sqrt, pi
from sympy.matrices import Matrix, MatrixBase
from moro.util import deg2rad, rad2deg

__all__ = [
    "axa2quat",
    "axa2rot",
    "dh",
    "eul2rot",
    "htmrot",
    "htmtra",
    "htm2rot",
    "htm2tra",
    "invhtm",
    "is_homogeneous_transform",
    "is_rotation_matrix",
    "quat2axa",
    "quat2rot",
    "rot2eul",
    "rot2quat",
    "rot2rotvec",
    "rot2axa",
    "rot2htm",
    "rot",
    "rotx",
    "roty",
    "rotz",
    "rotvec2rot",
    "rt2htm",
    "skew",
    "vex"
]
    
# ~ ==========================================
# ~ Transformation operations
# ~ ==========================================
def _normalize_axis(axis):
    if not isinstance(axis, str):
        raise ValueError("axis must be 'x', 'y' or 'z'.")

    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y' or 'z'.")

    return axis


def _as_vector(v, size, name="vector"):
    """Convert a supported vector input to a SymPy column matrix."""
    try:
        vector = Matrix(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a {size}D vector given as a list, tuple, "
            f"column matrix ({size}, 1) or row matrix (1, {size})."
        ) from exc

    if vector.shape == (size, 1):
        return vector
    if vector.shape == (1, size):
        return vector.T

    raise ValueError(
        f"{name} must be a {size}D vector with shape ({size}, 1) or "
        f"(1, {size}); got shape {vector.shape}."
    )


def _as_3d_vector(v, name="vector"):
    return _as_vector(v, 3, name=name)


def rot(theta, axis="z", deg=False):
    """
    Return a rotation matrix that represents a rotation of ``theta`` about ``axis``.

    Parameters
    ----------
    theta : float, int or symbolic
        Rotation angle. By default, the value is interpreted in radians.
    axis : str
        Rotation axis, ``"x"``, ``"y"`` or ``"z"``. Matching is
        case-insensitive. Default is ``"z"``.
    deg : bool, optional
        If True, ``theta`` is interpreted as degrees. Default is False.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix of shape (3, 3).
    """
    axis = _normalize_axis(axis)
    if axis=="x":
        return rotx(theta, deg=deg)
    elif axis=="y":
        return roty(theta, deg=deg)
    return rotz(theta, deg=deg)

def rotz(theta, deg=False):
    """
    Calculate the rotation matrix about the z-axis.

    Parameters
    ----------
    theta : float, int or symbolic
        Rotation angle. By default, the value is assumed to be given in radians.
    deg : bool, optional
        If True, `theta` is interpreted as degrees. Default is False.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix in SO(3).

    Examples
    --------
    Using angle in radians:

    >>> rotz(pi/2)
    ⎡0  -1  0⎤
    ⎢        ⎥
    ⎢1   0  0⎥
    ⎢        ⎥
    ⎣0   0  1⎦

    Using symbolic variables:

    >>> x = symbols("x")
    >>> rotz(x)
    ⎡cos(x)  -sin(x)  0⎤
    ⎢                  ⎥
    ⎢sin(x)   cos(x)  0⎥
    ⎢                  ⎥
    ⎣  0        0     1⎦

    Using angles in degrees:

    >>> rotz(45, deg=True)
    ⎡0.707106781186548  -0.707106781186547  0⎤
    ⎢                                        ⎥
    ⎢0.707106781186547   0.707106781186548  0⎥
    ⎢                                        ⎥
    ⎣        0                  0           1⎦
    """
    if deg: # If theta is given in degrees -> convert to radians
        theta = deg2rad(theta, False)
    ct = cos(theta)
    st = sin(theta)
    R = Matrix([[ct, -st, 0],
                  [st, ct, 0],
                  [0, 0, 1]])
    return R


def roty(theta, deg=False):
    """
    Calculates the rotation matrix about the y-axis

    Parameters
    ----------
    theta : float, int or `symbolic`
        Rotation angle (given in radians by default)

    deg : bool 
        If True, `theta` is interpreted as degrees. Default is False.   

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix in SO(3).
        
    Examples
    --------
    
    >>> roty(pi/3)
    ⎡         √3 ⎤
    ⎢1/2   0  ── ⎥
    ⎢         2  ⎥
    ⎢            ⎥
    ⎢ 0    1   0 ⎥
    ⎢            ⎥
    ⎢-√3         ⎥
    ⎢────  0  1/2⎥
    ⎣ 2          ⎦
    
    >>> roty(30, deg=True)
    ⎡0.866025403784439  0         0.5       ⎤
    ⎢                                       ⎥
    ⎢        0          1          0        ⎥
    ⎢                                       ⎥
    ⎣      -0.5         0  0.866025403784439⎦

    """
    if deg: # If theta is given in degrees -> convert to radians
        theta = deg2rad(theta, False)
    ct = cos(theta)
    st = sin(theta)
    R = Matrix([[ct, 0, st],
                [0, 1, 0],
                [-st, 0, ct]])
    return R


def rotx(theta, deg=False):
    """
    Calculates the rotation matrix about the x-axis

    Parameters
    ----------
    theta : float, int or `symbolic`
        Rotation angle (given in radians by default)

    deg : bool
        If True, `theta` is interpreted as degrees. Default is False.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix in SO(3).
        
    Examples
    --------
    >>> rotx(pi)
    ⎡1  0   0 ⎤
    ⎢         ⎥
    ⎢0  -1  0 ⎥
    ⎢         ⎥
    ⎣0  0   -1⎦
    >>> rotx(60, deg=True)
    ⎡1          0                  0         ⎤
    ⎢                                        ⎥
    ⎢0         0.5         -0.866025403784439⎥
    ⎢                                        ⎥
    ⎣0  0.866025403784439         0.5        ⎦

    """
    if deg: # If theta is given in degrees -> convert to radians
        theta = deg2rad(theta, False)

    ct = cos(theta)
    st = sin(theta)
    R = Matrix([[1, 0, 0],
                [0, ct, -st],
                [0, st, ct]])
    return R


def dh(a,alpha,d,theta):
    """
    Compute the Denavit-Hartenberg homogeneous transformation matrix.

    Parameters
    ----------
    a : int, float or symbolic
        Link length (DH parameter).
    alpha : int, float or symbolic
        Link twist (DH parameter).
    d : int, float or symbolic
        Link offset (DH parameter).
    theta : int, float or symbolic
        Joint angle (DH parameter).

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Denavit-Hartenberg homogeneous transformation matrix of shape (4, 4).

    Examples
    --------
    With numerical values:

    >>> dh(100, pi/2, 50, pi/2)
    ⎡0  0  1   0 ⎤
    ⎢            ⎥
    ⎢1  0  0  100⎥
    ⎢            ⎥
    ⎢0  1  0  50 ⎥
    ⎢            ⎥
    ⎣0  0  0   1 ⎦

    Using symbolic values:

    >>> a = symbols("a")
    >>> t = symbols("t")
    >>> dh(a, 0, 0, t)
    ⎡cos(t)  -sin(t)  0  a⋅cos(t)⎤
    ⎢                            ⎥
    ⎢sin(t)   cos(t)  0  a⋅sin(t)⎥
    ⎢                            ⎥
    ⎢  0        0     1     0    ⎥
    ⎢                            ⎥
    ⎣  0        0     0     1    ⎦
    """
    H = Matrix([[cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
                  [sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta)],
                  [0,sin(alpha),cos(alpha),d],
                  [0,0,0,1]])
    return H

    

_PROPER_EULER_SEQUENCES = (
    "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
)
_TAIT_BRYAN_SEQUENCES = (
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
)
_EULER_SEQUENCES = _PROPER_EULER_SEQUENCES + _TAIT_BRYAN_SEQUENCES


def _normalize_euler_sequence(seq):
    valid_sequences = ", ".join(repr(value) for value in _EULER_SEQUENCES)
    if not isinstance(seq, str):
        raise TypeError(f"seq must be a string; expected one of: {valid_sequences}.")

    seq = seq.lower()
    if seq not in _EULER_SEQUENCES:
        raise ValueError(f"seq must be one of: {valid_sequences}.")

    return seq


# Configuration for intrinsic proper Euler sequences under:
# R = R_a(phi) @ R_b(theta) @ R_a(psi).
# atan2 pairs are encoded as
# ((sin_sign, sin_i, sin_j), (cos_sign, cos_i, cos_j)).
_PROPER_EULER_CONFIG = {
    "xyx": {
        "cos_index": (0, 0),
        "phi": ((1, 1, 0), (-1, 2, 0)),
        "psi": ((1, 0, 1), (1, 0, 2)),
        "singular_positive": ((1, 2, 1), (1, 1, 1)),
        "singular_negative": ((1, 2, 1), (1, 1, 1)),
    },
    "xzx": {
        "cos_index": (0, 0),
        "phi": ((1, 2, 0), (1, 1, 0)),
        "psi": ((1, 0, 2), (-1, 0, 1)),
        "singular_positive": ((1, 2, 1), (1, 1, 1)),
        "singular_negative": ((-1, 2, 1), (-1, 1, 1)),
    },
    "yxy": {
        "cos_index": (1, 1),
        "phi": ((1, 0, 1), (1, 2, 1)),
        "psi": ((1, 1, 0), (-1, 1, 2)),
        "singular_positive": ((1, 0, 2), (1, 0, 0)),
        "singular_negative": ((-1, 0, 2), (1, 0, 0)),
    },
    "yzy": {
        "cos_index": (1, 1),
        "phi": ((1, 2, 1), (-1, 0, 1)),
        "psi": ((1, 1, 2), (1, 1, 0)),
        "singular_positive": ((1, 0, 2), (1, 0, 0)),
        "singular_negative": ((1, 0, 2), (-1, 0, 0)),
    },
    "zxz": {
        "cos_index": (2, 2),
        "phi": ((1, 0, 2), (-1, 1, 2)),
        "psi": ((1, 2, 0), (1, 2, 1)),
        "singular_positive": ((1, 1, 0), (1, 0, 0)),
        "singular_negative": ((1, 1, 0), (1, 0, 0)),
    },
    "zyz": {
        "cos_index": (2, 2),
        "phi": ((1, 1, 2), (1, 0, 2)),
        "psi": ((1, 2, 1), (-1, 2, 0)),
        "singular_positive": ((1, 1, 0), (1, 0, 0)),
        "singular_negative": ((-1, 1, 0), (-1, 0, 0)),
    },
}


# Configuration for intrinsic Tait-Bryan sequences under:
# R = R_a(phi) @ R_b(theta) @ R_c(psi), a != b != c.
_TAIT_BRYAN_CONFIG = {
    "xyz": {
        "sin_term": (1, 0, 2),
        "phi": ((-1, 1, 2), (1, 2, 2)),
        "psi": ((-1, 0, 1), (1, 0, 0)),
        "singular_positive": ((1, 1, 0), (1, 1, 1)),
        "singular_negative": ((-1, 1, 0), (1, 1, 1)),
        "singular_positive_sign": +1,
        "singular_negative_sign": -1,
    },
    "xzy": {
        "sin_term": (-1, 0, 1),
        "phi": ((1, 2, 1), (1, 1, 1)),
        "psi": ((1, 0, 2), (1, 0, 0)),
        "singular_positive": ((1, 2, 0), (1, 1, 0)),
        "singular_negative": ((-1, 2, 0), (-1, 1, 0)),
        "singular_positive_sign": -1,
        "singular_negative_sign": +1,
    },
    "yxz": {
        "sin_term": (-1, 1, 2),
        "phi": ((1, 0, 2), (1, 2, 2)),
        "psi": ((1, 1, 0), (1, 1, 1)),
        "singular_positive": ((1, 0, 1), (1, 0, 0)),
        "singular_negative": ((-1, 0, 1), (1, 0, 0)),
        "singular_positive_sign": -1,
        "singular_negative_sign": +1,
    },
    "yzx": {
        "sin_term": (1, 1, 0),
        "phi": ((-1, 2, 0), (1, 0, 0)),
        "psi": ((-1, 1, 2), (1, 1, 1)),
        "singular_positive": ((1, 0, 2), (1, 2, 2)),
        "singular_negative": ((1, 0, 2), (1, 2, 2)),
        "singular_positive_sign": +1,
        "singular_negative_sign": -1,
    },
    "zxy": {
        "sin_term": (1, 2, 1),
        "phi": ((-1, 0, 1), (1, 1, 1)),
        "psi": ((-1, 2, 0), (1, 2, 2)),
        "singular_positive": ((1, 1, 0), (1, 0, 0)),
        "singular_negative": ((1, 1, 0), (1, 0, 0)),
        "singular_positive_sign": +1,
        "singular_negative_sign": -1,
    },
    "zyx": {
        "sin_term": (-1, 2, 0),
        "phi": ((1, 1, 0), (1, 0, 0)),
        "psi": ((1, 2, 1), (1, 2, 2)),
        "singular_positive": ((1, 1, 2), (1, 1, 1)),
        "singular_negative": ((-1, 1, 2), (1, 1, 1)),
        "singular_positive_sign": -1,
        "singular_negative_sign": +1,
    },
}


def _validate_tol(tol):
    if isinstance(tol, bool) or not isinstance(tol, (int, float, sp.Number)):
        raise TypeError("tol must be a positive real number.")

    tol = sp.sympify(tol)
    if tol.is_number is not True:
        raise TypeError("tol must be a positive real number.")
    if tol.is_real is not True:
        raise ValueError("tol must be a positive real number.")

    tol_value = float(tol)
    if tol_value <= 0:
        raise ValueError("tol must be greater than 0.")
    return tol_value


def _validate_euler_tol(tol):
    return _validate_tol(tol)


def _is_numeric_real(value):
    value = sp.simplify(value)
    numeric_value = sp.N(value)
    return not value.free_symbols and numeric_value.is_real is True


def _has_float(value):
    return bool(sp.sympify(value).atoms(sp.Float))


def _matrix_zero_status(M):
    statuses = []
    for value in Matrix(M):
        zero = sp.trigsimp(sp.simplify(value)).is_zero
        statuses.append(zero)
    if all(status is True for status in statuses):
        return True
    if any(status is False for status in statuses):
        return False
    return None


def is_rotation_matrix(R, *, tol=1e-9):
    """Return True, False, or None according to membership in SO(3)."""
    tol = _validate_tol(tol)
    try:
        R = Matrix(R)
    except (TypeError, ValueError):
        return False

    if R.shape != (3, 3):
        return False

    if all(_is_numeric_real(value) for value in R):
        orthogonality_error = R.T * R - sp.eye(3)
        if any(abs(float(sp.N(value))) > float(tol) for value in orthogonality_error):
            return False
        determinant_error = sp.det(R) - 1
        return abs(float(sp.N(determinant_error))) <= float(tol)

    orthogonality = _matrix_zero_status(R.T * R - sp.eye(3))
    determinant = sp.trigsimp(sp.simplify(sp.det(R) - 1)).is_zero

    if orthogonality is True and determinant is True:
        return True
    if orthogonality is False or determinant is False:
        return False
    return None


def _validate_rotation_matrix(R, *, tol=1e-9):
    try:
        R = Matrix(R)
    except (TypeError, ValueError) as exc:
        raise TypeError("R must be convertible to a 3x3 matrix.") from exc

    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix.")

    status = is_rotation_matrix(R, tol=tol)
    if status is not True:
        if status is None:
            raise ValueError("R must be a rotation matrix; symbolic SO(3) membership is indeterminate.")
        raise ValueError("R must be a valid rotation matrix in SO(3).")
    return R



def _symbolic_zero_condition(value):
    """Classify whether a symbolic expression is provably zero."""
    return sp.simplify(value).is_zero


def is_homogeneous_transform(T, *, tol=1e-9):
    """Return True, False, or None according to membership in SE(3)."""
    tol = _validate_tol(tol)

    try:
        T = Matrix(T)
    except (TypeError, ValueError) as exc:
        raise TypeError("T must be convertible to a 4x4 matrix.") from exc

    if T.shape != (4, 4):
        return False

    entries = [sp.sympify(value) for value in T]
    fully_numeric = all(value.is_number is True for value in entries)

    if fully_numeric:
        if any(value.is_real is not True for value in entries):
            return False

        rotation_status = is_rotation_matrix(T[:3, :3], tol=tol)
        if rotation_status is not True:
            return False

        target_row = (0.0, 0.0, 0.0, 1.0)
        return all(
            abs(float(sp.N(T[3, j])) - target_row[j]) <= tol
            for j in range(4)
        )

    rotation_status = is_rotation_matrix(T[:3, :3], tol=tol)
    row_statuses = [
        _symbolic_zero_condition(T[3, 0]),
        _symbolic_zero_condition(T[3, 1]),
        _symbolic_zero_condition(T[3, 2]),
        _symbolic_zero_condition(T[3, 3] - 1),
    ]

    statuses = [rotation_status, *row_statuses]
    if any(status is False for status in statuses):
        return False
    if all(status is True for status in statuses):
        return True
    return None


def _validate_homogeneous_transform(T, *, tol=1e-9):
    """Validate and normalize a rigid homogeneous transformation."""
    tol = _validate_tol(tol)

    try:
        T = Matrix(T)
    except (TypeError, ValueError) as exc:
        raise TypeError("T must be convertible to a 4x4 matrix.") from exc

    if T.shape != (4, 4):
        raise ValueError("T must be a 4x4 matrix.")

    status = is_homogeneous_transform(T, tol=tol)
    if status is True:
        return T
    if status is None:
        raise ValueError(
            "T must be a homogeneous transformation; symbolic SE(3) "
            "membership is indeterminate."
        )
    raise ValueError("T must be a valid homogeneous transformation in SE(3).")


def _classify_trig_value(value, tol):
    value_simplified = sp.simplify(value)

    if _has_float(value_simplified) and _is_numeric_real(value_simplified):
        numeric_value = float(sp.N(value_simplified))
        if numeric_value > 1.0 + tol or numeric_value < -1.0 - tol:
            raise ValueError(
                "Trigonometric value is outside the valid range [-1, 1] beyond tolerance."
            )
        numeric_value = max(-1.0, min(1.0, numeric_value))

        if abs(numeric_value - 1.0) <= tol:
            return "positive", sp.S(1)
        if abs(numeric_value + 1.0) <= tol:
            return "negative", sp.S(-1)
        return "general", sp.Float(numeric_value)

    positive = sp.simplify(value_simplified - 1).is_zero
    negative = sp.simplify(value_simplified + 1).is_zero

    if positive is True:
        return "positive", sp.S(1)
    if negative is True:
        return "negative", sp.S(-1)
    if positive is False and negative is False:
        return "general", value_simplified

    return "symbolic", value_simplified


def _sqrt_one_minus_square(value):
    if _has_float(value) and _is_numeric_real(value):
        numeric = float(sp.N(value))
        return sqrt(sp.Float(max(0.0, 1.0 - numeric**2)))
    return sqrt(sp.simplify(1 - value**2))


def _signed_matrix_element(R, term):
    sign, i, j = term
    return sign * R[i, j]


def _atan2_from_config(R, pair):
    sin_term, cos_term = pair
    return atan2(
        _signed_matrix_element(R, sin_term),
        _signed_matrix_element(R, cos_term),
    )


def _negated_pair(pair):
    sin_term, cos_term = pair
    return (
        (-sin_term[0], sin_term[1], sin_term[2]),
        (-cos_term[0], cos_term[1], cos_term[2]),
    )


def _convert_euler_solutions_to_degrees(solutions):
    return [
        (rad2deg(phi), rad2deg(theta), rad2deg(psi))
        for phi, theta, psi in solutions
    ]


def _rot2proper_euler(R, seq, tol):
    config = _PROPER_EULER_CONFIG[seq]
    i, j = config["cos_index"]
    case, cos_theta = _classify_trig_value(R[i, j], tol)

    if case in ("general", "symbolic"):
        sin_theta = _sqrt_one_minus_square(cos_theta)
        theta1 = atan2(sin_theta, cos_theta)
        theta2 = atan2(-sin_theta, cos_theta)

        phi1 = _atan2_from_config(R, config["phi"])
        psi1 = _atan2_from_config(R, config["psi"])
        phi2 = _atan2_from_config(R, _negated_pair(config["phi"]))
        psi2 = _atan2_from_config(R, _negated_pair(config["psi"]))
        return [(phi1, theta1, psi1), (phi2, theta2, psi2)], None

    if case == "positive":
        phi = _atan2_from_config(R, config["singular_positive"])
        return [(phi, sp.S(0), sp.S(0))], "positive"

    phi = _atan2_from_config(R, config["singular_negative"])
    return [(phi, pi, sp.S(0))], "negative"


def _rot2tait_bryan(R, seq, tol):
    config = _TAIT_BRYAN_CONFIG[seq]
    sin_theta_raw = _signed_matrix_element(R, config["sin_term"])
    case, sin_theta = _classify_trig_value(sin_theta_raw, tol)

    if case in ("general", "symbolic"):
        cos_theta = _sqrt_one_minus_square(sin_theta)
        theta1 = atan2(sin_theta, cos_theta)
        theta2 = atan2(sin_theta, -cos_theta)

        phi1 = _atan2_from_config(R, config["phi"])
        psi1 = _atan2_from_config(R, config["psi"])
        phi2 = _atan2_from_config(R, _negated_pair(config["phi"]))
        psi2 = _atan2_from_config(R, _negated_pair(config["psi"]))
        return [(phi1, theta1, psi1), (phi2, theta2, psi2)], None

    if case == "positive":
        phi = _atan2_from_config(R, config["singular_positive"])
        return [(phi, pi / 2, sp.S(0))], "positive"

    phi = _atan2_from_config(R, config["singular_negative"])
    return [(phi, -pi / 2, sp.S(0))], "negative"


def _rot2eul_intrinsic(R, seq, tol):
    if seq in _PROPER_EULER_SEQUENCES:
        return _rot2proper_euler(R, seq, tol)
    return _rot2tait_bryan(R, seq, tol)


def _get_singular_relation_sign(seq, singular_case):
    if seq in _PROPER_EULER_SEQUENCES:
        return +1 if singular_case == "positive" else -1

    config = _TAIT_BRYAN_CONFIG[seq]
    return config[f"singular_{singular_case}_sign"]


def rot2eul(R, seq="zxz", deg=False, intrinsic=True, tol=1e-9):
    """Return Euler/Tait-Bryan angles that reconstruct a rotation matrix."""
    _validate_euler_tol(tol)
    seq = _normalize_euler_sequence(seq)
    if not isinstance(intrinsic, bool):
        raise TypeError("intrinsic must be a bool.")

    R = _validate_rotation_matrix(R, tol=tol)

    if intrinsic:
        solutions, singular_case = _rot2eul_intrinsic(R, seq, tol)
    else:
        internal_seq = seq[::-1]
        internal_solutions, singular_case = _rot2eul_intrinsic(
            R, internal_seq, tol
        )

        if singular_case is None:
            solutions = [
                (psi, theta, phi)
                for phi, theta, psi in internal_solutions
            ]
        else:
            alpha_eq, theta, _ = internal_solutions[0]
            sign = _get_singular_relation_sign(internal_seq, singular_case)
            solutions = [(sign * alpha_eq, theta, sp.S(0))]

    if deg:
        return _convert_euler_solutions_to_degrees(solutions)
    return solutions


def eul2rot(
    phi,
    theta,
    psi,
    seq="zxz",
    deg=False,
    intrinsic=True,
):
    """Build a rotation matrix from Euler or Tait-Bryan angles."""
    seq = _normalize_euler_sequence(seq)
    if not isinstance(intrinsic, bool):
        raise TypeError("intrinsic must be a bool.")

    if deg:
        phi, theta, psi = deg2rad(
            Matrix([phi, theta, psi]),
            evalf=False,
        )

    if not intrinsic:
        seq = seq[::-1]
        phi, psi = psi, phi

    return (
        rot(phi, seq[0])
        * rot(theta, seq[1])
        * rot(psi, seq[2])
    )


def htmtra(x=0, y=0, z=0):
    """
    Calculate the homogeneous transformation matrix of a translation.
    
    Parameters
    ----------
    x : int, float or symbolic, optional
        Translation along the x-axis. Default is 0.
    y : int, float or symbolic, optional
        Translation along the y-axis. Default is 0.
    z : int, float or symbolic, optional
        Translation along the z-axis. Default is 0.
    
    Returns
    -------
    H : :class:`sympy.matrices.dense.MutableDenseMatrix`
        Homogeneous transformation matrix

    Examples
    --------
    >>> htmtra()
    ⎡1  0  0  0⎤
    ⎢          ⎥
    ⎢0  1  0  0⎥
    ⎢          ⎥
    ⎢0  0  1  0⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦

    >>> htmtra(10,-40,50)
    ⎡1  0  0  10 ⎤
    ⎢            ⎥
    ⎢0  1  0  -40⎥
    ⎢            ⎥
    ⎢0  0  1  50 ⎥
    ⎢            ⎥
    ⎣0  0  0   1 ⎦

    >>> htmtra(z=100)
    ⎡1  0  0   0 ⎤
    ⎢            ⎥
    ⎢0  1  0   0 ⎥
    ⎢            ⎥
    ⎢0  0  1  100⎥
    ⎢            ⎥
    ⎣0  0  0   1 ⎦

    >>> a,b,c = symbols("a,b,c")
    >>> htmtra(x=a, y=b, z=c)
    ⎡1  0  0  a⎤
    ⎢          ⎥
    ⎢0  1  0  b⎥
    ⎢          ⎥
    ⎢0  0  1  c⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦

    """
    if isinstance(x, (list, tuple, MatrixBase)) or isinstance(y, (list, tuple, MatrixBase)) or isinstance(z, (list, tuple, MatrixBase)):
        raise TypeError("x, y and z must be scalar values.")

    M = Matrix([[1,0,0,x],
                [0,1,0,y],
                [0,0,1,z],
                [0,0,0,1]])
    return M
    

def htmrot(theta, axis="z", deg=False):
    """
    Return a homogeneous transformation matrix for a pure rotation.
    
    Parameters
    ----------
    theta : float, int or symbolic
        Rotation angle. By default, the value is interpreted in radians.
        
    axis : str
        Rotation axis, ``"x"``, ``"y"`` or ``"z"``. Matching is
        case-insensitive. Default is ``"z"``.

    deg : bool, optional
        If True, ``theta`` is interpreted as degrees. Default is False.
        
    Returns
    -------
    H : :class:`sympy.matrices.dense.MutableDenseMatrix`
        Homogeneous transformation matrix of shape (4, 4).
        
    
    Examples
    --------
    >>> htmrot(pi/2)
    ⎡0  -1  0  0⎤
    ⎢           ⎥
    ⎢1  0   0  0⎥
    ⎢           ⎥
    ⎢0  0   1  0⎥
    ⎢           ⎥
    ⎣0  0   0  1⎦
    >>> htmrot(pi/2, "x")
    ⎡1  0  0   0⎤
    ⎢           ⎥
    ⎢0  0  -1  0⎥
    ⎢           ⎥
    ⎢0  1  0   0⎥
    ⎢           ⎥
    ⎣0  0  0   1⎦
    >>> htmrot(30, "y", True)
    ⎡0.866025403784439  0         0.5         0⎤
    ⎢                                          ⎥
    ⎢        0          1          0          0⎥
    ⎢                                          ⎥
    ⎢      -0.5         0  0.866025403784439  0⎥
    ⎢                                          ⎥
    ⎣        0          0          0          1⎦
    >>> t = symbols("t")
    >>> htmrot(t, "x")
    ⎡1    0        0     0⎤
    ⎢                     ⎥
    ⎢0  cos(t)  -sin(t)  0⎥
    ⎢                     ⎥
    ⎢0  sin(t)  cos(t)   0⎥
    ⎢                     ⎥
    ⎣0    0        0     1⎦
    
    """
    return rot2htm(rot(theta, axis=axis, deg=deg))


def rot2htm(R):
    """
    Build a homogeneous transformation matrix from a rotation matrix.

    Parameters
    ----------
    R : array-like or sympy Matrix
        Rotation block. It is converted with ``Matrix(R)`` and must have shape
        (3, 3). No full SO(3) membership validation is performed.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Homogeneous transformation matrix with zero translation and shape (4, 4).
    """
    R = Matrix(R)
    if R.shape != (3, 3):
        raise ValueError(f"R must be a 3x3 matrix; got shape {R.shape}.")
    return R.row_join(Matrix([0, 0, 0])).col_join(Matrix([[0, 0, 0, 1]]))


def rt2htm(R, p):
    """
    Build a homogeneous transformation matrix from rotation and translation.

    Parameters
    ----------
    R : array-like or sympy Matrix
        Rotation block. It is converted with ``Matrix(R)`` and must have shape
        (3, 3). No full SO(3) membership validation is performed.
    p : list, tuple or sympy Matrix
        Translation vector. Accepted formats are a 3-element list, a 3-element
        tuple, a column matrix of shape (3, 1), or a row matrix of shape (1, 3).
        The vector is normalized internally to a column matrix.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Homogeneous transformation matrix of shape (4, 4).
    """
    R = Matrix(R)
    if R.shape != (3, 3):
        raise ValueError(f"R must be a 3x3 matrix; got shape {R.shape}.")
    p = _as_3d_vector(p, name="p")
    return R.row_join(p).col_join(Matrix([[0, 0, 0, 1]]))


def htm2rot(T):
    """
    Extract the rotation block from a homogeneous transformation matrix.

    Parameters
    ----------
    T : array-like or sympy Matrix
        Homogeneous transformation matrix. It is converted with ``Matrix(T)``
        and must have shape (4, 4). No full SE(3) membership validation is
        performed.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Upper-left rotation block of shape (3, 3).
    """
    T = Matrix(T)
    if T.shape != (4, 4):
        raise ValueError(f"T must be a 4x4 matrix; got shape {T.shape}.")
    return T[:3, :3]


def htm2tra(T):
    """
    Extract the translation vector from a homogeneous transformation matrix.

    Parameters
    ----------
    T : array-like or sympy Matrix
        Homogeneous transformation matrix. It is converted with ``Matrix(T)``
        and must have shape (4, 4). No full SE(3) membership validation is
        performed.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Translation column vector of shape (3, 1).
    """
    T = Matrix(T)
    if T.shape != (4, 4):
        raise ValueError(f"T must be a 4x4 matrix; got shape {T.shape}.")
    return T[:3, 3]


def invhtm(T, *, tol=1e-9):
    """Compute the structured inverse of a rigid homogeneous transform.

    Parameters
    ----------
    T : matrix-like, shape (4, 4)
        Homogeneous transformation in SE(3).
    tol : positive real, optional
        Numerical tolerance used for SE(3) validation.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Structured rigid-body inverse.
    """
    T = _validate_homogeneous_transform(T, tol=tol)
    R = T[:3, :3]
    p = T[:3, 3]

    R_inv = R.T
    p_inv = -R_inv * p
    return rt2htm(R_inv, p_inv)


def _normalize_quaternion(q, tol=1e-9):
    """Return a normalized scalar-first quaternion as a SymPy column vector."""
    tol = _validate_tol(tol)
    q = _as_vector(q, 4, name="quaternion")

    entries = [sp.sympify(value) for value in q]
    fully_numeric = all(value.is_number is True for value in entries)
    if fully_numeric:
        if any(value.is_real is not True for value in entries):
            raise ValueError("quaternion components must be real.")
        norm_value = float(sp.N(sp.sqrt(sp.simplify(q.dot(q)))))
        if norm_value <= tol:
            raise ValueError("The quaternion norm must be greater than tol.")
    else:
        norm_sq = sp.simplify(q.dot(q))
        if norm_sq.is_zero is True:
            raise ValueError("The quaternion cannot be the zero vector.")

    norm = sp.sqrt(sp.simplify(q.dot(q)))
    return sp.simplify(q / norm)


def _canonicalize_quaternion_sign(q):
    """Prefer a scalar-first quaternion with nonnegative scalar component."""
    q = Matrix(q)
    w = sp.simplify(q[0])

    if _is_numeric_real(w):
        if float(sp.N(w)) < 0:
            return -q
        return q

    if w.is_negative is True:
        return -q
    return q


def quat2rot(q, *, tol=1e-9):
    """Convert a scalar-first quaternion [w, x, y, z] to a rotation matrix."""
    q = _normalize_quaternion(q, tol=tol)
    w, x, y, z = q

    return sp.simplify(Matrix([
        [
            1 - 2 * (y**2 + z**2),
            2 * (x*y - w*z),
            2 * (x*z + w*y),
        ],
        [
            2 * (x*y + w*z),
            1 - 2 * (x**2 + z**2),
            2 * (y*z - w*x),
        ],
        [
            2 * (x*z - w*y),
            2 * (y*z + w*x),
            1 - 2 * (x**2 + y**2),
        ],
    ]))


def _rot2quat_numeric(R, tol):
    candidates = [
        1 + R[0, 0] + R[1, 1] + R[2, 2],
        1 + R[0, 0] - R[1, 1] - R[2, 2],
        1 - R[0, 0] + R[1, 1] - R[2, 2],
        1 - R[0, 0] - R[1, 1] + R[2, 2],
    ]
    values = [float(sp.N(value)) for value in candidates]
    index = max(range(4), key=values.__getitem__)

    dominant_sq = values[index]
    if dominant_sq < -tol:
        raise ValueError(
            "Rotation matrix produced an invalid quaternion component."
        )
    dominant = sp.Float(0.5) * sp.sqrt(
        sp.Float(max(0.0, dominant_sq))
    )
    denominator = 4 * dominant

    if abs(float(sp.N(denominator))) <= tol:
        raise ValueError(
            "Rotation matrix could not be converted to a stable quaternion."
        )

    if index == 0:
        w = dominant
        x = (R[2, 1] - R[1, 2]) / denominator
        y = (R[0, 2] - R[2, 0]) / denominator
        z = (R[1, 0] - R[0, 1]) / denominator
    elif index == 1:
        x = dominant
        w = (R[2, 1] - R[1, 2]) / denominator
        y = (R[0, 1] + R[1, 0]) / denominator
        z = (R[0, 2] + R[2, 0]) / denominator
    elif index == 2:
        y = dominant
        w = (R[0, 2] - R[2, 0]) / denominator
        x = (R[0, 1] + R[1, 0]) / denominator
        z = (R[1, 2] + R[2, 1]) / denominator
    else:
        z = dominant
        w = (R[1, 0] - R[0, 1]) / denominator
        x = (R[0, 2] + R[2, 0]) / denominator
        y = (R[1, 2] + R[2, 1]) / denominator

    q = _normalize_quaternion(Matrix([w, x, y, z]), tol=tol)
    return _canonicalize_quaternion_sign(q)


def rot2quat(R, tol=1e-9):
    """Convert a rotation matrix to a scalar-first unit quaternion."""
    tol = _validate_tol(tol)
    R = _validate_rotation_matrix(R, tol=tol)

    if all(_is_numeric_real(value) for value in R):
        return _rot2quat_numeric(R, tol)

    axis, angle = rot2axa(R, tol=tol)
    q = Matrix([
        sp.cos(angle / 2),
        axis[0] * sp.sin(angle / 2),
        axis[1] * sp.sin(angle / 2),
        axis[2] * sp.sin(angle / 2),
    ])
    q = _normalize_quaternion(q, tol=tol)
    return _canonicalize_quaternion_sign(q)


def axa2quat(k, theta, deg=False):
    """Convert an axis-angle orientation to a scalar-first unit quaternion."""
    k = _as_3d_vector(k, name="k")
    norm_sq = sp.simplify(k.dot(k))
    if norm_sq.is_zero is True:
        raise ValueError("The rotation axis cannot be the zero vector.")

    if deg:
        theta = deg2rad(theta, evalf=False)

    k = sp.simplify(k / k.norm())
    half = theta / 2
    return sp.simplify(Matrix([
        sp.cos(half),
        k[0] * sp.sin(half),
        k[1] * sp.sin(half),
        k[2] * sp.sin(half),
    ]))


def quat2axa(q, deg=False, *, tol=1e-9):
    """Convert a scalar-first quaternion to principal axis-angle form."""
    tol = _validate_tol(tol)
    q = _canonicalize_quaternion_sign(
        _normalize_quaternion(q, tol=tol)
    )

    w = sp.simplify(q[0])
    v = Matrix(q[1:4, 0])
    s = sp.sqrt(sp.simplify(v.dot(v)))

    if _is_numeric_real(s):
        if float(sp.N(s)) <= tol:
            axis = Matrix([1, 0, 0])
            angle = sp.S(0)
        else:
            axis = sp.simplify(v / s)
            angle = sp.simplify(2 * atan2(s, w))
    elif sp.simplify(s).is_zero is True:
        axis = Matrix([1, 0, 0])
        angle = sp.S(0)
    else:
        axis = sp.simplify(v / s)
        angle = sp.simplify(2 * atan2(s, w))

    if deg:
        angle = sp.simplify(rad2deg(angle, evalf=False))
    return axis, angle


def _rotation_angle_from_matrix(R, tol):
    """Return the principal rotation angle and its numerical/symbolic case."""
    cos_angle = sp.simplify((sp.trace(R) - 1) / 2)

    if _is_numeric_real(cos_angle):
        value = float(sp.N(cos_angle))
        if value > 1.0 + tol or value < -1.0 - tol:
            raise ValueError(
                "The rotation angle cosine is outside the valid range "
                "[-1, 1] beyond tolerance."
            )
        value = max(-1.0, min(1.0, value))
        angle = sp.acos(sp.Float(value))
        angle_value = float(sp.N(angle))

        if abs(angle_value) <= tol:
            return sp.S(0), "identity"
        if abs(angle_value - float(sp.pi)) <= tol:
            return sp.pi, "pi"
        return angle, "general"

    angle = sp.acos(cos_angle)
    angle_simplified = sp.simplify(angle)
    if angle_simplified.is_zero is True:
        return sp.S(0), "identity"
    if sp.simplify(angle_simplified - sp.pi).is_zero is True:
        return sp.pi, "pi"
    return angle_simplified, "general"


def _largest_rotation_axis_diagonal_index(diagonal):
    if all(_is_numeric_real(value) for value in diagonal):
        return max(range(3), key=lambda i: float(sp.N(diagonal[i])))

    known_nonzero = [
        i for i, value in enumerate(diagonal)
        if sp.simplify(value).is_zero is not True
    ]
    if not known_nonzero:
        return 0

    numeric_values = [sp.N(diagonal[i]) for i in known_nonzero]
    if all(value.is_number for value in numeric_values):
        return max(known_nonzero, key=lambda i: float(sp.N(diagonal[i])))
    return known_nonzero[0]


def _axis_at_pi(R):
    """Recover a unit rotation axis for an exact/numerical pi rotation."""
    A = sp.simplify((R + sp.eye(3)) / 2)
    diagonal = [sp.simplify(A[i, i]) for i in range(3)]
    i = _largest_rotation_axis_diagonal_index(diagonal)

    axis = Matrix([0, 0, 0])
    if _is_numeric_real(diagonal[i]):
        axis[i] = sp.sqrt(
            sp.Float(max(0.0, float(sp.N(diagonal[i]))))
        )
    else:
        axis[i] = sp.sqrt(diagonal[i])

    if sp.simplify(axis[i]).is_zero is True:
        raise ValueError("Could not recover a rotation axis at theta = pi.")

    for j in range(3):
        if j != i:
            axis[j] = sp.simplify(A[j, i] / axis[i])

    return sp.simplify(axis / axis.norm())


def _axis_from_rotation_matrix(R, angle, angle_case):
    if angle_case == "identity":
        return Matrix([1, 0, 0])
    if angle_case == "pi":
        return _axis_at_pi(R)

    axis = Matrix([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * sp.sin(angle))
    return sp.simplify(axis / axis.norm())


def rot2axa(R, deg=False, tol=1e-9):
    """Return the principal axis-angle representation of a rotation matrix."""
    tol = _validate_tol(tol)
    R = _validate_rotation_matrix(R, tol=tol)

    angle, angle_case = _rotation_angle_from_matrix(R, tol)
    axis = _axis_from_rotation_matrix(R, angle, angle_case)

    if deg:
        angle = sp.simplify(rad2deg(angle, evalf=False))
    return axis, sp.simplify(angle)


_ROTATION_VECTOR_SERIES_THRESHOLD = 1e-4


def rotvec2rot(phi):
    """Convert a three-component rotation vector to a rotation matrix."""
    phi = _as_3d_vector(phi, name="phi")
    entries = [sp.sympify(value) for value in phi]

    if all(value.is_number is True for value in entries):
        if any(
            value.is_real is not True or value.is_finite is not True
            for value in entries
        ):
            raise ValueError("phi components must be finite real values.")

    theta_sq = sp.simplify(phi.dot(phi))
    if theta_sq.is_zero is True:
        return sp.eye(3)

    Phi = skew(phi)

    if all(value.is_number is True for value in entries):
        theta = float(sp.N(sp.sqrt(theta_sq)))
        if theta < _ROTATION_VECTOR_SERIES_THRESHOLD:
            theta2 = theta * theta
            theta4 = theta2 * theta2
            A = 1.0 - theta2 / 6.0 + theta4 / 120.0
            B = 0.5 - theta2 / 24.0 + theta4 / 720.0
            return sp.simplify(
                sp.eye(3)
                + sp.Float(A) * Phi
                + sp.Float(B) * Phi**2
            )

    theta = sp.sqrt(theta_sq)
    A = sp.sin(theta) / theta
    B = (1 - sp.cos(theta)) / theta_sq
    return sp.simplify(sp.eye(3) + A * Phi + B * Phi**2)


def rot2rotvec(R, tol=1e-9):
    """Return the principal rotation vector of a rotation matrix."""
    tol = _validate_tol(tol)
    R = _validate_rotation_matrix(R, tol=tol)

    angle, angle_case = _rotation_angle_from_matrix(R, tol)
    if angle_case == "identity":
        return sp.zeros(3, 1)

    axis = _axis_from_rotation_matrix(R, angle, angle_case)
    return sp.simplify(angle * axis)


def axa2rot(k, theta, deg=False):
    """
    Build a rotation matrix from an axis-angle representation.

    Parameters
    ----------   
    k : list, tuple or sympy Matrix
        Rotation axis. Accepted formats are a 3-element list, a 3-element tuple,
        a column matrix of shape (3, 1), or a row matrix of shape (1, 3). The
        vector is normalized internally to a column matrix. The zero vector is
        rejected because it does not define a rotation axis.
    theta : float, int or symbolic
        Rotation angle in radians by default.
    deg : bool, optional
        If True, theta is interpreted in degrees. Default is False.

    Returns
    -------
    R : sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix of shape (3, 3) computed with Rodrigues' formula.
    """
    k = _as_3d_vector(k, name="k")
    norm_sq = sp.simplify(k.dot(k))
    if norm_sq.is_zero is True:
        raise ValueError("The rotation axis cannot be the zero vector.")

    if deg:
        theta = deg2rad(theta, evalf=False)

    k = k / k.norm()
    K = skew(k)
    return sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * K**2
    

def vex(S, *, tol=1e-9):
    """Return the vector associated with a 3x3 skew-symmetric matrix."""
    tol = _validate_tol(tol)

    try:
        S = Matrix(S)
    except (TypeError, ValueError) as exc:
        raise TypeError("S must be convertible to a 3x3 matrix.") from exc

    if S.shape != (3, 3):
        raise ValueError("S must be a 3x3 matrix.")

    residual = sp.simplify(S + S.T)
    entries = [sp.sympify(value) for value in residual]

    if all(value.is_number is True for value in entries):
        if any(value.is_real is not True for value in entries):
            raise ValueError("S must be a real skew-symmetric matrix.")
        if any(abs(float(sp.N(value))) > tol for value in entries):
            raise ValueError("S must be skew-symmetric within tolerance.")
    else:
        statuses = [sp.simplify(value).is_zero for value in entries]
        if any(status is False for status in statuses):
            raise ValueError("S must be skew-symmetric.")
        if not all(status is True for status in statuses):
            raise ValueError(
                "Skew symmetry of S could not be established symbolically."
            )

    return sp.simplify(Matrix([
        (S[2, 1] - S[1, 2]) / 2,
        (S[0, 2] - S[2, 0]) / 2,
        (S[1, 0] - S[0, 1]) / 2,
    ]))


def skew(u):
    """
    Return the skew-symmetric matrix associated with a 3D vector.

    Parameters
    ----------
    u : list, tuple or sympy Matrix
        Vector. Accepted formats are a 3-element list, a 3-element tuple, a
        column matrix of shape (3, 1), or a row matrix of shape (1, 3). The
        vector is normalized internally to a column matrix.

    Returns
    -------
    S : sympy.matrices.dense.MutableDenseMatrix
        Skew-symmetric matrix of shape (3, 3).
    """
    u = _as_3d_vector(u, name="u")
    ux,uy,uz = u
    S = Matrix([[0, -uz, uy],
                [uz, 0, -ux], 
                [-uy, ux, 0]])
    return S