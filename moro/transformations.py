"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import sympy as sp
from sympy import sin, cos, atan2, sqrt, pi
from sympy.matrices import Matrix, MatrixBase
from moro.util import deg2rad, is_SO3, rad2deg

__all__ = [
    "axa2rot",
    "dh",
    "eul2rot",
    "htmrot",
    "htmtra",
    "htm2rot",
    "htm2tra",
    "invhtm",
    "rot2eul",
    "rot2axa",
    "rot2htm",
    "rot",
    "rotx",
    "roty",
    "rotz",
    "rt2htm",
    "skew"
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


def _as_3d_vector(v, name="vector"):
    """
    Convert supported 3D vector inputs to a SymPy column matrix.
    """
    try:
        vector = Matrix(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a 3D vector given as a list, tuple, column matrix (3, 1) "
            "or row matrix (1, 3)."
        ) from exc

    if vector.shape == (3, 1):
        return vector
    if vector.shape == (1, 3):
        return vector.T

    raise ValueError(
        f"{name} must be a 3D vector with shape (3, 1) or (1, 3); got shape {vector.shape}."
    )


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
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
)


def _normalize_euler_sequence(seq):
    valid_sequences = "'xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz'"
    if not isinstance(seq, str):
        raise ValueError(f"seq must be one of: {valid_sequences}.")

    seq = seq.lower()
    if seq not in _PROPER_EULER_SEQUENCES:
        raise ValueError(f"seq must be one of: {valid_sequences}.")

    return seq


# Configuration for proper Euler sequences under the convention
# R = R_a(phi) @ R_b(theta) @ R_a(psi), with active rotations and column vectors.
# Each atan2 pair is encoded as ((sin_sign, sin_i, sin_j), (cos_sign, cos_i, cos_j)).
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


def rot2eul(R, seq="zxz", deg=False, tol=1e-9):
    """
    Calculate proper Euler angles from a rotation matrix.

    Parameters
    ----------
    R : matrix-like, shape (3, 3)
        Rotation matrix. The function validates only that the input has shape
        ``(3, 3)``; it does not yet perform a full SO(3) membership check.
    seq : str, optional
        Proper Euler sequence. Supported sequences are ``"xyx"``, ``"xzx"``,
        ``"yxy"``, ``"yzy"``, ``"zxz"`` and ``"zyz"``. Matching is
        case-insensitive.
    deg : bool, optional
        If True, returned angles are converted from radians to degrees.
    tol : float, optional
        Positive numerical tolerance used only for floating-point classification
        near the singularities ``theta = 0`` and ``theta = pi`` and for clipping
        small numerical excursions of ``cos(theta)`` outside ``[-1, 1]``.

    Returns
    -------
    list of tuple
        In the general case, returns two equivalent solutions
        ``[(phi1, theta1, psi1), (phi2, theta2, psi2)]``. In singular cases,
        returns a single representative solution with ``psi = 0``.

    Notes
    -----
    The convention matches :func:`eul2rot`: column vectors, active rotations and
    ``R = R_a(phi) @ R_b(theta) @ R_a(psi)`` for ``seq="aba"``. Euler angle
    representations are not unique; both general-case solutions reconstruct the
    same matrix, the second solution may contain a negative intermediate angle,
    and no additional range normalization is applied. At singularities, ``phi``
    and ``psi`` are not independently determined; setting ``psi = 0`` is only a
    representative convention.
    """
    _validate_euler_tol(tol)
    seq = _normalize_euler_sequence(seq)
    R = Matrix(R)
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix.")

    return _rot2proper_euler(R, seq, deg, tol)

def _validate_euler_tol(tol):
    if tol <= 0:
        raise ValueError("tol must be greater than 0.")


def _is_numeric_real(value):
    value = sp.simplify(value)
    numeric_value = sp.N(value)
    return not value.free_symbols and numeric_value.is_real is True


def _has_float(value):
    return bool(sp.sympify(value).atoms(sp.Float))


def _is_SO3_numeric_tol(R, tol):
    R = Matrix(R)
    if R.shape != (3, 3):
        return False
    if not all(_is_numeric_real(value) for value in R):
        return False

    orthogonality_error = R.T * R - sp.eye(3)
    if any(abs(float(sp.N(value))) > tol for value in orthogonality_error):
        return False

    determinant_error = sp.det(R) - 1
    return abs(float(sp.N(determinant_error))) <= tol


def _classify_euler_cos(value, tol):
    value_simplified = sp.simplify(value)

    if _has_float(value_simplified) and _is_numeric_real(value_simplified):
        numeric_value = float(sp.N(value_simplified))
        if numeric_value > 1.0 + tol or numeric_value < -1.0 - tol:
            raise ValueError("cos(theta) is outside the valid range [-1, 1] beyond tolerance.")
        numeric_value = max(-1.0, min(1.0, numeric_value))

        if abs(numeric_value - 1.0) <= tol:
            return "positive_singularity", sp.S(1)
        if abs(numeric_value + 1.0) <= tol:
            return "negative_singularity", sp.S(-1)
        return "general", sp.Float(numeric_value)

    is_positive_singularity = sp.simplify(value_simplified - 1).is_zero
    is_negative_singularity = sp.simplify(value_simplified + 1).is_zero

    if is_positive_singularity is True:
        return "positive_singularity", sp.S(1)
    if is_negative_singularity is True:
        return "negative_singularity", sp.S(-1)
    if is_positive_singularity is False and is_negative_singularity is False:
        return "general", value_simplified

    # Completely symbolic matrices without enough assumptions are processed
    # through the general branch to avoid undecidable boolean comparisons.
    return "symbolic", value_simplified


def _euler_sqrt_term(cos_theta):
    if _has_float(cos_theta) and _is_numeric_real(cos_theta):
        value = float(sp.N(cos_theta))
        radicand = max(0.0, 1.0 - value**2)
        return sqrt(sp.Float(radicand))
    return sqrt(sp.simplify(1 - cos_theta**2))


def _signed_matrix_element(R, term):
    sign, i, j = term
    return sign * R[i, j]


def _atan2_from_config(R, pair):
    sin_term, cos_term = pair
    return atan2(_signed_matrix_element(R, sin_term), _signed_matrix_element(R, cos_term))


def _negated_pair(pair):
    sin_term, cos_term = pair
    return ((-sin_term[0], sin_term[1], sin_term[2]), (-cos_term[0], cos_term[1], cos_term[2]))


def _convert_euler_solutions_to_degrees(solution):
    return [(rad2deg(a), rad2deg(b), rad2deg(c)) for a,b,c in solution]


def _rot2proper_euler(R, seq, deg=False, tol=1e-9):
    config = _PROPER_EULER_CONFIG[seq]
    i, j = config["cos_index"]
    cos_theta = R[i, j]
    theta_case, cos_theta = _classify_euler_cos(cos_theta, tol)

    if theta_case in ("general", "symbolic"):
        sqrt_term = _euler_sqrt_term(cos_theta)
        theta1 = atan2(sqrt_term, cos_theta)
        phi1 = _atan2_from_config(R, config["phi"])
        psi1 = _atan2_from_config(R, config["psi"])
        theta2 = atan2(-sqrt_term, cos_theta)
        phi2 = _atan2_from_config(R, _negated_pair(config["phi"]))
        psi2 = _atan2_from_config(R, _negated_pair(config["psi"]))
        solution = [(phi1,theta1,psi1), (phi2,theta2,psi2)]
    elif theta_case == "positive_singularity":
        theta = 0
        psi = 0
        phi = _atan2_from_config(R, config["singular_positive"])
        solution = [(phi,theta,psi)]
    elif theta_case == "negative_singularity":
        theta = pi
        psi = 0
        phi = _atan2_from_config(R, config["singular_negative"])
        solution = [(phi,theta,psi)]

    if deg:
        solution = _convert_euler_solutions_to_degrees(solution)

    return solution


def _rot2zxz(R, deg=False, tol=1e-9):
    """
    Calculates ZXZ Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "zxz", deg, tol)


def _rot2zyz(R, deg=False, tol=1e-9):
    """
    Calculates ZYZ Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "zyz", deg, tol)


def _rot2xyx(R, deg=False, tol=1e-9):
    """
    Calculates XYX Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "xyx", deg, tol)


def _rot2xzx(R, deg=False, tol=1e-9):
    """
    Calculates XZX Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "xzx", deg, tol)


def _rot2yxy(R, deg=False, tol=1e-9):
    """
    Calculates YXY Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "yxy", deg, tol)


def _rot2yzy(R, deg=False, tol=1e-9):
    """
    Calculates YZY Euler angles from a rotation matrix.
    """
    return _rot2proper_euler(R, "yzy", deg, tol)

def eul2rot(phi,theta,psi,seq="zxz",deg=False):
    """
    Build a rotation matrix from proper Euler angles.

    Parameters
    ----------
    phi : int, float or symbolic
        First Euler angle.
    theta : int, float or symbolic
        Intermediate Euler angle.
    psi : int, float or symbolic
        Third Euler angle.
    seq : str, optional
        Proper Euler sequence. Supported sequences are ``"xyx"``, ``"xzx"``,
        ``"yxy"``, ``"yzy"``, ``"zxz"`` and ``"zyz"``. Matching is
        case-insensitive. Tait-Bryan sequences such as ``"xyz"`` are not
        supported here.
    deg : bool, optional
        If True, the input angles are interpreted as degrees and converted to
        radians before constructing the matrix.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix.

    Notes
    -----
    This function uses column vectors and active rotations. For a sequence
    ``seq="abc"``, the convention is defined by the matrix product
    ``R = R_a(phi) @ R_b(theta) @ R_c(psi)``, where each elementary rotation is
    produced by :func:`rot`. For proper Euler sequences, ``a == c``.

    Examples
    --------
    >>> eul2rot(pi/2, pi/3, pi/4, seq="zxz")
    ⎡-√2   -√6         ⎤
    ⎢────  ────   √3/2 ⎥
    ⎢ 4     4          ⎥
    ⎢                  ⎥
    ⎢-√2    √6         ⎥
    ⎢────   ──   -1/2  ⎥
    ⎢ 4     4          ⎥
    ⎢                  ⎥
    ⎢ √6    √2         ⎥
    ⎢ ──    ──    1/2  ⎥
    ⎣ 4     4          ⎦

    >>> eul2rot(pi/6, pi/4, pi/3, seq="xyx")
    ⎡√2              √2        ⎤
    ⎢──      √6/4    ──        ⎥
    ⎢2               4         ⎥
    ⎢                          ⎥
    ⎢√2    3/8 + √3  1   3⋅√3 ⎥
    ⎢──    ────────  ─ - ──── ⎥
    ⎢4        4      8    8   ⎥
    ⎢                          ⎥
    ⎢-√6   1   3⋅√3  √3   3/8⎥
    ⎢────  ─ + ────  ── - ───⎥
    ⎣ 4    8    8    4     4 ⎦
    """
    if deg: # If angles are given in degrees -> convert to radians
        phi,theta,psi = deg2rad(Matrix([phi,theta,psi]), evalf=False)
    seq = _normalize_euler_sequence(seq)

    axis1 = seq[0]
    axis2 = seq[1]
    axis3 = seq[2]
    R = rot(phi,axis1) * rot(theta,axis2) * rot(psi,axis3)
    return R

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


def invhtm(T):
    """
    Compute the structured inverse of a homogeneous transformation matrix.

    Parameters
    ----------
    T : array-like or sympy Matrix
        Homogeneous transformation matrix. It is converted with ``Matrix(T)``
        and must have shape (4, 4). No full SE(3) membership validation is
        performed.

    Returns
    -------
    sympy.matrices.dense.MutableDenseMatrix
        Inverse homogeneous transformation matrix computed from the rigid-body
        structure, using ``R.T`` and ``-R.T*p`` instead of a general matrix
        inverse.
    """
    T = Matrix(T)
    if T.shape != (4, 4):
        raise ValueError(f"T must be a 4x4 matrix; got shape {T.shape}.")
    R = htm2rot(T)
    p = htm2tra(T)
    R_inv = R.T
    p_inv = -R_inv * p
    return rt2htm(R_inv, p_inv)

def rot2axa(R, deg=False, tol=1e-9):
    """
    Return the axis-angle representation of a rotation matrix.

    Parameters
    ---------- 

    R : sympy Matrix
        Rotation matrix in SO(3).

    deg : bool, optional
        If True, the angle is returned in degrees. Default is False.

    tol : float, optional
        Positive tolerance used to validate numeric rotation matrices, classify
        angles close to 0, classify angles close to pi, and tolerate small
        floating-point errors in trigonometric quantities. Default is 1e-9.

    Returns
    -------
    k : sympy.matrices.dense.MutableDenseMatrix
        Axis of rotation, a 3D vector.
    theta : float, int or symbolic
        Rotation angle in radians by default, or in degrees when ``deg=True``.
    """
    if tol <= 0:
        raise ValueError("tol must be greater than 0.")

    if not(is_SO3(R)) and not _is_SO3_numeric_tol(R, tol):
        raise ValueError("R must be a rotation matrix.")

    def _result(axis, angle):
        axis = sp.simplify(axis / axis.norm())
        angle = sp.simplify(angle)
        if deg:
            angle = sp.simplify(rad2deg(angle, evalf=False))
        return axis, angle

    def _largest_diagonal_index(diagonal):
        if all(_has_float(value) and _is_numeric_real(value) for value in diagonal):
            return max(range(3), key=lambda i: float(sp.N(diagonal[i])))

        known_nonzero = [i for i, value in enumerate(diagonal) if sp.simplify(value) != 0]
        if not known_nonzero:
            return 0
        numeric_values = [sp.N(diagonal[i]) for i in known_nonzero]
        if all(value.is_number for value in numeric_values):
            return max(known_nonzero, key=lambda i: sp.N(diagonal[i]))
        return known_nonzero[0]

    def _angle_from_cos(cos_angle):
        cos_angle = sp.simplify(cos_angle)
        if _has_float(cos_angle) and _is_numeric_real(cos_angle):
            value = float(sp.N(cos_angle))
            if value > 1.0 + tol or value < -1.0 - tol:
                raise ValueError("The rotation angle cosine is outside the valid range [-1, 1] beyond tolerance.")
            value = max(-1.0, min(1.0, value))
            return sp.acos(sp.Float(value)), value
        return sp.acos(cos_angle), None

    def _angle_case(angle, numeric_cos_angle):
        if numeric_cos_angle is not None:
            angle_value = float(sp.N(angle))
            if abs(angle_value) <= tol:
                return "identity"
            if abs(angle_value - float(sp.pi)) <= tol:
                return "pi"
            return "general"

        angle_simplified = sp.simplify(angle)
        is_zero = angle_simplified.is_zero
        is_pi = sp.simplify(angle_simplified - sp.pi).is_zero
        if is_zero is True:
            return "identity"
        if is_pi is True:
            return "pi"
        return "general"
    
    cos_angle = (sp.trace(R) - 1) / 2
    angle, numeric_cos_angle = _angle_from_cos(cos_angle)
    angle_case = _angle_case(angle, numeric_cos_angle)

    # Case 1: angle = 0
    # In this case, the rotation is the identity, so we can return any axis (we choose the x-axis) and an angle of 0.
    if angle_case == "identity":
        return _result(Matrix([1, 0, 0]), sp.S(0))

    # Case 2: angle = pi
    # In this case, R = 2*k*k.T - I, so A = (R + I)/2 = k*k.T.
    # Select the largest available diagonal term to recover the most stable component,
    # then use off-diagonal terms to preserve the relative signs of the axis components.
    if angle_case == "pi":
        A = sp.simplify((R + sp.eye(3)) / 2)
        diagonal = [sp.simplify(A[i, i]) for i in range(3)]
        i = _largest_diagonal_index(diagonal)
        axis = Matrix([0, 0, 0])
        axis[i] = sp.sqrt(max(0.0, float(sp.N(diagonal[i])))) if _has_float(diagonal[i]) and _is_numeric_real(diagonal[i]) else sp.sqrt(diagonal[i])

        for j in range(3):
            if j != i:
                axis[j] = sp.simplify(A[j, i] / axis[i])

        return _result(axis, angle)

    # Case 3: general case
    axis = Matrix([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2 * sp.sin(angle))

    return _result(axis, angle)
    
def axa2rot(k,theta):
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
        Rotation angle in radians.

    Returns
    -------
    R : sympy.matrices.dense.MutableDenseMatrix
        Rotation matrix of shape (3, 3) computed with Rodrigues' formula.
    """
    k = _as_3d_vector(k, name="k")
    norm_sq = sp.simplify(k.dot(k))
    if norm_sq.is_zero is True:
        raise ValueError("The rotation axis cannot be the zero vector.")

    k = k / k.norm()
    K = skew(k)
    return sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * K**2
    

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