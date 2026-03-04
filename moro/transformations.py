"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import sin,cos,atan2,acos,sqrt,pi
from sympy.matrices import Matrix,zeros
from moro.abc import *
from moro.util import *

__all__ = [
    "axa2rot",
    "dh",
    "eul2rot",
    "htmrot",
    "htmtra",
    "rot2eul",
    "rot2axa",
    "rot",
    "rotx",
    "roty",
    "rotz",
    "skew"
]
    
# ~ ==========================================
# ~ Transformation operations
# ~ ==========================================
def rot(theta, axis="z", deg=False):
    """
    Return a rotation matrix that represents a rotation of "theta" about "axis".

    Parameters
    ----------
    theta : float, int or `symbolic`
        Rotation angle (given in radians by default)
    axis : str
        Rotation axis, "x", "y" or "z" (default is "z")
    deg : bool
        ¿Is theta given in degrees?, False is default value.    
    """
    axis = axis.lower()
    if axis=="x":
        return rotx(theta, deg)
    elif axis=="y":
        return roty(theta, deg)
    elif axis=="z":
        return rotz(theta, deg)
    else:
        raise ValueError(f"{axis} is not a valid axis of rotation.")

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
        ¿Is theta given in degrees?, False is default value.

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
        ¿Is theta given in degrees?, False is default value.

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

    

def rot2eul(R, seq="zxz", deg=False):
    if seq in ("ZXZ","zxz"):
        return _rot2zxz(R, deg)
    elif seq in ("ZYZ","zyz"):
        return _rot2zyz(R, deg)
    else:
        raise ValueError("Currently only ZXZ and ZYZ sequence are supported")

def _rot2zxz(R, deg=False):
    """
    Calculates ZXZ Euler Angles from a rotation matrix
    """
    r33,r13,r23,r31,r32,r11,r12,r21 = R[2,2],R[0,2],R[1,2],R[2,0],R[2,1],R[0,0],R[0,1],R[1,0]
    if abs(r33) != 1:
        theta1 = atan2(sqrt(1-r33**2), r33)
        phi1 = atan2(r13, -r23)
        psi1 = atan2(r31, r32)
        theta2 = atan2(-sqrt(1-r33**2), r33)
        phi2 = atan2(-r13, r23)
        psi2 = atan2(-r31, -r32)
        solution = [(phi1,theta1,psi1), (phi2,theta2,psi2)]
    elif r33==1:
        theta = 0
        psi = 0
        phi = atan2(r21, r11)
        solution = [(phi,theta,psi)]
    elif r33==-1:
        theta = pi
        psi = 0
        phi = atan2(r21, r11)
        solution = [(phi,theta,psi)]
    else:
        pass # TODO raise an error
        
    if deg:
        solution = [(rad2deg(a), rad2deg(b), rad2deg(c)) for a,b,c in solution]
        
    return solution


def _rot2zyz(R, deg=False):
    """
    Calculates ZXZ Euler Angles from a rotation matrix
    """
    r33,r13,r23,r31,r32,r11,r12,r21 = R[2,2],R[0,2],R[1,2],R[2,0],R[2,1],R[0,0],R[0,1],R[1,0]
    if abs(r33) != 1:
        theta1 = atan2(sqrt(1-r33**2), r33)
        phi1 = atan2(r23, r13)
        psi1 = atan2(r32, -r31)
        theta2 = atan2(-sqrt(1-r33**2), r33)
        phi2 = atan2(-r23, -r13)
        psi2 = atan2(-r32, r31)
        solution = [(phi1,theta1,psi1), (phi2,theta2,psi2)]
    elif r33==1:
        theta = 0
        psi = 0
        phi = atan2(r21, r11)
        solution = [(phi,theta,psi)]
    elif r33==-1:
        theta = pi
        psi = 0
        phi = atan2(-r21, -r11)
        solution = [(phi,theta,psi)]
    else:
        pass # TODO raise an error
        
    if deg:
        solution = [(rad2deg(a), rad2deg(b), rad2deg(c)) for a,b,c in solution]
        
    return solution

def eul2rot(phi,theta,psi,seq="zxz",deg=False):
    if deg: # If angles are given in degrees -> convert to radians
        phi,theta,psi = deg2rad(Matrix([phi,theta,psi]), evalf=False)
    seq = seq.lower()

    if not seq in ("zxz","zyz","xyx","xzx","yxy","yzy"):
        raise ValueError(f"{seq} is not a valid sequence")

    axis1 = seq[0]
    axis2 = seq[1]
    axis3 = seq[2]
    R = rot(phi,axis1) * rot(theta,axis2) * rot(psi,axis3)
    return R

def htmtra(*args,**kwargs):
    """
    Calculate the homogeneous transformation matrix of a translation
    
    Parameters
    ----------
    *args : list, tuple, int, float
        Translation vector or components

    **kwargs : float, int
        dx, dy and dz keyword arguments
    
    Returns
    -------
    H : :class:`sympy.matrices.dense.MutableDenseMatrix`
        Homogeneous transformation matrix
        
        
    Examples
    --------
    >>> htmtra([50,-100,30])
    ⎡1  0  0   50 ⎤
    ⎢             ⎥
    ⎢0  1  0  -100⎥
    ⎢             ⎥
    ⎢0  0  1   30 ⎥
    ⎢             ⎥
    ⎣0  0  0   1  ⎦
    
    >>> a,b,c = symbols("a,b,c")
    >>> htmtra([a,b,c])
    ⎡1  0  0  a⎤
    ⎢          ⎥
    ⎢0  1  0  b⎥
    ⎢          ⎥
    ⎢0  0  1  c⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦

    Using float/integer arguments:

    >>> htmtra(10,-40,50)
    ⎡1  0  0  10 ⎤
    ⎢            ⎥
    ⎢0  1  0  -40⎥
    ⎢            ⎥
    ⎢0  0  1  50 ⎥
    ⎢            ⎥
    ⎣0  0  0   1 ⎦

    Using keyword arguments:

    >>> htmtra(dz=100,dx=300,dy=-200)
    ⎡1  0  0  300 ⎤
    ⎢             ⎥
    ⎢0  1  0  -200⎥
    ⎢             ⎥
    ⎢0  0  1  100 ⎥
    ⎢             ⎥
    ⎣0  0  0   1  ⎦

    """
    if args and not kwargs:
        if isinstance(args[0], (list,tuple)):
            d = args[0]
        elif len(args)==3:
            d = args
    elif kwargs and not args:
        d = [0,0,0]
        if "dx" in kwargs: 
            d[0] = kwargs.get("dx")
        if "dy" in kwargs:
            d[1] = kwargs.get("dy")
        if "dz" in kwargs:
            d[2] = kwargs.get("dz")
    else:
        raise ValueError("Only pass *args or **kwargs, not both")

    dx,dy,dz = d[0],d[1],d[2]
    M = Matrix([[1,0,0,dx],
                [0,1,0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    return M
    

def htmrot(theta, axis="z", deg=False):
    """
    Return a homogeneous transformation matrix that represents a 
    rotation "theta" about "axis". 
    
    Parameters
    ----------
    theta : float, int or `symbolic`
        Rotation angle (given in radians by default)
        
    axis : str
        Rotation axis

    deg : bool
        ¿Is theta given in degrees?
        
    Returns
    -------
    H : :class:`sympy.matrices.dense.MutableDenseMatrix`
        Homogeneous transformation matrix
        
    
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
    if deg: # Is theta given in degrees? -> then convert to radians
        theta = deg2rad(theta)
        
    if axis in ("z","Z",3,"3"):
        R = rotz(theta)
    elif axis in ("y","Y",2,"2"):
        R = roty(theta)
    elif axis in ("x","X",1,"1"):
        R = rotx(theta)
    else:
        raise ValueError("The axis is invalid, axis must be 'x', 'y' or 'z'")
    H = _rot2htm(R)
    return H


def _rot2htm(R):
    """
    Given a SO(3) matrix return a SE(3) homogeneous 
    transformation matrix.
    """
    _H = R.row_join(zeros(3,1))
    H = _H.col_join(Matrix([0,0,0,1]).T)
    return H
    

def rot2axa(R, deg=False):
    """
    Given a SO(3) matrix return the axis-angle representation.

    Parameters
    ---------- 

    R : `sympy.matrices.dense.MutableDenseMatrix`
        Rotation matrix in SO(3).

    deg : bool
        If True, the angle is returned in degrees. By default, it is False (angle in radians or symbolic).

    Returns
    -------
    k : `sympy.matrices.dense.MutableDenseMatrix`
        Axis of rotation, a 3D vector.
    theta : float, int or symbolic
        Rotation angle (given in radians by default, or symbolic).
    """
    if not(is_SO3(R)):
        raise ValueError("R must be a rotation matrix.")
    
    # trace
    trace = sp.trace(R)

    # angle 
    angle = sp.acos((trace - 1) / 2)

    # Case 1: angle = 0
    # In this case, the rotation is the identity, so we can return any axis (we choose the x-axis) and an angle of 0.
    if sp.simplify(angle) == 0:
        return Matrix([1, 0, 0]), sp.S(0)

    # Case 2: angle = pi
    # In this case, the axis can be computed from the diagonal elements of R, but we need to be careful with the signs.
    if sp.simplify(angle - sp.pi) == 0:
        axis = Matrix([
            sp.sqrt((R[0,0] + 1)/2),
            sp.sqrt((R[1,1] + 1)/2),
            sp.sqrt((R[2,2] + 1)/2)
        ])
        axis = axis / axis.norm()
        return axis, angle

    # Case 3: general case
    axis = Matrix([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2 * sp.sin(angle))

    axis = sp.simplify(axis / axis.norm())

    return axis, sp.simplify(angle)
    
def axa2rot(k,theta):
    """
    Given a R^3 vector (k) and an angle (theta), return 
    the SO(3) matrix associated.

    Parameters
    ----------   
    k : `sympy.matrices.dense.MutableDenseMatrix` or list or tuple
        Axis of rotation, must be a 3D vector. If k is given as a list or tuple, it will be converted to a sympy Matrix.
    theta : float, int or symbolic
        Rotation angle (given in radians by default).

    Returns
    -------
    R : `sympy.matrices.dense.MutableDenseMatrix`
        Rotation matrix in SO(3) corresponding to a rotation of "theta" about the axis defined by "k".
    """
    if isinstance(k,(list,tuple)):
        k = Matrix(k)
    ct = cos(theta)
    st = sin(theta)
    vt = 1 - cos(theta)
    kx,ky,kz = k.normalized()
    r11 = kx**2*vt + ct
    r21 = kx*ky*vt + kz*st
    r31 = kx*kz*vt - ky*st
    r12 = kx*ky*vt - kz*st
    r22 = ky**2*vt + ct
    r32 = ky*kz*vt + kx*st
    r13 = kx*kz*vt + ky*st 
    r23 = ky*kz*vt - kx*st 
    r33 = kz**2*vt + ct 
    R = Matrix([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    return R
    

def skew(u):
    """
    Return skew-symmetric matrix associated to u vector.

    Parameters
    ----------
    u : `sympy.matrices.dense.MutableDenseMatrix` or list or tuple
        A 3D vector. If u is given as a list or tuple, it will be converted to a sympy Matrix.

    Returns
    -------
    S : `sympy.matrices.dense.MutableDenseMatrix`
        Skew-symmetric matrix associated to u.  
    """
    if len(u) != 3:
        raise ValueError("The vector u must have three components.")
    ux,uy,uz = u
    S = Matrix([[0, -uz, uy],
                [uz, 0, -ux], 
                [-uy, ux, 0]])
    return S
    

if __name__=="__main__":
    pass