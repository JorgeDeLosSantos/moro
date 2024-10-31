"""
Numython R&D, (c) 2024
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sympy import sin,cos,atan2,acos,sqrt,pi
from sympy.matrices import Matrix,eye
from moro.abc import *
from moro.util import *

__all__ = [
    "axa2rot",
    "compose_rotations",
    "dh",
    "eul2htm",
    "eul2rot",
    "htm2eul",
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
    Calculates the rotation matrix about the z-axis

    Parameters
    ----------
    :param theta: Rotation angle (given in radians by default)
    :type theta: float, int or `symbolic`

    :param deg: ¿Is theta given in degrees?, False is default value.
    :type deg: bool

    Returns
    -------
    :return: `sympy.matrices.dense.MutableDenseMatrix`
    :rtype: Rotation matrix (SO3)

    Examples
    --------
    
    Using angle in radians,
    
    >>> rotz(pi/2)
    ⎡0  -1  0⎤
    ⎢        ⎥
    ⎢1  0   0⎥
    ⎢        ⎥
    ⎣0  0   1⎦

    Or symbolic variables,
    
    >>> x = symbols("x")
    >>> rotz(x)
    ⎡cos(x)  -sin(x)  0⎤
    ⎢                  ⎥
    ⎢sin(x)  cos(x)   0⎥
    ⎢                  ⎥
    ⎣  0        0     1⎦

    Using angles in degrees:
    
    >>> rotz(45, deg=True)
    ⎡0.707106781186548  -0.707106781186547  0⎤
    ⎢                                        ⎥
    ⎢0.707106781186547  0.707106781186548   0⎥
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
    :param theta: Rotation angle (given in radians by default)
    :type theta: float, int or `symbolic`

    :param deg: ¿Is theta given in degrees?, False is default value.
    :type deg: bool

    Returns
    -------
    :return: `sympy.matrices.dense.MutableDenseMatrix`
    :rtype: Rotation matrix (SO3)
        
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
    :param theta: Rotation angle (given in radians by default)
    :type theta: float, int or `symbolic`

    :param deg: ¿Is theta given in degrees?, False is default value.
    :type deg: bool

    Returns
    -------
    :return: `sympy.matrices.dense.MutableDenseMatrix`
    :rtype: Rotation matrix (SO3)
        
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


def _rot(theta, axis, deg=False):
    if axis in ("X","x",1,"1"):
        R = rotx(theta, deg)
    elif axis in ("Y","y",2,"2"):
        R = roty(theta, deg)
    elif axis in ("Z","z",3,"3"):
        R = rotz(theta, deg)
    else:
        R = eye(3)
    return R
    

def compose_rotations(*rotations):
    """
    Composes rotation matrices w.r.t. fixed or movable frames
    
    Parameters
    ----------
    :param rotations: A tuple that contains (angle, axis, frame, deg)
    :type rotations: tuple

    Returns
    -------
    :return: Rotation matrix
    :rtype: :class:`sympy.matrices.dense.MutableDenseMatrix`
        
    Examples
    --------
    >>> compose_rotations((45, "z", "fixed", True), (30, "x", "local", True))
    ⎡0.707106781186548  -0.612372435695794  0.353553390593274 ⎤
    ⎢                                                         ⎥
    ⎢0.707106781186547  0.612372435695795   -0.353553390593274⎥
    ⎢                                                         ⎥
    ⎣        0                 0.5          0.866025403784439 ⎦
    """
    R = eye(3) # I3x3 matrix
    for rot in rotations:
        angle,axis,frame,*_ = rot
        if len(rot)==4:
            deg = rot[-1]
        else:
            deg = False # default value
        crm = _rot(angle,axis,deg)
        if frame in ("world","fixed","global","w","0",0):
            R = crm*R
        elif frame in ("current","movable","local","c","1",1):
            R = R*crm
        else:
            pass # Nothing to do here -> raise except. (to impl.)
            
    return R


def dh(a,alpha,d,theta):
    """
    Calculates Denavit-Hartenberg matrix given the four parameters.

    Parameters
    ----------
    :param a: DH parameter
    :type a: int, float or symbol

    :param alpha: DH parameter
    :type alpha: int, float or symbol

    :param d: DH parameter
    :type d: int, float or symbol

    :param theta: DH parameter
    :type theta: int, float or symbol    

    Returns
    -------

    :return: Denavit-Hartenberg matrix (4x4)
    :rtype: :class:`sympy.matrices.dense.MutableDenseMatrix`
        

    Examples
    --------
    
    With numerical values:
    
    >>> dh(100,pi/2,50,pi/2)
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
    >>> dh(a,0,0,t)
    ⎡cos(t)  -sin(t)  0  a⋅cos(t)⎤
    ⎢                            ⎥
    ⎢sin(t)  cos(t)   0  a⋅sin(t)⎥
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
    
    
def eul2htm(phi,theta,psi,seq="zxz",deg=False):
    """
    Given a set of Euler Angles (phi,theta,psi) for specific 
    sequence this function returns the homogeneous transformation 
    matrix associated. Default sequence is ZXZ.

    Parameters
    ----------

    phi: int, float, symbol
        First angle of the set

    theta: int, float, symbol
        Second angle of the set

    psi: int, float, symbol
        Third angle of the set
        
    deg: bool
        This parameter is True if phi, theta, and psi are given in degrees, 
        by default it's assumed to be False (angles in radians).

    Returns
    -------

    H: `sympy.matrices.dense.MutableDenseMatrix`
        A homogeneous transformation matrix (SE(3)).
    
    
    Examples
    --------
    
    >>> eul2htm(90,90,90,"zxz",True)
    ⎡0  0   1  0⎤
    ⎢           ⎥
    ⎢0  -1  0  0⎥
    ⎢           ⎥
    ⎢1  0   0  0⎥
    ⎢           ⎥
    ⎣0  0   0  1⎦
    
    >>> eul2htm(pi/2,pi/2,pi/2)
    ⎡0  0   1  0⎤
    ⎢           ⎥
    ⎢0  -1  0  0⎥
    ⎢           ⎥
    ⎢1  0   0  0⎥
    ⎢           ⎥
    ⎣0  0   0  1⎦
        
    >>> eul2htm(0,pi/2,0,"zyz")
    ⎡0   0  1  0⎤
    ⎢           ⎥
    ⎢0   1  0  0⎥
    ⎢           ⎥
    ⎢-1  0  0  0⎥
    ⎢           ⎥
    ⎣0   0  0  1⎦
    """
    if deg: # If angles are given in degrees -> convert to radians
        phi,theta,psi = deg2rad(Matrix([phi,theta,psi]), evalf=False)
    seq = seq.lower()

    if not seq in ("zxz","zyz","xyx","xzx","yxy","yzy"):
        raise ValueError(f"{seq} is not a valid sequence")

    axis1 = seq[0]
    axis2 = seq[1]
    axis3 = seq[2]
    H = htmrot(phi,axis1) * htmrot(theta,axis2) * htmrot(psi,axis3)

    return H
    
    
def htm2eul(H, seq="zxz", deg=False):
    """
    Given a homogeneous transformation matrix this function 
    return the equivalent set of Euler Angles. 
    
    If "deg" is True then Euler Angles are converted to degrees.
    
    >>> H = htmrot(pi/3,"y")*htmrot(pi/4,"x")
    >>> H
    ⎡      √6   √6    ⎤
    ⎢1/2   ──   ──   0⎥
    ⎢      4    4     ⎥
    ⎢                 ⎥
    ⎢      √2  -√2    ⎥
    ⎢ 0    ──  ────  0⎥
    ⎢      2    2     ⎥
    ⎢                 ⎥
    ⎢-√3   √2   √2    ⎥
    ⎢────  ──   ──   0⎥
    ⎢ 2    4    4     ⎥
    ⎢                 ⎥
    ⎣ 0    0    0    1⎦
    >>> htm2eul(H)
    ⎛    ⎛√3⎞                     ⎞
    ⎜atan⎜──⎟, atan(√7), -atan(√6)⎟
    ⎝    ⎝2 ⎠                     ⎠
    >>> htm2eul(H, deg=True)
    (40.8933946491309, 69.2951889453646, -67.7923457014035)
    """
    if seq in ("ZXZ","zxz"):
        return _htm2zxz(H, deg)
    # elif seq in ("ZYZ","zyz"):
    #     return _htm2zyz(H, deg)
    else:
        raise ValueError("Currently only ZXZ sequence is supported")


def _htm2zxz(H, deg=False):
    """
    Calculates ZXZ Euler Angles from a homogeneous transformation matrix
    """
    R = H[:3,:3] # rotation sub-matrix
    r33,r13,r23,r31,r32,r11,r12,r21 = R[2,2],R[0,2],R[1,2],R[2,0],R[2,1],R[0,0],R[0,1],R[1,0]
    if abs(r33) != 1:
        theta = atan2(sqrt(1-r33**2), r33)
        phi = atan2(r13, -r23)
        psi = atan2(r31, r32)
    elif r33==1:
        theta = 0
        phi = 0
        psi = atan2(r21, r11)
    elif r33==-1:
        theta = pi
        psi = 0
        phi = atan2(r21, r11)
    else:
        theta = atan2(sqrt(1-r33**2), r33)
        phi = atan2(r13,-r23)
        psi = atan2(r31,r32)
        
    if deg:
        return rad2deg(phi), rad2deg(theta), rad2deg(psi)
        
    return phi,theta,psi


def rot2eul(R, seq="zxz", deg=False):
    if seq in ("ZXZ","zxz"):
        return _rot2zxz(R, deg)
    elif seq in ("ZYZ","zyz"):
        return _rot2zyz(R, deg)
    else:
        raise ValueError("Currently only ZXZ and ZYZ sequence is supported")

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
        phi = 0
        psi = atan2(r21, r11)
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
        phi = 0
        psi = atan2(r21, r11)
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
    """
    if not(is_SO3(R)):
        raise ValueError("R must be a rotation matrix.")
    r32,r23 = R[2,1],R[1,2]
    r13,r31 = R[0,2],R[2,0]
    r21,r12 = R[1,0],R[0,1]
    theta = acos((R.trace() - 1)/2)
    k = ( (1/(2*sin(theta)))*Matrix([r32-r23, r13-r31, r21-r12]) ).evalf()
    if deg:
        theta = rad2deg(theta)
    return k,theta
    
def axa2rot(k,theta):
    """
    Given a R^3 vector (k) and an angle (theta), return 
    the SO(3) matrix associated.
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
    Return skew-symmetric matrix associated to u vector 
    """
    if len(u) != 3:
        raise ValueError("The vector u must have three components.")
    ux,uy,uz = u
    S = Matrix([[0, -uz, uy],
                [uz, 0, -ux], 
                [-uy, ux, 0]])
    return S
    

if __name__=="__main__":
    print(eul2htm(0,0,pi,"zxz"))