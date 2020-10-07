"""

"""
from sympy import pi,sin,cos,tan
from sympy.matrices import Matrix,eye
from itertools import combinations
# ~ from scipy.spatial import Delaunay, ConvexHull
import numpy as np
import sympy.core as sc
import sympy.matrices as sm
from sympy.core.basic import Basic

__all__ = [
    "deg2rad",
    "ishtm",
    "isorthonormal",
    "isrot",
    "rad2deg",
    "sympy2float",
    "sympy_matrix_to_numpy_float",
    "issympyobject",
    "hcoords"
]

def deg2rad(theta, evalf=True):
    """
    Convert degrees to radians 
    
    Parameters
    ----------
    
    theta : float, int, symbolic
    
    Returns
    -------
    
    theta_rad : symbolic
    """
    if evalf:
        theta_rad = ( theta*(pi/180) ).evalf()
    else:
        theta_rad = theta*(pi/180)
    return theta_rad


def rad2deg(theta, evalf=True):
    """
    Convert radians to degrees 
    
    Parameters
    ----------
    
    theta : float, int, symbolic
    
    Returns
    -------
    
    theta_deg : symbolic
    """
    if evalf:
        theta_deg = ( theta*(180/pi) ).evalf()
    else:
        theta_deg = theta*(180/pi)
    return theta_deg


def issympyobject(obj):
    """
    Determine if input (obj) is a sympy object.
    
    Examples
    --------
    >>> from sympy import symbols
    >>> x = symbols("x")
    >>> issympyobject(x)
    True
    """
    if isinstance( obj, tuple(sc.all_classes ) ):
        return True
    elif isinstance(obj, Basic):
        return True
    elif isinstance(obj, sm.MatrixBase):
        return True
    else:
        return False

    
def ishtm(H):
    """
    Is H a homogeneous transformation matrix ?
    """
    nrow,ncol = H.shape
    if nrow == ncol == 4:
        if isrot(H[:3,:3]) and H[3,3]==1 and not any(H[3,:3]):
            return True
    return False
    

def isrot(R):
    """
    Is R a rotation matrix ?
    
    Parameters
    ----------
    
    R : `sympy.matrices.dense.MutableDenseMatrix`
    
    Returns
    -------
    
    False or True
    
    """
    nrow,ncol = R.shape
    if (nrow == ncol == 3) and isorthonormal(R):
        return True
    return False
    
    
def isorthonormal(R):
    """
    Check if R is orthonormal
    
    Parameters
    ----------
    
    R : `sympy.matrices.dense.MutableDenseMatrix`
    
    Returns
    -------
    
    False or True
    
    """
    _,ncol = R.shape
    for i,j in combinations(range(ncol), 2):
        if ( R[:,i].dot(R[:,j]) ).simplify() != 0:
            return False
    for i in range(ncol):
        if R[:,i].norm().simplify() != 1:
            return False
    return True
    
    
def hcoords(v):
    return v.col_join(Matrix([1]))
    

def sympy_matrix_to_numpy_float(H):
    """
    Convert SymPy Matrix (numerical) to NumPy array
    
    Parameters
    ----------
    
    H : `sympy.matrices.dense.MutableDenseMatrix`
    
    Returns
    -------
    
    Hf : array
    
    """
    Hf = np.array(H).astype("float64")
    return Hf
    
    
def sympy2float(sympy_object):
    """
    Convert a SymPy object to float object
    """
    if isinstance(sympy_object, Matrix):
        float_object = sympy_matrix_to_numpy_float(sympy_object)
    else:
        float_object = sympy_object
    return float_object
    
    


if __name__=="__main__":
    H = Matrix([[1,1,0,5],[1,0,0,4],[0,0,1,0],[0,0,0,1]])
    print(ishtm(H))
