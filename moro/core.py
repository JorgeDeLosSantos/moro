"""
Numython R&D, (c) 2026 
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import warnings
import math

import sympy as sp
from sympy import (
    prod,
    symbols,
    Matrix,
    eye,
    diag,
    trigsimp,
    zeros,
    simplify,
    nsimplify,
    Eq,
    MatAdd,
    MatMul,
)
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices.matrixbase import MatrixBase
# Moro core dependencies
from sympy.core.function import AppliedUndef
from moro.transformations import dh
from moro.util import (
    vector_in_hcoords,
)
from moro.abc import t

__all__ = ["Robot"]

class Robot:
    """
    Define a robot-serial-arm given the Denavit-Hartenberg parameters 
    and the joint type, as tuples (or lists). Each tuple must have the form:

    `(a_i, alpha_i, d_i, theta_i)`

    Or including the joint type:

    `(a_i, alpha_i, d_i, theta_i, joint_type)`

    All parameters are `int` or `floats`, or a symbolic variable of SymPy. Numeric angles must be passed in radians. If `joint_type` is not passed, the joint is assumed to be revolute.

    Examples
    --------
    
    >>> rr = Robot((l1,0,0,q1), (l2,0,0,q2))

    or

    >>> rr2 = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
    """
    _CACHE_CATEGORIES = ("kinematics", "dynamics")

    def __init__(self,*args):
        if len(args) == 0:
            raise ValueError("Robot must be initialized with at least one DH parameter row.")

        self._Ts = [] # Transformation matrices i to i-1
        self._joint_types = [] # Joint type -> "r" revolute, "p" prismatic
        self._qs = [] # Joint variables
        self._qis_range = None # qis_range (set via its setter when required)
        self._dh_parameters = [] # Store the DH parameters 

        for row_idx, k in enumerate(args, start=1):
            if not isinstance(k, (list, tuple)):
                raise ValueError(f"DH parameter row {row_idx} must be a list or tuple.")
            if len(k) not in (4, 5):
                raise ValueError(
                    f"DH parameter row {row_idx} must have exactly 4 or 5 elements "
                    "(a, alpha, d, theta[, joint_type])."
                )
            self._Ts.append(dh(k[0],k[1],k[2],k[3])) # Compute Ti->i-1
            self._dh_parameters.append(tuple(k[:4])) # Store the DH parameters as they were passed in the constructor
            if len(k)>4:
                joint_type = str(k[4]).strip().lower()
                if joint_type not in ("r", "p"):
                    raise ValueError(
                        f"Invalid joint type '{k[4]}'. Use 'r' for revolute or 'p' for prismatic."
                    )
                self._joint_types.append(joint_type)
            else: # By default, the joint type is assumed to be revolute
                self._joint_types.append('r')

            if self._joint_types[-1] == "r":
                self._qs.append(k[3])
            else:
                self._qs.append(k[2]) 
        self._dof = len(args) # Degree of freedom

        # Dynamic parameters (initially set to None, but they can be set using the corresponding methods)
        self._masses = None
        self._inertia_tensors = None
        self._cm_positions = None
        self._gravity = None
        # Flags reporting whether dynamic quantities were explicitly defined (True)
        # or auto-generated with a documented default assumption (False).
        self._masses_explicit = False
        self._inertia_tensors_explicit = False
        self._cm_positions_explicit = False
        self._gravity_explicit = False
        self._joint_limits_explicit = False
        self._set_default_joint_limits() # set default joint-limits on create

        # Cache for kinematics and dynamics computations
        self._cache = {category: {} for category in self._CACHE_CATEGORIES}

    @property
    def Ts(self):
        return [self._copy_matrix(T) for T in self._Ts]

    @property
    def joint_types(self):
        return list(self._joint_types)

    @property
    def dh_parameters(self):
        return list(self._dh_parameters)
    
    @property
    def dh_table(self):
        """
        Return the DH parameter table as a SymPy TableForm.
        """
        rows = [["i", "a_i", "alpha_i", "d_i", "theta_i"]]

        for i, (a, alpha, d, theta) in enumerate(self.dh_parameters, start=1):
            rows.append([i, a, alpha, d, theta])

        return Matrix(rows)
    
    def z(self,i):
        """
        Get the z_i axis direction w.r.t. {0}-Frame.
        
        Parameters
        ----------
        i: int
            {i}-th Frame
            
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            The direction of z_i axis
        """
        return self._get_cached(
            "kinematics",
            f"z_{i}",
            lambda: self.T_i0(i)[:3,2]
        )
    
    def r_o(self,i):
        """
        Get the position (of the origin of coordinates) of the {i}-Frame w.r.t. {0}-Frame
        
        Parameters
        ----------
        i: int
            {i}-th Frame
            
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            The position of {i}-Frame as a 3-component vector.
        """
        return self._get_cached(
            "kinematics",
            f"r_o_{i}",
            lambda: self.T_i0(i)[:3,3]
        )
    
    @property
    def J(self):
        """
        Get the geometric jacobian matrix of the end-effector. 
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Get the geometric jacobian matrix of the end-effector.
        """
        # Jacobian of the end-effector (point located at the origin of {n}-Frame)
        return self._get_cached(
            "kinematics",
            "J",
            lambda: self.J_point([0,0,0], self.dof)
        )

    @property
    def dof(self):
        """
        Get the degrees of freedom of the robot.
        
        Returns
        -------
        int
            Degrees of freedom of the robot
        """
        return self._dof

    @property
    def T(self):
        """ 
        Get the homogeneous transformation matrix of {n}-Frame (end-effector)
        w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            :math:`T_n^0`
        """
        return self._get_cached(
            "kinematics",
            "T",
            lambda: self.T_i0(self.dof)
        )
        
    def T_ij(self,i,j):
        """
        Get the homogeneous transformation matrix of {i}-Frame w.r.t. {j}-Frame. 
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Returns :math:`T_i^j`
        """
        if not (0 <= i <= self.dof) or not (0 <= j <= self.dof):
            raise ValueError(f"i and j must be between 0 and {self.dof} inclusive.")
        
        if i == j: 
            return eye(4)
        if i < j:
            T = prod(self._Ts[i:j])
            R = T[:3, :3]
            p = T[:3, 3]
            return R.T.row_join(-R.T * p).col_join(Matrix([[0, 0, 0, 1]]))
        
        return simplify(prod(self._Ts[j:i]))

    def T_i0(self,i):
        """
        Get the homogeneous transformation matrix of {i}-Frame w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Returns :math:`T_i^0`
        """
        if i == 0:
            return eye(4)
        return self._get_cached(
            "kinematics",
            f"T_i0_{i}",
            lambda: self.T_ij(i, 0)
        )
    
        
    def R_i0(self,i):
        """
        Get the rotation matrix of {i}-Frame w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Returns :math:`R_i^0`
        """
        return self._get_cached(
            "kinematics",
            f"R_i0_{i}",
            lambda: self.T_i0(i)[:3,:3]
        )
    
    @property
    def qs(self):
        return list(self._qs)
    
    @property
    def qis_range(self):
        if self._qis_range is None:
            raise ValueError(
                "qis_range has not been set. Assign a value via the qis_range setter first."
            )
        return self._qis_range
        
    @qis_range.setter
    def qis_range(self, value):
        self._qis_range = value

    @property
    def masses(self):
        """
        Get the masses of the links as a list like: [m1, m2, ..., mn], where 
        m1, m2, ..., mn, are numeric or symbolic values.
        
        Returns
        -------
        list
            A list of numerical or symbolic values that correspond to link masses.
        """
        if self._masses is None:
            raise ValueError("Link masses are not defined. Please set them using "
                             "the masses setter.")
        return list(self._masses)

    @masses.setter
    def masses(self,masses):
        """
        Set mass for each link using a list like: [m1, m2, ..., mn], where 
        m1, m2, ..., mn, are numeric or symbolic values.
        
        Parameters
        ----------
        masses: list, tuple
            A list of numerical or symbolic values that correspond to link masses.
        """
        if masses is None:
            masses = [ symbols(f"m_{i+1}") for i in range(self.dof) ]
            self._masses_explicit = False
        else:
            self._masses_explicit = True

        if len(masses) != self.dof:
            raise ValueError(f"Number of masses must be equal to the number of links ({self.dof}).")
        else:
            self._masses = list(masses)

        self._invalidate_dynamics_cache() # Invalidate dynamics cache since link masses affect the inertia matrix and potential energy

    def _as_column_vector3(self, value, name):
        """
        Convert a 3-component vector-like object to a defensive SymPy Matrix(3, 1).
        Row matrices with exactly three components are normalized to columns.
        """
        try:
            matrix = Matrix(value)
        except Exception as exc:
            raise ValueError(f"{name} must represent a three-dimensional vector.") from exc

        if matrix.shape == (3, 1):
            return Matrix(matrix)
        if matrix.shape == (1, 3):
            return Matrix(matrix.T)
        if len(matrix) == 3 and (matrix.rows == 3 or matrix.cols == 3):
            return Matrix(list(matrix)).reshape(3, 1)
        raise ValueError(f"{name} must have exactly three components.")

    def _copy_matrix(self, matrix):
        return Matrix(matrix)

    @property
    def inertia_tensors(self):
        """
        Get the inertia tensors of the links as a list like: [I1, I2, ..., In], where 
        I1, I2, ..., In, are 3x3 sympy matrices that correspond to the inertia tensor 
        of each link w.r.t. a frame located in its center of mass and aligned with the {i}-Frame.
        
        Returns
        -------

        list
            A list of 3x3 sympy matrices that correspond to the inertia tensor 
            of each link w.r.t. a frame located in its center of mass 
            and aligned with the {i}-Frame.
        """
        if self._inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Please set them using the inertia_tensors setter.")
        return [self._copy_matrix(tensor) for tensor in self._inertia_tensors]
    
    @inertia_tensors.setter
    def inertia_tensors(self,tensors):
        """
        Inertia tensor w.r.t. {i}'-Frame. Consider that the reference 
        frame {i}' is located at the center of mass of link [i] 
        and oriented in the same way as {i}-Frame. By default (if `tensors` argument
        is not passed), it is assumed that each link is symmetrical to, 
        at least, two planes of the reference frame located in its center of mass, 
        then the inertia tensor of each link is defined as a diagonal matrix with 
        the moments of inertia as diagonal elements, and the products of inertia as zero. 
        The moments of inertia are defined as symbolic variables of the form: 
        I_{x_ix_i}, I_{y_iy_i}, I_{z_iz_i}, where i is the link number.
        
        Parameters
        ----------
        tensors: sympy.matrices.dense.MutableDenseMatrix
            A list containinig `sympy.matrices.dense.MutableDenseMatrix` that 
            corresponds to each inertia tensor w.r.t. {i}'-Frame.

        Notes
        -----
        If ``tensors`` is None, diagonal inertia tensors are auto-generated as
        symbolic variables (assuming zero products of inertia, i.e. symmetric
        links). This is a modeling assumption; override it by providing explicit
        tensors when your links are not symmetric.
        """
        if tensors is None:
            self._generate_diagonal_inertia_tensors()
            self._inertia_tensors_explicit = False
        elif len(tensors) != self.dof:
            raise ValueError(f"Number of inertia tensors must be equal to the number of links ({self.dof}).")
        else:
            normalized_tensors = []
            for idx, tensor in enumerate(tensors, start=1):
                try:
                    tensor_matrix = Matrix(tensor)
                except Exception as exc:
                    raise ValueError(f"Inertia tensor for link {idx} must be convertible to a 3x3 Matrix.") from exc
                if tensor_matrix.shape != (3, 3):
                    raise ValueError(
                        f"Inertia tensor for link {idx} must be a 3x3 matrix; "
                        f"got shape {tensor_matrix.shape}."
                    )
                normalized_tensors.append(Matrix(tensor_matrix))
            self._inertia_tensors = normalized_tensors
            self._inertia_tensors_explicit = True

        self._invalidate_dynamics_cache() # Invalidate dynamics cache since inertia tensors affect the inertia matrix and Coriolis matrix

    def _generate_diagonal_inertia_tensors(self):
        """
        Generate diagonal inertia tensors for each link and store them in
        ``self._inertia_tensors``. Internal helper: diagonal tensors are a
        modeling assumption (zero products of inertia) for symmetric links.
        """
        inertia_tensors = []
        for k in range(self.dof):
            Istr = f"I_{{x_{k+1}x_{k+1}}}, I_{{y_{k+1}y_{k+1}}}, I_{{z_{k+1}z_{k+1}}}"
            Ix, Iy, Iz = symbols(Istr)
            inertia_tensors.append(diag(Ix, Iy, Iz))
        self._inertia_tensors = inertia_tensors

    @property
    def cm_positions(self):
        """
        Get the positions of the center of mass for each link. The position of the 
        center of mass of the i-th link is defined as a list or tuple of three elements 
        that correspond to the x, y, z coordinates of the center of mass w.r.t. {i}-Frame.
        
        Returns
        -------
        list
            A list of lists (or tuples) or a tuple of tuples (or lists) containing 
            each center of mass position w.r.t. its reference frame.
        """
        if self._cm_positions is None:
            raise ValueError("Center of mass locations are not defined. Please set them using the cm_positions setter.")
        return [self._copy_matrix(position) for position in self._cm_positions]
    
    @cm_positions.setter
    def cm_positions(self,positions):
        """
        Set the positions of the center of mass for each 
        link. The position of the center of mass of the i-th link must be 
        defined as a list or tuple of three elements that correspond to the x, y, z 
        coordinates of the center of mass w.r.t. {i}-Frame.
    
        Parameters
        ----------
        positions: list, tuple
            A list of lists (or tuples) or a tuple of tuples (or lists) containing 
            each center of mass position w.r.t. its reference frame.
        
        Examples
        --------
        >>> RR = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
        >>> RR.cm_positions = [(-lc1,0,0), (-lc2,0,0)]
        """
        if len(positions) != self.dof:
            raise ValueError(f"Number of center of mass locations must be equal to the number of links ({self.dof}).")
        
        self._cm_positions = [
            self._as_column_vector3(cm, f"Center of mass location for link {idx+1}")
            for idx, cm in enumerate(positions)
        ]
        self._cm_positions_explicit = True
        self._invalidate_kinematics_cache() # CoM affects kinematics-cached quantities (r_cm, J_cm)
        # Invalidate dynamics cache since CoM locations affect the inertia matrix and potential energy
        self._invalidate_dynamics_cache() 

    @property
    def gravity(self):
        """
        Get the gravity acceleration defined in the base frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Gravity vector defined in the base frame.
        """
        if self._gravity is None:
            raise ValueError("Gravity acceleration is not defined. Please set it using the gravity setter.")
        return self._copy_matrix(self._gravity)

    @gravity.setter
    def gravity(self,g):
        """
        Set the gravity acceleration in the base frame. 
        
        Parameters
        ----------
        g: list, tuple
            A list or tuple of three elements that define 
            the gravity acceleration in the base frame.
        
        Examples
        --------
        >>> RR = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
        >>> RR.gravity = (0, -g, 0)
        """
        self._gravity = self._as_column_vector3(g, "Gravity acceleration")
        self._gravity_explicit = True
        self._invalidate_dynamics_cache() # Invalidate dynamics cache since gravity vector affects potential energy and gravity torque vector

    def _r_cm_i(self,i):
        """
        Return the position of the center of mass of the i-th link w.r.t. {i}-Frame.
        
        Parameters
        ----------
        i: int
            Link number
        
        Returns
        -------
        `sympy.matrices.dense.MutableDenseMatrix`
            A column vector :math:`\\mathbf{r}_{G_i}^i`
        """
        self._check_index(i, name="link") 
        if self._cm_positions is None:
            raise ValueError("Center of mass locations are not defined. " \
                             "Please set them using the cm_positions setter.")
        
        return self._cm_positions[i-1]

    
    def r_cm(self,i):
        """
        Return the position of the center of mass of the 
        i-th link w.r.t. the base frame.
        
        Parameters
        ----------
        i: int
            Link number
        
        Returns
        -------
        `sympy.matrices.dense.MutableDenseMatrix`
            A column vector
        """
        return self._get_cached(
            "kinematics",
            f"r_cm_{i}",
            lambda: self._compute_r_cm(i)
        )
    
    def _compute_r_cm(self,i):
        """
        Internal method to compute the position of the center of mass of the i-th link w.r.t. the base frame. This method is called by r_cm() and its result is cached for future calls.
        """
        self._check_index(i, name="link") 
        if self._cm_positions is None:
            raise ValueError("Center of mass locations are not defined. " \
            "Please set them using the cm_positions setter.")  

        r_cm_i = self._r_cm_i(i) # vector r_{G_i}^i
        r_cm = ( self.T_i0(i) * vector_in_hcoords( r_cm_i ) )[:3,:]
        return simplify( r_cm )
        
    def v_cm(self,i):
        """
        Return the velocity of the center of mass of the 
        i-th link w.r.t. the base frame.
        
        Parameters
        ----------
        i: int
            Link number
        
        Returns
        -------
        `sympy.matrices.dense.MutableDenseMatrix`
            A column vector 
        """
        self._check_index(i)
        self._warn_static_joint_variables("v_cm")
        rcm_i = self.r_cm(i)
        vcm_i = rcm_i.diff(t)
        return simplify( vcm_i )
    
    def _J_cm_i(self,i):
        """
        Geometric Jacobian matrix of the center of mass of the i-th link.

        Parameters
        ----------
        i : int
            Link number.
        """
        return self.J_point(self._r_cm_i(i), i)
    
    def Jv_cm_i(self,i):
        """
        Return the linear velocity Jacobian matrix of the center of mass of the i-th link.

        Parameters
        ----------
        i : int
            Link number.
        """
        self._check_index(i)
        return self._get_cached(
            "kinematics",
            f"Jv_cm_{i}",
            lambda: self._J_cm_i(i)[:3,:]
        )
    
    def Jw_cm_i(self,i):
        """
        Return the angular velocity Jacobian matrix of the center of mass of the i-th link.

        Parameters
        ----------
        i : int
            Link number.
        """
        self._check_index(i)
        return self._get_cached(
            "kinematics",
            f"Jw_cm_{i}",
            lambda: self._J_cm_i(i)[3:,:]
        )
    
    def J_cm_i(self,i):
        """
        Compute the jacobian matrix of the center of mass of 
        the i-th link.
        
        Parameters
        ----------
        i : int
            Link number.
            
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Jacobian matrix of i-th CoM.     
        """
        self._check_index(i)
        return self._get_cached(
            "kinematics",
            f"J_cm_{i}",
            lambda: self._J_cm_i(i)
        )
    
    def J_point(self,point,i):
        """
        Compute the jacobian matrix of a specific point in the manipulator.
        
        Parameters
        ----------
        point : list 
            Coordinates of the point w.r.t. {i}-Frame. 

        i : int
            Link number in which the point is located.
            
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Jacobian matrix of the point.
        
        """
        self._check_index(i)
        point_wrt_i = self._as_column_vector3(point, "Point")
        point_wrt_0 = ( self.T_i0(i) * vector_in_hcoords( point_wrt_i ) )[:3,:]
        
        n = self.dof
        M_ = zeros(6,n)
        for j in range(1, n+1):
            idx = j - 1
            if j <= i:
                if self._joint_types[idx]=='r':
                    jp = self.z(j-1).cross(point_wrt_0 - self.r_o(j-1))
                    jo = self.z(j-1)
                else:
                    jp = self.z(j-1)
                    jo = zeros(3,1)
            else:
                jp = zeros(3,1)
                jo = zeros(3,1)
            jp = jp.col_join(jo)
            M_[:,idx] = jp
        return simplify(M_)
    
    def joint_type(self,i):
        """
        Return the type of the i-th joint. "r" for revolute, "p" for prismatic.
        
        Parameters
        ----------
        i : int
            Joint number.
        """
        self._check_index(i, name="joint")
        return self._joint_types[i-1]
    
    def q(self,i):
        """
        Return the i-th joint variable.
        
        Parameters
        ----------
        i : int
            Joint number.
        """
        self._check_index(i, name="joint")
        return self.qs[i-1]
    
    def q_dot(self,i):
        """
        Return the time derivative of the i-th joint variable.
        
        Parameters
        ----------
        i : int
            Joint number.
        """
        self._check_index(i, name="joint")
        return self.q(i).diff(t)
    
    def w_rel0(self,i):
        """
        Return the angular velocity of the [i]-link w.r.t. [i-1]-link, 
        described in {0}-Frame.
        
        Since we are using Denavit-Hartenberg frames, then:
        
        .. math:: 
            
            \\omega_{{i-i,i}} = \\dot{{q}}_i \\mathbf{z}_{i-1}
            
        If the i-th joint is revolute, or:
        
        .. math:: 
            
            \\omega_{{i-i,i}} = \\mathbf{0}
        
        If the i-th joint is a prismatic.
        
        Parameters
        ----------
        i : int
            Link number.
        """
        if self.joint_type(i) == "r":
            w_rel0 = self.z(i-1)*self.q_dot(i)
        else:
            w_rel0 = zeros(3,1)
        return w_rel0
    
    def w(self,i):
        """
        Compute the angular velocity of the [i]-link w.r.t. base {0}-Frame. 
        The angular velocity of the [i]-link w.r.t. base {0}-Frame can be 
        computed as the sum of the relative angular velocities of each link 
        w.r.t. its previous link, described in the base frame: 

        .. math::

            \\boldsymbol{\\omega}_i = \\sum_{{k=1}}^i \\boldsymbol{\\omega}_{{k-1,k}}
        
        Parameters
        ----------
        i: int 
            Link number.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Angular velocity of the [i]-link w.r.t. {0}-Frame.
        """
        self._check_index(i)
        self._warn_static_joint_variables("w")
        return self._get_cached(
            "kinematics",
            f"w_{i}",
            lambda: self._w(i)
        )
    
    def _w(self,i):
        """
        Internal method to compute the angular velocity of the [i]-link w.r.t. base {0}-Frame. This method is called by w() and its result is cached for future calls.
        """
        wi = Matrix([0,0,0])
        for k in range(1,i+1):
            wi += self.w_rel0(k)
        return wi
    
    
    def I_cm0(self,i):
        """
        Return the inertia tensor of [i-th] link w.r.t. a frame 
        located in the center of mass of link [i] and aligned with the base frame.
        
        Parameters
        ----------
        i: int 
            Link number.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Inertia tensor of the [i]-link w.r.t. {0}-Frame.
        """
        self._check_index(i)
        return self._get_cached(
            "dynamics",
            f"I_cm0_{i}",
            lambda: self._I_cm0(i)
        )

    def _I_cm0(self,i):
        """
        Internal method to compute the inertia tensor of [i-th] link w.r.t. a frame 
        located in the center of mass of link [i] and aligned with the base frame. 
        This method is called by I_cm0() and its result is cached for future calls.
        """
        if self._inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Please set them using the " \
            "inertia_tensors setter.")

        if i == 0:
            raise ValueError("i must be greater than 0")
        idx = i - 1
        Iii = self._inertia_tensors[idx]
        Ii = simplify( self.R_i0(i) * Iii * self.R_i0(i).T )
        return Ii
    
    def I_cm(self,i):
        """
        Return the inertia tensor of i-th link w.r.t. {i}' frame 
        (located in the center of mass of link [i] and aligned with 
        the {i}-Frame).
        
        Parameters
        ----------
        i: int 
            Link number.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Inertia tensor of the [i]-link w.r.t. {i}'-Frame.
        """
        self._check_index(i)
        if self._inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Please set them using the " \
            "inertia_tensors setter.") 
        
        idx = i - 1
        I_cm = self._copy_matrix(self._inertia_tensors[idx])
        return I_cm
    
    def m(self,i):
        """
        Return the mass of the i-th link. 

        Parameters
        ----------

        i: int  
            Link number.
        """
        if self._masses is None:
            raise ValueError("Link masses are not defined. Please set them using " \
            "the masses setter.")
        self._check_index(i, name="link")
        return self._masses[i-1]
        
    def inertia_matrix(self):
        """
        Return the inertia matrix M(q) of the robot. The inertia matrix is computed as:

        .. math::
            M(q) = \\sum_{{i=1}}^n m_i J_{v_i}^T J_{v_i} + J_{w_i}^T R_i I_i R_i^T J_{w_i}

        where :math:`m_i` is the mass of the i-th link, :math:`J_{v_i}` is the linear velocity Jacobian matrix of the center of mass of the i-th link, :math:`J_{w_i}` is the angular velocity Jacobian matrix of the center of mass of the i-th link, :math:`R_i` is the rotation matrix of the i-th link w.r.t. the base frame, and :math:`I_i` is the inertia tensor of the i-th link w.r.t. a frame located in its center of mass and aligned with the {i}-Frame.

        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Inertia matrix M(q)
        """
        return self._get_cached(
            "dynamics",
            "inertia_matrix",
            lambda: self._compute_inertia_matrix()
        )
    
    def _compute_inertia_matrix(self):
        """
        Internal method to compute the inertia matrix. This method is called by 
        inertia_matrix() and its result is cached for future calls.
        """
        if self._masses is None:
            raise ValueError("Link masses are not defined. Use masses setter.")
        if self._inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Use inertia_tensors setter.")
        if self._cm_positions is None:
            raise ValueError("Center of mass locations are not defined. Use cm_positions setter.")

        n = self.dof
        M = zeros(n)

        # Precompute Jacobians, Rotations, Inertia tensors, Masses
        Jv = [self.Jv_cm_i(i+1) for i in range(n)]
        Jw = [self.Jw_cm_i(i+1) for i in range(n)]
        R  = [self.R_i0(i+1)    for i in range(n)]
        I  = [self._inertia_tensors[i] for i in range(n)]
        m  = [self.m(i+1)     for i in range(n)]

        # Compute inertia matrix
        for i in range(n):
            M += m[i] * Jv[i].T * Jv[i]
            M += Jw[i].T * R[i] * I[i] * R[i].T * Jw[i]

        return trigsimp(M)


    def coriolis_matrix(self):
        """
        Return the Coriolis matrix C(q,q').
        The Coriolis matrix is computed using the Christoffel symbols of the first kind:

        .. math::
        
            C_{{i,j}} = \\sum_{{k=1}}^n c_{{i,j,k}} \\dot{{q}}_k
            
        """
        self._warn_static_joint_variables("coriolis_matrix")
        n = self.dof
        M = self.inertia_matrix()
        C = zeros(n)
        for i in range(1,n+1):
            for j in range(1,n+1):
                C[i-1,j-1] = 0
                for k in range(1,n+1):
                    C[i-1,j-1] += self.christoffel_symbols(i,j,k,M) * self.q_dot(k)
        return nsimplify(C)
        
    def christoffel_symbols(self,i,j,k,M):
        """
        Return the Christoffel symbol of the first kind:

        .. math::
            c_{{i,j,k}} = \\frac{1}{2} \\left( \\frac{{\\partial M_{{i,j}}}}{{\\partial q_k}} + \\frac{{\\partial M_{{i,k}}}}{{\\partial q_j}} - \\frac{{\\partial M_{{j,k}}}}{{\\partial q_i}} \\right)

        """
        q = self.qs
        idx_i, idx_j, idx_k = i-1, j-1, k-1 
        mij = M[idx_i, idx_j]
        mik = M[idx_i, idx_k]
        mjk = M[idx_j, idx_k]
        cijk = sp.Rational(1, 2)*( mij.diff(q[idx_k]) + mik.diff(q[idx_j]) - mjk.diff(q[idx_i]) )
        return cijk
    
    def gravity_vector(self):
        """
        Compute the gravity torque vector G(q). The gravity torque vector is computed 
        as the gradient of the potential energy of the system:

        .. math::
            G(q) = \\nabla U(q) = \\left[ \\frac{{\\partial U}}{{\\partial q_1}}, \\frac{{\\partial U}}{{\\partial q_2}}, ..., \\frac{{\\partial U}}{{\\partial q_n}} \\right]^T

        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Gravity torque vector G(q)
        """
        pot = self.potential_energy()
        gv = [nsimplify(pot.diff(k)) for k in self.qs]
        return Matrix(gv)
    
    def dynamic_model_matrix_form(self):
        """
        Return the dynamic model of the robot in matrix form:

        .. math::
            M(q) \\ddot{{q}} + C(q,\\dot{{q}}) \\dot{{q}} + G(q) = \\tau

        where :math:`M(q)` is the inertia matrix, :math:`C(q,q')` is the Coriolis matrix, 
        :math:`G(q)` is the gravity torque vector, and :math:`\\tau` is the vector of joint torques.

        """
        M = self.inertia_matrix()
        C = self.coriolis_matrix()
        G = self.gravity_vector()
        qdd = Matrix([q.diff(t,2) for q in self.qs])
        qd = Matrix([q.diff(t) for q in self.qs])
        tau = Matrix([ symbols(f"tau_{i+1}") for i in range(self.dof)])
        return Eq(MatAdd( MatMul(M,qdd), MatMul(C,qd),  G) , tau)
            
    def link_kinetic_energy(self,i):
        """
        Returns the kinetic energy of i-th link.

        .. math::
        
            K_i = \\frac{1}{2} m_i \\mathbf{v}_{G_i}^T \\mathbf{v}_{G_i} + \\frac{1}{2} \\boldsymbol{\\omega}_i^T I_i \\boldsymbol{\\omega}_i

        Parameters
        ----------
        i: int
            Link number.
        """
        self._check_index(i)
        mi = self.m(i)
        vi = self.v_cm(i)
        wi = self.w(i)
        I_cmi = self.I_cm(i)
        Ri = self.R_i0(i)
        
        half = sp.Rational(1, 2)
        Ktra_i = half * mi * vi.T * vi
        Krot_i = half * wi.T * Ri * I_cmi * Ri.T * wi
        Ki = Ktra_i + Krot_i
        return Ki

        
    def link_potential_energy(self,i):
        """
        Returns the potential energy of the [i-th] link.
        
        .. math::
        
            P_i = - m_i \\mathbf{g}^T \\mathbf{r}_{G_i} 
        
        Parameters
        ----------
        i: int
            Link number.
            
        Returns
        -------
        
        """
        self._check_index(i)
        if self._gravity is None:
            raise ValueError("Gravity acceleration is not defined. Please set it using " \
            "the gravity property.") 
        
        return - self.m(i) * self._gravity.T * self.r_cm(i)
        
    def kinetic_energy(self):
        """
        Returns the total kinetic energy of the robot
        """
        K = Matrix([0])
        for i in range(self.dof):
            K += self.link_kinetic_energy(i+1)
        return nsimplify(K)
        
    def potential_energy(self):
        """
        Returns the total potential energy of the robot:

        .. math::
            P(q) = \\sum_{{i=1}}^n P_i = - \\sum_{{i=1}}^n m_i \\mathbf{g}^T \\mathbf{r}_{G_i}

        """
        P = Matrix([0])
        for i in range(self.dof):
            P += self.link_potential_energy(i+1) 
        return nsimplify(P)
        
    def lagrangian(self):
        """
        Returns the Lagrangian of the system, defined as :math:`\\mathcal{L} = \\mathcal{K} - \\mathcal{P}`, where :math:`\\mathcal{K}` is the kinetic energy and :math:`\\mathcal{P}` is the potential energy.
        """
        K = self.kinetic_energy()
        P = self.potential_energy()
        L = K - P
        return nsimplify(L)[0]

    def dynamic_model(self):
        """
        Returns the dynamic model of the robot 
        using the Euler-Lagrange formulation. The returned value is a list of equations,
        one for each joint, of the form:

        .. math::   
            \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{{q}}_i} \\right) - \\frac{\\partial L}{\\partial q_i} = \\tau_i
        
        where :math:`\\mathcal{L}` is the Lagrangian of the system, defined as :math:`\\mathcal{L} = \\mathcal{K} - \\mathcal{P}`, where :math:`\\mathcal{K}` is the kinetic energy and :math:`\\mathcal{P}` is the potential energy.
        """
        self._warn_static_joint_variables("dynamic_model")
        L = self.lagrangian()
        equations = []
        for i in range(self.dof):
            q = self.qs[i]
            qp = self.qs[i].diff(t)
            dL_dqp = 0 if qp == 0 else L.diff(qp)
            equations.append( Eq( trigsimp(sp.diff(dL_dqp, t) - L.diff(q) ), symbols(f"tau_{i+1}") ) ) 
            
        return equations
    
    
    def _set_default_joint_limits(self):
        joint_limits = []
        for k in range(self.dof):
            if self._joint_types[k] == "r":  # for revolute joint
                lower_value = -sp.pi # -180°
                upper_value = sp.pi  # 180°
            else: # for prismatic joint
                lower_value = 0     # 
                upper_value = 1000  #
            joint_limits.append((lower_value, upper_value))
        self._joint_limits = joint_limits
        
    @property
    def joint_limits(self):
        """
        Get the joint limits of the robot. The joint limits are returned as a list of tuples, 
        where each tuple contains the lower and upper limits for the corresponding joint. 
        For revolute joints, the default limits are (-pi, pi) radians, and for prismatic joints, 
        the default limits are (0, 1000) units. If you want to set custom joint limits, 
        you can use the joint_limits setter.
        """
        return list(self._joint_limits)
    
    @joint_limits.setter
    def joint_limits(self,limits):
        """
        Set the joint limits of the robot. The joint limits should be provided 
        as a list of tuples, where each tuple contains the lower and upper limits 
        for the corresponding joint. For revolute joints, the limits should be in radians, 
        and for prismatic joints, the limits should be in the appropriate linear units.
        """
        if len(limits) != self.dof:
            raise ValueError("The number of joint limits must match DOF.")
        validated_limits = []
        for idx, limit in enumerate(limits, start=1):
            try:
                if len(limit) != 2:
                    raise ValueError
            except TypeError as exc:
                raise ValueError(f"Joint limit for joint {idx} must be a 2-tuple (lower, upper).") from exc
            except ValueError:
                raise ValueError(f"Joint limit for joint {idx} must be a 2-tuple (lower, upper).")

            lower, upper = limit
            try:
                lower_num = float(lower)
                upper_num = float(upper)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Joint limit for joint {idx} must contain numeric values.") from exc

            if math.isnan(lower_num) or math.isnan(upper_num):
                raise ValueError(f"Joint limit for joint {idx} cannot contain NaN.")
            if math.isinf(lower_num) or math.isinf(upper_num):
                raise ValueError(f"Joint limit for joint {idx} cannot contain infinite values.")
            if lower_num > upper_num:
                raise ValueError(f"Joint limit for joint {idx} must satisfy lower <= upper.")
            validated_limits.append((lower, upper))

        self._joint_limits = validated_limits
        self._joint_limits_explicit = True
    
    @property
    def _numerical_joint_limits(self):
        joint_limits = self.joint_limits 
        joint_limits_num = [(float(a), float(b)) for (a,b) in joint_limits] 
        return joint_limits_num
    
    def __str__(self):
        robot_type = "".join( self._joint_types ).upper()
        return f"Robot {robot_type}"
    
    def __repr__(self):
        robot_type = "".join( self._joint_types ).upper()
        return f"Robot {robot_type}"

    def model_summary(self):
        """
        Return a readable summary of the robot's modeling state.

        For each dynamic quantity it reports whether it is explicitly
        defined, assumed by default (auto-generated symbolic placeholder),
        or not set. Assumed values correspond to documented modeling
        assumptions (e.g. diagonal inertia tensors implying zero products
        of inertia) that should be overridden when they do not hold.

        Returns
        -------
        str
            A multi-line summary.
        """
        def _label(explicit, defined, assumed_text):
            if not defined:
                return "NOT SET"
            return "explicit" if explicit else f"assumed ({assumed_text})"

        robot_type = "".join(self._joint_types).upper()
        lines = [
            f"Model summary | Robot {robot_type} | DOF = {self.dof}",
            "  joint_limits     : " + ("custom" if self._joint_limits_explicit else "default"),
            "  masses           : " + _label(self._masses_explicit, self._masses is not None, "symbolic m_i"),
            "  inertia_tensors  : " + _label(self._inertia_tensors_explicit, self._inertia_tensors is not None, "diagonal symbolic"),
            "  cm_positions     : " + _label(self._cm_positions_explicit, self._cm_positions is not None, "-"),
            "  gravity          : " + _label(self._gravity_explicit, self._gravity is not None, "-"),
        ]
        return "\n".join(lines)

    # def _repr_latex_(self):
    #     return sp.latex(self.dh_table)
    
    def _has_static_joint_variables(self):
        'Return True if any joint variable is a static symbol (not a function of time).'
        return any(not isinstance(q, AppliedUndef) for q in self._qs)

    def _warn_static_joint_variables(self, operation):
        'Warn when a velocity-dependent operation is used with static joint variables.'
        if self._has_static_joint_variables():
            warnings.warn(
                f"{operation}() requires time-dependent joint variables, but at "
                "least one joint variable is a static (non-time) symbol; "
                "velocity-dependent results may be incorrect. Use dynamicsymbols "
                "(e.g. moro.abc.q1..q6) for dynamic analyses.",
                UserWarning,
                stacklevel=3,
            )

    def _check_index(self, i, name="Link"):
        """
        Check if the index i is a valid link index. If not, raise an appropriate error.
        """
        if not isinstance(i, int):
            raise TypeError(f"{name} index must be an integer, got {type(i)}")
        if i < 1 or i > self.dof:
            raise IndexError(f"{name} index {i} out of range. Valid range is 1 to {self.dof}.")
        
    def _invalidate_kinematics_cache(self):
        """
        Invalidate kinematics and dynamics cache when joint variables or DH parameters are updated.
        """
        self._cache["kinematics"] = {}
        self._cache["dynamics"] = {}  

    def _invalidate_dynamics_cache(self):
        """
        Invalidate dynamics cache when masses, inertia tensors, or gravity vector are updated
        """
        self._cache["dynamics"] = {} 

    def _get_cached(self, category, key, compute_fn):
        """
        Get a cached value for a given category and key. If the value is not in the cache, 
        compute it using the provided function, store it in the cache, and return it.
        
        Parameters
        ----------
        category : str
            The category of the cache (e.g., "kinematics", "dynamics").
        key : str
            The key that identifies the specific value within the category (e.g., "T_i0_1", "inertia_matrix").
        compute_fn : callable
            A function that computes the value if it's not already cached. This function 
            should take no arguments and return the computed value.
        
        Returns
        -------
        The cached or computed value corresponding to the given category and key.
        
        Raises
        ------
        ValueError
            If the provided category is not a valid cache category.
        """
        if category not in self._CACHE_CATEGORIES:
            raise ValueError(
                f"Invalid cache category: '{category}'. "
                f"Valid categories are: {self._CACHE_CATEGORIES}."
            )
        
        if key not in self._cache[category]:
            self._cache[category][key] = compute_fn()
        
        return self._copy_cached_value(self._cache[category][key])

    def _copy_cached_value(self, value):
        if isinstance(value, MatrixBase) and not isinstance(value, ImmutableMatrix):
            return Matrix(value)
        return value



if __name__=="__main__":
    pass