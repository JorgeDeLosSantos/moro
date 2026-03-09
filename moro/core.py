"""
Numython R&D, (c) 2026 
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import matplotlib.pyplot as plt
import sympy as sp
from sympy import (
    pi,
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
# Moro core dependencies
from moro.transformations import dh, htmrot, htmtra
from moro.util import (
    vector_in_hcoords,
    is_position_vector,
    is_SE3,
)
from moro.abc import t

__all__ = ["Robot", "RigidBody2D"]

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
        self.Ts = [] # Transformation matrices i to i-1
        self.joint_types = [] # Joint type -> "r" revolute, "p" prismatic
        self._qs = [] # Joint variables
        self._dh_parameters = [] # Store the DH parameters 

        for k in args:
            self.Ts.append(dh(k[0],k[1],k[2],k[3])) # Compute Ti->i-1
            self._dh_parameters.append(k[:4]) # Store the DH parameters as they were passed in the constructor
            if len(k)>4:
                self.joint_types.append(k[4])
            else: # By default, the joint type is assumed to be revolute
                self.joint_types.append('r')

            if self.joint_types[-1] == "r":
                self._qs.append(k[3])
            else:
                self._qs.append(k[2]) 
        self._dof = len(args) # Degree of freedom

        # Dynamic parameters (initially set to None, but they can be set using the corresponding methods)
        self._masses = None
        self._inertia_tensors = None
        self._cm_positions = None
        self._gravity = None
        self._set_default_joint_limits() # set default joint-limits on create

        # Cache for kinematics and dynamics computations
        self._cache = {category: {} for category in self._CACHE_CATEGORIES}

    @property
    def dh_parameters(self):
        return self._dh_parameters
    
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
        return self.T_i0(i)[:3,3]
    
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
        return self.J_point([0,0,0], self.dof) 

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
        return simplify(prod(self.Ts))
        
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
            return simplify(prod(self.Ts[i:j]).inv())
        
        return simplify(prod(self.Ts[j:i]))

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
        return self._qs
    
    @property
    def qis_range(self):
        return self._qis_range
        
    @qis_range.setter
    def qis_range(self, *args):
        self._qis_range = args

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
                             "the set_masses() method.")
        return self._masses

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

        if len(masses) != self.dof:
            raise ValueError(f"Number of masses must be equal to the number of links ({self.dof}).")
        else:
            self._masses = masses

        self._invalidate_dynamics_cache() # Invalidate dynamics cache since link masses affect the inertia matrix and potential energy

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
        return self._inertia_tensors
    
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
        """
        if tensors is not None and len(tensors) != self.dof:
            raise ValueError(f"Number of inertia tensors must be equal to the number of links ({self.dof}).")

        dof = self.dof
        self._inertia_tensors = []
        for k in range(dof):
            self._inertia_tensors.append( tensors[k] )
        
        self._invalidate_dynamics_cache() # Invalidate dynamics cache since inertia tensors affect the inertia matrix and Coriolis matrix

    def generate_diagonal_inertia_tensors(self):
        """
        Generate diagonal inertia tensors for each link.
        """
        inertia_tensors = []
        for k in range(self.dof):
            Istr = f"I_{{x_{k+1}x_{k+1}}}, I_{{y_{k+1}y_{k+1}}} I_{{z_{k+1}z_{k+1}}}"
            Ix, Iy, Iz = symbols(Istr)
            inertia_tensors.append(diag(Ix, Iy, Iz))
        return inertia_tensors

    @property
    def cm_positions(self):
        """
        Get the positions of the center of mass for each link. The position of the center of mass of the i-th link is defined as a list or tuple of three elements that correspond to the x, y, z coordinates of the center of mass w.r.t. {i}-Frame.
        
        Returns
        -------
        list
            A list of lists (or tuples) or a tuple of tuples (or lists) containing 
            each center of mass position w.r.t. its reference frame.
        """
        if self._cm_positions is None:
            raise ValueError("Center of mass locations are not defined. Please set them using the cm_positions setter.")
        return self._cm_positions
    
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
        
        # Validate that each center of mass location is a 3-element list or tuple
        for idx, cm in enumerate(positions):
            if not is_position_vector(cm):
                raise ValueError(f"Center of mass location for link {idx+1} must be a list or tuple of three elements (x, y, z).")
        
        # Convert each center of mass location to a sympy Matrix if it's a list or tuple
        for idx, cm in enumerate(positions):
            if not isinstance(cm, Matrix):
                positions[idx] = Matrix(cm)

        self._cm_positions = positions
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
        return self._gravity

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
        if len(g) != 3:
            raise ValueError("Gravity acceleration must have three components (x, y, z).")
        
        # Convert g to a sympy Matrix if it's a list or tuple
        if not isinstance(g, Matrix):
            g = Matrix(g)

        self._gravity = g
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
        if self.cm_positions is None:
            raise ValueError("Center of mass locations are not defined. " \
                             "Please set them using the cm_positions setter.")
        
        return self.cm_positions[i-1]

    
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
        if self.cm_positions is None:
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
        idx = i - 1
        point_wrt_i = Matrix( point )
        point_wrt_0 = ( self.T_i0(i) * vector_in_hcoords( point_wrt_i ) )[:3,:]
        
        n = self.dof
        M_ = zeros(6,n)
        for j in range(1, n+1):
            idx = j - 1
            if j <= i:
                if self.joint_types[idx]=='r':
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
        return self.joint_types[i-1]
    
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
        if self.inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Please set them using the " \
            "inertia_tensors setter.")

        if i == 0:
            raise ValueError("i must be greater than 0")
        idx = i - 1
        Iii = self.inertia_tensors[idx]
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
        if self.inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Please set them using the " \
            "inertia_tensors setter.") 
        
        idx = i - 1
        I_cm = self.inertia_tensors[idx]
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
            "the set_masses() method.")
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
        if self.inertia_tensors is None:
            raise ValueError("Inertia tensors are not defined. Use inertia_tensors setter.")
        if self.cm_positions is None:
            raise ValueError("Center of mass locations are not defined. Use cm_positions setter.")

        n = self.dof
        M = zeros(n)

        # Precompute Jacobians, Rotations, Inertia tensors, Masses
        Jv = [self.Jv_cm_i(i+1) for i in range(n)]
        Jw = [self.Jw_cm_i(i+1) for i in range(n)]
        R  = [self.R_i0(i+1)    for i in range(n)]
        I  = [self.I_cm(i+1)    for i in range(n)]
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
        n = self.dof
        M = self.inertia_matrix()
        C = zeros(n)
        for i in range(1,n+1):
            for j in range(1,n+1):
                C[i-1,j-1] = 0
                for k in range(1,n+1):
                    C[i-1,j-1] += self.christoffel_symbols(i,j,k,M) * self.qs[k-1].diff()
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
        cijk = (1/2)*( mij.diff(q[idx_k]) + mik.diff(q[idx_j]) - mjk.diff(q[idx_i]) )
        return cijk
    
    def gravity_vector(self):
        """
        Compute the gravity torque vector G(q). The gravity torque vector is computed as the gradient of the potential energy of the system:

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
        qpp = Matrix([q.diff(t,2) for q in self.qs])
        qp = Matrix([q.diff(t) for q in self.qs])
        tau = Matrix([ symbols(f"tau_{i+1}") for i in range(self.dof)])
        return Eq(MatAdd( MatMul(M,qpp), MatMul(C,qp),  G) , tau)
            
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
        mi = self.m(i)
        vi = self.v_cm(i)
        wi = self.w(i)
        I_cmi = self.I_cm(i)
        Ri = self.R_i0(i)
        
        Ktra_i = (1/2) * mi * vi.T * vi
        Krot_i = (1/2) * wi.T * Ri * I_cmi * Ri.T * wi
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
        if self.gravity is None:
            raise ValueError("Gravity acceleration is not defined. Please set it using " \
            "the gravity property.") 
        
        return - self.m(i) * self.gravity.T * self.r_cm(i)
        
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
        L = self.lagrangian()
        equations = []
        for i in range(self.dof):
            q = self.qs[i]
            qp = self.qs[i].diff()
            equations.append( Eq( trigsimp(L.diff(qp).diff(t) - L.diff(q) ), symbols(f"tau_{i+1}") ) ) 
            
        return equations
    
    def solve_inverse_kinematics(self,pose,q0=None):
        """
        Solve the inverse kinematics problem for a given end-effector pose. This method is not implemented yet and will be added in future versions of the library.
        """
        pass
    
    def _set_default_joint_limits(self):
        joint_limits = []
        for k in range(self.dof):
            if self.joint_types[k] == "r":  # for revolute joint
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
        return self._joint_limits
    
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
        for limit in limits:
            if len(limit) != 2:
                raise ValueError("Each joint-limit should be a 2-tuple.")
        self._joint_limits = limits
    
    @property
    def _numerical_joint_limits(self):
        joint_limits = self.joint_limits 
        joint_limits_num = [(float(a), float(b)) for (a,b) in joint_limits] 
        return joint_limits_num
    
    def plot_diagram(self,num_vals):
        """
        Draw a simple wire-diagram or kinematic-diagram of the manipulator.

        Parameters
        ----------

        num_vals : dict
            Dictionary like: {svar1: nvalue1, svar2: nvalue2, ...}, 
            where svar1, svar2, ... are symbolic variables that are 
            currently used in model, and nvalue1, nvalue2, ... 
            are the numerical values that will substitute these variables.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        # Ts = self.Ts
        points = []
        Ti_0 = []
        points.append(zeros(1,3))
        for i in range(self.dof):
            Ti_0.append(self.T_i0(i+1).subs(num_vals))
            points.append((self.T_i0(i+1)[:3,3]).subs(num_vals))
            
        X = [float(k[0]) for k in points]
        Y = [float(k[1]) for k in points]
        Z = [float(k[2]) for k in points]
        ax.plot(X,Y,Z, "o-", color="#778877", lw=3)
        ax.plot([0],[0],[0], "mo", markersize=6)
        # ax.set_axis_off()
        ax.view_init(30,30)
        
        px,py,pz = float(X[-1]),float(Y[-1]),float(Z[-1])
        dim = max([px,py,pz])
        
        self._draw_uvw(eye(4),ax, dim)
        for T in Ti_0:
            self._draw_uvw(T, ax, dim)
            
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_zlim(-dim, dim)
        plt.show()
    
    def _draw_uvw(self,H,ax,sz=1):
        """
        Draw the u,v,w axes of a frame defined by the homogeneous transformation matrix H.

        Parameters
        ----------

        H: sympy.matrices.dense.MutableDenseMatrix
            Homogeneous transformation matrix that defines the frame to be drawn.
        ax: matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis where the frame will be drawn.
        sz: float
            The length of the axes to be drawn.
        """
        u = H[:3,0]
        v = H[:3,1]
        w = H[:3,2]
        o = H[:3,3]
        L = sz/5
        ax.quiver(o[0],o[1],o[2],u[0],u[1],u[2],color="r", length=L)
        ax.quiver(o[0],o[1],o[2],v[0],v[1],v[2],color="g", length=L)
        ax.quiver(o[0],o[1],o[2],w[0],w[1],w[2],color="b", length=L)
    
    def __str__(self):
        robot_type = "".join( self.joint_types ).upper()
        return f"Robot {robot_type}"
    
    def __repr__(self):
        robot_type = "".join( self.joint_types ).upper()
        return f"Robot {robot_type}"

    # def _repr_latex_(self):
    #     return sp.latex(self.dh_table)
    
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
        
        return self._cache[category][key]

            


#### RigidBody2D

class RigidBody2D(object):
    """
    Defines a rigid body (two-dimensional) through a series of points that 
    make it up.
    
    Parameters
    ----------
    
    points: list, tuple
        A list of 2-lists (or list of 2-tuples) containing the 
        N-points that make up the rigid body.

    Examples
    --------

    >>> points = [(0,0), (1,0), (0,1)]
    >>> rb = RigidBody2D(points)

    """
    def __init__(self,points):
        self._points = points # Points
        self.Hs = [eye(4),] # Transformation matrices
        
    def restart(self):
        """
        Restart to initial coordinates of the rigid body
        """
        self.Hs = [eye(4),]
    
    @property
    def points(self):
        _points = []
        H = self.H #
        for p in self._points:
            Q = Matrix([p[0],p[1],0,1]) # Homogeneous coordinates
            _points.append(H*Q)
        return _points
    
    @property
    def H(self):
        _h = eye(4)
        for _mth in self.Hs:
            _h = _h*_mth
        return _h

    def rotate(self,angle):
        """
        Rotates the rigid body around z-axis.
        """
        R = htmrot(angle, axis="z") # Applying rotation
        self.Hs.append(R)
    
    def move(self,q):
        """
        Moves the rigid body
        """
        D = htmtra(q) # Applying translation
        self.Hs.append(D)
        
    def draw(self,color="r",kaxis=None):
        """
        Draw the rigid body
        """
        X,Y = [],[]
        cx,cy = self.get_centroid()
        for p in self.points:
            X.append(p[0])
            Y.append(p[1])
        plt.fill(X,Y,color,alpha=0.8)
        plt.plot(cx,cy,"r.")
        plt.axis('equal')
        plt.grid(ls="--")
        
        O = self.H[:3,3]
        U = self.H[:3,0]
        V = self.H[:3,1]
        plt.quiver(float(O[0]), float(O[1]), float(U[0]), float(U[1]), 
                   color="r", zorder=1000, scale=kaxis)
        plt.quiver(float(O[0]), float(O[1]), float(V[0]), float(V[1]), 
                   color="g", zorder=1001, scale=kaxis)
        self.ax = plt.gca()

    def _gca(self):
        return self.ax

    def get_centroid(self):
        """
        Return the centroid of the rigid body
        """
        n = len(self.points)
        sx,sy = 0,0
        for point in self.points:
            sx += point[0]
            sy += point[1]
        cx = sx/n
        cy = sy/n
        return cx,cy



if __name__=="__main__":
    pass