"""

Numython R&D, (c) 2026 
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 

"""
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import operator, functools
import sympy as sp
from sympy import pi
from sympy.matrices import Matrix,eye,diag,zeros
from sympy import simplify, nsimplify
from sympy import Eq,MatAdd,MatMul
from moro.abc import *
from moro.transformations import *
from moro.util import *
import moro.inverse_kinematics as ikin

__all__ = ["Robot", "RigidBody2D"]

class Robot(object):
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
    def __init__(self,*args):
        self.Ts = [] # Transformation matrices i to i-1
        self.joint_types = [] # Joint type -> "r" revolute, "p" prismatic
        self.qs = [] # Joint variables
        for k in args:
            self.Ts.append(dh(k[0],k[1],k[2],k[3])) # Compute Ti->i-1
            if len(k)>4:
                self.joint_types.append(k[4])
            else: # By default, the joint type is assumed to be revolute
                self.joint_types.append('r')

            if self.joint_types[-1] == "r":
                self.qs.append(k[3])
            else:
                self.qs.append(k[2])
        self._dof = len(args) # Degree of freedom
        self.__set_default_joint_limits() # set default joint-limits on create
    
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
        return self.T_i0(i)[:3,2]
    
    def p(self,i):
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
        n = self.dof
        M_ = zeros(6,n)
        for i in range(1, n+1):
            idx = i - 1
            if self.joint_types[idx]=='r': # If i-th joint is revolute
                jp = self.z(i-1).cross(self.p(n) - self.p(i-1))
                jo = self.z(i-1)
            else: # If i-th joint is prismatic
                jp = self.z(i-1)
                jo = zeros(3,1)
            jp = jp.col_join(jo)
            M_[:,idx] = jp
        return simplify(M_)

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
        Get the homogeneous transformation matrix of {N}-Frame (end-effector)
        w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            T_n^0
        """
        return simplify(functools.reduce(operator.mul, self.Ts))
        
    def T_ij(self,i,j):
        """
        Get the homogeneous transformation matrix of {i}-Frame w.r.t. {j}-Frame. 
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            T_i^j
        """
        if i == j: return eye(4)
        return simplify(functools.reduce(operator.mul, self.Ts[j:i]))
        
    def T_i0(self,i):
        """
        Get the homogeneous transformation matrix of {i}-Frame w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Returns T_i^0
        """
        if i == 0:
            return eye(4)
        else:
            return self.T_ij(i,0) 
        
    def R_i0(self,i):
        """
        Get the rotation matrix of {i}-Frame w.r.t. {0}-Frame.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Returns R_i^0
        """
        return self.T_i0(i)[:3,:3]
        
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
    
    def qi(self, i):
        """
        Get the i-th articular variable.

        Parameters
        ----------

        i: int
            Joint number (starting from 1)
        """
        idx = i - 1
        return self.qs[idx]
    
    @property
    def qis_range(self):
        return self._qis_range
        
    @qis_range.setter
    def qis_range(self, *args):
        self._qis_range = args
        
    def __plot_workspace(self):
        """ 
        TODO 
        """
        pass
        
    def set_masses(self,masses):
        """
        Set mass for each link using a list like: [m1, m2, ..., mn], where 
        m1, m2, ..., mn, are numeric or symbolic values.
        
        Parameters
        ----------
        masses: list, tuple
            A list of numerical or symbolic values that correspond to link masses.
        """
        self.masses = masses
        
    def set_inertia_tensors(self,tensors=None):
        """
        Inertia tensor w.r.t. {i}'-Frame. Consider that the reference 
        frame {i}' is located at the center of mass of link [i] 
        and oriented in the same way as {i}-Frame. By default (if tensors argument
        is not passed), it is assumed that each link is symmetrical to, 
        at least, two planes of the reference frame located in its center of mass, 
        then products of inertia are zero.
        
        Parameters
        ----------
        tensors: sympy.matrices.dense.MutableDenseMatrix
            A list containinig `sympy.matrices.dense.MutableDenseMatrix` that 
            corresponds to each inertia tensor w.r.t. {i}'-Frame.
        """
        dof = self.dof
        self.inertia_tensors = []
        if tensors is None: # Default assumption
            for k in range(dof):
                Istr = f"I_{{x_{k+1}x_{k+1}}}, I_{{y_{k+1}y_{k+1}}} I_{{z_{k+1}z_{k+1}}}"
                Ix,Iy,Iz = symbols(Istr)
                self.inertia_tensors.append( diag(Ix,Iy,Iz) )
        else:
            for k in range(dof):
                self.inertia_tensors.append( tensors[k] )
            
    def set_cm_locations(self,cmlocs):
        """
        Set the positions of the center of mass for each 
        link.
    
        Parameters
        ----------
        cmlocs: list, tuple
            A list of lists (or tuples) or a tuple of tuples (or lists) containing 
            each center of mass position w.r.t. its reference frame.
        
        Examples
        --------
        >>> RR = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
        >>> RR.set_cm_locations([(-lc1,0,0), (-lc2,0,0)])
        """
        self.cm_locations = cmlocs

    def set_gravity_vector(self,G):
        """
        Set the gravity vector in the base frame.
        
        Parameters
        ----------
        G: list, tuple
            A list or tuple of three elements that define 
            the gravity vector in the base frame.
        """
        self.G = G
    
    def rcm_i(self,i):
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
        idx = i - 1
        rcm_ii = Matrix( self.cm_locations[idx] )
        rcm_i = ( self.T_i0(i) * vector_in_hcoords( rcm_ii ) )[:3,:]
        return simplify( rcm_i )
        
    def vcm_i(self,i):
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
        rcm_i = self.rcm_i(i)
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
        n = self.dof
        M_ = zeros(6,n)
        for j in range(1, n+1):
            idx = j - 1
            if j <= i:
                if self.joint_types[idx]=='r':
                    jp = self.z(j-1).cross(self.rcm_i(i) - self.p(j-1))
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
    
    def Jv_cm_i(self,i):
        """
        Return the linear velocity Jacobian matrix of the center of mass of the i-th link.

        Parameters
        ----------
        i : int
            Link number.
        """
        return self._J_cm_i(i)[:3,:]
    
    def Jw_cm_i(self,i):
        """
        Return the angular velocity Jacobian matrix of the center of mass of the i-th link.

        Parameters
        ----------
        i : int
            Link number.
        """
        return self._J_cm_i(i)[3:,:]
    
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
        return self._J_cm_i(i)
    
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
        idx = i - 1
        point_wrt_i = Matrix( point )
        point_wrt_0 = ( self.T_i0(i) * vector_in_hcoords( point_wrt_i ) )[:3,:]
        
        n = self.dof
        M_ = zeros(6,n)
        for j in range(1, n+1):
            idx = j - 1
            if j <= i:
                if self.joint_types[idx]=='r':
                    jp = self.z(j-1).cross(point_wrt_0 - self.p(j-1))
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
        
    def w_ijj(self,i):
        """
        Return the angular velocity of the [i]-link w.r.t. [j]-link, 
        described in {j}-Frame, where j = i - 1. 
        
        Since we are using Denavit-Hartenberg frames, then:
        
        .. math:: 
            
            \\omega_{{i-i,i}}^{{i-1}} = \\begin{bmatrix} 0 \\\\ 0 \\\\ \\dot{{q}}_i \\end{bmatrix}
            
        If the i-th joint is revolute, or:
        
        .. math:: 
            
            \\omega_{{i-i,i}}^{{i-1}} = \\begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\end{bmatrix}
        
        If the i-th joint is a prismatic.
        
        Parameters
        ----------
        i : int
            Link number.
        """
        idx = i - 1 
        if self.joint_types[idx] == "r":
            wijj = Matrix([0,0,self.qs[idx].diff()])
        else:
            wijj = Matrix([0,0,0])
        return wijj
            
        
    def w_i(self,i):
        """
        Compute the angular velocity of the [i]-link w.r.t. {0}-Frame.
        
        Parameters
        ----------
        i: int 
            Link number.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Angular velocity of the [i]-link w.r.t. {0}-Frame.
        
        Examples
        --------
        >>> RR = Robot((l1,0,0,q1,"r"), (l2,0,0,q2,"r"))
        >>> pprint(RR.w_i(1))
        ⎡    0    ⎤
        ⎢         ⎥
        ⎢    0    ⎥
        ⎢         ⎥
        ⎢d        ⎥
        ⎢──(q₁(t))⎥
        ⎣dt       ⎦
        >>> pprint(RR.w_i(2))
        ⎡          0          ⎤
        ⎢                     ⎥
        ⎢          0          ⎥
        ⎢                     ⎥
        ⎢d           d        ⎥
        ⎢──(q₁(t)) + ──(q₂(t))⎥
        ⎣dt          dt       ⎦
        
        """
        wi = Matrix([0,0,0])
        for k in range(1,i+1):
            wi += self.R_i0(k-1)*self.w_ijj(k)
        return wi
        
    def I_i(self,i):
        """
        Return the inertia tensor of [i-th] link w.r.t. base frame.
        
        Parameters
        ----------
        i: int 
            Link number.
        
        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Inertia tensor of the [i]-link w.r.t. {0}-Frame.
        """
        if i == 0:
            raise ValueError("i must be greater than 0")
        idx = i - 1
        # self.set_inertia_tensors()
        Iii = self.inertia_tensors[idx]
        Ii = simplify( self.R_i0(i) * Iii * self.R_i0(i).T )
        return Ii
    
    def I_ii(self,i):
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
        if i == 0:
            raise ValueError("i must be greater than 0")
        idx = i - 1
        Iii = self.inertia_tensors[idx]
        return Iii
    
    def m_i(self,i):
        """
        Return the mass of the i-th link. 

        Parameters
        ----------

        i: int  
            Link number.
        """
        return self.masses[i-1]
        
    def get_inertia_matrix(self):
        """
        Return the inertia (mass) matrix

        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Inertia matrix M(q)
        """
        n = self.dof
        M = zeros(n)

        # Precompute Jacobians, Rotations, Inertia tensors, Masses
        Jv = [self.Jv_cm_i(i+1) for i in range(n)]
        Jw = [self.Jw_cm_i(i+1) for i in range(n)]
        R  = [self.R_i0(i+1)    for i in range(n)]
        I  = [self.I_ii(i+1)    for i in range(n)]
        m  = [self.m_i(i+1)     for i in range(n)]

        # Compute inertia matrix
        for i in range(n):
            M += m[i] * Jv[i].T * Jv[i]
            M += Jw[i].T * R[i] * I[i] * R[i].T * Jw[i]

        return simplify(M)

    def get_coriolis_matrix(self):
        n = self.dof
        M = self.get_inertia_matrix()
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
        # M = self.get_inertia_matrix()
        q = self.qs
        idx_i, idx_j, idx_k = i-1, j-1, k-1 
        mij = M[idx_i, idx_j]
        mik = M[idx_i, idx_k]
        mjk = M[idx_j, idx_k]
        cijk = (1/2)*( mij.diff(q[idx_k]) + mik.diff(q[idx_j]) - mjk.diff(q[idx_i]) )
        return cijk
    
    
    def get_coriolis_matrix_v2(self):
        """
        Calcula la matriz de Coriolis de manera completamente vectorial, sin bucles,
        y asegurando la correcta implementación de las derivadas cruzadas.
        """
        n = self.dof
        qs = self.qs                   # Coordenadas generalizadas
        qds = Matrix([q.diff() for q in qs])  # Derivadas de las coordenadas generalizadas (velocidades)
        M = self.get_inertia_matrix()   # Matriz de inercia n x n
        
        # Derivadas de la matriz de inercia respecto a cada q_k (matrices n x n)
        dM = [M.diff(q) for q in qs]  # Lista de derivadas de M respecto a q_k
        
        # Transpuestas de las derivadas de M
        dM_transpose = [dM_k.T for dM_k in dM]  # Transpuestas de las derivadas
        
        # Creamos la matriz de Coriolis usando la fórmula vectorial:
        # C_ij = 0.5 * sum_k( (dM/dq_k + dM/dq_k^T - dM_jk/dq_i) * qds_k )
        
        # Inicializamos la matriz C
        C = Matrix.zeros(n)
        
        for k in range(n):  # Iteramos solo sobre k
            # Multiplicamos cada término de la matriz derivada por la velocidad correspondiente
            # Aseguramos que las derivadas cruzadas se calculen correctamente
            term = 0.5 * (dM[k] + dM_transpose[k] - dM_transpose[k].T)
            C += term * qds[k]

        # Aseguramos que la matriz sea antisimétrica (esto es una propiedad de la matriz de Coriolis)
        C = (C - C.T) / 2

        return nsimplify(C)

    def christoffel_symbols_v2(self, i, j, k):
        M = self.get_inertia_matrix()
        q = self.qs
        # Ajuste de índices para Python
        idx_i, idx_j, idx_k = i-1, j-1, k-1
        mij = M[idx_i, idx_j]
        mik = M[idx_i, idx_k]
        mjk = M[idx_j, idx_k]
        
        # Fórmula de los símbolos de Christoffel
        cijk = (1/2) * (mij.diff(q[idx_k]) + mik.diff(q[idx_j]) - mjk.diff(q[idx_i]))
        return cijk
    
    def get_gravity_torque_vector(self):
        """
        Return the gravity torque vector G(q).

        Returns
        -------
        sympy.matrices.dense.MutableDenseMatrix
            Gravity torque vector G(q)
        """
        pot = self.get_potential_energy()
        gv = [nsimplify(pot.diff(k)) for k in self.qs]
        return Matrix(gv)
    
    def get_dynamic_model_matrix_form(self):
        """
        Return the dynamic model of the robot in matrix form:

        M(q) q'' + C(q,q') q' + G(q) = tau

        where M(q) is the inertia matrix, C(q,q') is the Coriolis matrix, 
        G(q) is the gravity torque vector, and tau is the vector of joint torques.

        """
        M = self.get_inertia_matrix()
        C = self.get_coriolis_matrix()
        G = self.get_gravity_torque_vector()
        qpp = Matrix([q.diff(t,2) for q in self.qs])
        qp = Matrix([q.diff(t) for q in self.qs])
        tau = Matrix([ symbols(f"tau_{i+1}") for i in range(len(qp))])
        return Eq(MatAdd( MatMul(M,qpp), MatMul(C,qp),  G) , tau)
            
    def kin_i(self,i):
        """
        Returns the kinetic energy of i-th link.

        .. math::
        
            K_i = \\frac{1}{2} m_i \\mathbf{v}_{G_i}^T \\mathbf{v}_{G_i} + \\frac{1}{2} \\boldsymbol{\\omega}_i^T I_i \\boldsymbol{\\omega}_i

        Parameters
        ----------
        i: int
            Link number.
        """
        idx = i - 1
        mi = self.masses[idx]
        vi = self.vcm_i(i)
        wi = self.w_i(i)
        Ii = self.I_i(i)
        
        Ktra_i = (1/2) * mi * vi.T * vi
        Krot_i = (1/2) * wi.T * Ii * wi
        Ki = Ktra_i + Krot_i
        return Ki
        
    def pot_i(self,i):
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
        idx = i - 1
        mi = self.masses[idx]
        G = Matrix( self.G )
        rcm_i = self.rcm_i(i)
        return - mi * G.T * rcm_i
        
    def get_kinetic_energy(self):
        """
        Returns the total kinetic energy of the robot
        """
        K = Matrix([0])
        for i in range(self.dof):
            K += self.kin_i(i+1) 
        return nsimplify(K)
        
    def get_potential_energy(self):
        """
        Returns the total potential energy of the robot
        """
        U = Matrix([0])
        for i in range(self.dof):
            U += self.pot_i(i+1) 
        return nsimplify(U)
        
    def get_dynamic_model(self):
        """
        Returns the dynamic model of the robot 
        using the Euler-Lagrange formulation. The returned value is a list of equations,
        one for each joint, of the form:

        .. math::   
            \\frac{d}{dt} \\left( \\frac{\\partial L}{\\partial \\dot{{q}}_i} \\right) - \\frac{\\partial L}{\\partial q_i} = \\tau_i
        
        where L is the Lagrangian of the system, defined as L = K - P, where K 
        is the kinetic energy and P is the potential energy.
        """
        K = self.get_kinetic_energy()
        U = self.get_potential_energy()
        L = ( K - U )[0]
        equations = []
        for i in range(self.dof):
            q = self.qs[i]
            qp = self.qs[i].diff()
            equations.append( Eq( simplify(L.diff(qp).diff(t) - L.diff(q) ), symbols(f"tau_{i+1}") ) ) 
            
        return equations
    
    def solve_inverse_kinematics(self,pose,q0=None):
        r_e = self.T[:3,3] # end-effector position
        if is_position_vector(pose):
            eqs = r_e - pose
            variables = self.qs # all joint variables
            joint_limits = self.__numerical_joint_limits # all joint limits
            if q0 is None:
                initial_guesses = ikin.generate_random_initial_guesses(variables, joint_limits)
            else:
                initial_guesses = q0
            # print(eqs, variables, initial_guesses, joint_limits)
            ikin_sol = ikin.solve_inverse_kinematics(eqs, variables, initial_guesses, joint_limits, method="GD")
        if is_SE3(pose) and self.dof == 6:
            variables = self.qs # all joint variables
            joint_limits = self.__numerical_joint_limits # all joint limits
            if q0 is None:
                initial_guesses = ikin.generate_random_initial_guesses(variables, joint_limits)
            else:
                initial_guesses = q0
            # If pose is a SE(3)
            # # raise NotImplementedError("This method hasn't been implemented yet")
            ikin_sol = ikin.pieper_method(pose,*self.Ts, variables, initial_guesses, joint_limits)
        return ikin_sol
    
    def __set_default_joint_limits(self):
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
        return self._joint_limits
    
    @joint_limits.setter
    def joint_limits(self,*limits):
        if len(limits) != self.dof:
            raise ValueError("The number of joint limits must match DOF.")
        for limit in limits:
            if len(limit) != 2:
                raise ValueError("Each joint-limit should be a 2-tuple.")
        self._joint_limits = limits
    
    @property
    def __numerical_joint_limits(self):
        joint_limits = self.joint_limits 
        joint_limits_num = [(float(a), float(b)) for (a,b) in joint_limits] 
        return joint_limits_num
    
    def __str__(self):
        repr = "".join( self.joint_types ).upper()
        return f"Robot {repr}"
    
    def __repr__(self):
        repr = "".join( self.joint_types ).upper()
        return f"Robot {repr}"
    
    def plot_diagram_threejs(self, num_vals, width=800, height=600):
        """
        Dibuja el diagrama cinemático del robot usando Three.js en Jupyter.
        
        Parameters
        ----------
        num_vals : dict
            Diccionario con valores numéricos para las variables simbólicas
        width : int
            Ancho del canvas en pixels
        height : int
            Alto del canvas en pixels
        """
        from IPython.display import HTML
        import json
        import uuid
        
        # Generar ID único para evitar conflictos
        unique_id = str(uuid.uuid4())[:8]
        
        # Extraer posiciones de joints
        joints = []
        frames = []
        
        # Frame base
        joints.append([0.0, 0.0, 0.0])
        frames.append({
            'position': [0.0, 0.0, 0.0],
            'x': [1.0, 0.0, 0.0],
            'y': [0.0, 1.0, 0.0],
            'z': [0.0, 0.0, 1.0]
        })
        
        # Para cada joint del robot
        for i in range(self.dof):
            Ti = self.T_i0(i + 1).subs(num_vals)
            
            # Posición del joint
            pos = [float(Ti[j, 3]) for j in range(3)]
            joints.append(pos)
            
            # Orientación del frame (ejes x, y, z)
            frames.append({
                'position': pos,
                'x': [float(Ti[j, 0]) for j in range(3)],
                'y': [float(Ti[j, 1]) for j in range(3)],
                'z': [float(Ti[j, 2]) for j in range(3)]
            })
        
        # Calcular dimensión para escalar la vista
        all_coords = [coord for joint in joints for coord in joint]
        max_coord = max(abs(c) for c in all_coords) if all_coords else 100
        dim = max(max_coord * 1.5, 50)
        
        robot_data = {
            'joints': joints,
            'frames': frames,
            'dimension': float(dim)
        }
        
        # Convertir a JSON
        robot_json = json.dumps(robot_data)
        
        # Template HTML con Three.js usando IDs únicos
        html_template = f"""
        <div id="container-{unique_id}" style="width: {width}px; height: {height}px; border: 1px solid #ccc; position: relative;">
            <div id="controls-{unique_id}" style="position: absolute; top: 10px; left: 10px; background: rgba(255, 255, 255, 0.95); padding: 10px; border-radius: 5px; font-family: Arial, sans-serif; font-size: 12px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <button onclick="window.robot_{unique_id}.toggleRotation()" style="margin: 2px; padding: 5px 10px; cursor: pointer; border: none; background: #4CAF50; color: white; border-radius: 3px;">⏯ Rotar</button>
                <button onclick="window.robot_{unique_id}.resetView()" style="margin: 2px; padding: 5px 10px; cursor: pointer; border: none; background: #4CAF50; color: white; border-radius: 3px;">🔄 Reset</button>
                <div id="status-{unique_id}" style="margin-top: 5px; padding: 5px; font-size: 10px; color: #666;">Cargando...</div>
            </div>
        </div>
        
        <script>
        (function() {{
            // Verificar si THREE ya está cargado
            if (typeof THREE !== 'undefined') {{
                initRobot_{unique_id}();
            }} else {{
                // Cargar Three.js
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
                script.onload = function() {{
                    initRobot_{unique_id}();
                }};
                script.onerror = function() {{
                    document.getElementById('status-{unique_id}').innerHTML = 'Error cargando Three.js ✗';
                    document.getElementById('status-{unique_id}').style.color = 'red';
                }};
                document.head.appendChild(script);
            }}
            
            function initRobot_{unique_id}() {{
                const robotData = {robot_json};
                const container = document.getElementById('container-{unique_id}');
                
                // Variables locales para este robot
                let scene, camera, renderer, robotGroup;
                let isRotating = false;
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                try {{
                    // Escena
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0xf5f5f5);
                    
                    // Cámara
                    camera = new THREE.PerspectiveCamera(
                        50,
                        {width} / {height},
                        0.1,
                        robotData.dimension * 10
                    );
                    const camDist = robotData.dimension * 2;
                    camera.position.set(camDist, camDist, camDist);
                    camera.lookAt(0, 0, robotData.dimension / 2);
                    
                    // Renderer
                    renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize({width}, {height});
                    renderer.setPixelRatio(window.devicePixelRatio);
                    container.appendChild(renderer.domElement);
                    
                    // Luces
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                    scene.add(ambientLight);
                    
                    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
                    dirLight1.position.set(robotData.dimension, robotData.dimension, robotData.dimension);
                    scene.add(dirLight1);
                    
                    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
                    dirLight2.position.set(-robotData.dimension, -robotData.dimension, -robotData.dimension);
                    scene.add(dirLight2);
                    
                    // Grid
                    const gridSize = robotData.dimension * 2;
                    const gridHelper = new THREE.GridHelper(gridSize, 20, 0x888888, 0xcccccc);
                    scene.add(gridHelper);
                    
                    // Ejes principales
                    const axesHelper = new THREE.AxesHelper(robotData.dimension / 4);
                    scene.add(axesHelper);
                    
                    // Grupo del robot
                    robotGroup = new THREE.Group();
                    scene.add(robotGroup);
                    
                    // Dibujar robot
                    drawRobot();
                    
                    // Event listeners
                    renderer.domElement.addEventListener('mousedown', onMouseDown);
                    renderer.domElement.addEventListener('mousemove', onMouseMove);
                    renderer.domElement.addEventListener('mouseup', onMouseUp);
                    renderer.domElement.addEventListener('mouseleave', onMouseUp);
                    renderer.domElement.addEventListener('wheel', onWheel, {{ passive: false }});
                    
                    // Animación
                    function animate() {{
                        requestAnimationFrame(animate);
                        
                        if (isRotating && robotGroup) {{
                            robotGroup.rotation.y += 0.005;
                        }}
                        
                        renderer.render(scene, camera);
                    }}
                    animate();
                    
                    document.getElementById('status-{unique_id}').innerHTML = 
                        'Listo - Arrastra para rotar | Scroll para zoom';
                    document.getElementById('status-{unique_id}').style.color = 'green';
                    
                }} catch(error) {{
                    console.error('Error:', error);
                    document.getElementById('status-{unique_id}').innerHTML = 
                        'Error: ' + error.message;
                    document.getElementById('status-{unique_id}').style.color = 'red';
                }}
                
                function drawRobot() {{
                    const {{ joints, frames }} = robotData;
                    
                    // Material para links
                    const linkMaterial = new THREE.MeshPhongMaterial({{
                        color: 0x778877,
                        shininess: 30,
                        side: THREE.DoubleSide
                    }});
                    
                    // Dibujar links
                    for (let i = 0; i < joints.length - 1; i++) {{
                        const start = new THREE.Vector3(...joints[i]);
                        const end = new THREE.Vector3(...joints[i + 1]);
                        
                        const direction = new THREE.Vector3().subVectors(end, start);
                        const length = direction.length();
                        
                        if (length > 0.001) {{
                            const radius = Math.max(robotData.dimension * 0.015, 1);
                            const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
                            const link = new THREE.Mesh(geometry, linkMaterial);
                            
                            const midpoint = start.clone().add(direction.clone().multiplyScalar(0.5));
                            link.position.copy(midpoint);
                            
                            const axis = new THREE.Vector3(0, 1, 0);
                            link.quaternion.setFromUnitVectors(axis, direction.clone().normalize());
                            
                            robotGroup.add(link);
                        }}
                    }}
                    
                    // Dibujar joints
                    joints.forEach((joint, index) => {{
                        const radius = index === 0 ? 
                            Math.max(robotData.dimension * 0.03, 2) : 
                            Math.max(robotData.dimension * 0.025, 1.5);
                        const geometry = new THREE.SphereGeometry(radius, 16, 16);
                        const material = new THREE.MeshPhongMaterial({{
                            color: index === 0 ? 0xff00ff : 0xff1493,
                            shininess: 50
                        }});
                        const sphere = new THREE.Mesh(geometry, material);
                        sphere.position.set(...joint);
                        robotGroup.add(sphere);
                    }});
                    
                    // Dibujar sistemas de coordenadas
                    const arrowLength = Math.max(robotData.dimension / 5, 10);
                    const arrowHeadLength = arrowLength * 0.2;
                    const arrowHeadWidth = arrowLength * 0.15;
                    
                    frames.forEach((frame) => {{
                        const origin = new THREE.Vector3(...frame.position);
                        
                        // Eje X (rojo)
                        const xDir = new THREE.Vector3(...frame.x).normalize();
                        const xArrow = new THREE.ArrowHelper(
                            xDir, origin, arrowLength, 0xff0000, 
                            arrowHeadLength, arrowHeadWidth
                        );
                        robotGroup.add(xArrow);
                        
                        // Eje Y (verde)
                        const yDir = new THREE.Vector3(...frame.y).normalize();
                        const yArrow = new THREE.ArrowHelper(
                            yDir, origin, arrowLength, 0x00ff00,
                            arrowHeadLength, arrowHeadWidth
                        );
                        robotGroup.add(yArrow);
                        
                        // Eje Z (azul)
                        const zDir = new THREE.Vector3(...frame.z).normalize();
                        const zArrow = new THREE.ArrowHelper(
                            zDir, origin, arrowLength, 0x0000ff,
                            arrowHeadLength, arrowHeadWidth
                        );
                        robotGroup.add(zArrow);
                    }});
                }}
                
                function onMouseDown(event) {{
                    isDragging = true;
                    previousMousePosition = {{ x: event.clientX, y: event.clientY }};
                }}
                
                function onMouseMove(event) {{
                    if (!isDragging || !robotGroup) return;
                    
                    const deltaX = event.clientX - previousMousePosition.x;
                    const deltaY = event.clientY - previousMousePosition.y;
                    
                    robotGroup.rotation.y += deltaX * 0.01;
                    robotGroup.rotation.x += deltaY * 0.01;
                    
                    previousMousePosition = {{ x: event.clientX, y: event.clientY }};
                }}
                
                function onMouseUp() {{
                    isDragging = false;
                }}
                
                function onWheel(event) {{
                    event.preventDefault();
                    if (camera) {{
                        const delta = event.deltaY * 0.001;
                        const scale = 1 + delta;
                        camera.position.multiplyScalar(scale);
                    }}
                }}
                
                // Exponer funciones públicas
                window.robot_{unique_id} = {{
                    toggleRotation: function() {{
                        isRotating = !isRotating;
                    }},
                    resetView: function() {{
                        const camDist = robotData.dimension * 2;
                        camera.position.set(camDist, camDist, camDist);
                        camera.lookAt(0, 0, robotData.dimension / 2);
                        robotGroup.rotation.set(0, 0, 0);
                    }}
                }};
            }}
        }})();
        </script>
        """
        
        return HTML(html_template)

            


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
        plt.quiver(float(O[0]), float(O[1]), float(U[0]), float(U[1]), color="r", zorder=1000, scale=kaxis)
        plt.quiver(float(O[0]), float(O[1]), float(V[0]), float(V[1]), color="g", zorder=1001, scale=kaxis)
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



def test_robot():
    # ABB = Robot((0,pi/2,330,q1), 
    #             (320,0,0,q2), 
    #             (0,pi/2,0,q3), 
    #             (0,-pi/2,300,q4), 
    #             (0,pi/2,0,q5), 
    #             (0,0,80,q6))
    r = Robot((0,pi/2,d1,q1),(l2,0,0,q2), (l3,0,0,q3))
    RRP = Robot((0,pi/2,d1,q1), (0,pi/2,0,q2), (0,0,q3,0))
    # r.plot_diagram({q1:0, q2:pi/4, q3:0, d1:100, l2:100, l3:100})
    RRP.plot_diagram({q1:0, q2:pi/2, q3:250, d1:100})
    # ABB.plot_diagram(
    #     {
    #         q1:deg2rad(33.69),
    #         q2:deg2rad(-26.13),
    #         q3:deg2rad(191.99),
    #         q4:deg2rad(180),
    #         q5:deg2rad(165.87),
    #         q6:deg2rad(-146.31)
    #     }
    # )
    
    
def test_rb2():
    points = [(0,0),(3,0),(0,1)]
    rb = RigidBody2D(points)
    rb.draw("r")
    rb.move([10,0,0])
    rb.draw("g")
    rb.rotate(pi/2)
    rb.move([5,0,0])
    rb.draw("b")
    plt.show()
    print(rb.Hs)


if __name__=="__main__":
    # test_robot()
    robot = Robot((l1, 0, 0, q1), (l2, 0, 0, q2), (l3, 0, 0, q3))

    # Visualizar directamente en Jupyter!
    print(robot.plot_diagram_threejs({
        q1: pi/6,
        q2: pi/4,
        q3: -pi/6,
        l1: 100,
        l2: 80,
        l3: 60
    }))
        
