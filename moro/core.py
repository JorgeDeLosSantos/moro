"""

Numython R&D, (c) 2020
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import operator, functools
from sympy import *
from sympy.matrices import Matrix,eye,diag
from moro.abc import *
from moro.transformations import *
from moro.util import *

__all__ = ["Robot", "RigidBody2D"]

class Robot(object):
    """
    Define a robot-serial-arm given the Denavit-Hartenberg parameters 
    and joint type, as tuples.

    Each tuple must have the form:

    (a_i, alpha_i, d_i, theta_i)

    Or including the joint type:

    (a_i, alpha_i, d_i, theta_i, joint_type)

    All parameters are `int` or `floats`, or a symbolic variable. 
    Numeric angles must be passed in radians.

    Examples
    --------
    
    >>> rr = Robot((l1,0,0,t1), (l2,0,0,t2))

    or

    >>> rr2 = Robot((l1,0,0,t1,"r"), (l2,0,0,t2,"r"))
    """
    def __init__(self,*args):
        self.Ts = [] # Transformation matrices i to i-1
        self.joint_types = [] # Joint type -> "r" revolute, "p" prismatic
        self.qs = []
        for k in args:
            self.Ts.append(dh(k[0],k[1],k[2],k[3])) # Compute Ti->i-1
            if len(k)>4:
                self.joint_types.append(k[4])
            else:
                self.joint_types.append('r')

            if self.joint_types[-1] is "r":
                self.qs.append(k[3])
            else:
                self.qs.append(k[2])
        self._dof = len(args) # Degree of freedom
    
    def z(self,i):
        """
        Get the z-dir of every {i}-Frame w.r.t. {0}-Frame
        """
        return self.T_i0(i)[:3,2]
    
    def p(self,i):
        """
        Get the position for every {i}-Frame w.r.t. {0}-Frame
        """
        return self.T_i0(i)[:3,3]
    
    @property
    def J(self):
        """
        Geometric Jacobian matrix
        """
        n = self.dof
        M_ = zeros(6,n)
        for i in range(1, n+1):
            idx = i - 1
            if self.joint_types[idx]=='r':
                jp = self.z(i-1).cross(self.p(n) - self.p(i-1))
                jo = self.z(i-1)
            else:
                jp = self.z(i-1)
                jo = zeros(3,1)
            jp = jp.col_join(jo)
            M_[:,idx] = jp
        return simplify(M_)

    @property
    def dof(self):
        """
        Get the degrees of freedom
        """
        return self._dof

    @property
    def T(self):
        """ 
        Get Homogeneous transformation matrix of {N}-Frame w.r.t. {0}-Frame
        """
        return simplify(functools.reduce(operator.mul, self.Ts))
        
    def T_ij(self,i,j):
        """
        Get the HTM of {i}-Frame w.r.t. {j}-Frame.
        """
        if i == j: return eye(4)
        return simplify(functools.reduce(operator.mul, self.Ts[j:i]))
        
    def T_i0(self,i):
        """
        Get the HTM of {i}-Frame w.r.t. {0}-Frame.
        """
        if i == 0:
            return eye(4)
        else:
            return self.T_ij(i,0) 
        
    def R_i0(self,i):
        """
        Get the rotation matrix of {i}-Frame w.r.t. {0}-Frame.
        """
        return self.T_i0(i)[:3,:3]
        
    def plot_diagram(self,vals):
        """
        Draw a simple wire-diagram or kinematic-diagram of the manipulator.

        Parameters
        ----------

        vals : dict
            Dictionary like: {svar1: nvalue1, svar2: nvalue2, ...}, 
            where svar1, svar2, ... are symbolic variables that are 
            currently used in model, and nvalue1, nvalue2, ... 
            are the numerical values that will substitute these variables.

        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        Ts = self.Ts
        points = []
        Ti_0 = []
        points.append(zeros(1,3))
        for i in range(self.dof):
            Ti_0.append(self.T_i0(i+1).subs(vals))
            points.append((self.T_i0(i+1)[:3,3]).subs(vals))
            
        X = [float(k[0]) for k in points]
        Y = [float(k[1]) for k in points]
        Z = [float(k[2]) for k in points]
        ax.plot(X,Y,Z, "o-", color="#778877", lw=3)
        ax.plot([0],[0],[0], "mo", markersize=6)
        ax.set_axis_off()
        ax.view_init(90,0)
        
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
        """ TODO """
        pass
        
    def set_masses(self,masses):
        """
        Set mass for each link using a list like: [m1, m2, ..., mn], where 
        m1, m2, ..., mn, are numeric or symbolic values.
        """
        self.masses = masses
        
    def set_inertia_tensors(self,tensors=None):
        dof = self.dof
        self.inertia_tensors = []
        if tensors is None:
            for k in range(dof):
                Istr = "I_{{xx{0}}}, I_{{yy{0}}}, I_{{zz{0}}}".format(k+1)
                Ix,Iy,Iz = symbols(Istr)
                self.inertia_tensors.append( diag(Ix,Iy,Iz) )
        else:
            for k in range(dof):
                self.inertia_tensors.append( tensors[k] )
            
    def set_cm_locations(self,cmlocs):
        self.cm_locations = cmlocs

    def set_gravity_vector(self,G):
        """
        Set the gravity vector in the base frame.
        """
        self.G = G
    
    def rcm_i(self,i):
        idx = i - 1
        rcm_ii = Matrix( self.cm_locations[idx] )
        rcm_i = ( self.T_i0(i) * hcoords( rcm_ii ) )[:3,:]
        return simplify( rcm_i )
        
    def vcm_i(self,i):
        rcm_i = self.rcm_i(i)
        vcm_i = rcm_i.diff(t)
        return simplify( vcm_i )
    
    def _J_cm_i(self,i):
        """
        Geometric Jacobian matrix
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
        return self._J_cm_i(i)[:3,:]
    
    def Jw_cm_i(self,i):
        return self._J_cm_i(i)[3:,:]
    
    def J_cm_i(self,i):
        return self._J_cm_i(i)
    
    def J_point(self,point,i):
        idx = i - 1
        point_wrt_i = Matrix( point )
        point_wrt_0 = ( self.T_i0(i) * hcoords( point_wrt_i ) )[:3,:]
        
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
        Angular velocity of {i}-Frame w.r.t. {j}-Frame, described 
        in {j}-Frame, where j = i - 1. 
        """
        # j = i - 1
        idx = i - 1 
        if self.joint_types[idx] == "r":
            wijj = Matrix([0,0,self.qs[idx].diff()])
        else:
            wijj = Matrix([0,0,0])
        return wijj
            
        
    def w_i(self,i):
        """
        Angular velocity of {i}-Frame w.r.t. {0}-Frame.
        """
        idx = i - 1
        wi = Matrix([0,0,0])
        for k in range(1,i+1):
            wi += self.R_i0(k-1)*self.w_ijj(k)
        return wi
        
    def I_i(self,i):
        """
        Returns the inertia tensor of i-th link w.r.t. base frame.
        """
        if i == 0:
            raise ValueError("i must be greater than 0")
        idx = i - 1
        # self.set_inertia_tensors()
        Iii = self.inertia_tensors[idx]
        Ii = simplify( self.R_i0(i) * Iii * self.R_i0(i).T )
        return Ii
            
    def kin_i(self,i):
        """
        Returns the kinetic energy of i-th link
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
        Returns the potential energy of i-th link
        """
        idx = i - 1
        mi = self.masses[idx]
        G = Matrix( self.G )
        rcm_i = self.rcm_i(i)
        return - mi * G.T * rcm_i
        
    def get_kinetic_energy(self):
        """
        Returns the kinetic energy of the robot
        """
        K = Matrix([0])
        for i in range(self.dof):
            K += self.kin_i(i+1) 
        return nsimplify(K)
        
    def get_potential_energy(self):
        """
        Returns the potential energy of the robot
        """
        U = Matrix([0])
        for i in range(self.dof):
            U += self.pot_i(i+1) 
        return nsimplify(U)
        
    def get_dynamic_model(self):
        """
        Returns the dynamic model of the robot
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



#### RigidBody2D

class RigidBody2D(object):
    """
    Defines a rigid body through a series of points that 
    make it up.

    :param points: List of lists or tuples
    :type points: list, tuple


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
        R = htmrot(angle, axis="z") # Aplicando rotación
        self.Hs.append(R)
    
    def move(self,q):
        """
        Moves the rigid body
        """
        D = htmtra(q) # Aplicando traslación
        self.Hs.append(D)
        
    def __scale(self,sf):
        """
        Escala el cuerpo rígido
        """
        # ~ S = self.scale_matrix(sf) # Aplicando escalado
        # ~ self.Hs.append(S)
        pass # nothing to do here

    def __scale_matrix(self,sf):
        M = Matrix([[sf,0,0,0],
                      [0,sf,0,0],
                      [0,0,sf,0],
                      [0,0,0,sf]])
        return M
        
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
    r = Robot((l1,0,0,t1), (l2,0,0,t2))
    r.plot_diagram({t1:pi/2, t2:pi/2, l1:100, l2:100})
    
    
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
    pass
    
