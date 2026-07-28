import sympy as sp
import random
from scipy.optimize import least_squares

def solve_inverse_kinematics(equations,
                             variables,
                             initial_guesses,
                             joint_limits,
                             method="nsolve",
                             max_attempts=5,
                             tol=1e-6):
    attempts = 0
    solution = []
    sci_eqs = sp.lambdify(variables, equations, "numpy")
    feqs = lambda x: sci_eqs(*tuple(x)).flatten()
    
    while attempts < max_attempts:
        ls_sol = least_squares(feqs, 
                                initial_guesses, 
                                bounds=tuple(zip(*joint_limits)))
        if ls_sol.cost < tol:
            solution = [dict( zip(variables, ls_sol.x) )]        
            break
        attempts += 1
    print(f"Attempts: {attempts}")
    if not(solution):
        raise ValueError("Could not find solution within given limits.")
    return solution

# def solve_inverse_kinematics_2(equations,
#                              variables,
#                              initial_guesses,
#                              joint_limits,
#                              method="nsolve",
#                              max_steps=10):
#     current_step = 1
#     try:
#         solution = nsolve(equations, variables, initial_guesses, method)
#     except ValueError:
#         initial_guesses = generate_random_initial_guesses(variables, joint_limits)
#         solution = nsolve(equations, variables, initial_guesses, method)
#     while current_step <= max_steps:
#         no_sol = 0
#         if len(solution) == 0:
#             no_sol += 1
#         for k in range(len(solution[0])):
#             if not(is_in_range(solution[0][variables[k]], joint_limits[k])):
#                 no_sol += 1
#         if no_sol > 0:
#             try:
#                 initial_guesses = generate_random_initial_guesses(variables, joint_limits)
#                 solution = nsolve(equations, variables, initial_guesses, method)
#             except ValueError:
#                 pass # skip current step
#         else:
#             break
#         current_step += 1
#     if current_step > max_steps:
#         raise ValueError("Could not find solution within given limits.")
#     return solution

# def nsolve(equations,variables,initial_guesses,method):
#     if method=="nsolve":
#         return sp.nsolve(equations, variables, initial_guesses, dict=True)
#     else:
#         return gradient_descent(equations, variables, initial_guesses)

# def gradient_descent(equations,variables,initial_guesses,eps=1e-8):
#     J = equations.jacobian(variables)
#     # print(J)
#     joint_pos = dict( zip(variables, initial_guesses) ) # joint pos
#     q = sp.Matrix(initial_guesses)
#     e = equations.subs(joint_pos)
#     beta = 0.01
#     k = 0
#     while e.norm() > eps:
#         JN = J.subs( joint_pos )
#         Jinv = JN.pinv()
#         De = beta*-e
#         Dq = Jinv*De
#         q = q + Dq
#         joint_pos = dict( zip(variables, q) ) # updating joint positions
#         e = equations.subs(joint_pos)
#         k += 1
#         if k > 10:
#             raise ValueError(f"Could not find solution. Last calculated: {joint_pos}")
#         print(q, e)
#     return joint_pos

# def ik_as_is(pose, fk, variables, initial_guesses, joint_limits):
#     equations = fk - pose
#     qsol = solve_inverse_kinematics(equations, 
#                                     variables, 
#                                     initial_guesses,
#                                     joint_limits
#                                     )
#     return qsol


def pieper_method(H,T10,T21,T32,T43,T54,T65,variables,initial_guesses,joint_limits):
    position_equations = (T10*T21*T32*T43)[:3,3] - (H*(T54*T65).inv())[:3,3]
    qsol_position = solve_inverse_kinematics(position_equations, 
                                             variables[:3], 
                                             initial_guesses[:3],
                                             joint_limits[:3]
                                             )
    # print(qsol_position)
    R30_sol = ( T10*T21*T32 ).subs(qsol_position[0])[:3,:3]
    orientation_equations = ( R30_sol * T43[:3,:3] * T54[:3,:3] * T65[:3,:3] ) - ( H[:3,:3] )
    # R_unk = R30_sol * T43[:3,:3] * T54[:3,:3] * T65[:3,:3]
    # R_des = H[:3,:3]
    # or_eq1 = R_unk[2,2] - R_des[2,2]
    # or_eq2 = R_unk[1,2] - R_des[1,2]
    # or_eq3 = R_unk[0,2] - R_des[0,2]
    # or_eq4 = R_unk[2,1] - R_des[2,1]
    # or_eq5 = R_unk[2,0] - R_des[2,0]
    # orientation_equations = sp.Matrix([or_eq1, or_eq2, or_eq3, or_eq4, or_eq5])
    qsol_orientation = solve_inverse_kinematics(orientation_equations,
                                                variables[3:],
                                                initial_guesses[3:],
                                                joint_limits[3:]
                                                )
    return [{**qsol_position[0], **qsol_orientation[0]}]

def normalize_solution_minus_pi_to_pi(q_sol, evalf=False):
    PI = sp.ones(len(q_sol), 1) * sp.pi
    q_sol_norm = ( q_sol + PI) % (2 * sp.pi) - PI  
    if evalf:
        return q_sol_norm.evalf(evalf)
    return q_sol

def is_in_range(x, limits):
    if x >= limits[0] and x <= limits[1]:
        return True
    return False
    
def generate_random_initial_guesses(variables, limits):
    N = len(variables)
    Q0 = []
    for k in range(N):
        guess = random.uniform(limits[k][0], limits[k][1])
        Q0.append(guess)
    return Q0



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
