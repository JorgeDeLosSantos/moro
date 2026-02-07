"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
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