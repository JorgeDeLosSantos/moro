"""

Numython R&D, (c) 2024
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 

"""
import sympy as sp
import random

def solve_inverse_kinematics(equations,
                             variables,
                             initial_guesses,
                             joint_limits,
                             method="nsolve",
                             max_steps=10):
    current_step = 1
    try:
        solution = nsolve(equations, variables, initial_guesses, method)
    except ValueError:
        initial_guesses = generate_random_initial_guesses(variables, joint_limits)
        solution = nsolve(equations, variables, initial_guesses, method)
    while current_step <= max_steps:
        no_sol = 0
        if len(solution) == 0:
            no_sol += 1
        for k in range(len(solution[0])):
            if not(is_in_range(solution[0][variables[k]], joint_limits[k])):
                no_sol += 1
        if no_sol > 0:
            try:
                initial_guesses = generate_random_initial_guesses(variables, joint_limits)
                solution = nsolve(equations, variables, initial_guesses, method)
            except ValueError:
                pass # skip current step
        else:
            break
        current_step += 1
    if current_step > max_steps:
        raise ValueError("Could not find solution within given limits.")
    return solution, current_step

def nsolve(equations,variables,initial_guesses,method):
    if method=="nsolve":
        return sp.nsolve(equations, variables, initial_guesses, dict=True)
    else:
        return gradient_descent(equations, variables, initial_guesses)

def gradient_descent(equations,variables,initial_guesses,eps=1e-8):
    J = equations.jacobian(variables)
    # print(J)
    joint_pos = dict( zip(variables, initial_guesses) ) # joint pos
    q = sp.Matrix(initial_guesses)
    e = equations.subs(joint_pos)
    beta = 0.01
    k = 0
    while e.norm() > eps:
        JN = J.subs( joint_pos )
        Jinv = JN.pinv()
        De = beta*-e
        Dq = Jinv*De
        q = q + Dq
        joint_pos = dict( zip(variables, q) ) # updating joint positions
        e = equations.subs(joint_pos)
        k += 1
        if k > 10:
            raise ValueError(f"Could not find solution. Last calculated: {joint_pos}")
        print(q, e)
    return joint_pos

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