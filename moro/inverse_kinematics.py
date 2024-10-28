"""

Numython R&D, (c) 2024
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 

"""
import sympy as sp

def solve_inverse_kinematics():
    pass

def nsolve():
    pass

def normalize_solution_minus_pi_to_pi(q_sol, evalf=False):
    PI = sp.ones(len(q_sol), 1) * sp.pi
    q_sol_norm = ( q_sol + PI) % (2 * sp.pi) - PI  
    if evalf:
        return q_sol_norm.evalf(evalf)
    return q_sol


