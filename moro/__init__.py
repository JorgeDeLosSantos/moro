"""
Numython R&D, (c) 2020

This module has been designed for academic purposes, using SymPy as base library. 
It's easy to check that SymPy is slower than NumPy specially in matrix algebra, 
however SymPy is more convenient to use as didactic tool due to the given facilities 
as the symbolic manipulation, calculation of partial and ordinary derivatives, 
matricial multiplication using asterisk symbol, "init_printing" function and so on.
"""
from .version import __version__

__author__ = "Pedro Jorge De Los Santos"

from sympy import solve, symbols, init_printing, pi, simplify
from sympy.matrices import Matrix, eye, zeros, ones
from sympy.physics.mechanics import init_vprinting
from .abc import * # To use common symbolic variables
from .core import *
from .plotting import * 
from .transformations import * 
from .util import *
# ~ from .ws import * # not yet ready
init_vprinting() # Get "pretty print" 
# vprinting for dot notation (Newton's notation)