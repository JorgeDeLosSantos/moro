"""
Numython R&D, (c) 2025
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
from .version import __version__

__author__ = "Pedro Jorge De Los Santos"

from .abc import * # To use common symbolic variables
from .core import *
from .plotting import * 
from .transformations import * 
from .util import *

# import sympy functions
from sympy import (solve, symbols, pi, 
                   simplify, nsimplify, 
                   trigsimp, expand, factor)
from sympy import sin,cos,tan,sqrt,atan2
from sympy.matrices import Matrix, eye, zeros, ones
from sympy.physics.mechanics import init_vprinting

# ~ from .ws import * # not yet ready
init_vprinting() # Get "pretty print" 
# vprinting for dot notation (Newton's notation)