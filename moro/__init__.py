"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
from .version import __version__

from .core import Robot
from .transformations import (
    rotx, roty, rotz,
    rot2eul, eul2rot,
    axa2rot, rot2axa,
    htmtra, htmrot,
    rot2htm, rt2htm,
    htm2rot, htm2tra,
    invhtm,
    dh
)

__all__ = [
    "__version__",
    "Robot",
    "rotx",
    "roty",
    "rotz",
    "rot2eul",
    "eul2rot",
    "axa2rot",
    "rot2axa",
    "htmtra",
    "htmrot",
    "rot2htm",
    "rt2htm",
    "htm2rot",
    "htm2tra",
    "invhtm",
    "dh"
]