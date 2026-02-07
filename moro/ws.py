"""

"""
# ~ from sympy import *
# ~ from sympy.matrices import Matrix,eye
# ~ from itertools import combinations
from scipy.spatial import Delaunay, ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from moro.core import *

__all__ = [""]


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib, True)
            add_edge(edges, ib, ic, True)
            add_edge(edges, ic, ia, True)
            
    return edges


def add_edge(edges, i, j, only_outer):
    """
    Add an edge between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        assert (j, i) in edges, "Can't go twice over same directed edge right?"
        if only_outer:
            # if both neighboring triangles are in shape, it's not a boundary edge
            edges.remove((j, i))
        return
    edges.add((i, j))


def RR_Example():
    from numpy import sin,cos
    l1,l2 = 100,50
    X, Y = [],[]
    for t1 in np.linspace(0,np.pi):
        for t2 in np.linspace(-np.pi/2,np.pi/2):
            X.append( l1*cos(t1) + l2*cos(t1+t2) )
            Y.append( l1*sin(t1) + l2*sin(t1+t2) )
    
    points = np.array( list(zip(X,Y)) )
    # Computing the alpha shape
    edges = alpha_shape(points, alpha=5, only_outer=True)
    # Plotting the output
    plt.figure()
    plt.axis('equal')
    # ~ plt.plot(points[:, 0], points[:, 1], '.')
    xc,yc = [],[]
    for i, j in edges:
        plt.plot(points[[i, j], 0], points[[i, j], 1])
    plt.show()


if __name__=="__main__":
    RR_Example()
