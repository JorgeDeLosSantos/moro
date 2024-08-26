"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.matrices import Matrix,eye
from moro.transformations import *
from moro.util import *

__all__ = ["plot_euler", "draw_uv", "draw_uvw"]

def plot_euler(phi,theta,psi,seq="zxz"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if seq in ("zxz","ZXZ","313",313):
        R1 = rotz(phi)
        R2 = R1*rotx(theta)
        R3 = R2*rotz(psi)
    elif seq in ("zyz","ZYZ","323",323):
        R1 = rotz(phi)
        R2 = R1*roty(theta)
        R3 = R2*rotz(psi)
    else:
        R1 = R2 = R3 = eye(4)
    draw_uvw(eye(4), ax, sz=6, alpha=0.4)
    draw_uvw(R1, ax, sz=6, alpha=0.6)
    draw_uvw(R2, ax, sz=6, alpha=0.8)
    draw_uvw(R3, ax, sz=6, alpha=1.0)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    # ax.set_aspect("equal")
    ax.axis('off')

    
def draw_uvw(H,ax,color=("r","g","b"),sz=1,alpha=1.0):
    u = H[:3,0]
    v = H[:3,1]
    w = H[:3,2]
    if ishtm(H):
        o = H[:3,3]
    else:
        o = Matrix([0,0,0])
    L = sz/5
    if isinstance(color,str):
        colorl = (color,color,color)
    else:
        colorl = color
    ax.quiver(o[0],o[1],o[2],u[0],u[1],u[2], color=colorl[0], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)
    ax.quiver(o[0],o[1],o[2],v[0],v[1],v[2], color=colorl[1], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)
    ax.quiver(o[0],o[1],o[2],w[0],w[1],w[2], color=colorl[2], 
              length=L, arrow_length_ratio=0.2, alpha=alpha)


def draw_xyz(*args, **kwargs):
    return draw_uvw(*args, **kwargs)

def draw_frame(*args, **kwargs):
    return draw_uvw(*args, **kwargs)

def draw_uv(H, ax, name="S0", color=("r","g"), sz=1):
    tpos = H*Matrix([1,1,0,1])
    H = sympy2float(H)
    u = H[:3,0]             
    v = H[:3,1]
    w = H[:3,2]
    if ishtm(H):
        o = H[:3,3]
    else:
        o = Matrix([0,0,0])
    L = sz/5
    if isinstance(color,str):
        colorl = (color,color)
    else:
        colorl = color
    # ~ print(o, u)
    ax.arrow(o[0],o[1],u[0],u[1], color=colorl[0])
    ax.arrow(o[0],o[1],v[0],v[1], color=colorl[1])
    ax.text(tpos[0], tpos[1], "{"+name+"}", fontsize=8)
    ax.set_aspect("auto")


if __name__=="__main__":
    plot_euler(pi/3, pi/3, 0.5)
    plt.show()
    # ~ fig = plt.figure()
    # ~ ax = fig.add_subplot(111)
    # ~ H1 = eye(4)*htmrot(pi/3)
    # ~ H2 = H1*htmtra([10,5,0])
    # ~ H3 = H2*htmtra([-4,5,0])*htmrot(pi/4)
    # ~ draw_uv(H1, ax, "A", "b")
    # ~ draw_uv(H2, ax, "B")
    # ~ draw_uv(H3, ax, "C")
    # ~ plt.grid(ls="--")
    # ~ plt.axis([-20,20,-20,20])
    # ~ plt.show()
