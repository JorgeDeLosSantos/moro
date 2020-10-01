"""

"""
from moro.core import Robot
from sympy import sin,cos,tan,atan2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from moro.ws import alpha_shape

__all__ = ["RR_Planar_Robot", ]


class RR_Planar_Robot(Robot):
    def __init__(self,L1,L2,q=[0,0]):
        self.L1 = L1
        self.L2 = L2
        self.q = q
        self.dof = 2
        # ----------------

    @property
    def type(self):
        self._type = ["r", "r"]
        return self._type

    @property
    def Ts(self):
        self._Ts = [dh(self.L1,0,0,self.q[0]), dh(self.L2,0,0,self.q[1])]
        return self._Ts

    @property
    def ex(self):
        q1, q2 = self.q
        L1, L2 = self.L1, self.L2
        self._ex = L1*cos(q1) + L2*cos(q1 + q2)
        return self._ex

    @property
    def ey(self):
        q1, q2 = self.q
        L1, L2 = self.L1, self.L2
        self._ey = L1*sin(q1) + L2*sin(q1 + q2)
        return self._ey

    def fkine(self):
        return self.ex, self.ey

    def ikine(self, pe, maxiter=100):
        goal = Matrix([pe[0], pe[1], 0, 0, 0, 0])
        e = Matrix([self.ex, self.ey, 0, 0, 0, 0])
        Phi = Matrix([self.q[0], self.q[1]])
        eps = 1e-6
        beta = 0.75
        k = 0
        # ~ clr = cm.gray(np.linspace(0,1,maxiter))
        while (goal - e).norm() > eps:
            J = self.J
            JI = self.J.pinv()
            De = beta*(goal - e)
            Dphi = JI * De
            Phi += Dphi
            # ~ self.draw(ls="--", color=clr[k])
            self.q = [Phi[0], Phi[1]]
            e[0],e[1] = self.fkine()
            k += 1
            if k == maxiter:
                print("Max. iterations has been reached")
                break
        q1,q2 = self.q
        return q1, q2
        
    def draw(self, **kwargs):
        q1,q2 = self.q
        L1,L2 = self.L1, self.L2
        O = [0, 0]
        O1 = [L1*cos(q1), L1*sin(q1)]
        O2 = [L1*cos(q1) + L2*cos(q1+q2), L1*sin(q1) + L2*sin(q1+q2)]
        plt.plot([O[0], O1[0], O2[0]], [O[1], O1[1], O2[1] ], **kwargs)
        plt.plot([O2[0] ],[O2[1] ], "ro")
        plt.grid(ls="--")
        # ~ self.show()
        
    def show(self):
        plt.show()
        
    def plot_workspace(self,q1r,q2r,n=20,**kwargs):
        L1,L2 = self.L1, self.L2
        q1min, q1max = q1r
        q2min, q2max = q2r
        EX = []
        EY = []
        for q1 in np.linspace(q1min, q1max, n):
            for q2 in np.linspace(q2min, q2max, n):
                self.q = [q1,q2]
                ex, ey = self.fkine()
                ex, ey = float(ex), float(ey)
                EX.append(ex), EY.append(ey)
                
        # Computing the alpha shape
        points = np.column_stack((EX,EY))
        k_alpha = self.calculate_alpha(points)
        edges = alpha_shape(points, alpha=k_alpha, only_outer=True)

        # Plotting the output
        # ~ plt.figure()
        # ~ plt.axis('equal')
        plt.plot(points[:, 0], points[:, 1], '.', color="#dadada")
        for i, j in edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1], **kwargs)
        # ~ plt.show()
        # ~ return points
        
        
    def calculate_alpha(self,points):
        mx = 0.1*np.max(points)
        return mx
        

if __name__=="__main__":
    r = RR_Planar_Robot(120,150)
    r.plot_workspace([0,np.pi], [0, np.pi/2], 50, color="r")
    r1 = RR_Planar_Robot(200,100)
    r1.plot_workspace([0,np.pi/2], [-np.pi/2, np.pi/2], 50, color="b" )
    r.show()
    
