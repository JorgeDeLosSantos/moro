from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols

t = symbols("t") # time 
q1,q2,q3,q4,q5,q6 = dynamicsymbols('q_1:7') # joint variables
qd1,qd2,qd3,qd4,qd5,qd6 = [q.diff(t) for q in (q1,q2,q3,q4,q5,q6)] # joint velocities
qdd1,qdd2,qdd3,qdd4,qdd5,qdd6 = [qd.diff(t) for qd in (qd1,qd2,qd3,qd4,qd5,qd6)] # joint accelerations
theta1,theta2,theta3,theta4,theta5,theta6 = symbols('theta_1:7', real=True) # theta parameter
l1,l2,l3,l4,l5,l6 = symbols('l_1:7', real=True) # parameters for constant lengths of the robot
lc1,lc2,lc3,lc4,lc5,lc6 = symbols('l_c1:7', real=True) # parameters for center of mass location
d1,d2,d3,d4,d5,d6 = symbols('d_1:7', real=True)  # for constant lengths of the robot
m1,m2,m3,m4,m5,m6 = symbols('m_1:7', real=True) # for mass symbols
a1,a2,a3,a4,a5,a6 = symbols('a_1:7', real=True) # for constant lengths of the robot
alpha1,alpha2,alpha3,alpha4,alpha5,alpha6 = symbols('alpha_1:7', real=True) # for DH parameters
g = symbols("g") # gravity
tau1,tau2,tau3,tau4,tau5,tau6 = symbols("tau_1:7", real=True) # for generalized forces/torques

# Some common greek letters 
alpha,beta,gamma,delta = symbols("alpha,beta,gamma,delta", real=True)
epsilon,zeta,eta,theta = symbols("epsilon,zeta,eta,theta", real=True)
iota,kappa,mu,nu = symbols("iota,kappa,mu,nu", real=True)
xi,omicron,rho,sigma = symbols("xi,omicron,rho,sigma", real=True)
tau,upsilon,phi,chi = symbols("tau,upsilon,phi,chi", real=True)
psi,omega = symbols("psi,omega", real=True)

# ~ del pi # Delete "pi" symbolic variable -> conflict with pi number
available_symvars = [g,t,
q1,q2,q3,q4,q5,q6,
qd1,qd2,qd3,qd4,qd5,qd6,
qdd1,qdd2,qdd3,qdd4,qdd5,qdd6,
theta1,theta2,theta3,theta4,theta5,theta6,
alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,
l1,l2,l3,l4,l5,l6,
lc1,lc2,lc3,lc4,lc5,lc6,
d1,d2,d3,d4,d5,d6,
m1,m2,m3,m4,m5,m6,
a1,a2,a3,a4,a5,a6,
alpha,beta,gamma,delta,
epsilon,zeta,eta,theta,
iota,kappa,mu,nu,
xi,omicron,rho,sigma,
tau,upsilon,phi,chi,
psi,omega]