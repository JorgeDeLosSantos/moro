"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import numpy as np
from sympy import lambdify, Matrix
from moro.util import is_position_vector

__all__ = ["solve_position_ik", "IKSolution"]


class IKSolution:
    """
    Represents the solution of an inverse kinematics problem.
    
    Attributes
    ----------
    q : list
        Joint variables that solve the IK problem.
    converged : bool
        Whether the solver converged to a solution.
    iterations : int
        Number of iterations performed.
    error : float
        Final position error (norm of the residual).
    method : str
        Solver method used ("newton", "lm", or "ccd").
    """
    
    def __init__(self, q, converged, iterations, error, method="lm"):
        self.q = list(q) if hasattr(q, '__iter__') else [q]
        self.converged = converged
        self.iterations = iterations
        self.error = error
        self.method = method
    
    def __repr__(self):
        status = "Converged" if self.converged else "Did not converge"
        return (
            f"IKSolution(q={self.q}, {status}, "
            f"method={self.method}, iters={self.iterations}, "
            f"error={self.error:.2e})"
        )


def solve_position_ik(
    robot,
    target_position,
    q0=None,
    joint_limits=None,
    tol=1e-6,
    max_iter=None,
    method="lm",
    damping=1.0,
    damping_scale=0.5,
):
    """
    Solve the position inverse kinematics problem using Newton-Raphson,
    Levenberg-Marquardt, or Cyclic Coordinate Descent (CCD).

    Parameters
    ----------
    robot : Robot
        A Robot instance with forward kinematics and Jacobian defined.
    target_position : list, tuple or numpy.ndarray
        Desired end-effector position :math:`[x, y, z]`.
    q0 : list or numpy.ndarray, optional
        Initial guess for joint variables. If None, a random guess 
        within joint limits is generated.
    joint_limits : list of tuples, optional
        Joint limits as ``[(q1_min, q1_max), (q2_min, q2_max), ...]``.
        If None, ``robot.joint_limits`` is used.
    tol : float, optional
        Convergence tolerance on the position error norm. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 100 for Newton and LM,
        500 for CCD.
    method : str, optional
        Solver method: ``"newton"``, ``"lm"`` (Levenberg-Marquardt), or
        ``"ccd"`` (Cyclic Coordinate Descent). Default is ``"lm"``.
    damping : float, optional
        Initial damping parameter :math:`\\lambda` for Levenberg-Marquardt.
        Only used when ``method="lm"``. Default is 1.0.
    damping_scale : float, optional
        Scaling factor for the damping parameter in Levenberg-Marquardt.
        Only used when ``method="lm"``. Default is 0.5.

    Returns
    -------
    IKSolution
        An object containing the solution joint variables, convergence 
        status, number of iterations, and final error.

    Raises
    ------
    ValueError
        If ``target_position`` is not a 3-element vector, or if an 
        unknown method is specified.

    Notes
    -----
    **Newton-Raphson** (``method="newton"``):

    .. math::

        \\mathbf{q}_{k+1} = \\mathbf{q}_k + 
        \\mathbf{J}_p^\\dagger(\\mathbf{q}_k) \\, 
        (\\mathbf{p}_d - \\mathbf{f}(\\mathbf{q}_k))

    **Levenberg-Marquardt** (``method="lm"``, default):

    .. math::

        \\mathbf{q}_{k+1} = \\mathbf{q}_k + 
        (\\mathbf{J}_p^T \\mathbf{J}_p + \\lambda^2 \\mathbf{I})^{-1}
        \\mathbf{J}_p^T \\, (\\mathbf{p}_d - \\mathbf{f}(\\mathbf{q}_k))

    **CCD** (``method="ccd"``) adjusts one joint at a time from the 
    end-effector toward the base. For each revolute joint it computes 
    the angle that rotates the end-effector toward the target in the 
    plane perpendicular to the joint axis. For prismatic joints, it 
    slides along the axis to reduce the error. CCD does not use a 
    Jacobian matrix and is robust near singularities, but converges 
    linearly.

    Examples
    --------
    >>> import moro as mr
    >>> from moro.abc import l1, l2, q1, q2
    >>> from moro.inverse_kinematics import solve_position_ik
    >>> 
    >>> # 2R planar robot
    >>> rr = mr.Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))
    >>> 
    >>> # Levenberg-Marquardt (default)
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1])
    >>> 
    >>> # Newton-Raphson
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="newton")
    >>> 
    >>> # CCD
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="ccd")
    """
    # Validate method
    if method not in ("newton", "lm", "ccd"):
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose 'newton', 'lm', or 'ccd'."
        )
    
    # Validate target position
    if not is_position_vector(target_position):
        raise ValueError(
            "target_position must be a 3-element vector (x, y, z)."
        )
    
    target = np.asarray(target_position, dtype=float).flatten()
    n = robot.dof
    
    # Use provided joint limits or fall back to robot defaults
    if joint_limits is None:
        limits = robot.joint_limits
    else:
        if len(joint_limits) != n:
            raise ValueError(
                f"Number of joint limits ({len(joint_limits)}) must match "
                f"DOF ({n})."
            )
        limits = joint_limits
    
    # Convert limits to numeric arrays
    lower_bounds = np.array([float(limits[i][0]) for i in range(n)])
    upper_bounds = np.array([float(limits[i][1]) for i in range(n)])
    
    # Default max_iter depending on method
    if max_iter is None:
        max_iter = 500 if method == "ccd" else 100
    
    # Build numerical functions
    sym_vars = tuple(robot.qs)
    fk_sym = robot.T[:3, 3]
    fk_func = lambdify(sym_vars, fk_sym, modules="numpy")
    
    # Initial guess
    if q0 is None:
        q = np.random.uniform(lower_bounds, upper_bounds)
    else:
        q = np.asarray(q0, dtype=float).flatten()
        q = np.clip(q, lower_bounds, upper_bounds)
    
    # ------------------------------------------------------------------ 
    #  CCD method (Jacobian-free)
    # ------------------------------------------------------------------ 
    if method == "ccd":
        # Precompute lambdified functions for joint positions and axes
        # r_o(i): position of {i}-Frame origin w.r.t. {0}-Frame
        ro_funcs = []
        for i in range(n + 1):  # i = 0, 1, ..., n (base to end-effector)
            sym = robot.r_o(i)
            ro_funcs.append(lambdify(sym_vars, sym, modules="numpy"))
        
        # z(i): z-axis direction of {i}-Frame in {0}-Frame
        z_funcs = []
        for i in range(n):  # i = 0, 1, ..., n-1
            sym = robot.z(i)
            z_funcs.append(lambdify(sym_vars, sym, modules="numpy"))
        
        # CCD iteration
        for iteration in range(max_iter):
            # Current end-effector position and error
            p_eff = fk_func(*q).flatten()
            error_vec = target - p_eff
            error_norm = np.linalg.norm(error_vec)
            
            # Check convergence
            if error_norm < tol:
                return IKSolution(q, converged=True,
                                  iterations=iteration + 1, error=error_norm,
                                  method=method)
            
            # Sweep from end-effector joint (n) down to base (1)
            for i in range(n, 0, -1):
                joint_idx = i - 1
                joint_type = robot.joint_type(i)
                
                # Joint origin position in base frame
                p_joint = ro_funcs[i - 1](*q).flatten()
                
                # End-effector position (updated after each joint adjustment)
                p_eff = fk_func(*q).flatten()
                
                if joint_type == "r":
                    # Revolute joint: rotate z-axis to align toward target
                    # Vectors from joint to end-effector and to target
                    r_ie = p_eff - p_joint
                    r_it = target - p_joint
                    
                    # Project onto plane perpendicular to z-axis
                    z_axis = z_funcs[i - 1](*q).flatten()
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    u_ie = r_ie - np.dot(r_ie, z_axis) * z_axis
                    u_it = r_it - np.dot(r_it, z_axis) * z_axis
                    
                    # Norms of projections
                    norm_ie = np.linalg.norm(u_ie)
                    norm_it = np.linalg.norm(u_it)
                    
                    if norm_ie > 1e-12 and norm_it > 1e-12:
                        # Normalize
                        u_ie = u_ie / norm_ie
                        u_it = u_it / norm_it
                        
                        # Angle between projected vectors
                        cos_theta = np.clip(np.dot(u_ie, u_it), -1.0, 1.0)
                        sin_theta = np.dot(z_axis, np.cross(u_ie, u_it))
                        delta = np.arctan2(sin_theta, cos_theta)
                        
                        # Update joint
                        q[joint_idx] += delta
                        q[joint_idx] = np.clip(q[joint_idx],
                                               lower_bounds[joint_idx],
                                               upper_bounds[joint_idx])
                else:
                    # Prismatic joint: slide along z-axis
                    z_axis = z_funcs[i - 1](*q).flatten()
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    # Project the error vector onto the joint axis
                    delta = np.dot(error_vec, z_axis)
                    
                    # Update joint
                    q[joint_idx] += delta
                    q[joint_idx] = np.clip(q[joint_idx],
                                           lower_bounds[joint_idx],
                                           upper_bounds[joint_idx])
        
        # Did not converge within max_iter
        p_final = fk_func(*q).flatten()
        final_error = np.linalg.norm(target - p_final)
        return IKSolution(q, converged=False,
                          iterations=max_iter, error=final_error,
                          method=method)
    
    # ------------------------------------------------------------------ 
    #  Newton-Raphson / Levenberg-Marquardt methods (Jacobian-based)
    # ------------------------------------------------------------------ 
    # Precompute Jacobian function
    J_sym = robot.J[:3, :]
    J_func = lambdify(sym_vars, J_sym, modules="numpy")
    
    # Initial error
    p_current = fk_func(*q).flatten()
    error_vec = target - p_current
    error_norm = np.linalg.norm(error_vec)
    
    lam = float(damping)
    
    def compute_step(J_p, error_vec):
        """Compute the joint update Δq for the chosen method."""
        if method == "newton":
            # Newton-Raphson: Δq = J^† · e
            if n == 3:
                try:
                    return np.linalg.solve(J_p, error_vec)
                except np.linalg.LinAlgError:
                    return np.linalg.pinv(J_p) @ error_vec
            else:
                return np.linalg.pinv(J_p) @ error_vec
        else:
            # Levenberg-Marquardt: Δq = (J^T J + λ²I)^{-1} J^T e
            H = J_p.T @ J_p
            H_reg = H + lam**2 * np.eye(n)
            try:
                return np.linalg.solve(H_reg, J_p.T @ error_vec)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(J_p) @ error_vec
    
    # Newton-Raphson / Levenberg-Marquardt iteration
    for iteration in range(max_iter):
        # Check convergence
        if error_norm < tol:
            return IKSolution(q, converged=True,
                              iterations=iteration + 1, error=error_norm,
                              method=method)
        
        # Evaluate Jacobian
        J_p = J_func(*q)
        
        # Ensure J_p is 2D (3×n)
        if J_p.ndim == 1:
            J_p = J_p.reshape(3, n)
        
        # Compute joint update
        dq = compute_step(J_p, error_vec)
        
        # Trial step
        q_trial = q + dq
        q_trial = np.clip(q_trial, lower_bounds, upper_bounds)
        
        # Evaluate trial error
        p_trial = fk_func(*q_trial).flatten()
        trial_error_vec = target - p_trial
        trial_error_norm = np.linalg.norm(trial_error_vec)
        
        if method == "lm":
            if trial_error_norm < error_norm:
                # Step accepted: reduce damping
                lam *= damping_scale
                q = q_trial
                error_vec = trial_error_vec
                error_norm = trial_error_norm
            else:
                # Step rejected: increase damping
                lam /= damping_scale
        else:
            # Newton-Raphson: always accept the step
            q = q_trial
            error_vec = trial_error_vec
            error_norm = trial_error_norm
    
    # Did not converge within max_iter
    p_final = fk_func(*q).flatten()
    final_error = np.linalg.norm(target - p_final)
    
    return IKSolution(q, converged=False,
                      iterations=max_iter, error=final_error,
                      method=method)