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
        Solver method used ("newton" or "lm").
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
    max_iter=100,
    method="lm",
    damping=1.0,
    damping_scale=0.5,
):
    """
    Solve the position inverse kinematics problem using either the 
    Newton-Raphson or Levenberg-Marquardt method.

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
        Maximum number of iterations. Default is 100.
    method : str, optional
        Solver method: ``"newton"`` for Newton-Raphson or ``"lm"`` for 
        Levenberg-Marquardt. Default is ``"lm"``.
    damping : float, optional
        Initial damping parameter :math:`\\lambda` for Levenberg-Marquardt.
        Only used when ``method="lm"``. Default is 1.0.
    damping_scale : float, optional
        Scaling factor for the damping parameter in Levenberg-Marquardt.
        When an iteration reduces the error, :math:`\\lambda` is multiplied 
        by ``damping_scale``. When the error increases, :math:`\\lambda` 
        is divided by ``damping_scale``.
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
    **Newton-Raphson** updates the joint variables as:

    .. math::

        \\mathbf{q}_{k+1} = \\mathbf{q}_k + 
        \\mathbf{J}_p^\\dagger(\\mathbf{q}_k) \\, 
        (\\mathbf{p}_d - \\mathbf{f}(\\mathbf{q}_k))

    **Levenberg-Marquardt** (default) uses a damped least-squares approach:

    .. math::

        \\mathbf{q}_{k+1} = \\mathbf{q}_k + 
        (\\mathbf{J}_p^T \\mathbf{J}_p + \\lambda^2 \\mathbf{I})^{-1}
        \\mathbf{J}_p^T \\, (\\mathbf{p}_d - \\mathbf{f}(\\mathbf{q}_k))

    The damping parameter :math:`\\lambda` is adapted at each iteration:
    it decreases when the error is reduced (more trust in the Gauss-Newton 
    step) and increases when the error grows (more regularization needed).

    Examples
    --------
    >>> import moro as mr
    >>> from moro.abc import l1, l2, q1, q2
    >>> from moro.inverse_kinematics import solve_position_ik
    >>> 
    >>> # 2R planar robot
    >>> rr = mr.Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))
    >>> 
    >>> # Solve IK using Levenberg-Marquardt (default)
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1])
    >>> sol.converged
    True
    >>> 
    >>> # Solve IK using Newton-Raphson
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1], 
    ...                         method="newton")
    >>> sol.converged
    True
    """
    # Validate method
    if method not in ("newton", "lm"):
        raise ValueError(
            f"Unknown method '{method}'. Choose 'newton' or 'lm'."
        )
    
    # Validate target position
    if not is_position_vector(target_position):
        raise ValueError(
            "target_position must be a 3-element vector (x, y, z)."
        )
    
    target = np.asarray(target_position, dtype=float).reshape(3, 1)
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
    
    # Build numerical functions for forward kinematics and Jacobian
    fk_sym = robot.T[:3, 3]
    J_sym = robot.J[:3, :]
    sym_vars = tuple(robot.qs)
    
    # Lambdify for fast numerical evaluation
    fk_func = lambdify(sym_vars, fk_sym, modules="numpy")
    J_func = lambdify(sym_vars, J_sym, modules="numpy")
    
    # Initial guess
    if q0 is None:
        q = np.random.uniform(lower_bounds, upper_bounds)
    else:
        q = np.asarray(q0, dtype=float).flatten()
        q = np.clip(q, lower_bounds, upper_bounds)
    
    # Levenberg-Marquardt state
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
            # Regularize
            H_reg = H + lam**2 * np.eye(n)
            try:
                return np.linalg.solve(H_reg, J_p.T @ error_vec)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular even with damping
                return np.linalg.pinv(J_p) @ error_vec
    
    # Evaluate initial error
    p_current = fk_func(*q).flatten()
    error_vec = target.flatten() - p_current
    error_norm = np.linalg.norm(error_vec)
    best_error = error_norm
    
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
        trial_error_vec = target.flatten() - p_trial
        trial_error_norm = np.linalg.norm(trial_error_vec)
        
        if method == "lm":
            # Levenberg-Marquardt adaptive damping
            if trial_error_norm < error_norm:
                # Step accepted: reduce damping (more trust in Gauss-Newton)
                lam *= damping_scale
                q = q_trial
                error_vec = trial_error_vec
                error_norm = trial_error_norm
            else:
                # Step rejected: increase damping (more regularization)
                lam /= damping_scale
                # Keep q unchanged, re-evaluate error at current q
                # error_vec and error_norm stay the same
                # (already evaluated above)
        else:
            # Newton-Raphson: always accept the step
            q = q_trial
            error_vec = trial_error_vec
            error_norm = trial_error_norm
    
    # Did not converge within max_iter
    # Compute final error for reporting (re-evaluate at final q)
    p_final = fk_func(*q).flatten()
    final_error = np.linalg.norm(target.flatten() - p_final)
    
    return IKSolution(q, converged=False,
                      iterations=max_iter, error=final_error,
                      method=method)