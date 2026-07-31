"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import numpy as np
from numbers import Integral, Real
from sympy import lambdify
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


def _is_finite_array(arr):
    """Return True when all values in arr are finite floats."""
    return np.all(np.isfinite(np.asarray(arr, dtype=float)))


def _as_finite_vector(value, size, name):
    """Convert input to a finite 1D float vector of a given size."""
    try:
        vec = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a {size}-element vector with real numeric values."
        ) from exc

    if vec.size != size:
        raise ValueError(
            f"{name} must contain exactly {size} values. "
            f"Expected size {size}, got {vec.size}."
        )

    if not _is_finite_array(vec):
        raise ValueError(f"{name} must contain only finite values.")

    return vec


def _validate_solver_options(method, tol, max_iter, damping, damping_scale):
    """Validate solver method and scalar options."""
    if method not in ("newton", "lm", "ccd"):
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose 'newton', 'lm', or 'ccd'."
        )

    if not isinstance(tol, Real) or isinstance(tol, bool) or not np.isfinite(float(tol)):
        raise ValueError("tol must be a finite real number.")
    if float(tol) <= 0:
        raise ValueError("tol must satisfy tol > 0.")

    if max_iter is None:
        max_iter = 500 if method == "ccd" else 100
    else:
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be a positive integer.")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must satisfy max_iter > 0.")
        max_iter = int(max_iter)

    if not isinstance(damping, Real) or isinstance(damping, bool) or not np.isfinite(float(damping)):
        raise ValueError("damping must be a finite real number.")
    if float(damping) <= 0:
        raise ValueError("damping must satisfy damping > 0.")

    if (
        not isinstance(damping_scale, Real)
        or isinstance(damping_scale, bool)
        or not np.isfinite(float(damping_scale))
    ):
        raise ValueError("damping_scale must be a finite real number.")
    if not (0 < float(damping_scale) < 1):
        raise ValueError("damping_scale must satisfy 0 < damping_scale < 1.")

    return float(tol), max_iter, float(damping), float(damping_scale)


def _prepare_joint_limits(robot, joint_limits):
    """Validate and return lower/upper joint limits as finite float arrays."""
    n = robot.dof
    limits = robot.joint_limits if joint_limits is None else joint_limits

    try:
        count = len(limits)
    except TypeError as exc:
        raise ValueError(
            "joint_limits must be a sequence with one (lower, upper) pair per joint."
        ) from exc

    if count != n:
        raise ValueError(
            "Joint limits must provide exactly one (lower, upper) pair per DOF. "
            f"Expected {n}, got {count}."
        )

    lower = []
    upper = []
    for idx, lim in enumerate(limits):
        try:
            lim_len = len(lim)
        except TypeError as exc:
            raise ValueError(
                f"Joint limit at index {idx} must be a sequence of two elements "
                "(lower, upper)."
            ) from exc

        if lim_len != 2:
            raise ValueError(
                f"Joint limit at index {idx} must contain exactly two elements."
            )

        try:
            low = float(lim[0])
            up = float(lim[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Joint limit at index {idx} must contain real numeric values."
            ) from exc

        if not np.isfinite(low) or not np.isfinite(up):
            raise ValueError(
                f"Joint limit at index {idx} must contain finite values."
            )

        if low > up:
            raise ValueError(
                f"Invalid joint limit at index {idx}: lower bound {low} "
                f"is greater than upper bound {up}."
            )

        lower.append(low)
        upper.append(up)

    lower_bounds = np.asarray(lower, dtype=float).reshape(-1)
    upper_bounds = np.asarray(upper, dtype=float).reshape(-1)

    if lower_bounds.shape != (n,) or upper_bounds.shape != (n,):
        raise ValueError(
            "Joint limits could not be converted to arrays with shape (robot.dof,)."
        )

    return lower_bounds, upper_bounds


def _prepare_initial_guess(q0, n, lower_bounds, upper_bounds):
    """Validate and clip the initial guess."""
    if q0 is None:
        return np.random.uniform(lower_bounds, upper_bounds)

    try:
        q = np.asarray(q0, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("q0 must be a one-dimensional numeric vector.") from exc

    if q.size != n:
        raise ValueError(
            f"q0 must contain exactly {n} values. Expected size {n}, got {q.size}."
        )

    if not _is_finite_array(q):
        raise ValueError("q0 must contain only finite values.")

    return np.clip(q, lower_bounds, upper_bounds)


def _apply_parameters(expr, parameters):
    """Apply symbolic substitutions to a local expression copy."""
    if parameters is None:
        return expr

    try:
        return expr.subs(parameters)
    except Exception as exc:
        raise ValueError(
            "parameters must be None or a dictionary-like substitution mapping "
            "compatible with sympy.Expr.subs."
        ) from exc


def _validate_free_symbols(expressions, joint_symbols):
    """Ensure only joint symbols remain free after parameter substitution."""
    joint_set = set(joint_symbols)
    allowed_aux_symbols = set()
    for q_sym in joint_symbols:
        allowed_aux_symbols.update(getattr(q_sym, "free_symbols", set()))

    missing = set()
    for expr in expressions:
        missing.update(
            sym
            for sym in expr.free_symbols
            if sym not in joint_set and sym not in allowed_aux_symbols
        )

    if missing:
        missing_str = ", ".join(str(sym) for sym in sorted(missing, key=str))
        raise ValueError(
            "Cannot build numerical IK functions because the following symbols "
            f"have no numerical value: {missing_str}. Provide them using the "
            "'parameters' argument."
        )


def _safe_eval_vector(func, q, size):
    """Evaluate a numerical function and return a finite 1D vector or None."""
    try:
        value = np.asarray(func(*q), dtype=float).reshape(-1)
    except Exception:
        return None

    if value.size != size or not _is_finite_array(value):
        return None

    return value


def _safe_eval_jacobian(func, q, n):
    """Evaluate Jacobian and return a finite (3, n) array or None."""
    try:
        j_val = np.asarray(func(*q), dtype=float)
    except Exception:
        return None

    if j_val.ndim == 1:
        if j_val.size != 3 * n:
            return None
        j_val = j_val.reshape(3, n)

    if j_val.shape != (3, n) or not _is_finite_array(j_val):
        return None

    return j_val


def _failure_solution(q, iterations, method):
    """Build a controlled non-converged solution with finite configuration."""
    q_safe = np.asarray(q, dtype=float).reshape(-1)
    if not _is_finite_array(q_safe):
        q_safe = np.zeros_like(q_safe)
    return IKSolution(q_safe, converged=False, iterations=iterations, error=np.inf, method=method)


def _solve_newton_or_lm(
    fk_func,
    j_func,
    target,
    q,
    lower_bounds,
    upper_bounds,
    tol,
    max_iter,
    method,
    damping,
    damping_scale,
):
    """Solve position IK using Newton-Raphson or Levenberg-Marquardt."""
    n = q.size
    completed_steps = 0

    p_current = _safe_eval_vector(fk_func, q, 3)
    if p_current is None:
        raise ValueError(
            "Cannot evaluate a finite end-effector position with the provided "
            "inputs. Check target_position, q0, and parameters."
        )

    error_vec = target - p_current
    if not _is_finite_array(error_vec):
        raise ValueError(
            "Initial position error is not finite. Check target_position and parameters."
        )

    error_norm = np.linalg.norm(error_vec)
    if not np.isfinite(error_norm):
        raise ValueError(
            "Initial position error norm is not finite. Check target_position and parameters."
        )

    if error_norm < tol:
        return IKSolution(q, converged=True, iterations=0, error=error_norm, method=method)

    lam = damping

    for _ in range(max_iter):
        j_val = _safe_eval_jacobian(j_func, q, n)
        if j_val is None:
            return _failure_solution(q, completed_steps, method)

        if method == "newton":
            if n == 3:
                try:
                    dq = np.linalg.solve(j_val, error_vec)
                except np.linalg.LinAlgError:
                    dq = np.linalg.pinv(j_val) @ error_vec
            else:
                dq = np.linalg.pinv(j_val) @ error_vec
        else:
            hessian = j_val.T @ j_val
            h_reg = hessian + lam**2 * np.eye(n)
            try:
                dq = np.linalg.solve(h_reg, j_val.T @ error_vec)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(j_val) @ error_vec

        dq = np.asarray(dq, dtype=float).reshape(-1)
        if dq.size != n or not _is_finite_array(dq):
            return _failure_solution(q, completed_steps, method)

        q_trial = np.clip(q + dq, lower_bounds, upper_bounds)
        if not _is_finite_array(q_trial):
            return _failure_solution(q, completed_steps, method)

        p_trial = _safe_eval_vector(fk_func, q_trial, 3)
        if p_trial is None:
            return _failure_solution(q, completed_steps, method)

        trial_error_vec = target - p_trial
        if not _is_finite_array(trial_error_vec):
            return _failure_solution(q, completed_steps, method)

        trial_error_norm = np.linalg.norm(trial_error_vec)
        if not np.isfinite(trial_error_norm):
            return _failure_solution(q, completed_steps, method)

        if method == "lm":
            # Rejected LM attempts still count as algorithm iterations.
            if trial_error_norm < error_norm:
                lam *= damping_scale
                q = q_trial
                error_vec = trial_error_vec
                error_norm = trial_error_norm
            else:
                lam /= damping_scale

            if not np.isfinite(lam) or lam <= 0:
                return _failure_solution(q, completed_steps, method)

            completed_steps += 1
        else:
            q = q_trial
            error_vec = trial_error_vec
            error_norm = trial_error_norm
            completed_steps += 1

        if error_norm < tol:
            return IKSolution(q, converged=True, iterations=completed_steps, error=error_norm, method=method)

    final_error = error_norm if np.isfinite(error_norm) else np.inf
    return IKSolution(q, converged=False, iterations=max_iter, error=final_error, method=method)


def _solve_ccd(
    robot,
    fk_func,
    ro_funcs,
    z_funcs,
    target,
    q,
    lower_bounds,
    upper_bounds,
    tol,
    max_iter,
):
    """Solve position IK using CCD with robust finite-value checks."""
    n = q.size
    completed_steps = 0
    method = "ccd"

    p_current = _safe_eval_vector(fk_func, q, 3)
    if p_current is None:
        raise ValueError(
            "Cannot evaluate a finite end-effector position with the provided "
            "inputs. Check target_position, q0, and parameters."
        )

    init_error_vec = target - p_current
    if not _is_finite_array(init_error_vec):
        raise ValueError(
            "Initial position error is not finite. Check target_position and parameters."
        )

    init_error_norm = np.linalg.norm(init_error_vec)
    if not np.isfinite(init_error_norm):
        raise ValueError(
            "Initial position error norm is not finite. Check target_position and parameters."
        )

    if init_error_norm < tol:
        return IKSolution(q, converged=True, iterations=0, error=init_error_norm, method=method)

    for _ in range(max_iter):
        p_eff = _safe_eval_vector(fk_func, q, 3)
        if p_eff is None:
            return _failure_solution(q, completed_steps, method)

        error_vec = target - p_eff
        if not _is_finite_array(error_vec):
            return _failure_solution(q, completed_steps, method)

        error_norm = np.linalg.norm(error_vec)
        if not np.isfinite(error_norm):
            return _failure_solution(q, completed_steps, method)

        if error_norm < tol:
            return IKSolution(q, converged=True, iterations=completed_steps, error=error_norm, method=method)

        # One CCD global step is a full sweep from joint n down to 1.
        for i in range(n, 0, -1):
            joint_idx = i - 1
            joint_type = robot.joint_type(i)

            p_joint = _safe_eval_vector(ro_funcs[joint_idx], q, 3)
            if p_joint is None:
                return _failure_solution(q, completed_steps, method)

            z_axis = _safe_eval_vector(z_funcs[joint_idx], q, 3)
            if z_axis is None:
                return _failure_solution(q, completed_steps, method)

            z_norm = np.linalg.norm(z_axis)
            if not np.isfinite(z_norm) or z_norm <= 1e-12:
                return _failure_solution(q, completed_steps, method)
            z_axis = z_axis / z_norm

            if joint_type == "r":
                p_eff = _safe_eval_vector(fk_func, q, 3)
                if p_eff is None:
                    return _failure_solution(q, completed_steps, method)

                r_ie = p_eff - p_joint
                r_it = target - p_joint

                if not _is_finite_array(r_ie) or not _is_finite_array(r_it):
                    return _failure_solution(q, completed_steps, method)

                u_ie = r_ie - np.dot(r_ie, z_axis) * z_axis
                u_it = r_it - np.dot(r_it, z_axis) * z_axis

                norm_ie = np.linalg.norm(u_ie)
                norm_it = np.linalg.norm(u_it)

                if (
                    not np.isfinite(norm_ie)
                    or not np.isfinite(norm_it)
                    or norm_ie <= 1e-12
                    or norm_it <= 1e-12
                ):
                    continue

                u_ie = u_ie / norm_ie
                u_it = u_it / norm_it

                cos_theta = np.clip(np.dot(u_ie, u_it), -1.0, 1.0)
                sin_theta = np.dot(z_axis, np.cross(u_ie, u_it))
                delta = np.arctan2(sin_theta, cos_theta)

                if not np.isfinite(delta):
                    return _failure_solution(q, completed_steps, method)

                q[joint_idx] = np.clip(
                    q[joint_idx] + delta,
                    lower_bounds[joint_idx],
                    upper_bounds[joint_idx],
                )
            else:
                # Recompute current error right before prismatic update to avoid
                # using stale values from previous joint updates in the same sweep.
                p_eff = _safe_eval_vector(fk_func, q, 3)
                if p_eff is None:
                    return _failure_solution(q, completed_steps, method)

                error_vec = target - p_eff
                if not _is_finite_array(error_vec):
                    return _failure_solution(q, completed_steps, method)

                delta = np.dot(error_vec, z_axis)
                if not np.isfinite(delta):
                    return _failure_solution(q, completed_steps, method)

                q[joint_idx] = np.clip(
                    q[joint_idx] + delta,
                    lower_bounds[joint_idx],
                    upper_bounds[joint_idx],
                )

            if not np.isfinite(q[joint_idx]):
                return _failure_solution(q, completed_steps, method)

        completed_steps += 1

        p_after = _safe_eval_vector(fk_func, q, 3)
        if p_after is None:
            return _failure_solution(q, completed_steps, method)

        error_after = target - p_after
        if not _is_finite_array(error_after):
            return _failure_solution(q, completed_steps, method)

        error_after_norm = np.linalg.norm(error_after)
        if not np.isfinite(error_after_norm):
            return _failure_solution(q, completed_steps, method)

        if error_after_norm < tol:
            return IKSolution(q, converged=True, iterations=completed_steps, error=error_after_norm, method=method)

    p_final = _safe_eval_vector(fk_func, q, 3)
    if p_final is None:
        return _failure_solution(q, max_iter, method)

    final_error_vec = target - p_final
    final_error = np.linalg.norm(final_error_vec)
    if not np.isfinite(final_error):
        final_error = np.inf

    return IKSolution(q, converged=False, iterations=max_iter, error=final_error, method=method)


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
    *,
    parameters=None,
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
    parameters : dict-like, optional
        Symbol substitutions for geometric constants and other non-joint
        symbols used by IK expressions. This mapping is applied locally using
        SymPy ``subs`` and does not modify ``robot`` or its cached expressions.

    Returns
    -------
    IKSolution
        An object containing the solution joint variables, convergence 
        status, number of iterations, and final error.

    Raises
    ------
    ValueError
        If inputs are invalid, if unresolved non-joint symbols remain in IK
        expressions, or if finite numeric functions cannot be built.

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

    ``tol`` is expressed in the same linear units as the robot DH parameters
    and ``target_position``.

    Joint limits are validated and used to clip every joint update. If ``q0``
    is provided, it is validated and clipped to joint limits before iterations.

    ``iterations`` counts completed global algorithm steps: Newton/LM count
    one per attempted update; CCD counts one per full sweep from joint n to 1.
    Therefore, if the initial guess already satisfies the tolerance,
    ``iterations`` is 0.

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
    >>> sol = solve_position_ik(
    ...     rr,
    ...     [1.5, 0.5, 0.0],
    ...     q0=[0.1, 0.1],
    ...     parameters={l1: 1.0, l2: 1.0},
    ... )
    >>> 
    >>> # Newton-Raphson
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="newton")
    >>> 
    >>> # CCD
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="ccd")
    """
    tol, max_iter, damping, damping_scale = _validate_solver_options(
        method, tol, max_iter, damping, damping_scale
    )

    # Validate target position shape and then enforce finite numeric values.
    if not is_position_vector(target_position):
        raise ValueError(
            "target_position must be a 3-element vector (x, y, z)."
        )

    target = _as_finite_vector(target_position, 3, "target_position")
    n = robot.dof

    lower_bounds, upper_bounds = _prepare_joint_limits(robot, joint_limits)

    sym_vars = tuple(robot.qs)
    fk_sym = _apply_parameters(robot.T[:3, 3], parameters)

    if method == "ccd":
        ro_syms = [_apply_parameters(robot.r_o(i), parameters) for i in range(n + 1)]
        z_syms = [_apply_parameters(robot.z(i), parameters) for i in range(n)]
        _validate_free_symbols([fk_sym] + ro_syms + z_syms, sym_vars)
    else:
        j_sym = _apply_parameters(robot.J[:3, :], parameters)
        _validate_free_symbols([fk_sym, j_sym], sym_vars)

    fk_func = lambdify(sym_vars, fk_sym, modules="numpy")

    q = _prepare_initial_guess(q0, n, lower_bounds, upper_bounds)

    if method == "ccd":
        ro_funcs = [lambdify(sym_vars, ro_syms[i], modules="numpy") for i in range(n)]
        z_funcs = [lambdify(sym_vars, z_syms[i], modules="numpy") for i in range(n)]
        return _solve_ccd(
            robot,
            fk_func,
            ro_funcs,
            z_funcs,
            target,
            q,
            lower_bounds,
            upper_bounds,
            tol,
            max_iter,
        )

    j_func = lambdify(sym_vars, j_sym, modules="numpy")
    return _solve_newton_or_lm(
        fk_func,
        j_func,
        target,
        q,
        lower_bounds,
        upper_bounds,
        tol,
        max_iter,
        method,
        damping,
        damping_scale,
    )