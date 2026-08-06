"""
Numython R&D, (c) 2026
Moro is a Python library for kinematic and dynamic modeling of serial robots. 
This library has been designed, mainly, for academic and research purposes, 
using SymPy as base library. 
"""
import numpy as np
from numbers import Integral, Real
from dataclasses import dataclass
from typing import Optional
from sympy import lambdify
from moro.util import is_position_vector

__all__ = [
    "solve_position_ik",
    "solve_position_trajectory",
    "IKSolution",
    "IKTrajectorySolution",
]

# --- Outcome Messages ---
MSG_CONVERGED = "Converged successfully."
MSG_MAX_ITER = "Maximum number of iterations reached."
MSG_STAGNATED_STEP = "Solver stagnated because the joint update became too small."
MSG_STAGNATED_ERROR = "Solver stagnated because the position error stopped improving."
MSG_NUMERICAL_FK = "Numerical failure while evaluating the forward kinematics."
MSG_NUMERICAL_JACOBIAN = "Numerical failure while evaluating the Jacobian."
MSG_NUMERICAL_UPDATE = "Numerical failure while computing the joint update."
MSG_NUMERICAL_CCD = "Numerical failure during CCD evaluation."
MSG_TRAJECTORY_CONVERGED = "Trajectory solved successfully."
MSG_TRAJECTORY_FAILED = "Trajectory failed at target index {index}: {reason}"


# --- Public Result Type ---
@dataclass
class IKSolution:
    """
    Represents the result of a position inverse kinematics solve.
    
    Attributes
    ----------
    q : list
        Joint variables for the final solver state.
    converged : bool
        Whether the solver reached the requested tolerance.
    iterations : int
        Number of completed global iterations.
    error : float
        Final position error norm. It equals ``norm(residual)`` when
        ``residual`` is finite; otherwise it is ``np.inf``.
    method : str
        Solver method used ("newton", "lm", or "ccd").
    residual : list, optional
        Final position residual ``target_position - current_position``.
        It has three elements when available, or ``None`` for numerical
        failures where no finite residual can be computed.
    message : str
        Human-readable solver outcome message.
    """
    q: list
    converged: bool
    iterations: int
    error: float
    method: str = "lm"
    residual: Optional[list] = None
    message: str = ""

    def __post_init__(self):
        self.q = list(self.q) if hasattr(self.q, "__iter__") else [self.q]

        if self.residual is not None:
            self.residual = list(self.residual) if hasattr(self.residual, "__iter__") else [self.residual]

        self.converged = bool(self.converged)
        self.iterations = int(self.iterations)
        self.error = float(self.error)
        self.method = str(self.method)
        self.message = str(self.message)

        if self.converged:
            q_finite = _is_finite_array(self.q)
            error_finite = np.isfinite(self.error)
            if not q_finite or not error_finite:
                raise ValueError(
                    "A converged IKSolution requires finite joint values and finite error."
                )

        if self.residual is not None:
            residual_arr = np.asarray(self.residual, dtype=float).reshape(-1)
            if residual_arr.size != 3 or not _is_finite_array(residual_arr):
                raise ValueError(
                    "residual must be None or a finite 3-element vector."
                )
            self.residual = residual_arr.tolist()
    
    def __repr__(self):
        status = "Converged" if self.converged else "Did not converge"
        return (
            f"IKSolution(q={self.q}, {status}, "
            f"method={self.method}, iters={self.iterations}, "
            f"error={self.error:.2e})"
        )


@dataclass
class IKTrajectorySolution:
    """
    Represents the global result of solving a sequence of position IK targets.

    Attributes
    ----------
    solutions : list of IKSolution
        Per-target IK results in processing order. The list includes the
        failing solution when convergence stops at an intermediate target.
    converged : bool
        True only when all targets converged.
    failed_index : int, optional
        Index of the first non-converged target. It is None when all targets
        converged.
    message : str
        Short global outcome message.
    """

    solutions: list
    converged: bool
    failed_index: Optional[int] = None
    message: str = ""

    def __post_init__(self):
        self.solutions = list(self.solutions)
        for idx, solution in enumerate(self.solutions):
            if not isinstance(solution, IKSolution):
                raise TypeError(
                    f"solutions[{idx}] must be an instance of IKSolution."
                )

        self.converged = bool(self.converged)

        if self.failed_index is not None:
            if isinstance(self.failed_index, bool) or not isinstance(self.failed_index, Integral):
                raise ValueError("failed_index must be None or a non-negative integer.")
            if int(self.failed_index) < 0:
                raise ValueError("failed_index must be None or a non-negative integer.")
            self.failed_index = int(self.failed_index)

        if self.converged and self.failed_index is not None:
            raise ValueError("failed_index must be None when converged is True.")

        self.message = str(self.message)

    @property
    def qs(self):
        """Return per-target joint vectors in processing order."""
        return [list(solution.q) for solution in self.solutions]

    @property
    def errors(self):
        """Return per-target final error norms in processing order."""
        return [solution.error for solution in self.solutions]

    @property
    def iterations(self):
        """Return per-target iteration counts in processing order."""
        return [solution.iterations for solution in self.solutions]

    def __repr__(self):
        status = "Converged" if self.converged else "Did not converge"
        return (
            "IKTrajectorySolution("
            f"points={len(self.solutions)}, "
            f"status={status}, "
            f"failed_index={self.failed_index}"
            ")"
        )


# --- Validation and Conversion Helpers ---
def _prepare_rng(random_state):
    """Prepare a local NumPy Generator for reproducible random initialization."""
    if random_state is None:
        return np.random.default_rng()

    if isinstance(random_state, bool):
        raise ValueError(
            "random_state must be None, an integer seed, or numpy.random.Generator."
        )

    if isinstance(random_state, Integral):
        return np.random.default_rng(int(random_state))

    if isinstance(random_state, np.random.Generator):
        return random_state

    raise TypeError(
        "random_state must be None, an integer seed, or numpy.random.Generator."
    )


def _validate_stagnation_options(step_tol, error_change_tol, stagnation_iterations):
    """Validate stagnation-detection options and normalize their types."""
    if not isinstance(step_tol, Real) or isinstance(step_tol, bool) or not np.isfinite(float(step_tol)):
        raise ValueError("step_tol must be a finite real number.")
    if float(step_tol) < 0:
        raise ValueError("step_tol must satisfy step_tol >= 0.")

    if (
        not isinstance(error_change_tol, Real)
        or isinstance(error_change_tol, bool)
        or not np.isfinite(float(error_change_tol))
    ):
        raise ValueError("error_change_tol must be a finite real number.")
    if float(error_change_tol) < 0:
        raise ValueError("error_change_tol must satisfy error_change_tol >= 0.")

    if isinstance(stagnation_iterations, bool) or not isinstance(stagnation_iterations, Integral):
        raise ValueError("stagnation_iterations must be an integer >= 1.")
    if int(stagnation_iterations) < 1:
        raise ValueError("stagnation_iterations must satisfy stagnation_iterations >= 1.")

    return float(step_tol), float(error_change_tol), int(stagnation_iterations)


def _compute_residual(target, position):
    """Compute finite 3D residual target - position or return None."""
    if target is None or position is None:
        return None

    try:
        target_vec = np.asarray(target, dtype=float).reshape(-1)
        pos_vec = np.asarray(position, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if target_vec.size != 3 or pos_vec.size != 3:
        return None

    residual = target_vec - pos_vec
    if not _is_finite_array(residual):
        return None

    return residual


def _make_solution(
    q,
    converged,
    iterations,
    method,
    *,
    message="",
    target=None,
    position=None,
    residual=None,
):
    """Create a consistent IKSolution with normalized fields and residual."""
    q_safe = np.asarray(q, dtype=float).reshape(-1)
    if not _is_finite_array(q_safe):
        raise ValueError("Cannot build IKSolution with non-finite joint values.")

    if residual is None:
        residual_vec = _compute_residual(target, position)
    else:
        try:
            residual_vec = np.asarray(residual, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            residual_vec = None

    if residual_vec is None or residual_vec.size != 3 or not _is_finite_array(residual_vec):
        residual_out = None
        error = np.inf
    else:
        residual_out = residual_vec.tolist()
        error = float(np.linalg.norm(residual_vec))

    return IKSolution(
        q=q_safe.tolist(),
        converged=bool(converged),
        iterations=int(iterations),
        error=error,
        method=method,
        residual=residual_out,
        message=message,
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


def _prepare_target_positions(target_positions):
    """Validate and normalize a trajectory target sequence to a list of 3D vectors."""
    if isinstance(target_positions, np.ndarray):
        arr = np.asarray(target_positions)

        if arr.ndim == 1 and arr.shape[0] == 3:
            raise ValueError(
                "target_positions must be a trajectory with shape (m, 3), "
                "not a single 3-element vector."
            )

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("target_positions must have shape (m, 3).")

        if arr.shape[0] == 0:
            raise ValueError("target_positions must contain at least one target.")

        normalized = []
        for idx, row in enumerate(arr):
            try:
                target = np.asarray(row, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"target_positions[{idx}] must contain exactly 3 finite numeric values."
                ) from exc

            if target.size != 3 or not _is_finite_array(target):
                raise ValueError(
                    f"target_positions[{idx}] must contain exactly 3 finite numeric values."
                )

            normalized.append(target.tolist())

        return normalized

    if isinstance(target_positions, (str, bytes)):
        raise ValueError("target_positions must be an iterable of 3D targets.")

    try:
        items = list(target_positions)
    except TypeError as exc:
        raise ValueError("target_positions must be an iterable of 3D targets.") from exc

    if len(items) == 0:
        raise ValueError("target_positions must contain at least one target.")

    if len(items) == 3:
        try:
            single_target = np.asarray(items, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            single_target = None

        if single_target is not None and single_target.size == 3 and _is_finite_array(single_target):
            raise ValueError(
                "target_positions must be a trajectory with shape (m, 3), "
                "not a single 3-element vector."
            )

    normalized = []
    for idx, target in enumerate(items):
        try:
            target_vec = np.asarray(target, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"target_positions[{idx}] must contain exactly 3 finite numeric values."
            ) from exc

        if target_vec.size != 3 or not _is_finite_array(target_vec):
            raise ValueError(
                f"target_positions[{idx}] must contain exactly 3 finite numeric values."
            )

        normalized.append(target_vec.tolist())

    return normalized


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


def _prepare_initial_guess(q0, n, lower_bounds, upper_bounds, rng):
    """Validate and clip the initial guess."""
    if q0 is None:
        return rng.uniform(lower_bounds, upper_bounds)

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


def _failure_solution(q, iterations, method, message, target=None, position=None):
    """Build a controlled non-converged solution for numerical failures."""
    return _make_solution(
        q,
        converged=False,
        iterations=iterations,
        method=method,
        message=message,
        target=target,
        position=position,
    )


# --- Solver Routines ---
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
    step_tol,
    error_change_tol,
    stagnation_iterations,
):
    """Solve position IK using Newton-Raphson or Levenberg-Marquardt."""
    n = q.size
    completed_steps = 0
    stalled_by_step = 0
    stalled_by_error = 0

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
        return _make_solution(
            q,
            converged=True,
            iterations=0,
            method=method,
            message=MSG_CONVERGED,
            target=target,
            position=p_current,
            residual=error_vec,
        )

    lam = damping

    for _ in range(max_iter):
        j_val = _safe_eval_jacobian(j_func, q, n)
        if j_val is None:
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_JACOBIAN,
                target=target,
                position=p_current,
            )

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
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_UPDATE,
                target=target,
                position=p_current,
            )

        q_trial = np.clip(q + dq, lower_bounds, upper_bounds)
        if not _is_finite_array(q_trial):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_UPDATE,
                target=target,
                position=p_current,
            )

        p_trial = _safe_eval_vector(fk_func, q_trial, 3)
        if p_trial is None:
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_FK,
                target=target,
            )

        trial_error_vec = target - p_trial
        if not _is_finite_array(trial_error_vec):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_FK,
                target=target,
            )

        trial_error_norm = np.linalg.norm(trial_error_vec)
        if not np.isfinite(trial_error_norm):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_FK,
                target=target,
            )

        prev_error_norm = error_norm
        accepted_step = True
        if method == "lm":
            # Rejected LM attempts still count as algorithm iterations.
            if trial_error_norm < error_norm:
                lam *= damping_scale
                q_next = q_trial
                p_next = p_trial
                error_vec_next = trial_error_vec
                error_norm_next = trial_error_norm
            else:
                lam /= damping_scale
                accepted_step = False
                q_next = q
                p_next = p_current
                error_vec_next = error_vec
                error_norm_next = error_norm

            if not np.isfinite(lam) or lam <= 0:
                return _failure_solution(
                    q,
                    completed_steps,
                    method,
                    MSG_NUMERICAL_UPDATE,
                    target=target,
                    position=p_current,
                )
        else:
            q_next = q_trial
            p_next = p_trial
            error_vec_next = trial_error_vec
            error_norm_next = trial_error_norm

        step_norm = np.linalg.norm(q_next - q)
        if not np.isfinite(step_norm):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_UPDATE,
                target=target,
                position=p_current,
            )

        completed_steps += 1

        if step_tol > 0 and step_norm <= step_tol and error_norm_next >= tol:
            stalled_by_step += 1
        else:
            stalled_by_step = 0

        improvement = prev_error_norm - error_norm_next
        if improvement <= error_change_tol:
            stalled_by_error += 1
        else:
            stalled_by_error = 0

        q = q_next
        p_current = p_next
        error_vec = error_vec_next
        error_norm = error_norm_next

        if error_norm < tol:
            return _make_solution(
                q,
                converged=True,
                iterations=completed_steps,
                method=method,
                message=MSG_CONVERGED,
                target=target,
                position=p_current,
                residual=error_vec,
            )

        if (
            stalled_by_step >= stagnation_iterations
            and error_norm >= tol
            and (method != "lm" or accepted_step or stagnation_iterations > 1)
        ):
            return _make_solution(
                q,
                converged=False,
                iterations=completed_steps,
                method=method,
                message=MSG_STAGNATED_STEP,
                target=target,
                position=p_current,
                residual=error_vec,
            )

        if stalled_by_error >= stagnation_iterations and error_norm >= tol:
            return _make_solution(
                q,
                converged=False,
                iterations=completed_steps,
                method=method,
                message=MSG_STAGNATED_ERROR,
                target=target,
                position=p_current,
                residual=error_vec,
            )

    return _make_solution(
        q,
        converged=False,
        iterations=max_iter,
        method=method,
        message=MSG_MAX_ITER,
        target=target,
        position=p_current,
        residual=error_vec,
    )


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
    step_tol,
    error_change_tol,
    stagnation_iterations,
):
    """Solve position IK using CCD with robust finite-value checks."""
    n = q.size
    completed_steps = 0
    method = "ccd"
    stalled_by_step = 0
    stalled_by_error = 0

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
        return _make_solution(
            q,
            converged=True,
            iterations=0,
            method=method,
            message=MSG_CONVERGED,
            target=target,
            position=p_current,
            residual=init_error_vec,
        )

    error_norm = init_error_norm
    error_vec = init_error_vec

    for _ in range(max_iter):
        q_before = q.copy()
        p_eff = _safe_eval_vector(fk_func, q, 3)
        if p_eff is None:
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        error_vec = target - p_eff
        if not _is_finite_array(error_vec):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        error_norm = np.linalg.norm(error_vec)
        if not np.isfinite(error_norm):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        if error_norm < tol:
            return _make_solution(
                q,
                converged=True,
                iterations=completed_steps,
                method=method,
                message=MSG_CONVERGED,
                target=target,
                position=p_eff,
                residual=error_vec,
            )

        # One CCD global step is a full sweep from joint n down to 1.
        for i in range(n, 0, -1):
            joint_idx = i - 1
            joint_type = robot.joint_type(i)

            p_joint = _safe_eval_vector(ro_funcs[joint_idx], q, 3)
            if p_joint is None:
                return _failure_solution(
                    q,
                    completed_steps,
                    method,
                    MSG_NUMERICAL_CCD,
                    target=target,
                )

            z_axis = _safe_eval_vector(z_funcs[joint_idx], q, 3)
            if z_axis is None:
                return _failure_solution(
                    q,
                    completed_steps,
                    method,
                    MSG_NUMERICAL_CCD,
                    target=target,
                )

            z_norm = np.linalg.norm(z_axis)
            if not np.isfinite(z_norm) or z_norm <= 1e-12:
                return _failure_solution(
                    q,
                    completed_steps,
                    method,
                    MSG_NUMERICAL_CCD,
                    target=target,
                )
            z_axis = z_axis / z_norm

            if joint_type == "r":
                p_eff = _safe_eval_vector(fk_func, q, 3)
                if p_eff is None:
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

                r_ie = p_eff - p_joint
                r_it = target - p_joint

                if not _is_finite_array(r_ie) or not _is_finite_array(r_it):
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

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
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

                q[joint_idx] = np.clip(
                    q[joint_idx] + delta,
                    lower_bounds[joint_idx],
                    upper_bounds[joint_idx],
                )
            elif joint_type == "p":
                # Recompute current error right before prismatic update to avoid
                # using stale values from previous joint updates in the same sweep.
                p_eff = _safe_eval_vector(fk_func, q, 3)
                if p_eff is None:
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

                error_vec = target - p_eff
                if not _is_finite_array(error_vec):
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

                delta = np.dot(error_vec, z_axis)
                if not np.isfinite(delta):
                    return _failure_solution(
                        q,
                        completed_steps,
                        method,
                        MSG_NUMERICAL_CCD,
                        target=target,
                    )

                q[joint_idx] = np.clip(
                    q[joint_idx] + delta,
                    lower_bounds[joint_idx],
                    upper_bounds[joint_idx],
                )

            else:
                raise ValueError(
                    f"Unsupported joint type '{joint_type}' for joint {i}. "
                    "Expected 'r' (revolute) or 'p' (prismatic)."
                )

            if not np.isfinite(q[joint_idx]):
                return _failure_solution(
                    q,
                    completed_steps,
                    method,
                    MSG_NUMERICAL_CCD,
                    target=target,
                )

        completed_steps += 1

        p_after = _safe_eval_vector(fk_func, q, 3)
        if p_after is None:
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        error_after = target - p_after
        if not _is_finite_array(error_after):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        error_after_norm = np.linalg.norm(error_after)
        if not np.isfinite(error_after_norm):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        step_norm = np.linalg.norm(q - q_before)
        if not np.isfinite(step_norm):
            return _failure_solution(
                q,
                completed_steps,
                method,
                MSG_NUMERICAL_CCD,
                target=target,
            )

        if error_after_norm < tol:
            return _make_solution(
                q,
                converged=True,
                iterations=completed_steps,
                method=method,
                message=MSG_CONVERGED,
                target=target,
                position=p_after,
                residual=error_after,
            )

        if step_tol > 0 and step_norm <= step_tol:
            stalled_by_step += 1
        else:
            stalled_by_step = 0

        if stalled_by_step >= stagnation_iterations and error_after_norm >= tol:
            return _make_solution(
                q,
                converged=False,
                iterations=completed_steps,
                method=method,
                message=MSG_STAGNATED_STEP,
                target=target,
                position=p_after,
                residual=error_after,
            )

        improvement = error_norm - error_after_norm
        if improvement <= error_change_tol:
            stalled_by_error += 1
        else:
            stalled_by_error = 0

        if stalled_by_error >= stagnation_iterations and error_after_norm >= tol:
            return _make_solution(
                q,
                converged=False,
                iterations=completed_steps,
                method=method,
                message=MSG_STAGNATED_ERROR,
                target=target,
                position=p_after,
                residual=error_after,
            )

        error_norm = error_after_norm
        error_vec = error_after

    p_final = _safe_eval_vector(fk_func, q, 3)
    return _make_solution(
        q,
        converged=False,
        iterations=max_iter,
        method=method,
        message=MSG_MAX_ITER,
        target=target,
        position=p_final,
    )


# --- Public API ---
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
    random_state=None,
    step_tol=1e-12,
    error_change_tol=1e-12,
    stagnation_iterations=5,
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
    random_state : None, int, or numpy.random.Generator, optional
        Random state used only when ``q0 is None``. When ``None``, a local
        generator from ``np.random.default_rng()`` is used. Integer seeds
        provide reproducible random initial guesses.
    step_tol : float, optional
        Stagnation threshold for effective joint movement. The effective step
        is measured after applying joint limits:
        ``norm(q_trial - q_current)`` for Newton/LM and
        ``norm(q_after_sweep - q_before_sweep)`` for CCD.
        Must satisfy ``step_tol >= 0``.
    error_change_tol : float, optional
        Stagnation threshold for error improvement. If
        ``previous_error - current_error <= error_change_tol`` consecutively,
        the solver can terminate due to stagnation.
        Must satisfy ``error_change_tol >= 0``.
    stagnation_iterations : int, optional
        Number of consecutive stalled iterations (or CCD sweeps) required
        before returning a stagnation result. Must be an integer >= 1.

    Returns
    -------
    IKSolution
        An object containing the final joint variables, convergence status,
        iteration count, final error norm, method, residual, and message.

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

    ``residual`` is the 3D vector ``target_position - current_position`` at
    termination whenever a finite value is available. ``error`` is the norm
    of that residual when available; otherwise, ``error`` is ``np.inf`` and
    ``residual`` is ``None`` (for example, after numerical failures).

    ``message`` is a stable short description of the solver outcome
    (converged, maximum iterations, stagnation reason, or numerical failure).

    Stagnation can stop the solver early when the effective step becomes too
    small or the position error stops improving for
    ``stagnation_iterations`` consecutive iterations/sweeps.

    If ``q0 is None``, random initialization is sampled with a local NumPy
    Generator, so integer ``random_state`` values make initialization
    reproducible without changing NumPy's global random state.

    ``iterations`` counts completed global algorithm steps: Newton/LM count
    one per attempted update; CCD counts one per full sweep from joint n to 1.
    Therefore, if the initial guess already satisfies the tolerance,
    ``iterations`` is 0. If convergence is not reached, ``iterations`` equals
    the number of completed attempts/sweeps at termination (or ``max_iter``
    when the iteration limit is reached).

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
    >>> # Reproducible random initialization (used only when q0 is None)
    >>> sol = solve_position_ik(
    ...     rr,
    ...     [1.5, 0.5, 0.0],
    ...     parameters={l1: 1.0, l2: 1.0},
    ...     random_state=42,
    ... )
    >>> 
    >>> # Newton-Raphson
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="newton", parameters={l1: 1.0, l2: 1.0})
    >>> 
    >>> # CCD
    >>> sol = solve_position_ik(rr, [1.5, 0.5, 0.0], q0=[0.1, 0.1],
    ...                         method="ccd", parameters={l1: 1.0, l2: 1.0})
    """
    tol, max_iter, damping, damping_scale = _validate_solver_options(
        method, tol, max_iter, damping, damping_scale
    )
    step_tol, error_change_tol, stagnation_iterations = _validate_stagnation_options(
        step_tol, error_change_tol, stagnation_iterations
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

    rng = _prepare_rng(random_state)
    q = _prepare_initial_guess(q0, n, lower_bounds, upper_bounds, rng)

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
            step_tol,
            error_change_tol,
            stagnation_iterations,
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
        step_tol,
        error_change_tol,
        stagnation_iterations,
    )


def solve_position_trajectory(
    robot,
    target_positions,
    q0,
    *,
    parameters=None,
    method="lm",
    joint_limits=None,
    tol=1e-6,
    max_iter=None,
    damping=1.0,
    damping_scale=0.5,
    random_state=None,
    step_tol=1e-12,
    error_change_tol=1e-12,
    stagnation_iterations=5,
):
    """
    Solve position IK for a sequence of Cartesian position targets.

    This function is a thin orchestration layer over ``solve_position_ik``.
    It solves each target sequentially and reuses each converged solution as
    the initial guess for the next target.

    Parameters
    ----------
    robot : Robot
        Robot model with forward kinematics and Jacobian support.
    target_positions : iterable or numpy.ndarray
        Sequence of position targets with shape ``(m, 3)``. A single vector
        with shape ``(3,)`` is rejected to avoid ambiguity with
        ``solve_position_ik``.
    q0 : list or numpy.ndarray
        Initial joint seed for the first target. This argument is required.
    parameters : dict-like, optional
        Symbol substitutions passed directly to ``solve_position_ik``.
    method : str, optional
        Solver method passed to ``solve_position_ik``.
    joint_limits : list of tuples, optional
        Joint limits passed to ``solve_position_ik``.
    tol : float, optional
        Position error tolerance passed to ``solve_position_ik``.
    max_iter : int, optional
        Maximum iterations passed to ``solve_position_ik``.
    damping : float, optional
        LM damping parameter passed to ``solve_position_ik``.
    damping_scale : float, optional
        LM damping scale passed to ``solve_position_ik``.
    random_state : None, int, or numpy.random.Generator, optional
        Forwarded for API consistency with ``solve_position_ik``. With
        mandatory ``q0`` and sequential seeding, it typically has no effect.
    step_tol : float, optional
        Stagnation step tolerance passed to ``solve_position_ik``.
    error_change_tol : float, optional
        Stagnation error-change tolerance passed to ``solve_position_ik``.
    stagnation_iterations : int, optional
        Stagnation iteration count passed to ``solve_position_ik``.

    Returns
    -------
    IKTrajectorySolution
        Trajectory-level result. Processing stops at the first non-converged
        target, and the failing solution is included in ``solutions``.

    Notes
    -----
    This utility does not perform interpolation, timing, smoothing, full-pose
    orientation IK, branch optimization, or global trajectory planning.
    Reusing the previous solution can improve local continuity but does not
    guarantee global branch continuity.

    The returned ``trajectory.qs`` can be used directly as a sequence of joint
    configurations for animation pipelines.

    Examples
    --------
    >>> import moro as mr
    >>> from moro.abc import l1, l2, q1, q2
    >>> from moro.inverse_kinematics import solve_position_trajectory
    >>>
    >>> robot = mr.Robot((l1, 0, 0, q1, "r"), (l2, 0, 0, q2, "r"))
    >>> targets = [
    ...     [1.5, 0.2, 0.0],
    ...     [1.4, 0.4, 0.0],
    ...     [1.2, 0.6, 0.0],
    ... ]
    >>> trajectory = solve_position_trajectory(
    ...     robot,
    ...     targets,
    ...     q0=[0.1, 0.1],
    ...     parameters={l1: 1.0, l2: 1.0},
    ... )
    >>> trajectory.qs
    """
    targets = _prepare_target_positions(target_positions)

    if isinstance(q0, np.ndarray):
        q_seed = np.asarray(q0, dtype=float).reshape(-1).copy()
    else:
        try:
            q_seed = list(q0)
        except TypeError:
            q_seed = q0

    solutions = []
    for index, target in enumerate(targets):
        solution = solve_position_ik(
            robot,
            target,
            q0=q_seed,
            parameters=parameters,
            method=method,
            joint_limits=joint_limits,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            damping_scale=damping_scale,
            random_state=random_state,
            step_tol=step_tol,
            error_change_tol=error_change_tol,
            stagnation_iterations=stagnation_iterations,
        )

        solutions.append(solution)
        if not solution.converged:
            reason = solution.message if solution.message else "Solver did not converge."
            return IKTrajectorySolution(
                solutions=solutions,
                converged=False,
                failed_index=index,
                message=MSG_TRAJECTORY_FAILED.format(index=index, reason=reason),
            )

        q_seed = list(solution.q)

    return IKTrajectorySolution(
        solutions=solutions,
        converged=True,
        failed_index=None,
        message=MSG_TRAJECTORY_CONVERGED,
    )