"""Numerical dynamics evaluation and time-domain simulation for Moro."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from moro.abc import t as symbolic_time


__all__ = [
    "DynamicsSolution",
    "inverse_dynamics",
    "forward_dynamics",
    "state_derivative",
    "simulate",
]


@dataclass
class _NumericalDynamicsModel:
    dof: int
    mass_matrix_func: object
    coriolis_matrix_func: object
    gravity_vector_func: object

    def mass_matrix(self, q):
        value = _evaluate_matrix(
            self.mass_matrix_func,
            q,
            shape=(self.dof, self.dof),
            name="Mass matrix",
        )
        return value

    def coriolis_matrix(self, q, qd):
        args = np.concatenate((q, qd))
        return _evaluate_matrix(
            self.coriolis_matrix_func,
            args,
            shape=(self.dof, self.dof),
            name="Coriolis matrix",
        )

    def gravity_vector(self, q):
        return _evaluate_vector(
            self.gravity_vector_func,
            q,
            size=self.dof,
            name="Gravity vector",
        )


@dataclass
class DynamicsSolution:
    """Time-major numerical solution of the robot equations of motion."""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    success: bool
    message: str
    method: str

    def __post_init__(self):
        self.t = _as_numeric_array(self.t, name="t", ndim=1)
        self.q = _as_numeric_array(self.q, name="q", ndim=2)
        self.qd = _as_numeric_array(self.qd, name="qd", ndim=2)
        self.qdd = _as_numeric_array(self.qdd, name="qdd", ndim=2)
        self.success = bool(self.success)
        self.message = str(self.message)
        self.method = str(self.method)

        if self.t.size < 1:
            raise ValueError("DynamicsSolution requires at least one time sample.")
        if self.t.size > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError("DynamicsSolution.t must be strictly increasing.")
        if self.q.shape != self.qd.shape or self.q.shape != self.qdd.shape:
            raise ValueError("q, qd, and qdd must have identical shapes.")
        if self.q.shape[0] != self.t.size:
            raise ValueError(
                "q, qd, and qdd must have one row per time sample."
            )
        if self.q.shape[1] < 1:
            raise ValueError("DynamicsSolution must contain at least one DOF.")

    @property
    def samples(self):
        return self.t.size

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    @property
    def dof(self):
        return self.q.shape[1]

    def __repr__(self):
        return (
            f"DynamicsSolution(samples={self.samples}, dof={self.dof}, "
            f"duration={self.duration}, success={self.success}, "
            f"method='{self.method}')"
        )


def _as_numeric_array(value, *, name, ndim=None):
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numerical.") from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(arr, dtype=float, copy=True)


def _as_state_vector(value, dof, *, name):
    """Normalize a generalized-coordinate vector to shape (dof,)."""
    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must contain exactly {dof} real finite value(s)."
        ) from exc

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain only real values.")

    if raw.ndim == 0:
        if dof != 1:
            raise ValueError(
                f"{name} must contain exactly {dof} values; scalar "
                "broadcasting is not supported."
            )
        arr = _as_numeric_array([value], name=name, ndim=1)
    elif raw.ndim == 1:
        arr = _as_numeric_array(value, name=name, ndim=1)
    else:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector.")

    if arr.size != dof:
        raise ValueError(
            f"{name} must contain exactly {dof} values. Got {arr.size}."
        )
    return arr


def _as_finite_scalar(value, *, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real scalar.")
    return numeric


def _positive_scalar_or_none(value, *, name):
    if value is None:
        return None
    numeric = _as_finite_scalar(value, name=name)
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return numeric


def _validate_dynamic_joint_variables(robot):
    qds = []
    for index, q in enumerate(robot.qs):
        qd = sp.diff(q, symbolic_time)
        if qd == 0:
            raise ValueError(
                "Numerical dynamics requires time-dependent joint variables; "
                f"joint {index + 1} is static."
            )
        qds.append(qd)
    return tuple(qds)


def _apply_parameters(expr, parameters):
    if parameters is None:
        return expr
    try:
        return expr.subs(parameters)
    except Exception as exc:
        raise ValueError(
            "parameters must be a SymPy-compatible substitution mapping."
        ) from exc


def _prepare_numerical_model(robot, *, parameters=None):
    """Compile the Robot symbolic M, C, and G expressions once."""
    q_exprs = tuple(robot.qs)
    qd_exprs = _validate_dynamic_joint_variables(robot)
    n = robot.dof

    M = _apply_parameters(robot.inertia_matrix(), parameters)
    C = _apply_parameters(robot.coriolis_matrix(), parameters)
    G = _apply_parameters(robot.gravity_vector(), parameters)

    q_syms = sp.symbols(f"_q0:{n}", real=True)
    qd_syms = sp.symbols(f"_qd0:{n}", real=True)

    replacements = {
        **dict(zip(q_exprs, q_syms)),
        **dict(zip(qd_exprs, qd_syms)),
    }

    M_num = M.xreplace(replacements)
    C_num = C.xreplace(replacements)
    G_num = G.xreplace(replacements)

    allowed_MG = set(q_syms)
    allowed_C = set(q_syms) | set(qd_syms)

    unresolved_M = set(M_num.free_symbols) - allowed_MG
    unresolved_C = set(C_num.free_symbols) - allowed_C
    unresolved_G = set(G_num.free_symbols) - allowed_MG
    unresolved = unresolved_M | unresolved_C | unresolved_G

    if unresolved:
        missing = ", ".join(str(s) for s in sorted(unresolved, key=str))
        raise ValueError(
            "Cannot build numerical dynamics because the following symbols "
            f"have no numerical value: {missing}. Provide them using "
            "'parameters'."
        )

    M_func = sp.lambdify(q_syms, M_num, modules="numpy")
    C_func = sp.lambdify((*q_syms, *qd_syms), C_num, modules="numpy")
    G_func = sp.lambdify(q_syms, G_num, modules="numpy")

    return _NumericalDynamicsModel(
        dof=n,
        mass_matrix_func=M_func,
        coriolis_matrix_func=C_func,
        gravity_vector_func=G_func,
    )


def _evaluate_matrix(func, args, *, shape, name):
    try:
        value = np.asarray(func(*args), dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} could not be evaluated numerically.") from exc

    if value.shape != shape:
        try:
            value = value.reshape(shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} must evaluate to shape {shape}."
            ) from exc
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values.")
    return value


def _evaluate_vector(func, args, *, size, name):
    try:
        value = np.asarray(func(*args), dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{name} could not be evaluated numerically.") from exc

    if value.size != size:
        raise ValueError(f"{name} must evaluate to {size} values.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values.")
    return value


def _inverse_dynamics_prepared(model, q, qd, qdd):
    M = model.mass_matrix(q)
    C = model.coriolis_matrix(q, qd)
    G = model.gravity_vector(q)
    tau = M @ qdd + C @ qd + G
    if tau.shape != (model.dof,) or not np.all(np.isfinite(tau)):
        raise ValueError("Inverse dynamics produced an invalid generalized force.")
    return tau


def _forward_dynamics_prepared(model, q, qd, tau):
    M = model.mass_matrix(q)
    C = model.coriolis_matrix(q, qd)
    G = model.gravity_vector(q)
    rhs = tau - C @ qd - G
    if not np.all(np.isfinite(rhs)):
        raise ValueError("Forward dynamics right-hand side is not finite.")
    try:
        qdd = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Mass matrix is singular at the requested state."
        ) from exc

    qdd = np.asarray(qdd, dtype=float).reshape(-1)
    if qdd.shape != (model.dof,) or not np.all(np.isfinite(qdd)):
        raise ValueError("Forward dynamics produced an invalid acceleration.")
    return qdd


def inverse_dynamics(robot, q, qd, qdd, *, parameters=None):
    """Evaluate numerical inverse dynamics at one robot state."""
    model = _prepare_numerical_model(robot, parameters=parameters)
    q = _as_state_vector(q, model.dof, name="q")
    qd = _as_state_vector(qd, model.dof, name="qd")
    qdd = _as_state_vector(qdd, model.dof, name="qdd")
    return _inverse_dynamics_prepared(model, q, qd, qdd)


def forward_dynamics(robot, q, qd, tau, *, parameters=None):
    """Evaluate numerical forward dynamics at one robot state."""
    model = _prepare_numerical_model(robot, parameters=parameters)
    q = _as_state_vector(q, model.dof, name="q")
    qd = _as_state_vector(qd, model.dof, name="qd")
    tau = _as_state_vector(tau, model.dof, name="tau")
    return _forward_dynamics_prepared(model, q, qd, tau)


def _prepare_tau(tau, dof):
    if tau is None:
        zero = np.zeros(dof, dtype=float)

        def zero_tau(_t, _q, _qd):
            return zero.copy()

        return zero_tau

    if callable(tau):
        def user_tau(current_t, q, qd):
            value = tau(float(current_t), q.copy(), qd.copy())
            return _as_state_vector(value, dof, name="tau return")

        return user_tau

    constant = _as_state_vector(tau, dof, name="tau").copy()

    def constant_tau(_t, _q, _qd):
        return constant.copy()

    return constant_tau


def _state_derivative_prepared(model, current_t, state, tau_func):
    if state.shape != (2 * model.dof,) or not np.all(np.isfinite(state)):
        raise ValueError(
            f"state must be a finite vector with shape ({2 * model.dof},)."
        )
    q = state[:model.dof]
    qd = state[model.dof:]
    tau = tau_func(current_t, q, qd)
    qdd = _forward_dynamics_prepared(model, q, qd, tau)
    return np.concatenate((qd, qdd))


def state_derivative(robot, t, state, tau=None, *, parameters=None):
    """Return [qd, qdd] for a state ordered as [q, qd]."""
    current_t = _as_finite_scalar(t, name="t")
    model = _prepare_numerical_model(robot, parameters=parameters)
    state = _as_numeric_array(state, name="state", ndim=1)
    if state.shape != (2 * model.dof,):
        raise ValueError(
            f"state must have shape ({2 * model.dof},)."
        )
    tau_func = _prepare_tau(tau, model.dof)
    return _state_derivative_prepared(model, current_t, state, tau_func)


def _prepare_t_span(t_span):
    try:
        values = list(t_span)
    except TypeError as exc:
        raise ValueError("t_span must contain exactly (t0, tf).") from exc
    if len(values) != 2:
        raise ValueError("t_span must contain exactly (t0, tf).")
    t0 = _as_finite_scalar(values[0], name="t_span[0]")
    tf = _as_finite_scalar(values[1], name="t_span[1]")
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")
    return t0, tf


def _prepare_t_eval(t_eval, t0, tf):
    if t_eval is None:
        return None
    values = _as_numeric_array(t_eval, name="t_eval", ndim=1)
    if values.size < 1:
        raise ValueError("t_eval must contain at least one sample.")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("t_eval must be strictly increasing.")
    if values[0] < t0 or values[-1] > tf:
        raise ValueError("t_eval values must lie inside t_span.")
    return values


def simulate(
    robot,
    t_span,
    q0,
    qd0=None,
    *,
    tau=None,
    parameters=None,
    t_eval=None,
    method="RK45",
    rtol=None,
    atol=None,
    max_step=None,
):
    """Integrate the robot equations of motion with SciPy solve_ivp."""
    t0, tf = _prepare_t_span(t_span)
    model = _prepare_numerical_model(robot, parameters=parameters)

    q0 = _as_state_vector(q0, model.dof, name="q0")
    if qd0 is None:
        qd0 = np.zeros(model.dof, dtype=float)
    else:
        qd0 = _as_state_vector(qd0, model.dof, name="qd0")

    t_eval = _prepare_t_eval(t_eval, t0, tf)
    tau_func = _prepare_tau(tau, model.dof)

    rtol = _positive_scalar_or_none(rtol, name="rtol")
    atol = _positive_scalar_or_none(atol, name="atol")
    max_step = _positive_scalar_or_none(max_step, name="max_step")

    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty SciPy solver method string.")

    initial_state = np.concatenate((q0, qd0))

    def rhs(current_t, state):
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        return _state_derivative_prepared(
            model,
            float(current_t),
            state_arr,
            tau_func,
        )

    options = {}
    if t_eval is not None:
        options["t_eval"] = t_eval
    if rtol is not None:
        options["rtol"] = rtol
    if atol is not None:
        options["atol"] = atol
    if max_step is not None:
        options["max_step"] = max_step

    result = solve_ivp(
        rhs,
        (t0, tf),
        initial_state,
        method=method,
        **options,
    )

    times = np.asarray(result.t, dtype=float).reshape(-1)
    states = np.asarray(result.y, dtype=float).T
    if states.ndim != 2 or states.shape[0] != times.size:
        raise ValueError("SciPy returned an invalid state history.")

    q = states[:, :model.dof]
    qd = states[:, model.dof:]

    qdd_rows = []
    for current_t, q_row, qd_row in zip(times, q, qd):
        tau_row = tau_func(current_t, q_row, qd_row)
        qdd_rows.append(
            _forward_dynamics_prepared(model, q_row, qd_row, tau_row)
        )
    qdd = np.asarray(qdd_rows, dtype=float)

    return DynamicsSolution(
        t=times,
        q=q,
        qd=qd,
        qdd=qdd,
        success=result.success,
        message=result.message,
        method=method,
    )
