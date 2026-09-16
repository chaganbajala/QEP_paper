"""Gradient estimators -- self-contained port of ``QEP.grad``.

Three QEP variants are compared throughout this study, exactly as in the
cluster project ``/u/qinwang/notebooks/QEP_sparse/Fixed_TN``:

``ED``   *exact QEP*: both the free and the nudged state are the **exact ground
         state** of the corresponding final Hamiltonian.  The only error left is
         the O(beta) error of the finite nudge.  Independent of T.
``AE``   *uncached QEP*: both states come from a full adiabatic sweep of duration
         T, the nudged one with the coupling ramped on smoothly during the sweep.
``CAE``  *cached QEP*: one free adiabatic sweep; the nudged state is then obtained
         by continuing from the free state under the static final Hamiltonian
         while the nudge is ramped on over ``T_nudge``.

All three feed the same estimator

    g = ( <psi_nudge| d_theta H |psi_nudge> - <psi_free| d_theta H |psi_free> ) / beta ,

which is the Onsager/equilibrium-propagation form of ``dL/d_theta``.
"""
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import dynamiqs as dq

__all__ = [
    "Exact_Diagonalization", "Adiabatic_Evolution", "Adiabatic_Evolution_Nudge",
    "QEP_grad", "Cached_QEP_grad", "exact_loss_gradient", "exact_y",
]

_OPTS = dq.Options(progress_meter=False, save_states=False)

# diffrax's adaptive integrator refuses to take more than `max_steps` steps.
# The library default (1e5) is far short of what the longest sweeps here need --
# a single leg at T = 5000, N = 9 takes of order 1e6 -- and the run aborts when
# it runs out.  The cap is a guard, not a tolerance: raising it changes no
# numerical result, it only lets the long runs finish.
_MAX_STEPS = 50_000_000


def default_method():
    return dq.method.Tsit5(max_steps=_MAX_STEPS)


# ---------------------------------------------------------------- solvers ---
class Exact_Diagonalization:
    """Return the exact ground state of the (possibly nudged) final Hamiltonian.

    Uses the dense numpy fast path ``qm.H_final_np``: at ``t = T`` the schedules
    have reached their end points, so the nudged final Hamiltonian is exactly
    ``Omega*H_Omega + Delta*H_Delta + H_int + beta*(dL/dy)*C``.  This is
    algebraically identical to evaluating ``qm.total_H(...)(qm.T)`` but avoids
    rebuilding a dynamiqs TimeQArray (and re-tracing a jaxpr) every epoch.
    """

    needs_psi0 = False

    def __init__(self, param_funcs):
        self.param_funcs = param_funcs

    def __call__(self, qm, params, psi0=None, beta=0.0, y=0.0, target=0.0, T=None):
        H = qm.H_final_np(params, y=y, target=target, beta=beta)
        return [qm.set_psi0(H)]


class Adiabatic_Evolution:
    """Full adiabatic sweep under ``total_H`` (nudge ramped on during the sweep)."""

    needs_psi0 = True

    def __init__(self, param_funcs, method=None):
        self.param_funcs = param_funcs
        self.method = default_method() if method is None else method

    @partial(jax.jit, static_argnames=["self", "qm"])
    def __call__(self, qm, params, psi0, beta, y, target, T=None):
        Tv = qm.T_of(T)
        H = qm.total_H(params, self.param_funcs, y, target, beta, Tv)
        tsave = jnp.stack([jnp.zeros(()), jnp.asarray(Tv, dtype=float)])
        return dq.sesolve(H, psi0, tsave, exp_ops=[], method=self.method,
                          options=_OPTS).final_state

    def __hash__(self):
        return hash((type(self).__name__, id(self.param_funcs), repr(self.method)))

    def __eq__(self, other):
        return self is other


class Adiabatic_Evolution_Nudge(Adiabatic_Evolution):
    """Nudge-only leg: static final H, nudge ramped over ``T_nudge``."""

    def __init__(self, param_funcs, nudge_func, method=None):
        super().__init__(param_funcs, method)
        self.nudge_func = nudge_func

    @partial(jax.jit, static_argnames=["self", "qm"])
    def __call__(self, qm, params, psi0, beta, y, target, T=None):
        # this study always uses T_nudge = T
        Tv = qm.T_of(T)
        Tn = qm.T_nudge if T is None else T
        H_tot = qm.set_H(params, self.param_funcs, Tv)(Tv)
        H = qm.nudge_H(H_tot, beta, y, target, self.nudge_func, Tn)
        tsave = jnp.stack([jnp.zeros(()), jnp.asarray(Tn, dtype=float)])
        return dq.sesolve(H, psi0, tsave, exp_ops=[], method=self.method,
                          options=_OPTS).final_state


# --------------------------------------------------------------- QEP grad ---
class QEP_grad:
    def __init__(self, beta, solver):
        self.beta = beta
        self.solver = solver
        self.param_funcs = solver.param_funcs

    def H_exp(self, params, psi0, qm, T):
        H = qm.set_H(params, self.param_funcs, T)
        return jnp.real(dq.expect(H(T), psi0))

    @partial(jax.jit, static_argnames=["self", "qm"])
    def H_params_gradient(self, params, psi0, qm, T):
        return jax.grad(self.H_exp, 0)(params, psi0, qm, T)

    def _diff(self, x, y):
        return (x - y) / self.beta

    def _psi0(self, qm):
        """Initial state of the sweep (parameter-independent; cached on ``qm``)."""
        return qm.psi_init() if getattr(self.solver, "needs_psi0", True) else None

    def states(self, params, qm, target, T=None):
        """Return ``(psi_free, psi_nudge, y)``."""
        psi0 = self._psi0(qm)
        psi_free = self._free(qm, params, psi0, target, T)
        y = dq.expect(qm.cost_ops, psi_free)
        psi_nudge = self._nudge(qm, params, psi0, psi_free, y, target, T)
        return psi_free, psi_nudge, y

    def _free(self, qm, params, psi0, target, T=None):
        out = self.solver(qm, params, psi0, 0.0, target, target, T)
        return out[-1] if isinstance(out, (list, tuple)) else out

    def _nudge(self, qm, params, psi0, psi_free, y, target, T=None):
        out = self.solver(qm, params, psi0, self.beta, y, target, T)
        return out[-1] if isinstance(out, (list, tuple)) else out

    def __call__(self, params, qm, target, T=None):
        Tv = float(qm.T_of(T))
        psi_free, psi_nudge, y = self.states(params, qm, target, T)
        dHdp_free = self.H_params_gradient(params, psi_free, qm, Tv)
        dHdp_nudge = self.H_params_gradient(params, psi_nudge, qm, Tv)
        params_grad = jax.tree_util.tree_map(self._diff, dHdp_nudge, dHdp_free)
        return y, params_grad

    def __hash__(self):
        return hash((type(self).__name__, float(self.beta), id(self.solver)))

    def __eq__(self, other):
        return self is other


class Cached_QEP_grad(QEP_grad):
    def __init__(self, beta, solver):
        free_solver, nudge_solver = solver
        super().__init__(beta, free_solver)
        self.free_solver, self.nudge_solver = free_solver, nudge_solver

    def _free(self, qm, params, psi0, target, T=None):
        out = self.free_solver(qm, params, psi0, 0.0, 0.0, target, T)
        return out[-1] if isinstance(out, (list, tuple)) else out

    def _nudge(self, qm, params, psi0, psi_free, y, target, T=None):
        out = self.nudge_solver(qm, params, psi_free, self.beta, y, target, T)
        return out[-1] if isinstance(out, (list, tuple)) else out


# ----------------------------------------------------- exact reference ------
def exact_y(qm, params, param_funcs=None):
    """``<AM>`` (or ``<AM^2>``) in the exact ground state of the final Hamiltonian."""
    psi = qm.set_psi0(qm.H_final_np(params))
    return float(np.real(np.asarray(dq.expect(qm.cost_ops, psi))))


def exact_loss_gradient(qm, params, target, param_funcs=None, h=1e-5):
    """``dL/d(Omega, Delta)`` through the *exact* ground state, by central differences.

    This is the beta -> 0, T -> infinity limit that every QEP estimator is trying
    to reproduce; it is the reference against which the O(1) bias is measured.
    """
    Omega, Delta, Vs = params
    out = []
    for i in range(2):
        pp = [Omega, Delta, Vs]
        pm = [Omega, Delta, Vs]
        pp[i] = pp[i] + h
        pm[i] = pm[i] - h
        Lp = float(qm.cost_func(exact_y(qm, tuple(pp), param_funcs), target))
        Lm = float(qm.cost_func(exact_y(qm, tuple(pm), param_funcs), target))
        out.append((Lp - Lm) / (2 * h))
    return np.array(out)
