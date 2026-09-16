"""Ordinary AE with checked state normalization before every expectation.

Endpoint normalization removes only overall amplitude error. The raw norm
guard rejects large integration errors; agreement under tighter tolerances
is still required to establish state/gradient accuracy.
"""
import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from .grad import Adiabatic_Evolution, QEP_grad
from .schedules import get_schedule


class NormDriftError(RuntimeError):
    pass


def checked_unit(psi, tolerance, leg):
    vector = psi.to_jax()
    norm2 = float(jnp.real(jnp.vdot(vector, vector)))
    if not np.isfinite(norm2) or norm2 <= 0 or abs(norm2 - 1) > tolerance:
        raise NormDriftError(
            f"{leg} state norm squared={norm2:.12g}; allowed drift={tolerance:g}. "
            "Rejecting the epoch; tighten the solver tolerances and rerun "
            "from a validated checkpoint or the original initial parameters.")
    return dq.unit(psi), norm2


class NormalizedAE:
    def __init__(self, beta, schedule, method, norm_tolerance=1e-5):
        if not np.isfinite(norm_tolerance) or norm_tolerance <= 0:
            raise ValueError('norm_tolerance must be positive and finite')
        f_omega, f_delta, _ = get_schedule(schedule)
        self.base = QEP_grad(beta, Adiabatic_Evolution((f_omega, f_delta), method))
        self.norm_tolerance = norm_tolerance
        self.last_diagnostics = {}

    def __call__(self, params, qm, target, T=None):
        self.last_diagnostics = {}
        psi0 = qm.psi_init()
        raw_free = self.base._free(qm, params, psi0, target, T)
        free, norm_free = checked_unit(raw_free, self.norm_tolerance, 'free')
        y = jnp.real(dq.expect(qm.cost_ops, free))
        self.last_diagnostics = {'norm2_free': norm_free,
                                 'y_raw': float(y) * norm_free}
        raw_nudge = self.base._nudge(qm, params, psi0, free, y, target, T)
        nudge, norm_nudge = checked_unit(raw_nudge, self.norm_tolerance, 'nudged')
        self.last_diagnostics['norm2_nudge'] = norm_nudge
        duration = float(qm.T_of(T))
        free_grad = self.base.H_params_gradient(params, free, qm, duration)
        nudge_grad = self.base.H_params_gradient(params, nudge, qm, duration)
        gradient = tuple((a-b)/self.base.beta for a, b in zip(nudge_grad, free_grad))
        if not all(np.isfinite(np.asarray(g)).all() for g in gradient):
            raise FloatingPointError('Non-finite normalized QEP gradient; epoch rejected')
        return y, gradient
