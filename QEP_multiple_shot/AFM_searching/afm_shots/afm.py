"""Open-chain AFM model, standalone copy for the fixed-N_g shot-number study.

Ported from Adaptive_bn/qep_shot/afm.py (itself ported from AFM_searching/code/models.py).
Only the two classes this study's research plan asks for are kept:

- Ising_Chain: open (non-periodic) Ising chain, cost operator AM = staggered
  magnetization. Coupling is the parent library's default open-chain
  Ns[k]@Ns[k+1] (model.Quantum_Model_DQ.set_coupling_ops), so no override needed here.
- Ising_Chain_AM2: same chain, cost operator AM^2 = AM@AM (fixes the even-N sign
  ambiguity; used unconditionally in this study per the research plan, for both the
  odd N=5 and N=9 chains studied here).

See ../reseach_plan_2026-09-11.md and ../../Adaptive_bn/PROJECT_SUMMARY.md for context.
"""
import jax
import jax.numpy as jnp
import dynamiqs as dq

try:
    from . import _pathfix  # noqa: F401  (side effect: makes `QEP` importable)
    from .schedules import ascend_func
except ImportError:
    import _pathfix  # noqa: F401  (fallback: run/imported as a flat script, not a package)
    from schedules import ascend_func
from QEP import model


class QM(model.Quantum_Model_DQ):
    def tune_func(self, t):
        return ascend_func(t / self.T)

    def total_H(self, params, funcs, y, target, beta):
        # baseline QEP nudge: fixes only the mean of cost_ops.
        H_sys = self.set_H(params, funcs)
        error_signal = jax.grad(self.cost_func, 0)(y, target)
        H_tot = H_sys + beta * error_signal * dq.modulated(self.tune_func, self.cost_ops)
        return H_tot


class Ising_Chain(QM):
    """Open (non-periodic) Ising chain: H = Omega*sum(X_i)/2 + Delta*sum(-N_i)
    + sum(V_i * N_i N_{i+1}), cost operator AM = sum_k (-1)^k Z_k / N.

    The interaction intentionally has no factor 1/2.  This is the convention of
    Adaptive_bn/qep_shot.afm.Ising_Chain and is why its phase diagram differs from
    the parent QEP.model.Quantum_Model_DQ model.
    """

    def get_cost_ops(self):
        Zs = self.block_ops[2]
        AM = sum([((-1) ** k) * Zs[k] for k in range(0, self.N)]) / self.N
        return AM

    def get_internal_H(self, params):
        _, _, V = params

        def scalar_multiply(a, op):
            return a * op

        S = sum(jax.tree_util.tree_map(scalar_multiply, list(V), self.COs))
        return S

    def set_H(self, params, funcs):
        Omega, Delta, _ = params
        fOmega, fDelta = funcs
        H = (dq.modulated(lambda t: fOmega(t, Omega, self.T), self.H_Omega)
             + dq.modulated(lambda t: fDelta(t, Delta, self.T), self.H_Delta)
             + self.H_int)
        return H

    def get_AM(self):
        Zs = self.block_ops[2]
        AM = sum([((-1) ** i) * Zs[i] for i in range(self.N)])
        return AM


class Ising_Chain_AM2(Ising_Chain):
    """Same open chain, but the QEP cost operator (nudged AND measured) is
    AM^2 = AM@AM instead of AM -- see qep_shot/afm.py's docstring for the full
    even-N sign-ambiguity motivation. Used for both N=5 and N=9 in this study
    per the research plan."""

    def get_cost_ops(self):
        AM = super().get_cost_ops()
        return AM @ AM
