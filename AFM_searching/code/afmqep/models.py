"""Quantum models for AFM searching -- self-contained (no import of ``QEP``).

The Hamiltonian swept during the free phase is

    H(t) = f_Omega(t) * sum_k X_k/2  +  f_Delta(t) * sum_k (-n_k)
                                     +  sum_<ij> V_ij n_i n_j ,       n = (1+Z)/2

with ``f_Omega``, ``f_Delta`` from :mod:`afmqep.schedules`.  Note the interaction
carries **no factor 1/2**: this reproduces ``qep_shot.afm.Ising_Chain``, and
differs by a factor 2 from ``QEP.model.Quantum_Model_DQ`` (which used
``0.5 * sum V n n``).  That factor is exactly why the phase diagram has to be
re-drawn for this project.

The QEP nudge is switched on **smoothly**:

    H_tot(t) = H(t) + beta * dL/dy * ascend_func(t/T) * C ,

with ``C`` the cost operator.  ``QEP.model.Quantum_Model_DQ.total_H`` switches it
on abruptly (a constant ``beta * dL/dy * C`` for the whole sweep); the smooth
version is ``qep_shot.afm.QM.total_H`` and is what this project uses.

Order parameters
----------------
``AM = (1/N) sum_k (-1)^k Z_k``  (chain) or ``(1/N) sum_k (-1)^(row+col) Z_k``
(lattice) -- the staggered magnetisation.  ``target = 1`` is perfect Neel order.
The ``*_AM2`` variants nudge and measure ``AM^2`` instead, which is blind to the
overall sign of the Neel pattern and therefore usable when the two checkerboard
fillings are degenerate.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import dynamiqs as dq

from .schedules import ascend_func
from .gs import exact_ground_state

__all__ = [
    "QM", "Ising_Chain", "Ising_Chain_AM2",
    "Ising_Lattice", "Ising_Lattice_AM2",
    "chain_pairs", "lattice_pairs",
]


def chain_pairs(N):
    """Open chain nearest-neighbour bonds."""
    return [(k, k + 1) for k in range(N - 1)]


def lattice_pairs(n_row, n_col):
    """Open 2D square lattice nearest-neighbour bonds, row-major site index."""
    pairs = []
    for k in range(n_row * n_col):
        row, col = divmod(k, n_col)
        if row < n_row - 1:
            pairs.append((k, (row + 1) * n_col + col))
        if col < n_col - 1:
            pairs.append((k, row * n_col + (col + 1)))
    return pairs


class QM:
    """Base model: operators, schedules, nudged Hamiltonian, ground states."""

    def __init__(self, N, T, dt, T_nudge=None, V=None, verbose=False):
        self.N = int(N)
        self.T = float(T)
        self.dt = float(dt)
        self.T_nudge = float(T) if T_nudge is None else float(T_nudge)
        self.dims = tuple([2] * self.N)

        t0 = time.time()
        self._prepare_ops()
        self.pairs = self.get_pairs()
        self.COs = self.set_coupling_ops()
        self.cost_ops = self.get_cost_ops()

        # The integrator is adaptive, so a save grid only decides what is
        # returned, and the QEP gradient reads the final state alone -- the
        # solvers therefore build a two-point tsave from T themselves.  These
        # are kept for callers that want the nominal grid; dt is metadata (it
        # fixed the save grid on the cluster, at no effect on the result).
        self.tlist = jnp.asarray([0.0, self.T])
        self.tlist_nudge = jnp.asarray([0.0, self.T_nudge])

        self.set_params(0.0, 0.0, V)
        if verbose:
            print(f"[{type(self).__name__}] N={self.N} built in {time.time()-t0:.2f}s")

    # ---------------------------------------------------------------- ops ---
    def _prepare_ops(self):
        N = self.N
        sx, sz = dq.sigmax(), dq.sigmaz()
        n_op = (dq.eye(2) + sz) / 2

        ids = [dq.eye(2)]
        for _ in range(1, N):
            ids.append(dq.tensor(ids[-1], dq.eye(2)))

        def embed(op, k):
            if N == 1:
                return op
            if k == 0:
                return dq.tensor(op, ids[N - 2])
            if k == N - 1:
                return dq.tensor(ids[N - 2], op)
            return dq.tensor(dq.tensor(ids[k - 1], op), ids[N - 2 - k])

        Xs = [embed(sx, k) for k in range(N)]
        Zs = [embed(sz, k) for k in range(N)]
        Ns = [embed(n_op, k) for k in range(N)]
        self.block_ops = [Xs, None, Zs, Ns]

        self.H_Omega = sum(0.5 * Xs[k] for k in range(N))
        self.H_Delta = sum(-Ns[k] for k in range(N))

    def get_pairs(self):
        raise NotImplementedError

    def set_coupling_ops(self):
        Ns = self.block_ops[3]
        return [Ns[i] @ Ns[j] for (i, j) in self.pairs]

    def get_cost_ops(self):
        raise NotImplementedError

    def get_AM(self):
        """Unnormalised staggered magnetisation ``sum_k (-1)^{s(k)} Z_k``."""
        raise NotImplementedError

    # ------------------------------------------------------------- params ---
    def set_params(self, Omega, Delta, Vs=None):
        """Return the parameter tuple and freeze the interaction term.

        ``Vs`` defaults to all-ones.  The couplings are held fixed throughout
        this study (as on the cluster): ``set_H`` reads the cached ``H_int``, so
        the V-component of the QEP gradient is identically zero and only
        ``(Omega, Delta)`` are trained.
        """
        if Vs is None:
            Vs = jnp.ones(len(self.COs))
        Vs = jnp.asarray(Vs, dtype=float)
        self.Vs = Vs
        self.H_int = self.get_internal_H((Omega, Delta, Vs))
        return Omega, Delta, Vs

    def get_internal_H(self, params):
        """``sum_k V_k * n_i n_j`` -- no 1/2 (qep_shot convention)."""
        _, _, V = params
        if len(self.COs) == 0:
            return 0.0 * self.H_Omega
        return sum(a * op for a, op in zip(list(V), self.COs))

    # -------------------------------------------------------- Hamiltonians ---
    def T_of(self, T=None):
        """Sweep duration: the stored one, or an override.

        ``T`` may be a *traced* value.  Passing it in rather than baking
        ``self.T`` into the graph means one compiled program serves every
        evolution time -- which matters for the bias scan of
        Section 4 of the report, where hundreds of values of T are visited and
        a per-T compilation both dominates the run time and eventually exhausts
        memory through the jit cache.
        """
        return self.T if T is None else T

    def set_H(self, params, funcs, T=None):
        Omega, Delta, _ = params
        fOmega, fDelta = funcs
        Tv = self.T_of(T)
        return (dq.modulated(lambda t: fOmega(t, Omega, Tv), self.H_Omega)
                + dq.modulated(lambda t: fDelta(t, Delta, Tv), self.H_Delta)
                + self.H_int)

    def set_H_f(self, params):
        """Static Hamiltonian at the end of the sweep."""
        Omega, Delta, _ = params
        return Omega * self.H_Omega + Delta * self.H_Delta + self.H_int

    def tune_func(self, t, T=None):
        """Smooth switch-on of the nudge over the whole sweep."""
        return ascend_func(t / self.T_of(T))

    def cost_func(self, y, target):
        return jnp.real(jnp.sum((y - target) ** 2))

    def total_H(self, params, funcs, y, target, beta, T=None):
        Tv = self.T_of(T)
        H_sys = self.set_H(params, funcs, Tv)
        error_signal = jax.grad(self.cost_func, 0)(y, target)
        return H_sys + beta * error_signal * dq.modulated(
            lambda t: self.tune_func(t, Tv), self.cost_ops)

    def nudge_H(self, H_tot, beta, y, target, nudge_func, T_nudge=None):
        """Cached variant: static final H plus a nudge ramped over ``T_nudge``."""
        Tn = self.T_nudge if T_nudge is None else T_nudge
        error_signal = jax.grad(self.cost_func, 0)(y, target)
        return (dq.modulated(lambda t: nudge_func(t, beta, Tn),
                             error_signal * self.cost_ops)
                + H_tot)

    # ------------------------------------------------------------- states ---
    def set_psi0(self, H):
        return exact_ground_state(H, self.dims)

    # ------------------------------------------------- dense numpy fast path ---
    #  At t = T every schedule reaches its end point (f_Omega = Omega,
    #  f_Delta = Delta, nudge ramp = 1), so the final Hamiltonian is just a
    #  linear combination of four fixed matrices.  Caching them as plain numpy
    #  arrays lets the "exact QEP" branch skip dynamiqs' TimeQArray machinery,
    #  which otherwise re-traces a jaxpr on every epoch and dominates its cost.
    _NP_CACHE_MAX_DIM = 4096

    @property
    def np_ops(self):
        if getattr(self, "_np_ops", None) is None:
            if 2 ** self.N > self._NP_CACHE_MAX_DIM:
                raise MemoryError(
                    f"refusing to densify N={self.N} (dim {2**self.N}); "
                    "use the sparse route in afmqep.gs instead")
            self._np_ops = {
                "H_Omega": np.asarray(self.H_Omega.to_jax()),
                "H_Delta": np.asarray(self.H_Delta.to_jax()),
                "H_int": np.asarray(self.H_int.to_jax()),
                "C": np.asarray(self.cost_ops.to_jax()),
            }
        return self._np_ops

    def H_final_np(self, params, y=0.0, target=0.0, beta=0.0):
        """Dense final Hamiltonian, optionally nudged (nudge ramp is 1 at t=T)."""
        Omega, Delta, _ = params
        o = self.np_ops
        H = float(Omega) * o["H_Omega"] + float(Delta) * o["H_Delta"] + o["H_int"]
        if beta:
            err = float(np.real(2.0 * (np.asarray(y) - target)))
            H = H + (beta * err) * o["C"]
        return H

    def psi_init(self):
        """Ground state at t = 0.

        Both schedules start at ``f_Omega = 0``, ``f_Delta = -1``, so
        ``H(0) = sum_k n_k + sum_<ij> V_ij n_i n_j`` -- independent of the
        trainable parameters.  It is therefore computed once and reused for
        every epoch of every run.
        """
        if getattr(self, "_psi_init", None) is None:
            o = self.np_ops
            H0 = -o["H_Delta"] + o["H_int"]
            self._psi_init = exact_ground_state(H0, self.dims)
        return self._psi_init

    def __hash__(self):
        return hash((type(self).__name__, self.N, float(self.T), float(self.T_nudge),
                     tuple(np.asarray(self.Vs).tolist())))

    def __eq__(self, other):
        return (type(self) is type(other) and self.N == other.N
                and self.T == other.T and self.T_nudge == other.T_nudge
                and np.array_equal(np.asarray(self.Vs), np.asarray(other.Vs)))


class Ising_Chain(QM):
    """Open Ising chain, cost operator = staggered magnetisation AM."""

    def get_pairs(self):
        return chain_pairs(self.N)

    def _stagger(self):
        return np.array([(-1) ** k for k in range(self.N)], dtype=float)

    def get_cost_ops(self):
        Zs = self.block_ops[2]
        s = self._stagger()
        return sum(s[k] * Zs[k] for k in range(self.N)) / self.N

    def get_AM(self):
        Zs = self.block_ops[2]
        s = self._stagger()
        return sum(s[k] * Zs[k] for k in range(self.N))


class Ising_Chain_AM2(Ising_Chain):
    """Same chain, cost operator = AM^2 (blind to the sign of the Neel pattern)."""

    def get_cost_ops(self):
        AM = super().get_cost_ops()
        return AM @ AM


class Ising_Lattice(Ising_Chain):
    """Open 2D square lattice.  ``shape = (n_row, n_col)``, ``N = n_row*n_col``."""

    def __init__(self, shape, T, dt, T_nudge=None, V=None, verbose=False):
        self.shape = tuple(shape)
        super().__init__(self.shape[0] * self.shape[1], T, dt, T_nudge, V, verbose)

    def get_pairs(self):
        return lattice_pairs(*self.shape)

    def _stagger(self):
        n_col = self.shape[1]
        return np.array([(-1) ** (k // n_col + k % n_col) for k in range(self.N)],
                        dtype=float)

    def __hash__(self):
        return hash((type(self).__name__, self.shape, float(self.T), float(self.T_nudge),
                     tuple(np.asarray(self.Vs).tolist())))

    def __eq__(self, other):
        return (type(self) is type(other) and self.shape == other.shape
                and self.T == other.T and self.T_nudge == other.T_nudge
                and np.array_equal(np.asarray(self.Vs), np.asarray(other.Vs)))


class Ising_Lattice_AM2(Ising_Lattice):
    def get_cost_ops(self):
        AM = super().get_cost_ops()
        return AM @ AM
