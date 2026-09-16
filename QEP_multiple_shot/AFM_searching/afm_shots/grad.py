"""Finite-shot QEP gradient estimator, standalone copy for the fixed-N_g shot-number
study (no dependency on Adaptive_bn/qep_shot).

`Shot_QEP_Grad` is ported verbatim from Adaptive_bn/qep_shot/grad.py (itself ported
from AFM_searching/code/shot_grad.py) -- the probe-field extensions
(`QEP_grad_PF`/`Shot_QEP_grad_PF`/`Inference_qm`) and `AFM_QEP_Grad` are dropped here
since this study's research plan only calls for `Shot_QEP_Grad` (AFM's target is
identifiable from a single fixed nudge, no probe-field sweep needed).

The additional ``call_with_moments`` method supports task 5.  It performs the same
finite-shot update as ``__call__``, and also returns the exact conditional mean and
variance of that estimator at the sampled output ``y``.  Thus the diagnostic curves
do not acquire a second layer of Monte-Carlo noise.
"""
import jax
import jax.numpy as jnp
import dynamiqs as dq

try:
    from . import _pathfix  # noqa: F401  (side effect: makes `QEP` importable)
except ImportError:
    import _pathfix  # noqa: F401  (fallback: run/imported as a flat script, not a package)
from QEP import grad as _qep_grad


class Shot_QEP_Grad(_qep_grad.QEP_grad):
    def _eigh_cached(self, O):
        # The operators measured here (cost_ops, and the dHdp leaves from grad_H)
        # are constant across a whole training run -- cache the eigendecomposition
        # instead of paying a fresh `eigh` every call.
        cache = getattr(self, '_eigh_cache', None)
        if cache is None:
            cache = self._eigh_cache = {}
        key = id(O)
        if key not in cache:
            cache[key] = jnp.linalg.eigh(dq.to_jax(dq.asqarray(O)))
        return cache[key]

    def _measure_single(self, vals, V, state, key, shots=1):
        if dq.isket(state):
            psi = dq.to_jax(dq.toket(state))[..., 0]
            p = jnp.abs(jnp.einsum('ji,...j->...i', V.conj(), psi)) ** 2
        else:
            rho = dq.to_jax(state)
            p = jnp.einsum('ji,...jk,ki->...i', V.conj(), rho, V).real
        p = jnp.clip(p, 0.0) / p.sum(-1, keepdims=True)
        idx = jax.random.categorical(key, jnp.log(p), shape=(shots, *p.shape[:-1]))
        return vals[idx]

    def measure(self, O, state, seed=100, shots=1):
        key = jax.random.PRNGKey(seed)
        vals, V = self._eigh_cached(O)
        if V.ndim == 2:
            return self._measure_single(vals, V, state, key, shots)

        # O carries leading batch axes (e.g. grad_H's Vs leaf: one operator per
        # coupling, stacked as (M, dim, dim)) -- measure each operator in the
        # batch independently (same state, split keys).
        batch_shape = V.shape[:-2]
        Vflat = V.reshape((-1,) + V.shape[-2:])
        valsflat = vals.reshape((-1, vals.shape[-1]))
        keys = jax.random.split(key, Vflat.shape[0])
        outs = jnp.stack(
            [self._measure_single(valsflat[i], Vflat[i], state, keys[i], shots) for i in range(Vflat.shape[0])],
            axis=0,
        )
        return outs.reshape(batch_shape + outs.shape[1:])

    def multiple_measure(self, Os, state, seed=100, shots=1):
        """Measure every operator leaf with an independent pseudo-random stream.

        Omega and Delta belong to noncommuting X and Z measurement groups.  The
        source implementation passed the same PRNG key to every pytree leaf, which
        creates an unphysical cross-group correlation.  Deterministic offsets keep
        runs reproducible while representing independent experimental shot sets.
        """
        is_qarray = lambda x: isinstance(x, dq.QArray)
        leaves, structure = jax.tree_util.tree_flatten(Os, is_leaf=is_qarray)
        measured = [
            self.measure(op, state, int(seed) + 104729 * i, shots)
            for i, op in enumerate(leaves)
        ]
        return jax.tree_util.tree_unflatten(structure, measured)

    def grad_H(self, params, qm):
        cache = getattr(self, '_dHdp_cache', None)
        if cache is None:
            cache = self._dHdp_cache = {}
        key = id(qm)
        if key in cache:
            return cache[key]

        def func(params):
            H = qm.set_H_f(params)
            return dq.to_jax(dq.asqarray(H))
        jac = jax.jacfwd(func)(params)

        def to_qarray(leaf):
            arr = jnp.moveaxis(leaf, (0, 1), (-2, -1))
            return dq.asqarray(arr, dims=qm.dims)

        result = jax.tree_util.tree_map(to_qarray, jac)
        cache[key] = result
        return result

    def H_params_gradient(self, params, psi0, qm, N_g=10, seed=10000):
        dHdp = self.grad_H(params, qm)
        res = self.multiple_measure(dHdp, psi0, seed, N_g)
        # Each leaf of `res` holds raw per-shot samples with the shots axis last: shape
        # (N_g,) for a scalar parameter, but (batch, N_g) for a vector-valued one (e.g.
        # the coupling V, one leaf per bond). Reducing over axis=-1 only (the shots
        # axis) keeps any leading batch axis intact.
        return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), res)

    def H_params_gradient_stats(self, params, psi0, qm, N_g=10, seed=10000):
        """Same shots as H_params_gradient, but returns (mean, var) instead of just mean --
        used by task 5 to report the shot-noise scale directly from the same draw used for
        the gradient, without a separate resampling pass."""
        dHdp = self.grad_H(params, qm)
        res = self.multiple_measure(dHdp, psi0, seed, N_g)
        mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), res)
        var = jax.tree_util.tree_map(lambda x: jnp.var(x, axis=-1), res)
        return mean, var

    def __call__(self, params, qm, target, N_y=10, N_g=10, seed=10):
        return self.call_with_seeds(params, qm, target, N_y=N_y, N_g=N_g,
                                    output_seed=seed, gradient_seed=seed)

    def call_with_seeds(self, params, qm, target, N_y=10, N_g=10,
                        output_seed=10, gradient_seed=10):
        """Finite-shot gradient with independently controllable output/gradient seeds.

        Normal training uses ``__call__`` and changes both together.
        """
        # N_y: number of shots when measuring the output <O_i>
        # N_g: number of shots for calculating the gradient
        H_sys = qm.set_H(params, self.solver.param_funcs)
        psi0 = qm.set_psi0(H_sys(0.))
        psi_free = self.solver(qm, params, psi0, 0., target, target)[-1]

        cost_ops = qm.cost_ops
        if isinstance(cost_ops, (tuple, list)):
            lam = self.measure(cost_ops[0], psi_free, seed=output_seed, shots=N_y)
            y = jnp.stack([lam.mean(), jnp.square(lam).mean()])
        else:
            lam = self.measure(cost_ops, psi_free, seed=output_seed, shots=N_y)
            y = lam.mean()

        return self._sample_gradient_given_output(
            params, qm, target, y, N_g, gradient_seed, psi0, psi_free)

    def call_with_exact_output(self, params, qm, target, N_g=10, gradient_seed=10):
        """Sample only the gradient observables; use exact ground-state ``<C>``.

        This is the estimator required for task 1: the free state is the exact
        ground state and its cost expectation is not replaced by a finite-shot
        output estimate.
        """
        H_sys = qm.set_H(params, self.solver.param_funcs)
        psi0 = qm.set_psi0(H_sys(0.))
        psi_free = self.solver(qm, params, psi0, 0., target, target)[-1]
        y = jnp.real(dq.expect(qm.cost_ops, psi_free))
        return self._sample_gradient_given_output(
            params, qm, target, y, N_g, gradient_seed, psi0, psi_free)

    def _sample_gradient_given_output(self, params, qm, target, y, N_g,
                                      gradient_seed, psi0, psi_free):
        """Shared free/nudged observable sampling for a supplied output value."""
        psi_nudge = self.solver(qm, params, psi0, self.beta, y, target)[-1]

        dHdp_free = self.H_params_gradient(
            params, psi_free, qm, N_g=N_g, seed=gradient_seed + 100)
        dHdp_nudge = self.H_params_gradient(
            params, psi_nudge, qm, N_g=N_g, seed=gradient_seed + 1000)

        params_grad = jax.tree_util.tree_map(self.grad_func, dHdp_nudge, dHdp_free)
        if len(params_grad) > 2:
            # model-specific: for the (Omega, Delta, V) Ising models we don't train
            # the fixed coupling V, so its gradient is zeroed.
            params_grad = params_grad[0], params_grad[1], 0 * params_grad[2]

        return y, params_grad

    def call_with_stats(self, params, qm, target, N_y=10, N_g=10, seed=10):
        """Same as __call__, but additionally returns the free/nudge-phase (mean, var)
        of each dH/dtheta leaf (task 5: track sigma(g_theta) alongside the gradient)."""
        H_sys = qm.set_H(params, self.solver.param_funcs)
        psi0 = qm.set_psi0(H_sys(0.))
        psi_free = self.solver(qm, params, psi0, 0., target, target)[-1]

        cost_ops = qm.cost_ops
        if isinstance(cost_ops, (tuple, list)):
            lam = self.measure(cost_ops[0], psi_free, seed=seed, shots=N_y)
            y = jnp.stack([lam.mean(), jnp.square(lam).mean()])
        else:
            lam = self.measure(cost_ops, psi_free, seed=seed, shots=N_y)
            y = lam.mean()

        psi_nudge = self.solver(qm, params, psi0, self.beta, y, target)[-1]

        mean_free, var_free = self.H_params_gradient_stats(params, psi_free, qm, N_g=N_g, seed=seed + 100)
        mean_nudge, var_nudge = self.H_params_gradient_stats(params, psi_nudge, qm, N_g=N_g, seed=seed + 1000)

        params_grad = jax.tree_util.tree_map(self.grad_func, mean_nudge, mean_free)
        if len(params_grad) > 2:
            params_grad = params_grad[0], params_grad[1], 0 * params_grad[2]

        # Var[(A_nudge - A_free)/beta] = (Var[A_nudge] + Var[A_free]) / (N_g * beta^2)
        # (independent free/nudge draws, sample means of N_g i.i.d. shots each).
        def combine_var(v_n, v_f):
            return (v_n + v_f) / (N_g * self.beta ** 2)
        grad_var = jax.tree_util.tree_map(combine_var, var_nudge, var_free)
        if len(grad_var) > 2:
            grad_var = grad_var[0], grad_var[1], 0 * grad_var[2]

        return y, params_grad, grad_var

    @staticmethod
    def _operator_moments(O, state):
        """Exact expectation and variance for one scalar or batched QArray."""
        arr = dq.to_jax(dq.asqarray(O))
        if arr.ndim == 2:
            mean = jnp.real(dq.expect(O, state))
            second = jnp.real(dq.expect(O @ O, state))
            return mean, jnp.maximum(second - mean ** 2, 0.0)

        leading = arr.shape[:-2]
        flat = arr.reshape((-1,) + arr.shape[-2:])
        means, variances = [], []
        for matrix in flat:
            op = dq.asqarray(matrix, dims=state.dims)
            mean = jnp.real(dq.expect(op, state))
            second = jnp.real(dq.expect(op @ op, state))
            means.append(mean)
            variances.append(jnp.maximum(second - mean ** 2, 0.0))
        return jnp.stack(means).reshape(leading), jnp.stack(variances).reshape(leading)

    def call_with_moments(self, params, qm, target, N_y=10, N_g=10, seed=10):
        """Return ``(y, sampled_gradient, expected_gradient, gradient_variance)``.

        The last two values are exact moments conditional on the one finite-shot
        output estimate ``y`` used to construct the nudged state.  Free and nudged
        measurement groups are independent, so

            Var(g) = (Var_free(A) + Var_nudge(A)) / (N_g * beta**2).
        """
        H_sys = qm.set_H(params, self.solver.param_funcs)
        psi0 = qm.set_psi0(H_sys(0.))
        psi_free = self.solver(qm, params, psi0, 0., target, target)[-1]
        lam = self.measure(qm.cost_ops, psi_free, seed=seed, shots=N_y)
        y = lam.mean()
        psi_nudge = self.solver(qm, params, psi0, self.beta, y, target)[-1]

        dHdp = self.grad_H(params, qm)
        samples_free = self.multiple_measure(dHdp, psi_free, seed + 100, N_g)
        samples_nudge = self.multiple_measure(dHdp, psi_nudge, seed + 1000, N_g)
        sample_mean_free = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), samples_free)
        sample_mean_nudge = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), samples_nudge)
        sampled_grad = jax.tree_util.tree_map(self.grad_func, sample_mean_nudge, sample_mean_free)

        # Build the two trees explicitly.  Returning a Python pair from
        # tree_map would make JAX treat that pair as two new tree nodes, losing
        # the original (Omega, Delta, V) parameter structure.
        derivative_leaves, derivative_structure = jax.tree_util.tree_flatten(
            dHdp, is_leaf=lambda x: isinstance(x, dq.QArray))
        moments_free = [self._operator_moments(O, psi_free) for O in derivative_leaves]
        moments_nudge = [self._operator_moments(O, psi_nudge) for O in derivative_leaves]
        mean_free = jax.tree_util.tree_unflatten(
            derivative_structure, [pair[0] for pair in moments_free])
        var_free = jax.tree_util.tree_unflatten(
            derivative_structure, [pair[1] for pair in moments_free])
        mean_nudge = jax.tree_util.tree_unflatten(
            derivative_structure, [pair[0] for pair in moments_nudge])
        var_nudge = jax.tree_util.tree_unflatten(
            derivative_structure, [pair[1] for pair in moments_nudge])
        expected_grad = jax.tree_util.tree_map(self.grad_func, mean_nudge, mean_free)
        gradient_variance = jax.tree_util.tree_map(
            lambda vn, vf: (vn + vf) / (N_g * self.beta ** 2), var_nudge, var_free)

        if len(sampled_grad) > 2:
            sampled_grad = sampled_grad[0], sampled_grad[1], 0 * sampled_grad[2]
            expected_grad = expected_grad[0], expected_grad[1], 0 * expected_grad[2]
            gradient_variance = (gradient_variance[0], gradient_variance[1],
                                 0 * gradient_variance[2])
        return y, sampled_grad, expected_grad, gradient_variance
