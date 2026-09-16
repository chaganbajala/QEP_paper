"""Fixed-N_g training loop, standalone copy for this study (no adaptive beta/N_g scheme
-- the research plan asks explicitly for fixed beta=0.1, eta=0.01, fixed N_g, no
adaptive methods).

`Shot_QEP_Training` is ported verbatim from Adaptive_bn/qep_shot/train.py (plain
gradient descent with a shot-noise-based grad_method; per-epoch reseeding via
`seed + k * 10000` so shot noise is i.i.d. across epochs, not replayed).

`Shot_QEP_Training_WithStats` is new here (task 5): identical training dynamics, but
at each epoch also records the exact conditional gradient mean and variance via
`grad_method.call_with_moments`.
"""
import time

import jax

try:
    from . import _pathfix  # noqa: F401  (side effect: makes `QEP` importable)
except ImportError:
    import _pathfix  # noqa: F401  (fallback: run/imported as a flat script, not a package)
from QEP import train as _qep_train


class Shot_QEP_Training(_qep_train.QEP_Training):
    """Plain gradient descent with a shot-noise-based grad_method."""

    def train(self, params_0, N_epoch, target, grad_method, qm, N_y=10, N_g=10, seed=0, verbose=True):
        paramsL = [params_0]
        costL = []
        run_params = params_0

        for k in range(0, N_epoch):
            t0 = time.time()
            y, grad_params = grad_method(run_params, qm, target, N_y, N_g, seed + k * 10000)
            run_params = self.update(run_params, grad_params)
            paramsL.append(run_params)
            costL.append(y)
            t1 = time.time()
            if verbose:
                print(k, "th round completed. Time concumption: ", t1 - t0, "s, Magnetization:", y)

        return paramsL, costL


class Shot_QEP_Training_WithStats(_qep_train.QEP_Training):
    """Same dynamics as Shot_QEP_Training, but grad_method must be a
    grad.Shot_QEP_Grad (its call_with_stats method is used) and each epoch additionally
    records the per-parameter gradient (mean, var) alongside the trajectory."""

    def train(self, params_0, N_epoch, target, grad_method, qm, N_y=10, N_g=10, seed=0, verbose=True):
        paramsL = [params_0]
        costL = []
        grad_sampleL = []
        grad_meanL = []
        grad_varL = []
        run_params = params_0

        for k in range(0, N_epoch):
            t0 = time.time()
            y, grad_params, grad_mean, grad_var = grad_method.call_with_moments(
                run_params, qm, target, N_y, N_g, seed + k * 10000)
            run_params = self.update(run_params, grad_params)
            paramsL.append(run_params)
            costL.append(y)
            grad_sampleL.append(grad_params)
            grad_meanL.append(grad_mean)
            grad_varL.append(grad_var)
            t1 = time.time()
            if verbose:
                print(k, "th round completed. Time concumption: ", t1 - t0, "s, Magnetization:", y)

        return paramsL, costL, grad_sampleL, grad_meanL, grad_varL
