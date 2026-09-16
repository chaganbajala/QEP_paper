"""Plain gradient-descent training loop (same as ``QEP.train.QEP_Training``)."""
import time

import jax
import numpy as np

__all__ = ["QEP_Training"]


class QEP_Training:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def _step(self, x, gx):
        return x - self.learning_rate * gx

    def update(self, params, grad_params):
        return jax.tree_util.tree_map(self._step, params, grad_params)

    def train(self, params_0, N_epoch, target, grad_method, qm,
              verbose_every=0, log=None):
        paramsL = [params_0]
        costL = []
        run_params = params_0
        t_start = time.time()
        for k in range(N_epoch):
            y, grad_params = grad_method(run_params, qm, target)
            run_params = self.update(run_params, grad_params)
            paramsL.append(run_params)
            costL.append(float(np.real(np.asarray(y))))
            if verbose_every and (k % verbose_every == 0 or k == N_epoch - 1):
                msg = (f"  epoch {k+1}/{N_epoch}  y={costL[-1]:+.6f}  "
                       f"Omega={float(run_params[0]):+.4f} Delta={float(run_params[1]):+.4f}"
                       f"  [{time.time()-t_start:.1f}s]")
                print(msg, flush=True)
                if log is not None:
                    log.write(msg + "\n")
                    log.flush()
        return paramsL, costL
