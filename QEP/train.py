import jax
import jax.numpy as jnp

import diffrax
import numpy as np
import dynamiqs as dq
import optax

import qutip
import matplotlib.pyplot as plt

from functools import partial
import time

import pickle

class QEP_Training:
    # Training with simple gradient descent
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.method = "gradient descent"
        
    def add_grad(self, x, gx):
        return x - self.learning_rate * gx
    
    def update(self, params, grad_params):
        return jax.tree_util.tree_map(self.add_grad, params, grad_params)
    
    def train(self, params_0, N_epoch, target, grad_method, qm):
        paramsL = [params_0]
        costL = []
        
        run_params = params_0
        
        for k in range(0, N_epoch):
            t0 = time.time()
            y, grad_params = grad_method(run_params, qm, target)
            run_params = self.update(run_params, grad_params)
            paramsL.append(run_params)
            costL.append(y)
            t1 = time.time()
            
            print(k, "th round completed. Time concumption: ", t1-t0, "s, Magnetization:", y)
            
        return paramsL, costL


class T_Scheduled_Training(QEP_Training):
    """Gradient descent with a geometric T-schedule.

    Evolution time starts at T_0 and grows by factor `r` each epoch,
    capped at T_max.  Early epochs use cheap, noisy gradients; precision
    increases as the optimisation converges.

    Args:
        learning_rate: gradient descent step size.
        T_0:           initial evolution time.
        T_max:         maximum (and final) evolution time.
        r:             geometric growth factor per epoch (r > 1).
    """

    def __init__(self, learning_rate, T_0, T_max, r=1.1):
        super().__init__(learning_rate)
        self.T_0 = T_0
        self.T_max = T_max
        self.r = r

    def _update_qm_T(self, qm, T):
        """Update qm's evolution time and time grid in-place."""
        dt = float(qm.tlist[1] - qm.tlist[0])
        qm.T = T
        qm.tlist = jnp.arange(0, T, dt)

    def train(self, params_0, N_epoch, target, grad_method, qm):
        paramsL = [params_0]
        costL = []
        TL = []

        run_params = params_0

        for k in range(N_epoch):
            T_k = float(min(self.T_0 * (self.r ** k), self.T_max))
            self._update_qm_T(qm, T_k)

            t0 = time.time()
            y, grad_params = grad_method(run_params, qm, target)
            run_params = self.update(run_params, grad_params)
            paramsL.append(run_params)
            costL.append(y)
            TL.append(T_k)
            t1 = time.time()

            print(k, "th round completed. T =", round(T_k, 3),
                  " Time consumption:", t1 - t0, "s, Magnetization:", y)

        return paramsL, costL, TL


class Optax_Training:
    """Training loop backed by any optax optimizer.

    Args:
        optimizer: an optax.GradientTransformation (e.g. optax.adam(lr)).
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def train(self, params_0, N_epoch, target, grad_method, qm):
        opt_state = self.optimizer.init(params_0)
        run_params = params_0
        paramsL = [params_0]
        costL = []

        for k in range(N_epoch):
            t0 = time.time()
            y, grad_params = grad_method(run_params, qm, target)
            updates, opt_state = self.optimizer.update(grad_params, opt_state, run_params)
            run_params = optax.apply_updates(run_params, updates)
            paramsL.append(run_params)
            costL.append(y)
            t1 = time.time()

            print(k, "th round completed. Time consumption:", t1 - t0, "s, Magnetization:", y)

        return paramsL, costL


class Adam_Training(Optax_Training):
    """Training with the Adam optimizer.

    Args:
        learning_rate: Adam step size.
        b1:            decay rate for first moment (default 0.9).
        b2:            decay rate for second moment (default 0.999).
        eps:           numerical stability constant (default 1e-8).
    """

    def __init__(self, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(optax.adam(learning_rate, b1=b1, b2=b2, eps=eps))