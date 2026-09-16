import jax
import jax.numpy as jnp

import diffrax
import numpy as np
import dynamiqs as dq

import qutip
import matplotlib.pyplot as plt

from functools import partial
import time

import pickle



#===================================Method Used for Getting the Ground State=========================
class Exact_Diagonalization:
    def __init__(self, param_funcs):
        self.param_funcs = param_funcs
        
    #@partial(jax.jit, static_argnames=["self", "qm"])
    def __call__(self, qm, params, psi0=0, beta=0, y=0, target=0):
        H = qm.total_H(params, self.param_funcs, y, target, beta)(qm.T)
        psi = qm.set_psi0(H)
        return [psi]
    
class Adiabatic_Evolution:
    def __init__(self, param_funcs):
        self.param_funcs = param_funcs
        
    @partial(jax.jit, static_argnames=["self", "qm"])
    def __call__(self, qm, params, psi0, beta, y, target):
        H = qm.total_H(params, self.param_funcs, y, target, beta)
        return dq.sesolve(H, psi0, qm.tlist, exp_ops=[], options=dq.Options(progress_meter=False)).states
    
class Adiabatic_Evolution_Nudge(Adiabatic_Evolution):
    def __init__(self, param_funcs, nudge_func):
        super().__init__(param_funcs)
        self.nudge_func = nudge_func
        
    @partial(jax.jit, static_argnames=["self", "qm"])
    def __call__(self, qm, params, psi0, beta, y, target):
        H_tot = qm.set_H(params, self.param_funcs)(qm.T)
        H = qm.nudge_H(H_tot, beta, y, target, self.nudge_func)
        return dq.sesolve(H, psi0, qm.tlist_nudge, exp_ops=[], options=dq.Options(progress_meter=False)).states
    
#=================================Get Gradient with QEP=======================

class QEP_grad:
    def __init__(self, beta, solver: Adiabatic_Evolution):
        self.beta = beta
        self.solver = solver
    
    def H_exp(self, params, psi0, qm):
        H = qm.set_H(params, self.solver.param_funcs)
        return jnp.real(dq.expect(H(qm.T), psi0))
    
    @partial(jax.jit, static_argnames=["self", "qm"])
    def H_params_gradient(self, params, psi0, qm):
        # Get <\partial_\theta H >
        return jax.grad(self.H_exp, 0)(params, psi0, qm)
    
    @partial(jax.jit, static_argnames=["self"])
    def grad_func(self, x, y):
        return (x-y)/self.beta
    
    def __call__(self, params, qm, target):
        t00 = time.time()
        H_sys = qm.set_H(params, self.solver.param_funcs)
        t0 = time.time()
        psi0 = qm.set_psi0(H_sys(0.))
        t1 = time.time()
        psi_free = self.solver(qm, params, psi0, 0., target, target)[-1]
        
        y = dq.expect(qm.cost_ops, psi_free)
        t2 = time.time()
        
        psi_nudge = self.solver(qm, params, psi0, self.beta, y, target)[-1]
        t3 = time.time()
        
        dHdp_free = self.H_params_gradient(params, psi_free, qm)
        dHdp_nudge = self.H_params_gradient(params, psi_nudge, qm)
        
        t31 = time.time()
        params_grad = jax.tree_util.tree_map(self.grad_func, dHdp_nudge, dHdp_free)
        t32 = time.time()
        #params_grad = params_grad[0], params_grad[1], params_grad[2]
        t4 = time.time()
        
        #print(t0-t00, t1-t0, t2-t1, t3-t2, t31-t3, t32-t31, t4-t32)
        
        return y, params_grad
    
class Cached_QEP_grad(QEP_grad):
    
    def __init__(self, beta, solver=(Adiabatic_Evolution, Adiabatic_Evolution_Nudge)):
        super().__init__(beta, solver)
        self.free_solver, self.nudge_solver = solver
        
    def H_exp(self, params, psi0, qm):
        H = qm.set_H(params, self.free_solver.param_funcs)
        return jnp.real(dq.expect(H(qm.T), psi0))
        
    def __call__(self, params, qm, target):
        
        t00 = time.time()
        H_sys = qm.set_H(params, self.free_solver.param_funcs)
        t0 = time.time()
        psi0 = qm.set_psi0(H_sys(0.))
        t1 = time.time()
        
        #H_tot = qm.total_H(params, self.solver.param_funcs, y, target, 0.)
        psi_free = self.free_solver(qm, params, psi0, 0., 0., target)[-1]
        
        y = dq.expect(qm.cost_ops, psi_free)
        t2 = time.time()
        
        psi_nudge = self.nudge_solver(qm, params, psi_free, self.beta, y, target)[-1]
        t3 = time.time()
        
        dHdp_free = self.H_params_gradient(params, psi_free, qm)
        dHdp_nudge = self.H_params_gradient(params, psi_nudge, qm)
        
        t31 = time.time()
        params_grad = jax.tree_util.tree_map(self.grad_func, dHdp_nudge, dHdp_free)
        t32 = time.time()
        #params_grad = params_grad[0], params_grad[1], params_grad[2]
        t4 = time.time()
        
        #print(t0-t00, t1-t0, t2-t1, t3-t2, t31-t3, t32-t31, t4-t32)
        return y, params_grad
    