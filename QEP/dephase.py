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
import gc


import QEP
from QEP import model, grad, train

# This includes codes for implementing dephasing systems

class Add_Dephase:
    # This add decoherence to the system. The dephasing rate is determined by the parameter gamma
    def __init__(self, obj, T_relax, gamma_list=None, dt=0.1):
        # obj: coherent network
        self.T_relax = T_relax
        self.obj = obj
        self.relax_tlist = jnp.arange(0, T_relax, dt)
        self.gamma_list = gamma_list if gamma_list is not None else jnp.ones(obj.N, dtype=jnp.float32)
        self.prepare_dephasing_ops(obj.N, obj.dims[0])
        

    def prepare_dephasing_ops(self, N, dim=2, dtype=jnp.complex64, device=None):
        # 1) define annihiator operator for a single qubit (shape 2x2 for example)
        # Since the number operator is defined as |up><up| = [[1,0],[0,0]] = (1+\sigma_z)/2, the annihilator should by [[0,0],[1,0]], which is the 
        # default creator in dynamiqs and many quantum computing literature, where |up> is taken as |0>. 
        annihilator = (dq.sigmaz() + dq.eye(dim))/2

        # 2) precompute identity powers: id_pows[k] = I^(k)  (shape 2^k x 2^k)
        # Start id_pows[0] = 1x1 identity (scalar identity matrix)
        eye2 = dq.eye(dim)

        ids = [eye2]
        for k in range(1, N):
            ids.append(dq.tensor(ids[-1], eye2))

        def insert_op(k, op, ids):
            if k==0:
                left_op = op
                return dq.tensor(left_op, ids[N-2])
            elif k==N-1:
                left_op = dq.tensor(ids[k-1], op)
                return left_op
            else:
                left_op = dq.tensor(ids[k-1], op)
                return dq.tensor(left_op, ids[N-2-k])

        As = [jnp.sqrt(self.gamma_list[k]) * insert_op(k, annihilator , ids) for k in range(N)]

        # 4) optional device placement (do this once, not inside loops)
        self.Ls = As

    def __getattr__(self, name):
        # forward unknown attributes/methods to wrapped object
        return getattr(self.obj, name)
    

class Add_OnD:
    # This add decoherence to the system. The dephasing rate is determined by the parameter gamma
    # One can also set the dephasing operators.
    def __init__(self, obj, T_relax, Ls, gamma_list=None, dt=0.1):
        # obj: coherent network
        # Ls: list of dephasing operators expressed as local operators
        self.T_relax = T_relax
        self.obj = obj
        self.relax_tlist = jnp.arange(0, T_relax, dt)
        self.gamma_list = gamma_list if gamma_list is not None else jnp.ones(obj.N, dtype=jnp.float32)
        self.raw_Ls = Ls
        self.prepare_dephasing_ops(obj.N, obj.dims[0])
        

    def prepare_dephasing_ops(self, N, dim=2, dtype=jnp.complex64, device=None):
        # 1) define annihiator operator for a single qubit (shape 2x2 for example)
        # Since the number operator is defined as |up><up| = [[1,0],[0,0]] = (1+\sigma_z)/2, the annihilator should by [[0,0],[1,0]], which is the 
        # default creator in dynamiqs and many quantum computing literature, where |up> is taken as |0>. 
        annihilator = (dq.sigmaz() + dq.eye(dim))/2

        # 2) precompute identity powers: id_pows[k] = I^(k)  (shape 2^k x 2^k)
        # Start id_pows[0] = 1x1 identity (scalar identity matrix)
        eye2 = dq.eye(dim)

        ids = [eye2]
        for k in range(1, N):
            ids.append(dq.tensor(ids[-1], eye2))

        def insert_op(k, op, ids):
            if k==0:
                left_op = op
                return dq.tensor(left_op, ids[N-2])
            elif k==N-1:
                left_op = dq.tensor(ids[k-1], op)
                return left_op
            else:
                left_op = dq.tensor(ids[k-1], op)
                return dq.tensor(left_op, ids[N-2-k])

        As = [jnp.sqrt(self.gamma_list[k]) * insert_op(k, self.raw_Ls[k], ids) for k in range(N)]

        # 4) optional device placement (do this once, not inside loops)
        self.Ls = As

    def __getattr__(self, name):
        # forward unknown attributes/methods to wrapped object
        return getattr(self.obj, name)
    

class Decoherent_Adiabatic_Evolution(grad.Adiabatic_Evolution):
    @partial(jax.jit, static_argnames=["self", "dcqm"])
    def __call__(self, dcqm, params, rho0, beta, y, target):
        H_t = dcqm.total_H(params, self.param_funcs, y, target, beta)
        return dq.mesolve(H_t, dcqm.Ls, rho0, tsave=dcqm.relax_tlist).states
    

class Big_DAE_run(Decoherent_Adiabatic_Evolution):
    def op_prod(self, scalar, op):
        return scalar * op
    def dephase_rho(self, rho, L):
        return L@rho@L.dag() - 0.5*(L.dag()@L@rho + rho@L.dag()@L)
    def liouvillian(self, dcqm, rho, H):
        rho1 = - 1j * (H@rho - rho@H) + sum([dcqm.gamma_list[k] * self.dephase_rho(rho, dcqm.Ls[k]) for k in range(len(dcqm.gamma_list))])
        return rho1
    
    @partial(jax.jit, static_argnames=["self", "dcqm"])
    def update(self, dcqm, t, rho0, H_t):
        H1 = H_t(t)
        H2 = H_t(t + 0.5 * dcqm.dt)

        k1 = self.liouvillian(dcqm, rho0, H1)
        rho1 = rho0 + 0.5 * dcqm.dt * k1
        k2 = self.liouvillian(dcqm, rho1, H2)
        return rho0 + dcqm.dt * k2

    def __call__(self, dcqm, params, rho0, beta, y, target):
        H_t = dcqm.total_H(params, self.param_funcs, y, target, beta)
        N_itr = len(dcqm.relax_tlist)
        rho = rho0
        for k in range(N_itr):
            rho = self.update(dcqm, dcqm.relax_tlist[k], rho, H_t)
        return rho
    
class Decoherent_QEP_grad:
    def __init__(self, beta, solver: Decoherent_Adiabatic_Evolution):
        self.beta = beta
        self.solver = solver
    
    def H_exp(self, params, rho0, dcqm):
        H = dcqm.set_H(params, self.solver.param_funcs)(dcqm.T)
        return jnp.real(dq.trace(H @ rho0))
    
    @partial(jax.jit, static_argnames=["self", "dcqm"])
    def H_params_gradient(self, params, rho0, dcqm):
        return jax.grad(self.H_exp, 0)(params, rho0, dcqm)
    
    @partial(jax.jit, static_argnames=["self"])
    def grad_func(self, x, y):
        return (x-y)/self.beta
    
    def __call__(self, params, dcqm, target):
        t00 = time.time()
        H_sys = dcqm.set_H(params, self.solver.param_funcs)
        t0 = time.time()
        gs_rho0 = dq.todm(dcqm.set_psi0(H_sys(0.)))
        rho0 = dq.mesolve(H_sys(0.), dcqm.Ls, gs_rho0, tsave=dcqm.relax_tlist, options=dq.Options(progress_meter=False)).states[-1]
        t1 = time.time()
        rho_free = self.solver(dcqm, params, rho0, 0., 0., target)[-1]
        
        y = dq.trace(dcqm.cost_ops @ rho_free)
        t2 = time.time()
        
        rho_nudge = self.solver(dcqm, params, rho0, self.beta, y, target)[-1]
        t3 = time.time()
        
        dHdp_free = self.H_params_gradient(params, rho_free, dcqm)
        dHdp_nudge = self.H_params_gradient(params, rho_nudge, dcqm)
        
        t31 = time.time()
        params_grad = jax.tree_util.tree_map(self.grad_func, dHdp_nudge, dHdp_free)
        t32 = time.time()
        #params_grad = params_grad[0], params_grad[1], params_grad[2]
        t4 = time.time()
        
        #print(t0-t00, t1-t0, t2-t1, t3-t2, t31-t3, t32-t31, t4-t32)
        
        return y, params_grad