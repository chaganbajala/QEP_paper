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

set_64 = False
jax.config.update("jax_enable_x64", set_64)
if set_64:
    complex_type = jnp.complex128
else:
    complex_type = jnp.complex64

#cpu_device = jax.devices('cpu')[0]
#gpu_device = jax.devices('gpu')[0]

#====================Functions to Get Ground State======================

def Lanczos_GS(H:jnp.ndarray, v0, itr_num=100):
    # One should rewrite it with sparse expression
    v = v0/jnp.linalg.norm(v0)
    for k in range(0, itr_num):
        v = H@v
        v = v/jnp.linalg.norm(v)
    
    eigv = v@H@v
    return eigv, v

def eigv_func(v, H):
    return jnp.real((v@H@v)/(v@v))

@jax.jit
def update_v(v0, lg, H, eta=0.05, r=0.9):
    gv = jax.grad(eigv_func, 0)(v0, H)
    tg = lg*r + gv
    v = v0 - eta * tg
    return v/jnp.linalg.norm(v), tg


def get_gs(H, qm, itr_num=100, eta=0.05, r=0.9):
    
    v = jnp.ones(H.shape[0])
    v = v/jnp.linalg.norm(v)
    lg = 0.
    for k in range(0,itr_num):
        v, lg = update_v(v, lg, H, eta=eta, r=r)
        #print(dq.overlap(psi0_qt, dq.asqarray(v[:,None], dims = qm.dims)))
    return dq.asqarray(jnp.asarray(v[:,None], dtype=complex_type), dims = qm.dims)

#----------------Calculate Ground State with Sparse Expression of Matrix---------------
@jax.jit
def sparse_update(v0, lg, H, eta=0.05, r=0.9):
    gv = 2*((dq.norm(v0)**2) * H@v0 - dq.expect(H, v0)*v0)/((dq.norm(v0))**4)
    tg = lg*r + gv
    v = v0 - eta * tg
    return v/dq.norm(v), tg

def get_gs_sparse(H, qm, itr_num=100, eta=0.05, r=0.9):
    v = jnp.ones(H.shape[0])
    v = v/jnp.linalg.norm(v)
    v = dq.asqarray(v[:,None], dims=qm.dims)
    #print(v.dims)
    lg = 0.* v
    for k in range(0, itr_num):
        v, lg = sparse_update(v, lg, H, eta=eta, r=r)
    return v


#==============================Models===============================

class Quantum_Model:
    # This implement a quantum system
    def __init__(self, N, T, dt):
        # N: number of sides
        self.N = N
        self.T = T
        #print(T)
        self.tlist = np.arange(0, T, dt)
        self.prepare_ops(N)
        
        self.tlist = jnp.arange(0, T, dt)
        self.COs = self.set_coupling_ops()
        self.measure_ops = self.set_measure_ops()
        self.dims = []
        for k in range(0, self.N):
            self.dims.append(2)
        self.dims = tuple(self.dims)
        
    def prepare_ops(self, N):
        # This prepare the necessary matrix for the system
        def insert_op(op, k, N, dim=2):
            if k==0:
                A_out = op
                for l in range(1,N):
                    A_out = qutip.tensor(A_out, qutip.qeye(dim))
                return A_out
            else:
                A_out = qutip.qeye(dim)
                for l in range(1,k):
                    A_out = qutip.tensor(A_out, qutip.qeye(dim))
                A_out = qutip.tensor(A_out, op)
                for l in range(k+1, N):
                    A_out = qutip.tensor(A_out, qutip.qeye(dim))
                return A_out
            
        Xs = []
        Ys = []
        Zs = []
        Ns = []
        for k in range(0, N):
            Xs.append(insert_op(qutip.sigmax(), k, N))
            Ys.append(insert_op(qutip.sigmay(), k, N))
            Zs.append(insert_op(qutip.sigmaz(), k, N))
            Ns.append(insert_op((1. + qutip.sigmaz())/2, k, N))
            
        self.block_ops = [Xs, Ys, Zs, Ns]
        
    def set_coupling_ops(self):
        COs = []
        Ns = self.block_ops[3]
        for k in range(self.N-1):
            COs.append(Ns[k]@Ns[k+1])

        return COs
    
    def set_measure_ops(self):
        M = sum(self.block_ops[2])/self.N
        AM = sum([((-1)**k) * self.block_ops[2][k] for k in range(0, self.N)])/self.N
        return M, AM
        
    def get_internal_H(self, params):
        _, _, V = params
        S = 0.
        Ns = self.block_ops[3]
        for k in range(0, self.N-1):
            S = S + 0.5 * V[k] * Ns[k]@Ns[k+1]
            
        S = sum([0.5 * V[k] * self.COs[k] for k in range(0, self.N-1)])
            
        return S
    
    def set_params(self, Omega, Delta, Vs=None, seed=0):
        if Vs==None:
            key = jax.random.PRNGKey(seed)
            Vs = jax.random.uniform(key, shape=(self.N-1,), minval=0., maxval=2.)
            Vs = np.ones(self.N-1)
        return Omega, Delta, Vs
        
    @staticmethod
    def func_Omega(t, Omega, T):
        Omega_0 = 1.
        y1 = Omega_0 * t/(0.2*T)
        y2 = Omega_0
        y3 = Omega_0 + (t-0.8*T)/(0.2*T)*(Omega-Omega_0)
        return np.min(np.array([y1, y2, y3]))

    @staticmethod
    def func_Delta(t, Delta, T):
        D0 = -1.
        y1 = D0
        y2 = (-D0+Delta) / (0.6*T) * (t - 0.2*T) + D0
        y3 = Delta
        return np.min(np.array([np.max(np.asarray([y1, y2])), y3]))
    
    #@partial(jax.jit, static_argnames=["self"])
    def set_H(self, params):
        H_int = self.get_internal_H(params)
        Omega, Delta, _ = params
        
        Xs, Ns = self.block_ops[0], self.block_ops[3]
        
            
        H_Omega = sum(0.5 * Xs[k] for k in range(self.N))
        H_Delta = sum(-Ns[k] for k in range(self.N))
            
        #H = [[H_Omega, func_Omega], [H_Delta, func_Delta], H_int]
        fOmega = partial(self.func_Omega, Omega=Omega, T=self.T)
        fDelta = partial(self.func_Delta, Delta=Delta, T=self.T)

        #H = dq.modulated(fOmega, H_Omega) + dq.modulated(fDelta, H_Delta) + H_int
            
        H = [[H_Omega, fOmega], [H_Delta, fDelta], H_int]
        return H
    
    def set_psi0(self, H):
        def generate_psi0(N, state):
            psi0 = state
            for k in range(1, N):
                psi0 = qutip.tensor(psi0, state)
                
            return psi0
        
        state = qutip.states.basis(2,1)
        #psi0 = generate_psi0(self.N, state)
        psi0 = H.groundstate()[1]
        return psi0
    
    def evolve(self, H, psi0):
        result = qutip.sesolve(H, psi0, self.tlist, _e_ops = [])
        return result
    
    def measure_op(self, state, op):
        return qutip.expect(op, state)
    
    def lab_measure_op(self, state, op, N_measure = 1):
        # This mimic the process of measurement in lab. 
        # The output is random
        return np.mean([qutip.measurement.measure_observable(state, op)[0] for k in range(N_measure)])
    
    def energy_decompose(self, H, state:qutip.Qobj, n=2):
        E_mean = qutip.expect(H, state)
        
        eigenvals, eigenkets = np.real(H.eigenstates(sparse=True, eigvals=0)[0:n])
        
        overlaps = [state.overlap(eigenkets[k]) for k in range(n)]
        
        return E_mean, eigenvals, overlaps
    
    def get_ground_state(self, H:qutip.Qobj):
        return H.groundstate()
    
    def __hash__(self):
        return hash((self.N, len(self.tlist)))

    def __eq__(self, other):
        return self.N == other.N and len(self.tlist) == len(other.tlist)

    def show_Omega_Delta(self, Omega=0., Delta=1):
        fo = lambda t: self.func_Omega(t, Omega, self.T)
        fd = lambda t: self.func_Delta(t, Delta, self.T)
        plt.plot(self.tlist, [fo(t) for t in self.tlist])
        plt.plot(self.tlist, [fd(t) for t in self.tlist])
        plt.show()
        
        
def evaluate_H(H, t0):
    Ht = 0.
    for term in H:
        if isinstance(term, list):
            op, f = term
            Ht += op * f(t0)
        else:  # static operator
            Ht += term
    return Ht


class Quantum_Model_DQ(Quantum_Model): #Implement quantum system with dynamiqs
    def __init__(self, N, T, dt):

        self.N = N
        self.T = T
        tm = time.time()
        #self.prepare_ops(N)
        tm1 = time.time()
        self.prepare_ops_dynamiqs(N)
        t0 = time.time()
        self.tlist = jnp.arange(0, T, dt)
        self.dt = dt

        self.dims = []
        for k in range(0, self.N):
            self.dims.append(2)
        self.dims = tuple(self.dims)
        
        self.tlist = jnp.arange(0, T, dt)
        t1 = time.time()
        self.COs = self.set_coupling_ops()
        t2 = time.time()
        self.measure_ops = self.set_measure_ops()
        t3 = time.time()
        self.dims = []
        t4 = time.time()
        for k in range(0, self.N):
            self.dims.append(2)
        t5 = time.time()
        self.dims = tuple(self.dims)
        t6 = time.time()
        self.cost_ops = self.get_cost_ops()
        t7 = time.time()
        
        print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t0-tm1, tm1-tm)
    
    def set_measure_ops(self):
        M = sum(self.block_ops[2])/self.N
        AM = sum([((-1)**k) * self.block_ops[2][k] for k in range(0, self.N)])/self.N
        return M, AM
        
    def prepare_ops(self, N):

        def insert_op(op, k, N, dim=2):
            if k==0:
                A_out = op
                for l in range(1,N):
                    A_out = dq.tensor(A_out, dq.eye(dim))
                return A_out
            else:
                A_out = dq.eye(dim)
                for l in range(1,k):
                    A_out = dq.tensor(A_out, dq.eye(dim))
                A_out = dq.tensor(A_out, op)
                for l in range(k+1, N):
                    A_out = dq.tensor(A_out, dq.eye(dim))
                return A_out
            
        Xs = []
        Ys = []
        Zs = []
        Ns = []
        '''
        for k in range(0, N):
            Xs.append(insert_op(dq.sigmax(), k, N))
            Ys.append(insert_op(dq.sigmay(), k, N))
            Zs.append(insert_op(dq.sigmaz(), k, N))
            Ns.append(insert_op((dq.eye(2) + dq.sigmaz())/2, k, N))
        '''
            
        Xs = [insert_op(dq.sigmax(), k, N) for k in range(0,N)]
        Ys = [insert_op(dq.sigmay(), k, N) for k in range(0,N)]
        Zs = [insert_op(dq.sigmaz(), k, N) for k in range(0,N)]
        Ns = [insert_op((dq.eye(2) + dq.sigmaz())/2, k, N) for k in range(0,N)]

        #Xs = [jax.device_put(op, gpu_device) for op in Xs]
        #Ys = [jax.device_put(op, gpu_device) for op in Ys]
        #Zs = [jax.device_put(op, gpu_device) for op in Zs]
        #Ns = [jax.device_put(op, gpu_device) for op in Ns]
            
        self.block_ops = [Xs, Ys, Zs, Ns]
        
    def prepare_ops_dynamiqs(self, N, dim=2, dtype=jnp.complex64, device=None):
        
        # 1) single-site ops (cast to requested dtype)
        sx = dq.sigmax()
        sy = dq.sigmay()
        sz = dq.sigmaz()
        n_op = ((dq.eye(dim) + sz) / 2)

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

        Xs = [insert_op(k, sx, ids) for k in range(N)]
        Ys = [insert_op(k, sy, ids) for k in range(N)]
        Zs = [insert_op(k, sz, ids) for k in range(N)]
        Ns = [insert_op(k, n_op, ids) for k in range(N)]

        # 4) optional device placement (do this once, not inside loops)
        self.block_ops = [Xs, Ys, Zs, Ns]
        
        self.H_Omega = sum(0.5 * Xs[k] for k in range(self.N))
        self.H_Delta = sum(-Ns[k] for k in range(self.N))
        
    def set_params(self, Omega, Delta, Vs=None, seed=0):
        if Vs==None:
            key = jax.random.PRNGKey(seed)
            #Vs = jax.random.uniform(key, shape=(len(self.COs),), minval=0., maxval=2.)
            Vs = jnp.ones(len(self.COs))
        self.H_int = self.get_internal_H((Omega, Delta, Vs))
        return Omega, Delta, Vs
        
    def set_coupling_ops(self):
        COs = []
        Ns = self.block_ops[3]
        for k in range(self.N-1):
            COs.append(Ns[k]@Ns[k+1])

        #self.block_ops = jax.device_put(self.block_ops, cpu_device)
        return COs
        
    def set_psi0(self, H, itr_num=100):
        #print(itr_num)
        def generate_psi0(N, state):
            psi0 = state
            for k in range(1, N):
                psi0 = dq.tensor(psi0, state)
                
            return psi0
        
        #state = qutip.states.basis(2,1)
        #psi0 = generate_psi0(self.N, state)
        #psi0 = dq.asqarray(H.to_qutip().groundstate()[1].full(), dims=self.dims)
        #psi0 = self.get_ground_state(H)
        psi0 = get_gs_sparse(H, self, itr_num)
        return psi0
    
    
    def get_internal_H(self, params):
        _, _, V = params
        
        def scalar_multiply(a, op):
            return a * op
        
        S = 0.5 * sum(jax.tree_util.tree_map(scalar_multiply, list(V), self.COs))

        return S

    def set_H_f(self, params):
        # Directly give the Hamiltonian at the end of adiabatic evolution.
        Omega, Delta, Vs = params
        S = self.get_internal_H(params)
        return Omega * self.H_Omega + Delta * self.H_Delta + S
    
    #@partial(jax.jit, static_argnames=["self"])
    def set_H(self, params, funcs):
        #H_int = self.get_internal_H(params)
        Omega, Delta, _ = params
        fOmega, fDelta = funcs
        
        Xs, Ns = self.block_ops[0], self.block_ops[3]
        
            
        #H_Omega = sum(0.5 * Xs[k] for k in range(self.N))
        #H_Delta = sum(-Ns[k] for k in range(self.N))

        H = dq.modulated(lambda t: fOmega(t, Omega, self.T), self.H_Omega) + dq.modulated(lambda t: fDelta(t, Delta, self.T), self.H_Delta) + self.H_int
        #H = dq.modulated(self.func_Omega, H_Omega) + dq.modulated(self.func_Delta, H_Delta) + H_int
        return H
    
    #@partial(jax.jit, static_argnames=["self"])
    def evolve(self, params):
        #self.H_int = self.get_internal_H(params)
        Omega, Delta, _ = params
        fOmega = lambda t: self.func_Omega(t, Omega, self.T)
        fDelta = lambda t: self.func_Delta(t, Delta, self.T)
        H = self.set_H(params, (fOmega, fDelta))
        psi0 = self.set_psi0(H(0))
        result = dq.sesolve(H, psi0, self.tlist, exp_ops=[])
        return result
    
    def measure_op(self, state, op):
        state_qt = state.to_qutip()
        op_qt = op.to_qutip()
        return super().measure_op(state_qt, op_qt)
    
    def lab_measure_op(self, state, op, N_measure=1):
        return super().lab_measure_op(state.to_qutip(), op.to_qutip(), N_measure)
    
    def energy_decompose(self, H, state: dq.QArray, n=2):
        H_qt = H.to_qutip()
        state_qt = state.to_qutip()
        return super().energy_decompose(H_qt, state_qt)
    
    def get_ground_state(self, H:dq.QArray, itr_num=100):
        H_jax = H.to_jax()
        v0 = jnp.ones(H_jax.shape[0])
        #E, v = Lanczos_GS(H_jax, v0, itr_num=itr_num)
        v = get_gs(H.to_jax(), self, itr_num=itr_num)
        return v
    
    def show_Omega_Delta(self, Omega, Delta):
        fo = lambda t: self.func_Omega(t, Omega, self.T)
        fd = lambda t: self.func_Delta(t, Delta, self.T)
        plt.plot(self.tlist, jax.vmap(fo)(self.tlist))
        plt.plot(self.tlist, jax.vmap(fd)(self.tlist))
        plt.show()
        
    #=============This Calculate the parameter Gradient============
    def cost_func(self, y, target):
        return jnp.real(jnp.sum((y-target)**2))
    
    def get_cost_ops(self):
        # This set the Hamiltonian implementing the cost function
        # During the nudge phase, the Hamiltonian is like H = H_0 + \beta * \sum_i \epsilon_i C_i
        # C_i is the output of this function
        
        AM = sum([((-1)**k) * self.block_ops[2][k] for k in range(0, self.N)])/self.N
        return AM
    
    def total_H(self, params, funcs, y, target, beta):
        H_sys = self.set_H(params, funcs)
        error_signal = jax.grad(self.cost_func, 0)(y, target)
        H_tot = H_sys + beta * error_signal *  self.cost_ops
        return H_tot

    
    
class Cached_QM(Quantum_Model_DQ):
    # Model for cached state in QEP
    def __init__(self, N, T, dt, T_nudge=100):
        super().__init__(N, T, dt)
        
        t0 = time.time()
        self.T_nudge = T_nudge
        self.tlist_nudge = jnp.arange(0, T_nudge, dt)
        t1 = time.time()
        print(t1-t0)
        
    def nudge_H(self, H_tot, beta, y, target, nudge_func):
        # This generate Hamiltonian in the nudge phase. 
        # A finite coupling to the target is switched on adiabatically.
        # H_tot is the total Hamiltonian at the end during the free phase.
        error_signal = jax.grad(self.cost_func, 0)(y, target)
        H = dq.modulated(lambda t: nudge_func(t, beta, self.T_nudge), error_signal * self.cost_ops) + H_tot
        return H

#=============================Concrete Systems============================
    
class L2Lattice(Quantum_Model_DQ):
    # Implement 2d Lattices
    def __init__(self, Ns, T, dt):
        length, width = Ns
        self.length, self.width = Ns
        super().__init__(length*width, T, dt)
        
        
    def set_coupling_ops(self):
        #print("This is a 2d Lattice")
        COs = []
        Ns = self.block_ops[3] #Particle number operators
        for k in range(self.width):
            for l in range(self.length-1):
                COs.append(Ns[k*self.length+l]@Ns[k*self.length+l+1])
                
        for k in range(self.width-1):
            for l in range(self.length):
                COs.append(Ns[k*self.length+l]@Ns[k*(self.length+1)+l])
    
        return COs
    
    def get_cost_ops(self):
        # This set the Hamiltonian implementing the cost function
        # During the nudge phase, the Hamiltonian is like H = H_0 + \beta * \sum_i \epsilon_i C_i
        # C_i is the output of this function
        
        AM = sum([((-1)**k) * self.block_ops[2][k] for k in range(0, self.length)])/self.length
        return AM
    

#======================Functions to Deal with model==================

def tree_norm(tree:list):
    S = 0.
    for term in tree:
        if isinstance(term, list) or isinstance(term, tuple):
            S = S + tree_norm(term)
        else:
            S = S + jnp.square(jnp.abs(term)).sum()
    return S

def extract_tree(tree, ind):
    res = []
    for term in tree:
        if isinstance(term, jnp.ndarray) or isinstance(term, np.ndarray):
            res.append(term[ind])
        else:
            res.append(extract_tree(term, ind))
    return res

def params_dot(params1, params2):
    vp1, _ = jax.flatten_util.ravel_pytree(params1)
    vp2, _ = jax.flatten_util.ravel_pytree(params2)
    return jnp.dot(vp1, vp2)

def params_cos(params1, params2):
    return params_dot(params1, params2)/jnp.sqrt(params_dot(params2, params2) * params_dot(params1, params1))

def v_grad_cos(v_params, exact_gradient, qep_gradient, dqm, target):
    def single_func(params, dqm, target):
        exact_grad = exact_gradient(dqm, params, target)
        qep_grad = qep_gradient(params, dqm, target)[1]
        return exact_grad, qep_grad, params_cos(exact_grad, qep_grad)
    return jax.vmap(single_func, (0, None, None))(v_params, dqm, target)