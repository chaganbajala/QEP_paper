"""Shared setup/helpers for the fixed-N_g shot-number study's task scripts
(../task1_gradient_distribution.py .. ../task5_single_run_stats.py).

Physical setup fixed across the whole study (reseach_plan_2026-09-11.md):
  - open Ising chain, cost operator AM^2 (Ising_Chain_AM2), N in {5, 9}
  - beta = 0.1, learning_rate (eta) = 0.01, target = 1.0
  - T = 100, dt = 0.1 (the project-wide default adiabatic ramp, see
    Adaptive_bn/qep_shot's test.ipynb / setup_model.py)
  - solver = QEP.grad.Exact_Diagonalization ("ED"): takes the ground state of the
    fully-ramped H(T) directly, i.e. assumes perfectly adiabatic following. This is
    the standard cheap default used throughout Adaptive_bn for shot-count studies
    (state prep cost is then independent of N_g -- only the categorical shot
    sampling depends on N_g, which is cheap), letting N_g range up to 10^4 without
    the cost exploding. A real-time-dynamics (Adiabatic_Evolution, "AE") cross-check
    at one representative setting is done separately (see README.md) to confirm the
    idealization doesn't change the qualitative conclusions.
"""
import os

import numpy as np
import jax.numpy as jnp
import dynamiqs as dq

from . import afm
from .schedules import make_modulation_funcs
from .grad import Shot_QEP_Grad

try:
    from . import _pathfix  # noqa: F401
except ImportError:
    import _pathfix  # noqa: F401
from QEP import grad as qep_grad

# ---- paths ----
AFM_SEARCHING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(AFM_SEARCHING_DIR, "data", "shots_2026-09-11")
FIG_DIR = os.path.join(AFM_SEARCHING_DIR, "figures", "shots_2026-09-11")
LOG_DIR = os.path.join(AFM_SEARCHING_DIR, "logs")

# ---- fixed physical/training setup ----
T = 100.
DT = 0.1
BETA = 0.1
LEARNING_RATE = 0.01
TARGET = 1.0
N_LIST = (5, 9)


def build_model_and_grad(N_sites, Omega0, Delta0, beta=BETA, solver_cls=qep_grad.Exact_Diagonalization):
    """Returns (nn, params_0, grad_method) for the open Ising_Chain_AM2 chain."""
    nn = afm.Ising_Chain_AM2(N_sites, T, DT)
    params_0 = nn.set_params(Omega0, Delta0)
    param_funcs = make_modulation_funcs()[:2]
    solver = solver_cls(param_funcs)
    grad_method = Shot_QEP_Grad(beta, solver)
    return nn, params_0, grad_method


def exact_am2(nn, params):
    """Exact (shot-free) ground-state <AM^2> at fixed params, via dense diagonalization
    of the fully-ramped H(T). Used as the phase-diagram background and as a noiseless
    reference against noisy training trajectories."""
    H_full = dq.to_jax(dq.asqarray(nn.set_H_f(params)))
    evals, evecs = jnp.linalg.eigh(H_full)
    psi0 = dq.asqarray(evecs[:, 0:1], dims=nn.dims)
    return float(dq.expect(nn.cost_ops, psi0).real)


def phase_diagram_am2(N_sites, omega_range, delta_range):
    """Exact <AM^2> over a (Omega, Delta) grid at fixed V=1, for the phase-diagram
    background used in task 2's trajectory plots. Returns AM2[i, j] for
    delta_range[i], omega_range[j] (matplotlib pcolormesh convention)."""
    nn = afm.Ising_Chain_AM2(N_sites, T, DT)
    V = np.ones(N_sites - 1)
    AM2 = np.zeros((len(delta_range), len(omega_range)))
    for i, Delta in enumerate(delta_range):
        for j, Omega in enumerate(omega_range):
            params = (float(Omega), float(Delta), V)
            AM2[i, j] = exact_am2(nn, params)
    return AM2


def random_init_points(n, seed, omega_range=(0.1, 1.5), delta_range=(0.5, 2.5)):
    """n random (Omega0, Delta0) initial points, uniform in the given box -- used by
    task 2 to pick "3 randomly chosen initial parameters"."""
    rng = np.random.default_rng(seed)
    Omegas = rng.uniform(*omega_range, size=n)
    Deltas = rng.uniform(*delta_range, size=n)
    return list(zip(Omegas.tolist(), Deltas.tolist()))


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
