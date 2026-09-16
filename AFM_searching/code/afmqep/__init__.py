"""afmqep -- self-contained code for the local AFM-searching study.

Nothing here imports the shared ``QEP`` library or any other project directory:
the model, gradient, training and schedule classes are all carried locally, as
required by ``AFM_searching/plan.md``.

Precision is chosen *before* JAX is configured, from the environment variable
``AFMQEP_X64``:

* ``AFMQEP_X64=0`` (default) -- complex64, matching the cluster runs; used for
  the long training sweeps.
* ``AFMQEP_X64=1`` -- complex128; used for the gradient-bias measurement, where
  the quantity of interest is a small difference of two gradients.
"""
import os

import jax

_X64 = os.environ.get("AFMQEP_X64", "0") not in ("0", "", "false", "False")
jax.config.update("jax_enable_x64", _X64)

from .schedules import (ascend_func, linear_func, make_modulation_funcs,  # noqa: E402
                        SMOOTH, STIFF, get_schedule)
from .gs import (exact_ground_state, sparse_hamiltonian, sparse_ground_state,  # noqa: E402
                 sparse_site_ops, sparse_low_spectrum)
from .models import (QM, Ising_Chain, Ising_Chain_AM2, Ising_Lattice,  # noqa: E402
                     Ising_Lattice_AM2, chain_pairs, lattice_pairs)
from .grad import (Exact_Diagonalization, Adiabatic_Evolution,  # noqa: E402
                   Adiabatic_Evolution_Nudge, QEP_grad, Cached_QEP_grad,
                   exact_loss_gradient, exact_y)
from .train import QEP_Training  # noqa: E402

X64 = _X64

MODELS = {
    "AM": Ising_Chain,
    "AM2": Ising_Chain_AM2,
    "lattice_AM": Ising_Lattice,
    "lattice_AM2": Ising_Lattice_AM2,
}


def make_grad_methods(beta, schedule="smooth", method=None):
    """Return ``{"ED": ..., "AE": ..., "CAE": ...}`` for a given nudge strength.

    ``method`` is a dynamiqs integrator (default adaptive ``Tsit5``); pass a
    tightened one for the gradient-bias measurement, where the signal is a small
    difference of two gradients and the integrator error must sit well below it.
    """
    fO, fD, fnudge = get_schedule(schedule)
    ed_solver = Exact_Diagonalization((fO, fD))
    ae_solver = Adiabatic_Evolution((fO, fD), method=method)
    aes_solver = Adiabatic_Evolution_Nudge((fO, fD), fnudge, method=method)
    return {
        "ED": QEP_grad(beta, ed_solver),
        "AE": QEP_grad(beta, ae_solver),
        "CAE": Cached_QEP_grad(beta, (ae_solver, aes_solver)),
    }


__all__ = [
    "X64", "MODELS", "make_grad_methods",
    "ascend_func", "linear_func", "make_modulation_funcs", "SMOOTH", "STIFF",
    "get_schedule",
    "exact_ground_state", "sparse_hamiltonian", "sparse_ground_state",
    "sparse_site_ops", "sparse_low_spectrum",
    "QM", "Ising_Chain", "Ising_Chain_AM2", "Ising_Lattice", "Ising_Lattice_AM2",
    "chain_pairs", "lattice_pairs",
    "Exact_Diagonalization", "Adiabatic_Evolution", "Adiabatic_Evolution_Nudge",
    "QEP_grad", "Cached_QEP_grad", "exact_loss_gradient", "exact_y",
    "QEP_Training",
]
