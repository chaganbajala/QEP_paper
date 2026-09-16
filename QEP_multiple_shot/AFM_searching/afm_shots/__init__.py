"""afm_shots: standalone package for the fixed-N_g / finite-shot AFM-searching study
(reseach_plan_2026-09-11.md).

Carries its own copies of the model/gradient/training classes it needs (`afm.py`,
`grad.py`, `train.py`, `schedules.py`), ported from `Adaptive_bn/qep_shot` -- see each
module's docstring for provenance. No import-time dependency on Adaptive_bn/qep_shot
at all, only on the shared `QEP` core library (`QEP_sparse/QEP`), resolved via
`_pathfix.py`. This satisfies research-plan item "Write python scripts that can run
without Adaptive_bn/qep_shot".

The nudge-phase ramp is smooth by construction (inherited unchanged from qep_shot):
`QM.tune_func` multiplies the nudge term by `ascend_func(t/T)`, a ramp with zero
derivative at both endpoints -- so the nudge switches on smoothly over the adiabatic
protocol rather than as a step function.
"""
from .schedules import ascend_func, make_modulation_funcs
from .afm import QM, Ising_Chain, Ising_Chain_AM2
from .grad import Shot_QEP_Grad
from .train import Shot_QEP_Training, Shot_QEP_Training_WithStats

__all__ = [
    "ascend_func", "make_modulation_funcs",
    "QM", "Ising_Chain", "Ising_Chain_AM2",
    "Shot_QEP_Grad",
    "Shot_QEP_Training", "Shot_QEP_Training_WithStats",
]
