"""Validated exact ground-state solver for post-hoc analysis (read-only).

Background
----------
Training and the plotted Figure-13 gradient moments all go through
``QEP.grad.Exact_Diagonalization``, which despite its name delegates to
``QEP.model.get_gs_sparse``: a fixed 100-step momentum-gradient power
iteration with no convergence check (see ``QEP/model.py``). Independent
checks in ``figure13_noise_audit_2026-09-14.md`` found several-percent
gradient-mean discrepancies between that iterative state and a genuine
dense diagonalization at some saved training points, and showed that 100
iterations is measurably insufficient there (see that file and the
"Ground-state solver accuracy" discussion in ``reports/2026-09-11/main.tex``).

Several standalone audit/analysis scripts in this study already worked
around this by independently calling ``scipy.linalg.eigh(H,
subset_by_index=[0, 0])`` for a genuine float64/complex128 lowest
eigenpair: ``audit_component_noise.py``, ``trajectory_expectations.py``,
``summarize_overlap.py``, and ``figure11b_resampling.py``. This module
extracts that ad hoc pattern into one documented, reusable place.

Scope and safety
-----------------
This module is a **read-only analysis utility**. It is not imported by
``afm_shots`` or by any training/sampling code, and it must never be used
to alter existing trajectory/production ``.npz`` data. Its purpose is
validated *exact* endpoint/reference evaluation for reports and future
corrected-solver work, not a silent correction of the approximate-solver
training trajectories already on disk.

Run this file directly to execute its self-validation:
    JAX_PLATFORM_NAME=cpu python -m AFM_searching.shot_study.exact_solver
"""
from __future__ import annotations

import numpy as np
import dynamiqs as dq
from scipy.linalg import eigh

from ..afm_shots.common import build_model_and_grad


def ground_state(H, *, residual_atol=1e-9, check_finite=True):
    """Lowest eigenpair of a Hermitian matrix via a genuine dense eigensolver.

    This wraps ``scipy.linalg.eigh(..., subset_by_index=[0, 0])`` at
    float64/complex128 precision -- not an iterative approximation, and
    independent of the JAX global ``x64`` flag used elsewhere in the
    project.

    Parameters
    ----------
    H : array_like, shape (d, d)
        Hermitian (or real-symmetric) matrix, any numeric input dtype; it
        is promoted to float64 (real input) or complex128 (complex input)
        before diagonalization.
    residual_atol : float or None, default 1e-9
        If not ``None``, raise ``RuntimeError`` when the eigenvector
        residual ``||H @ psi - energy * psi||`` exceeds this tolerance.
        Set to ``None`` to skip the check.
    check_finite : bool, default True
        Forwarded to ``scipy.linalg.eigh``.

    Returns
    -------
    energy : float
        Lowest eigenvalue.
    psi : numpy.ndarray, shape (d,)
        Normalized ground-state eigenvector (float64 or complex128).
    """
    H = np.asarray(H)
    dtype = np.complex128 if np.iscomplexobj(H) else np.float64
    H = H.astype(dtype, copy=False)
    energies, vectors = eigh(H, subset_by_index=[0, 0], check_finite=check_finite)
    energy = float(energies[0])
    psi = vectors[:, 0]
    if residual_atol is not None:
        residual = float(np.linalg.norm(H @ psi - energy * psi))
        if residual > residual_atol:
            raise RuntimeError(
                f"ground_state: eigenvector residual {residual:.3e} exceeds "
                f"tolerance {residual_atol:.3e}"
            )
    return energy, psi


def exact_ground_state(N, Omega, Delta, V=None, residual_atol=1e-9):
    """Exact free ground state of the N-site open ``Ising_Chain_AM2`` chain.

    Builds the fully-ramped free Hamiltonian at ``(Omega, Delta)`` (V
    defaults to the model's uniform coupling, V=1) exactly as the
    production model does, then diagonalizes it with :func:`ground_state`
    instead of the approximate 100-iteration training-state solver.

    Returns
    -------
    energy : float
    psi : numpy.ndarray
        Ground-state vector.
    am2 : float
        Exact ground-state ``<AFM^2> = <psi| cost_ops |psi>`` (real part).
    """
    nn, params, _ = build_model_and_grad(N, Omega, Delta)
    if V is not None:
        params = (params[0], params[1], np.asarray(V))
    H = np.asarray(dq.to_jax(nn.set_H_f(params)), dtype=np.complex128)
    energy, psi = ground_state(H, residual_atol=residual_atol)
    cost = np.asarray(dq.to_jax(nn.cost_ops), dtype=np.complex128)
    am2 = float(np.vdot(psi, cost @ psi).real)
    return energy, psi, am2


def exact_am2(N, Omega, Delta, V=None):
    """Convenience wrapper around :func:`exact_ground_state`: <AFM^2> only."""
    return exact_ground_state(N, Omega, Delta, V=V)[2]


def _self_validate():
    """Compare against the exact references quoted in GPT_relay.md under
    "Main production results", (Omega, Delta) = (0.5, 1.5), beta=0.1:
    N=5 exact ground-state <AFM^2>=0.6705; N=9 exact ground-state
    <AFM^2>=0.5072.

    Those two numbers were themselves computed before the solver-precision
    issue documented in figure13_noise_audit_2026-09-14.md was found, most
    likely via ``afm_shots.common.exact_am2`` under JAX's default complex64
    global precision. A later independent float64 recomputation at the same
    N=5 point ("Why the two time-budget ensembles differ", 2026-09-15) found
    <AFM^2>=0.671697, agreeing with the quoted 0.6705 only to ~1.2e-3, not
    machine precision. Running this module the same way found N=5 agrees to
    1.2e-3 and N=9 to 6.0e-3 -- larger for N=9's 512-dimensional Hilbert
    space, consistent with accumulated complex64 error rather than a bug
    here. The tolerance below (1e-2) matches that already-documented
    precision gap; this solver is float64 scipy.linalg.eigh throughout, i.e.
    more accurate than either prior number. Raises AssertionError on
    mismatch (which would indicate a real regression, not this known gap).
    """
    checks = {5: 0.6705, 9: 0.5072}
    results = {}
    for N, reference in checks.items():
        am2 = exact_am2(N, 0.5, 1.5)
        results[N] = am2
        assert abs(am2 - reference) < 1e-2, (
            f"N={N}: exact_am2(0.5,1.5)={am2:.6f} disagrees with the "
            f"GPT_relay.md reference {reference} by more than 1e-2"
        )
    return results


if __name__ == "__main__":
    values = _self_validate()
    for N, am2 in values.items():
        print(f"N={N}, (Omega,Delta)=(0.5,1.5): exact <AFM^2>={am2:.6f} (validated)")
