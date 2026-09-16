"""Exact ground states.

Two independent routes, used for different jobs:

* :func:`exact_ground_state` -- dense LAPACK diagonalisation of a dynamiqs
  ``QArray``.  Used inside the QEP gradient (the "exact QEP" method, and the
  initial state of every adiabatic sweep).  The cluster project used an
  iterative Rayleigh-quotient descent (``QEP.model.get_gs_sparse``, 100 fixed
  steps) here; we diagonalise instead so that "exact ground state" really is
  exact and cannot contaminate the O(1)-bias measurement.

* :func:`sparse_hamiltonian` / :func:`sparse_ground_state` -- a scipy.sparse
  construction of the *final* Hamiltonian only.  It never forms a dense matrix,
  so it reaches the 4x4 lattice (N = 16, dimension 65536) that dense
  diagonalisation cannot.  Used for phase diagrams and ground-state surveys.
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg
import jax.numpy as jnp
import dynamiqs as dq

__all__ = [
    "exact_ground_state", "sparse_hamiltonian", "sparse_ground_state",
    "sparse_site_ops", "sparse_low_spectrum",
]


# --------------------------------------------------------------------------- #
#  dense route (used inside the QEP gradient)
# --------------------------------------------------------------------------- #
def exact_ground_state(H, dims):
    """Lowest eigenvector of ``H`` as a normalised ket QArray.

    ``H`` may be a dynamiqs ``QArray`` or a plain dense (numpy/jax) matrix.
    """
    A = np.asarray(H.to_jax() if hasattr(H, "to_jax") else H)
    A = 0.5 * (A + A.conj().T)
    if np.max(np.abs(A.imag)) < 1e-12:
        w, v = scipy.linalg.eigh(A.real, subset_by_index=[0, 0])
        vec = v[:, 0].astype(np.complex64 if A.dtype == np.complex64 else np.complex128)
    else:
        w, v = scipy.linalg.eigh(A, subset_by_index=[0, 0])
        vec = v[:, 0]
    vec = vec / np.linalg.norm(vec)
    return dq.asqarray(jnp.asarray(vec[:, None], dtype=A.dtype), dims=dims)


# --------------------------------------------------------------------------- #
#  sparse route (phase diagrams, ground-state surveys, 2D lattices)
# --------------------------------------------------------------------------- #
def sparse_site_ops(N):
    """Return ``(Xs, Zs, Ns)``: lists of single-site operators on N qubits, CSR."""
    I2 = sp.identity(2, format="csr", dtype=np.float64)
    sx = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sz = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    n = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))  # (1 + sigma_z)/2

    def embed(op, k):
        out = sp.identity(2 ** k, format="csr", dtype=np.float64) if k else None
        out = op if out is None else sp.kron(out, op, format="csr")
        right = N - k - 1
        if right:
            out = sp.kron(out, sp.identity(2 ** right, format="csr", dtype=np.float64),
                          format="csr")
        return out.tocsr()

    Xs = [embed(sx, k) for k in range(N)]
    Zs = [embed(sz, k) for k in range(N)]
    Ns = [embed(n, k) for k in range(N)]
    return Xs, Zs, Ns


def sparse_hamiltonian(N, pairs, Omega, Delta, V=1.0, ops=None):
    """Final-time Hamiltonian ``Omega*sum(X/2) - Delta*sum(n) + V*sum_{<ij>} n_i n_j``.

    ``pairs`` is the coupling graph, a list of ``(i, j)`` index pairs.  The
    interaction carries **no factor 1/2** -- this is the ``qep_shot.afm``
    convention, which differs from ``QEP.model.Quantum_Model_DQ`` by a factor 2.
    """
    Xs, Zs, Ns = sparse_site_ops(N) if ops is None else ops
    H = Omega * 0.5 * sum(Xs) - Delta * sum(Ns)
    H = H + V * sum(Ns[i] @ Ns[j] for (i, j) in pairs)
    return H.tocsr()


def sparse_ground_state(H, k=1, v0=None):
    """Lowest ``k`` eigenpairs of a sparse symmetric ``H``; falls back to dense."""
    dim = H.shape[0]
    if dim <= 512:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = sp.linalg.eigsh(H, k=k, which="SA", v0=v0, tol=0, maxiter=50000)
    order = np.argsort(w)
    return w[order], v[:, order]


def sparse_low_spectrum(H, k=6):
    """Lowest ``k`` eigenvalues and eigenvectors -- for degeneracy surveys."""
    return sparse_ground_state(H, k=k)
