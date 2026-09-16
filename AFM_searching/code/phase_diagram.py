#!/usr/bin/env python
"""Ground-state phase diagram in the (Delta, Omega) plane.

Scans the *final* Hamiltonian

    H = Omega * sum_k X_k/2  -  Delta * sum_k n_k  +  V * sum_<ij> n_i n_j

exactly (no time evolution) and records, at every grid point, the staggered
magnetisation ``<AM>``, its second moment ``<AM^2>``, and the gap ``E1 - E0``.

The diagram has to be recomputed for this project: the cluster used
``QEP.model.Quantum_Model_DQ``, whose interaction is ``0.5 * sum V n n``,
whereas ``qep_shot.afm.Ising_Chain`` -- the model this study trains -- uses
``sum V n n``.  At V = 1 that is a factor 2 in the coupling and it moves the
lobe boundaries.

``AM`` is diagonal in the Z basis, so both moments are read straight off the
ground-state amplitudes; only ``H`` needs diagonalising.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afmqep.gs import sparse_site_ops           # noqa: E402
from afmqep.models import chain_pairs, lattice_pairs  # noqa: E402


def stagger_chain(N):
    return np.array([(-1) ** k for k in range(N)], dtype=float)


def stagger_lattice(n_row, n_col):
    return np.array([(-1) ** (k // n_col + k % n_col) for k in range(n_row * n_col)],
                    dtype=float)


def build_pieces(N, pairs, stagger):
    """Return (X_part sparse, diag_n, diag_int, am_diag) with AM already normalised."""
    Xs, Zs, Ns = sparse_site_ops(N)
    X_part = 0.5 * sum(Xs)

    # diagonal vectors, indexed by computational basis state
    idx = np.arange(2 ** N)
    bits = np.array([(idx >> (N - 1 - k)) & 1 for k in range(N)])  # 1 -> |0>_k? see below
    # embed() puts site k at tensor position k, so the most significant bit of the
    # basis index is site 0.  n_k has eigenvalue 1 on |0> (since n = (1+Z)/2 and
    # Z|0> = +|0>), i.e. on bit == 0.
    occ = 1 - bits                       # occupation of site k, shape (N, 2**N)
    zval = 2.0 * occ - 1.0               # Z eigenvalue

    diag_n = occ.sum(axis=0).astype(float)
    diag_int = sum(occ[i] * occ[j] for (i, j) in pairs).astype(float) if pairs \
        else np.zeros(2 ** N)
    am_diag = (stagger[:, None] * zval).sum(axis=0) / N
    return X_part.tocsr(), diag_n, diag_int, am_diag


def scan(N, pairs, stagger, Omega_list, Delta_list, V=1.0, want_gap=True,
         verbose=True):
    X_part, diag_n, diag_int, am_diag = build_pieces(N, pairs, stagger)
    dim = 2 ** N
    dense = dim <= 1024

    AM = np.zeros((len(Omega_list), len(Delta_list)))
    AM2 = np.zeros_like(AM)
    GAP = np.zeros_like(AM)

    Hbase = V * sp.diags(diag_int)
    t0 = time.time()
    for i, Omega in enumerate(Omega_list):
        HO = Omega * X_part
        # At Omega = 0 the Hamiltonian is diagonal and its classical spectrum is
        # massively degenerate; ARPACK (eigsh) can then return a spurious
        # "ground state" -- it did so in the empty-lattice region of the 4x4
        # scan.  Handle that column exactly instead: the ground state is simply
        # the lowest diagonal entry.
        diagonal_H = abs(float(Omega)) < 1e-12
        for j, Delta in enumerate(Delta_list):
            if diagonal_H and not dense:
                dvec = V * diag_int - Delta * diag_n
                order = np.argpartition(dvec, 1)[:2]
                order = order[np.argsort(dvec[order])]
                g = int(order[0])
                AM[i, j] = float(am_diag[g])
                AM2[i, j] = float(am_diag[g] ** 2)
                GAP[i, j] = float(dvec[order[1]] - dvec[order[0]]) if want_gap \
                    else np.nan
                continue
            H = HO + Hbase + sp.diags(-Delta * diag_n)
            if dense:
                w, v = scipy.linalg.eigh(H.toarray(),
                                         subset_by_index=[0, 1] if want_gap else [0, 0])
            else:
                w, v = sp.linalg.eigsh(H.tocsc(), k=2 if want_gap else 1,
                                       which="SA", tol=1e-10)
                order = np.argsort(w)
                w, v = w[order], v[:, order]
            p = np.abs(v[:, 0]) ** 2
            AM[i, j] = float(p @ am_diag)
            AM2[i, j] = float(p @ (am_diag ** 2))
            GAP[i, j] = float(w[1] - w[0]) if want_gap else np.nan
        if verbose and (i % 20 == 0 or i == len(Omega_list) - 1):
            print(f"  Omega row {i+1}/{len(Omega_list)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    return AM, AM2, GAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["chain", "lattice"], default="chain")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--shape", type=str, default="3x3")
    ap.add_argument("--Omega-range", type=str, default="-3,3",
                    help="Omega_min,Omega_max")
    ap.add_argument("--Delta-range", type=str, default="-1,3",
                    help="Delta_min,Delta_max (widen for 2D: higher coordination "
                         "pushes the lobe edge out to Delta = z*V)")
    ap.add_argument("--dOmega", type=float, default=0.02)
    ap.add_argument("--dDelta", type=float, default=0.02)
    ap.add_argument("--V", type=float, default=1.0)
    ap.add_argument("--no-gap", action="store_true")
    ap.add_argument("--outdir", default="../data/phase")
    args = ap.parse_args()

    if args.geom == "chain":
        N, pairs, stag = args.N, chain_pairs(args.N), stagger_chain(args.N)
        tag = f"chain_N{args.N}"
    else:
        n_row, n_col = (int(x) for x in args.shape.lower().split("x"))
        N = n_row * n_col
        pairs, stag = lattice_pairs(n_row, n_col), stagger_lattice(n_row, n_col)
        tag = f"lattice_{n_row}x{n_col}"

    o_lo, o_hi = (float(x) for x in args.Omega_range.split(","))
    d_lo, d_hi = (float(x) for x in args.Delta_range.split(","))
    Omega_list = np.arange(o_lo, o_hi, args.dOmega)
    Delta_list = np.arange(d_lo, d_hi, args.dDelta)
    print(f"{tag}: dim={2**N}, grid {len(Omega_list)} x {len(Delta_list)}", flush=True)

    AM, AM2, GAP = scan(N, pairs, stag, Omega_list, Delta_list, V=args.V,
                        want_gap=not args.no_gap)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"phase_{tag}.npz")
    np.savez_compressed(out, AM=AM, AM2=AM2, GAP=GAP, Omega_list=Omega_list,
                        Delta_list=Delta_list, N=N, V=args.V, tag=tag,
                        pairs=np.asarray(pairs))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
