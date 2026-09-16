#!/usr/bin/env python
"""One QEP training run.

Reproduces the cluster protocol of ``/u/qinwang/notebooks/QEP_sparse/fixTN_T=*_gd.py``
-- plain gradient descent on ``(Omega, Delta)``, ``lr = 0.01``, ``beta = 0.1``,
``target = 1``, fixed total *physical* time budget ``T_tot`` so that
``N_epoch = T_tot / T`` -- but with this project's model
(``afmqep.Ising_Chain`` / ``Ising_Chain_AM2``: interaction ``sum V n n``, smooth
nudge switch-on) and this project's order parameter (staggered magnetisation
``AM``, or ``AM^2``), rather than the nearest-neighbour ZZ correlator.

``--method ED`` is independent of T (the exact ground state of the final
Hamiltonian does not know how long the sweep was), so it is run once with the
largest epoch count and sliced afterwards; ``--T`` is then only bookkeeping.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afmqep as A          # noqa: E402
import dynamiqs as dq       # noqa: E402


def build_model(args):
    kw = dict(T=args.T, dt=args.dt, T_nudge=args.T if args.T_nudge is None
              else args.T_nudge)
    if args.geom == "chain":
        cls = A.Ising_Chain if args.cost == "AM" else A.Ising_Chain_AM2
        qm = cls(args.N, **kw)
        tag_geom = f"chain_N{args.N}"
    else:
        shape = tuple(int(x) for x in args.shape.lower().split("x"))
        cls = A.Ising_Lattice if args.cost == "AM" else A.Ising_Lattice_AM2
        qm = cls(shape, **kw)
        tag_geom = f"lattice_{shape[0]}x{shape[1]}"
    return qm, tag_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["chain", "lattice"], default="chain")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--shape", type=str, default="3x3")
    ap.add_argument("--cost", choices=["AM", "AM2"], default="AM")
    ap.add_argument("--method", choices=["ED", "AE", "CAE"], default="AE")
    ap.add_argument("--schedule", choices=["smooth", "stiff"], default="smooth")
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--T-nudge", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--rtol", type=float, default=None,
                    help="override the adaptive solver relative tolerance")
    ap.add_argument("--atol", type=float, default=None,
                    help="override the adaptive solver absolute tolerance")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the adaptive solver step cap")
    ap.add_argument("--Ttot", type=float, default=100000.0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override N_epoch (default: Ttot/T)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--target", type=float, default=1.0)
    # The cluster started at (Omega, Delta) = (0.5, 1.3) with an interaction
    # 0.5*sum(V n n).  This project's interaction is sum(V n n), i.e. twice as
    # strong, and the *final* Hamiltonian obeys
    #     H_here(Omega, Delta, V=1) = 2 * H_cluster(Omega/2, Delta/2, V=1),
    # so the same point of the phase diagram now sits at (1.0, 2.6).  Starting
    # literally at (0.5, 1.3) would begin deep inside the ordered lobe
    # (<AM> ~ 0.8 already) and leave the optimiser nothing to do.
    ap.add_argument("--Omega0", type=float, default=1.0)
    ap.add_argument("--Delta0", type=float, default=2.6)
    ap.add_argument("--init", type=str, default="base", help="label for the start point")
    ap.add_argument("--track-exact", type=int, default=1,
                    help="also record <cost_op> in the exact ground state each epoch")
    ap.add_argument("--outdir", default="../data/train")
    ap.add_argument("--verbose-every", type=int, default=0)
    args = ap.parse_args()

    N_epoch = args.epochs if args.epochs is not None else int(round(args.Ttot / args.T))
    qm, tag_geom = build_model(args)
    params = qm.set_params(args.Omega0, args.Delta0)
    method = None
    if args.rtol is not None or args.atol is not None or args.max_steps is not None:
        method = dq.method.Tsit5(
            rtol=1e-6 if args.rtol is None else args.rtol,
            atol=1e-6 if args.atol is None else args.atol,
            max_steps=50_000_000 if args.max_steps is None else args.max_steps,
        )
    gm = A.make_grad_methods(args.beta, args.schedule, method=method)[args.method]
    opt = A.QEP_Training(args.lr)

    T_label = "ED" if args.method == "ED" else f"{args.T:g}"
    tag = (f"{tag_geom}_{args.cost}_{args.method}_{args.schedule}"
           f"_T{T_label}_{args.init}")
    print(f"[{tag}] N_epoch={N_epoch} beta={args.beta} lr={args.lr} "
          f"start=({args.Omega0},{args.Delta0})", flush=True)

    OmegaL, DeltaL, yL, yexL = [], [], [], []
    run = params
    t0 = time.time()
    for k in range(N_epoch):
        y, g = gm(run, qm, args.target)
        OmegaL.append(float(run[0]))
        DeltaL.append(float(run[1]))
        yL.append(float(np.real(np.asarray(y))))
        if args.track_exact:
            yexL.append(A.exact_y(qm, run))
        run = opt.update(run, g)
        if args.verbose_every and (k % args.verbose_every == 0 or k == N_epoch - 1):
            print(f"  {k+1}/{N_epoch} y={yL[-1]:+.5f} "
                  f"O={OmegaL[-1]:+.4f} D={DeltaL[-1]:+.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    OmegaL.append(float(run[0]))
    DeltaL.append(float(run[1]))

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, tag + ".npz")
    np.savez_compressed(
        out,
        y=np.asarray(yL), y_exact=np.asarray(yexL) if args.track_exact else np.zeros(0),
        Omega=np.asarray(OmegaL), Delta=np.asarray(DeltaL),
        meta=json.dumps(vars(args) | {"N_epoch": N_epoch, "tag": tag,
                                      "wall_s": time.time() - t0}),
    )
    print(f"[{tag}] done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
