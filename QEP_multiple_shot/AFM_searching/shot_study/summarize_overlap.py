"""Validate complete expanded ensembles and cache Figures 9, 10, and 11b arrays.

Run after downloading cluster checkpoints. Incomplete ensembles never replace
the original plots. No training or new sampling is run. Budget endpoints are
evaluated with float64 exact diagonalization outside the plotting notebook.
"""
import argparse
import hashlib
import json

import numpy as np
import dynamiqs as dq
from threadpoolctl import threadpool_limits

from ..afm_shots import common
from . import config as C
from . import exact_solver
from .overlap_training import COMMON_BUDGET, EPOCHS, OUTPUT_DIR, POINTS, SEEDS
from .run import atomic_save


def summarize(N):
    arrays, sources = {}, {}
    budget_points = []
    missing = []
    for ng in C.TRAJECTORY_NG:
        curves = []
        for init_id in range(len(POINTS)):
            seeds = []
            for seed_id in range(SEEDS):
                path = OUTPUT_DIR / f'N{N}_init{init_id}_Ng{ng}_seed{seed_id}.npz'
                if not path.exists():
                    missing.append(f'{path.name}: missing')
                    continue
                with np.load(path, allow_pickle=False) as d:
                    meta = json.loads(str(d['metadata']))
                    if not meta['complete']:
                        missing.append(f'{path.name}: {meta["completed_epochs"]}/{EPOCHS[ng]} epochs')
                        continue
                    for key, value in dict(N=N, N_g=ng, N_y=C.N_Y, init_id=init_id,
                                           seed_id=seed_id, epochs=EPOCHS[ng],
                                           completed_epochs=EPOCHS[ng], smoke=False).items():
                        assert meta[key] == value, (path, key)
                    np.testing.assert_array_equal(meta['point'], POINTS[init_id])
                    y = d['measured_AM2']
                    assert y.shape == (EPOCHS[ng],)
                    assert d['Omega'].shape == d['Delta'].shape == (EPOCHS[ng]+1,)
                    assert d['sampled_gradient'].shape == (EPOCHS[ng],2)
                    for key in ('Omega','Delta','sampled_gradient','measured_AM2'):
                        assert np.isfinite(d[key]).all(), (path,key)
                    assert y.min() >= -1e-6 and y.max() <= 1+1e-6
                    if init_id < len(C.INITIAL_POINTS):
                        with np.load(C.DATA_DIR/'trajectories'/path.name, allow_pickle=False) as original:
                            for key in ('Omega','Delta','sampled_gradient','measured_AM2'):
                                np.testing.assert_array_equal(d[key][:len(original[key])], original[key])
                    seeds.append(y)
                    budget_points.append((ng, init_id, seed_id,
                                          d['Omega'].copy(), d['Delta'].copy()))
                    sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if len(seeds) == SEEDS:
                curves.append(seeds)
        if len(curves) == len(POINTS):
            # Treat initial points as the independent averaging groups; sampling
            # seeds replicate each point, rather than creating 50 distinct starts.
            curves = np.asarray(curves)
            start_means = curves.mean(axis=1)
            arrays[f'mean_Ng{ng}'] = start_means.mean(axis=0)
            arrays[f'sem_Ng{ng}'] = start_means.std(axis=0, ddof=1) / np.sqrt(len(POINTS))
            arrays[f'epoch_Ng{ng}'] = np.arange(1,EPOCHS[ng]+1)
    if missing:
        print(f'N={N}: {len(missing)}/200 runs missing/incomplete; summary not written.')
        for item in missing[:8]:
            print(' ',item)
        return False
    # Preserve the old budget checkpoints, but express caps in physical shots:
    # two independently measured bases in each of free/nudged states => 4*Ng.
    budgets = 2 * np.asarray(C.SHOT_BUDGETS)
    ngs = np.asarray(C.TRAJECTORY_NG)
    checkpoint_epochs = budgets[None, :] // (4 * ngs[:, None])
    endpoints = np.full((len(ngs), len(POINTS), SEEDS, len(budgets)), np.nan)
    nn, _, _ = common.build_model_and_grad(N, *POINTS[0])
    def matrix(op):
        value = np.asarray(dq.to_jax(op))
        np.testing.assert_allclose(value.imag, 0, atol=1e-12)
        return value.real.astype(np.float64)
    hx, hz, hi = map(matrix, (nn.H_Omega, nn.H_Delta, nn.H_int))
    cost = matrix(nn.cost_ops)
    max_residual = 0.
    with threadpool_limits(limits=1):
        for ng, init_id, seed_id, omega, delta in budget_points:
            g = list(ngs).index(ng)
            for b, k in enumerate(checkpoint_epochs[g]):
                if not 1 <= k < len(omega):
                    continue
                h = omega[k] * hx + delta[k] * hz + hi
                # Same real-float64 matrix through the identical
                # eigh(..., subset_by_index=[0, 0]) call and 1e-9 residual
                # gate as before; exact_solver.ground_state (validated in
                # exact_solver.py's self-check) now owns that logic so it is
                # not duplicated here. This is a pure refactor: no numerical
                # change for already-summarized N=5 data (never rerun here).
                energy, psi = exact_solver.ground_state(h, residual_atol=1e-9)
                residual = np.linalg.norm(h @ psi - energy * psi)
                max_residual = max(max_residual, float(residual))
                endpoints[g, init_id, seed_id, b] = psi @ cost @ psi
    valid = ((checkpoint_epochs >= 1) &
             (checkpoint_epochs <= np.array([EPOCHS[ng] for ng in ngs])[:, None]))
    assert np.array_equal(np.isfinite(endpoints).all(axis=(1, 2)), valid)
    finite = endpoints[np.isfinite(endpoints)]
    assert finite.min() >= -1e-7 and finite.max() <= 1 + 1e-7
    start_means = endpoints.mean(axis=2)
    arrays.update(N_g=ngs, physical_budgets=budgets,
                  budget_checkpoint_epochs=checkpoint_epochs,
                  budget_exact_AM2=endpoints,
                  budget_mean=start_means.mean(axis=1),
                  budget_sem=start_means.std(axis=1, ddof=1)/np.sqrt(len(POINTS)))
    path = C.DATA_DIR / f'trajectory_overlap_summary_N{N}.npz'
    atomic_save(path, metadata=json.dumps(dict(
        N=N, N_y=C.N_Y, initial_points=len(POINTS), seeds_per_point=SEEDS,
        N_g=list(C.TRAJECTORY_NG), epochs=EPOCHS, common_gradient_budget=COMMON_BUDGET,
        complete=True, source_sha256=sources,
        sem='SEM across 10 initial-point means, each averaging 5 seeds',
        x_convention='4*N_g*(k+1); y[k] is pre-update; allocation includes current epoch',
        physical_gradient_shots_per_epoch='4*N_g; excludes N_y output shots',
        common_physical_gradient_budget=2*COMMON_BUDGET,
        budget_alignment='Post-update parameters Omega[k], Delta[k], k=floor(S/(4*N_g)); k=0 omitted',
        budget_solver='scipy.linalg.eigh float64 lowest eigenpair',
        max_budget_eigenpair_residual=max_residual,
        training_solver='Historical 100-step approximate state preparation; not corrected',
    )), **arrays)
    print(f'N={N}: validated 200 complete runs; saved {path}; max residual={max_residual:.3g}')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', type=int, nargs='+', choices=C.N_LIST, default=[5])
    args = parser.parse_args()
    for N in args.N:
        summarize(N)
