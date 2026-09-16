"""Postprocess saved training paths; no training or new measurements.

Run from QEP_multiple_shot:
    JAX_PLATFORM_NAME=cpu python -m AFM_searching.shot_study.trajectory_expectations

Saved measured_AM2[k] precedes update k, so evaluate the ground state at
Omega[k], Delta[k], not at the post-update parameters k+1.
"""
import argparse
import hashlib
import json
import time

import dynamiqs as dq
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from ..afm_shots import common
from . import config as C


def process(N):
    started = time.monotonic()
    nn, params, _ = common.build_model_and_grad(N, *C.INITIAL_POINTS[0])

    def real_matrix(operator):
        matrix = np.asarray(dq.to_jax(operator))
        np.testing.assert_allclose(matrix.imag, 0, atol=1e-12)
        return matrix.real.astype(np.float64)

    hx, hz, hi = map(real_matrix, (nn.H_Omega, nn.H_Delta, nn.H_int))
    cost = real_matrix(nn.cost_ops)
    cost_diag = np.diag(cost)
    np.testing.assert_allclose(cost, np.diag(cost_diag), atol=1e-12)
    np.testing.assert_allclose(
        params[0] * hx + params[1] * hz + hi,
        real_matrix(nn.set_H_f(params)), atol=1e-6,
    )
    shape = (len(C.TRAJECTORY_NG), len(C.INITIAL_POINTS) * C.TRAJECTORY_SEEDS,
             C.TRAJECTORY_EPOCHS)
    measured, exact = np.empty(shape), np.empty(shape)
    measurement_variance = np.empty(shape)
    paths, hashes = [], []
    runs = [(i, s) for i in range(len(C.INITIAL_POINTS))
            for s in range(C.TRAJECTORY_SEEDS)]
    max_residual = 0.0
    with threadpool_limits(limits=1):
        for g, Ng in enumerate(C.TRAJECTORY_NG):
            for r, (init_id, seed_id) in enumerate(runs):
                path = C.DATA_DIR / 'trajectories' / f'N{N}_init{init_id}_Ng{Ng}_seed{seed_id}.npz'
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata']))
                    for key, value in dict(N=N, N_g=Ng, N_y=C.N_Y,
                                           init_id=init_id, seed_id=seed_id,
                                           epochs=C.TRAJECTORY_EPOCHS).items():
                        assert meta[key] == value, (path, key, meta[key], value)
                    omega, delta = data['Omega'], data['Delta']
                    assert omega.shape == delta.shape == (C.TRAJECTORY_EPOCHS + 1,)
                    assert data['measured_AM2'].shape == (C.TRAJECTORY_EPOCHS,)
                    measured[g, r] = data['measured_AM2']
                paths.append(str(path.relative_to(C.DATA_DIR)))
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                for k, (om, de) in enumerate(zip(omega[:-1], delta[:-1])):
                    H = om * hx + de * hz + hi
                    energies, vectors = eigh(H, subset_by_index=[0, 0], check_finite=True)
                    psi = vectors[:, 0]
                    probabilities = psi ** 2
                    exact[g, r, k] = probabilities @ cost_diag
                    measurement_variance[g, r, k] = max(
                        0., probabilities @ (cost_diag ** 2) - exact[g, r, k] ** 2
                    ) / C.N_Y
                    if k in (0, C.TRAJECTORY_EPOCHS - 1):
                        residual = np.linalg.norm(H @ psi - energies[0] * psi)
                        max_residual = max(max_residual, residual)
                        assert residual < 1e-9
                if g == r == 0:
                    np.testing.assert_allclose(exact[g, r, 0], common.exact_am2(nn, params), atol=2e-5)
            print(f'N={N}, N_g={Ng}: all 15 trajectories done ({time.monotonic()-started:.1f}s)', flush=True)
    for array in (measured, exact):
        assert np.isfinite(array).all()
        assert array.min() >= -1e-7 and array.max() <= 1 + 1e-7
    metadata = dict(
        N=N, N_y=C.N_Y, observable='AFM^2', epochs=C.TRAJECTORY_EPOCHS,
        axes=['N_g', 'run (initial point then seed)', 'pre-update epoch'],
        alignment='measured_AM2[k] and exact_AM2[k] both use Omega[k], Delta[k]',
        averaging='Equal weight over 3 initial points x 5 seeds, separately for each N_g',
        solver='scipy.linalg.eigh, float64, lowest eigenpair of full free Hamiltonian',
        measurements='Original saved N_y=100 measurements; no new sampling',
        max_checked_eigenpair_residual=max_residual,
        elapsed_seconds=time.monotonic()-started,
    )
    output = C.DATA_DIR / f'trajectory_expectations_N{N}.npz'
    np.savez_compressed(
        output, metadata=json.dumps(metadata), N_g=np.array(C.TRAJECTORY_NG),
        epoch=np.arange(1, C.TRAJECTORY_EPOCHS + 1), run_ids=np.array(runs),
        measured_AM2=measured, exact_AM2=exact,
        measured_mean_variance=measurement_variance,
        source_files=np.array(paths).reshape(shape[:2]),
        source_sha256=np.array(hashes).reshape(shape[:2]),
    )
    print(f'Saved {output}; max checked residual={max_residual:.3g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', nargs='+', type=int, choices=C.N_LIST, default=C.N_LIST)
    args = parser.parse_args()
    for n in args.N:
        process(n)
