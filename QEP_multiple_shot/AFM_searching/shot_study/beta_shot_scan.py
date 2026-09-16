"""Additional Figure 4 beta scans, reusing fixed states across measurements.

Run from QEP_multiple_shot with:
    python -m AFM_searching.shot_study.beta_shot_scan
No data generation is performed in the plotting notebooks.
"""
import argparse
import time

import dynamiqs as dq
import jax
import numpy as np

from AFM_searching.afm_shots import common
from AFM_searching.shot_study import config as C
from AFM_searching.shot_study.run import atomic_save, epoch_seed, metadata, scalar_pair


def run(N, repeats=256, smoke=False, shot_counts=(10, 100)):
    beta_values = C.BETA_GRID[:2] if smoke else C.BETA_GRID
    shot_counts = tuple(shot_counts)
    if not shot_counts or any(ng < 1 for ng in shot_counts):
        raise ValueError('Shot counts must be positive')
    gradients = {ng: np.empty((len(beta_values), repeats, 2)) for ng in shot_counts}
    outputs = np.empty((len(beta_values), repeats))
    exact = np.empty((len(beta_values), 2))
    analytic_variance = np.empty_like(exact)
    started = time.time()
    with np.load(C.DATA_DIR / f"gradient_beta_N{N}.npz") as original:
        for i, beta in enumerate(beta_values):
            nn, params, method = common.build_model_and_grad(N, *C.TYPICAL_POINT, beta=beta)
            H = nn.set_H(params, method.solver.param_funcs)
            initial = nn.set_psi0(H(0.))
            free = method.solver(nn, params, initial, 0., C.TARGET, C.TARGET)[-1]
            y = dq.expect(nn.cost_ops, free).real
            nudged = method.solver(nn, params, initial, beta, y, C.TARGET)[-1]
            operators = method.grad_H(params, nn)
            leaves = jax.tree_util.tree_leaves(operators, is_leaf=lambda x: isinstance(x, dq.QArray))
            for p, op in enumerate(leaves[:2]):
                mf, vf = method._operator_moments(op, free)
                mn, vn = method._operator_moments(op, nudged)
                exact[i, p] = float((mn - mf) / beta)
                analytic_variance[i, p] = float((vn + vf) / beta**2)
            outputs[i] = float(y)
            np.testing.assert_allclose(exact[i], original['exact_gradients'][i], rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(outputs[i], original['exact_outputs'][i], rtol=2e-5, atol=2e-5)

            def sample(ng, seed):
                gf = method.H_params_gradient(params, free, nn, N_g=ng, seed=seed + 100)
                gn = method.H_params_gradient(params, nudged, nn, N_g=ng, seed=seed + 1000)
                return scalar_pair(jax.tree_util.tree_map(method.grad_func, gn, gf))

            # Check that cached-state sampling agrees with the public estimator.
            seed0 = epoch_seed(910000 + 1000 * N, 0)
            check_ng = shot_counts[0]
            _, direct = method.call_with_exact_output(params, nn, C.TARGET, N_g=check_ng, gradient_seed=seed0)
            np.testing.assert_allclose(sample(check_ng, seed0), scalar_pair(direct), rtol=1e-6, atol=1e-6)
            for ng in shot_counts:
                for r in range(repeats):
                    gradients[ng][i, r] = sample(ng, epoch_seed(910000 + 1000 * N, r))
            print(f'N={N} beta={beta:g} complete ({time.time()-started:.1f}s)', flush=True)

    for ng in shot_counts:
        suffix = '_smoke' if smoke else ''
        atomic_save(
            C.DATA_DIR / f'gradient_beta_N{N}_Ng{ng}{suffix}.npz',
            metadata=metadata(task='gradient_distribution', N=N, scan='beta', N_g=ng,
                              repeats=repeats, smoke=smoke, point=C.TYPICAL_POINT,
                              N_y=0, output_shots_per_epoch=0,
                              output_estimator='exact_ground_state_expectation',
                              state_reuse=True, seed_base=910000 + 1000 * N),
            scan_values=np.asarray(beta_values), gradients=gradients[ng], outputs=outputs,
            exact_gradients=exact, exact_outputs=outputs[:, 0],
            analytic_gradient_variance=analytic_variance / ng,
            elapsed_seconds=np.asarray(time.time()-started),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N', type=int, nargs='+', default=[5, 9])
    parser.add_argument('--Ng', type=int, nargs='+', default=[10, 100])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for N in args.N:
        run(N, repeats=4 if args.smoke else C.GRADIENT_REPEATS,
            smoke=args.smoke, shot_counts=args.Ng)
