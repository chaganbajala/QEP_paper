"""Cache exact expectations along completed green-start paths; no new training.

Run from QEP_multiple_shot as AFM_searching.shot_study.green_trajectory_expectations.
All eigensolves stay outside the editable, plotting-only notebook.
"""
import hashlib
import json
import time

import dynamiqs as dq
import numpy as np
from scipy.linalg import eigh
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:
    # Cluster job scripts already constrain BLAS threads before Python starts.
    # Avoid making this optional performance helper a numerical dependency.
    from contextlib import nullcontext
    import os

    def threadpool_limits(*, limits):
        if limits != 1 or os.environ.get('OPENBLAS_NUM_THREADS') != '1':
            raise RuntimeError('Install threadpoolctl or launch with OPENBLAS_NUM_THREADS=1')
        print('threadpoolctl unavailable; using externally configured single-thread BLAS.', flush=True)
        return nullcontext()

from ..afm_shots import common
from . import config as C
from .green_start_budget import OUT, POINT, SUMMARY
from .run import atomic_save


def process(*, point=POINT, out=OUT, summary=SUMMARY, seed_offset=1000,
            output=C.DATA_DIR/'green_trajectory_expectations_N5_25seeds.npz'):
    started = time.monotonic()
    nn, params, _ = common.build_model_and_grad(5, *point)
    def matrix(op):
        a = np.asarray(dq.to_jax(op))
        np.testing.assert_allclose(a.imag, 0, atol=1e-12)
        return a.real.astype(np.float64)
    hx, hz, hi, cost = map(matrix, (nn.H_Omega, nn.H_Delta, nn.H_int, nn.cost_ops))
    np.testing.assert_allclose(cost, np.diag(np.diag(cost)), atol=1e-12)
    np.testing.assert_allclose(point[0]*hx+point[1]*hz+hi, matrix(nn.set_H_f(params)), atol=1e-6)
    cost_diag = np.diag(cost)
    with np.load(summary, allow_pickle=False) as d:
        baseline = json.loads(str(d['metadata']))
        assert baseline['complete'] and baseline['seeds'] == 25
        np.testing.assert_array_equal(baseline['point'], point)
    arrays, sources = {}, {}
    max_residual, max_checkpoint_error = 0., 0.
    with threadpool_limits(limits=1):
        for ng in C.NG_GRID:
            epochs = max(C.SHOT_BUDGETS)//(2*ng)
            measured = np.empty((25,epochs))
            exact = np.empty((25,epochs))
            final = np.empty(25)
            for seed in range(25):
                path = out/f'N5_Ng{ng}_seed{seed}.npz'
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == baseline['source_sha256'][path.name]
                sources[path.name] = digest
                with np.load(path, allow_pickle=False) as d:
                    meta = json.loads(str(d['metadata']))
                    for key, value in dict(N=5,N_g=ng,N_y=100,seed_id=seed,
                                           base_seed=seed_offset+seed,epochs=epochs,smoke=False).items():
                        assert meta[key] == value, (path,key)
                    np.testing.assert_array_equal(meta['point'],point)
                    om, de = d['Omega'], d['Delta']
                    assert om.shape == de.shape == (epochs+1,)
                    assert d['measured_AM2'].shape == (epochs,)
                    measured[seed] = d['measured_AM2']
                    checkpoints = d['checkpoint_epochs']
                    saved_endpoints = d['checkpoint_exact_AM2']
                whole = np.empty(epochs+1)
                for k, (omega, delta) in enumerate(zip(om,de)):
                    h = omega*hx+delta*hz+hi
                    energy, vec = eigh(h,subset_by_index=[0,0],check_finite=True)
                    psi = vec[:,0]
                    whole[k] = psi**2 @ cost_diag
                    if k in (0,epochs):
                        residual = float(np.linalg.norm(h@psi-energy[0]*psi))
                        max_residual = max(max_residual,residual)
                        assert residual < 1e-9
                exact[seed], final[seed] = whole[:-1], whole[-1]
                valid = (checkpoints>=1)&(checkpoints<=epochs)
                error = float(np.max(abs(whole[checkpoints[valid]]-saved_endpoints[valid])))
                max_checkpoint_error = max(max_checkpoint_error,error)
                # Old endpoint code assembled H in float32 before diagonalizing;
                # here H is assembled and diagonalized in float64.
                np.testing.assert_allclose(whole[checkpoints[valid]],saved_endpoints[valid],rtol=0,atol=2e-5)
            for a in (measured,exact,final):
                assert np.isfinite(a).all() and a.min()>=-1e-6 and a.max()<=1+1e-6
            arrays[f'epoch_Ng{ng}'] = np.arange(1,epochs+1)
            arrays[f'final_exact_Ng{ng}'] = final
            for label,values in (('measured',measured),('exact',exact)):
                arrays[f'{label}_AM2_Ng{ng}'] = values
                arrays[f'{label}_mean_Ng{ng}'] = values.mean(axis=0)
                arrays[f'{label}_sem_Ng{ng}'] = values.std(axis=0,ddof=1)/5
            print(f'Ng={ng}: 25 paths x {epochs} pre-update points; elapsed={time.monotonic()-started:.1f}s',flush=True)
    first = arrays['exact_AM2_Ng10'][0,0]
    for ng in C.NG_GRID:
        np.testing.assert_allclose(arrays[f'exact_AM2_Ng{ng}'][:,0],first,atol=1e-12)
    meta = dict(N=5,N_y=100,point=point,seeds=25,initial_points=1,complete=True,
                observable='AFM^2',alignment='Epoch k+1: saved measurement and exact ground state at Omega[k],Delta[k], before update',
                state_solver='Training retains historical 100-step approximation',
                expectation_solver='scipy.linalg.eigh float64 Hamiltonian and lowest eigenpair',
                measurements='Saved Ny=100 measurements; no resampling',
                uncertainty='Pointwise +/-1 SEM over 25 sampling seeds at one fixed start',
                max_checked_residual=max_residual,max_budget_checkpoint_error=max_checkpoint_error,
                initial_exact_AM2=float(first),source_sha256=sources,
                elapsed_seconds=time.monotonic()-started)
    atomic_save(output,metadata=json.dumps(meta,sort_keys=True),N_g=np.array(C.NG_GRID),**arrays)
    print(f'Saved {output}; residual={max_residual:.3g}, checkpoint difference={max_checkpoint_error:.3g}',flush=True)


if __name__ == '__main__':
    process()
