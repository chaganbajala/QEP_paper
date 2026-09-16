"""Additional N=5 seeds for Figure 11b; keep the historical 100-step algorithm.

The state-iteration loop is compiled, not replaced by an exact solver. Each
worker checks it against the original loop. Measurement/update code is reused.
No original data are overwritten. Use `run --case-id 0..199`, then `summarize`.
"""
import argparse
import json
import time

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from AFM_searching.afm_shots import common
from AFM_searching.afm_shots.train import Shot_QEP_Training
from AFM_searching.shot_study import config as C
from AFM_searching.shot_study.run import atomic_save, epoch_seed, metadata, scalar_pair
from QEP import model

OUT = C.DATA_DIR / "budget_noise_check_N5_2026-09-14"
SUMMARY = C.DATA_DIR / "figure11b_resampling_N5.npz"
EXTRA_SEEDS = tuple(range(5, 25))


@jax.jit
def compiled_state(h):
    v = jnp.ones(h.shape[0])
    v = v / jnp.linalg.norm(v)
    v = dq.asqarray(v[:, None], dims=h.dims)
    # The first update promotes the original real initial vector to complex.
    first = model.sparse_update(v, 0.*v, h)
    return jax.lax.fori_loop(
        1, 100, lambda i, carry: model.sparse_update(carry[0], carry[1], h), first)[0]


def run_case(case_id, smoke=False, *, point=C.TYPICAL_POINT, output_dir=OUT,
             seeds=EXTRA_SEEDS, seed_offset=0, task="figure11b_resampling"):
    ng, seed = [(ng, seed) for ng in C.NG_GRID for seed in seeds][case_id]
    path = output_dir / f"N5_Ng{ng}_seed{seed}{'_smoke' if smoke else ''}.npz"
    epochs = 4 if smoke else max(C.SHOT_BUDGETS)//(2*ng)
    if path.exists():
        with np.load(path) as d:
            assert len(d["measured_AM2"]) == epochs
            saved_meta = json.loads(str(d['metadata']))
            np.testing.assert_array_equal(saved_meta['point'], point)
            assert saved_meta.get('base_seed', seed) == seed + seed_offset
            assert saved_meta['N_g'] == ng and saved_meta['seed_id'] == seed
        print(f"already complete: {path}", flush=True)
        return
    nn, params, gradient = common.build_model_and_grad(5, *point)
    errors = []
    for check_point, y in ((point, .9), ((.1,1.4), .98)):
        pp = (check_point[0], check_point[1], params[2])
        for beta in (0., C.BETA):
            h = nn.total_H(pp, gradient.solver.param_funcs, y, 1., beta)(nn.T)
            original = np.asarray(dq.to_jax(nn.set_psi0(h)))
            fast = np.asarray(dq.to_jax(compiled_state(h)))
            error = float(np.max(abs(original-fast)))
            errors.append(error)
            np.testing.assert_allclose(fast, original, rtol=0, atol=2e-6)
    def prepare(h, itr_num=100):
        assert itr_num == 100
        return compiled_state(h)
    nn.set_psi0 = prepare  # This process-local model instance only.
    trainer = Shot_QEP_Training(C.LEARNING_RATE)
    omega, delta = np.empty(epochs+1), np.empty(epochs+1)
    measured, sampled = np.empty(epochs), np.empty((epochs,2))
    omega[0], delta[0] = scalar_pair(params)
    started = time.monotonic()
    for k in range(epochs):
        y, g = gradient(params, nn, C.TARGET, N_y=C.N_Y, N_g=ng,
                        seed=epoch_seed(seed + seed_offset,k))
        measured[k], sampled[k] = float(y), scalar_pair(g)
        params = trainer.update(params,g)
        omega[k+1],delta[k+1] = scalar_pair(params)
        if (k+1) % max(1,epochs//5) == 0:
            print(f"Ng={ng} seed={seed} epoch={k+1}/{epochs} elapsed={time.monotonic()-started:.1f}s",flush=True)
    budgets = np.array(C.SHOT_BUDGETS)
    checkpoint_epochs = budgets//(2*ng)
    exact = np.full(len(budgets),np.nan)
    for i,k in enumerate(checkpoint_epochs):
        if 0 < k <= epochs:
            h = np.asarray(dq.to_jax(nn.set_H_f((omega[k],delta[k],params[2]))),dtype=np.complex128)
            psi = eigh(h,subset_by_index=[0,0])[1][:,0]
            op = np.asarray(dq.to_jax(nn.cost_ops),dtype=np.complex128)
            exact[i] = np.vdot(psi,op@psi).real
    meta = json.loads(metadata(task=task,N=5,N_g=ng,seed_id=seed,
                              point=point,epochs=epochs,smoke=smoke))
    meta.update(state_solver="normalized_momentum_100_compiled",
                state_iterations=100,original_loop_max_state_error=max(errors),
                gradient_shots_per_epoch_expression="4*N_g",
                legacy_budget_definition="B=2*N_g*epochs; counts paired preparations",
                checkpoint_solver="scipy.linalg.eigh_complex128",
                base_seed=seed + seed_offset,
                jax_version=jax.__version__,backend=jax.default_backend())
    atomic_save(path,metadata=json.dumps(meta,sort_keys=True),budgets=budgets,
                checkpoint_epochs=checkpoint_epochs,checkpoint_exact_AM2=exact,
                Omega=omega,Delta=delta,measured_AM2=measured,sampled_gradient=sampled,
                elapsed_seconds=np.asarray(time.monotonic()-started))


def summarize():
    # Pair seed IDs across Ng, preserving any common-random-number dependence.
    all_values = np.empty((len(C.NG_GRID),25,len(C.SHOT_BUDGETS)))
    for i,ng in enumerate(C.NG_GRID):
        for seed in range(25):
            directory = C.DATA_DIR/"budgets" if seed < 5 else OUT
            with np.load(directory/f"N5_Ng{ng}_seed{seed}.npz") as d:
                meta=json.loads(str(d["metadata"]))
                assert (meta["N"],meta["N_g"],meta["seed_id"]) == (5,ng,seed)
                assert meta["N_y"] == 100 and meta["beta"] == .1
                assert not meta["smoke"]
                np.testing.assert_array_equal(d["budgets"],C.SHOT_BUDGETS)
                np.testing.assert_array_equal(d["checkpoint_epochs"],np.array(C.SHOT_BUDGETS)//(2*ng))
                all_values[i,seed] = d["checkpoint_exact_AM2"]
    valid=np.array(C.SHOT_BUDGETS)[None,:]//(2*np.array(C.NG_GRID)[:,None]) >= 1
    assert np.array_equal(np.isfinite(all_values).all(axis=1),valid)
    assert np.all((all_values[np.isfinite(all_values)]>=-1e-6)&(all_values[np.isfinite(all_values)]<=1+1e-6))
    rng=np.random.default_rng(20260914)
    indices=rng.integers(0,25,size=(20000,25))
    mean=all_values.mean(axis=1)
    boot=all_values[:,indices,:].mean(axis=2)
    lo,hi=np.quantile(boot,[.025,.975],axis=1)
    comparisons={}
    for budget,left,dip,right in ((50000,20,50,100),(100000,50,100,200)):
        j=C.SHOT_BUDGETS.index(budget)
        # Interpolate between neighboring log-Ng values; report exploratory CI.
        w=(np.log(dip)-np.log(left))/(np.log(right)-np.log(left))
        contrast=all_values[C.NG_GRID.index(dip),:,j]-(
            (1-w)*all_values[C.NG_GRID.index(left),:,j]+w*all_values[C.NG_GRID.index(right),:,j])
        ci=np.quantile(contrast[indices].mean(axis=1),[.025,.975])
        comparisons[f"B{budget}_Ng{dip}"]=dict(
            original_mean=float(contrast[:5].mean()),extra_mean=float(contrast[5:].mean()),
            combined_mean=float(contrast.mean()),paired_bootstrap_ci95=ci.tolist())
    meta=dict(N=5,N_y=100,seeds=25,initial_points=1,complete=True,
              state_solver="historical_100_step_approximation",
              uncertainty="pointwise percentile bootstrap over seed blocks, 20000 draws",
              comparisons=comparisons,legacy_budget_definition="B=2*N_g*epochs",
              physical_gradient_budget_definition="4*N_g*floor(B/(2*N_g))")
    atomic_save(SUMMARY,metadata=json.dumps(meta,sort_keys=True),N_g=np.array(C.NG_GRID),
                budgets=np.array(C.SHOT_BUDGETS),seed_ids=np.arange(25),values=all_values,
                mean=mean,ci95_low=lo,ci95_high=hi,original_mean=all_values[:,:5].mean(axis=1),
                extra_mean=all_values[:,5:].mean(axis=1))
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("run","summarize"))
    parser.add_argument("--case-id",type=int,choices=range(200))
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.action == "run":
        if args.case_id is None: parser.error("run requires --case-id")
        run_case(args.case_id,args.smoke)
    else: summarize()
