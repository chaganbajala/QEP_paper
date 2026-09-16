"""N5 green trajectory start: 25 seeds at all ten Ng values.

Run and postprocess outside figs_N5.ipynb. All 250 runs are newly generated;
seed bases 1000..1024 follow the init_id=1 trajectory convention. No baseline
data need to be uploaded or combined with a different initial condition.
"""
import argparse
import hashlib
import json

import numpy as np

from . import config as C
from .figure11b_resampling import run_case
from .run import atomic_save

POINT = C.INITIAL_POINTS[1]
OUT = C.DATA_DIR / 'budget_green_init1_N5_2026-09-15'
SUMMARY = C.DATA_DIR / 'time_budget_green_init1_N5_25seeds.npz'
SEEDS = tuple(range(25))


def summarize(*, point=POINT, out=OUT, summary=SUMMARY, init_id=1, seed_offset=1000):
    values = np.empty((len(C.NG_GRID), len(SEEDS), len(C.SHOT_BUDGETS)))
    sources = {}
    checkpoints = np.asarray(C.SHOT_BUDGETS)[None, :] // (2*np.asarray(C.NG_GRID)[:, None])
    for g, ng in enumerate(C.NG_GRID):
        epochs = max(C.SHOT_BUDGETS)//(2*ng)
        for s in SEEDS:
            path = out / f'N5_Ng{ng}_seed{s}.npz'
            with np.load(path, allow_pickle=False) as d:
                m = json.loads(str(d['metadata']))
                for key, expected in dict(N=5, N_g=ng, seed_id=s, base_seed=seed_offset+s,
                                          N_y=100, beta=.1, learning_rate=.01,
                                          epochs=epochs, smoke=False).items():
                    assert m[key] == expected, (path, key)
                np.testing.assert_array_equal(m['point'], point)
                assert m['original_loop_max_state_error'] <= 2e-6
                for key, shape in dict(Omega=(epochs+1,), Delta=(epochs+1,),
                                        measured_AM2=(epochs,), sampled_gradient=(epochs,2)).items():
                    assert d[key].shape == shape and np.isfinite(d[key]).all(), (path,key)
                np.testing.assert_allclose([d['Omega'][0],d['Delta'][0]],point,rtol=0,atol=1e-7)
                np.testing.assert_array_equal(d['checkpoint_epochs'],checkpoints[g])
                np.testing.assert_array_equal(d['budgets'],C.SHOT_BUDGETS)
                values[g,s] = d['checkpoint_exact_AM2']
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    valid = checkpoints >= 1
    assert np.array_equal(np.isfinite(values).all(axis=1),valid)
    finite = values[np.isfinite(values)]
    assert finite.min() >= -1e-6 and finite.max() <= 1+1e-6
    indices = np.random.default_rng(20260914).integers(0,25,size=(20000,25))
    boot = values[:,indices,:].mean(axis=2)
    lo, hi = np.quantile(boot,[.025,.975],axis=1)
    meta = dict(N=5,N_y=100,point=point,init_id=init_id,initial_points=1,seeds=25,
                base_seeds=list(range(seed_offset,seed_offset+25)),complete=True,
                uncertainty='pointwise percentile bootstrap over seed blocks, 20000 draws',
                state_solver='historical 100-step approximation, compiled with equivalence checks',
                endpoint_solver='scipy.linalg.eigh complex128',
                legacy_budget_definition='B=2*Ng*epochs; paired preparations',
                physical_gradient_shots='4*Ng*epochs; cap S=2*B; Ny output excluded',
                source_sha256=sources)
    atomic_save(summary,metadata=json.dumps(meta,sort_keys=True),N_g=np.asarray(C.NG_GRID),
                budgets=np.asarray(C.SHOT_BUDGETS),physical_budgets=2*np.asarray(C.SHOT_BUDGETS),
                checkpoint_epochs=checkpoints,seed_ids=np.asarray(SEEDS),values=values,
                mean=values.mean(axis=1),ci95_low=lo,ci95_high=hi)
    print(f'Validated all 250 cases; saved {summary}',flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(SUMMARY,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']))
        assert meta['complete'] and meta['seeds']==25
        fig,ax=plt.subplots(figsize=(4.8,3.8),layout='constrained')
        for b,cap in enumerate(d['physical_budgets']):
            valid=np.isfinite(d['mean'][:,b])
            x=d['N_g'][valid]
            line,=ax.plot(x,d['mean'][valid,b],'o-',lw=1.5,ms=4,label=rf'$S={cap:,}$')
            ax.fill_between(x,d['ci95_low'][valid,b],d['ci95_high'][valid,b],
                            color=line.get_color(),alpha=.12,linewidth=0)
        ax.set(xscale='log',xlabel=r'$N_g$ shots per basis per state',
               ylabel=r'exact ground-state $\langle AFM^2\rangle$',
               title=r'$N=5$, $N_y=100$; green start, 25 seeds'+'\n'+
                     rf'$(\Omega_0,\Delta_0)=({POINT[0]:.4f},{POINT[1]:.4f})$')
        ax.grid(alpha=.22)
        ax.legend(frameon=False,fontsize=8)
        C.FIG_DIR.mkdir(parents=True,exist_ok=True)
        for suffix in ('.pdf','.png'):
            path=C.FIG_DIR/f'time_budget_N5_green_init1_25seeds{suffix}'
            fig.savefig(path,bbox_inches='tight',**({'dpi':180} if suffix=='.png' else {}))
            print(f'Saved {path}',flush=True)
        plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','summarize','plot','finish'))
    parser.add_argument('--case-id',type=int,choices=range(250))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='run':
        if args.case_id is None: parser.error('run requires --case-id')
        run_case(args.case_id,args.smoke,point=POINT,output_dir=OUT,seeds=SEEDS,
                 seed_offset=1000,task='green_init1_budget')
    else:
        if args.action in ('summarize','finish'): summarize()
        if args.action in ('plot','finish'): plot()
