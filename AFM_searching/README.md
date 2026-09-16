# AFM searching with quantum equilibrium propagation

Local repeat of the cluster study `/u/qinwang/notebooks/QEP_sparse/Fixed_TN`
(zeropoint), following `plan.md`.

## Layout

```
AFM_searching/
├── plan.md                  the brief
├── QEP_GPT-12.pdf           the O(1)-bias theory this study tests (item 4)
├── code/                    self-contained -- imports nothing from other projects
│   ├── afmqep/              models, gradients, training loop, schedules, ground states
│   ├── phase_diagram.py     ground-state phase diagrams (sparse, no time evolution)
│   ├── gs_survey.py         degeneracy / Neel-weight survey  (plan item 6 gate)
│   ├── degeneracy_check.py  why <AM> is basis-dependent on even lattices
│   ├── train_run.py         one QEP training run (ED / AE / CAE)
│   ├── train_ed_sparse.py   exact-QEP training for the 4x4 lattice (dim 65536)
│   ├── bias_scan.py         gradient bias vs T and N        (plan item 4)
│   ├── spectrum.py          full instantaneous spectrum along the sweep
│   ├── cae_mechanism.py     what breaks the cached estimator (report appendix)
│   ├── even_lattice_check.py  training against a symmetry-blind readout
│   ├── summarize.py         every number quoted in the report
│   ├── make_tables.py       the report's tables, straight from the data
│   ├── figstyle.py          shared palette / matplotlib style
│   └── make_figs.py         all figures
├── cluster/                 job scripts (see below)
├── data/                    results (npz / json)
├── figs/                    one directory per system size and dimensionality
├── logs/                    per-task logs; logs/cluster/ mirrors the cluster's
└── report/                  LaTeX report -> report/main.pdf
```

## Where the data is

Everything was computed on **raven**, in

```
/ptmp/qinwang/notebooks/AFM_searching/
```

which holds the same `code/`, `cluster/`, `data/`, `logs/` tree. Two late pieces -- the
N=9 gradient-bias scans and the longest-`T` (2000, 5000) training runs at N=7, 9 and 3x3 --
were finished locally, on the same code and the same task lists, after the SSH connection to
the gateway dropped; `cluster/run_local.sh` runs any stage that way. The local
`data/` here is an rsync of the cluster's; `bash cluster/sync.sh down` refreshes
it and `bash cluster/sync.sh up` pushes code back (it never copies the task
lists, which contain absolute paths of whichever machine generated them).

## Running it

```bash
bash cluster/make_tasks.sh          # build the task lists (do this once)
sbatch cluster/run_phase.sbatch     # phase diagrams + ground-state surveys
sbatch cluster/run_train.sbatch     # the main T-sweep, chains and 2D lattices
sbatch cluster/run_multiinit.sbatch # the alternative starting points
sbatch cluster/run_extra.sbatch     # 4x4 exact QEP (sparse)
sbatch cluster/run_bias.sbatch      # gradient-bias scans
```

Every stage is **resumable**: `cluster/filter_done.py` drops tasks whose output
file already exists, so a stage can simply be resubmitted to fill gaps.
`bash cluster/run_local.sh <stage> [nproc]` runs the same lists on a laptop.

Then, locally:

```bash
python code/summarize.py            # -> data/summary.json
python code/make_figs.py all        # -> figs/
cd report && tectonic main.tex      # -> report/main.pdf
```

## Conventions that differ from the cluster study

* Interaction is `sum V n_i n_j`, not `0.5 * sum V n_i n_j` -- a factor 2, which
  is why the phase diagram is redrawn and why the start point is
  `(Omega, Delta) = (1.0, 2.6)` rather than the cluster's `(0.5, 1.3)`.
* Order parameter is the staggered magnetisation `AM` (and `AM^2`), not the
  nearest-neighbour ZZ correlator.
* The QEP nudge is switched on smoothly over the sweep, not abruptly.
* Ground states are obtained by diagonalisation rather than by a fixed number of
  Rayleigh-quotient steps.

Hyperparameters are otherwise unchanged: `beta = 0.1`, learning rate `0.01`,
target `1`, `dt = 0.1`, total physical time budget `T_tot = 1e5` so that
`N_epoch = T_tot / T`, with `T` from 10 to 5000.
