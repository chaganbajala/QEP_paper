# Fixed-shot AFM experiment pipeline

This directory implements the five tasks in `../reseach_plan_2026-09-11.md`.  It imports the
standalone `../afm_shots` package and never imports `Adaptive_bn/qep_shot`.

## Fixed physics and training settings

- open chains with `N=5,9` and unit nearest-neighbour coupling;
- target `AFM^2=1`;
- exact-diagonalization state preparation with the smooth nudge schedule inherited from
  `qep_shot.afm.Ising_Chain_AM2`;
- `beta=0.1`, gradient-descent learning rate `eta=0.01`, and fixed `N_g` during each run;
- `N_y=100`, matching `Adaptive_bn/test_richardson.ipynb`;
- Ω/X and Δ/Z are separate noncommuting measurement groups, so one epoch costs `2*N_g`
  gradient shots. Output shots are stored separately in metadata.

`config.py` is the single source of truth for grids, seeds, initial points, and budgets.  The
three initial points were drawn once with NumPy seed `20260911` and then frozen in the file.

## Files

- `run.py`: resumable local/cluster experiment driver. Each case writes one `.npz` with JSON
  metadata.
- `cluster_dispatch.py`: maps Slurm task IDs to cases.
- `submit_*.sbatch`: four Raven arrays (gradient distributions, trajectories, budgets, and
  training-time gradient moments).
- `analyze.py`: validates that all production cases exist, generates PDF/PNG figures, and writes
  `analysis_summary.json`.
- `validate.py`: checks all 226 expected production files, metadata, shapes/finiteness, and the
  intentional missing-budget mask before analysis.

Production output is written to `../data/shots_2026-09-11/`; figures go to
`../figures/shots_2026-09-11/`; Slurm logs go to `../logs/`. Reduced local checks use filenames
ending in `_smoke` and are ignored by analysis.

## Local smoke commands

Run from the repository root with the `jaxlab` Python:

```bash
MPLCONFIGDIR=/tmp/qep-mpl /Users/qwang/miniforge3/envs/jaxlab/bin/python \
  QEP_multiple_shot/AFM_searching/shot_study/run.py gradient --N 5 --scan ng --smoke
MPLCONFIGDIR=/tmp/qep-mpl /Users/qwang/miniforge3/envs/jaxlab/bin/python \
  QEP_multiple_shot/AFM_searching/shot_study/run.py trajectory --case-id 0 --smoke
MPLCONFIGDIR=/tmp/qep-mpl /Users/qwang/miniforge3/envs/jaxlab/bin/python \
  QEP_multiple_shot/AFM_searching/shot_study/run.py budget --case-id 0 --smoke
MPLCONFIGDIR=/tmp/qep-mpl /Users/qwang/miniforge3/envs/jaxlab/bin/python \
  QEP_multiple_shot/AFM_searching/shot_study/run.py stats --N 5 --smoke
```

## Statistical interpretation

Task 1 always uses the exact ground-state expectation of `AFM^2` as the output `y`, as requested;
only the free/nudged gradient observables are sampled. This isolates the setting in which the CLT
predicts `1/sqrt(N_g)`. Each distribution also contains and plots the exact shot-free QEP-gradient
expectation at the same finite beta. The beta summary plots standard deviation and RMSE separately: the variance
term is amplified roughly as
`1/beta^2`, whereas a U-shaped optimum is expected for total MSE after finite-beta bias is
included—not necessarily for variance alone.

Task 5 records the exact conditional mean and variance of the gradient estimator at each sampled
output `y`. This separates the underlying shot-noise scale from Monte-Carlo uncertainty in a
second diagnostic resampling pass.

## Additional Figure 4 beta scans

The original `gradient_beta_N5.npz` and `gradient_beta_N9.npz` use `N_g=1000`.
`beta_shot_scan.py` adds `gradient_beta_N{N}_Ng{10,100}.npz` using the same
seven beta values, fixed parameter point, exact ground-state output, and 256
repetitions. It computes the free/nudged states once per beta and reuses the
existing measurement sampler. It verifies the exact references against the
original files and cached-state sampling against `call_with_exact_output`.
These files also store the analytic gradient variance for validation.

From `QEP_multiple_shot`, run:

```bash
python -m AFM_searching.shot_study.beta_shot_scan
```

The plotting-only notebooks `figs_N5.ipynb` and `figs_N9.ipynb` contain separate
Figure 4, 4a, and 4b cells for `N_g=1000`, `10`, and `100`. Every title states
the shot count and exact-output convention. Exported supplemental figures use
the names `gradient_beta_tradeoff_N{N}_Ng{10,100}.{pdf,png}`.

The same driver accepts explicit shot counts, for example
`python -m AFM_searching.shot_study.beta_shot_scan --Ng 10000`.
The `N_g=10000` scans use the same grid and repeats and are plotted in separate
Figure 4c cells, with exports named `gradient_beta_tradeoff_N{N}_Ng10000`.

## Figure 11: measured and exact training order

`trajectory_expectations.py` evaluates exact free ground-state `<AFM^2>` at
every saved pre-update parameter point, without retraining or resampling:

```bash
# From QEP_multiple_shot; requires the project's numerical Python environment.
JAX_PLATFORM_NAME=cpu python -m AFM_searching.shot_study.trajectory_expectations
```

Outputs `trajectory_expectations_N{5,9}.npz` contain measured and exact arrays
with axes `(N_g, run, epoch)`, shape `(4, 15, 200)`, plus source-file hashes.
The 15 runs are 3 initial points times 5 seeds, equally weighted for each
`N_g = 10, 100, 1000, 10000`. Saved measurements use `N_y=100`, not `N_g`.
`measured_AM2[k]` is paired with the ground state at `Omega[k], Delta[k]`:
the plotted epochs 1--200 precede updates 1--200. Thus the final plotted point
is after 199 updates, not the terminal state after 200 updates.

Figure 11 in each plotting-only notebook loads these arrays and exports
`performance_measured_vs_exact_N{N}.{pdf,png}`. Solid lines are exact and dashed
lines are measured. The former fixed-budget Figure 11 is retained as 11b.

## Expanded Figure 10 and separated trajectory starts (2026-09-14)

Current scope is **N=5 only**; N=9 is deferred. Original training data remain
read-only. `overlap_training.py` expands the ensemble from 3 starts x 5 seeds
to 10 starts x 5 seeds for each of `N_g=(10,100,1000,10000)`. It preserves the
original three points and appends seven reproducible uniform points. The seed
rule, optimizer, beta=0.1, learning rate=0.01, and N_y=100 are unchanged.

The user requested more overlap, not matching the entire x range. The final
epoch targets are **5000, 500, 200, 200**, giving all four curves a common
gradient-shot range **20,000--100,000**. The preliminary 400,000-shot proposal
was shortened before long production started. Compatible 200-epoch prefixes
from that proposal can be resumed without discarding or altering measurements.
This shot budget excludes output shots, following the original Figure 10.

Outputs go to `data/shots_2026-09-11/trajectories_overlap_2026-09-14/`.
Each NPZ is atomically checkpointed every 200 epochs and records target/completed
epochs and a completion flag. Original prefixes are reused for starts 0--2.
Rerunning the same case/group resumes checkpoints. Smoke data are separate.

The N=5 phase panels now select starts **3, 5, 1**, with (Omega, Delta):

- (0.39073567793440267, 1.1664899930858177)
- (1.16771310144114, 1.0853474197535722)
- (0.6954632245521951, 1.9556663479392087)

The original separated-start preview selected (3,5,8) using geometry only.
The user subsequently chose start 1 instead of start 8 because its existing
trajectories reach higher AFM order in 200 epochs. The current panels are
therefore selected illustrations, not representative performance estimates.
Minimum normalized pairwise separation remains 0.4506, versus 0.2238 for the
original triple; the ten-start statistical ensemble is unchanged.

Panels show all five actual seeds per start through exactly 200 updates, with
no mean parameter path or mean endpoint. Circles mark shared starts and squares
mark individual endpoints. All panels share limits and an exact-AFM background.
A dashed `<AFM^2>=0.9` contour is a quantitative reference, not a phase boundary.
The helpers expose `seed_alpha` and `seed_lw`; each figure has its own editable cell.
For N_g=10000, the replacement green endpoints reach exact AFM squared values
0.9485--0.9705 (mean 0.9596), versus mean 0.3119 for the former start 8.

### Raven workflow

The isolated remote root is
`/raven/u/qinwang/notebooks/QEP_overlap_20260914/QEP_multiple_shot`.
Run submissions there. The preview script schedules 60 real N=5 prefixes,
then the continuation script schedules 25 grouped jobs (8 cases each), at
most 16 GPUs concurrently. Use an `afterok` dependency to avoid concurrent
writes between preview and continuation jobs. The grouped driver explicitly
filters `--N 5`; global case IDs remain stable for reproducibility.

```bash
sbatch AFM_searching/shot_study/submit_overlap_preview_raven.sbatch
# Substitute the returned preview job ID:
sbatch --dependency=afterok:PREVIEW_ID AFM_searching/shot_study/submit_overlap_raven.sbatch
# Substitute the returned continuation job ID:
sbatch --dependency=afterok:CONTINUATION_ID AFM_searching/shot_study/submit_overlap_summary_raven.sbatch
```

After downloading results, from local `QEP_multiple_shot`:

```bash
JAX_PLATFORM_NAME=cpu python -m AFM_searching.shot_study.summarize_overlap --N 5
```

The summarizer validates all 200 runs and reused prefixes before writing
`trajectory_overlap_summary_N5.npz`; incomplete ensembles are not plotted as
complete. Figure 10 in `figs_N5.ipynb` loads this summary when available,
otherwise retaining the original data. Bands are SEM over 10 initial-point
means, each averaging 5 seeds. Phase panels load starts 3 and 5 from the expanded
directory and start 1 from the original `trajectories/` directory. These data
are already available; no new training is required for the replacement.
The notebook remains plotting-only.
