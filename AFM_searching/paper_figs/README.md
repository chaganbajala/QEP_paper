# paper_figs

Figures requested in `figure_plan.md`, kept separate from the main report's
`figs/`/`data/`/`code/` so they can be assembled into a paper independently.

## Layout

- `code/` -- the plotting and data-packaging scripts that produced everything
  here (plus copies of `figstyle.py` and `make_figs.py` from
  `AFM_searching/code/`, reused as-is).
- `data/` -- the inputs each script reads: copies of the relevant `.npz` files
  from `AFM_searching/data/{bias,train,phase}`, plus the new lattice bias
  scans and stiff-schedule training runs generated for this task.
- `figs/` -- the output PDFs/PNGs.

## Editable figure notebooks

### Gradient-oscillation decay companions

**Current approach (1–2–5 benchmark windows):**
`python code/plot_benchmark_decay.py` produces `grad_benchmark_example_N3.pdf/png`
(one dense short-T demonstration above local amplitude-decay panels) and
`grad_benchmark_decay_chains.pdf/png` (amplitude comparison across sizes).
It also generates a separate three-panel figure for each chain:
`grad_benchmark_decay_chain_N{1,3,5,7,9}.pdf/png`, showing Ω amplitude,
Δ amplitude, and total relative RMS error (with centered oscillation dashed).
By default all five use the same completed, numerically checked benchmarks
across both schedules, with matching x and corresponding y limits. The current
shared grid is complete through T=2000: T=20,50,100,200,500,1000,2000
(15 September afternoon refresh). All five chains have all eleven input
benchmark windows, including the short-T windows excluded from decay plots.
All higher-T measurements remain stored. To render full-available-data
supplements separately, run
`python code/plot_benchmark_decay.py --coverage available --outdir figs/benchmark_available --summary data/bias_benchmarks/benchmark_available_summary.json`.
The input grid is T=1,2,5,10,20,50,100,200,500,1000,2000. Each window is
centered on its benchmark and has width `min(20, 0.8*T)`, so it stays positive
and reasonably local. Initial spacing is at most 0.25; the grid is doubled
only when a resolved oscillation's half-grid RMS check differs by more than
5%, with a maximum of one refinement. Plotted finite-window RMS values require
a passing resolution check and a tighter-solver endpoint check below 1% of
the measured vector RMS. A peak-count screen originally flagged stiff N=5/T=500
and N=9/T=20. Both have now received expanded-window checks at the same physical
parameters. For these audited points, hollow markers indicate over 10% variation
under shifts by ±12.5% of the original width or width changes of ±25% (or failed
component sampling checks). Flags apply separately to each displayed statistic.
N=9/T=20 passes and is filled; the N=5/T=500 component RMS and total error remain
hollow. Other points retain the original component peak-count screen; a filled
marker is not a claim that every point received an expanded-window audit.
The audit-derived vertical bars give the range across tested windows, **not
confidence intervals or numerical error bars**. All original central values
and original benchmark windows are unchanged. Numerical failures remain omitted;
no missing values are interpolated. These diagnostics are not rigorous
uncertainty bounds. T<20 remains outside the decay plots, with all raw data
retained.

Reproduce the two focused window checks with
`python code/check_benchmark_window_stability.py --N 5` and `--N 9`, then run
`python code/plot_benchmark_window_stability.py` for the expanded raw/RMS figures
`figs/grad_window_stability_chain_N5_T500.pdf/png` and
`figs/grad_window_stability_chain_N9_T20.pdf/png`. Separate data and hashed audit
manifests live under `data/bias_benchmarks/window_stability/`; the main plotter
validates their hashes before applying the diagnostic markers and range bars.
N=5 windows of width 20,30,40 remain sensitive; widening is not used to silently
replace its central estimate. N=9 widths 16,20,24 pass these local stability
checks, but the original width 16 is retained for locality and comparability.

`sample_benchmark_windows.py --N 3 --example --example-max 200` creates the
extended dense example (T=0.25–200, step=0.125); use `--N 1` or `--N 5` with
the same example options for the additional chains. Without `--example`,
the sampler computes benchmark windows. The plotter also creates standalone
two-panel `grad_oscillations_chain_N{1,3,5}.pdf/png` from completed dense scans,
using the same layout, schedule colors, ED reference and W20/W50 shading.
The original T≤60 example is preserved. The plotter uses the T≤200
example by default; pass `--example-max 60` to reproduce the earlier range. The example
and per-benchmark NPZs live separately in `data/bias_benchmarks/`, with one
resumable checkpoint per window. Exact matching samples from compatible,
normalized, tight-solver scans are reused without interpolation. The plotter
records source hashes, missing benchmarks and all quality flags in
`data/bias_benchmarks/benchmark_amplitude_summary.json`.

These figures describe finite-nudge QEP at beta=0.1, not the infinitesimal
response. The N=1 beta/sensitivity diagnostic recovers an approximately
constant stiff-schedule RMS as beta tends to zero; see
`../theory_check/n1_gradient_limits.md` for the order-of-limits explanation.

On the user's instruction, unfinished old refinement tasks `30248568_2,_3,_4`
(N=5,7,9) were canceled, with checkpoints preserved; N=1 and N=3 had already
completed. Replacement array **30248922** is now fully complete (all exit 0).
Its indices 0–3 = N=1,5,7,9 ran four
benchmark groups per allocation: T≤200, T=500, T=1000 and T=2000. N=3 is
computed locally. See `cluster/run_gradient_benchmarks.sbatch` (relative to
`AFM_searching/`). Each group retains the strict normalized AE solver and
uses the isolated Raven root listed below. Collect completed
`paper_figs/data/bias_benchmarks/benchmark_chain_N*_T*.npz` files, excluding
`.partial.npz` checkpoints, then rerun the new plotter. The baseline has 1238
gradient evaluations per N across both schedules, before reuse/refinement,
versus 2522 in the superseded scan; T-weighted compute cost is about 29%.

**Earlier half-octave-window companions (retained for comparison):**
`python code/plot_gradient_decay.py` generates `figs/grad_decay_chain_N{1,3,5,7,9}.pdf/png`
and `figs/grad_decay_chains.pdf/png`, without replacing the original `grad_vs_T_*`
figures. The component panels show local RMS oscillation about each window's
own mean, not a peak envelope or statistical uncertainty. The relative-error
panel distinguishes total RMS deviation from ED (solid) from centered
oscillation (dashed); their squared difference is the squared mean offset.
Segments connect window statistics only; no raw oscillations are interpolated
across unsampled gaps. Small nonmonotonic features are retained.

The plotter prefers completed, separately stored `data/bias_refined/` scans,
otherwise using the original data. Each figure identifies its source and
sampling density. `data/gradient_decay_summary.json` records input hashes,
per-window statistics, half-grid sensitivity, and available tighter-solver checks.
The chain comparison may therefore mix refined and original scans; consult
that summary for provenance. A half-grid check is a sampling diagnostic, not
a rigorous uncertainty bound.

`python code/refine_gradient_windows.py --N 5` computes new ordinary-AE windows
at the unchanged base point `(Omega, Delta)=(1, 2.6)`, beta=0.1 and target=1.
Defaults are 97 uniform samples per width-30 window and half-octave window
starts from 50 to 1600; `--starts` can extend the range. It uses x64 Dopri8,
rtol=1e-10, atol=1e-12, normalized endpoint expectations and a raw norm² guard,
with tighter-tolerance endpoint spot checks. Checkpoints every ten samples
allow resuming identical configurations. It never overwrites the original
bias scans or computes cached AE. Use `--benchmark` for a runtime estimate.

Tests: `python -m unittest discover -s code -p test_gradient_decay.py`.

The superseded Raven refinement submitted on 2026-09-15 used job array `30248568`, one
allocation per N=1,3,5,7,9, and four independent window groups per allocation.
It extends all chains to T=3230 with 13 windows × 97 samples per schedule.
The isolated remote root is
`/ptmp/qinwang/notebooks/AFM_searching/amplitude_refinement_20260915/`.
Each allocation calls `merge_gradient_windows.py` only after all four groups
finish successfully, producing `paper_figs/data/bias_refined/bias_chain_N*_AM_refined.npz`.
To collect results, preserve any previous local refined scan in an archive
before replacing it, copy only complete merged files, then rerun
`plot_gradient_decay.py`. Do not combine old and new samples inside a window:
the original scans used different integration tolerances and unnormalized
expectations. The original `grad_vs_T_*` plots and their fits remain unchanged.

### Training-figure notebooks

- `figs_N5.ipynb` reproduces the training figures in `figs/chain_N5/`.
- `figs_L3x3.ipynb` reproduces the training figures in `figs/lattice_3x3/`.

Each notebook includes the actual plotting code, grouped by figure family,
for both AM and AM2 at the base initial point. Section 3 has separate editable
trajectory cells for AFM (AM) and AFM² (AM2). Run top to bottom with a Python
kernel containing NumPy, Matplotlib, and IPython. They read the existing
`data/train/` and `data/phase/` files and reuse `code/figstyle.py`; they do not
launch training or download data. Figures display inline and, by default,
replace the corresponding PDF/PNG files using the original names. Set
`SAVE_FIGURES = False` for inline-only previews, or set the environment variable
`AFM_NOTEBOOK_FIGURE_ROOT` to use a different output root.

Section **2b** in both notebooks combines figures 1 and 2 side by side, separately for AM and
AM2, with a shared y-axis and one legend below both panels. Nine distinct colors
use a fixed T-to-color mapping across both panels and observables. Its additional
outputs are `figs/{chain_N5,lattice_3x3}/cost_vs_epoch_time_{AM,AM2}_base.pdf/png`;
the original separate figures remain available.
The 3×3 notebook also matches the current N5 figure 4 settings: compact
2.2-inch-square panels with an external two-column budget legend.

Figure 3b alone follows the multiple-shot `phase_trajectories_N5_Ng100_cell62`
appearance: `afmhot` background at alpha 0.7, blue ED / green AE paths with
white outlines, circular starts, square endpoints, and an AFM²=0.9 contour.
The original axis ranges are retained so the base start at Delta=2.6 stays
visible. Other notebook sections keep their existing styles.

Both notebooks additionally include their top-level combined-trajectory figure
in section **3c**, with its full editable plotting code. The separate
gradient-bias figures are not included. Run
`python code/test_figure_notebooks.py` to execute both notebooks in temporary
output directories and verify their output filenames (requires nbformat,
nbclient, and ipykernel in addition to the plotting dependencies).

The N5 combined-trajectory PDF/PNG also has the warm-background reference
layout. Reproduce just that figure with
`python code/plot_trajectory_combo.py --only chain_N5 --style cell62`.
The 3×3 counterpart uses the same layout; change `--only` to `lattice_3x3`
or run section 3c in `figs_L3x3.ipynb`.
It retains its original AM-trained paths but now uses the requested **AM²
equilibrium background**, colorbar [0,1], and AM²=0.9 contour. The background
change does not turn the paths into AM2-training results. Blue/magenta
encode smooth/stiff and dashed/solid encode T=20/1000. The default `study`
style remains available; use `--outdir` to render comparisons separately.

## The four figure_plan.md items

1. **`grad_vs_T_{chain_N1,3,5,7,9,lattice_2x2,lattice_3x3,lattice_4x4}.pdf`** -- gradient
   components vs evolution time T, smooth vs stiff schedule against the exact
   reference, plus the relative-error power-law fit (the
   `H_detecting/figures/bd_grad_vs_T.pdf` layout). Chains reuse the existing
   `code/bias_scan.py` output directly. Lattices needed new data: `code/
   bias_scan_2d.py` is the lattice counterpart of `bias_scan.py`, run for 2x2
   and 3x3. The 4x4 data are produced by the validated sparse, resumable
   `../../code/bias_scan_2d_sparse.py`; it omits the unused dense spectrum
   diagnostic and uses AM2 because AM is symmetry-forbidden on the even
   lattice. Regenerate
   with `python code/plot_grad_vs_T.py` (re-runs the lattice scans only if
   their output is missing from `data/bias/`).

   Every underlying scan already evaluates the gradient on a dense short-T
   grid *and* sparse windows out to T=800-3200 (`bias_scan.py`'s own
   docstring: the windows exist to resolve the bias amplitude at large T).
   The first version of `plot_grad_vs_T.py` only plotted the dense part, so
   every panel looked cut off around T=60-120 regardless of system size --
   fixed by plotting the full stored grid (log-x) and fitting the decay
   exponent over T>=200, matching the report's own settled methodology
   (`S4_bias.tex`) rather than the short-T range that is known to be a
   fit-range artifact.

2. **`trajectory_combo_{chain_N5,chain_N9,lattice_2x2,lattice_3x3,lattice_4x4}.pdf`** --
   one figure per geometry with four training paths over the phase diagram:
   {smooth, stiff} x {T=20 (short), T=1000 (long)}. The study never trained
   under the stiff schedule, so all four lines were run specifically for this
   figure, at **lr=0.1** rather than the report's lr=0.01 (`code/
   tasks_hilr_train2.txt`; beta=0.1, target=1, start (Omega,Delta)=(1.0,2.6),
   Ttot=1e5 otherwise). At lr=0.01, T=1000 gives only 100 epochs, and the
   displacement per run is ~`epochs*lr*|grad| ~ 100*0.01*0.1 = 0.1` -- the
   "long T" trajectories were single invisible dots. lr=0.1 was checked first
   in isolation (`data/train_lrtest/`): the N=9/T=1000/smooth run then moves
   all the way to y=0.98 with no instability, and the T=20 runs stay bounded
   (no NaNs, same range as at lr=0.01) even though -- expected, not a bug --
   they still don't converge there; the report's point that the short-T
   gradient is too corrupted to descend on holds regardless of step size.

   The report's own N_epoch = Ttot/T is also not used as-is: that gives only
   100 epochs at T=1000 and 5000 at T=20, neither chosen for what actually
   happens during training at lr=0.1 (it is bookkeeping for items 3/4's fixed
   *physical-time* budget, not this figure). Checking the epoch-by-epoch y
   trace showed T=1000 already flat (slope ~2e-4/epoch) by epoch ~70-100, and
   T=20 already at its long-run oscillating character by epoch ~100 -- so
   N_epoch is clipped to [200, 2000] instead of used raw: a floor so a long T
   is never cut short before it settles, a cap so a short T that has already
   saturated into its failure mode doesn't keep running for free. Since
   Ttot/T is 100 at T=1000 and 5000 at T=20 at every one of the four
   geometries, the floor raises every T=1000 run to 200 epochs and the cap
   lowers every T=20 run to 2000 epochs, uniformly (`tasks_hilr_train2.txt`).
   The trajectories barely changed shape from the first (unclipped) lr=0.1
   batch -- confirming both epoch counts were already past their
   post-transient plateau, not still evolving.
   The 4x4 counterpart uses the same start, learning rate, schedules, T values,
   and epoch clipping, but AM2 as its objective. Regenerate with
   `python code/plot_trajectory_combo.py`.

3. **`{geometry}/physical_time_{AM,AM2}_base.pdf`** -- order parameter vs
   accumulated evolution time (`N_epoch * T`), one curve per T. Reuses
   `code/make_figs.py train` unchanged against the copied smooth-schedule
   training data.

4. **`{geometry}/time_budget_{AM,AM2}_base.pdf`** and
   **`{geometry}/time_budget_heat_{AM,AM2}_base.pdf`** -- order parameter vs
   T at fixed total time budgets, and the full (T, N_epoch) grid. Same source
   as item 3.

Items 3 and 4 cover chains N=1,3,5,7,9 and lattices 2x2, 3x3. Ordinary-AE AM2
runs for 4x4 are in progress on Raven and will produce the same filenames.

## 4x4 status

Preliminary checkpoint figures are now available in
`preliminary/20260914-162748/4x4_preliminary.pdf` (eight pages), with individual
PDF/PNG panels and hashed input snapshots alongside it. They explicitly label
partial AE trajectories and the existing 0.1-spaced phase background; these
are not the final high-resolution figures. Regenerate a fresh timestamped
snapshot using `python code/plot_4x4_preliminary.py` after downloading the
latest corrected checkpoints from both clusters. The earlier
`preliminary/20260914-162628` directory is the initial rendering before minor
label/legend readability adjustments.

Numerical audit: the initial long-T AE runs with Tsit5 at rtol=atol=1e-6
are unreliable. At T=1000, the measured free-state norm squared was 3.361,
and normalizing that state did not recover the accurate result. The corrected
runner supports Dopri8 with tighter tolerances, normalization before the nudge
and derivative expectations, and rejection of excessive raw norm drift.
Corrected runs must start in a fresh output directory from their original
parameters; an old raw checkpoint cannot be resumed with changed settings.
The packager now rejects AE data without the norm diagnostics. See the relay
notes for the convergence test and current job state.

The original `AFM_searching/data` tree did contain three sparse 4x4 results;
they had accidentally been left out when `paper_figs/data` was assembled. They
are now included here:

- `data/phase/phase_lattice_4x4.npz` -- the 60 x 70 sparse phase diagram on a
  0.1-spaced grid.
- `data/train/lattice_4x4_AM2_ED_smooth_TED_base.npz` -- 10,000 epochs of
  sparse exact-QEP training, from `<AM^2>=0.1521` to `0.99944`.
- `data/train/lattice_4x4_AM_ED_smooth_TED_base.npz` -- the corresponding
  `<AM>` diagnostic. On this even lattice it is not a physically valid success
  curve: the near-degenerate checkerboard doublet lets the sparse eigensolver
  select a symmetry-broken vector. Use the AM2 result for scientific claims.

The packaged plots are under `figs/lattice_4x4/`. Regenerate the copies and
plots with `python code/package_4x4_exact.py`. The underlying calculations are
produced by `../code/phase_diagram.py` and `../code/train_ed_sparse.py`.

The original small-system validation and loose-solver timings do not establish
accuracy or runtime for corrected 4x4 production. The converged Dopri8
rtol=1e-10/atol=1e-12 reference suggests about 12 hours per standard run on an
unshared Raven A100, with parameter-dependent variation. Corrected runs use
complex128, normalized expectations, and a raw norm-squared drift guard of
1e-5. The 26 standard paths get individual A100s; the two short high-lr paths
share an A100, while the two long high-lr paths now use separate Zeropoint
V100s with seven-day allocations, following a successful cross-cluster
numerical validation gate. They resumed their checked epoch-two checkpoints
after the original shared-GPU job was stopped with user approval. Existing
epoch counts and initial parameters are unchanged. Only ordinary AE is used.

The base T grid is `10,20,50,100,200,500,1000,2000`. T=5000 was dropped at the
user's request to limit the computational cost for this larger system. The established
alternative starts use `T=20,100,1000`. A 0.02-spaced 300x350 phase scan and
the gradient inputs are also running on Raven. See
`../GPT_relay-0914.md` for job IDs, validation, timing, and packing details.

After downloading the completed results, run `python code/package_4x4.py`.
Download corrected training from both clusters into `AFM_searching/data/train_corrected/`;
the Zeropoint project is `/ptmp/qinwang/AFM_searching_checked_0914`.
The packager uses this directory for AE inputs, preserves any superseded
canonical AE files in a content-addressed archive, and promotes only validated
completed results into the usual training/figure directories.
It validates that all 38 required training/bias files are complete and separately
checks the phase grid, archives the old coarse
phase grid, installs the new grid, verifies every paper-data copy by SHA-256,
and renders the standard main-study and paper figures. It deliberately refuses
to make partial plots.

## Regenerating everything

```bash
cd code
python plot_grad_vs_T.py         # item 1 (chains instant; lattices ~10 min if data/bias/ is empty)
python plot_trajectory_combo.py  # item 2 (instant; re-run tasks_hilr_train.txt first if the *_hilr.npz files are missing)
python make_figs.py train        # items 3 and 4
python package_4x4.py             # validate/package all completed 4x4 data and figures
python package_4x4_exact.py       # ED-only fallback while finite-time jobs run
```
