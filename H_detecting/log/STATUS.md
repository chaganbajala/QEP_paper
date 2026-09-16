# H_detecting — project status & handoff

Reproducible study of **Hamiltonian detection with Quantum Equilibrium Propagation (QEP)**:
unlike the companion `../single_qubit` study (linear cost → recovers only the *ray*
`(A,B) ∝ (h_z,h_x)`), here a **probing field** breaks the amplitude degeneracy so both the
amplitude *and* direction of the target field `h=(h_z,h_x)` are recovered.

Self-contained: depends only on `code/qep.py` (not the parent `QEP/` package).

## TL;DR for a fresh session
- Core code is `code/qep.py` (read its module docstring first — it is the source of truth).
- Two experiment drivers, two reports:
  - **Detection study**: `code/run_experiments.py` → `data/*.npz` → `code/draw_figs.py`
    → `figures/*` → `report/report.tex` → `report/report.pdf`.
  - **Training study** (intro.md §Training): `code/train_experiments.py` → `data/train_*.npz`
    → `code/draw_train_figs.py` → `figures/train_*` → `report/training.tex` (→ training.pdf).
- `H_detecting.ipynb` (top level) is the original exploratory/debug notebook; still runnable.

## Physics / model (not obvious from code)
- Model `H_sys(A,B) = -A σz - B σx`; adiabatic schedule from fixed start `(A0,B0)=(1,0)` to the
  trainable `(A,B)` along the **smooth** ramp `½(1-cos πτ)` (zero boundary derivative → cancels the
  O(1) adiabatic boundary bias; lesson inherited from single_qubit).
- Probe: add `-f_x σx`, measure `y=<σz>`. Exact single-qubit response
  `y(f_x) = A/√(A²+(B+f_x)²)` (carries a **square root**; peak at `f_x=-B`, width ~A). The shape
  encodes the *absolute* `(A,B)` — this is what lifts the amplitude degeneracy.
- Quadratic cost `C = ½ Σ_i (y_i - y^τ_i)²`, error signal `ε_i = y_i - y^τ_i`, QEP nudge along the
  **measured** operator `σz`: `H_β = H_sys - f_x σx + β ε σz`.
- QEP gradient `g = (<∂_θH>_nudge - <∂_θH>_free)/β`, `∂_A H=-σz`, `∂_B H=-σx`.

## Key conventions / decisions
- **Average vs sum gradient.** `qep_grad_batch(..., reduce=)` and `train_scan(..., reduce=)`:
  - `reduce="sum"` is the **default** (keeps `run_experiments.py` reproducible; it compares the
    summed QEP gradient against `num_grad_batch` of the summed cost).
  - `reduce="mean"` is used throughout the **training study** (`train_experiments.py`) — the update
    is the AVERAGE gradient over the dataset (intro.md §Training). Reported cost is the AVERAGE cost
    `C = mean_i ½(y_i-y^τ_i)²`. With mean grad the effective step is 1/K smaller, so `LR=2.0`
    (vs `0.3` for the summed-grad detection study).
- **Degeneracy edge case.** At `f_x=-B` with `A=0` the whole Hamiltonian vanishes → degenerate GS,
  `<σz>` undefined. Probe grids avoid this point (e.g. the `H_0=-σx` test uses a grid excluding -1).
- **Solver tol** `rtol=atol=1e-9` (QEP divides an O(β) signal by β → amplifies solver error by 1/β).
- `NS=2` (endpoint-only output; the adaptive solver handles accuracy), `BETA=0.02`, target
  `h=(0.8,1.5)` (|h|≈1.70 ≠ 1 so amplitude recovery is non-trivial), probes `f_x∈linspace(-2,2,9)`.

## Key results
- QEP gradient matches numerical gradient: cosine **>0.9999** (exact and adiabatic).
- **Amplitude identifiability**: K=1 probe fixes only direction (`|p-h|≈0.34`); K≥2 pins amplitude,
  error → ~1e-4 by K=9 (`n_probes`).
- Mean-gradient training converges by **~epoch 100** (epochs 150/200/300 identical) → `N=150` used.
- Exact equilibria reach machine-precision cost; adiabatic reach `|p-h|~1e-3`, improving
  monotonically with `T`; smooth schedule beats stiff by orders of magnitude.
- Adiabatic prep infidelity `1-F` decreases ~`1/T²` with the smooth ramp (`train_prep_fid_vs_T`).
  **[CORRECTED 2026-09-09 -- the true exponent is 4; the `1/T²` reading is the 1e-9 solver floor.
  See "Boundary cancellation on the STATE" at the end of this file.]**

## Performance notes (speed)
- Cost is dominated by adiabatic `train_scan`, scaling ~linearly in `T` (T=20: ~28 s for 16 inits ×
  N=300; T=80: ~135 s). `train_experiments.exp_perf_vs_T` is the bottleneck (~5 min).
- Speedup applied: `N=300 → 150` (verified output-preserving: params identical past epoch ~100).
- DO NOT pipe long runs through `tail` (it buffers all output to EOF — looks frozen). Use
  `python -u driver.py > log 2>&1` and poll the log.

## File map
```
code/qep.py                core: ramp, htau, gs_of, y_analytic, equilibrium, measure,
                           response_curve, make_dataset, qep_grad_single, qep_grad_persample,
                           qep_grad_batch(reduce=), cost, num_grad_batch, train_scan(reduce=),
                           param_error
code/run_experiments.py    detection study (reduce='sum'); `python run_experiments.py [test|full]`
code/draw_figs.py          detection figures -> figures/
code/train_experiments.py  training study (reduce='mean'); `python train_experiments.py [test|full|<names>]`
code/draw_train_figs.py    training figures -> figures/train_*
data/*.npz                 detection outputs
data/train_*.npz           training outputs
figures/*                  rendered png+pdf
report/report.tex,.pdf     detection report
report/training.tex,.pdf   training report
log/STATUS.md              THIS FILE
H_detecting.ipynb          original exploratory/debug notebook (top level)
intro.md                   project spec (target, protocol, test example, training section)
```

## How to run
- Env: `/Users/qwang/miniforge3/envs/jaxlab/bin/python` (jax 0.9.1, dynamiqs 0.3.3, optax 0.2.6,
  `jax_enable_x64=True`). From `code/` with `PYTHONPATH=.`.
- Detection: `PYTHONPATH=. python run_experiments.py full && python draw_figs.py`
- Training: `PYTHONPATH=. python train_experiments.py full && python draw_train_figs.py`
  (run subsets: `python train_experiments.py multi_init_paths response_evolution`)
- Reports: `/Users/qwang/miniforge3/envs/tex/bin/tectonic report/report.tex` (and `training.tex`).

## STATUS: COMPLETE (2026-06-21)

All training experiments ran; `data/train_*.npz` are final (perf_vs_T uses physical-basin A>0 inits;
mean avg-cost 3.3e-3 -> 3.9e-9 over T=5..80, clean monotonic decrease, mean tracks best).

## Status of each deliverable
- code/qep.py            DONE (reduce='mean', qep_grad_persample added; verified)
- code/train_experiments.py DONE (6 exps; vmapped avg_cost_traj, 15x faster, verified identical)
- code/draw_train_figs.py   DONE (mathtext \tfrac/\overline avoided; per-fig try/except)
- data/train_*.npz       DONE (all 6, perf_vs_T uses A>0 inits)
- figures/train_*        DONE (all re-rendered from final data)
- report/training.tex,.pdf  DONE (compiled, 1.7 MB)
- report/report.tex,.pdf DONE (detection study, earlier)
- README.md              DONE (detailed model + all objects/functions)
- draw_all_figs.ipynb    DONE (21 cells; executes end-to-end, 0 errors)
- log/STATUS.md          THIS FILE

## Bugfix (2026-06-21): grad_cos_vs_T cosine
`exp_grad_cos_vs_T` used `np.linalg.norm(a, -1)` — the 2nd positional arg is `ord`, NOT `axis`, so
the per-point vector norms collapsed to one scalar (matrix norm, ord=-1) and the median cosine came
out ~3e-4 (flat near 0) instead of the true rising trend. The raw saved gradients (`g_exact`,
`gq_T`) were correct; only the reduction was wrong. Fixed to `axis=-1`; regenerated data + figure
+ recompiled report.pdf. Correct median_cos: 0.84 (T=5) -> 0.97 (T=20) -> 0.99 (T=50).
(Only that one 2-D norm call had the bug; all other linalg.norm calls act on 1-D vectors.)

## descent_field_vs_T figure (2026-06-21)
Added exp_descent_field_vs_T + fig_descent_field_vs_T (detection study): descent-direction field
(unit -grad) over (A,B) grid 7x7, overlaying exact-QEP (black) and adiabatic-QEP (red) at
T=5,10,20,40, with median angle (theta) between them in titles. Like single_qubit/descent_field_vs_T
but reference is exact-QEP (not numerical), and marks the target POINT (star) since the cost has a
unique minimum (not a ray). Median angle 6.1->1.9->1.0->0.3 deg as T grows. In report.tex
(fig:descentfield). Notebook now 24 cells.

## grad_cos_vs_T now references exact-QEP (consistent with grad_cos_scatter)
grad_cos_vs_T reference changed from the numerical exact gradient to the EXACT-QEP gradient, so it is
literally the vs-T summary (median/IQR/mean/variance) of the grad_cos_scatter data: cos(adiabatic-QEP,
exact-QEP) at 200 independent (A,B) points, NO training. Same +0.3*sign sampling convention as the
scatter. Median rises 0.89 (T=5) -> 0.98 (T=20) -> 0.997 (T=50). (Numbers ~ identical to the old
numerical-reference version since exact-QEP ~ true gradient up to O(beta).)

## grad_cos_vs_T: sample size + error bars (2026-06-21)
Originally 30 random points -> noisy median (apparent dip at T=32). Raised to 200 points
(`NPTS` env var, default 200; 27s locally -> no cluster needed). Now saves the full per-point
cosine array `cos_all` (nT, n_pts). Figure is 2-panel: (left) median + IQR band + mean+-std;
(right) mean & variance vs T (twin axes). Findings: MEDIAN is monotonic 0.84->0.996 (T=32 dip was
sampling noise); MEAN ~0.6 with large, ~T-independent variance because a minority of points sit in
flat-cost/saturated-response regions where the EXACT gradient direction is ill-defined (cosine
random in [-1,1]). Use median/IQR for the genuine T-trend. Smooth schedule IS used throughout
(qep.ramp default 'smooth', equilibrium schedule='smooth', nudge ramped) -> single_qubit fix.

## grad scatter figures + smooth-schedule verification (2026-06-21)
Added `exp_grad_scatter` (QEP vs numerical gradient, exact|adiabatic panels) and
`exp_grad_cos_scatter` (adiabatic-QEP vs exact-QEP gradient at T=5,10,20,40), both evaluated at 40
INDEPENDENT random (A,B) points via qep_grad_batch (NO training). Result: scatter TIGHTENS with T,
median cos 0.76->0.94->0.99->0.995 -> dispersion shrinks with T (intuitive; matches single_qubit).
Verified smooth schedule is genuinely applied: smooth-vs-linear rel-err vs T at well-behaved points
gave smooth 0.21->0.0052 (decays) vs linear 0.22->0.12 (flat). NO coding bug. The apparent
"dispersion grows with T" is the ill-conditioned tail (near-origin/saturated points, |grad|~1/|p|,
small gap, LZS susceptibility) whose magnitude grows with T; it inflates mean/variance but not the
median. Figures in report.tex (fig:gradscatter, fig:gradcosscatter); notebook now 23 cells.

## response_evolution: "before" oscillation explained (2026-06-21)
The oscillating "before" curve was NOT a bug. Exact-GS response = analytic A/sqrt(A^2+(B+fx)^2) to
1e-16. The wiggle is non-adiabatic STATE PREP at T=20 for the deliberately-bad init (0.2,2.1): small
A=0.2 means for fx~=-B the transverse field B(t)+fx crosses zero late in the ramp while A(t)~0.2 ->
min gap ~0.45 -> Landau-Zener (T*gap^2~4, borderline) -> prep fidelity F=0.75 at fx=-2 (vs >0.9999 at
well-conditioned gap>1.7 points). F(fx=-1.8): 0.77->0.99->0.9999 for T=20->40->80. This IS the
small-gap tail that lowers the MEAN (not median) cosine in grad_cos_vs_T. Figure redone as 2 panels:
left = exact GS response (smooth, always), right = adiabatic prep T=20 (before wiggles). exp now saves
curves_exact & curves_adiab. training.tex fig:resp updated, training.pdf recompiled.

## perf_vs_T note
exp_perf_vs_T samples inits with A>0 (target h_z=0.8>0). Random inits including A<0 land in the
wrong sign sector and converge too slowly in N=150, inflating the MEAN with optimisation (not T)
error; best/median stay clean. A>0 basin makes the average reflect the adiabatic-time floor.

## Possible extensions
- noisy/sampled readout of y, optimised probe placement, multi-qubit targets.

## Nudge switch-on audit (2026-09-09) -> report-2026-09-09/
Reported defect: "QEP in this project is not done with smooth switch on in the nudging phase; fix
using QEP_multiple_shot/Adaptive_bn/qep_shot/afm.py::QM". **AUDIT RESULT: the defect is NOT in this
project.** qep.py:88 already reads `beta*err*ramp(t/T, schedule)`, and `qep.ramp(x,'smooth')` is
bit-for-bit identical to afm.py's `ascend_func` (max diff exactly 0 over 1001 points); QM.total_H
adds it the same way. Only two copies of the construction exist here (code/qep.py + the notebook)
and both carry the ramp. The requested fix is a NO-OP; no published number changed.

THE UNRAMPED NUDGE IS REAL BUT LIVES IN ../single_qubit: `data/make_param_paths.py:18` and
`data/make_assets.py:63` add `(beta*err)*Htau` at full strength from t=0, inconsistent with
`single_qubit/code/qep.py` (which ramps it and says so in a comment). NOT fixed -- out of scope
here, but worth fixing there and re-checking any figure regenerated from those two helpers.

New in qep.py: two OPT-IN switches threaded through every helper, defaults reproduce the study
BIT-FOR-BIT (verified: make_dataset/response_curve/qep_grad_batch(exact+adiabatic,T=5,20)/cost/
train_scan all bitwise-equal vs the pre-patch module, max|diff|=0):
  - `nudge_ramp=True`  : shipped behaviour. False = the single_qubit bug, kept as a CONTROL.
  - `probe_ramp=False` : shipped behaviour (probe constant, psi0 includes it). True = the reference's
                         QEP_grad_PF convention (probe folded into B and ramped with it).
Both are no-ops for method='exact'. Also: num_grad_batch now does jnp.asarray(p) so a plain numpy
`p` no longer raises AttributeError on `p.at[i]`.

Findings (code/run_nudge_audit.py -> data/audit_*.npz -> code/draw_audit_figs.py -> figures/audit_*):
- THE RAMP IS ESSENTIAL. Unramped nudge DESTROYS convergence in T: median cos(adiab-QEP, exact-QEP)
  over 200 random points STALLS at ~0.82 with no trend (5->80), and median rel-err GROWS 1.28->1.92,
  vs ramped 0.89->0.9989 and 0.90->0.11. Mechanism: psi0 is the GS of the UNNUDGED start H, so a
  full-strength nudge at t=0 leaves an O(beta*eps/gap) excited admixture that does NOT decay with T;
  dividing it by beta gives an O(1) gradient error (the "O(1) boundary bias"), and the winding
  relative phase makes it grow rather than settle.
- PROBE CONVENTION IS A REAL TRADE-OFF, not an improvement (this is the ONLY genuine difference from
  the reference). Constant probe: transverse coeff g(t)=B*ramp+fx runs fx -> B+fx, so it crosses ZERO
  mid-ramp for every probe with sign(fx) != sign(B+fx) -> gap collapses to 2|A(x*)|. At (A,B)=(0.2,2.1)
  exactly the 4 probes fx in {-2,-1.5,-1,-0.5} cross (min gap 1.51->0.445, worse the later the
  crossing) and exactly those 4 have elevated prep infidelity. Ramped probe: g(t)=(B+fx)*ramp is
  monotone from 0, never crosses -> probe-avg 1-F at (0.2,2.1) is 34x better at T=12, 72x at T=50.
  BUT for fx>0 constant is BETTER (min gap 4.47 vs 1.96 at fx=+2), and in bulk over 200 points
  constant WINS: median cos 0.9989 vs 0.9920 at T=80, median rel-err 0.114 vs 0.243. KEEP THE
  CURRENT (constant-probe) DEFAULT.
- TRAINING IS FORGIVING; all 3 variants converge, median final |p-h| within 2x at every T
  (T=80: cur 3.45e-4, bug 4.27e-4, ref 3.29e-4) -> the published training conclusions are robust to
  all three protocol choices. Variants separate only on the ONE ill-conditioned init (0.2,2.1)
  (5 of 6 inits behave identically): it converges from T=10 (ref), T=20 (cur), T=40 (bug).
- WHY training survives cos~0.82: the 200-point statistic samples the whole (A,B) plane incl.
  flat-cost/saturated regions where the exact gradient direction is ill-defined; training doesn't
  linger there. On the 13 descent-phase iterates with |g_exact| > 5% of max, median cos at T=20 is
  0.99994 (cur) and still 0.99394 (bug). Do NOT measure cosine near the minimum -- |g|->0 there and
  the cosine is pure noise (a run from (0.3,0.5) gives median cos ~ -0.08 over the full path).

New files: code/run_nudge_audit.py, code/draw_audit_figs.py, data/audit_{ramp_identity,grad_vs_T,
prep_fid_vs_T,gap_profile,training_vs_T}.npz, figures/audit_*.{png,pdf},
report-2026-09-09/report.{tex,pdf} (9 pp). Runtime ~5 min total.

## Bandwidth / boundary-curvature study (2026-09-09) -> report-2026-09-09 sec. 7
Follow-up to QEP_GPT-12.pdf (the O(1) gradient bias from a stiff turning point). The note's
criterion is on the FIRST derivative: the bias ~ B_n0(0) = <phi_n|d_tau H|phi_0>/(E_0-E_n)^2 at
tau=0, and d_tau H = lambda'(tau) d_lambda H, so any schedule with lambda'(0)=lambda'(1)=0 kills it.
QUESTION ANSWERED HERE: what sets the RESIDUAL once that is done? Answer: the boundary SECOND
derivative, via dev ~ 0.25*lambda/T.

Theory: the exact series solution (Eq. 36-37 of the note) builds higher terms by the recursion
F^k = d_tau(F^{k-1}/(E_n-E_m)) -- each order DIFFERENTIATES the previous one -- so with lambda'(0)=0
the leading survivor picks up d_tau^2 H ~ lambda''(0), one order higher in 1/T.

Schedule family: the notebook's `parametrized_ascend(x, lam)` (H_detecting.ipynb cells 25-26), now
in qep.py as `parametrized_ascend`/`d_parametrized_ascend`/`sched_param(lam)`, plus
`SMOOTH_LAM = pi^2/2`. ALL 5 claimed properties verified numerically: f(0)=0, f(1)=1,
f'(0)=f'(1)=0 (<1e-15), monotone, f''(0)=+lam and f''(1)=-lam EXACTLY, f->x as lam->inf
(max|f-x|=1e-3 at lam=1e3). Numerically stable to lam=1e-3 despite the singular coth(lam/2) and
2/lam pieces. `ramp()` now accepts a CALLABLE schedule (applied to the (A,B) ramp AND the nudge,
which is what the boundary analysis requires); 'smooth'/'linear' unchanged -- published
grad_cos_vs_T.npz re-ran BITWISE IDENTICAL across all 7 arrays after this change.

Setup: gradient at fixed (Delta,Omega)==(A,B) [A = sigma^z coeff, B = sigma^x coeff], ZERO probe
field fx=0, target h=(0.8,1.5), beta=0.02; reference = exact-equilibrium QEP grad at the same point
and same beta (so the O(beta) QEP error cancels and only the adiabatic error is measured).

CRITICAL METHOD POINT: the deviation OSCILLATES in T (period ~1.15 here) with amplitude set by lam
-- a single-T reading is meaningless and can be off by 100x. Every number below is the RMS over a
dense 24-point T window of width 6 (the envelope). See figures/bw_oscillation.

Findings (code/run_bandwidth.py -> data/bw_*.npz -> code/draw_bandwidth_figs.py -> figures/bw_*):
- lam^1 LAW: local slope dlog(dev)/dlog(lam) peaks at 0.928/0.972/0.989/0.995 for T=10/20/40/80
  (-> 1), and the slope>0.95 window WIDENS with T ([5.1,10.4] at T=20 -> [1.6,33.8] at T=80).
  Fit: dev = C*lam/T with C = 0.2483/0.2524/0.2590 at T=20/40/80  =>  dev ~ 0.25 lam/T.
- SATURATION: at lam=3000 the deviation equals the LINEAR schedule's to 0.05/0.05/0.24/0.21% and is
  T-independent (~0.53). Once the 1/lam boundary layer is unresolvable the O(1) bias is fully back.
- MASTER CURVE: dev = Phi(lam/T). Spread across T=10..80 at matched lam/T is only
  1.03x/1.16x/1.15x/1.10x at lam/T = 0.25/0.5/1/2. Crossover at lam ~ 2T.
- f''(0) DOMINATES, but is NOT the sole parameter at finite T. The cosine ramp and this family's
  lam=pi^2/2 member agree to 0.5/0.1/0.2/0.3% at T=10/20/40/80 -- but that alone proves NOTHING,
  because the two functions differ by only 6.8e-4 pointwise (I initially over-claimed here; the
  agreement is continuity, not evidence). PROPER TEST (data/bw_shape_test.npz): perturb the INTERIOR
  at fixed boundary data via g_a(x) = f_{pi^2/2}(x) + a*x^3*(1-x)^3 -- x^3(1-x)^3 has value, 1st AND
  2nd derivative zero at both ends, so g_a(0),g_a(1),g_a'(0),g_a'(1),g_a''(0),g_a''(1) are ALL fixed
  (g''(0)=4.93480 exactly) for every a; a=8 displaces mid-ramp by 0.125 and is the largest monotone
  amplitude. Result: dev changes by only 1.20x (T=20) and 1.10x (T=40) over that whole range, and
  the sensitivity SHRINKS with T (interior enters at higher order in 1/T). For comparison a BOUNDARY
  change of the same magnitude (smooth->linear, max|f-x|=0.105 vs 0.125) multiplies dev by 17
  (3.13e-2 -> 5.34e-1 at T=40). Per unit shape change the boundary is ~170x more consequential.
  Corollary: the project's standard ramp IS the lam=pi^2/2 member, so its published adiabatic
  gradient error is ~1.2/T.
- SMALL-lam FLOOR: below lam~1 the curves flatten onto a lam-INDEPENDENT floor that is NOT part of
  the lam^1 law (the lam-linear term has been tuned away; a steeper higher-order term dominates).
  Measured T^-1.71 at lam=0.3, between the 1/T of the curvature term and the 1/T^2 the note's
  hierarchy gives when the first TWO derivatives are suppressed. The lam/T collapse degrades there
  (1.95x at lam/T=0.1, 3.8x at 0.05). To push past this floor one must flatten MORE derivatives at
  the boundary, not merely lower the curvature (the N-th derivative hierarchy of Eq. 37).
- GENERIC: same curve shape at (Delta,Omega) = (1.2,0.4), (0.5,1.0), (1.0,1.0), each saturating onto
  its OWN linear-schedule value (matched to 0.2%).

PERF: run_bandwidth.py takes ~15 min. vmap over lam WORKS and is ~20x faster than a Python loop
(verified identical to 1e-13); T cannot be vmapped (sesolve tlist depends on it) so T is looped.
Large lam is stiff -> the solver slows down, so the lam=3000 end dominates the cost.

New files: code/run_bandwidth.py, code/draw_bandwidth_figs.py, data/bw_{schedule_shapes,dev_vs_lam,
dev_vs_lam_points,shape_test}.npz, figures/bw_*.{png,pdf}. report-2026-09-09/report.{tex,pdf} now 14 pp
(sec. 7 = this study).

## Boundary cancellation on the STATE (2026-09-09) -> bd_* / run_boundary.py
The state-level precursor to the bandwidth study, requested as figure 1 of the O(1)-deviation
sequence: show the two [0,1]->[0,1] schedules of ../../networks.py (ascend_func = smooth
1/2(1-cos pi x), ascend_func_st = stiff x; bit-identical to qep.ramp 'smooth'/'linear'), then the
prep-fidelity bias each one leaves.

KEY IDENTITY that makes the note concrete for this project: with
H(tau) = -[A0+(A-A0)f(tau)] sz - [B0+(B-B0)f(tau)] sx  we have
    d_tau H = -f'(tau) [ (A-A0) sz + (B-B0) sx ]
so the note's boundary coefficient B_n0(tau) = <phi_n|d_tau H|phi_0>/(E_0-E_n)^2 is EXACTLY
proportional to the schedule slope f'(tau). Measured at (A,B)=(0.8,1.5): |B_10(0)| = 0.375 (stiff)
vs EXACTLY 0.0000 at both endpoints (smooth). Note the smooth ramp has the LARGER |B_10| in the
bulk (peak 0.48 vs 0.38) -- only the endpoints matter.

RESULTS (fx=0, start (A0,B0)=(1,0) i.e. H(0)=-sz, overlap with the exact GS of -A sz - B sx):
- 1-F ~ T^-2 for stiff (fitted -2.002 at (0.8,1.5), -2.008 at (1.2,0.4))
  1-F ~ T^-4 for smooth (fitted -4.004, -3.991)
  = Campos Venuti-Lidar boundary cancellation = Sec. 4.4 of the note at N=1. Local log-log slope
  locks onto -2 / -4 as 1/T -> 0 (figure panel d).
- BOTH oscillate in T at the period the theory predicts from e^{-iT(gamma_1-gamma_0)},
  2 pi / int_0^1 (E_1-E_0) dtau:  predicted 2.553 vs measured 2.551 at (0.8,1.5);
  2.798 vs 2.796 at (1.2,0.4). Direct confirmation of the dynamical-phase factor.
- THE STATE ERROR DECAYS EITHER WAY (T^-2 is still ->0). The O(1) bias is a GRADIENT effect only,
  because d_theta hits e^{-iT gamma} and pulls out a factor T. Do not conflate the two figures:
  bd_* is the state, bw_* is the gradient.

NUMERICS -- the one real trap. The smooth T^-4 curve reaches 1.4e-9 by T=160, BELOW the study-wide
rtol=atol=1e-9 solver floor: at tol=1e-9 the solver reports 7.6e-8, i.e. 53x too large, and the
apparent exponent flattens. Converged from tol=1e-12 (1e-13 used, 0.5% off the 1e-14 value).
THIS INVALIDATES THE OLD "1-F ~ 1/T^2" READING under Key results above (train_prep_fid_vs_T):
that experiment runs at the 1e-9 default, so past T~30 it is measuring the SOLVER, not the physics.
Verified directly from data/train_prep_fid_vs_T.npz:
  - the TRIVIAL config p=(1.0,0.0) (= the schedule start, so H barely changes and 1-F should be ~0)
    instead reads 9.2e-9 -> 1.1e-7 and RISES as T^(+1.00) for all three fx. Accumulated solver
    error growing linearly in T -- there is no other candidate.
  - every other curve bottoms out at ~1e-7 near T=80-130 and then turns UPWARD, i.e. hits that
    same floor; the fitted slopes (-1.3 to -3.1) are floor-contaminated averages, not exponents.
  - the ONE curve still well above the floor at T=130, p=(0.3,1.0) fx=-1 (1-F = 3.9e-6), fits
    slope -4.13 -- T^-4, agreeing with the bd_* measurement.
So the true smooth-ramp exponent is 4, not 2. train_prep_fid_vs_T is not wrong about the TRAINING
study (the states it feeds are fine at 1e-9) -- it simply cannot resolve the exponent, and its T>=32
points are meaningless. Re-run it with tol=1e-13 before reusing it for any scaling claim.
`qep.equilibrium` therefore grew a `tol=1e-9` keyword -- default unchanged, so every published
number is bit-for-bit identical; it is threaded nowhere else (only equilibrium needs it).

Dense T grid (dT=0.25, T=3..160, 629 points) is required: the oscillation period is ~2.5, so a
coarse or log-spaced grid aliases it. Reductions are an RMS envelope over a one-period boxcar; the
local exponent is a SLIDING LEAST-SQUARES fit over a factor-1.6 window in log T (np.gradient is
useless here -- dT=0.25 against T~100 gives dlogT~2e-3 and the envelope ripple swamps it).

New files: code/run_boundary.py, code/draw_boundary_figs.py,
data/bd_{schedule_shapes,prep_fid_vs_T}.npz, figures/bd_{schedule_shapes,prep_fid_vs_T,
prep_fid_points}.{png,pdf}. README §3.5. Runtime ~20 s total.
Figure palette (stiff #C0392B / smooth #1F6FB2) passes the categorical CVD checks (OKLab dE 20.4
protan, 28.6 normal vision); schedule is also encoded by dash pattern, so it never rests on colour.
Figures 1 and 2 also live in paper_figs.ipynb (top level, 10 cells): SELF-CONTAINED -- it runs the
evolutions itself rather than reading data/bd_*.npz, compute cells separated from plot cells,
executes end-to-end in ~17 s with 0 errors and reproduces the script numbers exactly (-2.002 /
-4.004, period 2.553 vs 2.551 measured). It resolves its own paths, so unlike
draw_all_figs.ipynb (which hardcodes ../data and only works with cwd=code/) it runs from either
H_detecting/ or H_detecting/code/. Not yet in a report .tex.

## State-prep overlap vs T at different lam (2026-09-09) -> report sec. 7.5, figures/bw_prep_vs_T
Independent cross-check of the lam/T gradient law, via the FREE-PHASE prep overlap
<psi(T)|phi_0> (beta=0, fx=0; |phi_0> = exact GS of the FINAL H) at (Delta,Omega)=(1.2,0.4).
The two observables must be linked by ONE amplitude:
    1 - |<psi|phi_0>|^2 ~ |a_exc|^2        and        gradient deviation ~ T*|a_exc|
(the factor T because d_theta on either state in <psi|O|psi> gives an O(T) term, note sec. 4.3).

RESULT: THREE REGIMES, and the two observables agree in all of them.
  regime                       1-F exponent   implied |a_exc|   grad dev (pred / measured)
  stiff (lam >~ T, or linear)     -1.87          ~1/T            O(1)   / flat
  curvature (1 <~ lam << T)       -4.15          ~lam/T^2        lam/T  / T^-0.93
  lam -> 0 (curvature gone)       -5.72          ~1/T^3          1/T^2  / T^-1.71
The genuine 'linear' schedule also fits -1.87, matching lam=1e5 (which is a valid numerical
stand-in for linear: exp(-lam)->0, I_lam->1, f->x).

THIS IDENTIFIES THE SMALL-lam FLOOR of the gradient study: with f''(0)=lam driven to 0 the
curvature term is tuned away and the THIRD derivative takes over -> |a|~1/T^3 -> grad dev ~1/T^2.
The earlier gradient-only measurement could only bracket it as "between 1/T and 1/T^2" (T^-1.71);
it is now pinned.

lam^2 PREFACTOR: 1-F ~ lam^2 holds, but ONLY where BOTH lam values are in the curvature regime
(1 <~ lam << T). Local slope dlog(1-F)/dlog(lam): 1.77 (lam=pi^2/2->10 at T=14.3), 1.93 (same pair
at T=21.2), 1.84 (lam=10->30 at T=31.4). Outside that window the slope drops well below 2 -- NOT a
failure of the law, just one endpoint being in a different regime. Do not fit lam-slopes across
regime boundaries.

*** NUMERICAL FLOOR -- IMPORTANT ***
1-F flattens near 3e-8 and then RISES ~T (at lam=0.3 a naive large-T fit gives exponent +0.98!).
That is the solver's accumulated local error at rtol=atol=1e-9, NOT physics. All exponent fits use
only 1-F > 1e-6. Consequence: small-lam asymptotics cannot be followed far (lam=1 hits the floor by
T~30, before its lam^2 regime develops). The GRADIENT results are unaffected -- those deviations are
1e-3..1e-1, five orders of magnitude above this floor. If the floor ever needs lowering, tighten
the sesolve tolerance in qep.equilibrium (currently hardcoded rtol=atol=1e-9).

New: code/run_bandwidth.py::exp_prep_vs_T_lam ("prep_vs_T"), data/bw_prep_vs_T.npz,
figures/bw_prep_vs_T.{png,pdf}. Needed `import dynamiqs as dq` in run_bandwidth.py. ~5 min.
report-2026-09-09/report.pdf now 16 pp.

## Small-lam limit + optimal lam (2026-09-09) -> report sec. 7.6
OBJECTION RAISED: "if lam is very small the function should be abrupt somewhere between 0 and 1,
which should decrease adiabatic performance." Two parts, DIFFERENT answers.

PART 1 -- NO ABRUPTNESS. As lam->0: B_lam(x) -> lam*x(1-x) -> 0 and I_lam = coth(lam/2)-2/lam ->
lam/6 -> 0, so H_lam -> 0 and
    f_lam(x)  ->  S(x) = 10x^3 - 15x^4 + 6x^5    (the QUINTIC SMOOTHSTEP).
Verified on a 4001-pt grid (data/bw_small_lam_limit.npz): max|f_lam - S| = 8.9e-6 at lam=1e-3,
convergence is O(lam). Peak interior slope rises only pi/2=1.5708 (cosine ramp) -> max S' = 15/8 =
1.875 (+19%) and SATURATES; peak interior |f''| saturates at 5.773. So the interior does steepen --
the intuition is directionally right -- but by a bounded, modest amount. Nothing diverges.

*** THIS EXPLAINS THE SMALL-lam FLOOR ***  S has S'(0)=S'(1)=0 AND S''(0)=S''(1)=0, S'''(0)=60, so
it is the N=2 member of the note's hierarchy -> |a|~1/T^(N+1)=1/T^3 -> 1-F ~ T^-6. That IS the -5.72
infidelity exponent and the T^-1.71 gradient floor measured earlier. The lam->0 end is not a
pathology, it is the next rung of the same ladder.

PART 2 -- BUT SMALL lam IS NOT FREE: THERE IS AN OPTIMUM (data/bw_optimal_lam.npz, lam over 4
decades 0.01..100, T=3..40, both observables, RMS over T windows):
   T  | lam*(1-F)  min 1-F  | lam*(grad)  min dev  | penalty at lam=0.01 (1-F / grad)
   3  |   10.0     9.7e-4   |   31.6      5.70e-1  |  3.86x / 1.89x
   5  |    2.15    1.9e-4   |    4.64     2.91e-1  |  1.16x / 1.44x
  10  |    1.00    2.5e-6   |    0.68     6.05e-2  |  1.23x / 1.05x
  20  |  (0.15)    8.3e-8   |    0.10     1.77e-2  |  1.01x / 1.00x
  40  |  (0.03)    3.8e-8   |    0.046    4.87e-3  |  1.00x / 1.00x
(parenthesised lam* are ON the solver floor ~3e-8 -> NOT physical; the grad column is fine, 1e-2..1e-3)
- At T=3 the objection is exactly right: best is a STIFF lam~10, and lam=0.01 costs 3.9x in prep
  infidelity. Mechanism = the 19% steeper interior + at such short T nothing is adiabatic anyway, so
  the INTERIOR (not the boundary) dominates.
- The optimum falls steeply, lam* ~ T^-2.5 (gradient; ~T^-1.9 for 1-F above the floor), crossing
  below the standard ramp's pi^2/2 around T~5.
- The minimum FLATTENS: penalty for lam=0.01 vs lam* goes 3.9x -> 1.2x -> 1.01x -> 1.00x over
  T=3..40. By T >~ 10 the whole small-lam side is flat (the T^-6 scaling outruns everything).

PRACTICAL: do NOT minimise lam at short T. For T <~ 5 the standard cosine ramp is already near
optimal and stiffer is better. For T >~ 10 (this project's regime) the optimum is well below
pi^2/2, small lam is penalty-free, and the sec.-7 gains stand.

New: run_bandwidth.py::{exp_small_lam_limit ("small_lam_limit"), exp_optimal_lam ("optimal_lam")},
data/bw_{small_lam_limit,optimal_lam}.npz, figures/bw_{small_lam_limit,optimal_lam}.{png,pdf}.
report-2026-09-09/report.pdf now 18 pp. GOTCHA when editing these drivers: a docstring containing
S'''(0) breaks a '''-quoted heredoc patch -- write the block to a file and insert it instead.

## Fig. 1(b) of the note, replicated (2026-09-09) -> bd_grad_vs_T
exp_grad_vs_T in run_boundary.py + fig_grad_vs_T in draw_boundary_figs.py + Figure 3 of
paper_figs.ipynb. Same model, same (A,B)=(0.8,1.5), fx=0 (as specified), start (A0,B0)=(1,0).
y = <sz> of the adiabatic final state; dy/d(A,B) by CENTRAL DIFFERENCE (eps=1e-4), which is the
note's own estimator. Reference is closed-form on the exact GS: y=A/R gives dy/dA = B^2/R^3,
dy/dB = -A B/R^3 -> [0.45796865, -0.24424995]; agrees with a central difference on the exact GS to
3.8e-10 (asserted in both the driver and the notebook).

THREE schedules, not two: the note's Fig. 1 uses lambda=3tau^2-2tau^3 for its smooth case while
this project uses (1-cos pi tau)/2. Both have lambda'(0)=lambda'(1)=0, so carrying both shows the
effect keys on the VANISHING BOUNDARY SLOPE and not on the particular smooth function.

RESULT (relative deviation ||g_ad - g_ex||/||g_ex||, RMS envelope, fitted on T>=20):
  stiff  lambda=tau        T^-0.001   <- FLAT. mean deviation 0.879, i.e. 88% wrong at EVERY T
  smooth (1-cos pi tau)/2  T^-1.001   mean 0.0322
  smooth 3tau^2-2tau^3     T^-1.000   mean 0.0391
So the stiff O(1) bias does not decay at all, while both smooth schedules go as 1/T. This is the
note's headline and it reproduces cleanly. Consistent with run_bandwidth's "dev ~ 0.25 lam/T" (1/T
for any finite lam) saturating at the stiff O(1) value.

Period: measure it on the SIGNED component deviation (G[:,0]-gex[0]), predicted 2.553 vs measured
2.551. Measuring it on |dev| gives 1.275 -- HALF -- because the modulus rectifies the oscillation.
That is not a discrepancy; do not report it as one.

Grid: T=2..160, dT=0.25 (633 points). Value panels window to T<=40 so the oscillation stays legible;
panel (c) uses the full range for the envelope. 4 evolutions per (T, schedule) for the two central
differences -> ~12 s per schedule under jit+vmap.

## Data-saving audit (2026-09-09)
Swept all 5 drivers, 5 draw scripts and both notebooks against data/ and figures/.
CLEAN: 33 npz saved by the drivers, 33 on disk; 37 figure stems written, 37 on disk; exact 1:1 both
ways. No figure lacks its npz, no draw script expects a missing npz (verified at ARRAY-KEY level,
including the dynamic keys rms_T*/path_T*/inf|env|slope_*_p*), and no orphan npz. No draw_*.py
imports qep or computes anything -- they are pure load+plot, so every figure is regenerable.

THE ONE GAP FOUND, NOW FIXED: paper_figs.ipynb computed its data in-notebook (qep/jax, real
Schroedinger evolutions) and saved NO npz -- the only place in the project where figure data was
generated without being persisted. Fixed: each compute cell now calls save_npz() ->
data/paper_fig{1_schedules,2_prep_fid,3_grad}.npz. Deliberately NOT the bd_* names: the driver's
bd_prep_fid_vs_T.npz holds two (A,B) points plus the tolerance sweep and is read by
draw_boundary_figs.fig_prep_fid_points, so writing the notebook's single-point subset over it would
silently break that figure. Separate names, no clobbering; the docstring on save_npz says why.
(draw_all_figs.ipynb saves no npz either, but it is pure load+plot, so nothing is unpersisted.)

STILL OPEN, not touched (belongs to the detection study, currently being edited):
data/grad_cos_vs_T.npz was regenerated Sep 9 14:58 but figures/grad_cos_vs_T.{png,pdf} date from
Jun 21 15:21 -- the FIGURE IS OLDER THAN ITS DATA and is the only such pair in the project. Re-run
`python draw_figs.py` (or the notebook cell) to bring it up to date, and re-check report.tex if the
numbers in the caption moved.

## paper_figs.ipynb restructured: light data + plotting ONLY (2026-09-09)
Per instruction, the notebook no longer computes anything heavy. Division of labour is now:
  Fig 1 (schedules + |B_10(tau)|)  -- generated IN the notebook: analytic curves plus a 2x2 eigh
                                      per tau. No time evolution, so it is "simple data".
  Fig 2 (prep fidelity vs T)       -- loads data/bd_prep_fid_vs_T.npz from run_boundary.py.
  Fig 3 (gradient vs T)            -- loads data/bd_grad_vs_T.npz from run_boundary.py.
Effect: 13 cells/54 s -> 10 cells/7 s, 0 errors, 3 figures. No dynamiqs import in the notebook at
all now; only qep.ramp/A0/B0 and one jax.grad check. This matches draw_all_figs.ipynb's convention
(pure load+plot) and undoes the self-contained design of the earlier version.

Consequences of the restructure:
- The save_npz() helper and data/paper_fig{1_schedules,2_prep_fid,3_grad}.npz are GONE (deleted).
  They were only needed while the notebook generated its own data; the drivers already save
  bd_*.npz, so those three were pure duplicates and would have been orphans under the audit's
  1:1 invariant. The project is back to every npz being read by a draw script.
- local_slope() stays in the notebook (Fig 2 panel d) -- it is post-processing of LOADED arrays,
  exactly as draw_boundary_figs.py does it, not data generation.

## Dropped the 3tau^2-2tau^3 series (2026-09-09)
exp_grad_vs_T and fig_grad_vs_T carried a third schedule (the note's own smooth ramp
lambda=3tau^2-2tau^3, colour #0f9d6e) as evidence that the effect keys on lambda'(0)=lambda'(1)=0
rather than on the particular smooth function. REMOVED on request -- not plotted and not computed.
GSCHEDS is back to {"linear","smooth"}; the palette is back to the validated red/blue pair.
For the record, before removal it measured T^-1.000 with mean rel deviation 0.0391 at T>=20,
against the cosine ramp's T^-1.001 / 0.0322 -- i.e. the two smooth schedules agreed to ~20%, which
was the point. Re-add by putting the callable back in GSCHEDS if that evidence is ever wanted.

## NOT DONE: the note's own Fig. 1 (its own model)
Started and then reverted on instruction. Worth recording what it WOULD take, since the
distinction is easy to lose: bd_grad_vs_T replicates the note's Fig. 1(b) CLAIM in THIS project's
model (H = -A sz - B sx, two parameters, (A,B)=(0.8,1.5)). The note's own Fig. 1 uses a DIFFERENT
Hamiltonian -- Sec. 1.2/Sec. 5:
    H(tau,theta) = 1/2 [ A sigma^z + B sigma^x ],  A = 1-lam+lam cos theta,  B = lam sin theta
i.e. a 1/2, a PLUS sign, a single angle parameter theta, and |field|=1 at BOTH ends (R(0)=R(1)=1,
min gap |cos(theta/2)|, closing at theta=pi). Its exact GS at tau=1 is
-sin(theta/2)|up>+cos(theta/2)|down>, so y_ex = -cos theta and dy_ex/dtheta = sin theta in closed
form. Its schedules are lambda=tau and lambda=3tau^2-2tau^3. It cannot reuse qep.equilibrium; it
needs its own Hamiltonian construction. The note does not state which theta Fig. 1(b) uses.
All of that machinery was removed from run_boundary.py (registry is back to the 3 experiments).

## The ACTUAL QEP gradient (2026-09-09) -> bd_qep_grad_vs_T
exp_grad_vs_T differentiates the OUTPUT y = <sz> by central difference. That is NOT what QEP
extracts. Added exp_qep_grad_vs_T + qep_grad() in run_boundary.py and fig_qep_grad_vs_T in
draw_boundary_figs.py for the real two-phase quantity:
    g_theta = (<d_theta H>_nudge - <d_theta H>_free)/beta ,  d_A H = -sz, d_B H = -sx,
with the nudge on the MEASURED operator (beta*sigma^z, unit error signal) and RAMPED with the same
schedule as (A,B). beta = 0.02.

Onsager (Eq. 6 of the note) makes this equal dy/dtheta as beta -> 0, so the beta->0 limit is the
same closed form (B^2/R^3, -A B/R^3) = [0.45796865, -0.24424995]. At beta=0.02 the exact-equilibrium
QEP value is [0.46177493, -0.24319300], an O(beta) offset of 3.8e-3. THE REFERENCE USED IS THE
EXACT-EQUILIBRIUM QEP GRADIENT AT THE SAME BETA (run_bandwidth's convention) so the plotted
deviation is the adiabatic error alone and not that O(beta) offset.

Result over T=2..160, envelope fitted on T>=20:
  stiff   mean rel deviation 0.625, envelope fit T^-0.095
  smooth  mean rel deviation 0.0219, envelope fit T^-1.079
Same qualitative split as the output-derivative figure (stiff O(1), smooth 1/T), so the note's
conclusion survives the change of estimator. Magnitudes are smaller for stiff (0.63 vs 0.88):
the two phases' O(1) boundary biases largely CANCEL in the difference, and dividing by beta
re-amplifies only what is left.

*** DO NOT READ THE STIFF T^-0.095 AS SLOW CONVERGENCE. IT IS A BEAT. ***
The ramped nudge shifts A(tau) -> A(tau) - beta*lambda(tau), so the free and nudge phases have
DIFFERENT dynamical phases: dgamma_free = 2.461525, dgamma_nudge = 2.448333, difference -1.319e-2.
The envelope therefore beats with period 2 pi/|d(dgamma)| = 476.3 in T, and T<=160 is only 0.34 of
one beat -- the "droop" past T~100 is its onset. VERIFIED by running the stiff case out to T=782:
the envelope collapses to 0.0016 at T = 477 (predicted 476.3, 0.15% agreement) and RECOVERS to
~0.14 by T=700. It sloshes on two timescales (fast 2.55, slow 476); it does not converge. The
figure therefore labels the stiff curve "flat, mean 0.63" and NOT with a fitted exponent; a naive
fit over any window shorter than a beat is meaningless. Beat period scales as 1/(beta*eps), so a
smaller beta pushes it further out.

CONCURRENT-EDIT INCIDENT (worth remembering): draw_boundary_figs.py was rewritten by the user while
this was in progress, dropping fig_qep_grad_vs_T. A follow-up string patch intended for it then
matched uniquely inside fig_grad_vs_T instead and landed there (the assert count==1 passed only
BECAUSE the intended function had vanished). Reverted fig_grad_vs_T and re-added fig_qep_grad_vs_T.
Lesson: when both sides are editing, re-read the file and anchor patches on a function-scoped
slice, not on a string that exists in two near-identical figure functions.

NOT added to paper_figs.ipynb: the notebook was mid-edit (empty placeholder cells 10-11 between
Figure 3 and Figure 4) so writing to it would have clobbered work in progress.

## Exponent-vs-lambda figure, and an O(beta) FLOOR that corrupts Figure 6 (2026-09-09)
New: fig_index_vs_lam in draw_boundary_figs.py -> figures/bd_index_vs_lam.*, and Figure 8 of
paper_figs.ipynb. x = lambda, y = the fitted exponent n in envelope ~ T^n. PURE POST-PROCESSING --
the raw per-lambda gradients grad_lam{j} (633,2) are already saved in bd_qep_grad_vs_T_lam.npz, so
the exponents are re-fitted from them with no new evolutions (same precedent as local_slope).

*** THE FINDING: bd_qep_grad_vs_T_lam's saved slopes are WRONG for lambda <~ 5. ***
That file measures its deviation against g_exact = the ANALYTIC (beta->0) gradient
[0.45796865, -0.24424995]. At finite beta the QEP estimator sits an O(beta) systematic distance
from that -- exactly the file's own floor_lam* = 7.6109e-3 -- and that offset is NOT an adiabatic
error and does not decay with T. For small lambda the adiabatic error falls BELOW the offset, the
deviation curve saturates, and the power-law fit then measures the approach to the floor.
Re-referencing to the EXACT-EQUILIBRIUM QEP gradient at the same beta (bd_qep_grad_vs_T's
g_exact = [0.46177493, -0.24319300]) removes the floor entirely:

   lambda   as-measured   floor-removed   env(T=160)/floor
      0.1      -0.321        -2.010            1.03   <- saturated
      0.5      -0.349        -1.515            1.07   <- saturated
      1        -0.437        -1.254            1.14   <- saturated
      5        -0.935        -1.078            0.92   <- saturated
     20        -1.040        -1.050            4.42
    100        -0.707        -0.708           22.78
   cosine      -0.933        -1.079                   <- saturated
   stiff       -0.095        -0.095           75.60

As-measured, the exponent RISES toward 0 as lambda -> 0, which reads as "a flatter boundary
converges worse" -- backwards. Floor-removed it is monotonic the right way: -2.01 at lambda=0.1,
through -1.08 at the cosine ramp (lambda=pi^2/2), flattening to -0.71 at lambda=100 and collapsing
to -0.09 for stiff. CROSS-CHECK: the floor-removed values track the OUTPUT-DERIVATIVE exponents of
bd_grad_vs_T_lam (-1.940, -1.442, -1.182, -1.000, -0.962, -0.615) to within ~0.08 -- which is the
Onsager consistency check, and is itself evidence the correction is the right one (the output
derivative has no beta and therefore no floor).

RULE OF THUMB: when the reference is the beta->0 analytic gradient, any deviation within ~2x of
the O(beta) offset is meaningless. Either reference the exact-equilibrium QEP gradient at the SAME
beta (preferred -- the convention of run_bandwidth.py and of exp_qep_grad_vs_T), or shrink beta.
The figure marks the floor-limited points with a red X so the artifact cannot be read as physics.

Also: Figure 7's title/text corrected. It uses err=1.0, i.e. eps == 1, which is the QEP gradient of
the LINEAR cost C = y -- there is NO target. That is the bare Onsager primitive (Eq. 6 of the note)
and is the right thing to compare against Figure 3, but it is not the detection-task cost gradient.
A target-based version CANNOT be evaluated at (A,B) = (0.8,1.5), because that point IS the study
target h: y - y_tau = -1.7e-16, the nudge vanishes and the gradient is identically zero (correctly
-- it is the cost minimum). A real target-based figure needs (A,B) != h; suggested (1.2, 0.4),
which gives eps = 0.478. NOT BUILT -- waiting on a choice of evaluation point.

## Repeat of grad_scatter / grad_cos_scatter per schedule (2026-09-09) -> bd_grad_scatter_sched
New: exp_grad_scatter_sched in run_boundary.py -> data/bd_grad_scatter_sched.npz, and three
figures in draw_boundary_figs.py: bd_grad_cos_scatter_sched, bd_grad_scatter_sched,
bd_grad_scatter_summary. Faithful repeat -- same PRNGKey(0), same 40 points, same 9-probe set and
target as run_experiments.py -- but per schedule and over T = 5,10,20,40,80. NOTE the schedule
argument ramps BOTH (A,B) AND the nudge (qep.equilibrium has no separate control), so "stiff" here
means a stiff switch-on of the nudging phase too.

### WHY THE TWO PUBLISHED FIGURES SEEM TO DISAGREE AT THE SAME T=20
Three separate reasons, none of them physics:
1. THE REFERENCE. grad_scatter's x-axis is num_grad_batch with method="adiabatic" -- the finite
   difference of the ADIABATIC cost. That is NOT ground truth: it carries the same O(1) boundary
   error as the QEP estimate. Measured range at T=20: [-9.2,+2.8] (stiff), [-8.6,+8.4] (smooth),
   against the exact-QEP range [-4.55,+1.51]. Both axes are contaminated, so the cloud spreads in
   both directions. grad_cos_scatter's x-axis is the exact-equilibrium QEP gradient: clean and
   T-independent. Control: with EXACT equilibria max|QEP - numerical| = 8.75e-2, a clean diagonal,
   so the QEP estimator itself is fine.
2. THE AXIS CLIP. grad_cos_scatter clips to [-2,2] (its own title says so). That hides 30-40% of
   the components -- and 11% of the EXACT components already exceed 2, so it is discarding
   legitimately large gradients, not just outliers. The tight diagonal is the surviving bulk.
3. SUMMED COST. Both use reduce="sum" over K=9 probes, so nine O(1) per-probe errors add.

### THE REAL FINDING: THE 40 RANDOM POINTS ARE TWO DIFFERENT POPULATIONS
18 of the 40 have A < 0. The ramp starts at (A0,B0) = (1,0), so for those A(tau) = 1+(A-1)f(tau)
must CROSS ZERO, and at the crossing the gap is only 2|B(tau*)|: median min gap 0.815 for A<0
against 1.994 for A>0. That is a Landau-Zener failure of state preparation, NOT a boundary-term
effect, and the QEP estimator amplifies it (divide by beta, and d_theta pulls a factor T out of the
dynamical phase), so its error GROWS with T. The 8 worst points at T=80 are 8/8 in this sector;
max|g_ad| grows 6.8 -> 54.8 over T=5..80. This is the "ill-conditioned tail" noted earlier in this
file, now identified concretely as the A<0 sector. (exp_perf_vs_T already restricts to A>0 inits
for the same reason.)

### WITH THE SECTORS SEPARATED, THE EXPECTATION IS CONFIRMED EXACTLY
median relative deviation ||g_ad - g_ex||/||g_ex||:
        A>0 (22 pts)            A<0 (18 pts)
   T    stiff    smooth         stiff    smooth
   5    0.843    0.758          1.091    1.387
  10    0.445    0.230          1.942    2.375
  20    0.215    0.068          1.690    3.021
  40    0.174    0.025          3.335    3.423
  80    0.164    0.007          6.568   11.910
A>0: STIFF PLATEAUS at ~0.16 from T=20 on (the O(1) boundary bias -- more T does not help), while
SMOOTH decays as ~T^-2 to 0.0066, i.e. 25x better at T=80 and still falling. Median cosine at
T=80: stiff 0.9981, smooth 1.0000. So in the physical sector the smooth scatter does collapse onto
y=x and the stiff one does not -- exactly as predicted.
A<0: both schedules get WORSE with T; the schedule cannot fix a closing gap.

LESSON FOR FUTURE SCATTERS: never sample (A,B) uniformly on [-2,2]^2 for this ramp -- half the
points cross A=0 and their LZ error swamps the boundary effect being measured. Either restrict to
A>0, or split the population as bd_grad_cos_scatter_sched does (marker shape = sector). And do not
clip the axes silently; bd_grad_cos_scatter_sched prints the off-scale count per panel instead.

## Scatter rerun on T = 1,2,5,10,20,50 (2026-09-09)
SCAT_TS changed from (5,10,20,40,80) to (1,2,5,10,20,50); data + all three figures regenerated.
The "N off-scale" annotation was REMOVED from bd_grad_cos_scatter_sched on request (it counted
markers, not points: 80 markers per panel = 40 points x 2 components, tested on |y| only, which
was sufficient because |x| <= 4.549 < 5 always. 100% of them were A<0 points in every panel).

median rel dev, A>0 (22 pts) | A<0 (18 pts)
   T      stiff   smooth   |   stiff   smooth
   1      2.446    3.107   |   0.606    0.669
   2      2.132    3.209   |   0.981    1.334
   5      0.843    0.758   |   1.091    1.387
  10      0.445    0.230   |   1.942    2.375
  20      0.215    0.068   |   1.690    3.021
  50      0.116    0.021   |   4.984    4.799

TWO THINGS THIS T GRID CHANGES, both worth remembering:

1. T=1 and T=2 are OUTSIDE THE ADIABATIC REGIME and the ordering INVERTS there: smooth is WORSE
   than stiff (3.11 vs 2.45 at T=1, 3.21 vs 2.13 at T=2). That is not a contradiction -- the whole
   boundary analysis is an expansion in 1/T and simply does not apply yet. Do not quote T<5 numbers
   as evidence for or against the schedule.

2. *** THE STIFF PLATEAU IS NO LONGER VISIBLE, AND THAT IS AN ARTEFACT OF THE T GRID. ***
   Capped at T=50 the stiff sequence reads 0.843, 0.445, 0.215, 0.116 -- a clean-looking decay.
   The previous grid (5,10,20,40,80) gave 0.843, 0.445, 0.215, 0.174, 0.164, i.e. a PLATEAU near
   0.16. T=50 lands in a dip of the O(1) oscillating bias. The stiff curve is not converging, it
   is oscillating, and a 6-point grid with one point per decade cannot tell those apart. If the
   stiff-vs-smooth conclusion has to be read off THIS figure, add T=40 and T=80 back, or reduce to
   an RMS envelope over a dense T window the way exp_grad_vs_T does. The smooth curve is
   unaffected (monotone T^-2 from T=5 on, 0.021 at T=50, median cos 0.9999).

max|g_ad| over the 40 points: stiff 2.6 -> 31.5, smooth 3.1 -> 45.3 across T=1..50; the 8 worst
points at T=50 are still 8/8 in the A<0 sector.

## Scatter study, final setting (2026-09-09)
Hyperparameters now: K=7 probes f_x in [-1.5,1.5] step 0.5; (A,B) uniform on [-3,3]^2 (the
+0.3*sign origin guard of run_experiments.py is GONE, so points can sit near the degenerate
origin -- closest is A=0.008, tightest min gap along the ramp 0.73). BOTH SECTORS shown.

NO TARGET. The nudge is a constant beta=0.02 on the measured operator with a standardised error
signal eps == 1, and the gradient is the AVERAGE over probes:
   g = (1/K) sum_i ( <d_theta H>_{beta,i} - <d_theta H>_{0,i} ) / beta   = (1/K) sum_i dy_i/dtheta.
New helpers qep_grad_probes / num_grad_probes in run_boundary.py replace qep.qep_grad_batch /
qep.num_grad_batch, which build eps = y - y_tau from a target.

*** MEAN vs SUM CHANGES NOTHING STATISTICALLY. *** It is a factor 1/K applied identically to
g_ad and g_ex, so every cosine and every relative deviation is bit-identical; only the plotted
magnitudes shrink by K. Verified by rerunning: all 12 rows unchanged. Do not expect the switch to
"suppress the bad cases" -- it cannot.

*** DOES AVERAGING OVER PROBES SUPPRESS THE BAD CASES? MEASURED, AND IT DEPENDS ON THE SECTOR. ***
median rel dev, avg-of-7 vs the single-probe distribution over the same points:
  sched  T  sector | avg of 7 | best fx  median fx  worst fx
 linear 10    A>0  |  0.2949  |  0.0863    0.4648     2.0225
 linear 10    A<0  |  4.8451  |  0.1528    2.1250    37.4621
 linear 50    A>0  |  0.0851  |  0.0464    0.4618     1.4662
 linear 50    A<0  | 17.1658  |  0.1416    0.9318   142.1034
 smooth 10    A>0  |  0.2582  |  0.0134    0.1093     1.2108
 smooth 10    A<0  |  4.0061  |  0.0301    3.0681    34.8462
 smooth 50    A>0  |  0.0128  |  0.0032    0.0142     0.1087
 smooth 50    A<0  |  4.6706  |  0.0033    0.0602   103.3372
A>0: averaging HELPS, sometimes a lot (linear T=50: 0.085 avg vs 0.462 median single, 5.4x better)
     -- the per-probe boundary errors partially cancel.
A<0: averaging HURTS. At T=50 stiff the median single probe is 0.93 but the worst is 142, and the
     mean is LINEAR in the per-probe gradients, so it inherits the catastrophic probe and lands at
     17.2 -- an order of magnitude worse than a typical single probe. Averaging suppresses errors
     that are independent and comparable in size; it cannot suppress one term that is 100x the
     rest. This is the answer to "shouldn't the average suppress the bad cases": only where the
     gap stays open.

Figures: colour is now the SIGNED A on a diverging PuOr scale centred at 0 (orange A<0, purple
A>0) -- every point still distinct, and the sector readable off the marker; PuOr not RdBu so it
cannot be read as the stiff/smooth red/blue of Figs 1-8. Markers: dot = g_Omega (B/sigma^x
component), open circle = g_Delta (A/sigma^z component). bd_grad_cos_scatter_sched uses SYMLOG on
both axes (linthresh = 90th pct of |g_exact|): with A<0 back the range spans ~40x and a linear
axis squashes the A>0 bulk into an invisible blob; symlog keeps both visible and y=x is still the
diagonal since both axes transform identically. New figure bd_scatter_points shows where the 40
points sit in the (A,B) plane in the same colours.

## Wedge sampling: the right idea, applied to the wrong path (2026-09-09)
Sampling changed to: (A,B) uniform on [-3,3]^2 REJECTING the wedge {A < 1 and |B| < (1-A)/3}
(16.1% of candidates). The geometry behind it is correct and worth keeping in mind:

  The ramp is (A(tau),B(tau)) = (1,0) + f(tau)*(A-1, B) -- a straight SEGMENT in the (A,B) plane
  from the start to the endpoint. The gap 2|(A(tau),B(tau))| vanishes only at the ORIGIN, so the
  dangerous endpoints are exactly those whose segment passes near (0,0). The perpendicular
  distance from the origin to that segment is |B|/sqrt((A-1)^2+B^2), so the rays B = +-(A-1)/3 out
  of (1,0) bound a cone of half-angle arctan(1/3) about the -A direction. Excluding it forces a
  closest approach >= 1/sqrt(10) ~ 0.316, min gap >~ 0.63. VERIFIED on the kept sample: closest
  approach min 0.362, median 0.924. The wedge does exactly what it claims.

*** BUT IT DID NOT FIX THE A<0 SECTOR. *** median rel dev at T=50: A<0 stiff 16.40 (was 17.17
before the cut), smooth 4.13 (was 4.67); cosines still negative. A>0 is unchanged and clean
(smooth 0.016 at T=50, cos 1.0000).

WHY: THE PROBE SHIFTS THE PATH. With a probe on, H = -A sz - (B+fx) sx, so the segment runs from
(1, fx) to (A, B+fx) -- displaced vertically by fx. The wedge protects only the fx=0 member of
that family. Measured closest approach to the origin on the kept sample:
    unprobed path            min 0.362  median 0.924
    worst probe, A>0 points  min 0.215  median 1.000
    worst probe, A<0 points  min 0.036  median 0.105   <-- an order of magnitude smaller
All six worst points are A<0 with their worst probe at fx = +-0.5 driving the path to within
0.036-0.15 of the origin. Spearman rho against the deviation: -0.673 for the worst-probe distance
vs -0.609 for the unprobed one, so the probed distance is the better predictor.

THE CORRECTED RULE (tested, NOT adopted -- the shipped sampling is the wedge as specified):
reject an endpoint if ANY probe's path (1,fx) -> (A,B+fx) comes within 1/sqrt(10) of the origin.
Rejects 54.3% of candidates instead of 16.1%, and only 1 of 40 kept points still has A<0. Result
at T=50: A<0 rel dev 0.168 (stiff) / 0.528 (smooth) with cosines 0.9999 / 0.9929, against 16.40 /
4.13 and negative cosines under the wedge. Mechanism confirmed.

GENERAL LESSON: any geometric safety criterion for this model has to be applied to the PROBED
Hamiltonian, over the whole probe set -- not to H_sys alone. The probe is not a small perturbation
here: fx ranges over +-1.5 while |B| itself is O(1).

## train_multi_init_paths over T x schedule (2026-09-09) -> figures/train_multi_init_paths_TS
Repeat of train_multi_init_paths with the ramped nudge, for T = 1, 2, 5, 10, 20 x {smooth, stiff}.
code/train_experiments.py::exp_multi_init_paths_TS ("multi_init_paths_TS", 48s) ->
data/train_multi_init_paths_TS.npz -> code/draw_train_figs.py::fig_multi_init_paths_TS.
Same training rule as the rest of the training study (mean grad, LR=2.0, N=150, beta=0.02, K=9,
same 6 inits). NOTE "stiff" here means the SCHEDULE f(x)=x; the nudge is ramped in BOTH rows
(nudge_ramp=True applies ramp(t/T, schedule) to the nudge too), so the comparison is purely about
the schedule shape.

median |p-h| after 150 epochs:
   T     |    1     |    2     |    5     |   10     |   20
  smooth | 2.40e+00 | 6.17e-01 | 1.15e-01 | 1.01e-02 | 5.89e-03
  stiff  | 2.30e+00 | 5.31e-01 | 5.42e-02 | 2.13e-02 | 2.09e-02

- T=1 and T=2 FAIL, and they fail in an interesting way: not by wandering but by converging
  RELIABLY TO THE WRONG POINT. 5 of 6 inits land on the same spurious fixed point, (3.14, 2.00) at
  T=1 and (1.42, 1.46) at T=2 (smooth). At T=1 the paths make a large excursion to B~4.2 first.
  So short T gives a systematic bias in the fixed point, not noise -- gradient descent is happily
  minimising the wrong function, since the free phase is not preparing a ground state at all.
- The ill-conditioned init (0.2,2.1) runs away completely at T=1,2 (|p-h| ~ 14..22) and is the only
  off-frame trajectory; it needs T>=20 to converge, matching the audit's finding for that point.
- At T=5 the STIFF schedule is BETTER (5.4e-2 vs 1.15e-1). Consistent with the small-T result of the
  bandwidth study (report sec. 7.6): when nothing is adiabatic, stiffer/larger-lam wins.
- *** DO NOT read a plateau from T=10 vs T=20. *** Stiff gives 2.13e-2 -> 2.09e-2 there, which LOOKS
  like saturation at the O(1) bias, but extending the same run to T=40, 80 shows it is the
  OSCILLATION, not a floor:
      T     |  10      |  20      |  40      |  80
     smooth | 1.01e-02 | 5.89e-03 | 1.12e-03 | 3.45e-04   <- monotone, ~T^-1.5
     stiff  | 2.13e-02 | 2.09e-02 | 7.95e-04 | 2.58e-03   <- ERRATIC; beats smooth at T=40
  The stiff O(1) gradient bias oscillates in T (period ~2.5), so the training endpoint inherits
  that: a single T can look good or bad by luck. The real statement is that SMOOTH decays
  monotonically and predictably with T while STIFF does not decay systematically at all -- which is
  the note's own point, now visible at the level of the trained parameters.
  (I nearly reported a plateau off the two figure points; the T=40/80 check refuted it.)

Frame note for the figure: the 10 panels share one window, sized to hold every trajectory point
with |p-h| < 4 so the short-T spurious fixed points and the T=1 excursion are visible while the one
runaway init is excluded; panels annotate how many endpoints are off-frame. GOTCHA: \square and
\circ are NOT in matplotlib mathtext (same class as \tfrac) -- use words in the suptitle.

## Wedge as a LABEL, not a filter (2026-09-09, final state of the scatter study)
Sampling is back to plain uniform [-3,3]^2, all 40 points kept. WEDGE_SLOPE = 0.5 in
run_boundary.py; exp_grad_scatter_sched now saves `in_wedge` (bool, 40) and `wedge_slope`, and the
figures draw the 13 in-wedge points RED. Colour ramp switched PuOr -> VIRIDIS specifically because
viridis contains no red, so the ramp and the wedge marker cannot be confused; red #C0392B is
>= dE 12.8 (CVD) from every colour on the viridis ramp. Statistics in all figures now split on
wedge membership rather than on the sign of A.

Median rel dev, outside | inside the wedge:
   T      out stiff  out smooth  |  in stiff  in smooth
   1        1.124      1.247      |   0.538     0.713
   2        0.950      1.060      |   3.051     3.348
   5        0.412      0.474      |   2.503     3.844
  10        0.635      0.399      |   4.419     4.678
  20        0.501      0.142      |   4.360     3.864
  50        0.555      0.017      |  13.06      6.07
Outside: smooth converges to 0.017 (cos 1.0000); stiff stalls near 0.5 with cos 0.9967 -- direction
right, magnitude wrong. Inside: both diverge with T. NOTE THE INVERSION AT T=1-2: in-wedge points
are BETTER there (0.54 vs 1.12) because nothing is adiabatic yet and the gap crossing has not had
time to do damage; the penalty only appears once the evolution is slow enough for Landau-Zener to
matter.

WEDGE SLOPE COMPARISON (when it was still used as a rejection rule), A<0 median rel dev at T=50:
   no cut          stiff 17.17  smooth 4.67   cos +0.169 / +0.122
   slope 1/3 cut   stiff 16.40  smooth 4.13   cos -0.212 / -0.179   (rejects 16.1%)
   slope 1/2 cut   stiff  7.25  smooth 4.07   cos -0.819 / -0.187   (rejects 23.5%)
   per-probe cut   stiff  0.17  smooth 0.53   cos +1.000 / +0.993   (rejects 54.3%)
Widening the wedge helps but does not solve it; only the per-probe criterion does. The wedge is a
statement about the UNPROBED path and the probe shifts it by fx, which is the whole gap.

## Probe-set economy: can fewer f_x train? (2026-09-10) -> figures/train_probe_sets_{paths,summary}
code/train_experiments.py::exp_probe_sets_TS ("probe_sets_TS", 157s) -> data/train_probe_sets_TS.npz
-> draw_train_figs.py::fig_probe_sets_{paths,summary}. Also paper_figs.ipynb Figs 14-15 (cached).
Four sets, everything else at the training-study values (mean grad so the step is K-INDEPENDENT,
LR=2.0, N=150, beta=0.02, same 6 inits, ramped nudge, T=1,2,5,10,20, both schedules):
  K9 = linspace(-2,2,9) | K7 = +-1.5 step .5 | K5 = +-2 step 1 | K3 = -1,0,1

*** ALL FOUR SETS IDENTIFY (A,B) EXACTLY. *** y_tau is generated FROM h, so C(h)=0 exactly for any
probe set -> h is always a global minimum, even for K=3. Nothing here is an identifiability limit.
What differs is the CONVERGENCE RATE at fixed LR/epochs. Verified with exact equilibria:
   set | N=150    | N=600    | N=2400
   K9  | 2.23e-07 | 1.34e-14 | 1.34e-14
   K7  | 4.85e-03 | 6.40e-08 | 7.63e-14
   K5  | 1.23e-11 | 1.21e-14 | 1.21e-14
   K3  | 9.81e-03 | 9.36e-11 | 8.90e-14
So the K7/K3 "exact" residuals at N=150 are SLOW CONVERGENCE, not bias. Do NOT draw them as floor
lines (I nearly did; they span 9 decades and read as identifiability limits).

WHY, and it is NOT the point count -- it is whether the set BRACKETS the response peak. The curve
y(f_x)=A/sqrt(A^2+(B+f_x)^2) peaks at f_x=-B with width ~A; at h=(0.8,1.5) that is f_x=-1.5,
width 0.8. Singular values of dy/d(A,B) over each set (predicts the ordering BEFORE any training):
   set | K | sing. values     | cond | worst-dir sens. | nearest probe to the peak
   K9  | 9 | 1.1615, 0.6413   | 1.8  | 0.6413          | -1.5  (0.00 widths off)
   K7  | 7 | 1.1335, 0.2937   | 3.9  | 0.2937          | -1.5  (0.00 widths off)
   K5  | 5 | 0.8490, 0.6207   | 1.4  | 0.6207          | -2.0  (0.62 widths off)
   K3  | 3 | 0.8101, 0.2458   | 3.3  | 0.2458          | -1.0  (0.62 widths off)
=> K5 (5 points) is BETTER CONDITIONED THAN K7 (7 points): cond 1.4 vs 3.9, worst-direction
sensitivity 0.62 vs 0.29. K7 has the peak on the BOUNDARY of its range so it sees one flank only;
K5 straddles it (-2 and -1). Span/bracketing beats count.

ADIABATIC training, median |p-h|, smooth schedule:
   set |  exact   |  T=1     |  T=2     |  T=5     |  T=10    |  T=20
   K9  | 2.23e-07 | 2.40e+00 | 6.17e-01 | 1.14e-01 | 1.01e-02 | 5.89e-03
   K7  | 4.85e-03 | 1.67e+00 | 8.67e-01 | 1.65e-01 | 2.12e-02 | 8.80e-03
   K5  | 1.23e-11 | 2.40e+00 | 5.34e-01 | 1.67e-01 | 1.78e-02 | 6.20e-03
   K3  | 9.81e-03 | 1.88e+00 | 1.13e+00 | 8.00e-01 | 2.40e-02 | 1.11e-02
- PRACTICAL ANSWER: K=5 is the good trade. It costs 5/9 of the evolutions per epoch (QEP does 2
  solves per probe) and at T=20 it matches K9 to 5% (6.20e-3 vs 5.89e-3); at T=10 it is 1.8x worse
  (1.78e-2 vs 1.01e-2), so the saving is not entirely free at intermediate T. K7 is worse than K5
  at BOTH (1.19x at T=10, 1.42x at T=20) despite having two more points -- the conditioning, not
  the count.
- K=3 is the only set that materially hurts, and mainly at INTERMEDIATE T (T=5: 8.0e-1 vs 1.1e-1
  for K9 -- 7x). By T=20 it is within 2x (1.11e-2). Its ill-conditioned init (0.2,2.1) zig-zags
  badly at T=2,5 (visible in the paths figure).
- Ordering at T>=10 tracks the worst-direction sensitivity above, not K.

## Probe sets: stiff trajectories + cost-vs-epoch (2026-09-10)
figures/train_probe_sets_paths_linear (stiff companion to _paths; SAME data, no re-training --
draw_train_figs.py::fig_probe_sets_paths now takes a `sched` argument) and
figures/train_probe_sets_cost_epoch{,_ref} from code/train_experiments.py::exp_probe_sets_costs
("probe_sets_costs", 8s, pure post-processing of train_probe_sets_TS.npz). paper_figs.ipynb
Figs 16-18.

STIFF schedule, median |p-h| (compare the smooth table above):
   set |  T=1     |  T=2     |  T=5     |  T=10    |  T=20
   K9  | 2.30e+00 | 5.31e-01 | 5.42e-02 | 2.13e-02 | 2.09e-02
   K7  | 1.60e+00 | 7.93e-01 | 6.75e-02 | 3.80e-02 | 2.21e-02
   K5  | 2.30e+00 | 5.82e-01 | 4.76e-02 | 2.13e-02 | 3.34e-02
   K3  | 1.26e+00 | 7.22e-01 | 7.96e-01 | 4.36e-01 | 4.09e-02
- K3 + STIFF is the worst combination by far: 4.36e-01 at T=10, vs 2.40e-02 for K3+smooth (18x) and
  2.13e-02 for K9+stiff. The two weaknesses compound -- a weakly-constrained parameter direction
  plus a gradient whose bias does not decay.
- With stiff, none of the sets improve from T=10 to T=20 (K9 2.09e-2, K5 even worsens to 3.34e-2):
  the oscillating O(1) bias again, as in the multi_init_paths_TS note.

COST vs EPOCH (figures/train_probe_sets_cost_epoch). The average cost along the SAME stored paths,
evaluated with EXACT equilibria, on EACH SET'S OWN probes -- the objective that run minimises.
(An earlier version also scored every path on a common linspace(-2,2,21) grid; the user asked for
K3/K5/K7/K9 only, so that was removed from the experiment, the figure and the notebook. For the
record it had shown K3 reading 2.6e-3 on its own 3 probes against 1.9e-2 on the common grid at
T=5 -- and that gap was robust to the grid choice, incl. a grid containing none of the training
probes. If a generalisation measure is ever wanted again, |p-h| already serves as one.)

*** Do NOT compare absolute cost values ACROSS sets: C_K3 averages 3 probes and C_K9 nine, so they
are different functions. Compare the DESCENT of each curve; |p-h| is the set-independent measure. ***

final median cost, smooth schedule (own probes):
   set |   T=1    |   T=2    |   T=5    |   T=10   |   T=20
   K9  | 5.10e-02 | 1.38e-02 | 6.48e-04 | 3.38e-06 | 1.77e-06
   K7  | 3.49e-02 | 2.80e-02 | 4.87e-04 | 3.50e-06 | 2.23e-06
   K5  | 4.91e-02 | 1.17e-02 | 1.71e-03 | 1.75e-05 | 2.37e-06
   K3  | 5.47e-02 | 3.24e-02 | 2.58e-03 | 1.72e-05 | 4.26e-06
final median cost, stiff schedule (own probes) -- note it gets WORSE from T=10 to T=20 for K9/K7/K5,
the oscillating O(1) bias again:
   K9  | 4.16e-02 | 6.79e-03 | 1.59e-04 | 1.27e-05 | 3.17e-05
   K7  | 3.78e-02 | 1.85e-02 | 3.77e-04 | 2.48e-05 | 4.16e-05
   K5  | 3.99e-02 | 5.39e-03 | 1.39e-04 | 2.29e-05 | 7.39e-05
   K3  | 4.24e-02 | 5.12e-03 | 1.29e-03 | 5.65e-04 | 5.28e-05
- Dynamics (Fig 17): T=1,2 plateau high (~1e-2..5e-2) and never descend -- the trajectory failure
  seen as a cost curve. At T=20 K3's curve only starts falling around epoch 100 (slow convergence,
  matching its worse conditioning); K9 is essentially converged by epoch 40.

## Two outlier populations distinguished: Landau-Zener (red) vs sigma^x-saturation (black x) (2026-09-11)
New: fig_baxis_points in draw_boundary_figs.py -> figures/bd_baxis_points.{png,pdf}, Figure 20 of
paper_figs.ipynb. Addresses a question about Figure 11: besides the red wedge (Landau-Zener)
points, some points near the B-axis also show elevated relative deviation -- are they ALSO broken?

ANSWER: no, different mechanism, not a failure. At A=0 the ground state of H=-A sz-(B+fx)sx is a
pure sigma^x eigenstate (<sz>=0 identically), so the sigma^z NUDGE (which is what QEP always
perturbs, d_A H=-sz) only weakly moves <sx> there. Verified the EXACT power law by scanning A=2
fixed, B->0: g_Delta (dC/dA) ~ B^2 (ratio->4.00 as B halves), g_Omega (dC/dB) ~ B^1 (ratio->2.00).
So near the B-axis it is specifically g_Omega that gets small, while g_Delta stays O(1) -- the two
components do NOT vanish at the same rate, worth keeping straight in any writeup.

Checked that the ADIABATIC estimate at these points is NOT dynamically broken: at a concrete
near-B-axis point (A,B)=(0.05,2.5) (min gap along every probed path stays >=0.80, nowhere near
degenerate), the ABSOLUTE adiabatic-exact deviation decays like an ORDINARY 1/T boundary effect --
0.697(T=5) -> 0.335 -> 0.166 -> 0.088 -> 0.041 -> 0.001(T=160), i.e. converges completely normally.
It only LOOKS bad in Figure 11's RELATIVE-deviation reporting because the denominator (the exact
gradient) is legitimately close to zero there -- dividing an ordinary-sized absolute error by a
near-zero exact value inflates the ratio, no dynamical failure involved. Contrast the wedge, where
the absolute deviation does NOT shrink with T at all.

Criterion for marking (a cross, black, distinct from the red wedge fill): |A| < BAXIS_THR = 0.3,
outside the wedge. Chosen because it is STABLE (0.25-0.35 give the identical 4-point set, so it is
not sitting on a data point) and because it isolates points where g_Omega is measurably small
without being contaminated by the wedge's much larger, unrelated failure. 4 of the 40 points
qualify: (0.193,-0.871), (0.203,-1.850), (-0.250,-1.426), (0.008,2.489); relative deviation at
K=1, smooth, T=20 is only 2-8% for all four (vs the wedge's >=2x, often >>10x at K=1).

Figure built specifically at K=1 (f_x={0}) as requested, since a single zero-field probe isolates
the sigma^x-saturation mechanism most cleanly (with K=7 the effect is partly masked/mixed by
probe-averaging). Panel (a) = the (A,B) map from Figure 9, crosses added. Panel (b) = the K=1,
smooth, T=20 gradient scatter (data already existed in bd_grad_scatter_probes.npz from Figure
11', no recomputation needed), crosses on the same 4 points. Axis limits in panel (b) taken from
the NON-wedge points only (K=1's wedge failures reach |g|~15 and would otherwise squash the 4
crosses -- the actual subject -- into an unreadable clump); this follows the same convention
already used in fig_grad_cos_scatter_sched/probes. Result: the 4 crosses sit visibly ON the y=x
diagonal in panel (b), confirming by eye that they are well-behaved, not outliers in any
meaningful sense once relative-vs-absolute error is disentangled.
