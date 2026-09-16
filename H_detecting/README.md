# H_detecting — Hamiltonian detection with Quantum Equilibrium Propagation

This project trains the controls of a single-qubit Hamiltonian to reproduce an unknown **target**
Hamiltonian, using **Quantum Equilibrium Propagation (QEP)** gradients and a **probing field** that
makes both the *amplitude* and the *direction* of the target field identifiable. This README
documents the physical model and **every object and function** used in the code.

---

## 1. Physical model

### 1.1 System Hamiltonian
The trainable ("model") Hamiltonian is a single qubit:
```
H_sys(A, B) = -A·σz - B·σx
```
The two real numbers `p = (A, B)` are the **trainable parameters** (`A` multiplies σz, `B` multiplies σx).

### 1.2 Target Hamiltonian
The unknown Hamiltonian to be detected is
```
H_τ = -h_z·σz - h_x·σx,        h = (h_z, h_x)
```
The goal of training is to drive `(A, B) → (h_z, h_x)`.

### 1.3 Probing field and readout
We interrogate a Hamiltonian by adding a **probe field** `f_x` along x and measuring the
z-magnetisation of the resulting **ground state**:
```
H(f_x) = H_sys(A,B) - f_x·σx = -A·σz - (B + f_x)·σx
y(f_x) = ⟨σz⟩ in the ground state of H(f_x)
```
For a single qubit the ground state aligns with the total field `(B+f_x, 0, A)`, giving the
closed-form **response curve**
```
y(f_x) = A / sqrt(A² + (B + f_x)²)        (exact)
```
This curve is peaked at `f_x = -B`, has width `~A`, saturates at `±1`, and its tails decay as
`1/|f_x|`. Crucially its **shape depends on the absolute values of A and B**, not just their ratio —
this is what removes the amplitude degeneracy that the single-qubit linear-cost study suffered from.

> **Degeneracy note.** At `f_x = -B` the σx coefficient vanishes; if additionally `A = 0` the whole
> Hamiltonian is the zero matrix and its ground state is undefined (degenerate). Probe grids avoid
> this point (e.g. the `H_0=-σx` test grid excludes `f_x=-1`).

### 1.4 Cost and error signal
Given a dataset of probe/response pairs `{(f_x^(i), y_τ^(i))}` measured from the target, we fit the
model's response with the **quadratic cost**
```
C = (1/K) · Σ_i  ½ (y_i - y_τ^(i))²          (average over the K probes)
ε_i = ∂C/∂y_i = y_i - y_τ^(i)               (error signal of probe i)
```

### 1.5 QEP (Equilibrium Propagation) gradient
EqProp computes the parameter gradient from two equilibrium states. The **nudge** couples the error
signal to the **measured** operator (here σz):
```
free  phase:  equilibrium of  H_sys - f_x·σx
nudge phase:  equilibrium of  H_sys - f_x·σx + β·ε·σz       (β small)
```
The per-probe QEP gradient is
```
g_i = ( ⟨∂_θ H⟩_nudge − ⟨∂_θ H⟩_free ) / β ,   with ∂_A H = -σz,  ∂_B H = -σx
    = ( (-⟨σz⟩,-⟨σx⟩)_nudge − (-⟨σz⟩,-⟨σx⟩)_free ) / β
```
and the update uses the **average** over the dataset:
```
ḡ = (1/K) Σ_i g_i ,        (A,B) ← (A,B) − η·ḡ
```

### 1.6 Two ways to obtain the equilibrium
- **`method="exact"`** — exact diagonalisation of the (static) Hamiltonian. T-independent; the
  reference ground state.
- **`method="adiabatic"`** — Schrödinger evolution (`dynamiqs.sesolve`) under a schedule that ramps
  `(A,B)` from the fixed start `(A0,B0)=(1,0)` to the trained value over total time `T`, using the
  **smooth ramp** `λ(τ)=½(1−cos πτ)`. The smooth ramp has zero time-derivative at the endpoints,
  which cancels the O(1) adiabatic boundary-term bias; the nudge is likewise ramped on smoothly. The
  ODE tolerance is tight (`rtol=atol=1e-9`) because the QEP gradient divides an O(β) signal by β.

---

## 2. Core module — `code/qep.py`

All functions take plain arrays; `p` is `(A,B)`, `h` is `(h_z,h_x)`, `fxs` is a 1-D array of probe
fields, `T` the evolution time, `Nsteps` the output-grid length (use `NS=2`, endpoints only — the
adaptive solver handles accuracy), `method ∈ {"exact","adiabatic"}`, `schedule ∈ {"smooth","linear"}`.

| Object | Signature | Definition / returns |
|---|---|---|
| `SZ, SX` | module constants | `dynamiqs` Pauli `σz`, `σx` (QArrays). |
| `A0, B0` | `= 1.0, 0.0` | Fixed start of the adiabatic schedule (not trained). |
| `ramp(x, schedule="smooth")` | x∈[0,1] → [0,1] | Schedule shape. `"smooth"`=½(1−cos πx) (zero boundary slope); `"linear"`=x (stiff, biased); or **any callable** `f(x)` — used for the bandwidth family, applied to the `(A,B)` ramp *and* the nudge. |
| `parametrized_ascend(x, lam)` | → array | Bandwidth family `f_λ` with `f'(0)=f'(1)=0` for every λ but `f''(0)=+λ`, `f''(1)=−λ` exactly; `λ→∞` gives the linear schedule. See §7. |
| `d_parametrized_ascend(x, lam)` | → array | Analytic `f_λ'(x) = B_λ(x) + 30(1−I_λ)x²(1−x)²`. |
| `sched_param(lam)` | → callable | The `schedule=` argument for `parametrized_ascend` at bandwidth λ. |
| `SMOOTH_LAM` | `= π²/2 ≈ 4.9348` | `f''(0)` of the standard ½(1−cos πx) ramp — its member of the λ family. |
| `htau(h)` | → QArray | Target Hamiltonian `-h_z σz - h_x σx`. |
| `E0(h)` | → float | Target ground energy `-|h|`. |
| `gs_of(Hq)` | QArray → ket QArray | Ground state (lowest eigenvector) via `eigh`. |
| `y_analytic(p, fxs)` | → array | Closed-form response `A/√(A²+(B+f_x)²)`. |
| `equilibrium(p, fx, T, beta, err, Nsteps, method, schedule="smooth", nudge_ramp=True, probe_ramp=False, tol=1e-9)` | → ket QArray | Ground/equilibrium state of `-A σz -(B+fx) σx + β·err·σz`. Exact: diagonalisation. Adiabatic: `sesolve` from `gs_of` of the start Hamiltonian, ramping `(A,B)` and the nudge. `nudge_ramp`/`probe_ramp` are the audit switches of §6 (defaults = shipped behaviour, both no-ops for `method="exact"`). `tol` is the Tsit5 `rtol=atol`; the `1e-9` default is the study-wide setting — tighten it only for observables that sit below that floor (§3.5). |
| `measure(p, fx, T, Nsteps, method, schedule="smooth")` | → float | `y = ⟨σz⟩` of the **free** equilibrium (β=0). |
| `response_curve(p, fxs, T, Nsteps, method, schedule="smooth")` | → array | `measure` vmapped over `fxs`. |
| `make_dataset(h, fxs)` | → array | Target response `y_τ(f_x) = ⟨σz⟩` in the exact GS of `H_τ − f_x σx`. The "experimental data". |
| `qep_grad_single(p, fx, y_tau, T, beta, Nsteps, method, schedule="smooth")` | → (y, grad(2,)) | One-probe QEP gradient of `C_i=½(y−y_τ)²`. Free phase gives `y` and `ε=y−y_τ`; nudge phase uses `β·ε·σz`; returns `(dn−df)/β`. |
| `qep_grad_persample(p, fxs, y_taus, T, beta, Nsteps, method, schedule="smooth")` | → (ys(K,), grads(K,2)) | `qep_grad_single` vmapped over the dataset. Used to visualise per-probe gradient **directions**. |
| `qep_grad_batch(p, fxs, y_taus, T, beta, Nsteps, method, schedule="smooth", reduce="sum")` | → (ys(K,), grad(2,)) | Reduces per-probe gradients. `reduce="mean"` → **average** gradient (training rule); `reduce="sum"` → total (default; matches `num_grad_batch` of the summed cost in the detection study). |
| `cost(p, fxs, y_taus, T, Nsteps, method, schedule="smooth")` | → float | **Summed** cost `Σ_i ½(y_i−y_τ_i)²`. (Divide by K for the average cost.) |
| `num_grad_batch(p, fxs, y_taus, T, Nsteps, method, eps=1e-4, schedule="smooth")` | → grad(2,) | Central-difference gradient of `cost` — ground-truth reference for the QEP gradient. |
| `train_scan(p0, fxs, y_taus, T, beta, lr, N, Nsteps, method, use_adam=False, schedule="smooth", momentum=0.0, reduce="sum")` | → traj(N+1,2) | `lax.scan` gradient descent for `N` epochs. Uses `qep_grad_batch(..., reduce)`; supports plain GD, heavy-ball (`momentum`), or Adam (`use_adam`). The **training study uses `reduce="mean"`** with `lr=2.0`. |
| `param_error(P, h)` | → array | Euclidean `|p − h|` for a `(...,2)` array of params. |

`nudge_ramp` and `probe_ramp` are threaded through **every** helper above (`measure`,
`response_curve`, `qep_grad_single/persample/batch`, `cost`, `num_grad_batch`, `train_scan`) with
the same defaults, so the whole stack can be switched with one keyword. `num_grad_batch` calls
`jnp.asarray(p)` on entry, so a plain NumPy `p` is accepted (it previously raised
`AttributeError` on `p.at[i]`).

---

## 3. Experiment drivers

### 3.1 Detection study — `code/run_experiments.py` (`reduce="sum"`)
Run: `PYTHONPATH=. python run_experiments.py [test|full]`. Writes `data/<name>.npz`.

| Experiment | What it produces |
|---|---|
| `prep_fidelity` | adiabatic state-prep fidelity for several `(p, f_x)`. |
| `response_curves` | response curves for several amplitudes (same direction) + exact/adiabatic vs `y_τ`. |
| `grad_vs_beta` | QEP-vs-numerical gradient relative error vs β (exact ∝β; adiabatic floor). |
| `grad_cos_vs_T` | median cosine(QEP, exact gradient) vs T. |
| `schedule_bias` | gradient error vs T for stiff vs smooth ramp. |
| `n_probes` | final `|p−h|` and radius error vs number of probes K (amplitude identifiability). |
| `training_simple` | cost & `|p−h|` vs epoch (exact). |
| `param_paths` | (A,B) trajectories: exact + several T + single-probe. |
| `perf_vs_T` | final `|p−h|` vs T over many target/init configs. |
| `test_example` | recover `H_τ=−σz` from `H_0=−σx`; response fit. |

Figures: `code/draw_figs.py` → `figures/<name>.{png,pdf}`. Report: `report/report.tex` → `report.pdf`.

### 3.2 Training study — `code/train_experiments.py` (`reduce="mean"`)
Run: `PYTHONPATH=. python train_experiments.py [test|full|<names>]` (e.g.
`python train_experiments.py multi_init_paths response_evolution`). Writes `data/train_<name>.npz`.

| Experiment | What it produces |
|---|---|
| `grad_directions` | per-probe gradient directions `−g_i` vs the average `−ḡ` at a point; averaged descent vector field + trajectory; single-probe vs averaged fields. |
| `prep_fid_vs_T` | adiabatic ground-state-prep infidelity `1−F` vs T. **The `~1/T²` originally reported here is the `tol=1e-9` solver floor, not the physics — the true smooth-ramp exponent is 4 (§3.5); this experiment's `T ≥ 32` points are floor-limited.** |
| `training_mean` | average cost & `|p−h|` vs epoch (exact + adiabatic T=10,20,40). |
| `perf_vs_T` | final **average cost** vs T over 16 initial parameters (individual / mean±std / best). |
| `multi_init_paths` | (A,B) trajectories from 6 initial parameters (exact and adiabatic T=20). |
| `response_evolution` | model response curve before / during / after training vs the target. |

Helpers in this driver:
- `avg_cost(p, …)` — average cost `cost(...)/K` for one params point.
- `avg_cost_traj(P, …)` — vmapped average cost over a `(M,2)` params array (≈15× faster than a Python
  loop; verified to give identical values).

Figures: `code/draw_train_figs.py` → `figures/train_<name>.{png,pdf}`. Report: `report/training.tex`.

### 3.3 Nudge/probe switch-on audit — `code/run_nudge_audit.py` (2026-09-09)
Run: `PYTHONPATH=. python -u run_nudge_audit.py [all|<names>]` (~5 min). Writes `data/audit_*.npz`.
Figures: `code/draw_audit_figs.py` → `figures/audit_*`. Report: `report-2026-09-09/report.tex`.

Compares three protocol variants — **cur** (shipped: ramped nudge, constant probe), **bug**
(un-ramped nudge, as a deliberate control) and **ref** (the `afm.py`/`h_learning` convention:
ramped nudge *and* ramped probe):

| Experiment | What it produces |
|---|---|
| `ramp_identity` | machine-level proof that `qep.ramp` **is** `afm.py`'s `ascend_func` (max diff exactly 0). |
| `grad_vs_T` | cos & rel-error of each variant's gradient vs exact-QEP, 200 random points, T=5…80. |
| `prep_fid_vs_T` | free-phase prep infidelity vs T for both probe conventions (isolates `probe_ramp`). |
| `gap_profile` | instantaneous gap along the schedule — *why* the probe convention matters. |
| `training_vs_T` | final `|p−h|` vs T per variant, 6 initial parameters. |

**Headline:** the reported "missing smooth switch-on" is **not** a defect in this project (the nudge
is already ramped, identically to the reference); the un-ramped nudge is in `../single_qubit`'s two
`data/` helpers. The ramp is nonetheless essential — disabling it makes the median gradient cosine
**stall at ~0.82 and never converge in T**. Probe ramping is a trade-off, not an improvement, so the
current default is kept. See `log/STATUS.md` and the report for the full findings.

### 3.4 Bandwidth / boundary-curvature study — `code/run_bandwidth.py` (2026-09-09)
Run: `PYTHONPATH=. python -u run_bandwidth.py [all|<names>]` (~25 min). Writes `data/bw_*.npz`.
Figures: `code/draw_bandwidth_figs.py` → `figures/bw_*`. Report: `report-2026-09-09/report.tex` §7.

Follow-up to `QEP_GPT-12.pdf`: the note's criterion for killing the O(1) gradient bias is on the
**first** derivative of the schedule. This asks what sets the **residual**, using the
`parametrized_ascend` family (λ = boundary second derivative).

| Experiment | What it produces |
|---|---|
| `schedule_shapes` | `f_λ`, `f_λ'`, `f_λ''` — λ tunes only the boundary curvature. |
| `dev_vs_lam` | RMS gradient deviation vs λ at T=10…80 (the main result). |
| `dev_vs_lam_points` | the same sweep at two more `(Δ,Ω)` points. |
| `prep_vs_T` | state-prep overlap `⟨ψ(T)|φ₀⟩` vs T per λ — the independent cross-check. |
| `small_lam_limit` | proves `f_λ → 10x³−15x⁴+6x⁵` (quintic smoothstep) as λ→0; no abruptness. |
| `optimal_lam` | λ scanned over 4 decades at several T — locates the optimum λ*(T). |
| `shape_test` | interior perturbed at fixed boundary data; shows `f''(0)` dominates but isn't the whole story. |

**Headline:** `dev ≈ 0.25 λ/T`, saturating at the stiff-schedule O(1) value for `λ ≳ 2T`, so all
`T` collapse onto one master curve `Φ(λ/T)`. The cosine ramp is the `λ=π²/2` member (agrees to
0.3%). Reshaping the schedule's *interior* at fixed boundary data moves the bias by only 10–20%
(and less as `T` grows) against a factor of 17 for a boundary change of the same size — the
boundary curvature dominates, without being the sole parameter at finite `T`. Below `λ≈1` a
λ-independent higher-order floor (`~T^-1.71`) takes over. **The deviation oscillates in `T`
(period ≈1.15), so every number is an RMS over a dense 24-point `T` window** — a single-`T`
reading is meaningless.

### 3.5 Boundary cancellation on the prepared state — `code/run_boundary.py` (2026-09-09)
Run: `PYTHONPATH=. python -u run_boundary.py [all|<names>]` (~20 s). Writes `data/bd_*.npz`.
Figures: `code/draw_boundary_figs.py` → `figures/bd_*`.

The **state**-level statement behind §3.4, and the entry point to the whole boundary story. For
this project's ramp `d_tau H = -f'(tau)[(A-A0) sigma^z + (B-B0) sigma^x]`, so the note's boundary
coefficient `B_n0(tau) = <phi_n|d_tau H|phi_0>/(E_0-E_n)^2` is **exactly proportional to the
schedule slope** `f'(tau)`. The two schedules of `../../networks.py` — `ascend_func` (smooth,
`f'(0)=f'(1)=0`) and `ascend_func_st` (stiff, `f'≡1`), bit-identical to
`qep.ramp(...,'smooth'/'linear')` — therefore differ in whether the leading boundary term survives
at all. Setup: `f_x=0`, start `(A0,B0)=(1,0)` i.e. `H(0)=-sigma^z`, evolve to `(A,B)`, overlap with
the exact GS of `-A sigma^z - B sigma^x`.

| Experiment | What it produces |
|---|---|
| `schedule_shapes` | the two `[0,1]→[0,1]` schedules, their slopes, and `|B_10(tau)|` along the ramp. |
| `prep_fid_vs_T` | prep fidelity vs T, dense grid (dT=0.25, T=3…160), both schedules, two `(A,B)`; RMS envelope, fitted exponent, predicted vs measured oscillation period, tolerance-convergence check. |
| `grad_vs_T` | `∂y/∂(A,B)` vs T by central difference, both schedules, against the closed-form exact-GS gradient — the same comparison as Fig. 1(b) of the note, in this project's model. |

**Results.** `|B_10(0)| = 0.375` (stiff) vs **exactly 0** (smooth, at both endpoints). Prep
infidelity `1-F ~ T^-2` for stiff (fitted −2.002, −2.008) and `~T^-4` for smooth (−4.004, −3.991)
at both `(A,B)` — Campos Venuti–Lidar boundary cancellation, i.e. §4.4 of the note at N=1. Both
oscillate in T at the predicted period `2 pi / int_0^1 (E_1-E_0) d tau`: 2.553 predicted vs 2.551
measured, and 2.798 vs 2.796 — a direct confirmation of the `e^{-iT(gamma_1-gamma_0)}` factor.
Note the **state** error decays either way; only the **gradient** keeps an O(1) bias, because
`d_theta` hits the dynamical phase and pulls out a factor T (that is §3.4).

**The gradient result (`grad_vs_T`).** This is the figure that carries the note's headline.
Relative deviation of the adiabatic gradient from the exact one, envelope-fitted on `T ≥ 20`:

| schedule | envelope vs T | mean rel. deviation, `T ≥ 20` |
|---|---|---|
| stiff `λ=τ` | `T^-0.001` — **flat, no decay** | **0.879** |
| smooth `½(1−cos πτ)` | `T^-1.001` | 0.032 |

So the stiff schedule sits **88 % off the true gradient at every T** — increasing T does not help,
exactly the note's "infinite T does not guarantee the accurate gradient", while the smooth ramp
decays as `1/T`. The signed component deviation oscillates at the predicted `2π/∫(E₁−E₀)dτ`
(2.553 vs 2.551 measured; note `|dev|` itself is rectified and so shows *half* that period). The
exact-GS reference is closed-form (`∂y/∂A = B²/R³`, `∂y/∂B = −AB/R³`) and agrees with a central
difference to 3.8e-10.

Contrast §3.5's Figure 2: the **state** error decays for both schedules. The gradient's O(1) bias
survives because `∂_θ` hits the dynamical phase `e^{-iTγ}` and pulls out a factor T.

**Notebook.** `paper_figs.ipynb` (top level) holds Figures 1–3. It contains **only light data
generation and plotting** — nothing in it solves a Schrödinger equation:

| Figure | data source | generated in the notebook? |
|---|---|---|
| 1 — schedules + `B₁₀(τ)` | analytic curves + a 2×2 `eigh` per τ | yes (no evolution, costs nothing) |
| 2 — prep fidelity vs T | `run_boundary.py prep_fid_vs_T` → `data/bd_prep_fid_vs_T.npz` | no, loaded |
| 3 — gradient vs T | `run_boundary.py grad_vs_T` → `data/bd_grad_vs_T.npz` | no, loaded |

10 cells, ~7 s end-to-end, 0 errors. `code/draw_boundary_figs.py` is the scripted twin of the plot
cells. The notebook resolves its own paths, so it runs from either `H_detecting/` or
`H_detecting/code/` (unlike `draw_all_figs.ipynb`, §3.6, which hardcodes `../data` and therefore
needs `cwd=code/`).

**Numerics.** These runs need `tol=1e-13`, not the study-wide `1e-9`: the smooth `T^-4` curve
reaches 1.4e-9 by T=160, where a `1e-9` solver reports 7.6e-8 — 53× too large. The right panel of
`figures/bd_prep_fid_points` is the convergence check. This is the only reason `qep.equilibrium`
grew a `tol` keyword; the `1e-9` default leaves every published number bit-for-bit unchanged.

### 3.6 Figure notebook — `draw_all_figs.ipynb`
Self-contained notebook with **all** plotting code (one cell per figure, same code as the two
`draw_*.py` scripts), reading `data/` and writing `figures/`. Run the Setup cell first.

### 3.7 Exploratory notebook — `H_detecting.ipynb`
The original sketch/debug notebook (core functions + analytic test). Kept for reference.

---

## 4. Key parameters & defaults
| Symbol | Value | Meaning |
|---|---|---|
| `h` | `(0.8, 1.5)` | target field (\|h\|≈1.70 ≠ 1, so amplitude recovery is non-trivial) |
| `FXS` | `linspace(-2,2,9)` | probe fields (K=9) |
| `BETA` | `0.02` | nudge strength β |
| `NS` | `2` | solver output points (endpoints; adaptive stepping) |
| `LR` | `0.3` (sum) / `2.0` (mean) | gradient-descent step |
| `N` | `150` | training epochs (converged by ~epoch 100) |
| solver | `Tsit5(rtol=atol=1e-9)` | tight, because QEP divides by β |

---

## 5. Environment & reproducibility
- Python: `/Users/qwang/miniforge3/envs/jaxlab/bin/python` — `jax 0.9.1`, `dynamiqs 0.3.3`,
  `optax 0.2.6`, `jax_enable_x64=True`.
- LaTeX: `/Users/qwang/miniforge3/envs/tex/bin/tectonic`.
- Run from `code/` with `PYTHONPATH=.`. Don't pipe long runs through `tail` (it buffers to EOF and
  looks frozen); use `python -u driver.py > log 2>&1` and poll the log.
- Project status / handoff notes: `log/STATUS.md`.

## 6. Protocol switches (added 2026-09-09)
Two opt-in keywords on `qep.equilibrium` and every helper that calls it. **Both defaults reproduce
the published study bit-for-bit** (verified against the pre-patch module, and by re-running
`exp_grad_cos_vs_T` and comparing all 7 saved arrays — bitwise identical).

| switch | default | meaning |
|---|---|---|
| `nudge_ramp` | `True` | `True`: nudge switched on as `β·ε·λ(t/T)` — shipped behaviour, identical to `afm.py`'s `QM.total_H`/`tune_func`. `False`: nudge at full strength from `t=0` (the `../single_qubit` bug), kept **only** as a control — it breaks convergence in `T`. |
| `probe_ramp` | `False` | `False`: probe `f_x` constant for the whole run, `ψ₀` is the exact GS *including* the probe — shipped behaviour. `True`: the reference's `QEP_grad_PF` convention, probe folded into `B` and ramped with it. Recommended only for deliberately ill-conditioned inits or small `T`. |

Both are no-ops for `method="exact"`, which sees only the `t=T` Hamiltonian.

## 7. Schedule bandwidth λ (added 2026-09-09)
`ramp()` accepts a callable, so any schedule can be used; `sched_param(lam)` supplies the
bandwidth family `parametrized_ascend`. λ is *exactly* the boundary second derivative
`f_λ''(0)`, while `f_λ'(0)=f_λ'(1)=0` holds for every λ — so λ isolates the curvature that sets
the residual adiabatic gradient bias once the O(1) term has been cancelled.

| λ | meaning |
|---|---|
| `λ → 0` | limit is the **quintic smoothstep** `10x³−15x⁴+6x⁵` (N=2: `f'=f''=0` at both ends) → `\|a\|~1/T³`, so deviation hits the `~T^-1.71` floor. Interior slope saturates at 1.875 (+19% vs the cosine ramp) — it does *not* become abrupt. |
| `λ = π²/2 ≈ 4.93` | the standard `½(1−cos πx)` ramp (reproduced to 0.3%); deviation `≈1.2/T` |
| `λ ≳ 2T` | saturated — indistinguishable from the linear/stiff schedule (O(1) bias, T-independent) |

Master law: `dev ≈ 0.25 λ/T` for `λ ≲ 2T`. Applied to the nudge as well as the `(A,B)` ramp, as
the boundary analysis requires.

The state-prep overlap confirms this independently — one excited-state amplitude `|a|` explains
both, via `1−F ≈ |a|²` and `dev ~ T|a|`:

| regime | condition | `1−F` exponent | `\|a\|` | `dev` |
|---|---|---|---|---|
| stiff | `λ ≳ T`, or linear | −1.87 | `1/T` | O(1) |
| curvature | `1 ≲ λ ≪ T` | −4.15 | `λ/T²` | `λ/T` |
| `λ→0` | curvature also gone | −5.72 | `1/T³` | `1/T²` |

⚠️ **Do not minimise λ at short T.** The λ-dependence has an optimum that falls as
`λ* ~ T^-2.5`: at T=3 the best λ is ≈10 and λ=0.01 costs 3.9× in prep infidelity. The minimum
flattens with T (penalty 3.9× → 1.0× over T=3…40), so for `T ≳ 10` small λ is penalty-free.

⚠️ `1−F` hits a **solver-tolerance floor** near 3e−8 and then *rises* ∝T (accumulated error at
`rtol=atol=1e-9`). Fit exponents only where `1−F > 1e-6`. Gradient results are unaffected (they sit
5 orders above it). See `report-2026-09-09/report.tex` §7 and `log/STATUS.md`.
