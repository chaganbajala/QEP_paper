# Figure 13: component independence and scientific interpretation

Audit date: 2026-09-14. Scope: existing N=5/N=9 Figure 13 data and the sampler
used by normal and expanded training. No production code, notebooks, data,
or cluster jobs were changed. This audit supersedes earlier unqualified claims
in README/GPT_relay that training uses exact diagonalization and costs 2*N_g
individual gradient shots per epoch.

## Conclusions

1. The current standalone sampler does **not** reuse the Omega key for Delta.
   Normal training and Figure 13 use the same independent component streams.
2. Figure 13 plots analytic **conditional** marginal moments, not empirical
   cross-component correlations. Similar relative-noise curves are plausible
   and survive checks using independently diagonalized ground states.
3. There is a different genuine accuracy problem: the solver named
   `Exact_Diagonalization` delegates to a fixed 100-step iterative approximation,
   not an eigendecomposition. This affects Figure 13 **and training**. The saved
   analytic moments agree with that approximate-state calculation, but differ
   by several percent from ground-state gradient means at some checked points.
4. Experimental shot accounting is also wrong as currently described: the two
   noncommuting groups each require N_g shots in each of two states, giving
   **4*N_g**, not 2*N_g, gradient preparations per epoch, plus N_y=100 output
   shots for the implemented protocol.

## Actual measurement streams

For per-epoch seed s, `afm_shots/grad.py` calls:

| Measurement | Seed | Number of shots |
|---|---:|---:|
| Output AFM squared | s | N_y |
| Free Omega/X | s+100 | N_g |
| Free Delta/Z | s+104829 | N_g |
| Nudged Omega/X | s+1000 | N_g |
| Nudged Delta/Z | s+105729 | N_g |

`multiple_measure` offsets parameter-tree leaves by 104729; the first two
leaves are the actual derivatives A_Omega=sum(X_i)/2 and A_Delta=-sum(n_i).
They do not commute (Frobenius norm of their commutator is 6.32456 for N=5,
33.94113 for N=9). The free and nudged states also use distinct streams.
Unused V measurements have further distinct keys and are zeroed before updating.

Instrumenting the actual public sampler verified identical call/seed sequences
for normal training and `call_with_moments`, and identical local sampled outputs
between these two routes with the same inputs. Expanded training imports this
same class and changes the seed each epoch through `epoch_seed`.

Two legacy files **still contain the shared-key bug**:

- `QEP_multiple_shot/Adaptive_bn/qep_shot/grad.py`, `multiple_measure`;
- `QEP_multiple_shot/AFM_searching/code/shot_grad.py`, `multiple_measure`.

Neither is imported by the current standalone shot study. The earlier relay
records the standalone fix before production submission. The legacy defect
should not be generalized to all old experiments without auditing their imports.
Reusing a key really can cause correlations; see the
[JAX PRNG documentation](https://docs.jax.dev/en/latest/random-numbers.html).

## Conditional Monte Carlo check

Reproducible diagnostic: `audit_component_noise.py`, run from QEP_multiple_shot:

```bash
JAX_PLATFORM_NAME=cpu MPLCONFIGDIR=/tmp/codex-mpl \
  /Users/qwang/miniforge3/envs/jaxlab/bin/python \
  -m AFM_searching.shot_study.audit_component_noise
```

N=5, (Omega,Delta)=(0.5,1.5), beta=0.1, N_g=1000, 2048 repetitions,
fixed sampled y=0.635199964; JAX 0.9.1 CPU. Each repetition uses the production
`_measure_single` implementation and actual component/phase seed offsets.
Vectorized draws were also checked against public `measure` calls.

| Quantity | Result |
|---|---:|
| Correlation between independently sampled gradient components | -0.001614 |
| Approximate 95% correlation interval (Fisher transform) | [-0.04493, 0.04170] |
| Control: deliberately reuse Omega keys for Delta | 0.19849 |
| Empirical / analytic conditional variance, Omega | 0.97680 |
| Empirical / analytic conditional variance, Delta | 0.98028 |

Thus this test detects the deliberately introduced shared-key correlation but
finds no evidence of it in the current sampler. It is a bounded diagnostic,
not a proof of independence for every possible input.

Historical bitwise replay is not claimed: local CPU replay of the first sampled
gradient gives (1.3600004,0.3500009), versus saved GPU (0.9100020,0.3500009).
The output y agrees within 6e-8 and conditional means within small numerical
errors. Backend-dependent eigendecompositions/sampling can change individual
draws; this audit did not reproduce the historical GPU environment. Provenance
evidence additionally comes from matching archived source and conditional moments.

## What the curves actually mean

At fixed parameters and fixed measured output y, write f/n for the free/nudged
states. For each i in {Omega, Delta},

    g_i = (sample_mean(A_i,n) - sample_mean(A_i,f)) / beta
    mu_i = (<A_i>_n - <A_i>_f) / beta
    sigma_i^2 = [Var_n(A_i) + Var_f(A_i)] / (N_g * beta^2)

Figure 13 loads the two different columns of mu and sigma squared and plots
sigma_i/abs(mu_i). There is no duplicated-column plotting error. It does not
estimate correlation between simultaneously sampled gradient components.

Both means respond to the same nudge, H_n=H_f+2*beta*(y-1)*AFM^2. In the weak
nudge limit both signals contain the small common factor y-1. This naturally
drives both relative-noise curves upward together as the target is approached;
shared fluctuations of y can produce matching spikes. Independent marginal
noise levels are not required to have different shapes.

Across finite plotted values, median (relative noise of Omega)/(relative noise
of Delta) is 0.84762 for N=5 and 0.79139 for N=9. They are similar, not equal.
For N=5 the first relative values are (0.58729,0.75712), the last are
(16.63031,19.13239). On a logarithmic axis those differences appear small.
The current division masks near-zero means at 23 epochs for N=5 and 1 for N=9;
those are undefined/divergent ratios, not evidence of vanishing shot noise.

The moments are conditional on the single measured y at each visited point.
They exclude the extra variability from resampling N_y output shots:

    Var(g_i | params) = E_y[Var(g_i | params,y)] + Var_y[mu_i(params,y)]
    Cov(g_Omega,g_Delta | params) = Cov_y[mu_Omega(params,y),mu_Delta(params,y)]

The latter can be nonzero even when X/Z experiments are independent. Figure 13
must be described as conditional gradient-measurement noise, not total training
noise. The Monte Carlo independence check fixes y for exactly this reason.

## Independent ground-state check: a real discrepancy

The call chain is `QEP.grad.Exact_Diagonalization.__call__` -> `qm.set_psi0`
-> `QEP.model.get_gs_sparse(...,itr_num=100)`. The last function performs 100
normalized momentum-gradient iterations from a uniform state, with no residual
convergence test. Its name does not make it an eigensolver.

The diagnostic independently builds the Hamiltonian and uses SciPy float64
`eigh` for its lowest eigenvector at saved pre-update parameters and saved y.
It verifies those Hamiltonians against the production Hamiltonian matrices.
Checks cover zero-based epochs 0,20,100,200,499 for each N, not all training.

| N; epoch index | Saved mean (Omega,Delta) | Ground-state mean (Omega,Delta) |
|---|---|---|
| 5; 0 | (0.759542,0.364020) | (0.752488,0.357070) |
| 5; 20 | (0.351274,0.132518) | (0.341963,0.127773) |
| 5; 499 | (0.029870,0.0050354) | (0.030584,0.0051963) |
| 9; 0 | (1.675777,0.793209) | (1.606324,0.757700) |
| 9; 200 | (0.017450,0.0025272) | (0.017928,0.0026832) |

Recomputing moments with the production iterative states instead reproduces
saved means to max absolute error 6.65e-6 (N=5), 1.75e-5 (N=9), across checked
points. Iterative state residual norms are about 0.0011--0.0029, despite high
ground-state fidelities. Small state errors matter when subtracting expectations
and dividing by beta or a small gradient signal.

Importantly, the **exact-ground-state** relative noises remain similar: N=5
first (0.59300,0.77125), last (16.24998,18.50241). The solver error therefore
does not explain away the qualitative overlap at these checked points.

Normal and current expanded training use the same approximate solver, so this
is not merely a plotting-label problem. Exact moments recalculated on saved
parameters do not turn those trajectories into exact-ground-state training.
The phase-background/endpoint helper `exact_am2` does call `jnp.linalg.eigh`;
that separate path should not be conflated with state preparation in training.

## Provenance and scope

### Follow-up: are 100 iterations sufficient?

The user specifically asked this during the audit. Added and ran
`check_iteration_convergence(n,k)` in the diagnostic script. It holds the
Hamiltonians and saved y fixed and compares 100/200/500-step states against
float64 diagonalization of those same production Hamiltonians. The largest
absolute relative error of the two gradient means is:

| Test point | 100 iterations | 200 iterations | 500 iterations |
|---|---:|---:|---:|
| N=5, first epoch | 1.9460% | 0.01696% | 0.000991% |
| N=5, last epoch | 3.0509% | 0.03704% | 0.03887% |
| N=9, first epoch | 4.6841% | 0.004943% | 0.000121% |

These results directly show inadequate convergence at 100 steps for these
points, not merely suspicion based on the iteration count. More iterations
reduce eigenvector residuals from about 1e-3 to 1e-8--3e-7. Late in N=5
training the small Delta signal exposes the remaining float32 normalization/
expectation-subtraction sensitivity: adding iterations alone is not a guarantee
of monotonically improving relative gradient accuracy. This is not a universal
certification of 200 or 500 iterations across parameter space.

For N=5 the Hamiltonian is only 32x32 (N=9: 512x512). An actual float64 Hermitian
eigensolver is the recommended ground-state reference, rather than replacing
one unvalidated fixed count by another. An iterative alternative needs both
convergence/residual checks and checks that it selects the ground state.

The four existing pipeline regression tests pass, but do not test convergence
or component independence. The new diagnostic module also passes syntax compilation.

Read-only SSH checks found matching SHA256 hashes locally, in original Raven
production root `/ptmp/qinwang/notebooks/QEP_fixed_shot_20260911/`, and in current
training root `/raven/u/qinwang/notebooks/QEP_overlap_20260914/`:

- `QEP/model.py`: d6b535c7d7d549b96b1643cccbf6653ea70638ddc96b61342148c32590bd82fe
- `QEP/grad.py`: adf36b859115cb5c66689888bd982d81e83fbee6942a343f6b591a49fdbd5310
- `QEP_multiple_shot/AFM_searching/afm_shots/grad.py`: b21ccf827cf6bbe955f186611b614035fdbd0583f5bf581e38e880243b4b513e
- `QEP_multiple_shot/AFM_searching/shot_study/run.py`: 2c3bf6100ca0c06b9df21534838975e76c12be962f10bf570475b2f5b635bc49

Present archive hashes are corroboration, not immutable historical job records.
No new N=9 training was launched; its existing diagnostic data were only audited.

## Required follow-up before claiming exact experimental benchmarks

- With user approval, provide an explicitly validated ground-state solver for
  this standalone study; avoid silently changing shared QEP core behavior.
- Recompute Figure 13 moments with that solver and label their conditional
  meaning, distinguishing them from an exact energy/cost derivative at beta=0.
- Establish a new, separately versioned N=5 training dataset using the corrected
  solver; do not mix approximate-solver prefixes with corrected continuations.
- Correct gradient-shot axes/metadata to 4*N_g*epochs for individual state
  preparations, or explicitly call 2*N_g*epochs a count of *paired* preparations.
  Counting the implemented independent output measurements gives
  (4*N_g+N_y)*epochs. No extra physical V measurements are needed since V is frozen.
- Add independence and eigenvector-residual regression tests; the existing four
  pipeline tests check neither scientific condition.

The user requested diagnosis, so these production changes and any decision to
cancel/restart jobs are intentionally left for approval.
