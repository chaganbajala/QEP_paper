#!/usr/bin/env python
"""Run reproducible cases for the five fixed-shot AFM research tasks.

Invoke from any directory.  Each case writes one self-describing compressed
``.npz`` file, so Slurm array tasks are independent and interrupted work can be
resumed by rerunning only missing case IDs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Import from QEP_multiple_shot, not from AFM_searching itself.  The latter has
# a legacy directory named ``code`` that otherwise shadows Python's stdlib code
# module while dynamiqs imports IPython/pdb.
QEP_MULTIPLE_SHOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or os.curdir).resolve() != Path(__file__).resolve().parents[1]]
sys.path.insert(0, str(QEP_MULTIPLE_SHOT))

import jax
import numpy as np

from AFM_searching.afm_shots import common
from AFM_searching.afm_shots.train import Shot_QEP_Training
from AFM_searching.shot_study import config as C
from QEP import grad as core_grad


def scalar_pair(tree):
    return float(np.asarray(tree[0])), float(np.asarray(tree[1]))


def epoch_seed(base_seed: int, epoch: int) -> int:
    return int(np.random.SeedSequence([int(base_seed), int(epoch)]).generate_state(1)[0])


def metadata(**extra):
    base = dict(
        schema_version=1,
        beta=C.BETA,
        learning_rate=C.LEARNING_RATE,
        N_y=C.N_Y,
        target=C.TARGET,
        T=C.T,
        dt=C.DT,
        gradient_shots_per_epoch_expression="2*N_g",
        output_shots_per_epoch=C.N_Y,
        estimator_groups="Omega:X;Delta_and_V:Z (independent PRNG streams)",
    )
    base.update(extra)
    return json.dumps(base, sort_keys=True)


def atomic_save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)
    print(f"saved {path}", flush=True)


def train_case(N, point, N_g, epochs, seed, with_moments=False, verbose_every=0):
    nn, params, grad_method = common.build_model_and_grad(N, *point, beta=C.BETA)
    trainer = Shot_QEP_Training(C.LEARNING_RATE)

    omega = np.empty(epochs + 1)
    delta = np.empty(epochs + 1)
    measured_am2 = np.empty(epochs)
    sampled_grad = np.empty((epochs, 2))
    expected_grad = np.empty((epochs, 2)) if with_moments else None
    gradient_var = np.empty((epochs, 2)) if with_moments else None
    omega[0], delta[0] = point

    started = time.time()
    for k in range(epochs):
        current_seed = epoch_seed(seed, k)
        if with_moments:
            y, grad, mean, variance = grad_method.call_with_moments(
                params, nn, C.TARGET, N_y=C.N_Y, N_g=N_g, seed=current_seed)
            expected_grad[k] = scalar_pair(mean)
            gradient_var[k] = scalar_pair(variance)
        else:
            y, grad = grad_method(
                params, nn, C.TARGET, N_y=C.N_Y, N_g=N_g, seed=current_seed)
        sampled_grad[k] = scalar_pair(grad)
        params = trainer.update(params, grad)
        omega[k + 1], delta[k + 1] = scalar_pair(params)
        measured_am2[k] = float(np.real(np.asarray(y)))
        if verbose_every and ((k + 1) % verbose_every == 0 or k + 1 == epochs):
            print(f"epoch {k+1}/{epochs}: AM2={measured_am2[k]:.6g} "
                  f"Omega={omega[k+1]:.6g} Delta={delta[k+1]:.6g}", flush=True)

    result = dict(
        Omega=omega,
        Delta=delta,
        measured_AM2=measured_am2,
        sampled_gradient=sampled_grad,
        elapsed_seconds=np.asarray(time.time() - started),
    )
    if with_moments:
        result.update(expected_gradient=expected_grad, gradient_variance=gradient_var)
    return nn, params, result


def run_gradient(N, scan, repeats, smoke=False):
    values = C.NG_GRID if scan == "ng" else C.BETA_GRID
    if smoke:
        values, repeats = values[:2], min(repeats, 4)
    point = C.TYPICAL_POINT
    # Task 1 explicitly uses the exact ground-state expectation of AFM^2.  Only
    # the free/nudged gradient observables are sampled.
    gradients = np.empty((len(values), repeats, 2))
    outputs = np.empty((len(values), repeats))
    exact_gradients = np.empty((len(values), 2))
    exact_outputs = np.empty(len(values))
    started = time.time()
    for i, value in enumerate(values):
        beta = C.BETA if scan == "ng" else float(value)
        N_g = int(value) if scan == "ng" else C.STATS_NG
        nn, params, grad_method = common.build_model_and_grad(N, *point, beta=beta)
        exact_method = core_grad.QEP_grad(beta, grad_method.solver)
        exact_y, exact_gradient = exact_method(params, nn, C.TARGET)
        exact_outputs[i] = float(np.real(np.asarray(exact_y)))
        exact_gradients[i] = scalar_pair(exact_gradient)
        for r in range(repeats):
            seed = epoch_seed(910000 + 1000 * N, r)
            y, grad = grad_method.call_with_exact_output(
                params, nn, C.TARGET, N_g=N_g, gradient_seed=seed)
            gradients[i, r] = scalar_pair(grad)
            outputs[i, r] = float(np.real(np.asarray(y)))
        print(f"{scan} {i+1}/{len(values)} value={value} "
              f"std={gradients[i].std(axis=0)}", flush=True)

    suffix = "_smoke" if smoke else ""
    path = C.DATA_DIR / f"gradient_{scan}_N{N}{suffix}.npz"
    atomic_save(
        path,
        metadata=metadata(task="gradient_distribution", N=N, scan=scan,
                          repeats=repeats, smoke=smoke, point=point,
                          N_y=0, output_estimator="exact_ground_state_expectation"),
        scan_values=np.asarray(values), gradients=gradients, outputs=outputs,
        exact_gradients=exact_gradients, exact_outputs=exact_outputs,
        elapsed_seconds=np.asarray(time.time() - started),
    )


def trajectory_cases(smoke=False):
    Ns = C.N_LIST[:1] if smoke else C.N_LIST
    points = C.INITIAL_POINTS[:1] if smoke else C.INITIAL_POINTS
    ngs = C.TRAJECTORY_NG[:1] if smoke else C.TRAJECTORY_NG
    seeds = range(1 if smoke else C.TRAJECTORY_SEEDS)
    return [(N, i, point, ng, seed) for N in Ns for i, point in enumerate(points)
            for ng in ngs for seed in seeds]


def run_trajectory(case_id, smoke=False):
    cases = trajectory_cases(smoke)
    N, init_id, point, N_g, seed_id = cases[case_id]
    epochs = 2 if smoke else C.TRAJECTORY_EPOCHS
    _, _, result = train_case(N, point, N_g, epochs, seed=seed_id + 1000 * init_id,
                              verbose_every=max(1, epochs // 10))
    suffix = "_smoke" if smoke else ""
    path = C.DATA_DIR / "trajectories" / (
        f"N{N}_init{init_id}_Ng{N_g}_seed{seed_id}{suffix}.npz")
    atomic_save(path, metadata=metadata(task="trajectory", N=N, init_id=init_id,
                                       point=point, N_g=N_g, seed_id=seed_id,
                                       epochs=epochs, smoke=smoke), **result)


def budget_cases(smoke=False):
    Ns = C.N_LIST[:1] if smoke else C.N_LIST
    ngs = C.NG_GRID[:1] if smoke else C.NG_GRID
    seeds = range(1 if smoke else C.BUDGET_SEEDS)
    return [(N, ng, seed) for N in Ns for ng in ngs for seed in seeds]


def run_budget(case_id, smoke=False):
    cases = budget_cases(smoke)
    N, N_g, seed_id = cases[case_id]
    max_budget = 40 if smoke else max(C.SHOT_BUDGETS)
    epochs = max_budget // (2 * N_g)
    nn, params, result = train_case(N, C.TYPICAL_POINT, N_g, epochs, seed=seed_id,
                                    verbose_every=max(1, epochs // 10))
    budgets = np.asarray((20, 40) if smoke else C.SHOT_BUDGETS, dtype=int)
    checkpoint_epochs = budgets // (2 * N_g)
    checkpoint_exact = np.full(len(budgets), np.nan)
    for i, ep in enumerate(checkpoint_epochs):
        if ep < 1 or ep > epochs:
            continue
        checkpoint_params = nn.set_params(float(result["Omega"][ep]),
                                          float(result["Delta"][ep]))
        checkpoint_exact[i] = common.exact_am2(nn, checkpoint_params)
    suffix = "_smoke" if smoke else ""
    path = C.DATA_DIR / "budgets" / f"N{N}_Ng{N_g}_seed{seed_id}{suffix}.npz"
    atomic_save(path, metadata=metadata(task="shot_budget", N=N, point=C.TYPICAL_POINT,
                                       N_g=N_g, seed_id=seed_id, epochs=epochs,
                                       budgets=budgets.tolist(), smoke=smoke),
                budgets=budgets, checkpoint_epochs=checkpoint_epochs,
                checkpoint_exact_AM2=checkpoint_exact, **result)


def run_stats(N, smoke=False):
    epochs = 2 if smoke else C.STATS_EPOCHS
    _, _, result = train_case(N, C.TYPICAL_POINT, C.STATS_NG, epochs, seed=20260911,
                              with_moments=True, verbose_every=max(1, epochs // 10))
    suffix = "_smoke" if smoke else ""
    path = C.DATA_DIR / f"gradient_during_training_N{N}{suffix}.npz"
    atomic_save(path, metadata=metadata(task="gradient_during_training", N=N,
                                       point=C.TYPICAL_POINT, N_g=C.STATS_NG,
                                       epochs=epochs, conditional_on_sampled_y=True,
                                       smoke=smoke), **result)


def print_cases(kind, smoke=False):
    cases = trajectory_cases(smoke) if kind == "trajectory" else budget_cases(smoke)
    for i, case in enumerate(cases):
        print(i, case)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gradient")
    p.add_argument("--N", type=int, choices=C.N_LIST, required=True)
    p.add_argument("--scan", choices=("ng", "beta"), required=True)
    p.add_argument("--repeats", type=int, default=C.GRADIENT_REPEATS)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("trajectory")
    p.add_argument("--case-id", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("budget")
    p.add_argument("--case-id", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("stats")
    p.add_argument("--N", type=int, choices=C.N_LIST, required=True)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("list-cases")
    p.add_argument("kind", choices=("trajectory", "budget"))
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    C.ensure_dirs()
    if args.command == "gradient":
        run_gradient(args.N, args.scan, args.repeats, args.smoke)
    elif args.command == "trajectory":
        run_trajectory(args.case_id, args.smoke)
    elif args.command == "budget":
        run_budget(args.case_id, args.smoke)
    elif args.command == "stats":
        run_stats(args.N, args.smoke)
    else:
        print_cases(args.kind, args.smoke)


if __name__ == "__main__":
    # The eigensolvers in this codebase use float32 by design.  Record the
    # backend in logs because local smoke tests are CPU and Raven runs are GPU.
    print("jax backend:", jax.default_backend(), flush=True)
    main()
