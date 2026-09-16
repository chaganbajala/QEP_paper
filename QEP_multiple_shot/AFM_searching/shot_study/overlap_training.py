"""Checkpointed expansion of Figure 10's training ensemble and shot range.

Run from QEP_multiple_shot: python -m AFM_searching.shot_study.overlap_training CASE_ID
Original data are read-only. Rerunning a case resumes its own checkpoint.
"""
import argparse
import json
import time

import numpy as np

from ..afm_shots import common
from ..afm_shots.train import Shot_QEP_Training
from . import config as C
from .run import atomic_save, epoch_seed, metadata, scalar_pair


OUTPUT_DIR = C.DATA_DIR / 'trajectories_overlap_2026-09-14'
COMMON_BUDGET = 100_000
SEEDS = 5
# Keep the original three points; append seven uniform points drawn with
# default_rng(20260914), Omega in [0.1,1.5], Delta in [0.5,2.5].
POINTS = C.INITIAL_POINTS + (
    (0.39073567793440267, 1.1664899930858177),
    (0.7428905688845931, 0.9724975887947553),
    (1.16771310144114, 1.0853474197535722),
    (1.3889938137051474, 1.9727996978088573),
    (0.9881055865974885, 1.4554621131779657),
    (0.9351227120364359, 2.4672234788611176),
    (1.0339896133989557, 1.8813281332374763),
)
EPOCHS = {ng: max(C.TRAJECTORY_EPOCHS, COMMON_BUDGET // (2 * ng))
          for ng in C.TRAJECTORY_NG}


def cases():
    # Interleave both system sizes and gradient-shot counts in the queue.
    return [(N, i, ng, seed) for i in range(len(POINTS))
            for seed in range(SEEDS) for ng in C.TRAJECTORY_NG for N in C.N_LIST]


def run_case(case_id, smoke=False, epochs=None, max_new_epochs=None,
             checkpoint_every=200, output_dir=OUTPUT_DIR):
    N, init_id, ng, seed_id = cases()[case_id]
    point = POINTS[init_id]
    target_epochs = epochs if smoke and epochs is not None else (20 if smoke else EPOCHS[ng])
    assert target_epochs > 0 and checkpoint_every > 0
    suffix = '_smoke' if smoke else ''
    name = f'N{N}_init{init_id}_Ng{ng}_seed{seed_id}{suffix}.npz'
    path = output_dir / name
    original = C.DATA_DIR / 'trajectories' / name
    source = path if path.exists() else (original if not smoke and init_id < 3 else None)
    om, de = np.empty(target_epochs + 1), np.empty(target_epochs + 1)
    measured = np.empty(target_epochs)
    gradients = np.empty((target_epochs, 2))
    om[0], de[0] = point
    start, elapsed_prior, origin = 0, 0., ''
    if source is not None:
        with np.load(source, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            expected = dict(N=N, N_g=ng, N_y=C.N_Y, init_id=init_id,
                            seed_id=seed_id, beta=C.BETA, learning_rate=C.LEARNING_RATE)
            for key, value in expected.items():
                assert meta[key] == value, (source, key, meta[key], value)
            np.testing.assert_array_equal(meta['point'], point)
            if source == path:
                assert meta['smoke'] == smoke
                # The user shortened the requested overlap while the 200-epoch
                # preview prefixes were running. Reuse those prefixes, but do
                # not truncate any recorded epochs or accept arbitrary plans.
                previous_target = max(C.TRAJECTORY_EPOCHS, 400_000 // (2 * ng))
                assert meta['epochs'] == target_epochs or (
                    not smoke and meta.get('common_gradient_budget') == 400_000
                    and meta['epochs'] == previous_target
                )
            start = len(data['measured_AM2'])
            assert 0 <= start <= target_epochs
            assert data['Omega'].shape == data['Delta'].shape == (start + 1,)
            assert data['sampled_gradient'].shape == (start, 2)
            om[:start+1], de[:start+1] = data['Omega'], data['Delta']
            measured[:start], gradients[:start] = data['measured_AM2'], data['sampled_gradient']
            elapsed_prior = float(data['elapsed_seconds'])
            origin = meta.get('original_source', str(source) if source == original else '')
    if (start == target_epochs and source == path
            and meta.get('common_gradient_budget') == COMMON_BUDGET):
        print(f'Already complete: {path}', flush=True)
        return path

    started = time.monotonic()
    completed = start

    def save():
        arrays = dict(Omega=om[:completed+1], Delta=de[:completed+1],
                      measured_AM2=measured[:completed], sampled_gradient=gradients[:completed])
        assert all(np.isfinite(a).all() for a in arrays.values())
        atomic_save(path, metadata=metadata(
            task='trajectory_overlap', N=N, N_g=ng, init_id=init_id, seed_id=seed_id,
            point=point, epochs=target_epochs, completed_epochs=completed,
            complete=completed == target_epochs, smoke=smoke,
            common_gradient_budget=COMMON_BUDGET, initial_points=10, seeds_per_point=SEEDS,
            original_source=origin, base_seed=seed_id + 1000 * init_id,
            alignment='measured_AM2[k] is before update, at Omega[k], Delta[k]',
        ), elapsed_seconds=np.asarray(elapsed_prior + time.monotonic()-started), **arrays)

    if start == target_epochs:
        save()
        return path
    nn, params, grad_method = common.build_model_and_grad(N, float(om[start]), float(de[start]))
    trainer = Shot_QEP_Training(C.LEARNING_RATE)
    stop = target_epochs if max_new_epochs is None else min(target_epochs, start + max_new_epochs)
    assert stop > start
    print(f'case={case_id} N={N} init={init_id} Ng={ng} seed={seed_id}: '
          f'epochs {start} -> {stop} (target {target_epochs})', flush=True)
    for k in range(start, stop):
        y, grad = grad_method(params, nn, C.TARGET, N_y=C.N_Y, N_g=ng,
                              seed=epoch_seed(seed_id + 1000 * init_id, k))
        gradients[k] = scalar_pair(grad)
        measured[k] = float(np.real(np.asarray(y)))
        params = trainer.update(params, grad)
        om[k+1], de[k+1] = scalar_pair(params)
        completed = k+1
        if completed % checkpoint_every == 0 or completed == stop:
            save()
            duration = time.monotonic()-started
            print(f'epoch {completed}/{target_epochs}, new seconds/epoch='
                  f'{duration/(completed-start):.4f}, AFM2={measured[k]:.6f}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, help='Smoke-test epoch target only')
    parser.add_argument('--max-new-epochs', type=int, help='Checkpoint and stop early for timing/resume tests')
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--groups', type=int, help='Treat case_id as group ID and partition selected cases into groups')
    parser.add_argument('--N', type=int, choices=C.N_LIST, help='Limit production to one system size')
    args = parser.parse_args()
    if not 0 <= args.case_id < len(cases()):
        parser.error(f'case_id must be 0..{len(cases())-1}')
    if args.epochs is not None and not args.smoke:
        parser.error('--epochs is only allowed for smoke tests')
    if args.groups is not None:
        selected = [i for i, case in enumerate(cases()) if args.N is None or case[0] == args.N]
        if not 1 <= args.groups <= len(selected) or not 0 <= args.case_id < args.groups:
            parser.error('group ID must be smaller than the positive group count')
        case_ids = selected[args.case_id::args.groups]
    else:
        case_ids = [args.case_id]
        if args.N is not None and cases()[args.case_id][0] != args.N:
            parser.error('case_id does not match --N')
    for case_id in case_ids:
        run_case(case_id, args.smoke, args.epochs, args.max_new_epochs, args.checkpoint_every)


if __name__ == '__main__':
    main()
