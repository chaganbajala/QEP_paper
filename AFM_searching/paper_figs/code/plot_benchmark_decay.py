#!/usr/bin/env python
"""Dense N=1,3,5 demonstrations and RMS decay from checked 1-2-5 windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import figstyle as F
from sample_benchmark_windows import BENCHMARKS, SCHEDULES, amplitude_statistics

PAPER = Path(__file__).resolve().parents[1]
NS = (1, 3, 5, 7, 9)


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def load_windows(directory):
    windows, summary = {}, {}
    for N in NS:
        windows[N], summary[str(N)] = {}, {'windows': {}, 'missing': []}
        for T in BENCHMARKS:
            path = directory / f'benchmark_chain_N{N}_T{T}.npz'
            if not path.exists():
                summary[str(N)]['missing'].append(T)
                continue
            data = read(path)
            if not bool(data.get('complete', False)):
                summary[str(N)]['missing'].append(T)
                continue
            meta = json.loads(str(data['meta']))
            assert meta['N'] == N and meta['benchmark'] == T
            stored = json.loads(str(data['statistics']))
            for s in SCHEDULES:
                assert np.isfinite(data[f'g_{s}_AE']).all()
                check = amplitude_statistics(data['TL'], data[f'g_{s}_AE'])
                np.testing.assert_allclose(check['rms'], stored[s]['rms'])
            data['stats'] = stored
            audit_path=directory/'window_stability'/f'window_stability_chain_N{N}_T{T}.json'
            if audit_path.exists():
                audit=json.loads(audit_path.read_text())
                assert audit['N']==N and audit['T']==T and audit['schedule']=='stiff'
                assert audit['benchmark_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                audit_data=directory/'window_stability'/Path(audit['data_source']).name
                assert audit['data_sha256']==hashlib.sha256(audit_data.read_bytes()).hexdigest()
                assert bool(read(audit_data)['complete'])
                baseline=audit['records'][0]
                np.testing.assert_allclose(baseline['trials'][0]['rms'],stored['stiff']['rms'])
                data['window_audit']=audit
            windows[N][T] = data
            summary[str(N)]['windows'][str(T)] = dict(
                source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                statistics=stored, convergence_checks=json.loads(str(data['convergence_checks'])))
            if 'window_audit' in data:
                summary[str(N)]['windows'][str(T)]['window_stability_audit']=data['window_audit']
    return windows, summary


def numerically_resolved(data, schedule):
    st = data['stats'][schedule]
    return bool(st['sampling_resolved'] and st['solver_spotcheck_resolved'])


def common_benchmarks(windows):
    return [T for T in BENCHMARKS if T >= 20 and all(
        T in windows[N] and all(numerically_resolved(windows[N][T], s) for s in SCHEDULES)
        for N in NS)]


def component_curve(windows, schedule, component=None):
    values = []
    for T in BENCHMARKS:
        d = windows.get(T)
        if d is None or not numerically_resolved(d, schedule):
            values.append(np.nan)
            continue
        rms = np.array(d['stats'][schedule]['rms'])
        values.append(float(rms[component]) if component is not None
                      else float(np.linalg.norm(rms) / np.linalg.norm(d['g_ED'])))
    return np.array(values)


def relative_error_curve(windows, schedule):
    """Total RMS deviation from ED, distinct from the centered oscillation."""
    values = []
    for T in BENCHMARKS:
        d = windows.get(T)
        if d is None or not numerically_resolved(d, schedule):
            values.append(np.nan)
            continue
        bias = d[f'g_{schedule}_AE'] - d['g_ED']
        values.append(float(np.sqrt(np.mean(np.sum(bias ** 2, axis=1))) / np.linalg.norm(d['g_ED'])))
    return np.array(values)


def display_curve(y, benchmarks):
    return np.where(np.isin(BENCHMARKS, benchmarks), y, np.nan)


def window_caution(data, schedule, component=None, kind='centered'):
    """Use measured sensitivity for audited points; retain original screen otherwise."""
    if schedule=='stiff' and 'window_audit' in data:
        record=data['window_audit']['records'][0]
        if component is not None:
            return bool(record['component_sensitive'][component] or
                        record['component_max_half_grid_change'][component]>.05)
        return bool(record[f'{kind}_sensitive'] or max(record['component_max_half_grid_change'])>.05)
    st=data['stats'][schedule]
    if component is not None:
        return min(st['peaks'][component],st['troughs'][component])<3
    return not st['resolved_cycles']


def draw_estimates(ax, values, windows, schedule, color, benchmarks, component=None, kind='centered'):
    """Hollow: window caution; bars: tested window range, not statistical error."""
    x = np.asarray(BENCHMARKS)
    y = display_curve(values, benchmarks)
    finite = np.isfinite(y)
    limited = np.array([T in windows and window_caution(windows[T],schedule,component,kind)
                        for T in BENCHMARKS])
    ax.plot(x, y, '-', color=color, lw=1.3)
    if schedule=='stiff':
        for i,T in enumerate(BENCHMARKS):
            if not finite[i] or 'window_audit' not in windows[T]:continue
            record=windows[T]['window_audit']['records'][0]
            bounds=np.array(record['component_range'])[:,component] if component is not None else np.array(record[f'{kind}_range'])
            ax.errorbar(T,y[i],yerr=np.array([[max(0.,y[i]-bounds[0])],[max(0.,bounds[1]-y[i])]]),
                        fmt='none',ecolor=color,elinewidth=1.,capsize=3,alpha=.8,zorder=3)
    for mask, face in ((finite & ~limited, color), (finite & limited, 'white')):
        ax.plot(x[mask], y[mask], 'o', color=color, markerfacecolor=face,
                markeredgewidth=1., ms=4, zorder=4)


def shared_decay_limits(windows, benchmarks):
    limits = []
    for j in range(3):
        values = []
        for N in NS:
            for s in SCHEDULES:
                curves = [component_curve(windows[N], s, j)] if j < 2 else [
                    relative_error_curve(windows[N], s), component_curve(windows[N], s)]
                for y in curves:
                    y = display_curve(y, benchmarks)
                    values.extend(y[np.isfinite(y) & (y > 0)])
                if s=='stiff':
                    for T in benchmarks:
                        if T not in windows[N] or 'window_audit' not in windows[N][T]:continue
                        r=windows[N][T]['window_audit']['records'][0]
                        bounds=np.array(r['component_range'])[:,j] if j<2 else np.r_[r['total_range'],r['centered_range']]
                        values.extend(bounds[bounds>0])
        if not values:
            limits.append((1e-4, 1.))
            continue
        lo, hi = np.log10([min(values), max(values)])
        pad = max(.12, .06 * (hi - lo))
        limits.append((float(10 ** (lo - pad)), float(10 ** (hi + pad))))
    return limits


def save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=PAPER / 'data/bias_benchmarks')
    ap.add_argument('--outdir', type=Path, default=PAPER / 'figs')
    ap.add_argument('--example-max', type=float, default=200., help='Requested dense-example range; must be complete to render.')
    ap.add_argument('--coverage', choices=('common', 'available'), default='common',
                    help='Use the same completed numerical benchmarks across all five N by default.')
    ap.add_argument('--summary', type=Path, help='Optional separate provenance file for supplementary renders.')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    windows, summary = load_windows(args.data)
    display_T = common_benchmarks(windows) if args.coverage == 'common' else [T for T in BENCHMARKS if T >= 20]
    if not display_T:
        raise ValueError('No common completed benchmarks yet; use --coverage available for labeled previews.')
    ylimits = shared_decay_limits(windows, display_T)
    xlimits = (min(display_T) / 1.25, max(display_T) * 1.25)
    summary['display'] = dict(coverage=args.coverage, benchmarks=display_T,
                             shared_xlimits=xlimits, shared_ylimits=ylimits,
                             window_caution='hollow: >10% window sensitivity where audited; otherwise component cycle screen',
                             window_bars='range from +/-12.5% shifts and +/-25% widths; not confidence intervals',
                             window_audit_scope='originally cycle-flagged N5/T500 and N9/T20, stiff only')
    F.use_style()
    with plt.rc_context({'figure.constrained_layout.use': False}):
        # Standalone copies of the upper two panels, with identical layout.
        for N in (1, 3, 5):
            suffix = '' if args.example_max == 60. else f'_T{args.example_max:g}'
            path = args.data / f'benchmark_chain_N{N}_example{suffix}.npz'
            if not path.exists():
                continue
            raw = read(path)
            if not bool(raw.get('complete', False)):
                print('Dense example still incomplete:', path, flush=True)
                continue
            summary[f'raw_example_N{N}'] = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
            fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.79, wspace=.24)
            for j, symbol in enumerate((r'\Omega', r'\Delta')):
                ax = axes[j]
                for s in SCHEDULES:
                    ax.plot(raw['TL'], raw[f'g_{s}_AE'][:, j], color=F.SCHEDULE_COLOR[s], lw=.9)
                ax.axhline(raw['g_ED'][j], color=F.INK, lw=1)
                ax.set_xlim(0, args.example_max)
                for center in (20, 50):
                    if center in windows[N]:
                        t = windows[N][center]['TL']
                        ax.axvspan(t[0], t[-1], color='#777777', alpha=.10, zorder=0)
                        ax.text(center, .98, f'$W_{{{center}}}$', transform=ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=7, color='#666666')
                F.finish(ax, 'evolution duration $T$ (linear)', f'$g_{{{symbol}}}$',
                         f'({chr(97+j)}) Resolved oscillations', legend=False)
            handles = [Line2D([], [], color=F.SCHEDULE_COLOR[s], label=F.SCHEDULE_LABEL[s]) for s in SCHEDULES]
            handles.append(Line2D([], [], color=F.INK, label='ED reference'))
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .05), ncol=3)
            fig.suptitle(f'Open chain $N={N}$: gradient oscillations through $T={args.example_max:g}$',
                         y=.98, fontsize=11)
            save(fig, args.outdir / f'grad_oscillations_chain_N{N}.pdf')

        example_path = args.data / 'benchmark_chain_N3_example.npz'
        if args.example_max != 60.:
            example_path = args.data / f'benchmark_chain_N3_example_T{args.example_max:g}.npz'
        if example_path.exists():
            example = read(example_path)
            assert bool(example['complete'])
            summary['example'] = dict(source=str(example_path), sha256=hashlib.sha256(example_path.read_bytes()).hexdigest())
            fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.2))
            fig.subplots_adjust(left=.11, right=.98, bottom=.20, top=.89, hspace=.56, wspace=.29)
            for j, symbol in enumerate((r'\Omega', r'\Delta')):
                ax = axes[0, j]
                for s in SCHEDULES:
                    ax.plot(example['TL'], example[f'g_{s}_AE'][:, j], color=F.SCHEDULE_COLOR[s], lw=.9)
                ax.axhline(example['g_ED'][j], color=F.INK, lw=1, label='ED')
                ax.set_xlim(0, args.example_max)
                for center in (20, 50):
                    if center in windows[3]:
                        TL = windows[3][center]['TL']
                        ax.axvspan(TL[0], TL[-1], color='#777777', alpha=.10, zorder=0)
                        ax.text(center, .98, f'$W_{{{center}}}$', transform=ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=7, color='#666666')
                F.finish(ax, 'evolution duration $T$ (linear)', f'$g_{{{symbol}}}$',
                         f'({chr(97+j)}) Resolved oscillations', legend=False)
                ax = axes[1, j]
                for s in SCHEDULES:
                    ax.plot(BENCHMARKS, component_curve(windows[3], s, j), 'o-',
                            color=F.SCHEDULE_COLOR[s], ms=3.5)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(15, 2500)
                F.finish(ax, 'benchmark $T$', f'RMS oscillation $A_{{{symbol}}}(T)$',
                         f'({chr(99+j)}) Local amplitude decay', legend=False)
            handles = [Line2D([], [], color=F.SCHEDULE_COLOR[s], label=F.SCHEDULE_LABEL[s]) for s in SCHEDULES]
            handles.append(Line2D([], [], color=F.INK, label='ED reference (top)'))
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .073), ncol=3)
            fig.suptitle('$N=3$: show the oscillations once, then measure local amplitude', y=.98, fontsize=11)
            missing = summary['3']['missing']
            note = 'Local RMS about the window mean; numerically unresolved windows are omitted.'
            if missing:
                note += '\nPending benchmarks: ' + ', '.join(map(str, missing))
            fig.text(.54, .018, note, ha='center', fontsize=8)
            save(fig, args.outdir / 'grad_benchmark_example_N3.pdf')

        # Separate three-panel decay figure for each chain, as requested.
        for N in NS:
            fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))
            fig.subplots_adjust(left=.075, right=.99, bottom=.30, top=.79, wspace=.38)
            for s in SCHEDULES:
                color = F.SCHEDULE_COLOR[s]
                for j in (0, 1):
                    draw_estimates(axes[j], component_curve(windows[N], s, j),
                                   windows[N], s, color, display_T,component=j)
                draw_estimates(axes[2], relative_error_curve(windows[N], s),
                               windows[N], s, color, display_T,kind='total')
                axes[2].plot(BENCHMARKS, display_curve(component_curve(windows[N], s), display_T),
                             '--', color=color, lw=1.1, alpha=.75)
            labels = [r'Local RMS variation $A_\Omega(T)$', r'Local RMS variation $A_\Delta(T)$',
                      r'RMS / $\|g_{\rm ED}\|$']
            titles = [r'(a) $\Omega$ amplitude', r'(b) $\Delta$ amplitude', '(c) Relative gradient error']
            for ax, label, title, ylim in zip(axes, labels, titles, ylimits):
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(*xlimits)
                ax.set_ylim(*ylim)
                ax.set_xticks(display_T, [f'{T:g}' for T in display_T])
                ax.xaxis.set_minor_formatter(NullFormatter())
                F.finish(ax, 'benchmark $T$', label, title, legend=False)
            handles = [Line2D([], [], color=F.SCHEDULE_COLOR[s], marker='o', ms=3.5,
                              label=F.SCHEDULE_LABEL[s]) for s in SCHEDULES]
            handles += [Line2D([], [], color=F.INK, label='(c) total RMS error'),
                        Line2D([], [], color=F.INK, ls='--', label='(c) centered oscillation'),
                        Line2D([], [], color=F.INK, ls='', marker='o', markerfacecolor='white', label='Hollow: window caution'),
                        Line2D([], [], color=F.INK, marker='|', label='Bars: tested window range')]
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .08), ncol=3)
            fig.suptitle(f'Open chain $N={N}$: local benchmark amplitude decay', y=.98, fontsize=11)
            note = ('Shared completed benchmarks: ' if args.coverage == 'common' else 'Available benchmark data: ')
            note += ', '.join(f'{T:g}' for T in display_T) + '. Window-range bars are not confidence intervals.'
            if args.coverage == 'available' and summary[str(N)]['missing']:
                note += '\nPending benchmarks: ' + ', '.join(map(str, summary[str(N)]['missing']))
            fig.text(.52, .012, note, ha='center', fontsize=8)
            save(fig, args.outdir / f'grad_benchmark_decay_chain_N{N}.pdf')

        fig, axes = plt.subplots(1, 2, figsize=(7.7, 4.1), sharey=True)
        fig.subplots_adjust(left=.11, right=.98, bottom=.29, top=.84, wspace=.17)
        colors = ['#0072B2', '#D55E00', '#009E73', '#CC0077', '#6F4C9B']
        shown = []
        for N, color in zip(NS, colors):
            visible = False
            for ax, s in zip(axes, SCHEDULES):
                y = display_curve(component_curve(windows[N], s), display_T)
                if np.isfinite(y).any():
                    draw_estimates(ax, y, windows[N], s, color, display_T)
                    visible = True
            if visible:
                shown.append(Line2D([], [], color=color, marker='o', ms=3.5, label=f'$N={N}$'))
        for ax, s in zip(axes, SCHEDULES):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(*xlimits)
            ax.set_xticks(display_T, [f'{T:g}' for T in display_T])
            ax.xaxis.set_minor_formatter(NullFormatter())
            F.finish(ax, 'benchmark $T$', title=F.SCHEDULE_LABEL[s], legend=False)
        axes[0].set_ylabel(r'$\sqrt{\langle\|g-\bar g_W\|^2\rangle_W}/\|g_{\rm ED}\|$')
        fig.legend(handles=shown, loc='lower center', bbox_to_anchor=(.54, .11), ncol=5)
        fig.suptitle('Amplitude decay from local 1–2–5 benchmark windows', y=.98, fontsize=11)
        incomplete = [str(N) for N in NS if summary[str(N)]['missing']]
        footer = ('Shared completed benchmarks: ' if args.coverage == 'common' else 'Available benchmarks: ')
        footer += ', '.join(f'{T:g}' for T in display_T) + '. Hollow: window caution; bars: tested window range.'
        if incomplete and args.coverage == 'available':
            footer += '\nScans still incomplete: N=' + ', '.join(incomplete)
        fig.text(.54, .015, footer, ha='center', fontsize=8)
        save(fig, args.outdir / 'grad_benchmark_decay_chains.pdf')
    summary_path = args.summary or args.data / 'benchmark_amplitude_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
