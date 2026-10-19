"""Registered synthetic association-memory protocol. Run with -m aibrain.benchmark."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time
import numpy as np
from .memory import ConnectionMemory, cue, split
from .baselines import HebbianMemory, DeltaMemory, HistoryAttention

SEEDS = [41, 97, 127, 211, 307]
LOADS = [8, 16, 32, 64, 128]
METHODS = {'connection': ConnectionMemory, 'hebbian': HebbianMemory,
           'delta': DeltaMemory, 'attention': HistoryAttention}


def episode(seed, load, dim=64, values=8):
    if load < values or load % values or dim < 16:
        raise ValueError('load must be a positive multiple of values; dim >= 16')
    rng = np.random.default_rng(seed)
    keys, seen = [], set()
    while len(keys) < load:
        indices = tuple(sorted(rng.choice(dim, 8, replace=False).tolist()))
        if indices not in seen:
            key = np.zeros(dim, dtype=np.float32)
            key[list(indices)] = 1
            keys.append(key)
            seen.add(indices)
    labels = np.tile(np.arange(values), load // values)
    rng.shuffle(labels)
    return np.asarray(keys), labels


def corrupt(keys, swaps, rng):
    if not isinstance(swaps, int) or swaps < 0:
        raise ValueError('swaps must be a nonnegative integer')
    out = keys.copy()
    for key in out:
        on, off = np.flatnonzero(key == 1), np.flatnonzero(key == 0)
        if swaps > min(len(on), len(off)):
            raise ValueError('more swaps requested than available bits')
        key[rng.choice(on, swaps, replace=False)] = 0
        key[rng.choice(off, swaps, replace=False)] = 1
    return out


def evaluate(memory, queries, labels):
    return float(np.mean([int(np.argmax(memory.read(q))) == int(y)
                          for q, y in zip(queries, labels)]))


def rewrite_targets(labels):
    updated = labels.copy()
    mask = np.arange(len(labels)) < len(labels) // 4
    updated[mask] = (updated[mask] + 1) % 8
    return updated, mask


def trained(keys, labels):
    models = {name: cls() for name, cls in METHODS.items()}
    eye = np.eye(8)
    for x, y in zip(keys, labels):
        for m in models.values():
            m.write(x, eye[y])
    return models


def mechanism_gate():
    x, y = episode(0, 16)
    m = trained(x, y)['connection']
    original = m.weights.copy()
    before = np.array([m.read(k) for k in x])
    m.clear_activity()
    after = np.array([m.read(k) for k in x])
    iterative_error, direct_error, energy_increase = 0., 0., 0.
    for k in x[:8]:
        direct = m.read(k)
        iterative, energies = m.relax(k)
        iterative_error = max(iterative_error, float(np.max(np.abs(iterative - direct))))
        w = m.weights.astype(np.float64)
        equivalent = w.T @ cue(k) / (m.leak + w.sum(axis=0))
        direct_error = max(direct_error, float(np.max(np.abs(direct - equivalent))))
        energy_increase = max(energy_increase, float(np.max(np.diff(energies))))
    frozen_accuracy = evaluate(ConnectionMemory(), x, y)
    clean_accuracy = evaluate(m, x, y)
    connections_after_queries = m.weights.copy()
    m.erase()
    erased_accuracy = evaluate(m, x, y)
    reconstruct_error = max(float(np.max(np.abs(sum(split(k)) - k))) for k in x)
    passed = {
        'exact_split_reconstruction': reconstruct_error < 1e-12,
        'matched_scalar_core': float(np.ptp(x.mean(axis=1))) == 0,
        'activity_reset_preserves_echo': bool(np.array_equal(before, after)),
        'queries_preserve_connections': bool(np.array_equal(original, connections_after_queries)),
        'erasing_connections_removes_recall': clean_accuracy > erased_accuracy,
        'erased_and_frozen_at_balanced_floor': erased_accuracy == frozen_accuracy == .125,
        'relaxation_matches_matrix': iterative_error < 1e-9 and direct_error < 1e-12,
        'energy_descends': energy_increase < 1e-12,
    }
    return {'passed': passed, 'clean_accuracy': clean_accuracy,
            'erased_accuracy': erased_accuracy, 'frozen_accuracy': frozen_accuracy,
            'reconstruction_max_error': reconstruct_error,
            'iterative_max_error': iterative_error, 'direct_max_error': direct_error,
            'max_energy_increase': energy_increase}


def summarize(rows, dimensions):
    groups = {}
    for r in rows:
        key = tuple(r[d] for d in dimensions)
        groups.setdefault(key, []).append(r['accuracy'])
    return [dict(zip(dimensions, key), mean=float(np.mean(acc)),
                 seed_std=float(np.std(acc, ddof=1)) if len(acc) > 1 else 0,
                 minimum=float(min(acc)), maximum=float(max(acc)))
            for key, acc in sorted(groups.items())]


def run_benchmark(seeds=None, episodes=20, loads=None):
    seeds = SEEDS if seeds is None else list(seeds)
    loads = LOADS if loads is None else list(loads)
    if (not seeds or not loads or not isinstance(episodes, int) or episodes < 1
            or any(not isinstance(n, int) or n < 8 or n % 8 for n in loads)
            or any(not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError('need nonnegative integer seeds, episodes >= 1, loads divisible by 8')
    start = time.perf_counter()
    records, rewrites = [], []
    for seed in seeds:
        for load in loads:
            totals = {}
            for ep in range(episodes):
                derived = int(np.random.SeedSequence([seed, load, ep]).generate_state(1)[0])
                keys, labels = episode(derived, load)
                models = trained(keys, labels)
                m = models['connection']
                frozen = ConnectionMemory()  # never receives the observations
                erased = copy.deepcopy(m)
                erased.erase()
                rng = np.random.default_rng(derived + 1)
                for swaps in (0, 1, 3):
                    queries = corrupt(keys, swaps, rng)
                    scores = {name: evaluate(model, queries, labels) for name, model in models.items()}
                    w = m.weights.astype(np.float64)
                    q = np.asarray([cue(k) for k in queries])
                    direct = q @ w / (m.leak + w.sum(axis=0))
                    scores['direct_equivalent'] = float(np.mean(direct.argmax(axis=1) == labels))
                    scores['raw_query'] = float(np.mean([m.read(k, centered=False).argmax() == y
                                                        for k, y in zip(queries, labels)]))
                    shuffled = queries[rng.permutation(load)]
                    scores['shuffled'] = evaluate(m, shuffled, labels)
                    # Constant output: all keys have equal pooled scalar and balanced labels.
                    scores['pooled'] = float(np.mean(labels == 0))
                    scores['erased'] = evaluate(erased, queries, labels)
                    scores['frozen'] = evaluate(frozen, queries, labels)
                    for name, accuracy in scores.items():
                        totals.setdefault((swaps, name), []).append(accuracy)
            for (swaps, method), acc in totals.items():
                records.append({'seed': seed, 'load': load, 'swaps': swaps,
                                'method': method, 'accuracy': float(np.mean(acc)),
                                'episode_accuracies': acc, 'queries_per_episode': load})
        rewrite_totals = {}
        for ep in range(episodes):
            derived = int(np.random.SeedSequence([seed, 999, ep]).generate_state(1)[0])
            keys, labels = episode(derived, 32)
            original_models = trained(keys, labels)
            updated, changed = rewrite_targets(labels)
            for repeats in (1, 4):
                models = copy.deepcopy(original_models)
                for _ in range(repeats):
                    for x, label in zip(keys[changed], updated[changed]):
                        for model in models.values():
                            model.write(x, np.eye(8)[label])
                for name, model in models.items():
                    for subset, mask in (('changed', changed), ('retained', ~changed)):
                        acc = evaluate(model, keys[mask], updated[mask])
                        rewrite_totals.setdefault((repeats, name, subset), []).append(acc)
        for (repeats, method, subset), acc in rewrite_totals.items():
            rewrites.append({'seed': seed, 'repeats': repeats, 'method': method,
                             'subset': subset, 'accuracy': float(np.mean(acc)),
                             'episode_accuracies': acc})
    return {
        'protocol': {'seeds': seeds, 'episodes_per_seed': episodes, 'loads': loads,
                     'dim': 64, 'active_bits': 8, 'values': 8, 'eta': .5, 'leak': 1.,
                     'query_bit_swaps': [0, 1, 3], 'rewrite_load': 32,
                     'rewrite_fraction': .25, 'training_exposures': 1,
                     'query_target_available_to_model': False,
                     'attention_temperature': 12., 'delta_learning_rate': 1.,
                     'hyperparameters_tuned_on_evaluation': False},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'platform': platform.platform()},
        'state_bytes': {'connection': 2048, 'hebbian': 2048, 'delta': 2048,
                        'attention_by_load': {str(n): n * (64 + 8) * 4 for n in loads},
                        'scope': 'persistent numeric float32 arrays; excludes object overhead and transient buffers'},
        'recall': records, 'rewrite': rewrites,
        'recall_summary': summarize(records, ['load', 'swaps', 'method']),
        'rewrite_summary': summarize(rewrites, ['repeats', 'subset', 'method']),
        'mechanism': mechanism_gate(), 'runtime_seconds': time.perf_counter() - start,
    }


def latency_report():
    keys, labels = episode(1234, 32)
    models = trained(keys, labels)
    rows = {}
    for name, m in models.items():
        for _ in range(20):
            m.read(keys[0])
        start = time.perf_counter()
        for i in range(1000):
            m.read(keys[i % 32])
        rows[name] = (time.perf_counter() - start) * 1e6 / 1000
    m = models['connection']
    start = time.perf_counter()
    for i in range(100):
        m.relax(keys[i % 32])
    rows['relaxation_80_steps'] = (time.perf_counter() - start) * 1e6 / 100
    return {'microseconds_per_query': rows, 'load': 32,
            'scope': 'single local run, Python validation/conversion included; not hardware benchmarking'}


def plot_report(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'connection': '#176f87', 'delta': '#b4472b', 'hebbian': '#7e66aa', 'attention': '#479960'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for swaps, ax in zip((0, 3), axes[0]):
        for method, color in colors.items():
            rows = [r for r in report['recall_summary'] if r['method'] == method and r['swaps'] == swaps]
            ax.errorbar([r['load'] for r in rows], [100*r['mean'] for r in rows],
                        yerr=[100*r['seed_std'] for r in rows], label=method,
                        color=color, marker='o', capsize=2)
        ax.axhline(12.5, color='#aaa', linestyle=':')
        ax.set(title=f'Recall: {swaps} bit swaps in cue', xlabel='Stored associations',
               ylabel='Accuracy (%)', ylim=(0, 105))
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    methods = ['connection', 'direct_equivalent', 'raw_query', 'shuffled', 'pooled', 'erased']
    rows = [next(r for r in report['recall_summary'] if r['load'] == 32 and r['swaps'] == 1 and r['method'] == m)
            for m in methods] if 32 in report['protocol']['loads'] else []
    if rows:
        ax.barh(methods, [100*r['mean'] for r in rows], color=['#176f87']*3 + ['#aaa']*3)
        ax.invert_yaxis()
    ax.set(title='Controls at load 32, one bit swap', xlabel='Accuracy (%)', xlim=(0, 100))
    ax = axes[1, 1]
    names = list(colors)
    positions = np.arange(len(names))
    for offset, subset, shade in ((-.18, 'changed', '#176f87'), (.18, 'retained', '#c5ac5a')):
        rows = [next(r for r in report['rewrite_summary'] if r['repeats'] == 4 and r['subset'] == subset and r['method'] == n)
                for n in names]
        ax.bar(positions + offset, [100*r['mean'] for r in rows], width=.35, label=subset, color=shade)
    ax.set_xticks(positions, names)
    ax.set(title='Rewrite 8 of 32 pairs, four exposures', ylabel='Accuracy (%)', ylim=(0, 105))
    ax.legend(fontsize=8)
    fig.suptitle('AIBrain Gate 1: connection memory versus ordinary association memories', fontsize=14)
    fig.savefig(path)
    svg_path = Path(path)
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/gate1'))
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--loads', type=int, nargs='+', default=LOADS)
    args = p.parse_args()
    try:
        report = run_benchmark(args.seeds, args.episodes, args.loads)
    except ValueError as exc:
        p.error(str(exc))
    report['latency'] = latency_report()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_report(report, args.output / 'gate1.svg')
    print(f"Wrote {args.output}/metrics.json and gate1.svg ({report['runtime_seconds']:.1f}s)")
    print('Mechanism gates:', report['mechanism']['passed'])


if __name__ == '__main__':
    main()
