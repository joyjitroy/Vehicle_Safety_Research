"""Pairwise Wilcoxon signed-rank tests vs B1, with Holm-Bonferroni correction.

Metric: per-sample Brier score contribution (lower = better).
Test (one-sided): is config Ax significantly better than B1?
  d[i] = brier_B1[i] - brier_Ax[i]  (positive = Ax wins on sample i)
  scipy.stats.wilcoxon(d, alternative='greater')

Requires per-sample 'probs' and 'labels' in ablations.json holdout entries.
Run scripts/run_ablations.py first if those keys are missing.

Usage:
    python compute_wilcoxon.py

Output:
    Prints summary table to stdout.
    Writes outputs/results/wilcoxon.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon as scipy_wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_PATH = ROOT / 'outputs' / 'results' / 'ablations.json'
OUT_PATH = ROOT / 'outputs' / 'results' / 'wilcoxon.json'
ALPHA = 0.05


def holm_bonferroni(p_raw: dict, alpha: float = 0.05):
    """Holm-Bonferroni step-down correction. Returns (adjusted_p dict, reject dict)."""
    names = list(p_raw.keys())
    raw = np.array([p_raw[n] for n in names])
    n = len(raw)
    order = np.argsort(raw)
    adj = np.ones(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = float(min(raw[idx] * (n - rank), 1.0))
        a = max(a, running_max)
        running_max = a
        adj[idx] = a
    adjusted = {names[i]: adj[i] for i in range(n)}
    reject = {n: adjusted[n] < alpha for n in names}
    return adjusted, reject


def main():
    with open(ABLATIONS_PATH, encoding='utf-8') as f:
        ablations = json.load(f)

    b1_holdout = ablations.get('B1', {}).get('holdout', {})
    if 'probs' not in b1_holdout:
        raise RuntimeError(
            "Per-sample scores not found in ablations.json under B1.holdout.probs.\n"
            "Re-run:  python run_ablations.py\n"
            "(evaluation-only pass, no retraining — uses existing checkpoints)"
        )

    b1_probs = np.array(b1_holdout['probs'])
    b1_labels = np.array(b1_holdout['labels'])
    b1_brier = (b1_probs - b1_labels) ** 2

    configs = [k for k in ablations if k != 'B1']
    raw_pvals = {}
    results = {}

    for name in configs:
        entry = ablations[name].get('holdout', {})
        if 'probs' not in entry:
            print(f'[WARN] {name}: no per-sample scores in ablations.json — skipping.')
            continue

        ax_probs = np.array(entry['probs'])
        ax_labels = np.array(entry['labels'])

        if len(ax_probs) != len(b1_probs):
            print(f'[WARN] {name}: holdout size mismatch ({len(ax_probs)} vs {len(b1_probs)}) — skipping.')
            continue

        ax_brier = (ax_probs - ax_labels) ** 2
        d = b1_brier - ax_brier

        if np.all(d == 0):
            p_raw = 1.0
            stat = 0.0
        else:
            stat, p_raw = scipy_wilcoxon(d, alternative='greater', zero_method='wilcox')
            stat, p_raw = float(stat), float(p_raw)

        raw_pvals[name] = p_raw
        results[name] = {
            'auprc_b1': float(b1_holdout['auprc']),
            'auprc_ax': float(entry['auprc']),
            'delta_auprc': float(entry['auprc']) - float(b1_holdout['auprc']),
            'mean_brier_b1': float(b1_brier.mean()),
            'mean_brier_ax': float(ax_brier.mean()),
            'delta_brier': float(ax_brier.mean()) - float(b1_brier.mean()),
            'wilcoxon_stat': stat,
            'p_raw': p_raw,
        }

    adj_pvals, reject = holm_bonferroni(raw_pvals, alpha=ALPHA)

    for name in results:
        results[name]['p_adj_holm'] = float(adj_pvals.get(name, float('nan')))
        results[name]['significant_vs_b1'] = bool(reject.get(name, False))

    print(f'\nWilcoxon signed-rank test vs B1 (one-sided, Holm-Bonferroni corrected, α={ALPHA})')
    print(f'Metric: per-sample Brier score contribution (N={len(b1_probs)} holdout samples)')
    print()
    header = f'{"Config":<12}  {"ΔAUPRC":>8}  {"ΔBrier":>8}  {"p_raw":>9}  {"p_adj(HB)":>10}  {"Sig":>5}'
    print(header)
    print('-' * len(header))

    for name in sorted(results, key=lambda x: results[x]['delta_auprc'], reverse=True):
        r = results[name]
        sig = '*' if r['significant_vs_b1'] else ''
        print(
            f'{name:<12}  {r["delta_auprc"]:>+8.4f}  {r["delta_brier"]:>+8.4f}'
            f'  {r["p_raw"]:>9.4f}  {r["p_adj_holm"]:>10.4f}  {sig:>5}'
        )

    print()
    print('* p_adj < 0.05 after Holm-Bonferroni correction (Ax significantly better than B1)')

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nFull results saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
