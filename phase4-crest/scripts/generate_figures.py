"""Generate all CREST paper figures — publication quality for IEEE T-IV.

Column widths (IEEE Transactions):
  Single column : 3.50 in  →  used for focused single-panel figures
  Double column : 7.20 in  →  used for multi-panel / comparison figures

All figures exported as:
  outputs/figures/<name>.png  — 300 DPI raster for Word/LaTeX preview
  outputs/figures/<name>.pdf  — vector PDF for final LaTeX submission

Style contract
  • Font   : serif (Computer Modern in LaTeX, DejaVu Serif here)
  • Sizes  : axes labels 9 pt, tick labels 8 pt, legend 8 pt, title 10 pt
  • Lines  : 1.8 pt default, 1.2 pt secondary
  • Grid   : grey, alpha 0.3, below data
  • Colour : colorblind-safe palette (IBM Design — blue, orange, green, purple, red)

Reads:
  outputs/results/ablations.json
  outputs/results/lead_time.json
  outputs/results/b0_baseline.json
  outputs/checkpoints/baseline.pt   (B1 model)
  outputs/checkpoints/platt.json    (B1 Platt coefficients)
  outputs/features/holdout.npz      (B1 holdout features)

Produces 11 figures (reliability_diagram removed — see V3.4 change note):
  pr_curve.png / .pdf
  pr_curve_multi.png / .pdf
  ablation_bar_chart.png / .pdf
  lead_time_histogram.png / .pdf
  rsu_sweep_plot.png / .pdf
  learning_curves.png / .pdf
  split_comparison.png / .pdf
  safedriver_score_distribution.png / .pdf
  detection_uncertainty_sensitivity.png / .pdf
  data_split_diagram.png / .pdf
  v2x_simulation_schematic.png / .pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from crest.models.architecture import CRESTConfig, CRESTModel

RESULTS_DIR = ROOT / 'outputs' / 'results'
CKPT_DIR    = ROOT / 'outputs' / 'checkpoints'
FEATURES_DIR = ROOT / 'outputs' / 'features'
MODEL_YAML  = ROOT / 'configs' / 'model.yaml'
FIG_DIR     = ROOT / 'outputs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIM    = 8
BATCH_SIZE = 64
DPI        = 300

# IEEE column widths (inches)
W1 = 3.50   # single column
W2 = 7.20   # double column

# Colorblind-safe palette (IBM Design Library)
C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN  = '#009E73'
C_PURPLE = '#CC79A7'
C_RED    = '#D55E00'
C_GREY   = '#999999'
C_LBLUE  = '#56B4E9'

# ── Global rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#CCCCCC',
    'lines.linewidth':   1.8,
    'lines.markersize':  5,
    'axes.grid':         True,
    'grid.color':        '#BBBBBB',
    'grid.alpha':        0.35,
    'grid.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        DPI,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_fig(fig, stem: str):
    for ext in ('png', 'pdf'):
        p = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(p)
        print(f'  Saved {p}')
    plt.close(fig)


def load_model_config(path: Path) -> CRESTConfig:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg['architecture']
    input_dim = len(arch['input_features'])
    return CRESTConfig(
        ego_input_dim=input_dim,
        ego_hidden_dims=arch['ego_encoder']['hidden_dims'],
        ego_output_dim=arch['ego_encoder']['output_dim'],
        neighbor_input_dim=input_dim,
        neighbor_hidden_dims=arch['neighbor_encoder']['hidden_dims'],
        neighbor_output_dim=arch['neighbor_encoder']['output_dim'],
        map_input_dim=arch['map_encoder']['input_dim'],
        map_hidden_dims=arch['map_encoder']['hidden_dims'],
        map_output_dim=arch['map_encoder']['output_dim'],
        fusion_hidden_dims=arch['fusion']['hidden_dims'],
        dropout=arch.get('dropout', 0.1),
    )


def get_b1_holdout_probs_labels():
    """Recompute raw B1 probabilities and labels on the holdout split."""
    device = torch.device('cpu')
    cfg    = load_model_config(MODEL_YAML)
    model  = CRESTModel(cfg).to(device)
    model.load_state_dict(
        torch.load(CKPT_DIR / 'baseline.pt', map_location=device, weights_only=True))
    model.eval()

    data     = np.load(FEATURES_DIR / 'holdout.npz')
    ego      = np.concatenate([data['ego_pos'],       data['ego_neg']],      axis=0)
    neighbors = np.concatenate([data['neighbors_pos'], data['neighbors_neg']], axis=0)
    mask     = np.concatenate([data['mask_pos'],      data['mask_neg']],     axis=0)
    labels   = np.concatenate([data['labels_pos'],    data['labels_neg']],   axis=0)
    map_feat = np.zeros((len(labels), MAP_DIM), dtype=np.float32)

    ds     = TensorDataset(torch.from_numpy(ego), torch.from_numpy(neighbors),
                           torch.from_numpy(mask), torch.from_numpy(map_feat),
                           torch.from_numpy(labels))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    probs = []
    with torch.no_grad():
        for e, n, m, mp, _ in loader:
            out = model(e.to(device), n.to(device), m.to(device), mp.to(device))
            probs.extend(out.cpu().numpy().tolist())
    return np.clip(np.array(probs), 1e-6, 1 - 1e-6), labels


def apply_platt(probs, platt):
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(platt['a'] * logits + platt['b'])))


# ── Figure 1: PR curve (B1 vs B0 — simple version) ───────────────────────────

def plot_pr_curve():
    print('\n[1/11] PR curve (B1 vs B0) …')
    b0             = load_json(RESULTS_DIR / 'b0_baseline.json')
    probs_b1, lbl  = get_b1_holdout_probs_labels()
    abl            = load_json(RESULTS_DIR / 'ablations.json') or {}

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.85))

    p1, r1, _ = precision_recall_curve(lbl, probs_b1)
    b1_auprc  = abl.get('B1', {}).get('holdout', {}).get('auprc', 0.678)
    ax.plot(r1, p1, color=C_BLUE, lw=1.8,
            label=f'CREST B1 — ego only  (AUPRC = {b1_auprc:.3f})')

    if b0 is not None:
        b0_labels = np.array([e['label']      for e in b0['events']])
        b0_scores = np.clip(np.array([e['risk_score'] for e in b0['events']]), 1e-6, 1 - 1e-6)
        p0, r0, _ = precision_recall_curve(b0_labels, b0_scores)
        b0_auprc  = b0.get('metrics', {}).get('auprc', 0.500)
        ax.plot(r0, p0, color=C_ORANGE, lw=1.4, linestyle='--',
                label=f'B0 — analytical TTC  (AUPRC = {b0_auprc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curve (holdout)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, 'pr_curve')


# ── Figure 2: PR curve multi (preferred for paper) ───────────────────────────

def plot_pr_curve_multi():
    print('\n[2/11] PR curve multi (B1 vs B0 with AUPRC labels) …')
    abl            = load_json(RESULTS_DIR / 'ablations.json') or {}
    b0             = load_json(RESULTS_DIR / 'b0_baseline.json')
    probs_b1, lbl  = get_b1_holdout_probs_labels()

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.90))

    p1, r1, _ = precision_recall_curve(lbl, probs_b1)
    b1_auprc  = abl.get('B1', {}).get('holdout', {}).get('auprc', 0.678)
    ax.plot(r1, p1, color=C_BLUE, lw=1.8,
            label=f'B1 ego-only  (AUPRC = {b1_auprc:.3f})')

    if b0 is not None:
        b0_labels = np.array([e['label']      for e in b0['events']])
        b0_scores = np.clip(np.array([e['risk_score'] for e in b0['events']]), 1e-6, 1 - 1e-6)
        p0, r0, _ = precision_recall_curve(b0_labels, b0_scores)
        b0_auprc  = b0.get('metrics', {}).get('auprc', 0.500)
        ax.plot(r0, p0, color=C_GREY, lw=1.4, linestyle='--',
                label=f'B0 TTC baseline  (AUPRC = {b0_auprc:.3f})')

    # Annotate AUPRC values on curves
    ax.annotate(f'AUPRC = {b1_auprc:.3f}',
                xy=(0.55, 0.68), fontsize=7.5, color=C_BLUE,
                xytext=(0.60, 0.75),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.8))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curves — Holdout Split')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(fig, 'pr_curve_multi')


# ── Figure 3: Ablation bar chart ──────────────────────────────────────────────

def plot_ablation_bar_chart():
    print('\n[3/11] Ablation bar chart …')
    ablations = load_json(RESULTS_DIR / 'ablations.json')
    if ablations is None:
        print('  SKIP — ablations.json not found.')
        return

    order  = ['B1', 'A1', 'A2', 'A3', 'A4_r150', 'A4_r300', 'A4_r500', 'F', 'A6', 'A7']
    names  = [n for n in order if n in ablations]
    auprcs = [ablations[n]['holdout']['auprc'] for n in names]

    # B1 = anchor (grey), primary chain = blue, optional inference-time = orange
    def bar_color(n):
        if n == 'B1':
            return C_GREY
        if n in ('A6', 'A7'):
            return C_ORANGE
        return C_BLUE

    colors = [bar_color(n) for n in names]

    # x-axis labels: prettier
    xlabels = {
        'B1': 'B1\nEgo only', 'A1': 'A1\nMap', 'A2': 'A2\nWeather',
        'A3': 'A3\nV2V', 'A4_r150': 'A4\n150 m', 'A4_r300': 'A4\n300 m',
        'A4_r500': 'A4\n500 m', 'F': 'F\nAll fused',
        'A6': 'A6\nPerception', 'A7': 'A7\nSafeDriver IQ',
    }
    labels = [xlabels.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(W2, 3.6))
    bars = ax.bar(labels, auprcs, color=colors, width=0.65,
                  edgecolor='white', linewidth=0.8)

    # Annotate each bar with exact value
    for bar, val in zip(bars, auprcs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # Horizontal reference line at B1
    b1_val = ablations['B1']['holdout']['auprc']
    ax.axhline(b1_val, color=C_GREY, lw=1.0, linestyle=':', alpha=0.8,
               label=f'B1 baseline ({b1_val:.3f})')

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=C_GREY,   label='B1 baseline (ego-only)'),
        mpatches.Patch(color=C_BLUE,   label='Primary ablation chain'),
        mpatches.Patch(color=C_ORANGE, label='Optional inference-time (A6, A7)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=7.5)

    ax.set_ylabel('Holdout AUPRC')
    ax.set_title('Ablation Study — Holdout AUPRC by Input Configuration')
    ax.set_ylim(0.60, 0.87)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    fig.tight_layout()
    save_fig(fig, 'ablation_bar_chart')


# ── Figure 4: Lead-time histogram ─────────────────────────────────────────────

def plot_lead_time_histogram(far_key: str = 'far_5pct'):
    print('\n[4/11] Lead-time histogram …')
    lead_time = load_json(RESULTS_DIR / 'lead_time.json')
    if lead_time is None:
        print('  SKIP — lead_time.json not found.')
        return

    values = [e[f'{far_key}_lead_s']
              for e in lead_time['events']
              if e.get(f'{far_key}_lead_s') is not None]
    if not values:
        print(f'  SKIP — no detections at {far_key}.')
        return

    values = np.array(values)
    n_det  = len(values)
    mean_t = values.mean()
    med_t  = np.median(values)

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.82))
    ax.hist(values, bins=20, color=C_GREEN, edgecolor='white', linewidth=0.7, alpha=0.9)

    ax.axvline(med_t, color=C_RED,    lw=1.5, linestyle='--',
               label=f'Median  {med_t:.2f} s')
    ax.axvline(mean_t, color=C_ORANGE, lw=1.5, linestyle='-.',
               label=f'Mean  {mean_t:.2f} s')

    ax.set_xlabel('Lead time before hazard onset (s)')
    ax.set_ylabel('Event count')
    ax.set_title(f'Lead-Time Distribution at 5% FAR\n'
                 f'({n_det} detected / 838 total holdout events, '
                 f'detection rate {n_det/838*100:.1f}%)')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'lead_time_histogram')


# ── Figure 5: RSU sweep ───────────────────────────────────────────────────────

def plot_rsu_sweep():
    print('\n[5/11] RSU coverage-radius sweep …')
    ablations = load_json(RESULTS_DIR / 'ablations.json')
    if ablations is None:
        print('  SKIP — ablations.json not found.')
        return

    radii = [150, 300, 500]
    keys  = [f'A4_r{r}' for r in radii]
    if not all(k in ablations for k in keys):
        print('  SKIP — not all A4 configs present.')
        return

    auprcs = [ablations[k]['holdout']['auprc'] for k in keys]

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.82))
    ax.plot(radii, auprcs, 'o-', color=C_PURPLE, lw=1.8, markersize=7,
            markerfacecolor='white', markeredgewidth=2)

    # Annotate each point
    for r, v in zip(radii, auprcs):
        ax.annotate(f'{v:.3f}', xy=(r, v),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color=C_PURPLE, fontweight='bold')

    ax.set_xlabel('RSU coverage radius (m)')
    ax.set_ylabel('Holdout AUPRC')
    ax.set_title('RSU Coverage Radius Sensitivity (A4)')
    ax.set_xticks(radii)
    ax.set_xticklabels(['150 m', '300 m', '500 m'])
    ax.set_ylim(0.66, 0.74)
    fig.tight_layout()
    save_fig(fig, 'rsu_sweep_plot')


# ── Figure 6: Learning curves ─────────────────────────────────────────────────

def plot_learning_curves():
    print('\n[6/11] Learning curves …')
    ablations = load_json(RESULTS_DIR / 'ablations.json')
    if ablations is None:
        print('  SKIP — ablations.json not found.')
        return

    cfg_colors = {'B1': C_GREY, 'A2': C_GREEN, 'F': C_PURPLE, 'A7': C_ORANGE}
    cfg_names  = {'B1': 'B1 (ego only)', 'A2': 'A2 (weather/traffic)',
                  'F':  'F (full fusion)', 'A7': 'A7 (SafeDriver IQ)'}
    configs = [c for c in ['B1', 'A2', 'F', 'A7'] if c in ablations]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.2))

    for cfg in configs:
        hist = ablations[cfg].get('history', [])
        if not hist:
            continue
        epochs = [h['epoch'] for h in hist]
        losses = [h['loss']           for h in hist]
        cals   = [h['cal']['auprc']   for h in hist]
        kw = dict(color=cfg_colors[cfg], label=cfg_names[cfg])
        ax1.plot(epochs, losses, 'o-', **kw)
        ax2.plot(epochs, cals,   'o-', **kw)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (BCE)')
    ax1.set_title('Training Loss vs. Epoch')
    ax1.legend(loc='upper right')
    ax1.set_xticks([1, 2, 3, 4, 5])

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Calibration Split AUPRC')
    ax2.set_title('Calibration AUPRC vs. Epoch')
    ax2.legend(loc='upper right')
    ax2.set_xticks([1, 2, 3, 4, 5])

    fig.tight_layout()
    save_fig(fig, 'learning_curves')


# ── Figure 7: Train / Cal / Holdout comparison ───────────────────────────────

def plot_split_comparison():
    print('\n[7/11] Train/Cal/Holdout split comparison …')
    ablations = load_json(RESULTS_DIR / 'ablations.json')
    if ablations is None:
        print('  SKIP — ablations.json not found.')
        return

    order = ['B1', 'A1', 'A2', 'A3', 'A4_r500', 'F', 'A6', 'A7']
    names = [n for n in order if n in ablations]

    xlabels = {
        'B1': 'B1\nEgo', 'A1': 'A1\nMap', 'A2': 'A2\nWeather',
        'A3': 'A3\nV2V', 'A4_r500': 'A4\n500 m', 'F': 'F\nAll',
        'A6': 'A6\nPercept.', 'A7': 'A7\nSafeDriver',
    }

    def best(history, split):
        vals = [h[split]['auprc'] for h in history if split in h]
        return max(vals) if vals else None

    train_v   = [best(ablations[n].get('history', []), 'train') for n in names]
    cal_v     = [best(ablations[n].get('history', []), 'cal')   for n in names]
    holdout_v = [ablations[n]['holdout']['auprc']               for n in names]

    x = np.arange(len(names))
    w = 0.24

    fig, ax = plt.subplots(figsize=(W2, 3.4))
    b1 = ax.bar(x - w, train_v,   w, label='Train (best epoch)', color=C_BLUE,   alpha=0.85)
    b2 = ax.bar(x,     cal_v,     w, label='Calibration',        color=C_ORANGE, alpha=0.85)
    b3 = ax.bar(x + w, holdout_v, w, label='Holdout',            color=C_GREEN,  alpha=0.85)

    # Annotate holdout bars only (most important)
    for bar, val in zip(b3, holdout_v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.004, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6.5, color=C_GREEN, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([xlabels.get(n, n) for n in names])
    ax.set_ylabel('AUPRC')
    ax.set_title('AUPRC by Split and Configuration\n'
                 '(Holdout values annotated)')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left')
    fig.tight_layout()
    save_fig(fig, 'split_comparison')


# ── Figure 8: SafeDriver IQ score distribution ────────────────────────────────

def plot_safedriver_score_distribution():
    print('\n[8/11] SafeDriver IQ score distribution …')
    rng = np.random.default_rng(42)
    n   = 5000
    base  = rng.beta(2, 5, n) * 100.0
    neg_s = np.clip(base                  + rng.normal(0, 5, n), 0, 100)
    pos_s = np.clip(base + 20.0           + rng.normal(0, 5, n), 0, 100)

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.78))
    ax.hist(neg_s, bins=40, alpha=0.6, color=C_BLUE,  density=True,
            label='Negative (safe context)',   edgecolor='white', lw=0.4)
    ax.hist(pos_s, bins=40, alpha=0.6, color=C_RED,   density=True,
            label='Positive (crash-adjacent)', edgecolor='white', lw=0.4)

    ax.axvline(neg_s.mean(), color=C_BLUE,  lw=1.2, linestyle='--',
               label=f'Neg mean = {neg_s.mean():.1f}')
    ax.axvline(pos_s.mean(), color=C_RED,   lw=1.2, linestyle='--',
               label=f'Pos mean = {pos_s.mean():.1f}')

    ax.set_xlabel('SafeDriver IQ risk score  (0 = safe, 100 = high risk)')
    ax.set_ylabel('Density')
    ax.set_title('SafeDriver IQ Score Distribution\n'
                 '(Simulated — illustrative only, not experimental data)')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, 'safedriver_score_distribution')


# ── Figure 9: Detection-uncertainty sensitivity ───────────────────────────────

def plot_detection_uncertainty_sensitivity():
    print('\n[9/11] Detection-uncertainty sensitivity …')
    # Interpolated between B1 (0.678) and A4_r500 (0.722) bounds
    miss_probs  = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30])
    auprc_miss  = 0.722 - (0.722 - 0.678) * (miss_probs / 0.30)

    pos_noises  = np.array([0.0, 0.3, 0.6, 1.0, 2.0, 3.0])
    auprc_noise = 0.722 - (0.722 - 0.678) * (pos_noises / 3.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.0))

    ax1.plot(miss_probs * 100, auprc_miss, 'o-', color=C_RED, lw=1.8,
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax1.axvline(10, color=C_GREY, lw=1.2, linestyle='--',
                label='Frozen param (10%)')
    ax1.fill_between(miss_probs * 100, auprc_miss, 0.678,
                     alpha=0.08, color=C_RED)
    ax1.set_xlabel('Missed Detection Probability (%)')
    ax1.set_ylabel('Holdout AUPRC')
    ax1.set_title('Sensitivity: Missed Detection Rate\n'
                  '(Illustrative — interpolated, A4_r500 bounds)')
    ax1.set_ylim(0.66, 0.74)
    ax1.legend(fontsize=7.5)

    # Annotate operating point
    idx = np.where(miss_probs == 0.10)[0][0]
    ax1.annotate(f'{auprc_miss[idx]:.3f}',
                 xy=(10, auprc_miss[idx]),
                 xytext=(14, auprc_miss[idx] + 0.003),
                 fontsize=7.5, color=C_RED,
                 arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    ax2.plot(pos_noises, auprc_noise, 'o-', color=C_BLUE, lw=1.8,
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax2.axvline(0.3, color=C_GREY, lw=1.2, linestyle='--',
                label='Frozen param (0.3 m)')
    ax2.fill_between(pos_noises, auprc_noise, 0.678,
                     alpha=0.08, color=C_BLUE)
    ax2.set_xlabel('Position Noise Std (m)')
    ax2.set_ylabel('Holdout AUPRC')
    ax2.set_title('Sensitivity: Position Noise\n'
                  '(Illustrative — interpolated, A4_r500 bounds)')
    ax2.set_ylim(0.66, 0.74)
    ax2.legend(fontsize=7.5)

    idx2 = np.where(pos_noises == 0.3)[0][0]
    ax2.annotate(f'{auprc_noise[idx2]:.3f}',
                 xy=(0.3, auprc_noise[idx2]),
                 xytext=(0.7, auprc_noise[idx2] + 0.003),
                 fontsize=7.5, color=C_BLUE,
                 arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.8))

    fig.tight_layout()
    save_fig(fig, 'detection_uncertainty_sensitivity')


# ── Figure 10: Dataset split diagram ─────────────────────────────────────────

def plot_data_split_diagram():
    print('\n[10/11] Dataset split diagram …')
    split_colors  = {
        'train':    '#0072B2',
        'cal':      '#E69F00',
        'holdout':  '#009E73',
        'neg_only': '#AAAAAA',
    }
    split_labels = {
        'train':    'Train',
        'cal':      'Calibration',
        'holdout':  'Holdout',
        'neg_only': 'Neg-only (no labels)',
    }

    fig, ax = plt.subplots(figsize=(W2, 2.8))
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.3, 2.8)
    ax.axis('off')

    # MiTra sessions T1-T9 on row y=1.7
    mitra = [
        ('T1', 'neg_only'), ('T2', 'neg_only'), ('T3', 'neg_only'),
        ('T4', 'train'), ('T5', 'train'), ('T6', 'train'), ('T7', 'train'),
        ('T8', 'cal'), ('T9', 'holdout'),
    ]
    for i, (label, split) in enumerate(mitra):
        x = i * 1.05
        ax.barh(1.7, 1.0, left=x, height=0.55,
                color=split_colors[split], edgecolor='white', linewidth=1.5)
        ax.text(x + 0.5, 1.7, label,
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')

    # NGSIM on row y=0.6
    ngsim = [
        ('US-101\nEarly (70%)', 'train'),
        ('US-101\nLate 30%', 'cal'),
        ('I-80\nAll', 'holdout'),
    ]
    x_starts = [0.0, 3.2, 7.3]
    widths    = [3.0, 2.8, 3.2]
    for (label, split), xs, w in zip(ngsim, x_starts, widths):
        ax.barh(0.6, w, left=xs, height=0.55,
                color=split_colors[split], edgecolor='white', linewidth=1.5)
        ax.text(xs + w / 2, 0.6, label,
                ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold')

    # Row labels
    ax.text(-0.4, 1.7, 'MiTra', ha='right', va='center', fontsize=9, fontweight='bold')
    ax.text(-0.4, 0.6, 'NGSIM', ha='right', va='center', fontsize=9, fontweight='bold')

    # Event counts annotation
    ax.text(10.8, 1.7, '5,320 train\n1,284 cal\n838 holdout',
            ha='right', va='center', fontsize=6.8, color='#444444')

    # Legend
    legend_patches = [mpatches.Patch(facecolor=split_colors[s],
                                     label=split_labels[s])
                      for s in split_colors]
    ax.legend(handles=legend_patches, loc='lower center',
              ncol=4, fontsize=7.5, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.10))

    ax.set_title('CREST Dataset Split: MiTra Sessions and NGSIM Corridors',
                 fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save_fig(fig, 'data_split_diagram')


# ── Figure 11: V2X simulation schematic ──────────────────────────────────────

def plot_v2x_simulation_schematic():
    print('\n[11/11] V2X simulation schematic …')
    fig, ax = plt.subplots(figsize=(W1, W1))
    ax.set_xlim(-620, 620)
    ax.set_ylim(-620, 620)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('#F5F5F5')

    # Road
    road = plt.Rectangle((-620, -45), 1240, 90, color='#CCCCCC', zorder=0)
    ax.add_patch(road)
    ax.axhline(0, color='white', linestyle='--', linewidth=1.2, zorder=1)

    # RSU
    rsu_x, rsu_y = 0, 210
    rsu_circle   = plt.Circle((rsu_x, rsu_y), 300,
                               color=C_PURPLE, alpha=0.10, zorder=1)
    ax.add_patch(rsu_circle)
    ax.plot(rsu_x, rsu_y, 's', color=C_PURPLE, markersize=13, zorder=3)
    ax.text(rsu_x - 10, rsu_y + 38, 'RSU (300 m)',
            ha='center', fontsize=8.5, color=C_PURPLE, fontweight='bold')

    # Ego vehicle
    ego_x, ego_y = -60, 0
    v2v_circle   = plt.Circle((ego_x, ego_y), 300,
                               color=C_BLUE, alpha=0.08, zorder=1)
    ax.add_patch(v2v_circle)
    ax.plot(ego_x, ego_y, 'D', color=C_BLUE, markersize=13, zorder=4)
    ax.text(ego_x, ego_y - 62, 'Ego vehicle',
            ha='center', fontsize=8.5, color=C_BLUE, fontweight='bold')

    # V2V range arrow
    ax.annotate('', xy=(ego_x + 300, ego_y), xytext=(ego_x, ego_y),
                arrowprops=dict(arrowstyle='<->', color=C_BLUE, lw=1.4))
    ax.text(ego_x + 150, ego_y + 22, 'V2V 300 m',
            ha='center', fontsize=7.5, color=C_BLUE)

    # Neighbors: (x, y, detected)
    for nx, ny, detected in [(210, 5, True), (360, -8, False),
                              (-210, 3, True), (95, 18, True)]:
        color  = C_GREEN if detected else C_RED
        marker = 'o'     if detected else 'x'
        msize  = 10      if detected else 11
        ax.plot(nx, ny, marker, color=color, markersize=msize, zorder=4,
                markeredgewidth=2 if not detected else 1)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=C_GREEN, markersize=9,
                   label='Detected vehicle'),
        plt.Line2D([0], [0], marker='x', color=C_RED,
                   markersize=9, markeredgewidth=2,
                   label='Missed detection (p = 0.10)'),
        plt.Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=C_BLUE, markersize=9,
                   label='Ego vehicle'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=C_PURPLE, markersize=9,
                   label='RSU (fixed infrastructure)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left',
              fontsize=7.5, framealpha=0.95)

    ax.set_title('V2X Simulation Schematic\n'
                 'RSU 300 m · V2V 300 m · Missed detection p = 0.10',
                 fontsize=9.5, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'v2x_simulation_schematic')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Generating CREST publication figures (IEEE T-IV quality) …')
    print(f'Output directory: {FIG_DIR}\n')

    # Reliability diagram intentionally omitted — see V3.4 change note.
    # Calibration quality is reported via Brier score in the results table.

    plot_pr_curve()
    plot_pr_curve_multi()
    plot_ablation_bar_chart()
    plot_lead_time_histogram()
    plot_rsu_sweep()
    plot_learning_curves()
    plot_split_comparison()
    plot_safedriver_score_distribution()
    plot_detection_uncertainty_sensitivity()
    plot_data_split_diagram()
    plot_v2x_simulation_schematic()

    print(f'\nDONE — 11 figures × 2 formats (PNG + PDF) = 22 files in {FIG_DIR}')


if __name__ == '__main__':
    main()
