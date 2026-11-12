"""Plot only completed epoch records; no fitting or model selection changes."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/full_context_encoder_v2/full_finetune'
OUT = ROOT / 'results/full_context_encoder_v2/learning_curve'


def main():
    paths = sorted(SOURCE.glob('epoch_*.json'))
    pairs = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in paths]
    pairs = [(p, r) for p, r in pairs if r['epoch'] > 0]
    if not pairs:
        print('WAIT_FOR_COMPLETED_TRAINED_EPOCH', flush=True)
        return
    OUT.mkdir(exist_ok=True)
    rows = [r for _, r in pairs]
    epochs = [r['epoch'] for r in rows]
    window_fit = [r['fit_at_cal_thresholds']['windows']['f1'] for r in rows]
    window_cal = [r['calibration']['windows']['f1'] for r in rows]
    answer_cal = [r['calibration']['answers']['f1'] for r in rows]
    losses = [r['fit_weighted_bce'] for r in rows]
    complete = (SOURCE / 'complete.json').exists()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 160})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    ax = axes[0]
    ax.plot(epochs, window_fit, 'o-', color='#3267a8', label='Training: 4-BPE window F1')
    ax.plot(epochs, window_cal, 'o-', color='#d97620', label='Calibration: 4-BPE window F1')
    ax.plot(epochs, answer_cal, 's--', color='#43815c', label='Calibration: whole-answer F1')
    ax.axhline(.75, color='#858585', linestyle=':', linewidth=1, label='Localization target: 0.75')
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 (risk-positive class)')
    ax.set_title('Same frozen labels and calibration set')
    ax.legend(loc='lower right', fontsize=8)
    ax = axes[1]
    ax.plot(epochs, losses, 'o-', color='#3267a8')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Weighted training BCE')
    ax.set_title('Training objective')
    for ax in axes:
        ax.set_xticks(epochs)
        ax.set_xlabel('Completed training epoch')
        ax.grid(alpha=.18)
    state = 'All six epochs complete' if complete else f'In progress: {len(epochs)} / 6 epochs complete'
    fig.suptitle(f'Full-context ModernBERT QA baseline | {state}', fontsize=12)
    fig.savefig(OUT / 'learning_curve.png')
    fig.savefig(OUT / 'learning_curve.pdf')
    plt.close(fig)
    artifact = {'training_finished': complete, 'epochs': epochs,
                'training_window_f1_at_calibration_threshold': window_fit,
                'calibration_window_f1': window_cal,
                'calibration_answer_f1': answer_cal,
                'weighted_training_bce': losses,
                'source_epoch_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in pairs},
                'official_test_opened': False, 'models_or_thresholds_changed': False,
                'limits': 'Repeated development calibration, not held-out test. Fit/calibration class proportions differ. Training and calibration F1 share each epoch calibration threshold. The answer all-risk baseline is0.7722; its target is not the0.75 localization line.'}
    (OUT / 'curve_data.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'epochs_plotted': epochs, 'training_finished': complete,
                      'plot': str((OUT / 'learning_curve.png').resolve())}), flush=True)


if __name__ == '__main__':
    main()
