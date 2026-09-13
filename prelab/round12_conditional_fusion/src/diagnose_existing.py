"""Describe R11 additive terms on train/dev only; no fitting or test scoring."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'round11_logprob'


def main():
    spec = importlib.util.spec_from_file_location('r11_readonly_diagnostic', OLD / 'src/run11.py')
    r11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r11)
    path = OLD / 'results/frozen_models.pkl'
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    frozen = json.loads((OLD / 'results/freeze11.json').read_text(encoding='utf-8'))
    assert checksum == frozen['model_sha256']
    models = pickle.loads(raw)
    meta = r11.r10.metadata(r11.SOURCE)
    bank = r11.Bank(r11.SOURCE, meta[2])
    result = {
        'status': 'descriptive_diagnostic_only',
        'new_fusion_trained': False,
        'test_labels_or_predictions_used': False,
        'old_models_sha256': checksum,
        'measure': 'Weighted SD of each additive logit term, and their ratio. Not causal importance, percentage attribution, or predicted improvement.',
        'splits': {},
    }
    for split in ['train', 'validation']:
        items, tokens, _, _ = r11.r10.cohort(r11.SOURCE, split, meta)
        for gran, rows in [('item', items), ('token', tokens)]:
            rows = [row for row in rows if row['main_eligible']]
            model = models['lb_nll__' + gran]
            matrix = bank.item_matrix if gran == 'item' else bank.token_matrix
            x = matrix(rows, 'lb_nll')
            z = model['scaler'].transform(x).astype(np.float32)
            coef = model['model'].coef_[0]
            lb = z[:, :784] @ coef[:784]
            nll = z[:, -1] * coef[-1]
            weights = r11.r10.base_weights(rows, gran)
            weights = weights / weights.sum()

            def sd(values):
                mean = np.sum(weights * values)
                return float(np.sqrt(np.sum(weights * (values - mean) ** 2)))

            expected = model['model'].decision_function(z)
            assert np.allclose(lb + nll + model['model'].intercept_[0], expected, atol=1e-9, rtol=1e-9)
            result['splits'][split + '_' + gran] = {
                'observations': len(rows),
                'nll_standardized_coefficient': float(coef[-1]),
                'lookback_contribution_weighted_sd': sd(lb),
                'nll_contribution_weighted_sd': sd(nll),
                'nll_to_lookback_sd_ratio': sd(nll) / sd(lb),
            }
    out = ROOT / 'results/contribution_diagnostic.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
