"""Real frozen-model implementation check on predeclared old TRAIN rows only."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import run10 as runner
import feature10


def main():
    root = runner.ROOT
    rows = [r for r in runner.readl(root/'data/dev_inputs.jsonl') if r['split'] == 'train'][:2]
    tok, model = runner.model7.load_model()
    report = {'model': str(runner.model7.MODEL), 'rows': [], 'new_test_used': False,
              'feature_source_sha256': runner.sha(feature10.__file__), 'passed': False}
    for row in rows:
        g = json.loads((root/'data/generation_records'/(row['row_id']+'.json')).read_text('utf-8'))
        visible = runner.run9.visible(row)
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        arrays, meta = feature10.extract_features(tok, model, visible, runner.run9.extractor_generation(g))
        runner.validate_new(arrays, meta, g)
        with np.load(root/'data/features'/(row['row_id']+'.npz'), allow_pickle=False) as old:
            agreement = {name: float(np.max(np.abs(arrays[name]-old[name])))
                         for name in ['lookback_features', 'hidden_28', 'token_nll', 'token_entropy']}
        assert agreement['lookback_features'] <= 1e-6 and agreement['hidden_28'] <= 1e-6, agreement
        assert agreement['token_nll'] <= 1e-5 and agreement['token_entropy'] <= 1e-5, agreement
        changed = dict(runner.run9.extractor_generation(g))
        ids = list(g['response_token_ids']); cut = max(1, len(ids)//2)
        replacement = tok.encode('hello', add_special_tokens=False)[0]
        assert ids[cut] != replacement
        ids[cut] = replacement
        changed['response_token_ids'] = ids
        changed['response'] = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        changed['response_token_offsets'] = runner.model7.token_offsets(tok, ids, changed['response']).tolist()
        other, _ = feature10.extract_features(tok, model, visible, changed)
        prefix = {name: float(np.max(np.abs(arrays[name][:cut]-other[name][:cut])))
                  for name in ['new_features', 'surface_features', 'lookback_features', 'hidden_28', 'token_nll', 'token_entropy']}
        assert max(prefix.values()) <= 1e-6, prefix
        prior = np.asarray(meta['semantic_sentence_prior'])
        assert (prior > 0).all() and np.allclose(prior.sum(-1), 1)
        record = {'row_id': row['row_id'], 'split': 'train', 'core_max_abs_difference': agreement,
                  'future_changed_at_token': cut, 'unchanged_prefix_max_abs_difference': prefix,
                  'soft_prior_positive_normalized': True, 'new_shape': list(arrays['new_features'].shape),
                  'surface_shape': list(arrays['surface_features'].shape),
                  'seconds_for_two_replays': time.perf_counter()-started,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        report['rows'].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report['passed'] = True
    runner.save(root/'results/gpu_implementation_audit10.json', report)
    print('GPU_AUDIT10_PASSED', flush=True)


if __name__ == '__main__': main()
