"""Use the two saved auxiliary heads from one already selected frozen model."""
from pathlib import Path
import argparse
import numpy as np
import torch
import run_development as q

SOURCE = q.ROOT / 'results/semantic_multitask_v1'
INDEX = q.ROOT / 'results/sequence_v1/token_index.json'
OUT = q.ROOT / 'results/existing_auxiliary_readout_v1'
METHOD = 'semantic_tcn_aux_types_w32'


def selected():
    complete = q.read(SOURCE / 'complete.json')
    assert not complete['test_opened']
    summary = q.read(SOURCE / 'summary.json')
    assert q.sha(SOURCE / 'summary.json') == complete['files_sha256']['summary.json']
    assert len(summary['all_epochs'][METHOD]) == 30
    entry = summary['selected'][METHOD]
    assert entry == max(summary['all_epochs'][METHOD], key=lambda e: e['selection_key'])
    path = SOURCE / METHOD / f"epoch_{entry['epoch']:03d}_scores.npz"
    assert q.sha(path) == entry['scores_sha256']
    return entry, path


def prepare():
    assert not (OUT / 'protocol.json').exists()
    entry, path = selected()
    q.save(OUT / 'protocol.json', {
        'scope': 'Original634 fit/159 calibration only; no new model, data, labels, or epoch choice.',
        'source_sha256': q.sha(Path(__file__)), 'selected_epoch': entry['epoch'],
        'score_sha256': q.sha(path), 'index_sha256': q.sha(INDEX),
        'control': 'Exact saved primary binary-risk output of the same frozen selected model.',
        'candidate': 'For each raw token, max(sigmoid(saved baseless logit),sigmoid(saved conflict logit)). Original lexical4rawBPE window max, then answermax.',
        'thresholds': 'One candidate only, original two calibration F1 thresholds. No per-type threshold or further pooling rule.',
        'limits': 'The original model was selected using its primary head, trained634 answers with auxiliary coefficient.25 and head-specific auxiliary weighting. This is a frozen readout experiment, not a matched retraining of direct-type heads. Repeated development calibration.',
        'new_fits': 0, 'GPU_used': False, 'official_test_opened': False})
    print('EXISTING_AUX_READOUT_PREPARED_ONE_FIXED_EPOCH', entry['epoch'], flush=True)


def run():
    protocol = q.read(OUT / 'protocol.json')
    assert protocol['source_sha256'] == q.sha(Path(__file__))
    assert not (OUT / 'complete.json').exists()
    entry, path = selected()
    assert entry['epoch'] == protocol['selected_epoch']
    assert q.sha(path) == protocol['score_sha256'] and q.sha(INDEX) == protocol['index_sha256']
    meta = q.metadata(); index = q.read(INDEX)['answers']
    assert [a['response_id'] for a in index] == [a['response_id'] for a in meta['answers']]
    offsets = {a['response_id']: a['left'] for a in index}
    def windows(p):
        return np.asarray([max(float(p[offsets[w['response_id']] + j]) for j in w['token_indices']
            if meta['by_response'][w['response_id']]['tokens']['lexical_mask'][j])
            for w in meta['windows']], np.float64)
    with np.load(path, allow_pickle=False) as z:
        original = windows(z['token_scores'])
        assert np.array_equal(original, z['window_scores'])
        assert q.metrics(meta, original, entry['thresholds']) == entry['metrics']
        aux = z['auxiliary_token_logits'].copy()
    assert aux.shape == (index[-1]['right'], 2) and np.isfinite(aux).all()
    p = torch.sigmoid(torch.from_numpy(aux)).numpy().max(1)
    scores = windows(p); answers = q.answer_scores(meta, scores)
    lo, hi = meta['bounds']['calibration']
    thresholds = {
        'window': q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]], scores[lo:hi]),
        'answer': q.choose_threshold([a['label'] for a in meta['answers'][634:]], answers[634:])}
    metrics = q.metrics(meta, scores, thresholds)
    np.savez_compressed(OUT / 'scores.npz', token_scores=p, window_scores=scores, answer_scores=answers)
    q.save(OUT / 'summary.json', {'control': entry, 'aux_max': {'thresholds': thresholds, 'metrics': metrics},
        'same_frozen_model_and_epoch': True, 'control_window_scores_and_metrics_exact': True,
        'auxiliary_token_logits_sha256': q.sha(path), 'scores_sha256': q.sha(OUT / 'scores.npz'),
        'new_fits': 0, 'official_test_opened': False, 'GPU_used': False})
    assert not torch.cuda.is_initialized()
    q.save(OUT / 'complete.json', {'summary_sha256': q.sha(OUT / 'summary.json'), 'new_fits': 0,
                                  'GPU_used': False, 'official_test_opened': False})
    print('EXISTING_AUX_MAX_RESULT', metrics['calibration']['windows']['f1'], metrics['calibration']['answers']['f1'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'run'])
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
