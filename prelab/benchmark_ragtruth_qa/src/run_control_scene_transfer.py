"""Frozen-QA-model transfer into existing R16 train development; no fitting."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
import run_development as q
import run_full_context_encoder_v2 as base
import prepare_control_scene_transfer as scene

OUT = scene.OUT
SOURCES = {
    'qa_only': (q.ROOT / 'results/full_context_encoder_v2/full_finetune', 'epoch_{epoch:02d}.pt'),
    'auxiliary_transfer': (q.ROOT / 'results/full_context_aux_transfer_v1/transfer', 'qa_epoch_{epoch:02d}.pt'),
}


def check_prepared():
    manifest = q.read(OUT / 'preparation_complete.json')
    for name, value in manifest['artifacts_sha256'].items():
        assert q.sha(OUT / name) == value
    for name, value in manifest['sources_sha256'].items():
        assert q.sha(Path(name)) == value
    return manifest


def infer(method):
    """GPU only when explicitly scheduled after all other owners have exited."""
    manifest = check_prepared()
    source, template = SOURCES[method]
    if not (source / 'complete.json').exists():
        print('WAIT_UPSTREAM_QA_SELECTION_NO_INFERENCE', flush=True)
        return
    complete = q.read(source / 'complete.json')
    selected = complete['selected']
    assert selected['epoch'] > 0 and not complete['official_test_opened']
    checkpoint = source / template.format(epoch=selected['epoch'])
    assert q.sha(checkpoint) == selected['artifacts_sha256']['.pt']
    directory = OUT / method
    assert not (directory / 'started.json').exists()
    directory.mkdir(exist_ok=True)
    freeze = {'method': method, 'upstream_complete_sha256': q.sha(source / 'complete.json'),
              'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': q.sha(checkpoint),
              'QA_selected_epoch': selected['epoch'], 'QA_thresholds': selected['thresholds'],
              'scene_preparation_sha256': q.sha(OUT / 'preparation_complete.json'),
              'code_sha256': q.sha(Path(__file__)), 'selection_uses_R16': False,
              'original_heldouts_opened': False}
    q.save(directory / 'started.json', freeze)
    base.configure_gpu()
    model = base.load_model()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    model.cuda().eval()
    rows = q.lines(OUT / 'inputs.jsonl')
    probabilities = {}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for i, row in enumerate(rows):
            probability = base.logits(model, row, 'cuda').sigmoid().cpu().numpy()
            assert probability.shape == (row['raw_token_count'],) and np.isfinite(probability).all()
            probabilities[row['response_id']] = probability
            if (i+1) % 100 == 0:
                print('CONTROL_TRANSFER_INFER', method, i+1, len(rows), flush=True)
    assert len(probabilities) == manifest['answers'] == 602
    np.savez_compressed(directory / 'token_predictions.npz', **probabilities)
    q.save(directory / 'inference_complete.json', {'status': 'complete_no_model_fitting',
        'answers': len(probabilities), 'seconds': time.perf_counter()-start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'source_freeze': freeze, 'token_predictions_sha256': q.sha(directory / 'token_predictions.npz'),
        'original_heldouts_opened': False})
    del model
    torch.cuda.empty_cache()
    print('CONTROL_TRANSFER_INFERENCE_COMPLETE_GPU_RELEASED', method, flush=True)


def aggregate(predictions, windows, answers):
    values = np.asarray([max(float(predictions[w['row_id']][j]) for j in w['lexical_token_indices']) for w in windows])
    by_item = {}
    for w, score in zip(windows, values):
        item = w['item_ids'][0]
        by_item[item] = max(by_item.get(item, -np.inf), score)
    answer_values = np.asarray([by_item[a['item_id']] for a in answers])
    assert np.isfinite(values).all() and np.isfinite(answer_values).all()
    return values, answer_values


def score(method):
    check_prepared()
    directory = OUT / method
    run = q.read(directory / 'inference_complete.json')
    assert q.sha(directory / 'token_predictions.npz') == run['token_predictions_sha256']
    assert not (directory / 'score_complete.json').exists()
    windows, answers = q.lines(OUT / 'windows.jsonl'), q.lines(OUT / 'answers.jsonl')
    with np.load(directory / 'token_predictions.npz') as z:
        assert set(z.files) == {a['row_id'] for a in answers}
        wv, av = aggregate(z, windows, answers)
    wm = np.asarray([w['main_eligible'] for w in windows], bool)
    am = np.asarray([a['main_eligible'] for a in answers], bool)
    wy = np.asarray([w['gold'] if w['main_eligible'] else -1 for w in windows])
    ay = np.asarray([a['gold'] if a['main_eligible'] else -1 for a in answers])
    qa_thresholds = run['source_freeze']['QA_thresholds']
    direct = {'windows': scene.counts(wy[wm], wv[wm] >= qa_thresholds['window']['threshold']),
              'answers': scene.counts(ay[am], av[am] >= qa_thresholds['answer']['threshold'])}
    wp, ap = np.zeros(len(wv), bool), np.zeros(len(av), bool)
    wc, ac = np.zeros(len(wv), int), np.zeros(len(av), int)
    fold_results = []
    for fold, groups in enumerate(q.read(OUT / 'folds.json')):
        calibration = set(groups['calibration_groups'])
        evaluation = set(groups['evaluation_groups'])
        wi = np.asarray([w['group_id'] in calibration for w in windows]) & wm
        ai = np.asarray([a['group_id'] in calibration for a in answers]) & am
        wt = q.choose_threshold(wy[wi], wv[wi])
        at = q.choose_threshold(ay[ai], av[ai])
        we = np.asarray([w['group_id'] in evaluation for w in windows])
        ae = np.asarray([a['group_id'] in evaluation for a in answers])
        wp[we] = wv[we] >= wt['threshold']
        ap[ae] = av[ae] >= at['threshold']
        wc[we] += 1
        ac[ae] += 1
        fold_results.append({'fold': fold, 'thresholds': {'window': wt, 'answer': at},
            'windows': scene.counts(wy[we & wm], wp[we & wm]),
            'answers': scene.counts(ay[ae & am], ap[ae & am]), 'model_fitted_on_R16': False})
    assert (wc == 1).all() and (ac == 1).all()
    calibrated = {'windows': scene.counts(wy[wm], wp[wm]), 'answers': scene.counts(ay[am], ap[am])}
    old = q.read(scene.R26 / 'summary.json')['methods']
    comparisons = {name: {unit: {k: old[name][unit][k] for k in ('precision', 'recall', 'f1')}
                                  for unit in ('windows', 'answers')}
                   for name in ('lookback_tuned', 'redeep_tuned', 'local_slots_fusion_smooth_global')}
    np.savez_compressed(directory / 'scores.npz', window_scores=wv, answer_scores=av,
                        fivefold_window_predictions=wp, fivefold_answer_predictions=ap)
    result = {'method': method, 'unchanged_QA_thresholds': direct,
              'existing_R16_fivefold_threshold_calibration': calibrated, 'folds': fold_results,
              'existing_development_comparisons': comparisons,
              'scope': 'R16 repeatedly exposed train development only; not a fresh test. External semantic training differs from native-Qwen probe training.',
              'model_selected_using_R16': False, 'model_fitted_on_R16': False,
              'original_R16_validation_test_opened': False, 'RAGTruth_test_opened': False,
              'human_gold': False, 'labels_or_windows_changed': False}
    q.save(directory / 'summary.json', result)
    q.save(directory / 'score_complete.json', {'summary_sha256': q.sha(directory / 'summary.json'),
        'scores_sha256': q.sha(directory / 'scores.npz'), 'inference_sha256': q.sha(directory / 'inference_complete.json')})
    print('CONTROL_TRANSFER_SCORED', method, calibrated['windows']['f1'], calibrated['answers']['f1'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['infer', 'score'])
    parser.add_argument('method', choices=tuple(SOURCES))
    args = parser.parse_args()
    {'infer': infer, 'score': score}[args.stage](args.method)
