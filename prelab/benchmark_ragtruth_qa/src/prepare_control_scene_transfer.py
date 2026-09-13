"""Prepare ONLY the already exposed R16 train scene for semantic transfer.

R16 original validation/test and RAGTruth official test remain unopened. No
model weights, training, inference, or new performance evaluation is involved.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer
import run_development as q
import tail_finetune as mapping

ROOT = q.ROOT
R16 = ROOT.parent / 'round16_dataset_expansion'
R26 = ROOT.parent / 'round26_global_local_fusion/results'
CANDIDATES = ROOT.parent / 'round23b_local_evidence_probe/results/candidate_windows.jsonl'
OUT = ROOT / 'control_scene_transfer_v1'
MODEL = ROOT.parent / 'models/ModernBERT-base'


def counts(labels, predictions):
    y = np.asarray(labels, int)
    p = np.asarray(predictions, bool)
    tp, fp = int(((y == 1) & p).sum()), int(((y == 0) & p).sum())
    fn, tn = int(((y == 1) & ~p).sum()), int(((y == 0) & ~p).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'started.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [R16 / 'data/dataset_train.jsonl', R16 / 'data/tokens_train.jsonl', CANDIDATES,
              R26 / 'answer_scores_oof.jsonl', R26 / 'window_scores_oof.jsonl', R26 / 'summary.json',
              *[R26 / f'fold_{f}_calibration.json' for f in range(5)],
              MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json', Path(__file__)]
    sources = {str(p.resolve()): q.sha(p) for p in inputs}
    q.save(OUT / 'started.json', {'source_sha256': sources, 'scope': 'R16 existing train only'})
    original = q.lines(R16 / 'data/dataset_train.jsonl')
    assert len(original) == 602 and all(r['input']['split'] == 'train' for r in original)
    by_id = {r['input']['row_id']: r for r in original}
    assert len(by_id) == 602 and len({r['input']['question_id'] for r in original}) == 301
    assert len({r['input']['group_id'] for r in original}) == 278
    native = defaultdict(dict)
    for token in q.lines(R16 / 'data/tokens_train.jsonl'):
        assert token['split'] == 'train' and token['row_id'] in by_id
        native[token['row_id']][token['token_index']] = token
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rendered = []
    for r in original:
        row = r['input']
        rid, text = row['row_id'], r['response']
        offsets = r['response_token_offsets']
        assert len(offsets) == len(r['response_token_ids']) == len(native[rid])
        for j, (left, right) in enumerate(offsets):
            t = native[rid][j]
            assert [left, right] == [t['start'], t['end']] and text[left:right] == t['text']
            assert r['response_token_ids'][j] == t['token_id']
        marker = '\n\nSearch results:\n'
        assert row['prompt'].count(marker) == 1
        task, evidence = row['prompt'].split(marker)
        # Model-visible content comes only from the actual neutral generation prompt.
        expected_evidence = '\n\n'.join(f'[{j+1}] {p["title"]}\n{p["text"]}' for j, p in enumerate(row['passages']))
        assert evidence == expected_evidence
        prefix = evidence + tokenizer.sep_token + task + tokenizer.sep_token
        encoded = tokenizer(prefix + text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        eo = np.asarray(encoded['offset_mapping'], np.int64)
        inside = (eo[:, 1] > len(prefix)) & (eo[:, 0] < len(prefix) + len(text))
        begin = np.where(inside, np.maximum(0, eo[:, 0] - len(prefix)), -1)
        end = np.where(inside, np.minimum(len(text), eo[:, 1] - len(prefix)), -1)
        charmap = mapping.character_map(text, offsets, begin, end)
        assert len(encoded['input_ids']) <= tokenizer.model_max_length
        rendered.append({'response_id': rid, 'input_ids': encoded['input_ids'], 'raw_token_count': len(offsets),
                         'mapping': [x.tolist() for x in charmap], 'answer_sha256': q.digest(text),
                         'prompt_sha256': q.digest(row['prompt']),
                         'raw_token_ids': r['response_token_ids'], 'raw_token_offsets': offsets})
    windows = q.lines(CANDIDATES)
    prior_windows = {w['window_key']: w for w in q.lines(R26 / 'window_scores_oof.jsonl')}
    prior_answers = q.lines(R26 / 'answer_scores_oof.jsonl')
    answers = []
    aw = defaultdict(list)
    geometry = []
    for i, w in enumerate(windows):
        assert w['split'] == 'train' and len(w['item_ids']) == 1
        old = prior_windows[w['window_key']]
        assert all(old[k] == v for k, v in w.items())
        rid = w['row_id']
        lexical = [j for j in w['raw_token_indices'] if native[rid][j]['lexical']]
        assert lexical
        # Gold follows the old eligibility rule, including None for refused/unresolved spans.
        if w['main_eligible']:
            assert w['gold'] == int(any(native[rid][j]['gold'] == 1 for j in lexical))
        else:
            assert w['gold'] is None
        geometry.append(dict(w, lexical_token_indices=lexical))
        aw[w['item_ids'][0]].append(i)
    for a in prior_answers:
        original_annotation = by_id[a['row_id']]['annotation']
        assert original_annotation['item_id'] == a['item_id']
        assert original_annotation['answer_gold'] == a['gold']
        assert original_annotation['human_gold'] is False
        assert aw[a['item_id']]
        answers.append({k: v for k, v in a.items() if k not in ('scores', 'predictions')})
    assert len(windows) == 12222 and sum(w['main_eligible'] for w in windows) == 9526
    assert len(answers) == 602 and sum(a['main_eligible'] for a in answers) == 598
    assert sum(a['reviewed_safe_refusal'] for a in answers) == 117
    folds = [q.read(R26 / f'fold_{f}_calibration.json')['groups'] for f in range(5)]
    evaluation_groups = [g for fold in folds for g in fold['evaluation_groups']]
    assert len(evaluation_groups) == len(set(evaluation_groups)) == 278
    for f in folds:
        sets = [set(f[k]) for k in ('fit_groups', 'calibration_groups', 'evaluation_groups')]
        assert len(set.union(*sets)) == 278 and not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    # Replay existing R26 predictions. This verifies geometry/counts, not new-model quality.
    old_summary = q.read(R26 / 'summary.json')['methods']
    replayed = {}
    for name in ('lookback_tuned', 'slots_base_smooth_global'):
        for a, saved in zip(answers, prior_answers):
            maximum = max(prior_windows[windows[j]['window_key']]['scores'][name] for j in aw[a['item_id']])
            assert maximum == saved['scores'][name]
        wm = counts([w['gold'] for w in windows if w['main_eligible']],
                    [prior_windows[w['window_key']]['predictions'][name] for w in windows if w['main_eligible']])
        am = counts([a['gold'] for a in prior_answers if a['main_eligible']],
                    [a['predictions'][name] for a in prior_answers if a['main_eligible']])
        for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1'):
            assert wm[key] == old_summary[name]['windows'][key]
            assert am[key] == old_summary[name]['answers'][key]
        replayed[name] = {'windows': wm, 'answers': am, 'all_answer_maxima_exact': True}
    q.savel(OUT / 'inputs.jsonl', rendered)
    q.savel(OUT / 'windows.jsonl', geometry)
    q.savel(OUT / 'answers.jsonl', answers)
    q.save(OUT / 'folds.json', folds)
    q.save(OUT / 'EXISTING_BASELINE_GEOMETRY_CHECK.json', {'passed': True, 'methods': replayed, 'new_inference': False})
    protocol = {'scope': 'Only original R16 train602 answers/301questions/278groups. Repeated development, not a fresh test.',
        'purpose': 'Transfer an externally trained semantic detector into the existing neutral incomplete-retrieval scene.',
        'input': 'Exact visible search-results rendering, original neutral task/questions, unchanged full Qwen answer; no condition/category/annotation fields enter model input.',
        'labels': 'Existing assistant annotation and independent assistant review, human_gold=false. No relabeling or new human-gold claim.',
        'geometry': 'Exact original Qwen raw BPE IDs/offsets and12222 candidate4-BPE windows. Window score=max lexical token risk; all candidate windows feed answer max, including safe refusals.',
        'eligibility': 'Original9526 asserted windows/1063positive;598resolved answers/151positive including117safe-refusal negatives. Unresolved labels stay None, no zero imputation.',
        'model_selection': 'Choose QA checkpoint by its already-fixed QA rule only, before any R16 inference. No R16 epoch/model selection.',
        'thresholds': 'Report both unchanged QA window/answer thresholds and separately the existing5fold R16 development calibration: each threshold selected only on that fold calibration groups; evaluate its original outer groups, no neural fitting.',
        'comparison_limits': 'External QA or auxiliary supervision and another encoder differ from original Qwen-state baselines. This transfer alone does not prove a pure native-whitebox advantage or fresh-event test performance.',
        'original_R16_validation_test_opened': False, 'RAGTruth_test_opened': False,
        'no_new_generation': True, 'no_truncation': True}
    q.save(OUT / 'protocol.json', protocol)
    files = ['inputs.jsonl', 'windows.jsonl', 'answers.jsonl', 'folds.json', 'EXISTING_BASELINE_GEOMETRY_CHECK.json', 'protocol.json']
    manifest = {'status': 'prepared_not_inferred', 'answers': len(rendered),
        'input_tokens': sum(len(r['input_ids']) for r in rendered),
        'max_input_tokens': max(len(r['input_ids']) for r in rendered),
        'raw_answer_tokens': sum(r['raw_token_count'] for r in rendered),
        'candidate_windows': len(windows), 'scored_windows': 9526, 'scored_answers': 598,
        'sources_sha256': sources, 'artifacts_sha256': {name: q.sha(OUT / name) for name in files},
        'GPU_used': False, 'original_heldouts_opened': False}
    assert sources == {str(p.resolve()): q.sha(p) for p in inputs}
    q.save(OUT / 'preparation_complete.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'answers', 'max_input_tokens', 'raw_answer_tokens', 'candidate_windows')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
