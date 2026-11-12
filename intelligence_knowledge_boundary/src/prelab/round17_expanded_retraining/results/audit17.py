"""Independent post-freeze audit. No imports of experiment evaluators; no fitting.

Default execution is blocked until test_complete17.json exists. --self-test uses
synthetic objects only. Raw annotations/BPE offsets reconstruct every denominator.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import hashlib
import json
import math
import os
import pickle

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '4')
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'round10_dual_granularity'
NEW = ROOT.parent/'round16_dataset_expansion'
FEATURES = ('lb', 'nll', 'lb_nll')
METHODS = tuple(c+'_'+f for c in ('legacy', 'expanded') for f in FEATURES)


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text('utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')).hexdigest()


def close(a, b, label='', atol=1e-10, rtol=1e-8):
    if a is None or b is None:
        assert a is b, (label, a, b)
    else:
        assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False), (label, a, b)


def verify_tree(base, hashes):
    for relative, expected in hashes.items():
        assert sha(base/relative) == expected, relative


def assert_metrics(expected, actual, label):
    for key, value in expected.items():
        assert key in actual, (label, key)
        close(value, actual[key], label+'.'+key)


def counts(y, prediction, scores=None, unit='tokens'):
    y, p = np.asarray(y, int), np.asarray(prediction, bool)
    assert y.shape == p.shape and len(y) and set(y) <= {0, 1}
    tp = int(np.count_nonzero((y == 1) & p))
    fp = int(np.count_nonzero((y == 0) & p))
    fn = int(np.count_nonzero((y == 1) & ~p))
    tn = int(np.count_nonzero((y == 0) & ~p))
    out = {unit: len(y), 'risk_'+unit: int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
           'precision': tp/(tp+fp) if tp+fp else None,
           'recall': tp/(tp+fn) if tp+fn else None,
           'f1': 2*tp/(2*tp+fp+fn) if tp+fn else None,
           'alert_rate': float(p.mean()), 'risk_rate': float(y.mean())}
    if scores is not None:
        scores = np.asarray(scores, float)
        assert np.isfinite(scores).all() and scores.shape == y.shape
        out.update(auroc=float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
                   average_precision=float(average_precision_score(y, scores)) if y.sum() else None)
    return out


def threshold(y, values):
    """Independent exhaustive candidate scan, including all/none endpoints."""
    y, values = np.asarray(y, int), np.asarray(values, float)
    assert set(y) == {0, 1} and np.isfinite(values).all()
    candidates = [math.nextafter(float(values.max()), math.inf),
                  *np.unique(values).tolist(), math.nextafter(float(values.min()), -math.inf)]
    best = None
    for t in candidates:
        p = values >= t
        tp = int(np.count_nonzero(p & (y == 1)))
        count = int(p.sum())
        key = (2*tp/(count+int(y.sum())), tp/count if count else 0., t)
        best = key if best is None or key > best else best
    return {'threshold': best[2], 'validation_f1': best[0], 'validation_precision': best[1],
            'tokens': len(y), 'scorable_tokens': len(y), 'risk_tokens': int(y.sum())}


def weights(rows, target):
    """Independently rebuild group->condition->answer->window equal masses."""
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for j, row in enumerate(rows):
        buckets[row['group_id']][row['condition']][row['item_ids'][0]].append(j)
    base = np.empty(len(rows), float)
    for conditions in buckets.values():
        for answers in conditions.values():
            for ix in answers.values():
                base[ix] = 1/(len(conditions)*len(answers)*len(ix))
    base /= base.mean()
    y = np.asarray([r['gold'] for r in rows], int)
    class_mass = np.asarray([base[y == label].sum() for label in (0, 1)])
    assert (class_mass > 0).all()
    factors = base.sum()/(2*class_mass)
    loss = base*factors[y]
    for conditions in buckets.values():
        ix = [i for answers in conditions.values() for vv in answers.values() for i in vv]
        loss[ix] *= (len(rows)/len(buckets))/loss[ix].sum()
    loss *= target/loss.sum()
    return base, loss, factors


def label_geometry(row, g, a, safe):
    """Raw-character gold and label-independent BPE candidate window geometry."""
    assert len(g['items']) == 1
    canonical = g['items'][0]
    for key in ('item_id', 'text', 'start', 'end'):
        assert canonical[key] == a[key], (row['row_id'], key)
    text = g['response']
    if canonical['parse_ok']:
        assert text[canonical['start']:canonical['end']] == canonical['text']
        left, right = canonical['start'], canonical['end']
    else:
        left, right = 0, len(text)
        assert a['localization_status'] != 'resolved'
    asserted = a['original_stance'] == 'asserted' and a['original_risk'] in (0, 1)
    safe_refusal = a['item_id'] in safe
    if safe_refusal:
        assert a['original_stance'] == 'abstained' and a['original_risk'] is None
        assert a['localization_status'] == 'excluded' and not a['risk_spans']
    eligible = asserted or safe_refusal
    resolved = asserted and a['localization_status'] == 'resolved'
    fields = {k: row[k] for k in ('row_id', 'question_id', 'group_id', 'split', 'condition', 'category')}
    item = dict(canonical, **fields, gold=a['original_risk'] if asserted else 0 if safe_refusal else None,
                main_eligible=eligible, asserted_eligible=asserted, reviewed_safe_refusal=safe_refusal,
                original_stance=a['original_stance'], localization_status=a['localization_status'])
    known = {i for i in range(canonical['start'], canonical['end']) if text[i].isalnum()} if resolved else set()
    bad = set()
    regions = []
    merged = []
    for span in a['risk_spans']:
        assert text[span['start']:span['end']] == span['text']
        chars = {i for i in range(span['start'], span['end']) if text[i].isalnum()}
        assert chars and chars <= known
        bad |= chars
    for start, end in sorted((s['start'], s['end']) for s in a['risk_spans']):
        if merged and merged[-1][1] >= start:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for start, end in merged:
        regions.append({'characters': {j for j in range(start, end) if text[j].isalnum()}, 'token_keys': []})
    assert bool(a['risk_spans']) == (a['original_risk'] == 1) if resolved else not a['risk_spans']
    tokens = []
    for j, ((l, r), token_id) in enumerate(zip(g['response_token_offsets'], g['response_token_ids'])):
        assert 0 <= l <= r <= len(text)
        chars = {k for k in range(l, r) if text[k].isalnum()}
        main = bool(chars) and chars <= known and resolved
        token = dict(fields, token_key=row['row_id']+f'__token{j}', token_index=j,
                     token_id=token_id, start=l, end=r, text=text[l:r], lexical=bool(chars),
                     main_eligible=main, gold=int(bool(chars & bad)) if main else None)
        tokens.append(token)
        for region in regions:
            if main and chars & region['characters']:
                region['token_keys'].append(token['token_key'])
    ix = [j for j, (l, r) in enumerate(g['response_token_offsets']) if r > left and l < right]
    windows = []
    if ix:
        assert ix == list(range(ix[0], ix[-1]+1))
        for first in range(max(1, len(ix)-3)):
            raw = ix[first:first+4]
            lexical = [tokens[j] for j in raw if tokens[j]['lexical']]
            if not lexical:
                continue
            if resolved:
                assert all(t['main_eligible'] for t in lexical)
            l, r = max(left, tokens[raw[0]]['start']), min(right, tokens[raw[-1]]['end'])
            windows.append(dict(fields, window_key=item['item_id']+f'__w4__start{raw[0]}',
                 item_ids=[item['item_id']], width=4, actual_width=len(raw), short_window=len(raw) < 4,
                 raw_token_indices=raw, token_keys=[t['token_key'] for t in lexical],
                 start=l, end=r, text=text[l:r], main_eligible=resolved,
                 gold=int(any(t['gold'] == 1 for t in lexical)) if resolved else None))
    return item, tokens, windows, regions, bool(ix)


class Rebuild:
    def __init__(self, manifest):
        self.manifest = manifest
        self.files = {}

    def features(self, source, g, generation_path):
        directory = OLD if source == OLD else ROOT
        p = directory/'data/features'/(g['row_id']+'.npz')
        side = p.with_suffix('.json')
        meta = read(side)
        assert meta['source_generation_sha256'] == sha(generation_path)
        assert meta['arrays_sha256'] == sha(p)
        if source == NEW:
            declared = self.manifest['records'][g['row_id']]
            assert p.resolve() == (ROOT/declared['npz']).resolve()
            assert side.resolve() == (ROOT/declared['json']).resolve()
            assert sha(side) == declared['json_sha256'] and sha(p) == declared['npz_sha256']
            assert declared['source_generation_sha256'] == sha(generation_path)
            assert meta['extraction_signature_sha256'] == self.manifest['extraction_signature_sha256']
            assert meta['labels_read'] is False and meta['response_regenerated'] is False
            assert meta['attention_timing'].startswith('post-read query P+j')
            assert 'P+j-1' in meta['nll_timing']
        with np.load(p, allow_pickle=False) as arr:
            assert arr['token_ids'].tolist() == g['response_token_ids']
            off = arr['response_token_offsets'] if 'response_token_offsets' in arr else np.column_stack((arr['token_start'], arr['token_end']))
            assert off.tolist() == g['response_token_offsets']
            lb = arr['lb'] if 'lb' in arr else arr['lookback_features']
            nll = arr['nll'] if 'nll' in arr else arr['token_nll']
            assert lb.shape == (len(off), 784) and nll.shape == (len(off),)
            assert lb.dtype == nll.dtype == np.float32
            assert np.isfinite(lb).all() and np.isfinite(nll).all()
            assert (lb >= 0).all() and (lb <= 1).all() and (nll >= 0).all()
            x = np.column_stack((lb, nll)).astype(np.float32)
        for path in (p, side):
            self.files[str(path.resolve())] = sha(path)
        return x

    def pack(self, source, split):
        source_rows = lines(source/'data/inputs.jsonl')
        groups, pairs = defaultdict(set), defaultdict(list)
        for row in source_rows:
            groups[row['group_id']].add(row['split'])
            pairs[row['question_id']].append(row['condition'])
        assert all(len(ss) == 1 for ss in groups.values())
        assert all(sorted(cc) == ['complete', 'partial'] for cc in pairs.values())
        rows = [r for r in source_rows if r['split'] == split]
        annotations = lines(source/'data'/f'annotations_{split}.jsonl')
        gold = {a['row_id']: a for a in annotations}
        assert len(gold) == len(annotations) == len(rows)
        assert set(gold) == {r['row_id'] for r in rows}
        safe_path = source/'data'/f'safe_refusals_{split}.json'
        policy = read(source/'data/question_label_policy.json')
        assert policy['status'] == 'reviewed_frozen'
        assert sha(safe_path) == policy['reviewed_safe_refusal_files_sha256'][split]
        safe_doc = read(safe_path)
        if 'source_annotation_sha256' in safe_doc:
            assert safe_doc['source_annotation_sha256'] == sha(source/'data'/f'annotations_{split}.jsonl')
        safe = set(safe_doc['safe_refusal_item_ids'])
        result = {key: [] for key in ('items', 'tokens', 'windows', 'regions')}
        matrix = []
        for row in rows:
            path = source/'data/generation_records'/(row['row_id']+'.json')
            g, a = read(path), gold[row['row_id']]
            assert a['source_generation_sha256'] == sha(path)
            assert not a.get('token_scores_viewed', False)
            for key in ('row_id', 'question_id', 'split', 'condition'):
                assert g[key] == row[key]
            item, tokens, windows, regions, has_range = label_geometry(row, g, a, safe)
            result['items'].append(item)
            result['tokens'].extend(tokens)
            result['windows'].extend(windows)
            result['regions'].extend(regions)
            if has_range:
                x = self.features(source, g, path)
                matrix.extend(x[w['raw_token_indices']].mean(axis=0) for w in windows)
        result['matrix'] = np.asarray(matrix, np.float32)
        assert result['matrix'].shape == (len(result['windows']), 785)
        result['questions'] = len(rows)//2
        return result


def merge(a, b):
    return {**{k: a[k]+b[k] for k in ('items', 'tokens', 'windows', 'regions')},
            'matrix': np.concatenate((a['matrix'], b['matrix'])), 'questions': a['questions']+b['questions']}


def coverage(pack):
    return {'questions': pack['questions'], 'groups': len({a['group_id'] for a in pack['items']}),
            'planned_answers': len(pack['items']), 'item_evaluable': sum(a['main_eligible'] for a in pack['items']),
            'risk_answers': sum(a['gold'] == 1 for a in pack['items']),
            'asserted_evaluable': sum(a['asserted_eligible'] for a in pack['items']),
            'reviewed_safe_refusals': sum(a['reviewed_safe_refusal'] for a in pack['items']),
            'total_bpe_tokens': len(pack['tokens']), 'eligible_tokens': sum(t['main_eligible'] for t in pack['tokens']),
            'risk_tokens': sum(t['gold'] == 1 for t in pack['tokens']),
            'failed_parse_answers': sum(not a['parse_ok'] for a in pack['items']),
            'candidate_windows': len(pack['windows']),
            'eligible_windows': sum(w['main_eligible'] for w in pack['windows']),
            'risk_windows': sum(w['gold'] == 1 for w in pack['windows'])}


def probability(model, x):
    """Use saved coefficients, with explicit float32 preprocessing; no fit calls."""
    x = x[:, model['columns']].copy()
    x -= model['scaler'].mean_
    x /= model['scaler'].scale_
    return expit(x @ model['model'].coef_[0] + model['model'].intercept_[0])


def item_max(pack, window_scores):
    parts = defaultdict(list)
    for row, score in zip(pack['windows'], window_scores):
        parts[row['item_ids'][0]].append(float(score))
    return np.asarray([max(parts[i['item_id']]) if parts[i['item_id']] else np.nan for i in pack['items']])


def measures(pack, values, thresholds):
    av = item_max(pack, values)
    wi = np.asarray([w['main_eligible'] for w in pack['windows']])
    ai = np.asarray([a['main_eligible'] for a in pack['items']])
    assert np.isfinite(values).all() and np.isfinite(av[ai]).all()
    wp, ap = values >= thresholds['window']['threshold'], av >= thresholds['answer']['threshold']
    window = counts([w['gold'] for w in pack['windows'] if w['main_eligible']], wp[wi], values[wi], 'windows')
    answer = counts([a['gold'] for a in pack['items'] if a['main_eligible']], ap[ai], av[ai], 'answers')
    union = {key for w, p in zip(pack['windows'], wp) if p for key in w['token_keys']}
    tt = [t for t in pack['tokens'] if t['main_eligible']]
    highlight = counts([t['gold'] for t in tt], [t['token_key'] in union for t in tt])
    highlight.update(marked_token_fraction=sum(t['token_key'] in union for t in tt)/len(tt),
                     risk_regions=len(pack['regions']),
                     risk_regions_any_hit=sum(bool(set(r['token_keys']) & union) for r in pack['regions']),
                     risk_regions_fully_hit=sum(bool(r['token_keys']) and set(r['token_keys']) <= union for r in pack['regions']))
    extra = {}
    for label, select in [('safe_refusals', lambda a: a['reviewed_safe_refusal']),
                          ('supported_assertions', lambda a: a['asserted_eligible'] and a['gold'] == 0)]:
        jj = [j for j, a in enumerate(pack['items']) if select(a)]
        rids = {pack['items'][j]['row_id'] for j in jj}
        lexical = [t for t in pack['tokens'] if t['row_id'] in rids and t['lexical']]
        marked = sum(t['token_key'] in union for t in lexical)
        extra[label] = {'answers': len(jj), 'answer_false_alarms': int(ap[jj].sum()),
                        'lexical_tokens': len(lexical), 'highlighted_tokens': marked}
    return {'windows': window, 'answers': answer, 'highlight_tokens': highlight}, extra, av, wp, ap


def compare_saved(pack, score_rows, answer_rows, model, name, metrics):
    assert [w['window_key'] for w in score_rows] == [w['window_key'] for w in pack['windows']]
    assert [a['item_id'] for a in answer_rows] == [a['item_id'] for a in pack['items']]
    for expected, actual in zip(pack['windows'], score_rows):
        for k, v in expected.items():
            assert actual[k] == v, (name, expected['window_key'], k)
    for expected, actual in zip(pack['items'], answer_rows):
        for k in ('item_id', 'row_id', 'question_id', 'group_id', 'split', 'condition', 'category',
                  'text', 'start', 'end', 'gold', 'main_eligible', 'reviewed_safe_refusal'):
            assert expected[k] == actual[k], (name, expected['item_id'], k)
    saved = np.asarray([w['scores'][name] for w in score_rows])
    actual = probability(model, pack['matrix'])
    close(actual, saved, name+'.window probabilities', atol=1e-10)
    rebuilt, extra, av, wp, ap = measures(pack, actual, model['thresholds'])
    assert [bool(v) for v in wp] == [w['predictions'][name] for w in score_rows]
    for j, row in enumerate(answer_rows):
        if np.isfinite(av[j]):
            close(av[j], row['scores'][name], name+'.answer max')
            assert bool(ap[j]) == row['predictions'][name]
        else:
            assert row['scores'][name] is None and row['predictions'][name] is None
            assert not row['main_eligible']
    for unit in rebuilt:
        assert_metrics(rebuilt[unit], metrics[unit], name+'.'+unit)
    return {'counts': rebuilt, 'descriptive_false_alarms': extra,
            'maximum_score_difference': float(np.max(np.abs(actual-saved))),
            'missing_known_window_scores': 0, 'missing_known_answer_scores': 0}


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            'defined_draws': len(finite)}


def bootstrap(rows, config, reported):
    groups = sorted({r['group_id'] for r in rows})
    assert reported['groups'] == len(groups) == 49
    assert reported['seed'] == config['seed'] and reported['draws'] == config['draws'] == 2000
    by_group = {g: [r for r in rows if r['group_id'] == g and r['main_eligible']] for g in groups}
    count = np.zeros((len(groups), len(METHODS), 3), np.int64)
    for i, g in enumerate(groups):
        for j, name in enumerate(METHODS):
            for row in by_group[g]:
                y, p = row['gold'], row['predictions'][name]
                assert p is not None
                count[i, j] += (int(y == 1 and p), int(y == 0 and p), int(y == 1 and not p))
    sampled = np.random.default_rng(config['seed']).integers(0, len(groups), (config['draws'], len(groups)))
    # Direct indexing/summing, independent of evaluator's frequency/einsum path.
    totals = count[sampled].sum(axis=1)
    tp, fp, fn = totals[:, :, 0], totals[:, :, 1], totals[:, :, 2]
    denominator = 2*tp+fp+fn
    f1 = np.divide(2*tp, denominator, out=np.full(tp.shape, np.nan), where=denominator > 0)
    result = {'f1': {}, 'expanded_minus_legacy': {}}
    for j, name in enumerate(METHODS):
        ci = interval(f1[:, j]); result['f1'][name] = ci
        assert_metrics(ci, reported['f1'][name], name+'.bootstrap')
    for f in FEATURES:
        ci = interval(f1[:, METHODS.index('expanded_'+f)]-f1[:, METHODS.index('legacy_'+f)])
        result['expanded_minus_legacy'][f] = ci
        assert_metrics(ci, reported['expanded_minus_legacy'][f], f+'.difference bootstrap')
    return result


def run():
    out = ROOT/'results'
    assert (out/'test_complete17.json').exists(), 'Do not read real cohorts until final test is complete'
    complete, freeze = read(out/'test_complete17.json'), read(out/'fit_freeze17.json')
    assert complete['fit_freeze_sha256'] == sha(out/'fit_freeze17.json')
    verify_tree(out, complete['files_sha256']); verify_tree(out, freeze['files_sha256'])
    assert list(freeze['methods']) == list(METHODS) and freeze['primary'] == 'expanded_lb'
    assert freeze['test_labels_used'] is False and complete['retuning_allowed'] is False
    started, test_started = read(out/'fit_started17.json'), read(out/'test_started17.json')
    assert started['test_labels_used'] is False and test_started['retuning_allowed'] is False
    assert test_started['fit_freeze_sha256'] == sha(out/'fit_freeze17.json')
    moments = [datetime.fromisoformat(x['utc']) for x in (started, freeze, test_started, complete)]
    assert moments == sorted(moments)
    snapshot = read(out/'source_snapshot17.json')
    assert started['source_snapshot_sha256'] == sha(out/'source_snapshot17.json')
    verify_tree(Path('.'), snapshot['code'])
    gold_freeze = read(NEW/'data/annotation_freeze.json')
    input_freeze = read(NEW/'data/input_freeze.json')
    verify_tree(NEW, gold_freeze['files_sha256']); verify_tree(NEW, input_freeze['files_sha256'])
    verify_tree(ROOT.parent, read(NEW/'data/legacy_files_snapshot.json')['files_sha256'])
    checks = {'annotation_freeze_sha256': NEW/'data/annotation_freeze.json',
              'input_freeze_sha256': NEW/'data/input_freeze.json',
              'generation_manifest_sha256': NEW/'data/generation_manifest.json',
              'feature_manifest_sha256': ROOT/'data/feature_manifest.json',
              'legacy_snapshot_sha256': NEW/'data/legacy_files_snapshot.json'}
    for key, path in checks.items():
        assert snapshot['sources'][key] == sha(path), key
    manifest = read(ROOT/'data/feature_manifest.json')
    assert manifest['complete'] and manifest['completed_count'] == len(manifest['records']) == 800
    assert manifest['labels_read'] is False and manifest['response_regenerated'] is False
    assert digest(manifest['extraction_signature']) == manifest['extraction_signature_sha256']
    verify_tree(Path('.'), manifest['extraction_signature']['code_sha256'])
    for record in manifest['records'].values():
        assert sha(ROOT/record['npz']) == record['npz_sha256']
        assert sha(ROOT/record['json']) == record['json_sha256']
    conf = read(ROOT/'protocol.json')
    assert conf['C'] == .01 and conf['target_loss_mass'] == 3854 and conf['width'] == 4
    assert conf['primary_method'] == 'expanded_lb' and conf['seed'] == 20260913
    rebuilder = Rebuild(manifest)
    legacy, new = rebuilder.pack(OLD, 'train'), rebuilder.pack(NEW, 'train')
    validation = rebuilder.pack(NEW, 'validation')
    fit_files = dict(rebuilder.files)
    test = rebuilder.pack(NEW, 'test')
    expanded = merge(legacy, new)
    assert (legacy['questions'], expanded['questions'], validation['questions'], test['questions']) == (120, 421, 49, 50)
    assert sum(t['main_eligible'] for t in legacy['tokens']) == 3854
    group_sets = [{a['group_id'] for a in p['items']} for p in (expanded, validation, test)]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    assert fit_files == read(out/'feature_files_fit.json')
    expected_test_files = {k: v for k, v in rebuilder.files.items() if k not in fit_files}
    assert expected_test_files == read(out/'feature_files_test.json')
    models = pickle.loads((out/'frozen_models.pkl').read_bytes())
    assert list(models) == list(METHODS)
    vm, tm, fm = read(out/'validation_metrics.json'), read(out/'metrics_test.json'), read(out/'fit_metrics.json')
    for pack, actual, label in [(legacy, fm['coverage']['legacy_train'], 'legacy'),
                                 (new, fm['coverage']['new_train'], 'new'),
                                 (validation, vm['coverage'], 'validation'), (test, tm['coverage'], 'test')]:
        assert_metrics(coverage(pack), actual, label+'.coverage')
    expanded_cov = coverage(expanded)
    assert_metrics({k: expanded_cov[k] for k in fm['coverage']['expanded_train']},
                   fm['coverage']['expanded_train'], 'expanded.coverage')
    vw, va = lines(out/'window_scores_validation.jsonl'), lines(out/'answer_scores_validation.jsonl')
    tw, ta = lines(out/'window_scores_test.jsonl'), lines(out/'answer_scores_test.jsonl')
    result = {'status': 'passed', 'model_checks': {}, 'validation': {}, 'test': {}, 'paired_bootstrap': {}}
    for name, model in models.items():
        c, f = name.split('_', 1)
        pack = legacy if c == 'legacy' else expanded
        ix = [j for j, w in enumerate(pack['windows']) if w['main_eligible']]
        rows = [pack['windows'][j] for j in ix]
        columns = list(range(784)) if f == 'lb' else [784] if f == 'nll' else list(range(785))
        assert model['columns'].tolist() == columns
        assert model['fit_window_keys'] == [w['window_key'] for w in rows]
        assert set(model['fit_groups']) == {w['group_id'] for w in rows}
        assert model['C'] == model['model'].C == .01 and model['target_loss_mass'] == 3854
        assert model['model'].solver == 'liblinear' and model['model'].penalty == 'l2'
        assert model['model'].max_iter == 2000 and model['model'].random_state == 20260913
        assert model['model'].classes_.tolist() == [0, 1] and model['components'] is None
        assert int(model['model'].n_iter_.max()) < 2000
        b, loss, factors = weights(rows, 3854)
        close(b, model['base_weights'], name+'.base weights')
        close(loss, model['loss_weights'], name+'.loss weights')
        close(factors, model['class_factors'], name+'.class factors')
        x = pack['matrix'][ix][:, columns].astype(np.float64)
        # StandardScaler casts sample_weight to X.dtype (float32) before accumulation.
        sb = b.astype(np.float32).astype(np.float64)
        mean = np.average(x, axis=0, weights=sb)
        variance = np.average((x-mean)**2, axis=0, weights=sb)
        close(mean, model['scaler'].mean_, name+'.train-only mean', atol=2e-8)
        close(variance, model['scaler'].var_, name+'.train-only variance', atol=2e-8)
        close(sb.sum(), model['scaler'].n_samples_seen_, name+'.scaler weight mass', atol=1e-3)
        eps = np.finfo(np.float64).eps
        constant = variance <= sb.sum()*eps*variance + (sb.sum()*mean*eps)**2
        expected_scale = np.sqrt(variance)
        expected_scale[constant] = 1.
        close(expected_scale, model['scaler'].scale_, name+'.train-only scale', atol=2e-8)
        own, _, _, _, _ = measures(pack, probability(model, pack['matrix']), model['thresholds'])
        common, _, _, _, _ = measures(legacy, probability(model, legacy['matrix']), model['thresholds'])
        for unit in own:
            assert_metrics(own[unit], fm['own_training'][name][unit], name+'.own fit.'+unit)
            assert_metrics(common[unit], fm['common_legacy_training'][name][unit], name+'.common fit.'+unit)
        validation_scores = np.asarray([w['scores'][name] for w in vw])
        av = item_max(validation, validation_scores)
        for unit, records, scores in [('window', validation['windows'], validation_scores), ('answer', validation['items'], av)]:
            jj = [j for j, r in enumerate(records) if r['main_eligible']]
            best = threshold([records[j]['gold'] for j in jj], scores[jj])
            assert best['threshold'] == model['thresholds'][unit]['threshold'], (name, unit, 'Threshold differs')
            assert_metrics(best, model['thresholds'][unit], name+'.validation threshold.'+unit)
            assert_metrics(best, freeze['thresholds'][name][unit], name+'.freeze threshold.'+unit)
        result['model_checks'][name] = {'fit_windows': len(rows), 'fit_groups': len(model['fit_groups']),
                     'total_loss_weight': float(loss.sum()), 'dimensions': len(columns),
                     'maximum_mean_difference': float(np.max(np.abs(mean-model['scaler'].mean_))),
                     'maximum_variance_difference': float(np.max(np.abs(variance-model['scaler'].var_)))}
        result['validation'][name] = compare_saved(validation, vw, va, model, name, vm['methods'][name])
        result['test'][name] = compare_saved(test, tw, ta, model, name, tm['methods'][name])
        for category, category_metrics in tm['categories'].items():
            for unit, saved in [('windows', tw), ('answers', ta)]:
                rr = [r for r in saved if r['category'] == category and r['main_eligible']]
                m = counts([r['gold'] for r in rr], [r['predictions'][name] for r in rr],
                           [r['scores'][name] for r in rr], unit)
                assert_metrics(m, category_metrics[name][unit], name+'.'+category+'.'+unit)
    for unit, rows in [('windows', tw), ('answers', ta)]:
        result['paired_bootstrap'][unit] = bootstrap(rows, conf['bootstrap'], tm['paired_bootstrap'][unit])
    result.update(auditor_sha256=sha(__file__), fit_freeze_sha256=sha(out/'fit_freeze17.json'),
                  test_complete_sha256=sha(out/'test_complete17.json'),
                  checks=['source/feature/model/prediction hashes unchanged', 'fit before test freeze chronology',
                     'legacy train only; R16 question pairs and event groups not split',
                     'same mass3854; train-only scaler and class weights reconstructed',
                     'all six probabilities independently calculated from saved coefficients',
                     'two thresholds chosen on validation only', 'raw BPE4 mean/OR and gold-independent candidates',
                     'safe refusals scored by all-window max', 'window/answer/union-token counts and paired group CIs'],
                  limitations=['No model refitting or new tests.', 'Labels remain assistant judgments.',
                     'Window/highlight results are delayed until the window ends; answer max uses full output.',
                     'Audit checks consistency, not whether source statements or assistant gold are true.'])
    path = out/'INDEPENDENT_AUDIT17.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({'status': 'passed', 'models': len(models), 'path': str(path)}, ensure_ascii=False))


def self_test():
    row = dict(row_id='synthetic', question_id='q', group_id='g', split='train', condition='complete', category='x')
    g = dict(response='12, bad.', response_token_ids=[1, 2, 3, 4, 5, 6],
             response_token_offsets=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 7], [7, 8]],
             items=[dict(item_id='synthetic__1', text='12, bad.', start=0, end=8, parse_ok=True)])
    a = dict(g['items'][0], original_stance='asserted', original_risk=1, localization_status='resolved',
             risk_spans=[dict(start=4, end=7, text='bad')])
    item, tt, ww, rr, has_range = label_geometry(row, g, a, set())
    assert has_range and [w['gold'] for w in ww] == [0, 1, 1]
    assert ww[0]['raw_token_indices'] == [0, 1, 2, 3]
    a.update(original_stance='abstained', original_risk=None, localization_status='excluded', risk_spans=[])
    refusal, rt, rw, _, _ = label_geometry(row, g, a, {'synthetic__1'})
    assert [w['raw_token_indices'] for w in rw] == [w['raw_token_indices'] for w in ww]
    assert refusal['gold'] == 0 and all(w['gold'] is None for w in rw)
    assert all(t['gold'] is None for t in rt)
    assert item_max({'items': [refusal], 'windows': rw}, [.1, .99, .2]).tolist() == [.99]
    assert threshold([0, 1, 1], [.9, .8, .8])['threshold'] == .8
    br = [dict(group_id='a', condition='complete', item_ids=['a1'], gold=0),
          dict(group_id='a', condition='partial', item_ids=['a2'], gold=1),
          dict(group_id='b', condition='complete', item_ids=['b1'], gold=1),
          dict(group_id='b', condition='complete', item_ids=['b1'], gold=0)]
    b, w, _ = weights(br, 3854)
    assert np.isclose(w[:2].sum(), 1927) and np.isclose(w[2:].sum(), 1927)
    assert counts([0, 1, 1], [1, 1, 0])['f1'] == .5
    print('Synthetic audit checks passed; no real labels/features/results opened.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        self_test() if args.self_test else run()
