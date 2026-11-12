"""Independent R19 artifact audit, never refits classifiers or reads old test labels.

Reuses earlier independent geometry/count utilities, not the runtime's feature,
prediction or metric functions. --self-test is synthetic; --geometry-smoke reads
only the already frozen R16 train cohort and the R18 window catalog.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import importlib.util
import json
import pickle
import os
import re
import hashlib

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
R18 = ROOT.parent/'round18_input_information_diagnostics'
NEW = ROOT.parent/'round16_dataset_expansion'
UTILITY = R18/'src/audit18.py'
spec = importlib.util.spec_from_file_location('independent_r18_audit_utilities', UTILITY)
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
u = prior.u
read, lines, sha, close = u.read, u.lines, u.sha, u.close
METHODS = ('base', 'base_mmd', 'base_ecs', 'base_pks', 'base_all', 'signals_only')
BLOCKS = {'base': ('base',), 'base_mmd': ('base', 'mmd'),
          'base_ecs': ('base', 'ecs'), 'base_pks': ('base', 'pks'),
          'base_all': ('base', 'mmd', 'ecs', 'pks'),
          'signals_only': ('mmd', 'ecs', 'pks')}
WIDTHS = {'base': 785, 'mmd': 2, 'ecs': 4, 'pks': 4}
# Alter only the independently imported auditor registry, never runtime files.
prior.METHODS = METHODS


def train_rows(path):
    pattern = re.compile(r'(?<!\\)"split"\s*:\s*"([^"\\]+)"')
    selected = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip(): continue
            matches = list(pattern.finditer(line)); assert len(matches) == 1
            if matches[0].group(1) == 'train': selected.append(json.loads(line))
    return selected


def content_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest() if isinstance(value, str) else u.digest(value)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', 'utf-8')


def token_signals(a):
    """Explicit contiguous layer slices; source aggregation precedes windows."""
    mmd = a['token_lumina_mmd']
    out = {'mmd': np.column_stack((mmd.mean(1), mmd.max(1)))}
    out['ecs'] = np.column_stack([
        a['token_redeep_ecs'][:, band*196:(band+1)*196].mean(1)
        for band in range(4)])
    out['pks'] = np.column_stack([
        a['token_redeep_pks'][:, band*7:(band+1)*7].mean(1)
        for band in range(4)])
    return out


def load_arrays(base, manifest, rid, g, gp, required, files):
    entry = manifest['records'][rid]
    path, side = base/entry['npz'], base/entry['json']
    meta = read(side)
    assert sha(path) == entry['npz_sha256'] == meta['arrays_sha256']
    assert sha(side) == entry['json_sha256']
    assert meta['source_generation_sha256'] == sha(gp)
    assert meta['extraction_signature_sha256'] == manifest['extraction_signature_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert z['token_ids'].tolist() == g['response_token_ids']
        assert z['response_token_offsets'].tolist() == g['response_token_offsets']
        result = {key: z[key].copy() for key in required}
    n = len(g['response_token_ids'])
    for key, width in required.items():
        value = result[key]
        assert value.shape == ((n,) if width is None else (n, width)), (rid, key)
        assert value.dtype == np.float32 and np.isfinite(value).all(), (rid, key)
    for path in (path, side): files[str(path.resolve())] = sha(path)
    return result


def rebuild(features=True):
    rows = train_rows(NEW/'data/inputs.jsonl')
    labels = lines(NEW/'data/annotations_train.jsonl')
    ann = {a['row_id']: a for a in labels}
    assert len(ann) == len(labels) == len(rows) == 602
    safe_file = NEW/'data/safe_refusals_train.json'
    assert sha(safe_file) == read(NEW/'data/question_label_policy.json')['reviewed_safe_refusal_files_sha256']['train']
    safe = set(read(safe_file)['safe_refusal_item_ids'])
    generations = read(NEW/'data/generation_manifest.json')['record_sha256']
    pack = {k: [] for k in ('items', 'tokens', 'windows', 'regions')}
    design = {k: [] for k in WIDTHS}; files = {'r18': {}, 'r19': {}}
    lengths, pairs = {}, defaultdict(list)
    if features:
        manifests = [read(p/'data/feature_manifest.json') for p in (R18, ROOT)]
        for m in manifests:
            assert m['complete'] and m['completed_count'] == 602
            assert set(m['records']) == {r['row_id'] for r in rows}
    for row in rows:
        rid = row['row_id']; pairs[row['question_id']].append(row['condition'])
        gp = NEW/'data/generation_records'/(rid+'.json'); g, a = read(gp), ann[rid]
        assert g['split'] == 'train' and sha(gp) == generations[rid] == a['source_generation_sha256']
        assert not a.get('token_scores_viewed', False)
        item, tt, ww, rr, _ = u.label_geometry(row, g, a, safe)
        pack['items'].append(item); pack['tokens'].extend(tt); pack['windows'].extend(ww)
        for r in rr: r.update(row_id=rid, group_id=row['group_id'])
        pack['regions'].extend(rr); lengths[rid] = len(g['response_token_ids'])
        if not features: continue
        old = load_arrays(R18, manifests[0], rid, g, gp, {'lb': 784, 'nll': None}, files['r18'])
        assert (old['lb'] >= 0).all() and (old['lb'] <= 1).all() and (old['nll'] >= 0).all()
        new = load_arrays(ROOT, manifests[1], rid, g, gp,
              {'token_lumina_mmd': 2, 'token_redeep_ecs': 784, 'token_redeep_pks': 28}, files['r19'])
        signals = token_signals(new)
        signals['base'] = np.column_stack((old['lb'], old['nll']))
        for window in ww:
            raw = window['raw_token_indices']
            for key in design: design[key].append(signals[key][raw].mean(0))
    assert all(sorted(values) == ['complete', 'partial'] for values in pairs.values())
    pack['questions'] = len(pairs)
    assert pack['questions'] == 301 and len(pack['windows']) == 12222
    assert pack['windows'] == lines(R18/'results/candidate_windows.jsonl')
    assert sum(w['main_eligible'] for w in pack['windows']) == 9526
    if features:
        design = {k: np.asarray(v, np.float32) for k, v in design.items()}
        for k, width in WIDTHS.items():
            assert design[k].shape == (12222, width) and np.isfinite(design[k]).all()
    return pack, design, files, lengths


def raw_matrix(name, design, ix=None):
    blocks = [design[k] if ix is None else design[k][ix] for k in BLOCKS[name]]
    return blocks[0] if len(blocks) == 1 else np.column_stack(blocks)


def predict(model, raw):
    z = raw.copy(); z -= model['scaler'].mean_; z /= model['scaler'].scale_
    z = z.astype(np.float32)
    return expit(z @ model['model'].coef_[0]+model['model'].intercept_[0])


def inspect_fit(model, design, ix, rows):
    name = model['method']; assert model['kind'] == 'lr'
    assert model['design_blocks'] == list(BLOCKS[name])
    assert np.array_equal(model['fit_ix'], ix)
    assert model['fit_keys'] == model['fit_window_keys'] == [r['window_key'] for r in rows]
    y = np.asarray([r['gold'] for r in rows], int); assert np.array_equal(y, model['fit_y'])
    b, loss, factors = u.weights(rows, 3854)
    for actual, expected in [(model['base_weights'], b), (model['loss_weights'], loss), (model['class_factors'], factors)]: close(actual, expected)
    close(model['base_weight_sum'], b.sum()); close(model['loss_weight_sum'], 3854)
    close(model['loss_mass_by_label'], np.bincount(y, weights=loss, minlength=2))
    assert model['fit_only'] and model['target_loss_mass'] == 3854 and model['C'] == .01
    raw = raw_matrix(name, design, ix)
    assert raw.shape[1] == model['input_width']
    weight = b.astype(raw.dtype).astype(float)
    mean = np.average(raw.astype(float), axis=0, weights=weight)
    var = np.average((raw.astype(float)-mean)**2, axis=0, weights=weight)
    close(model['scaler'].mean_, mean, 'fit scaler mean', atol=2e-8)
    close(model['scaler'].var_, var, 'fit scaler variance', atol=2e-8)
    close(model['scaler'].n_samples_seen_, weight.sum(), 'scaler mass')
    eps = np.finfo(float).eps
    constant = var <= weight.sum()*eps*var+(weight.sum()*mean*eps)**2
    scale = np.sqrt(var); scale[constant] = 1
    close(model['scaler'].scale_, scale, 'fit scaler scale', atol=2e-8)
    c = model['model']
    assert c.C == .01 and c.solver == 'liblinear' and c.penalty == 'l2'
    assert c.random_state == 20260913 and c.max_iter == 2000 and c.n_iter_.max() < 2000
    assert c.classes_.tolist() == [0, 1]
    for a, z in [('coefficients_standardized', c.coef_), ('intercept_standardized', c.intercept_), ('iterations', c.n_iter_)]:
        assert np.array_equal(model[a], z)
    fit_scores = predict(model, raw)
    assert np.array_equal(fit_scores, model['final_fit_scores'])
    return {'fit_windows': len(rows), 'input_width': raw.shape[1], 'loss_mass': float(loss.sum()),
            'fit_only_scaler_and_weights_verified': True, 'fit_probabilities_exact': True}


def baseline(model, design, values, fold, report):
    old_path = R18/'results'/f'fold_{fold}_frozen.pkl'
    old = pickle.loads(old_path.read_bytes())['models']['mean_lr']
    for key in ('fit_ix', 'fit_y', 'base_weights', 'loss_weights', 'class_factors'):
        assert np.array_equal(model[key], old[key]), ('baseline', fold, key)
    assert model['fit_window_keys'] == old['fit_window_keys']
    for key in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
        assert np.array_equal(getattr(model['scaler'], key), getattr(old['scaler'], key))
    for key in ('coef_', 'intercept_', 'classes_', 'n_iter_'):
        assert np.array_equal(getattr(model['model'], key), getattr(old['model'], key))
    old_values = predict(old, design['base'])
    assert np.array_equal(values, old_values) and model['thresholds'] == old['thresholds']
    assert report['passed'] and report['coefficients_scaler_weights_indices_thresholds_exact']
    assert report['all_candidate_scores_max_absolute_difference'] == 0
    assert report['r18_fold_sha256'] == sha(old_path)
    return {'exact': True, 'all_candidate_probability_maximum_difference': 0.0,
            'r18_fold_sha256': sha(old_path)}


def verify_sources():
    out = ROOT/'results'; complete = read(out/'complete19.json'); snap = read(out/'source_snapshot19.json')
    assert complete['original_validation_or_test_used'] is False
    u.verify_tree(out, complete['files_sha256']); u.verify_tree(Path('.'), snap['code_sha256'])
    assert sha(R18/'results/complete18.json') == snap['r18_complete_sha256']
    old_complete = read(R18/'results/complete18.json')
    u.verify_tree(R18/'results', old_complete['files_sha256'])
    old_snap = read(R18/'results/source_snapshot18.json')
    assert snap['r18_source_snapshot'] == old_snap
    u.verify_tree(Path('.'), old_snap['code_sha256'])
    af = read(NEW/'data/annotation_freeze.json')
    assert old_snap['annotation_freeze_sha256'] == sha(NEW/'data/annotation_freeze.json')
    assert af['input_freeze_sha256'] == sha(NEW/'data/input_freeze.json')
    u.verify_tree(NEW, af['files_sha256'])
    u.verify_tree(NEW, read(NEW/'data/input_freeze.json')['files_sha256'])
    u.verify_tree(ROOT.parent, read(NEW/'data/legacy_files_snapshot.json')['files_sha256'])
    for base, expected in [(ROOT, snap['feature_manifest_sha256']), (R18, old_snap['feature_manifest_sha256'])]:
        p = base/'data/feature_manifest.json'; assert sha(p) == expected
        fm = read(p); assert fm['complete'] and fm['completed_count'] == 602
        assert u.digest(fm['extraction_signature']) == fm['extraction_signature_sha256']
        u.verify_tree(Path('.'), fm['extraction_signature']['code_sha256'])
    assert sha(ROOT/'data/fold_assignment.json') == sha(R18/'data/fold_assignment.json')
    return complete, snap


def verify_plans():
    """Independently rebuild exact masks/streams with the frozen CPU tokenizer."""
    from transformers import AutoTokenizer
    model_path = ROOT.parent/'models/Qwen2.5-7B-Instruct-bnb-4bit'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pm = read(ROOT/'data/perturbation_manifest.json')
    assert pm['complete'] and pm['rows'] == 602 and pm['interventions'] == 1204
    assert pm['plans_sha256'] == sha(ROOT/'data/perturbation_plans.jsonl')
    assert pm['donor_pool_sha256'] == sha(ROOT/'data/donor_pool.json')
    assert pm['source_inputs_sha256'] == sha(NEW/'data/inputs.jsonl')
    assert pm['extract19_sha256'] == sha(ROOT/'src/extract19.py')
    assert pm['tokenizer_config_sha256'] == sha(model_path/'tokenizer_config.json')
    donors = read(ROOT/'data/donor_pool.json')
    review = read(ROOT/'data/donor_independence_review.json')
    assert donors['review_sha256'] == sha(ROOT/'data/donor_independence_review.json')
    assert donors['candidate_file_sha256'] == review['candidate_file_sha256'] == sha(ROOT/'data/donor_candidates.json')
    assert donors['inventory_sha256'] == review['source_inventory_sha256'] == sha(ROOT/'data/train_visible_inventory.json')
    r6path = ROOT.parent/'round6_evidence_grounding/data/inputs.jsonl'
    assert donors['source_r6_inputs_sha256'] == sha(r6path)
    assert donors['source_r16_inputs_sha256'] == sha(NEW/'data/inputs.jsonl')
    originals = {r['row_id']: r for r in train_rows(r6path)}
    allowed = {r['donor_id'] for r in review['records'] if r['status'] == 'pass'}
    assert len(allowed) == donors['count'] == len(donors['donors'])
    assert {d['donor_id'] for d in donors['donors']} == allowed
    seen = set()
    for donor in donors['donors']:
        assert donor['source_split'] == 'train'
        pair = (donor['title'], donor['text']); assert pair not in seen; seen.add(pair)
        raw = json.dumps(dict(title=pair[0], text=pair[1]), ensure_ascii=False, sort_keys=True)
        assert content_digest(raw) == donor['passage_sha256']
        for rid in donor['source_rows']:
            assert any((p['title'], p['text']) == pair for p in originals[rid]['passages'])
    rows, plans = train_rows(NEW/'data/inputs.jsonl'), lines(ROOT/'data/perturbation_plans.jsonl')
    assert [r['row_id'] for r in rows] == [p['row_id'] for p in plans]
    streams = {}; mask_sizes = []; compared = 0
    for row, plan in zip(rows, plans):
        rid = row['row_id']; gp = NEW/'data/generation_records'/(rid+'.json'); g = read(gp)
        assert plan['actual_split'] == 'train' and plan['source_generation_sha256'] == sha(gp)
        assert plan['original_prefix_token_ids'] == g['input_token_ids']
        assert plan['response_token_ids'] == g['response_token_ids']
        question_hash = content_digest(row['questions'][0]); assert plan['question_text_sha256'] == question_hash
        rendered = tokenizer.apply_chat_template([dict(role='system', content=row['system']), dict(role='user', content=row['prompt'])],
                                                 tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == g['input_token_ids']
        prompt_start = rendered.index(row['prompt']); assert rendered.count(row['prompt']) == 1
        prefix, body = row['prompt'].split('\n\nSearch results:\n')
        assert body == '\n\n'.join(f'[{i+1}] {p["title"]}\n{p["text"]}' for i, p in enumerate(row['passages']))
        cursor = prompt_start+len(prefix)+len('\n\nSearch results:\n')
        assert len(plan['interventions']) == 2
        for slot, (passage, intervention) in enumerate(zip(row['passages'], plan['interventions'])):
            header = f'[{slot+1}] {passage["title"]}\n'; assert rendered[cursor:cursor+len(header)] == header
            begin, end = cursor+len(header), cursor+len(header)+len(passage['text'])
            assert rendered[begin:end] == passage['text']; cursor = end+2
            mask = [j for j, (a, b) in enumerate(encoded['offset_mapping']) if begin <= a < b <= end]
            assert mask and mask == list(range(mask[0], mask[-1]+1))
            assert intervention['rendered_body_character_interval'] == [begin, end]
            assert intervention['eligible_body_token_positions'] == mask
            assert intervention['passage_index'] == slot and intervention['title'] == passage['title']
            key = (question_hash, slot)
            if key not in streams:
                ranked = sorted(donors['donors'], key=lambda d: (content_digest(f'{question_hash}\0{slot}\0{d["donor_id"]}'), d['donor_id']))
                text = ''.join(d['title']+'\n'+d['text']+'\n\n' for d in ranked)
                tokens = tokenizer(text, add_special_tokens=False)['input_ids']
                assert not set(tokens)&set(tokenizer.all_special_ids)
                streams[key] = (tokens, content_digest(text), [d['donor_id'] for d in ranked])
            tokens, text_hash, ordered_ids = streams[key]
            assert intervention['donor_stream_total_tokens'] == len(tokens) >= len(mask)
            assert intervention['donor_stream_tokens_used'] == len(mask)
            assert intervention['donor_stream_sha256'] == content_digest(tokens)
            assert intervention['donor_stream_text_sha256'] == text_hash
            assert intervention['replacement_body_token_ids'] == tokens[:len(mask)]
            used = [d['donor_id'] for d in intervention['used_donors']]
            assert used == ordered_ids[:len(used)]
            expected = list(g['input_token_ids'])
            for j, token in zip(mask, tokens): expected[j] = token
            assert expected == intervention['replacement_prefix_token_ids']
            assert content_digest(expected) == intervention['replacement_prefix_sha256']
            changed = [j for j, (a, b) in enumerate(zip(expected, g['input_token_ids'])) if a != b]
            assert changed == intervention['changed_token_positions'] and changed and set(changed) <= set(mask)
            compared += 1; mask_sizes.append(len(mask))
    assert compared == 1204 and len(rows) == 602 and len(streams) == 602
    return {'status': 'passed', 'rows': len(rows), 'interventions': compared, 'distinct_question_slot_streams': len(streams),
            'donors': donors['count'], 'mask_token_min': min(mask_sizes), 'mask_token_max': max(mask_sizes),
            'independent_tokenizer_mask_and_stream_reconstruction': True, 'old_heldout_examples_parsed': False,
            'perturbation_manifest_sha256': sha(ROOT/'data/perturbation_manifest.json'),
            'limit': 'Checks source provenance and exact controlled token replacement; semantic donor review is independently recorded elsewhere.'}


def strata(windows, answers, lengths, actual):
    checked = {}
    selectors = {'category': lambda row: row['category'], 'condition': lambda row: row['condition'],
                 'answer_length': lambda row: '<=16' if lengths[row['row_id']] <= 16 else '17..32' if lengths[row['row_id']] <= 32 else '>32'}
    for field, selector in selectors.items():
        checked[field] = {}
        for label in sorted({selector(a) for a in answers}):
            checked[field][label] = {}
            for name in METHODS:
                checked[field][label][name] = {}
                for unit, records in [('windows', windows), ('answers', answers)]:
                    rr = [r for r in records if r['main_eligible'] and selector(r) == label]
                    expected = u.counts([r['gold'] for r in rr], [r['predictions'][name] for r in rr], unit=unit)
                    u.assert_metrics(expected, actual[field][label][name][unit], '.'.join((field, label, name, unit)))
                    checked[field][label][name][unit] = expected
    return checked


def run():
    out = ROOT/'results'
    assert (out/'complete19.json').exists(), 'Wait for the frozen 30-model run; audit never trains.'
    complete, snap = verify_sources(); config = read(ROOT/'protocol.json')
    assert tuple(complete['methods']) == tuple(config['methods']) == METHODS
    assert config['primary_method'] == 'base_all' and config['primary_contrast'] == 'all_vs_base'
    assert config['width'] == 4 and config['stride'] == 1 and config['folds'] == 5
    pack, design, files, lengths = rebuild()
    assert files == read(out/'feature_files.json') and lengths == read(out/'response_lengths.json')
    assert pack['windows'] == lines(out/'candidate_windows.jsonl')
    summary = read(out/'summary.json'); u.assert_metrics(u.coverage(pack), summary['coverage'], 'coverage')
    assignment = read(ROOT/'data/fold_assignment.json')['groups']
    assert set(assignment) == {a['group_id'] for a in pack['items']} and len(assignment) == 278
    report = read(out/'baseline_reproduction.json'); assert report == summary['baseline_exact_reproduction']
    audit = {'status': 'passed', 'model_count': 30, 'folds': {}, 'baseline_exact_reproduction': {},
             'original_validation_or_test_parsed': False, 'classifiers_refitted': False}
    gid = np.asarray([w['group_id'] for w in pack['windows']]); eligible = np.asarray([w['main_eligible'] for w in pack['windows']])
    all_windows, all_answers, observed = [], [], []
    for fold in range(5):
        stored = pickle.loads((out/f'fold_{fold}_frozen.pkl').read_bytes())
        detail = read(out/f'fold_{fold}_metrics.json'); assert detail == summary['folds'][str(fold)]
        eg = sorted(g for g, f in assignment.items() if f == fold)
        cg = sorted(g for g, f in assignment.items() if f == (fold+1)%5)
        fg = sorted(set(assignment)-set(eg)-set(cg))
        assert not (set(fg)&set(cg) or set(fg)&set(eg) or set(cg)&set(eg))
        for key, values in [('fit_groups', fg), ('calibration_groups', cg), ('evaluation_groups', eg)]: assert stored[key] == values
        tr, tx = prior.subset(pack, fg); ca, cx = prior.subset(pack, cg); ev, ex = prior.subset(pack, eg)
        fit_ix = np.flatnonzero(np.isin(gid, fg)&eligible); fit_rows = [pack['windows'][j] for j in fit_ix]
        wr, ar = lines(out/f'fold_{fold}_window_scores.jsonl'), lines(out/f'fold_{fold}_answer_scores.jsonl')
        assert all(r['fold'] == fold for r in wr+ar) and tuple(stored['models']) == METHODS
        audit['folds'][str(fold)] = {}
        for name in METHODS:
            model = stored['models'][name]
            assert model['method'] == name and model['fold'] == fold
            assert model['fit_groups'] == fg and model['calibration_groups'] == cg and model['evaluation_groups'] == eg
            info = inspect_fit(model, design, fit_ix, fit_rows)
            values = predict(model, raw_matrix(name, design)); ts = model['thresholds']
            assert ts == detail[name]['thresholds']
            for unit, rr, ss in [('window', ca['windows'], values[cx]), ('answer', ca['items'], u.item_max(ca, values[cx]))]:
                ii = [j for j, r in enumerate(rr) if r['main_eligible']]
                best = u.threshold([rr[j]['gold'] for j in ii], ss[ii])
                u.assert_metrics(best, ts[unit], f'{fold}.{name}.{unit}.calibration threshold')
            for stage, pp, ix in [('fit', tr, tx), ('calibration', ca, cx), ('evaluation', ev, ex)]:
                got = u.measures(pp, values[ix], ts)[0]
                for unit in got: u.assert_metrics(got[unit], detail[name][stage][unit], f'{fold}.{name}.{stage}.{unit}')
            info['maximum_saved_probability_difference'] = prior.verify_saved(ev, values[ex], ts, wr, ar, name)
            assert np.array_equal(values[ex], np.asarray([w['scores'][name] for w in wr]))
            if name == 'base':
                audit['baseline_exact_reproduction'][str(fold)] = baseline(model, design, values, fold, report[str(fold)])
                oldw = lines(R18/'results'/f'fold_{fold}_window_scores.jsonl')
                olda = lines(R18/'results'/f'fold_{fold}_answer_scores.jsonl')
                for current, old, key in [(wr, oldw, 'window_key'), (ar, olda, 'item_id')]:
                    assert [r[key] for r in current] == [r[key] for r in old]
                    assert [r['scores']['base'] for r in current] == [r['scores']['mean_lr'] for r in old]
                    assert [r['predictions']['base'] for r in current] == [r['predictions']['mean_lr'] for r in old]
            audit['folds'][str(fold)][name] = info
        observed.extend(eg); all_windows.extend(wr); all_answers.extend(ar)
        print('AUDITED19_FOLD', fold, flush=True)
    assert len(observed) == len(set(observed)) == 278
    assert all_windows == lines(out/'window_scores_oof.jsonl') and all_answers == lines(out/'answer_scores_oof.jsonl')
    assert len(all_windows) == len({w['window_key'] for w in all_windows}) == 12222
    assert len(all_answers) == len({a['item_id'] for a in all_answers}) == 602
    rebuilt = prior.pooled(pack, all_windows, all_answers)
    for name in METHODS:
        actual = summary['methods'][name]
        for unit in ('windows', 'answers', 'highlight_tokens'): u.assert_metrics(rebuilt[name][unit], actual[unit], name+'.'+unit)
        for key in ('safe_refusals', 'safe_refusal_false_positives'): assert rebuilt[name][key] == actual[key]
        wanted, got = rebuilt[name]['conditional_ranking'], actual['conditional_ranking']
        for key in ('answers_with_both_labels', 'peak_hit_answers', 'mean_auroc', 'mean_average_precision'): close(wanted[key], got[key])
        assert {r['item_id']: r for r in wanted['details']} == {r['item_id']: r for r in got['details']}
        for stage in ('fit', 'calibration', 'evaluation'):
            for key in ('f1', 'auroc', 'average_precision'):
                close(np.mean([summary['folds'][str(f)][name][stage]['windows'][key] for f in range(5)]), summary['fold_mean'][name][stage][key])
    audit['pooled'] = rebuilt
    audit['paired_bootstrap'] = {unit: prior.bootstrap(rr, config['bootstrap'], config['primary_contrasts'], summary['paired_bootstrap'][unit])
                                for unit, rr in [('windows', all_windows), ('answers', all_answers)]}
    audit['strata'] = strata(all_windows, all_answers, lengths, summary['strata'])
    audit['perturbation_plans'] = verify_plans()
    verify_sources()
    for paths in files.values(): u.verify_tree(Path('.'), paths)
    audit.update(auditor_sha256=sha(__file__), independent_utilities_sha256={str(UTILITY): sha(UTILITY), str(prior.UTILITY): sha(prior.UTILITY)},
        complete_sha256=sha(out/'complete19.json'),
        limits=['No refitting or original validation/test example inspection.',
                'This checks cached axes/aggregation and artifact consistency, not a second GPU extraction.',
                'Existing assistant labels are not independently verified human truth.',
                'R18-informed reuse is exploratory; fixed-model group CIs exclude refitting and selection uncertainty.'])
    write(out/'INDEPENDENT_AUDIT19.json', audit)
    print('INDEPENDENT_AUDIT19_PASSED', flush=True)


def geometry_smoke():
    pack, _, _, _ = rebuild(features=False)
    assignment = read(ROOT/'data/fold_assignment.json')['groups']; seen = []
    assert sha(ROOT/'data/fold_assignment.json') == sha(R18/'data/fold_assignment.json')
    for fold in range(5):
        eg = {g for g, f in assignment.items() if f == fold}; cg = {g for g, f in assignment.items() if f == (fold+1)%5}
        fg = set(assignment)-eg-cg
        assert not (eg&cg or eg&fg or cg&fg)
        for groups in (fg, cg, eg):
            pp, ix = prior.subset(pack, groups)
            assert all(r['row_id'] in {a['row_id'] for a in pp['items']} for r in pp['regions'])
            assert set(w['gold'] for w in pp['windows'] if w['main_eligible']) == {0, 1}
        rows = [w for w in pack['windows'] if w['main_eligible'] and w['group_id'] in fg]
        b, loss, _ = u.weights(rows, 3854); close(loss.sum(), 3854)
        seen.extend(eg)
    assert len(seen) == len(set(seen)) == 278
    print('GEOMETRY19_SMOKE_PASSED', u.coverage(pack), 'No real features, probabilities or classifier fitted.', flush=True)


def self_test():
    # Different per-token source maxima: mean(max) must be 10, not max(mean)=5.
    source = np.asarray([[10, 0], [0, 10]], np.float32)
    ecs = np.tile(np.arange(784, dtype=np.float32), (2, 1))
    pks = np.tile(np.arange(28, dtype=np.float32), (2, 1))
    got = token_signals({'token_lumina_mmd': source, 'token_redeep_ecs': ecs, 'token_redeep_pks': pks})
    close(got['mmd'].mean(0), [5, 10]); close(got['ecs'][0], [97.5, 293.5, 489.5, 685.5])
    close(got['pks'][0], [3, 10, 17, 24])
    design = {'base': np.zeros((2, 785), np.float32), **got}
    assert [raw_matrix(m, design).shape[1] for m in METHODS] == [785, 787, 789, 789, 795, 10]
    row = dict(row_id='r', question_id='q', group_id='g', split='train', condition='complete', category='test')
    g = {'response': 'a, 12.', 'response_token_ids': [1, 2, 3, 4, 5, 6],
         'response_token_offsets': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
         'items': [dict(item_id='r__1', text='a, 12.', start=0, end=6, parse_ok=True)]}
    a = dict(g['items'][0], original_stance='asserted', original_risk=1, localization_status='resolved', risk_spans=[dict(start=3, end=5, text='12')])
    _, tt, ww, _, _ = u.label_geometry(row, g, a, set())
    assert [t['gold'] for t in tt] == [0, None, None, 1, 1, None]
    assert len(ww) == 3 and all(w['gold'] == 1 and len(w['raw_token_indices']) == 4 for w in ww)
    a.update(original_stance='abstained', original_risk=None, localization_status='excluded', risk_spans=[])
    item, _, ww, _, _ = u.label_geometry(row, g, a, {'r__1'})
    assert item['main_eligible'] and item['gold'] == 0 and all(not w['main_eligible'] for w in ww)
    assert u.item_max({'items': [item], 'windows': ww}, np.asarray([.1, .9, .2])).tolist() == [.9]
    print('SYNTHETIC_AUDIT19_PASSED; source-max order, layer bands, six dimensions, BPE geometry and refusal maximum.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--self-test', action='store_true'); parser.add_argument('--geometry-smoke', action='store_true'); parser.add_argument('--plans-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.self_test: self_test()
        elif args.geometry_smoke: geometry_smoke()
        elif args.plans_only: print(json.dumps(verify_plans(), ensure_ascii=False), flush=True)
        else: run()
