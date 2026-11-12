"""Independent CPU-only checks for ReDeEP/LUMINA plans and completed scores."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
sys.path.insert(0, str(OUT))
from evaluate_extensions import sha, load, save, indexed, canonical_sha


def audit_completed_scores(white, plans, config):
    result = {}
    for method in ['lumina', 'redeep']:
        method_root = white / method
        if not (method_root / 'PREDICTIONS_COMPLETE.json').exists():
            continue
        caches = {}
        relevant = {rid: p for rid, p in plans.items() if method == 'redeep' or p['partition'] != 'fit'}
        for rid, plan in relevant.items():
            path = method_root / 'features' / plan['partition'] / (rid + '.npz')
            receipt = load(path.with_suffix('.json'))
            assert receipt['npz_sha256'] == sha(path)
            assert receipt['config_sha256'] == sha(white / 'CONFIG.json')
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['token_ids'], plan['answer_token_ids'])
                assert np.array_equal(z['offsets'], plan['response_token_offsets'])
                assert np.array_equal(z['predictor_positions'], np.asarray(plan['original_answer_positions']) - 1)
                keys = ['ecs_paper', 'pks_paper'] if method == 'redeep' else ['signals']
                caches[rid] = {key: z[key].astype(np.float64) for key in keys}
                assert all(np.isfinite(v).all() for v in caches[rid].values())
        fit_check = None
        if method == 'redeep':
            fitted = load(method_root / 'FIT.json')
            fit_ids = [rid for rid, p in plans.items() if p['partition'] == 'fit']
            seq = indexed(RUN / 'data/fit/sequences.jsonl')
            y = np.concatenate([seq[rid]['risk_mask'] for rid in fit_ids]).astype(np.float64)
            e = np.concatenate([caches[rid]['ecs_paper'] for rid in fit_ids])
            p = np.concatenate([caches[rid]['pks_paper'] for rid in fit_ids])
            er = np.corrcoef(e.T, 1 - y)[-1, :-1]
            pr = np.corrcoef(p.T, y)[-1, :-1]
            heads = sorted(range(32), key=lambda i: (-er[i], tuple(config['candidate_heads'][i])))[:1]
            layers = sorted(range(32), key=lambda i: (-pr[i], i))[:10]
            assert heads == fitted['selected_head_indices'] and layers == fitted['selected_layers']
            assert np.allclose(er, fitted['head_correlations'], rtol=0, atol=1e-12)
            assert np.allclose(pr, fitted['layer_correlations'], rtol=0, atol=1e-12)
            selected_e, selected_p = e[:, heads].sum(1), p[:, layers].sum(1)
            eb = [float(selected_e.min()), float(selected_e.max())]
            pb = [float(selected_p.min()), float(selected_p.max())]
            assert eb == fitted['ecs_bounds'] and pb == fitted['pks_bounds']
            assert fitted['fit_answers'] == 1292 and fitted['fit_tokens'] == len(y)
            assert fitted['fit_labels_sha256'] == sha(RUN / 'data/fit/sequences.jsonl')
            fit_check = {'raw_fit_tokens': len(y), 'selected_heads': heads, 'selected_layers': layers,
                         'fit_only_pearson_selection_and_minmax_independently_recomputed': True}
        score_checks = {}
        for partition in ['validation', 'test']:
            path = method_root / f'{partition}_predictions.jsonl'
            predictions = indexed(path)
            assert predictions.keys() == {rid for rid, plan in plans.items() if plan['partition'] == partition}
            for rid, pred in predictions.items():
                data = caches[rid]
                assert pred['response_token_offsets'] == plans[rid]['response_token_offsets']
                if method == 'lumina':
                    signals = data['signals']
                    # The port combines its primitives in FP32 before export.
                    # Reproduce that dtype rather than imposing an FP64 identity.
                    expected = (np.float32(.5) * signals[:, 0].astype(np.float32) -
                                np.float32(.5) * signals[:, 1].astype(np.float32)).astype(np.float64)
                    assert np.array_equal(signals[:, 2], expected)
                    scores = signals[:, 2]
                else:
                    es = data['ecs_paper'][:, heads].sum(1)
                    ps = data['pks_paper'][:, layers].sum(1)
                    scores = (ps - pb[0]) / (pb[1] - pb[0]) - .2 * (es - eb[0]) / (eb[1] - eb[0])
                assert np.allclose(scores, pred['token_scores'], rtol=0, atol=1e-12)
                assert abs(float(scores.mean()) - pred['native_answer_score']) < 1e-12
            score_checks[partition] = {'rows': len(predictions), 'predictions_sha256': sha(path),
                                      'actual_native_scores_recomputed': True}
        result[method] = {'status': 'PASS', 'raw_feature_files': len(caches), 'fit_check': fit_check,
                          'scores': score_checks}
    return result


def main():
    white = OUT.parent / 'whitebox'
    config = load(white / 'CONFIG.json')
    for mapping in ['source_files', 'code_sha256']:
        for path, expected in config[mapping].items():
            assert sha(path) == expected, path
    assert sha(white / 'plans.jsonl') == config['plans_sha256']
    plans = indexed(white / 'plans.jsonl')
    members = indexed(RUN / 'data/split_assignments.jsonl')
    inputs, geoms = {}, {}
    for part in ['fit', 'validation', 'test']:
        inputs.update(indexed(RUN / 'data' / part / 'inputs.jsonl'))
        geoms.update(indexed(RUN / 'data' / part / 'geometry.jsonl'))
    assert plans.keys() == inputs.keys() == geoms.keys() == members.keys()
    # Tokenizer only. No pretrained model or CUDA context is loaded.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config['model'], use_fast=True, local_files_only=True)
    donors_checked = raw_fit_tokens = 0
    for rid, plan in plans.items():
        inp, geo, mem = inputs[rid], geoms[rid], members[rid]
        assert plan['partition'] == mem['split'] and plan['group_id'] == mem['group_id']
        assert plan['original_input_ids'] == geo['full_input_ids']
        assert plan['answer_token_ids'] == geo['response_token_ids']
        assert plan['response_token_offsets'] == geo['response_token_offsets']
        assert plan['response_token_offsets_raw'] == geo['response_token_offsets_raw']
        assert plan['lexical_mask'] == geo['lexical_mask']
        assert plan['original_answer_positions'] == geo['answer_token_positions']
        assert plan['context_token_positions'] == geo['context_token_positions']
        assert plan['original_input_ids'] == plan['original_prefix_ids'] + plan['answer_token_ids']
        assert plan['original_answer_positions'] == list(range(len(plan['original_prefix_ids']), len(plan['original_input_ids'])))
        if mem['split'] == 'fit':
            raw_fit_tokens += len(plan['answer_token_ids'])
            continue
        donor_id = plan['donor_response_id']
        donor, donor_mem = inputs[donor_id], members[donor_id]
        assert donor_mem['split'] == 'fit'
        assert mem['group_id'] != donor_mem['group_id'] == plan['donor_group_id']
        assert not ({d['title'].strip().casefold() for d in inp['documents']} &
                    {d['title'].strip().casefold() for d in donor['documents']})
        assert not ({d['text'] for d in inp['documents']} & {d['text'] for d in donor['documents']})
        assert inp['retrieved_passages'] != donor['retrieved_passages']
        a, b = inp['document_char_spans'][0]
        assert inp['prompt_prefix'][a:b] == inp['retrieved_passages']
        prefix = inp['prompt_prefix'][:a] + donor['retrieved_passages'] + inp['prompt_prefix'][b:]
        encoded = tok(prefix + inp['original_response'], add_special_tokens=False,
                      return_offsets_mapping=True, truncation=False)
        assert encoded['input_ids'] == plan['random_input_ids']
        response_positions = [i for i, (s, e) in enumerate(encoded['offset_mapping']) if e > len(prefix) and e > s]
        assert response_positions == plan['random_answer_positions']
        assert [encoded['input_ids'][i] for i in response_positions] == geo['response_token_ids']
        offsets = [[max(s, len(prefix)) - len(prefix), e - len(prefix)]
                   for i, (s, e) in enumerate(encoded['offset_mapping']) if i in set(response_positions)]
        assert offsets == geo['response_token_offsets']
        assert plan['random_answer_positions'][-1] == len(plan['random_input_ids']) - 1
        donors_checked += 1
    assert donors_checked == 550 and raw_fit_tokens == 141282
    assert config['redeep']['selection'] == 'all raw fit-token Pearson, native token risk-inverse ECS; no window pooling'
    for filename in ['CPU_CHECK.json', 'LUMINA_CPU_CHECK.json', 'REDEEP_CPU_CHECK.json']:
        assert load(white / filename)['passed']
    assert load(white / 'CPU_CHECK.json')['config_sha256'] == sha(white / 'CONFIG.json')
    score_checks = audit_completed_scores(white, plans, config)
    result = {'status': 'PASS', 'plans': len(plans), 'random_contexts_independently_retokenized': donors_checked,
        'fit_raw_tokens_for_redeep': raw_fit_tokens, 'all_donors_fit_only': True,
        'all_donors_disjoint_groups_titles_documents': True, 'answer_tokens_offsets_and_final_token_preserved': True,
        'config_sha256': sha(white / 'CONFIG.json'), 'plans_sha256': sha(white / 'plans.jsonl'),
        'GPU_used': False, 'stage': 'input preparation/formula plus available completed raw-score audit',
        'completed_score_checks': score_checks,
        'formula_review': {
            'LUMINA': '0.5 IPR - 0.5 cosine top100 MMD; unnormalized top100 masses retained; official all-output-layer convention; predictor t-1',
            'ReDeEP': 'paper standard JSD + retrieved-document ECS; raw fit-token Pearson; fit-only MinMax; fixed alpha1 beta0.2 khead1 klayer10'},
        'adaptation_notes': ['LUMINA NF4 is precision adaptation', 'ReDeEP follows paper formula rather than contradictory released-code divergence',
            'Exact shared original prompt used; author re-templating/truncation not applied',
            'Common answer metric is max4window, not native arithmetic mean; retained native answer_score is supplementary']}
    save(OUT / 'WHITEBOX_INPUT_AUDIT.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
