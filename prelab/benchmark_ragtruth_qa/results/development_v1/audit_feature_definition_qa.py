"""Read-only hash/coordinate linkage audit; no model import, GPU, or fitting."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(x):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    started = time.perf_counter()
    manifest = read(ROOT / 'data/feature_manifest.json')
    audit = read(ROOT / 'data/feature_audit.json')
    signature = read(ROOT / 'data/feature_signature.json')
    gold = read(ROOT / 'data/gold_manifest.json')
    independent = read(ROOT / 'data/gold_independent_review.json')
    gpu = read(ROOT / 'data/replay_selfcheck/manifest.json')
    assert manifest['complete'] and manifest['completed_records'] == 793
    assert audit['passed'] and audit['feature_manifest_sha256'] == sha(ROOT / 'data/feature_manifest.json')
    assert manifest['signature_sha256'] == digest(signature)
    assert signature['feature_code_sha256'] == sha(ROOT / 'src/feature_qa.py')
    assert signature['loader_code_sha256'] == sha(ROOT / 'src/run_feature_qa.py')
    assert signature['runner_sha256'] == sha(ROOT / 'src/run_feature_qa_all.py')
    assert signature['passed_selfcheck_manifest_sha256'] == sha(ROOT / 'data/replay_selfcheck/manifest.json')
    assert signature['selfcheck_signature_sha256'] == gpu['signature_sha256']
    assert gpu['passed']
    assert independent['status'] == 'passed'
    assert independent['export_comparison']['gold_manifest_sha256'] == sha(ROOT / 'data/gold_manifest.json')
    for key, h in gold['signature']['files_sha256'].items():
        assert sha(Path(gold['source_paths'][key])) == h, key
    for rel, h in independent['input_sha256'].items():
        assert sha(ROOT / rel) == h, rel
    for row in gold['outputs']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    plans = rows(ROOT / 'data/feature_preparation/plans.jsonl')
    assert sha(ROOT / 'data/feature_preparation/plans.jsonl') == signature['plans_sha256']
    plans = {p['response_id']: p for p in plans}
    sources, tokens, answers = {}, {}, {}
    for split in ('fit', 'calibration'):
        assert sha(ROOT / f'data/{split}.jsonl') == signature['source_development_files_sha256'][split]
        sources.update({r['response_id']: r for r in rows(ROOT / f'data/{split}.jsonl')})
        tokens.update({r['response_id']: r for r in rows(ROOT / f'data/tokens_{split}.jsonl')})
        answers.update({r['response_id']: r for r in rows(ROOT / f'data/answers_{split}.jsonl')})
    ids = [r['response_id'] for r in manifest['records']]
    assert len(set(ids)) == len(ids) == 793
    assert ids == signature['response_ids_in_order']
    assert set(ids) == set(plans) == set(sources) == set(tokens) == set(answers)
    total_tokens = 0
    first_boundary = 0
    boundary_context_tokens = 0
    for rec in manifest['records']:
        rid = rec['response_id']; p, t, src = plans[rid], tokens[rid], sources[rid]
        side_path = ROOT / f'data/features/{rid}.json'
        npz_path = ROOT / f'data/features/{rid}.npz'
        assert sha(side_path) == rec['metadata_sha256']
        assert sha(npz_path) == rec['npz_sha256']
        side = read(side_path)
        assert side['complete'] and side['response_id'] == rid
        assert side['plan_sha256'] == rec['plan_sha256'] == digest(p)
        assert side['signature_sha256'] == manifest['signature_sha256']
        assert side['npz_sha256'] == rec['npz_sha256']
        for key in ('partition', 'source_id', 'group_id'):
            assert side[key] == p[key] == t[key] == src[key] == answers[rid][key]
        assert side['official_split'] == src['official_split'] == 'train'
        assert src['quality'] == 'good'
        assert src['original_response'] == p['original_response'] == t['original_response']
        assert digest(src['original_response']) == p['answer_sha256'] == t['answer_sha256']
        assert src['labels'] == t['original_labels'] == answers[rid]['original_labels']
        assert bool(src['labels']) == bool(answers[rid]['label'])
        v = p['original']; pos = v['answer_token_positions']; n = len(pos)
        total_tokens += n
        with np.load(npz_path, allow_pickle=False) as z:
            for stored, key in [('token_ids', 'answer_token_ids'), ('answer_token_positions', 'answer_token_positions'),
                                ('response_token_offsets', 'response_token_offsets'), ('response_token_offsets_raw', 'response_token_offsets_raw')]:
                assert np.array_equal(z[stored], v[key]), (rid, stored)
                assert np.array_equal(z[stored], t['token_ids' if stored == 'token_ids' else stored])
            assert z['lb'].shape == (n, 1024)
            assert z['nll'].shape == (n,)
            assert z['hidden_last'].shape == (n, 4096)
        ctx = v['context_token_positions']; rg = v['rendered_reference_character_range']
        offsets = np.asarray(v['input_token_offsets'])
        expected_ctx = np.flatnonzero((offsets[:, 1] > rg[0]) & (offsets[:, 0] < rg[1]) & (offsets[:, 1] > offsets[:, 0])).tolist()
        assert ctx == expected_ctx and max(ctx) < min(pos)
        boundary_context_tokens += sum(offsets[j, 0] < rg[0] or offsets[j, 1] > rg[1] for j in ctx)
        first_boundary += int(v['response_token_offsets_raw'][0][0] < 0)
        assert pos[0] > 0 and v['response_token_offsets'][0][0] == 0 and t['first_answer_token_preserved']
    assert total_tokens == 213159 and first_boundary == 793
    fit_groups = {r['group_id'] for r in sources.values() if r['partition'] == 'fit'}
    cal_groups = {r['group_id'] for r in sources.values() if r['partition'] == 'calibration'}
    assert not fit_groups & cal_groups
    official = ROOT.parent / 'references/Lookback-Lens'
    files = ['src/feature_qa.py', 'src/run_feature_qa.py', 'src/run_feature_qa_all.py', 'data/feature_signature.json',
             'data/feature_manifest.json', 'data/feature_audit.json', 'data/replay_selfcheck/manifest.json',
             'data/gold_manifest.json', 'data/gold_independent_review.json', 'data/feature_preparation/plans.jsonl']
    result = {
        'status': 'passed_with_explicit_adaptations', 'scope': 'Fit634/calibration159 only; no official-test content, model execution, fitting, or frozen-file modifications',
        'lookback': {
            'arithmetic': 'mean(context attention) / [mean(context attention) + mean(answer attention)]',
            'mean_vs_sum': 'Mean/mean agrees with official equation and pinned step01_extract_attns.py. Sum/sum would be a different length-dependent feature.',
            'official_reference': 'https://aclanthology.org/2024.emnlp-main.84/',
            'official_code': 'https://github.com/voidism/Lookback-Lens/blob/e0a1fa3a898fbf6512af7be5567dea8ffe7a6620/step01_extract_attns.py',
            'official_commit': 'e0a1fa3a898fbf6512af7be5567dea8ffe7a6620',
            'qa_query': 'Post-read actual response-token position i. Causal mask allows keys <= i, including diagonal/self-attention.',
            'official_query': 'Generation step predicts y_t from the last prefix/prior-answer position; y_t not appended until after attentions are stored.',
            'qa_context_pool': 'Only tokens overlapping exact released retrieved_passages block. Question, instructions, BOS/chat wrapper excluded from both ratio pools.',
            'official_nq_context_pool': 'Whole input prefix through document/question/task instruction, excluding fixed #Answer#: suffix; that suffix is in the new-token pool along with prior generated tokens.',
            'qa_answer_pool': 'All actual response tokens from the first through current token, including diagonal; denominator uses this causal count, not full eventual answer length.',
            'context_boundary_token_overlap_count': int(boundary_context_tokens),
            'boundary_policy': 'Token overlap, not strict character containment; boundary tokens may include neighboring separator characters.',
            'unselected_input_tokens': 'Still visible to model and included in full causal attention softmax; excluded from ratio pools. Shared softmax normalization cancels algebraically in ratio; input content still affects q/k.',
            'precision': 'NF4 BF16 q/k matmul and float32 softmax/reduction. Official generation used unquantized FP16 settings; historic trace is not available.',
            'native_bf16_cast_max_differences_in_existing_two_row_oracle': [r['library_oracle']['native_bf16_cast_lb_max_difference'] for r in gpu['checks']],
            'claim_limit': 'A source-only post-read Lookback adaptation, not exact reproduction of original experimental features. No evidence yet that these differences explain development performance.',
        },
        'timing': {'lb': 'post-read i', 'hidden_last': 'post-read i, final model RMSNorm output', 'nll': 'pre-read i-1 predicts actual token i',
                   'future': 'Causal masking excludes future content. Full teacher-forced reconstruction is not asserted numerically identical to every variable-length online prefix run.'},
        'proof_revalidation': {'records': 793, 'raw_response_tokens': total_tokens, 'fit_groups': len(fit_groups), 'calibration_groups': len(cal_groups),
            'groups_disjoint': True, 'all_feature_npz_and_sidecar_hashes_checked': True, 'all_feature_plan_source_gold_token_coordinates_checked': True,
            'all_original_human_spans_unchanged': True, 'all_gold_output_hashes_checked': True, 'all_first_boundary_crossing_tokens_retained': first_boundary,
            'existing_independent_gold_oracle_rebound': True, 'existing_gpu_two_row_oracle_rebound': True,
            'gpu_oracle_scope': 'Existing two development answers; LB sampled query rows, NLL first16; float32-softmax LB/NLL/hidden differences zero, whole-repeat differences zero. All793 were hash/coordinate audited, not independently recomputed with another model implementation.'},
        'gold_counts': gold['counts'], 'gold_edge_cases': gold['edge_case_counts'],
        'metric_adaptations': ['Four raw BPE tokens stride1 versus original Lookback experimental eight-token windows.',
            'Risk=1 and all four official span types retained; original Lookback code factual=1 is an orientation convention, not a mismatch when labels/scores are consistently trained.',
            'All quality-good unlabeled refusals are evaluated as negative, unlike R16 refusal localization exclusion.',
            'Calibration F1 is selection-optimistic; paper AUROC is not an F1 benchmark; original test remains sealed.'],
        'files_sha256': {rel: sha(ROOT / rel) for rel in files},
        'official_files_sha256': {rel: sha(official / rel) for rel in ['step01_extract_attns.py', 'generation.py', 'transformers-4.32.0/src/transformers/generation/utils.py']},
        'audit_code_sha256': sha(__file__), 'seconds': time.perf_counter() - started,
    }
    (OUT / 'FEATURE_DEFINITION_AUDIT_QA.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'proof_revalidation': result['proof_revalidation']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
