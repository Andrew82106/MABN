"""CPU-only LUMINA QA input plans; no model loading, scoring or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
import feature_qa as q

ROOT = q.ROOT
OUT = ROOT / 'results/lumina_qa_preparation_v1'
OLD = ROOT / 'data/feature_preparation/plans.jsonl'
NEW = ROOT / 'fit_expansion/data/new_token_plans.jsonl'
R7 = ROOT.parent / 'round7_evidence_grounding'
SEED = 20261011
NATIVE_FIT = 634
FIT = 3680
TOTAL = 3839


def lines(path):
    return [json.loads(x) for x in Path(path).read_text('utf-8').splitlines() if x]


def write_lines(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as h:
        for row in rows:
            h.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def original_rows():
    q.assert_cpu_only()
    old_manifest = q.read(ROOT / 'data/feature_preparation/manifest.json')
    expansion = q.read(ROOT / 'fit_expansion/data/export_freeze.json')
    assert q.sha(OLD) == old_manifest['plans_jsonl_sha256']
    assert q.sha(NEW) == expansion['output_files_sha256'][str(NEW.resolve())]
    assert q.sha(Path(q.__file__)) == old_manifest['signature']['code_sha256']
    old, new = lines(OLD), lines(NEW)
    assert len(old) == 793 and len(new) == 3046
    rows = old[:NATIVE_FIT] + new + old[NATIVE_FIT:]
    assert len(rows) == TOTAL and len({r['response_id'] for r in rows}) == TOTAL
    assert all(r['partition'] == 'fit' for r in rows[:FIT])
    assert all(r['partition'] == 'calibration' for r in rows[FIT:])
    fg, cg = ({r['group_id'] for r in part} for part in (rows[:FIT], rows[FIT:]))
    assert len(fg) == 615 and len(cg) == 154 and not fg & cg
    for r in rows:
        assert r['official_split'] == 'train' and r['labels_used'] is False
        assert not set(r) & {'labels', 'risk', 'gold', 'original_labels'}
        assert q.digest(r['released_prompt']) == r['prompt_sha256']
        assert q.digest(r['original_response']) == r['answer_sha256']
    return rows, old_manifest


def material(row):
    """Use the frozen exact reference span, not a guessed prompt delimiter."""
    view = row['original']
    left, right = view['rendered_reference_character_range']
    left -= len(q.WRAPPER_LEFT)
    right -= len(q.WRAPPER_LEFT)
    prompt = row['released_prompt']
    assert 0 <= left < right <= len(prompt)
    block = prompt[left:right]
    assert block.strip() and prompt.count(block) == 1
    noctx = row['no_context']
    if noctx is not None:
        assert noctx['released_prompt_without_passages'] == prompt[:left] + prompt[right:]
        assert noctx['removed_passages_sha256'] == q.digest(block)
    return {'source_id': row['source_id'], 'group_id': row['group_id'],
            'partition': row['partition'], 'retrieved_passages': block,
            'material_sha256': q.digest(block), 'reference_range': [left, right],
            'released_prompt_sha256': row['prompt_sha256'],
            'prompt_prefix': prompt[:left], 'prompt_suffix': prompt[right:]}


def source_pool(rows):
    sources = {}
    for row in rows:
        value = material(row)
        sid = row['source_id']
        if sid in sources:
            assert sources[sid] == value, ('Generator rows disagree on visible material', sid)
        else:
            sources[sid] = value
    fit = {sid: x for sid, x in sources.items() if x['partition'] == 'fit'}
    assert len(sources) == 793 and len(fit) == 634
    assert {r['source_id'] for r in rows[:NATIVE_FIT]} == set(fit)
    # All expanded source materials/prompt pieces matched the native first634
    # above; unlike those native plans, expansion may omit a no-context view.
    native = rows[:NATIVE_FIT] + rows[FIT:]
    assert len(native) == 793 and all(r['no_context'] is not None for r in native)
    return sources, fit


def assignments(sources, pool):
    result = {}
    for sid, target in sorted(sources.items()):
        eligible = [d for d in pool.values()
                    if d['source_id'] != sid and d['group_id'] != target['group_id']
                    and d['material_sha256'] != target['material_sha256']]
        assert eligible
        # Pseudorandom stable hash priority: only seed and material identities.
        chosen = min(eligible, key=lambda d: (q.digest(f'{SEED}|{sid}|{d["source_id"]}'), d['source_id']))
        result[sid] = {'source_id': sid, 'group_id': target['group_id'],
                       'partition': target['partition'], 'donor_source_id': chosen['source_id'],
                       'donor_group_id': chosen['group_id'], 'donor_partition': 'fit',
                       'material_sha256': target['material_sha256'],
                       'donor_material_sha256': chosen['material_sha256'],
                       'eligible_donors': len(eligible)}
    return result


def validate_view(view, response):
    ids, positions, answer = (view[k] for k in ('input_ids', 'answer_token_positions', 'answer_token_ids'))
    assert view['attention_mask'] == [1] * len(ids)
    assert 0 < positions[0] <= positions[-1] == len(ids) - 1, 'Unexpected suffix tokens; stop rather than trim'
    assert positions == list(range(positions[0], len(ids)))
    assert answer == [ids[p] for p in positions]
    assert len(answer) == len(view['response_token_offsets']) == len(view['response_token_offsets_raw'])
    assert q.digest(ids) == view['input_ids_sha256']
    assert len(ids) <= 4096, 'No truncation or length-based donor reselection is allowed'
    assert all(0 <= a < b <= len(response) for a, b in view['response_token_offsets'])


def make_plan(tokenizer, row, sources, selected):
    target = sources[row['source_id']]
    assignment = selected[row['source_id']]
    donor = sources[assignment['donor_source_id']]
    left, right = target['reference_range']
    replacement = donor['retrieved_passages']
    prompt = row['released_prompt']
    random_prompt = prompt[:left] + replacement + prompt[right:]
    rv = q.encode_view(tokenizer, random_prompt, row['original_response'],
                       reference_range=(left, left + len(replacement)))
    ov = row['original']
    validate_view(ov, row['original_response'])
    validate_view(rv, row['original_response'])
    for key in ('answer_token_ids', 'response_token_offsets', 'response_token_offsets_raw'):
        assert ov[key] == rv[key], (row['response_id'], key)
    op, rp = ov['answer_token_positions'][0], rv['answer_token_positions'][0]
    return {'response_id': row['response_id'], 'source_id': row['source_id'],
            'group_id': row['group_id'], 'partition': row['partition'],
            'official_split': 'train', 'old_plan_sha256': q.digest(row),
            'original_prompt': prompt, 'random_prompt': random_prompt,
            'original_response': row['original_response'],
            'answer_sha256': row['answer_sha256'],
            'original_prompt_sha256': row['prompt_sha256'],
            'random_prompt_sha256': q.digest(random_prompt),
            'material_sha256': target['material_sha256'],
            'donor_source_id': assignment['donor_source_id'],
            'donor_group_id': assignment['donor_group_id'],
            'donor_material_sha256': assignment['donor_material_sha256'],
            'original_reference_range': [left, right],
            'random_reference_range': [left, left + len(replacement)],
            'original_prefix_ids': ov['input_ids'][:op],
            'random_prefix_ids': rv['input_ids'][:rp],
            'answer_token_ids': ov['answer_token_ids'],
            'original_input_ids': ov['input_ids'], 'random_input_ids': rv['input_ids'],
            'original_input_ids_sha256': ov['input_ids_sha256'],
            'random_input_ids_sha256': rv['input_ids_sha256'],
            'original_answer_positions': ov['answer_token_positions'],
            'random_answer_positions': rv['answer_token_positions'],
            'original_predictor_positions': [p - 1 for p in ov['answer_token_positions']],
            'random_predictor_positions': [p - 1 for p in rv['answer_token_positions']],
            'response_token_offsets': ov['response_token_offsets'],
            'response_token_offsets_raw': ov['response_token_offsets_raw'],
            'position_shift': rp - op, 'suffix_tokens_after_answer': 0,
            'labels_used': False, 'exact_original_generation_trace': False}


def protocol():
    return {'status': 'CPU_preparation_only', 'seed': SEED,
            'cohort': {'fit': 3680, 'fit_material_groups': 615, 'calibration': 159,
                       'calibration_material_groups': 154, 'donor_fit_materials': 634},
            'selection': 'One donor per source_id shared by all generator answers. Lowest SHA256 of seed|target_source_id|donor_source_id over fit-only materials excluding same source_id, known group_id, exact whole-material hash. No answer, question, gold, score, length or topic criterion.',
            'intervention': 'Replace the complete frozen retrieved_passages character block, including its original headers/body, with another fit material block. Preserve original prompt prefix/suffix, question/instructions, answer bytes and wrapper. No original title retention, tiling, truncation or length matching.',
            'boundary': 'Frozen rendered_reference_character_range minus literal wrapper length; prove unique occurrence. Native793 verify exact frozen no-context deletion; expansion3046 verify the same source block, full prompt and boundary as the native source because expansion may omit no-context. Preserve the answer-boundary crossing raw token and its negative raw offset.',
            'timing': 'Both sides predictor P+i-1 for each frozen answer token i; full sequence is prefix_ids+same_answer_token_ids. Any trailing suffix or changed answer IDs/offsets is a blocking error, never silently removed.',
            'formula': {'official_commit': 'c43ff41d872b05f659dcb3ad3a6dd78226954319',
                        'port': str((R7 / 'src/lumina7.py').resolve()),
                        'official_sha256': q.sha(R7 / 'references/lumina_official.py'),
                        'port_sha256': q.sha(R7 / 'src/lumina7.py'),
                        'functions': ['collect_response_hidden', 'final_statistics', 'ipr_from_hidden', 'cosine_mmd_from_topk'],
                        'top_k': 100, 'lambda': 0.5, 'vocabulary_chunk': 16,
                        'score': '0.5*IPR-0.5*MMD_squared; higher risk',
                        'author_code_convention': 'All output hidden layers including final state; norm applied again to last hidden. Paper Eq8 ends at L-1. Pin code convention rather than silently merge definitions.',
                        'aggregation_status': 'No scores or metrics produced. Paper answer aggregation is mean over all answer tokens. Any future 4BPE window mean and answer-max variant must be reported separately as localization adaptation, not substituted silently.'},
            'model': 'Existing Llama-2-7B-chat NF4/BF16 reconstruction, not historical native traces; same checkpoint and explicit wrapper as frozen QA extraction.',
            'limitations': ['Different source/group identities do not establish semantic irrelevance.',
                           'Whole-material replacement changes sequence length, position and topic together; it is an evidence-reliance probe, not a pure semantic or factuality causal estimate.',
                           'Original instructions are preserved, including any source refusal wording. Do not add insufficient-context guidance.',
                           'Official script adds an extra space and truncates prompts at 12000 characters; this QA adapter does neither and retains every frozen raw BPE.'],
            'GPU_used': False, 'fits': 0, 'official_test_opened': False,
            'future_execution': 'Root-owned formula/extraction layer and explicit GPU scheduling; this file has no model-loader, scoring, training or GPU CLI.'}


def prepare():
    q.assert_cpu_only()
    assert not (OUT / 'preparation_complete.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve an incomplete attempt for inspection'
    q.save(OUT / 'protocol.json', protocol())
    started = time.perf_counter()
    rows, old_manifest = original_rows()
    sources, pool = source_pool(rows)
    selected = assignments(sources, pool)
    tok = AutoTokenizer.from_pretrained(q.MODEL, local_files_only=True, use_fast=True)
    for name, expected in old_manifest['signature']['tokenizer']['files_sha256'].items():
        assert q.sha(q.MODEL / name) == expected
    # Six fixed identity anchors, not samples chosen by any label or score.
    anchors = []
    for ix in (0, 633, 634, 3679, 3680, 3838):
        row = rows[ix]
        target = sources[row['source_id']]
        encoded = q.encode_view(tok, row['released_prompt'], row['original_response'], target['reference_range'])
        assert encoded == row['original']
        p = make_plan(tok, row, sources, selected)
        anchors.append({'index': ix, 'response_id': row['response_id'], 'original_reencode_exact': True,
                        'answer_and_both_offsets_exact': True, 'suffix_tokens': 0,
                        'position_shift': p['position_shift']})
    q.save(OUT / 'CPU_BOUNDARY_SELFCHECK.json', {'anchors': anchors, 'GPU_used': False,
           'full_pretrained_weights_loaded': False, 'tested_before_full_plan_export': True})
    write_lines(OUT / 'donor_assignments.jsonl', [selected[s] for s in sorted(selected)])
    write_lines(OUT / 'source_materials.jsonl', [sources[s] for s in sorted(sources)])
    counts, lengths, shifts, ids = Counter(), [], [], []
    with (OUT / 'feature_inputs.jsonl').open('w', encoding='utf-8', newline='\n') as h:
        for ix, row in enumerate(rows):
            p = make_plan(tok, row, sources, selected)
            h.write(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + '\n')
            ids.append(p['response_id'])
            counts['raw_answer_tokens'] += len(p['answer_token_ids'])
            counts['original_input_tokens'] += len(p['original_input_ids'])
            counts['random_input_tokens'] += len(p['random_input_ids'])
            counts['boundary_crossing_answers'] += int(p['response_token_offsets_raw'][0][0] < 0)
            lengths.append((len(p['original_input_ids']), len(p['random_input_ids'])))
            shifts.append(p['position_shift'])
            if (ix + 1) % 500 == 0:
                print(json.dumps({'planned': ix + 1, 'total': TOTAL}), flush=True)
    assert counts['raw_answer_tokens'] == 708506 and counts['original_input_tokens'] == 2408466
    assert counts['boundary_crossing_answers'] == TOTAL
    counts['two_pass_input_tokens'] = counts['original_input_tokens'] + counts['random_input_tokens']
    counts['all_layer_IPR_token_projections'] = 32 * counts['raw_answer_tokens']
    artifacts = [OUT / name for name in ('protocol.json', 'CPU_BOUNDARY_SELFCHECK.json',
                 'donor_assignments.jsonl', 'source_materials.jsonl', 'feature_inputs.jsonl')]
    upstream = [Path(__file__), Path(q.__file__), OLD, NEW,
                ROOT / 'data/feature_preparation/manifest.json', ROOT / 'fit_expansion/data/export_freeze.json',
                R7 / 'src/lumina7.py', R7 / 'references/lumina_official.py']
    complete = {'status': 'CPU_ready_not_extracted', 'answers': TOTAL, 'fit': FIT,
                'calibration': 159, 'target_materials': len(sources), 'donor_fit_materials': len(pool),
                'counts': dict(counts), 'max_original_input_tokens': max(a for a, _ in lengths),
                'max_random_input_tokens': max(b for _, b in lengths),
                'position_shift_min': min(shifts), 'position_shift_max': max(shifts),
                'zero_position_shift_answers': shifts.count(0),
                'all_answer_IDs_and_raw_clipped_offsets_preserved': True,
                'trailing_tokens_after_answer_all_zero': True,
                'response_order_sha256': q.digest(ids),
                'artifacts_sha256': {p.name: q.sha(p) for p in artifacts},
                'source_sha256': {str(p.resolve()): q.sha(p) for p in upstream},
                'seconds': time.perf_counter() - started, 'GPU_used': False, 'fits': 0,
                'official_test_opened': False, 'gold_labels_read': False}
    q.save(OUT / 'preparation_complete.json', complete)
    q.assert_cpu_only()
    print(json.dumps(complete), flush=True)


def check():
    q.assert_cpu_only()
    complete = q.read(OUT / 'preparation_complete.json')
    for name, expected in complete['artifacts_sha256'].items():
        assert q.sha(OUT / name) == expected
    for path, expected in complete['source_sha256'].items():
        assert q.sha(path) == expected
    rows, _ = original_rows()
    sources, pool = source_pool(rows)
    selected = assignments(sources, pool)
    records = lines(OUT / 'feature_inputs.jsonl')
    assert len(records) == TOTAL
    for p, r in zip(records, rows):
        v, sid = r['original'], r['source_id']
        assert p['response_id'] == r['response_id'] and p['old_plan_sha256'] == q.digest(r)
        assert p['donor_source_id'] == selected[sid]['donor_source_id']
        target, donor = sources[sid], pool[p['donor_source_id']]
        assert p['random_prompt'] == target['prompt_prefix'] + donor['retrieved_passages'] + target['prompt_suffix']
        assert p['original_input_ids'] == v['input_ids']
        assert p['answer_token_ids'] == v['answer_token_ids']
        assert p['response_token_offsets'] == v['response_token_offsets']
        assert p['response_token_offsets_raw'] == v['response_token_offsets_raw']
        for side in ('original', 'random'):
            prefix, ids = p[side + '_prefix_ids'], p[side + '_input_ids']
            assert ids == prefix + p['answer_token_ids']
            assert q.digest(ids) == p[side + '_input_ids_sha256']
            assert p[side + '_answer_positions'] == list(range(len(prefix), len(ids)))
            assert p[side + '_predictor_positions'] == list(range(len(prefix) - 1, len(ids) - 1))
    q.save(OUT / 'CPU_OUTPUT_CHECK.json', {'status': 'passed', 'answers': TOTAL,
           'source_generators_share_donor': True, 'all_answer_tokens_including_final_retained': True,
           'old_coords_and_input_exact': True, 'donors_fit_only_no_known_group_overlap': True,
           'preparation_complete_sha256': q.sha(OUT / 'preparation_complete.json'),
           'GPU_used': False, 'fits': 0, 'official_test_opened': False})
    q.assert_cpu_only()
    print('passed3839 CPU plans; no GPU, scores or fitting', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'check'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        check()
