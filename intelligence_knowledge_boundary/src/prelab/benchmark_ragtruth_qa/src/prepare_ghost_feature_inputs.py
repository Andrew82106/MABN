"""CPU-only input freeze for a potential GHOST-inspired replay.

This command cannot load pretrained weights, extract GPU features or train.
"""
from pathlib import Path
from collections import Counter
import json
import time
import torch
import feature_qa as q
import ghost_geometry

ROOT = q.ROOT
OUT = ROOT / 'results/ghost_geometry_preparation_v1'
OLD = ROOT / 'data/feature_preparation/plans.jsonl'
NEW = ROOT / 'fit_expansion/data/new_token_plans.jsonl'


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def run():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'preparation_complete.json').exists()
    cpu = q.read(OUT / 'CPU_SELFCHECK.json')
    assert cpu['repeat_exact'] and not cpu['GPU_used'] and not cpu['real_data_read']
    old_manifest = q.read(ROOT / 'data/feature_preparation/manifest.json')
    expansion = q.read(ROOT / 'fit_expansion/data/export_freeze.json')
    assert q.sha(OLD) == old_manifest['plans_jsonl_sha256']
    assert q.sha(NEW) == expansion['output_files_sha256'][str(NEW.resolve())]
    old, new = lines(OLD), lines(NEW)
    assert len(old) == 793 and len(new) == 3046
    records = old[:634] + new + old[634:]
    assert len({r['response_id'] for r in records}) == 3839
    assert all(r['partition'] == 'fit' for r in records[:3680])
    assert all(r['partition'] == 'calibration' for r in records[3680:])
    fit_groups = {r['group_id'] for r in records[:3680]}
    cal_groups = {r['group_id'] for r in records[3680:]}
    assert len(fit_groups) == 615 and len(cal_groups) == 154 and not fit_groups & cal_groups
    exported, counts = [], Counter()
    for r in records:
        assert r['official_split'] == 'train' and not r['labels_used']
        view = r['original']
        ids, positions, answer = view['input_ids'], view['answer_token_positions'], view['answer_token_ids']
        assert view['attention_mask'] == [1] * len(ids) and len(ids) <= 4096
        assert q.digest(ids) == view['input_ids_sha256']
        assert len(positions) == len(answer) == len(view['response_token_offsets'])
        assert positions == list(range(positions[0], positions[-1] + 1)) and positions[0] > 0
        assert [ids[j] for j in positions] == answer
        assert all(0 <= a < b <= len(r['original_response']) for a, b in view['response_token_offsets'])
        assert not set(r) & {'labels', 'risk', 'gold', 'original_labels'}
        exported.append({'response_id': r['response_id'], 'source_id': r['source_id'],
                         'group_id': r['group_id'], 'partition': r['partition'],
                         'input_ids': ids, 'answer_token_positions': positions, 'answer_token_ids': answer,
                         'response_token_offsets': view['response_token_offsets'],
                         'response_token_offsets_raw': view['response_token_offsets_raw'],
                         'old_plan_sha256': q.digest(r), 'answer_sha256': r['answer_sha256']})
        counts['input_tokens'] += len(ids)
        counts['raw_answer_tokens'] += len(answer)
        counts['boundary_crossing_first_tokens'] += int(view['response_token_offsets_raw'][0][0] < 0)
    protocol = {'status': 'CPU_input_and_formula_preparation_only',
                'model': 'Existing Llama-2-7B-chat NF4/BF16 frozen reconstruction checkpoint; no new model download.',
                'scope': '3680 fit answers /615 material groups and159 cal /154 disjoint groups; all raw answer tokens retained. No official test or gold labels opened.',
                'layer_convention': 'HF hidden_states indices3..29 inclusive, after decoder blocks3..29 of32; final RMSNorm state used as final reference.',
                'token_timing': 'All four signals use the state/logits at position P+i-1, before the target answer token. Same frozen complete sequence and BPE boundaries, no regeneration.',
                'features': cpu['features'],
                'adaptation': 'GHOST originally averages token features into an answer vector. Local4BPE window supervision would be a new adaptation, not completed author reproduction. No calibrated F1 currently exists.',
                'future_gate': 'Requires explicit GPU scheduling after the live crossfit queue, identity-aligned dense-vs-streaming and repeat checks on fixed short fit rows, then a longest-input resource check. No automatic GPU command exists here.',
                'downstream': 'Fit classifiers only after all3839 features finish. Preserve existing labels/windows/weights, group split and answermax. Give baseline families equivalent features and training budgets.',
                'feature_payload_bytes_float32': counts['raw_answer_tokens'] * 4 * 4,
                'performance_estimate': 'Old793-row all-attention extraction used662.55seconds GPU plus serialization. New four-signal hook omits attention reconstruction and large hidden exports; speed must be measured, not inferred as an accuracy gain.'}
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / 'protocol.json', protocol)
    path = OUT / 'feature_inputs.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in exported), encoding='utf-8')
    bindings = [Path(__file__), Path(ghost_geometry.__file__), OLD, NEW,
                ROOT / 'data/feature_preparation/manifest.json', ROOT / 'fit_expansion/data/export_freeze.json',
                OUT / 'CPU_SELFCHECK.json', OUT / 'protocol.json', path]
    complete = {'time': time.time(), 'status': 'CPU_ready_not_GPU_extracted', 'answers': 3839,
                'native_fit': 634, 'added_fit': 3046, 'calibration': 159, 'fit_groups': 615, 'calibration_groups': 154,
                'stats': dict(counts), 'max_input_tokens': max(len(r['input_ids']) for r in exported),
                'source_sha256': {str(p.resolve()): q.sha(p) for p in bindings},
                'GPU_used': False, 'new_fits': 0, 'official_test_opened': False}
    q.save(OUT / 'preparation_complete.json', complete)
    assert not torch.cuda.is_initialized()
    print(json.dumps({k: complete[k] for k in ('status', 'answers', 'stats', 'max_input_tokens')}), flush=True)


if __name__ == '__main__':
    run()
