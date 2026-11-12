"""Bounded CPU audit; never imports a production runner or evaluates/trains a model."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
DEST = Path(__file__).resolve().parent
OUT = DEST.parent
ROOT = OUT.parent.parent
PREVIOUS = ROOT / 'results/minicheck_tail_all_docs_v2'
SRC = ROOT / 'src/tail_finetune_all_docs_v3.py'
OLD = ROOT / 'src/tail_finetune.py'
READ_HASHES = {}


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    READ_HASHES[str(path.resolve())] = h.hexdigest()
    return h.hexdigest()


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path, fields):
    sha(path)
    result = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            result.append({k: x[k] for k in fields})
    return result


def npz(path):
    sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
protocol = read(OUT / 'protocol.json')
previous_protocol = read(PREVIOUS / 'protocol.json')
freeze = read(OUT / 'protocol_freeze.json')
snapshot = read(OUT / 'source_snapshot.json')
prepared = read(OUT / 'preparation_complete.json')
started = read(DEST / 'started.json')
complete = read(DEST / 'complete.json')
export = read(ROOT / 'fit_expansion/data/export_freeze.json')
gold = read(ROOT / 'data/gold_manifest.json')
config = read(ROOT / 'semantic_baseline/model/config.json')
sha(PREVIOUS / 'protocol_freeze.json')
sha(SRC)
sha(OLD)
seed = protocol['seed']
assert seed == 20261004 and protocol['epochs'] == 3 and protocol['effective_answer_batch'] == 8
fixed_keys = ('epochs', 'seed', 'effective_answer_batch', 'optimizer', 'selection', 'weights', 'token_loss', 'aggregation')
assert all(protocol[k] == previous_protocol[k] for k in fixed_keys)

afields = ('response_id', 'partition', 'group_id', 'answer_sha256', 'label', 'original_labels')
tfields = ('response_id', 'partition', 'answer_sha256', 'answer_risk', 'token_count', 'lexical_mask', 'risk_mask')
fit_a = rows(ROOT / 'fit_expansion/data/answers_fit.jsonl', afields)
fit_t = rows(ROOT / 'fit_expansion/data/tokens_fit.jsonl', tfields)
cal_a = rows(ROOT / 'data/answers_calibration.jsonl', afields)
cal_t = rows(ROOT / 'data/tokens_calibration.jsonl', tfields)
original = rows(ROOT / 'data/answers_fit.jsonl', ('response_id',))
original_ids = {a['response_id'] for a in original}
assert len(fit_a) == len(fit_t) == 3680 and len(cal_a) == len(cal_t) == 159
assert len(original_ids) == len(original) == 634
for partition, aa, tt in [('fit', fit_a, fit_t), ('calibration', cal_a, cal_t)]:
    for a, t in zip(aa, tt):
        assert a['response_id'] == t['response_id'] and a['partition'] == t['partition'] == partition
        assert a['answer_sha256'] == t['answer_sha256']
        assert a['label'] == t['answer_risk'] == int(bool(a['original_labels']))
        assert t['token_count'] == len(t['lexical_mask']) == len(t['risk_mask'])
assert len({a['response_id'] for a in fit_a + cal_a}) == 3839
fit_groups = sorted({a['group_id'] for a in fit_a})
cal_groups = {a['group_id'] for a in cal_a}
assert len(fit_groups) == 615 and len(cal_groups) == 154 and not set(fit_groups) & cal_groups
assert original_ids <= {a['response_id'] for a in fit_a}
for rel in ('answers_fit.jsonl', 'tokens_fit.jsonl'):
    p = (ROOT / 'fit_expansion/data' / rel).resolve()
    assert READ_HASHES[str(p)] == export['output_files_sha256'][str(p)]
for rel in ('answers_calibration.jsonl', 'tokens_calibration.jsonl'):
    p = (ROOT / 'data' / rel).resolve()
    record = next(x for x in gold['outputs'] if (ROOT / x['path']).resolve() == p)
    assert READ_HASHES[str(p)] == record['sha256']

w = npz(OUT / 'training_weights.npz')
v2 = npz(PREVIOUS / 'training_weights.npz')
assert set(w) == set(v2)
assert all(w[k].dtype == v2[k].dtype and np.array_equal(w[k], v2[k]) for k in w)
sizes = np.asarray([t['token_count'] for t in fit_t], np.int64)
ends = np.r_[0, sizes.cumsum()]
bounds = np.column_stack((ends[:-1], ends[1:]))
lex = np.concatenate([np.asarray(t['lexical_mask'], bool) for t in fit_t])
y = np.concatenate([np.asarray(t['risk_mask'], np.int64) for t in fit_t])
membership = np.asarray([a['response_id'] in original_ids for a in fit_a])
assert len(y) == 665708 and int(lex.sum()) == 560300 and int(y.sum()) == 47398
assert not y[~lex].any() and set(np.unique(y)) == {0, 1}
assert np.array_equal(w['bounds'], bounds) and np.array_equal(w['y'], y)
assert np.array_equal(w['original_membership'], membership)
assert int(w['target_mass']) == 560300 and int(w['fit_answers']) == 3680 and int(w['groups']) == 615
group_index = {g: j for j, g in enumerate(fit_groups)}
answer_group = np.asarray([group_index[a['group_id']] for a in fit_a])
token_group = np.repeat(answer_group, sizes)
token_answer = np.repeat(np.arange(3680), sizes)
lex_counts = np.asarray([sum(t['lexical_mask']) for t in fit_t], np.int64)
assert (lex_counts > 0).all()
strata = Counter((int(g), bool(o)) for g, o in zip(answer_group, membership))
assert all(strata[g, True] for g in range(615))
without_aux = sum(strata[g, False] == 0 for g in range(615))
assert int(w['groups_without_auxiliary']) == without_aux
# Independent absolute-mass derivation: group -> original/auxiliary -> answer -> lexical token.
M, G = 560300, 615
answer_mass = np.asarray([(M / G) * (0.5 if strata[int(g), False] else 1.0) / strata[int(g), bool(o)]
                          for g, o in zip(answer_group, membership)])
base = np.where(lex, (answer_mass / lex_counts)[token_answer], 0.0)
class_mass = np.bincount(y, weights=base, minlength=2)
factors = M / (2 * class_mass)
pre_group_loss = base * factors[y]
group_denoms = np.bincount(token_group, weights=pre_group_loss, minlength=G)
loss = pre_group_loss * ((M / G) / group_denoms)[token_group]
errors = {'base': maxerr(base, w['base']), 'class_factors': maxerr(factors, w['class_factors']),
          'loss': maxerr(loss, w['loss'])}
assert max(errors.values()) < 1e-9, errors
assert np.isfinite(w['base']).all() and np.isfinite(w['loss']).all()
assert (w['base'][~lex] == 0).all() and (w['loss'][~lex] == 0).all()
assert (w['base'][lex] > 0).all() and (w['loss'][lex] > 0).all()
group_base = np.bincount(token_group, weights=w['base'], minlength=G)
group_loss = np.bincount(token_group, weights=w['loss'], minlength=G)
answer_base = np.bincount(token_answer, weights=w['base'], minlength=3680)
group_errors = {'base': maxerr(group_base, np.full(G, M/G)), 'loss': maxerr(group_loss, np.full(G, M/G)),
                'answer_base': maxerr(answer_base, answer_mass)}
assert max(group_errors.values()) < 1e-7, group_errors
assert abs(float(w['base'].sum()) - M) < 1e-7 and abs(float(w['loss'].sum()) - M) < 1e-7

sha(OUT / 'answer_orders.npy')
sha(PREVIOUS / 'answer_orders.npy')
orders = np.load(OUT / 'answer_orders.npy', allow_pickle=False)
previous_orders = np.load(PREVIOUS / 'answer_orders.npy', allow_pickle=False)
rng = np.random.default_rng(seed)
expected_orders = np.stack([rng.permutation(3680) for _ in range(3)])
assert orders.shape == (3, 3680) and np.array_equal(orders, previous_orders) and np.array_equal(orders, expected_orders)
assert all(np.array_equal(np.sort(order), np.arange(3680)) for order in orders)
assert all(len(order[j:j+8]) == 8 for order in orders for j in range(0, 3680, 8))

# Only a fresh 1025-parameter CPU head, no feature input, backward, or optimizer step.
assert config['hidden_size'] == 1024
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    head = torch.nn.Linear(1024, 1)
    torch.nn.init.normal_(head.weight, mean=0.0, std=config['initializer_range'])
    torch.nn.init.zeros_(head.bias)
expected_rng = torch.Generator(device='cpu').manual_seed(seed).get_state()
expected_group = {'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01,
                  'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False,
                  'differentiable': False, 'fused': None, 'params': [0, 1]}
checkpoints = []
epoch0_errors = {}
for epoch in range(4):
    path = DEST / f'epoch_{epoch:02d}.pt'
    digest = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    entry = read(DEST / f'epoch_{epoch:02d}.json')
    assert state['epoch'] == entry['epoch'] == epoch
    assert state['protocol_freeze_sha256'] == started['protocol_freeze_sha256'] == sha(OUT/'protocol_freeze.json')
    assert state['source_snapshot_sha256'] == started['source_snapshot_sha256'] == sha(OUT/'source_snapshot.json')
    assert entry['artifacts_sha256']['.pt'] == digest and complete['all_epochs'][epoch] == entry
    assert state['cuda_rng_state'] is None and torch.equal(state['torch_rng_state'], expected_rng)
    model = state['model_state_dict']
    assert list(model) == ['weight', 'bias']
    assert tuple(model['weight'].shape) == (1, 1024) and tuple(model['bias'].shape) == (1,)
    assert all(v.dtype == torch.float32 and v.device.type == 'cpu' and torch.isfinite(v).all() for v in model.values())
    opt = state['optimizer_state_dict']
    assert opt['param_groups'] == [expected_group]
    if epoch == 0:
        assert not opt['state'] and entry['optimizer_steps'] == 0
        for key, expected in head.state_dict().items():
            epoch0_errors[key] = float((model[key] - expected).abs().max())
            assert torch.equal(model[key], expected)
    else:
        assert set(opt['state']) == {0, 1} and entry['optimizer_steps'] == 460
        for j, key in enumerate(('weight', 'bias')):
            item = opt['state'][j]
            assert set(item) == {'step', 'exp_avg', 'exp_avg_sq'}
            assert float(item['step']) == epoch * 460
            assert all(v.dtype == torch.float32 and v.device.type == 'cpu' and torch.isfinite(v).all() for v in item.values())
            assert item['exp_avg'].shape == item['exp_avg_sq'].shape == model[key].shape
            assert (item['exp_avg_sq'] >= 0).all()
    checkpoints.append({'epoch': epoch, 'checkpoint_sha256': digest,
                        'epoch_optimizer_steps': int(entry['optimizer_steps']),
                        'cumulative_steps_per_parameter': [] if epoch == 0 else [int(float(opt['state'][j]['step'])) for j in (0, 1)],
                        'fp32_parameters_and_adam': True, 'seed_cpu_rng_exact': True,
                        'freeze_and_source_snapshot_bound': True})
assert started['device'] == 'cpu' and not complete['GPU_used'] and not complete['test_opened']
assert complete['trainable_parameters'] == 1025 and len(complete['all_epochs']) == 4
assert snapshot['protocol_freeze_sha256'] == sha(OUT/'protocol_freeze.json')
assert prepared['files_sha256'][str((OUT/'source_snapshot.json').resolve())] == sha(OUT/'source_snapshot.json')

# Verify precisely the read bounded sources against the original frozen hashes.
bound_checks = {}
for path, expected in freeze['files_sha256'].items():
    if path in READ_HASHES:
        assert READ_HASHES[path] == expected, path
        bound_checks[path] = expected
sha(Path(__file__))
assert not torch.cuda.is_initialized()
report = {
    'status': 'passed', 'passed': True, 'audit_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Only frozen0 training weights, fixed orders, initial head, and epoch0..3 optimizer/checkpoint records; independent CPU formulas, no production import.',
    'metadata': {'fit_answers': 3680, 'calibration_answers': 159, 'fit_groups': 615, 'calibration_groups': 154,
                 'fit_calibration_groups_disjoint': True, 'original_fit_answers': 634, 'auxiliary_fit_answers': 3046,
                 'fit_raw_tokens': 665708, 'fit_lexical_tokens': 560300, 'fit_risk_tokens': 47398,
                 'nonlexical_zero_weight_tokens': int((~lex).sum()), 'groups_without_auxiliary': without_aux},
    'weights': {'independent_formula': 'M/G per group; within group original/aux each half if auxiliary exists, otherwise original gets all; equal answers then lexical tokens; fit binary class balance followed by equal group loss.',
                'absolute_tolerance': 1e-9, 'independent_max_abs_errors': errors, 'group_and_answer_mass_max_abs_errors': group_errors,
                'base_sum': float(w['base'].sum()), 'loss_sum': float(w['loss'].sum()),
                'class_factors': w['class_factors'].tolist(), 'fit_labels_bounds_membership_exact': True,
                'previous_v2_all_10_arrays_bit_exact': True, 'previous_v2_npz_bytes_exact': sha(OUT/'training_weights.npz') == sha(PREVIOUS/'training_weights.npz'),
                'calibration_excluded_from_weight_computation': True,
                'precision_boundary': 'Frozen weights are float64; runner explicitly casts per-answer BCE weights and labels to float32. No training replay was performed.'},
    'orders': {'seed': seed, 'shape': [3, 3680], 'sequential_generator_rebuild_exact': True,
               'previous_v2_array_and_bytes_exact': np.array_equal(orders, previous_orders) and sha(OUT/'answer_orders.npy') == sha(PREVIOUS/'answer_orders.npy'),
               'each_epoch_each_fit_answer_once': True, 'batch_size': 8, 'batches_per_epoch': 460,
               'total_optimizer_steps': 1380, 'batch_objective_factor': 3680/(8*560300),
               'no_per_batch_weight_normalization': True},
    'head_initialization': {'seed': seed, 'hidden_size': 1024, 'initializer_range': config['initializer_range'],
                            'parameters': 1025, 'independent_cpu_epoch0_max_abs_errors': epoch0_errors,
                            'same_head_as_tail_constructor': 'Static comparison: both reset the same seed inside fork_rng, create Linear1024->1, then normal_(0, config.initializer_range) and zero bias; no live tail2 result read.'},
    'optimizer': {'name': 'AdamW', 'parameter_group': expected_group, 'checkpoints': checkpoints},
    'frozen_protocol_keys_exact_to_v2': list(fixed_keys),
    'verified_frozen_source_hashes': bound_checks, 'files_read_sha256': READ_HASHES,
    'limitations': ['Checkpoint step counters, fixed orders and frozen source agree; this is not an independent replay of every minibatch update or proof from an execution trace.',
                    'No live tail2 results, test files, lower-layer caches or mapped hidden matrix were read. Thresholds, metrics and selected-model score replay belong to the separate parent audit.',
                    'Same head initialization for tail2 is established from frozen source logic, not by reading its running checkpoints. Only listed bounded source hashes were rehashed.'],
    'GPU_used': False, 'training_run': False, 'production_runner_imported': False, 'official_test_opened': False,
}
target = DEST / 'INDEPENDENT_TRAINING_RECORD_AUDIT.json'
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'report_sha256': sha(target),
                  'weight_errors': errors, 'group_errors': group_errors, 'class_factors': w['class_factors'].tolist(),
                  'head_max_abs_errors': epoch0_errors, 'steps': [0,460,920,1380]}, ensure_ascii=False))
