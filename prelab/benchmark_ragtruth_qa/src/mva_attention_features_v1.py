"""Pinned-author raw MVA statistics and explicit CPU/GPU preparation stages.

No full-dataset extraction or training CLI exists in this version. A future
coordinator may call extract_one/save_record only after separate authorization.
Input probabilities are reconstructed from the frozen NF4 model and cast to
FP16 before the pinned author's FP32 reductions/FP16 intermediate round trips.
These are offline post-token features, not original generation traces.
"""
from __future__ import annotations

import argparse
import ast
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import torch

import feature_qa as q
import run_feature_qa as loader

ROOT = q.ROOT
OUT = ROOT / 'results/mva_attention_features_v1'
OFFICIAL = ROOT / 'references/mva_f8b871a/gen_features.py'
COMMIT = 'f8b871a06b6c18dabe5881bd02a68854b2940b81'
OFFICIAL_SHA = 'd04da5ec2e457ed6105bc57a7a73b348274db9834179234cb941722f151eb696'
OFFICIAL_URL = f'https://raw.githubusercontent.com/Ogamon958/mva_hal_det/{COMMIT}/features/gen_features.py'
OLD = ROOT / 'data/feature_preparation/plans.jsonl'
NEW = ROOT / 'fit_expansion/data/new_token_plans.jsonl'
COMMON = ROOT / 'results/ghost_geometry_features_v1/signature.json'
NAMES = ('key_avg', 'key_entropy', 'query_entropy')
VERSION = 'mva-author-raw-code-nf4-offline-v1'
HEAD_CHUNK, QUERY_CHUNK, THREADS = 2, 32, 4
TOTAL, FIT, RAW_TOKENS = 3839, 3680, 708506
EPS = 1e-9


def atomic_json(path, value):
    q.save(path, value)


def freeze(path, value):
    if Path(path).exists():
        assert q.read(path) == value, ('Refusing to overwrite frozen definition', str(path))
    else:
        atomic_json(path, value)


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_lines_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        assert q.sha(path) == q.sha(pending), ('Existing input plan differs', str(path))
        pending.unlink()
    else:
        pending.replace(path)


def _normalize_with_zeros(x, dim):
    # Preserve BOTH intermediate FP16 quantization and the subsequent second
    # normalization. A one-pass entropy identity is not equivalent here.
    mask = (x != 0).float()
    masked = (x * mask).float()
    denominator = torch.sum(masked, dim=dim, keepdim=True) + EPS
    return ((masked / denominator) * mask).to(torch.float16)


def _normalized_entropy(x, dim):
    mask = (x != 0).float()
    masked = (x * mask).float()
    denominator = torch.sum(masked, dim=dim, keepdim=True) + EPS
    probabilities = masked / denominator
    entropy = -torch.sum(probabilities * torch.log(probabilities + EPS) * mask.float(), dim=dim)
    count = torch.sum(mask, dim=dim).clamp(min=1)
    return torch.where(count > 1, entropy / torch.log(count + EPS),
                       torch.zeros_like(entropy)).to(torch.float16)


def statistics_from_attention(attention, answer_positions=None):
    """FP16 causal attention [heads,T,T] -> FP16 [3,heads,selected_T].

    This intentionally computes each head as a 2-D matrix, exactly as the
    pinned function. All rows/columns are retained for normalization; slicing
    to answers is the final operation. Input need not have unit-mass rows
    (zero and sparse edge cases are supported), but must be causal and finite.
    """
    assert isinstance(attention, torch.Tensor) and attention.dtype == torch.float16
    assert attention.ndim == 3 and attention.shape[-1] == attention.shape[-2]
    heads, total, _ = attention.shape
    assert heads > 0 and total > 0
    assert bool(torch.isfinite(attention).all()) and bool((attention >= 0).all())
    assert bool((attention <= 1).all()) and not bool(torch.triu(attention, diagonal=1).any())
    rows = torch.arange(1, total + 1, device=attention.device, dtype=torch.float32).unsqueeze(1)
    result = torch.empty((3, heads, total), device=attention.device, dtype=torch.float16)
    for head in range(heads):
        original = attention[head]
        result[0, head] = torch.mean((original * rows).float(), dim=0).to(torch.float16)
        result[1, head] = _normalized_entropy(_normalize_with_zeros(original, dim=0), dim=0)
        result[2, head] = _normalized_entropy(_normalize_with_zeros(original, dim=1), dim=1)
    # The author nan_to_num is redundant on these valid inputs. Rejecting an
    # invalid input/output avoids concealing extraction corruption.
    assert bool(torch.isfinite(result).all()), 'Do not silently replace a nonfinite feature'
    if answer_positions is not None:
        positions = torch.as_tensor(answer_positions, device=attention.device, dtype=torch.long)
        assert positions.ndim == 1 and positions.numel() > 0
        assert bool((positions >= 0).all()) and bool((positions < total).all())
        assert bool((positions[1:] > positions[:-1]).all())
        result = result.index_select(-1, positions)
    return result


def statistics_numpy(attention, answer_positions=None):
    """CPU NumPy adapter; no implicit dtype or device conversion semantics."""
    assert isinstance(attention, np.ndarray) and attention.dtype == np.float16
    return statistics_from_attention(torch.from_numpy(attention.copy()), answer_positions).numpy()


def author_reference(attention):
    """Execute the unmodified pinned get_features AST, not its imports/demo.

    find_split_position only feeds the fourth, unused Lookback output. Setting
    its return value to zero supplies a legal dummy boundary for that branch;
    the three MVA branches and function AST are untouched.
    """
    assert q.sha(OFFICIAL) == OFFICIAL_SHA
    tree = ast.parse(OFFICIAL.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_features']
    assert len(nodes) == 1
    env = {'torch': torch, 'find_split_position': lambda tokens: 0}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OFFICIAL), 'exec'), env)
    values = env['get_features']([attention.unsqueeze(0)], list(range(attention.shape[-1])),
                                  False, str(attention.device))
    # Author tuple order: mean, outgoing entropy, incoming entropy, Lookback.
    return torch.stack((values[0][0], values[2][0], values[1][0]))


def reconstruct_attention_block(query, key, scaling, query_chunk=QUERY_CHUNK):
    """Rotated/repeated Q,K [head_chunk,T,D] -> full FP16 [head_chunk,T,T].

    Only the head axis/query computation is blocked. The two complete key
    columns remain available for the author's global, twice-normalized entropy.
    """
    assert query.shape == key.shape and query.ndim == 3 and query_chunk > 0
    heads, total, _ = query.shape
    assert heads > 0 and total > 0
    matrix = torch.empty((heads, total, total), device=query.device, dtype=torch.float16)
    absolute = torch.arange(total, device=query.device)
    for begin in range(0, total, query_chunk):
        end = min(begin + query_chunk, total)
        logits = torch.matmul(query[:, begin:end], key.transpose(-2, -1)) * scaling
        logits.masked_fill_(absolute[None, None, :] > absolute[None, begin:end, None], float('-inf'))
        matrix[:, begin:end] = logits.float().softmax(-1).to(torch.float16)
    assert bool(torch.isfinite(matrix).all())
    return matrix


class MVAHooks(q.LookbackHooks):
    """One layer and at most HEAD_CHUNK full attention heads resident at a time."""
    def __init__(self, model, plan, head_chunk=HEAD_CHUNK, query_chunk=QUERY_CHUNK):
        super().__init__(model, plan, query_chunk)
        self.head_chunk = head_chunk
        assert isinstance(head_chunk, int) and head_chunk > 0
        self.values = np.empty((3, self.spec['layers'], self.spec['heads'], len(self.positions)),
                               dtype=np.float16)

    def capture(self, layer, module, keys):
        state = self.pending.pop(layer)
        assert keys.shape[0] == state['q'].shape[0] == 1 and layer not in self.seen
        heads, kv, dim = self.spec['heads'], self.spec['kv_heads'], self.spec['head_dim']
        query = state['q'].view(1, -1, heads, dim).transpose(1, 2)
        key = keys.view(1, -1, kv, dim).transpose(1, 2)
        cos, sin = state['rope']
        query = (query * cos[:, None] + q.rotate_half(query) * sin[:, None])[0]
        key = key * cos[:, None] + q.rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(heads // kv, dim=1)[0]
        for begin in range(0, heads, self.head_chunk):
            end = min(begin + self.head_chunk, heads)
            matrix = reconstruct_attention_block(query[begin:end], key[begin:end], module.scaling,
                                                   self.query_batch)
            features = statistics_from_attention(matrix, self.positions)
            self.values[:, layer, begin:end] = features.cpu().numpy()
            del matrix, features
        self.seen.add(layer)


def validate_row(row):
    assert set(row) == {'response_id', 'source_id', 'group_id', 'partition', 'official_split',
        'old_plan_sha256', 'source_plan_file', 'source_plan_index', 'prompt_sha256', 'answer_sha256',
        'original', 'labels_used', 'exact_original_generation_trace'}
    assert row['official_split'] == 'train' and row['partition'] in ('fit', 'calibration')
    assert row['labels_used'] is False and row['exact_original_generation_trace'] is False
    view = row['original']; ids, positions = view['input_ids'], view['answer_token_positions']
    assert len(ids) > 1 and view['attention_mask'] == [1] * len(ids)
    assert 0 < positions[0] and positions == list(range(positions[0], len(ids)))
    assert view['answer_token_ids'] == [ids[i] for i in positions]
    assert all(type(i) is int and 0 <= i < 32000 for i in ids)
    assert q.digest(ids) == view['input_ids_sha256']
    assert view['response_text_sha256'] == row['answer_sha256']
    offsets, raw = np.asarray(view['response_token_offsets']), np.asarray(view['response_token_offsets_raw'])
    assert offsets.shape == raw.shape == (len(positions), 2)
    assert np.all(offsets[:, 0] >= 0) and np.all(offsets[:, 1] > offsets[:, 0])
    assert np.all(raw[:, 1] > raw[:, 0]) and np.array_equal(offsets[:, 1], raw[:, 1])
    assert np.array_equal(offsets[:, 0], np.maximum(raw[:, 0], 0))


@torch.inference_mode()
def extract_one(model, row):
    """Production single-answer skeleton. No labels, pooling or dataset loop."""
    validate_row(row)
    assert not model.training
    ids, mask, positions = q.tensor_input(model, row['original'])
    started = time.perf_counter()
    with MVAHooks(model, row) as hooks:
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             output_attentions=False, output_hidden_states=False).last_hidden_state[0]
    assert len(hooks.seen) == hooks.spec['layers'] and not hooks.pending
    features = hooks.values.transpose(3, 0, 1, 2).reshape(len(positions), -1).copy()
    view = row['original']
    arrays = {'mva_raw': features, 'token_ids': np.asarray(view['answer_token_ids'], dtype=np.int64),
        'answer_positions': np.asarray(view['answer_token_positions'], dtype=np.int64),
        'token_start': np.asarray(view['response_token_offsets'], dtype=np.int64)[:, 0],
        'token_end': np.asarray(view['response_token_offsets'], dtype=np.int64)[:, 1],
        'token_start_raw': np.asarray(view['response_token_offsets_raw'], dtype=np.int64)[:, 0],
        'token_end_raw': np.asarray(view['response_token_offsets_raw'], dtype=np.int64)[:, 1]}
    assert features.dtype == np.float16 and np.isfinite(features).all()
    anchor = hidden.index_select(0, positions[[0, -1]]).float().cpu().numpy()
    return arrays, {'seconds': time.perf_counter() - started, 'architecture': hooks.spec,
                    'hidden_first_last_anchor': anchor}


def source_rows():
    old_manifest = q.read(ROOT / 'data/feature_preparation/manifest.json')
    new_manifest = q.read(ROOT / 'fit_expansion/data/export_freeze.json')
    assert q.sha(OLD) == old_manifest['plans_jsonl_sha256']
    assert q.sha(NEW) == new_manifest['output_files_sha256'][str(NEW.resolve())]
    assert q.sha(q.__file__) == old_manifest['signature']['code_sha256']
    old, new = jsonlines(OLD), jsonlines(NEW)
    assert len(old) == 793 and len(new) == 3046
    ordered = [(OLD, i, r) for i, r in enumerate(old[:634])]
    ordered += [(NEW, i, r) for i, r in enumerate(new)]
    ordered += [(OLD, i, old[i]) for i in range(634, 793)]
    rows = []
    for path, index, record in ordered:
        assert record['official_split'] == 'train' and record['labels_used'] is False
        assert not {'labels', 'gold', 'risk_mask', 'answer_risk'} & set(record)
        assert q.digest(record['original_response']) == record['answer_sha256']
        assert q.digest(record['released_prompt']) == record['prompt_sha256']
        row = {k: record[k] for k in ('response_id', 'source_id', 'group_id', 'partition', 'official_split',
                                     'prompt_sha256', 'answer_sha256', 'original')}
        row.update(old_plan_sha256=q.digest(record), source_plan_file=str(path.resolve()),
                   source_plan_index=index, labels_used=False, exact_original_generation_trace=False)
        validate_row(row)
        rows.append(row)
    assert len(rows) == len({r['response_id'] for r in rows}) == TOTAL
    assert all(r['partition'] == ('fit' if i < FIT else 'calibration') for i, r in enumerate(rows))
    fg, cg = ({r['group_id'] for r in part} for part in (rows[:FIT], rows[FIT:]))
    assert len(fg) == 615 and len(cg) == 154 and not fg & cg
    assert sum(len(r['original']['answer_token_ids']) for r in rows) == RAW_TOKENS
    return rows


def definition():
    assert q.sha(OFFICIAL) == OFFICIAL_SHA
    common, downloads = q.read(COMMON), q.read(ROOT / 'model_download_manifest.json')
    assert downloads['status'] == 'complete'
    assert common['model']['repo'] == loader.REPO == downloads['repo_id']
    assert common['model']['revision'] == loader.REVISION == downloads['revision']
    assert common['load_config'] == loader.LOAD_CONFIG
    for entry in downloads['files']:
        path = loader.MODEL / entry['filename']; saved = common['model']['assets'][entry['filename']]
        assert path.parent.resolve() == loader.MODEL.resolve()
        assert path.stat().st_size == saved['bytes'] == entry['actual_bytes']
        assert saved['sha256'] == entry['actual_sha256'] and entry['source_hash_match']
    for name, version in common['software'].items():
        assert importlib.metadata.version(name) == version
    assert common['python'] == platform.python_version() and common['torch_cuda_runtime'] == torch.version.cuda
    paths = [Path(__file__), Path(q.__file__), Path(loader.__file__), OFFICIAL]
    code = {str(p.resolve()): q.sha(p) for p in paths}
    for path, digest in common['code_sha256'].items():
        if 'site-packages' in path or 'bitsandbytes' in path:
            assert q.sha(path) == digest
            code[path] = digest
    source_files = [OLD, NEW, ROOT / 'data/feature_preparation/manifest.json',
                    ROOT / 'fit_expansion/data/export_freeze.json', COMMON,
                    ROOT / 'model_download_manifest.json']
    return {'version': VERSION, 'official_commit': COMMIT, 'official_url': OFFICIAL_URL,
        'official_sha256': OFFICIAL_SHA, 'model': common['model'], 'load_config': loader.LOAD_CONFIG,
        'model_asset_check': 'prepare checks current sizes and frozen SHA identities; gpu-smoke rehashes actual assets before loading',
        'seed': loader.SEED, 'threads': THREADS, 'software': common['software'],
        'python': platform.python_version(), 'torch_cuda_runtime': torch.version.cuda,
        'code_sha256': code, 'source_sha256': {str(p.resolve()): q.sha(p) for p in source_files},
        'groups': {'fit_answers': FIT, 'calibration_answers': TOTAL - FIT, 'fit_groups': 615, 'calibration_groups': 154},
        'feature_names': list(NAMES), 'feature_order': 'feature-major, layer-major, head-major; raw answer token is row',
        'storage_dtype': 'float16', 'attention_probability_dtype': 'float16 after FP32 softmax of model-dtype QK product',
        'head_chunk': HEAD_CHUNK, 'query_chunk': QUERY_CHUNK, 'width': 3072, 'epsilon': EPS,
        'reduction': 'One complete causal head; author FP32 row/column reductions, normalization-to-FP16 then entropy renormalization, result FP16',
        'key_avg': 'mean over all T rows of (absolute 1-based query index * A), denominator T, not T-j+1',
        'timing': 'post-token Q row for outgoing; own and future Q rows for incoming; full sequence before answer slicing',
        'no_norm_variant': True, 'no_random_context': True, 'labels_read': False,
        'offline_future_answer_used': True, 'exact_original_generation_trace': False, 'official_test_opened': False,
        'stages': ['prepare', 'cpu-selfcheck', 'gpu-smoke'], 'automatic_stage_chaining': False,
        'real_extraction_cli_available': False, 'training_available': False,
        'baseline_status': 'Author-code-exact statistics on declared reconstructed FP16 attention are only MVA inputs, not a complete MVA baseline. No classifier or postprocessing is implemented here.',
        'future_baseline_boundary': 'A formal MVA baseline must use the pinned author classifier/structure/postprocessing in a separate directory. A 256-wide lightweight Transformer, partial-label CRF, new thresholds or other method changes are ours adaptation, never the formal baseline.',
        'cpu_oracle': 'unmodified get_features AST; dummy boundary used only by unused fourth Lookback output',
        'cpu_statistic_equality': 'exact FP16 arrays for pinned author and block/core paths',
        'gpu_smoke': 'first, last, longest frozen input, no labels; same-path repeat exact; finite all features and actual peak memory; no dataset extraction',
        'checkpoint_write': 'NPZ.pending fsync/replace, metadata with full NPZ hash, then per-record commit JSON; commit last; no overwrite of mismatched existing records'}


def resource_plan(rows):
    longest = max(range(len(rows)), key=lambda i: len(rows[i]['original']['input_ids']))
    length = len(rows[longest]['original']['input_ids'])
    count = sum(len(r['original']['answer_token_ids']) for r in rows)
    return {'longest_record_index': longest, 'longest_response_id': rows[longest]['response_id'],
        'longest_input_tokens': length, 'input_tokens': sum(len(r['original']['input_ids']) for r in rows),
        'raw_answer_tokens': count, 'feature_bytes_fp16': count * 3072 * 2,
        'feature_bytes_fp32_if_expanded': count * 3072 * 4,
        'coordinates_bytes_int64_seven_arrays_upper': count * 7 * 8,
        'resident_attention_block_bytes': HEAD_CHUNK * length * length * 2,
        'one_head_fp32_matrix_bytes': length * length * 4,
        'one_layer_rotated_qk_bytes_approx': 2 * 32 * length * 128 * 2,
        'feature_workspace_budget_bytes_estimate': 512 * 2**20,
        'estimate_note': '512MiB is a conservative engineering workspace allowance, not measured GPU peak; model/runtime/prefill allocations are additional. Actual longest-input smoke is still required.',
        'forward_budget': TOTAL, 'wall_clock_estimate': None,
        'disk_reserve_recommendation_bytes': 6 * 10**9,
        'GPU_measured_this_stage': False}


def prepare():
    q.assert_cpu_only(); torch.set_num_threads(THREADS)
    started = time.perf_counter(); rows = source_rows(); proto = definition()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', proto)
    write_lines_atomic(OUT / 'feature_inputs.jsonl', rows)
    resources = resource_plan(rows)
    assert resources['longest_input_tokens'] == 1232 and resources['input_tokens'] == 2408466
    freeze(OUT / 'RESOURCE_PLAN.json', resources)
    freeze(OUT / 'source_download.json', {'url': OFFICIAL_URL, 'commit': COMMIT,
        'sha256': OFFICIAL_SHA, 'bytes': OFFICIAL.stat().st_size, 'downloaded_not_executed_as_module': True})
    complete = {'status': 'prepared_not_extracted', 'answers': TOTAL, 'fit': FIT, 'calibration': TOTAL - FIT,
        'raw_answer_tokens': RAW_TOKENS, 'width': 3072,
        'response_order_sha256': q.digest([r['response_id'] for r in rows]),
        'raw_axis_sha256': q.digest([[r['response_id'], r['original']['answer_token_ids'],
                                    r['original']['response_token_offsets_raw']] for r in rows]),
        'files_sha256': {name: q.sha(OUT / name) for name in
                        ('protocol.json', 'feature_inputs.jsonl', 'RESOURCE_PLAN.json', 'source_download.json')},
        'seconds': time.perf_counter() - started, 'GPU_used': False, 'labels_read': False,
        'official_test_opened': False, 'formal_records_extracted': 0, 'fits': 0}
    atomic_json(OUT / 'preparation_complete.json', complete)
    q.assert_cpu_only(); print(json.dumps(complete), flush=True)


def check_prepared():
    complete = q.read(OUT / 'preparation_complete.json')
    assert complete['status'] == 'prepared_not_extracted' and complete['answers'] == TOTAL
    assert complete['raw_answer_tokens'] == RAW_TOKENS and not complete['GPU_used']
    for name, digest in complete['files_sha256'].items():
        assert Path(name).name == name and q.sha(OUT / name) == digest
    assert q.read(OUT / 'protocol.json') == definition()
    rows = jsonlines(OUT / 'feature_inputs.jsonl')
    assert len(rows) == TOTAL
    for i, row in enumerate(rows):
        validate_row(row)
        assert row['partition'] == ('fit' if i < FIT else 'calibration')
    assert q.digest([r['response_id'] for r in rows]) == complete['response_order_sha256']
    assert q.digest([[r['response_id'], r['original']['answer_token_ids'],
                      r['original']['response_token_offsets_raw']] for r in rows]) == complete['raw_axis_sha256']
    return rows, complete


def validate_arrays(arrays, row, width=3072):
    view = row['original']; n = len(view['answer_token_ids'])
    assert set(arrays) == {'mva_raw', 'token_ids', 'answer_positions', 'token_start', 'token_end',
                           'token_start_raw', 'token_end_raw'}
    assert arrays['mva_raw'].dtype == np.float16 and arrays['mva_raw'].shape == (n, width)
    assert np.isfinite(arrays['mva_raw']).all()
    assert np.array_equal(arrays['token_ids'], view['answer_token_ids'])
    assert np.array_equal(arrays['answer_positions'], view['answer_token_positions'])
    for suffix, source in (('', 'response_token_offsets'), ('_raw', 'response_token_offsets_raw')):
        assert np.array_equal(arrays['token_start' + suffix], np.asarray(view[source])[:, 0])
        assert np.array_equal(arrays['token_end' + suffix], np.asarray(view[source])[:, 1])


def read_record(directory, index, row, signature_sha, width=3072):
    directory = Path(directory); stem = f'{index:05d}'
    commit = directory / (stem + '.commit.json')
    if not commit.exists():
        return None
    record = q.read(commit)
    assert record['signature_sha256'] == signature_sha and record['record_sha256'] == q.digest(row)
    assert record['record_index'] == index and record['response_id'] == row['response_id']
    assert q.sha(directory / (stem + '.npz')) == record['npz_sha256']
    assert q.sha(directory / (stem + '.json')) == record['metadata_sha256']
    with np.load(directory / (stem + '.npz'), allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    validate_arrays(arrays, row, width)
    return arrays


def save_record(directory, index, row, arrays, signature_sha, width=3072):
    """Atomic per-record writer for a future authorized extraction coordinator."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    validate_arrays(arrays, row, width)
    existing = read_record(directory, index, row, signature_sha, width)
    if existing is not None:
        assert all(np.array_equal(existing[k], arrays[k]) for k in arrays)
        return
    stem = f'{index:05d}'; path = directory / (stem + '.npz')
    # An uncommitted completed artifact is not silently replaced. A .pending
    # file may be rewritten because it has never been a valid cache record.
    assert not path.exists() and not (directory / (stem + '.json')).exists(), 'Uncommitted final artifact: inspect before resume'
    pending = path.with_suffix('.npz.pending')
    with pending.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    pending.replace(path)
    metadata = {'record_index': index, 'response_id': row['response_id'], 'partition': row['partition'],
        'record_sha256': q.digest(row), 'signature_sha256': signature_sha, 'npz_sha256': q.sha(path),
        'old_plan_sha256': row['old_plan_sha256'], 'input_ids_sha256': row['original']['input_ids_sha256'],
        'answer_sha256': row['answer_sha256'], 'feature_names': list(NAMES), 'width': width,
        'tokens': len(row['original']['answer_token_ids']), 'dtype': 'float16', 'labels_used': False}
    atomic_json(directory / (stem + '.json'), metadata)
    commit = dict(metadata, metadata_sha256=q.sha(directory / (stem + '.json')))
    atomic_json(directory / (stem + '.commit.json'), commit)
    assert read_record(directory, index, row, signature_sha, width) is not None


def cpu_selfcheck():
    q.assert_cpu_only(); torch.set_num_threads(THREADS); started = time.perf_counter()
    rows, complete = check_prepared()
    cases = {
        'hand_causal': torch.tensor([[[1., 0, 0, 0], [.25, .75, 0, 0], [.5, .25, .25, 0],
                                      [.125, .125, .25, .5]]], dtype=torch.float16),
        'all_zero': torch.zeros((2, 5, 5), dtype=torch.float16),
        'single_token': torch.ones((1, 1, 1), dtype=torch.float16),
        'single_nonzero_per_row': torch.eye(7, dtype=torch.float16).unsqueeze(0),
        'sparse_underflow': torch.tensor([[[1., 0, 0], [2**-24, 1., 0],
                                         [2**-24, 2**-14, .5]]], dtype=torch.float16)}
    gen = torch.Generator().manual_seed(20261014)
    random = torch.randn((3, 19, 19), generator=gen)
    random.masked_fill_(torch.triu(torch.ones(19, 19, dtype=torch.bool), 1), float('-inf'))
    cases['random_causal'] = random.softmax(-1).half()
    reports = []
    for name, attention in cases.items():
        oracle = author_reference(attention)
        result = statistics_from_attention(attention)
        assert torch.equal(result, oracle), (name, (result.float() - oracle.float()).abs().max().item())
        pos = sorted({0, attention.shape[-1] // 2, attention.shape[-1] - 1})
        selected = statistics_from_attention(attention, pos)
        assert torch.equal(selected, oracle[:, :, pos])
        assert np.array_equal(statistics_numpy(attention.numpy(), pos), selected.numpy())
        reports.append({'case': name, 'heads': attention.shape[0], 'total_tokens': attention.shape[-1],
                        'max_abs_difference': 0.0, 'exact': True, 'selected_positions': pos})
    assert torch.equal(statistics_from_attention(cases['all_zero']), torch.zeros((3, 2, 5), dtype=torch.float16))
    assert not bool(statistics_from_attention(cases['single_nonzero_per_row'])[1:].any())
    hand_mean = torch.tensor([.875, .6875, .4375, .5], dtype=torch.float16)
    assert torch.equal(statistics_from_attention(cases['hand_causal'])[0, 0], hand_mean)
    failures = []
    for name, bad in [('nan', torch.tensor([[[float('nan')]]], dtype=torch.float16)),
                      ('future_nonzero', torch.ones((1, 2, 2), dtype=torch.float16)),
                      ('float32_not_silently_cast', torch.ones((1, 1, 1))),
                      ('negative', -torch.ones((1, 1, 1), dtype=torch.float16))]:
        try:
            statistics_from_attention(bad)
        except AssertionError:
            failures.append(name)
        else:
            raise AssertionError(('Invalid input accepted', name))

    # Actual HF tiny model: independently returned eager attention, all layers,
    # grouped-query heads, arbitrary head/query chunks, and no-hook hidden state.
    from transformers import LlamaConfig, LlamaForCausalLM
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261014)
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.0)
        cfg._attn_implementation = 'eager'
        model = LlamaForCausalLM(cfg).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
    view = {'input_ids': ids[0].tolist(), 'attention_mask': [1] * ids.shape[1],
            'answer_token_positions': [7, 8, 9, 10], 'answer_token_ids': ids[0, 7:].tolist(),
            'context_token_positions': [1, 2, 3, 4, 5]}
    tiny_plan = {'original': view}
    with torch.inference_mode():
        dense = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            output_attentions=True, output_hidden_states=False)
        expected = torch.stack([author_reference(a[0].half())[:, :, 7:] for a in dense.attentions], dim=1).numpy()
        hook_checks = []
        for hc, qc in ((1, 1), (2, 3), (4, 32)):
            with MVAHooks(model, tiny_plan, hc, qc) as hook:
                actual = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                     output_attentions=False).last_hidden_state
            assert len(hook.seen) == 2 and not hook.pending
            assert np.array_equal(hook.values, expected), ('Dense HF mismatch', hc, qc, np.abs(hook.values - expected).max())
            assert torch.equal(actual, dense.last_hidden_state)
            hook_checks.append({'head_chunk': hc, 'query_chunk': qc, 'features_exact': True, 'hidden_exact': True})
        handles_before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
        try:
            with MVAHooks(model, tiny_plan):
                raise RuntimeError('intentional cleanup check')
        except RuntimeError:
            pass
        assert handles_before == sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())

    # Character/ID axes and atomic writer are exercised with one existing plan,
    # synthetic feature values, and a temporary directory; no formal cache made.
    anchor = rows[0]; n = len(anchor['original']['answer_token_ids'])
    arrays = {'mva_raw': np.zeros((n, 3), dtype=np.float16),
        'token_ids': np.asarray(anchor['original']['answer_token_ids'], dtype=np.int64),
        'answer_positions': np.asarray(anchor['original']['answer_token_positions'], dtype=np.int64)}
    for suffix, key in (('', 'response_token_offsets'), ('_raw', 'response_token_offsets_raw')):
        off = np.asarray(anchor['original'][key], dtype=np.int64)
        arrays['token_start' + suffix], arrays['token_end' + suffix] = off[:, 0], off[:, 1]
    with tempfile.TemporaryDirectory(prefix='mva_cpu_commit_') as directory:
        save_record(directory, 0, anchor, arrays, 'cpu-synthetic-only', width=3)
        save_record(directory, 0, anchor, arrays, 'cpu-synthetic-only', width=3)
        assert read_record(directory, 0, anchor, 'cpu-synthetic-only', width=3) is not None
    q.assert_cpu_only()
    report = {'status': 'passed', 'source_sha256': q.sha(__file__), 'official_sha256': OFFICIAL_SHA,
        'preparation_complete_sha256': q.sha(OUT / 'preparation_complete.json'), 'cases': reports,
        'invalid_inputs_rejected': failures, 'actual_tiny_HF_dense_and_hooks': hook_checks,
        'hook_cleanup_on_exception': True, 'atomic_writer_roundtrip': True,
        'records_identity_checked': len(rows), 'raw_answer_tokens': RAW_TOKENS,
        'seconds': time.perf_counter() - started, 'GPU_used': False, 'official_test_opened': False,
        'real_model_loaded': False, 'real_records_extracted': 0, 'fits': 0}
    atomic_json(OUT / 'CPU_SELFCHECK.json', report)
    print(json.dumps(report), flush=True)


def verify_assets_before_gpu():
    for name, asset in q.read(OUT / 'protocol.json')['model']['assets'].items():
        assert q.sha(loader.MODEL / name) == asset['sha256'], name


def gpu_smoke():
    # Explicit command only. It never calls a full-data extractor or trainer.
    rows, complete = check_prepared()
    check = q.read(OUT / 'CPU_SELFCHECK.json')
    assert check['status'] == 'passed' and check['source_sha256'] == q.sha(__file__)
    assert check['preparation_complete_sha256'] == q.sha(OUT / 'preparation_complete.json')
    assert not (OUT / 'GPU_SMOKE.json').exists(), 'GPU smoke already finished; do not rerun automatically'
    verify_assets_before_gpu()
    chosen = sorted({0, len(rows) - 1, q.read(OUT / 'RESOURCE_PLAN.json')['longest_record_index']})
    started = time.perf_counter(); model = None
    reports = []
    try:
        model, load_meta = loader.load_nf4()
        for index in chosen:
            row = rows[index]; torch.cuda.reset_peak_memory_stats()
            arrays, meta = extract_one(model, row)
            repeated, repeat_meta = extract_one(model, row)
            validate_arrays(arrays, row)
            assert all(np.array_equal(arrays[k], repeated[k]) for k in arrays), 'Same-path repeat changed'
            assert np.array_equal(meta['hidden_first_last_anchor'], repeat_meta['hidden_first_last_anchor'])
            torch.cuda.synchronize()
            reports.append({'index': index, 'response_id': row['response_id'],
                'input_tokens': len(row['original']['input_ids']), 'answer_tokens': len(arrays['token_ids']),
                'shape': list(arrays['mva_raw'].shape), 'same_path_repeat_exact': True,
                'all_finite': True, 'seconds_first': meta['seconds'], 'seconds_repeat': repeat_meta['seconds'],
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            print('MVA_SMOKE_RECORD', json.dumps(reports[-1]), flush=True)
        report = {'status': 'passed', 'protocol_sha256': q.sha(OUT / 'protocol.json'),
            'cpu_check_sha256': q.sha(OUT / 'CPU_SELFCHECK.json'), 'records': reports,
            'model_load': load_meta, 'seconds': time.perf_counter() - started,
            'GPU_used': True, 'formal_records_extracted': 0, 'fits': 0, 'official_test_opened': False,
            'scope': 'Resource/finite/repeat smoke; independent dense attention/statistics oracle was CPU TinyLlama, not a claim of a second full GPU oracle'}
        atomic_json(OUT / 'GPU_SMOKE.json', report)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cpu-selfcheck', 'gpu-smoke'))
    stage = parser.parse_args().stage
    try:
        {'prepare': prepare, 'cpu-selfcheck': cpu_selfcheck, 'gpu-smoke': gpu_smoke}[stage]()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {'stage': stage, 'exception': repr(exc), 'traceback': traceback.format_exc(),
                   'source_sha256': q.sha(__file__), 'GPU_initialized': torch.cuda.is_initialized()}
        path = OUT / f'FAILURE_{stage}_{time.time_ns()}.json'
        atomic_json(path, failure)
        raise


if __name__ == '__main__':
    main()
