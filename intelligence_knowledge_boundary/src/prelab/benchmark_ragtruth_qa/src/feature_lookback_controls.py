"""Definition controls for frozen QA reconstruction; no GPU during prepare/check.

One base forward emits a 2x2 extent/timing comparison plus the exact old anchor.
All old files are read-only. --run is explicit and must be scheduled by root.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import feature_qa as base
import run_feature_qa as loader

ROOT = base.ROOT
OUT = ROOT / 'data/lookback_controls_v1'
PROTOCOL = ROOT / 'lookback_controls_protocol.json'
VERSION = 'qa-lookback-definition-controls-v1'
VARIANTS = ('lb_source_pre_header', 'lb_prefix_pre_header', 'lb_source_post_header',
            'lb_prefix_post_header', 'lb_source_post_legacy')
COORDS = ('token_ids', 'answer_token_positions', 'response_token_offsets', 'response_token_offsets_raw')


def freeze(path, value):
    if path.exists():
        assert base.read(path) == value, ('Frozen controls file changed', str(path))
    else:
        base.save(path, value)


def layout(plan):
    """Derive actual chat-header positions, never standalone-tokenize the suffix."""
    view = plan['original']; offsets = np.asarray(view['input_token_offsets'])
    answer = np.asarray(view['answer_token_positions'], dtype=np.int64)
    header_start = len(base.WRAPPER_LEFT) + len(plan['released_prompt'])
    answer_start = view['answer_character_range'][0]
    assert answer_start == header_start + len(base.WRAPPER_RIGHT)
    assert view['prefix_character_length'] == answer_start
    header = np.flatnonzero((offsets[:, 0] >= header_start) & (offsets[:, 1] <= answer_start) &
                            (offsets[:, 1] > offsets[:, 0]))
    assert len(header) > 0 and header[-1] == answer[0] - 1
    assert np.array_equal(header, np.arange(header[0], answer[0]))
    assert offsets[header[0], 0] == header_start, 'Header start cuts a token; stop for explicit policy review'
    assert answer[0] not in header
    assert offsets[answer[0], 0] <= answer_start < offsets[answer[0], 1]
    prefix = np.arange(int(header[0]), dtype=np.int64)
    context = np.asarray(view['context_token_positions'], dtype=np.int64)
    assert len(prefix) > 0 and len(context) > 0 and set(context).issubset(set(prefix))
    assert np.all(answer > 0) and np.array_equal(answer, np.arange(answer[0], answer[-1] + 1))
    return {'response_id': plan['response_id'], 'source_id': plan['source_id'], 'group_id': plan['group_id'],
        'partition': plan['partition'], 'original_plan_sha256': base.digest(plan),
        'header_character_range': [header_start, answer_start], 'header_token_positions': header.tolist(),
        'prefix_context_token_positions': prefix.tolist(), 'source_context_token_positions': context.tolist(),
        'answer_token_positions': answer.tolist(), 'pre_query_positions': (answer - 1).tolist(),
        'post_query_positions': answer.tolist(), 'first_response_raw_offset': view['response_token_offsets_raw'][0],
        'header_token_ids': [view['input_ids'][i] for i in header]}


class ControlHooks(base.LookbackHooks):
    def __init__(self, model, plan, view, query_batch=base.QUERY_BATCH):
        super().__init__(model, plan, query_batch)
        self.prefix = torch.tensor(view['prefix_context_token_positions'], device=self.device, dtype=torch.long)
        self.header = torch.tensor(view['header_token_positions'], device=self.device, dtype=torch.long)
        self.new = torch.cat((self.header, self.positions))
        n = len(self.positions)
        self.controls = {key: np.empty((n, self.spec['layers'], self.spec['heads']), dtype=np.float32)
                         for key in VARIANTS}

    def capture(self, layer, module, keys):
        state = self.pending.pop(layer)
        heads, kv, dim = self.spec['heads'], self.spec['kv_heads'], self.spec['head_dim']
        assert keys.shape[0] == state['q'].shape[0] == 1
        query = state['q'].view(1, -1, heads, dim).transpose(1, 2)
        key = keys.view(1, -1, kv, dim).transpose(1, 2)
        cos, sin = state['rope']
        key = key * cos[:, None] + base.rotate_half(key) * sin[:, None]
        key = key.repeat_interleave(heads // kv, dim=1)[0]
        absolute = torch.arange(key.shape[-2], device=self.device)
        for timing in ('post', 'pre'):
            for begin in range(0, len(self.positions), self.query_batch):
                end = min(begin + self.query_batch, len(self.positions))
                pos = self.positions[begin:end] - int(timing == 'pre')
                q = query[0].index_select(1, pos)
                qc, qs = cos[0].index_select(0, pos), sin[0].index_select(0, pos)
                q = q * qc[None] + base.rotate_half(q) * qs[None]
                logits = (q @ key.transpose(-2, -1)) * module.scaling
                logits.masked_fill_(absolute[None, None] > pos[None, :, None], float('-inf'))
                weights = logits.float().softmax(-1)
                causal_new = self.new[None, :] <= pos[:, None]
                assert bool(torch.all(causal_new.sum(-1) > 0))
                new_mean = (weights.index_select(-1, self.new) * causal_new[None]).sum(-1) / causal_new.sum(-1)[None]
                source_mean = weights.index_select(-1, self.context).mean(-1)
                prefix_mean = weights.index_select(-1, self.prefix).mean(-1)
                for extent, context_mean in [('source', source_mean), ('prefix', prefix_mean)]:
                    ratio = context_mean / (context_mean + new_mean).clamp_min(torch.finfo(torch.float32).tiny)
                    self.controls[f'lb_{extent}_{timing}_header'][begin:end, layer] = ratio.T.cpu().numpy()
                if timing == 'post':
                    # Copy the original arithmetic and chunk shape exactly for old-cache comparison.
                    attention = weights.index_select(-1, self.positions)
                    previous = self.positions[None, :] <= pos[:, None]
                    answer_mean = (attention * previous[None]).sum(-1) / previous.sum(-1)[None]
                    ratio = source_mean / (source_mean + answer_mean).clamp_min(torch.finfo(torch.float32).tiny)
                    self.controls['lb_source_post_legacy'][begin:end, layer] = ratio.T.cpu().numpy()
        self.seen.add(layer)


@torch.inference_mode()
def extract(model, plan, current_layout):
    assert not model.training
    ids, mask, _ = base.tensor_input(model, plan['original'])
    with ControlHooks(model, plan, current_layout) as hooks:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                    output_attentions=False, output_hidden_states=False)
    assert len(hooks.seen) == hooks.spec['layers']
    arrays = {k: x.reshape(len(hooks.positions), -1) for k, x in hooks.controls.items()}
    v = plan['original']
    arrays.update(token_ids=np.asarray(v['answer_token_ids'], dtype=np.int64),
                  answer_token_positions=np.asarray(v['answer_token_positions'], dtype=np.int64),
                  response_token_offsets=np.asarray(v['response_token_offsets'], dtype=np.int32),
                  response_token_offsets_raw=np.asarray(v['response_token_offsets_raw'], dtype=np.int32))
    return arrays


def validate(arrays, plan, width=1024):
    assert set(arrays) == set(VARIANTS + COORDS)
    n = len(plan['original']['answer_token_ids'])
    for key in VARIANTS:
        a = arrays[key]
        assert a.shape == (n, width) and a.dtype == np.float32 and np.isfinite(a).all()
        assert np.all((a >= 0) & (a <= 1))
    for key in COORDS:
        target = plan['original']['answer_token_ids' if key == 'token_ids' else key]
        assert np.array_equal(arrays[key], target)
    return n


def prepare():
    base.assert_cpu_only()
    previous = base.read(ROOT / 'data/feature_manifest.json')
    old_signature = base.read(ROOT / 'data/feature_signature.json')
    assert previous['complete'] and previous['completed_records'] == 793
    assert previous['signature_sha256'] == base.digest(old_signature)
    assert old_signature['feature_code_sha256'] == base.sha(Path(base.__file__))
    assert old_signature['loader_code_sha256'] == base.sha(Path(loader.__file__))
    old_path = ROOT / 'data/feature_preparation/plans.jsonl'
    assert base.sha(old_path) == old_signature['plans_sha256']
    plans = [json.loads(s) for s in old_path.read_text('utf-8').splitlines() if s]
    assert len(plans) == 793 and [p['response_id'] for p in plans] == old_signature['response_ids_in_order']
    assert all(p['partition'] in ('fit', 'calibration') and p['official_split'] == 'train' for p in plans)
    assert {x: sum(p['partition'] == x for p in plans) for x in ('fit', 'calibration')} == {'fit': 634, 'calibration': 159}
    layouts = [layout(p) for p in plans]
    old_records = {r['response_id']: r for r in previous['records']}
    for p in plans:
        rid = p['response_id']; rec = old_records[rid]; npz = ROOT / f'data/features/{rid}.npz'
        side = ROOT / f'data/features/{rid}.json'; meta = base.read(side)
        assert base.sha(npz) == rec['npz_sha256'] == meta['npz_sha256']
        assert base.sha(side) == rec['metadata_sha256']
        assert meta['response_id'] == rid and meta['plan_sha256'] == base.digest(p)
        assert meta['signature_sha256'] == previous['signature_sha256']
        with np.load(npz, allow_pickle=False) as z:
            for key in COORDS:
                assert np.array_equal(z[key], p['original']['answer_token_ids' if key == 'token_ids' else key])
    OUT.mkdir(parents=True, exist_ok=True)
    layout_path = OUT / 'layouts.jsonl'
    text = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in layouts)
    if layout_path.exists():
        assert layout_path.read_text('utf-8') == text
    else:
        layout_path.write_text(text, encoding='utf-8')
    signature = {'version': VERSION, 'protocol_sha256': base.sha(PROTOCOL),
        'code_sha256': {str(p.resolve()): base.sha(p) for p in [Path(__file__), Path(base.__file__), Path(loader.__file__)]},
        'plans_sha256': base.sha(old_path), 'layouts_sha256': base.sha(layout_path),
        'old_feature_manifest_sha256': base.sha(ROOT / 'data/feature_manifest.json'),
        'old_feature_signature_sha256': base.sha(ROOT / 'data/feature_signature.json'),
        'gold_manifest_sha256': base.sha(ROOT / 'data/gold_manifest.json'),
        'development_protocol_sha256': base.sha(ROOT / 'development_protocol.json'),
        'model_download_manifest_sha256': base.sha(ROOT / 'model_download_manifest.json'),
        'repo_id': loader.REPO, 'revision': loader.REVISION, 'load_config': loader.LOAD_CONFIG,
        'variants': list(VARIANTS), 'response_ids_in_order': [p['response_id'] for p in plans],
        'source_files_sha256': old_signature['source_development_files_sha256'],
        'old_records': old_records, 'test_or_withheld_read': False, 'annotation_values_read': False}
    for split, h in signature['source_files_sha256'].items():
        assert base.sha(ROOT / f'data/{split}.jsonl') == h
    freeze(OUT / 'signature.json', signature)
    counts = sum(len(p['original']['answer_token_ids']) for p in plans)
    report = {'ready': True, 'records': 793, 'partitions': {'fit': 634, 'calibration': 159},
        'raw_response_tokens': counts, 'all_old_feature_hashes_and_coordinates_verified': True,
        'header_token_count_values': sorted(set(len(x['header_token_positions']) for x in layouts)),
        'header_token_id_sequences': sorted(set(tuple(x['header_token_ids']) for x in layouts)),
        'first_answer_boundary_records': sum(x['first_response_raw_offset'][0] < 0 for x in layouts),
        'all_pre_first_new_pools_nonempty': True, 'all_original_ids_offsets_preserved': True,
        'signature_sha256': base.digest(signature),
        'resource_estimate': {'production_backbone_forwards': 793, 'gpu_gate_extra_repeat_forwards': 2,
            'lb_float32_raw_bytes': counts * 1024 * 4 * len(VARIANTS),
            'lb_raw_gib': counts * 1024 * 4 * len(VARIANTS) / 2**30,
            'npz_space_to_reserve_gib': 7, 'gpu_planning_minutes': [15, 30],
            'estimate_not_measured': True, 'query_readouts': 'two timing streams, batch8; no extra backbone per variant'},
        'gpu_started': False, 'labels_used': False, 'test_read': False}
    freeze(OUT / 'preparation.json', report)
    return plans, layouts, signature, report


def cpu_check():
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    base.assert_cpu_only(); torch.set_num_threads(4); torch.manual_seed(20260912)
    checks = []
    for family, cfg_cls, model_cls in [('llama', LlamaConfig, LlamaForCausalLM), ('qwen2', Qwen2Config, Qwen2ForCausalLM)]:
        cfg = cfg_cls(vocab_size=128, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=128,
                      attention_dropout=0.0, use_sliding_window=False, sliding_window=None)
        cfg._attn_implementation = 'eager'; model = model_cls(cfg).cpu().eval()
        ids = [1, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        answer = [8, 9, 10, 11, 12]; context = [2, 3, 4]; header = [6, 7]; prefix = list(range(6))
        v = {'input_ids': ids, 'attention_mask': [1] * len(ids), 'answer_token_positions': answer,
             'answer_token_ids': [ids[i] for i in answer], 'context_token_positions': context,
             'response_token_offsets': [[i, i+1] for i in range(5)],
             'response_token_offsets_raw': [[-1, 1]] + [[i, i+1] for i in range(1, 5)]}
        p = {'partition': 'fit', 'official_split': 'train', 'response_id': 'synthetic', 'original': v}
        lo = {'prefix_context_token_positions': prefix, 'header_token_positions': header}
        got = extract(model, p, lo); validate(got, p, 12)
        old, _ = base.extract_features(model, p)
        assert np.array_equal(old['lb'], got['lb_source_post_legacy'])
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                           use_cache=False, output_attentions=True)
        errors = {}
        for variant in VARIANTS:
            extent = context if 'source' in variant else prefix
            is_pre = '_pre_' in variant; use_header = variant != 'lb_source_post_legacy'
            expected = []
            for i in answer:
                qpos = i - int(is_pre)
                new = (header if use_header else []) + [j for j in answer if j <= qpos]
                assert new and (not is_pre or i not in new)
                layers = []
                for attention in output.attentions:
                    w = attention[0, :, qpos].float()
                    c = w[:, extent].mean(-1); y = w[:, new].mean(-1)
                    layers.append((c / (c + y)).numpy())
                expected.append(np.stack(layers).reshape(-1))
            expected = np.stack(expected)
            err = float(np.max(np.abs(expected - got[variant])))
            assert np.allclose(expected, got[variant], atol=2e-7, rtol=1e-6), (variant, err)
            errors[variant] = err
        # Same-length alteration of the unread last token: all pre-read rows
        # and post-read rows preceding it are invariant; post last may differ.
        changed = {**p, 'original': {**v, 'input_ids': ids[:-1]+[61],
                                    'answer_token_ids': v['answer_token_ids'][:-1]+[61]}}
        alt = extract(model, changed, lo)
        for key in VARIANTS:
            kept = len(answer) if '_pre_' in key else len(answer)-1
            assert np.array_equal(got[key][:kept], alt[key][:kept]), key
        checks.append({'family': family, 'dense_oracle_max_abs_error': errors,
            'old_anchor_exact': True, 'pre_first_pool_is_header_only': True,
            'same_length_unread_future_content_invariance_exact': True})
    result = {'passed': True, 'version': VERSION, 'checks': checks,
              'protocol_sha256': base.sha(PROTOCOL), 'code_sha256': base.sha(__file__),
              'only_tiny_random_cpu_models': True, 'gpu_initialized': torch.cuda.is_initialized(), 'test_read': False}
    freeze(OUT / 'cpu_selfcheck.json', result)
    print(json.dumps(result), flush=True)


def old_anchor(arrays, plan, signature):
    rid = plan['response_id']; path = ROOT / f'data/features/{rid}.npz'
    assert base.sha(path) == signature['old_records'][rid]['npz_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(arrays['lb_source_post_legacy'], z['lb']), ('Legacy anchor drift', rid)


def cached(plan, lo, signature):
    path = OUT / 'features' / f"{plan['response_id']}.npz"
    meta_path = path.with_suffix('.json')
    if not meta_path.exists():
        assert not path.exists(), ('Uncommitted cache requires inspection', str(path))
        return None
    meta = base.read(meta_path)
    assert meta['complete'] and meta['signature_sha256'] == base.digest(signature)
    assert meta['response_id'] == plan['response_id'] and meta['partition'] == plan['partition']
    assert meta['plan_sha256'] == base.digest(plan) and meta['layout_sha256'] == base.digest(lo)
    assert base.sha(path) == meta['npz_sha256']
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    validate(arrays, plan); old_anchor(arrays, plan, signature)
    return meta


def save_row(arrays, plan, lo, signature):
    validate(arrays, plan); old_anchor(arrays, plan, signature)
    path = OUT / 'features' / f"{plan['response_id']}.npz"
    loader.save_npz(path, arrays)
    meta = {'complete': True, 'response_id': plan['response_id'], 'source_id': plan['source_id'],
        'group_id': plan['group_id'], 'partition': plan['partition'], 'signature_sha256': base.digest(signature),
        'plan_sha256': base.digest(plan), 'layout_sha256': base.digest(lo), 'npz_sha256': base.sha(path),
        'response_tokens': len(arrays['token_ids']), 'legacy_anchor_exact': True,
        'labels_used': False, 'test_read': False}
    base.save(path.with_suffix('.json'), meta)
    return meta


def run():
    # Explicit GPU action only. prepare is read-only for old state and fails
    # if a frozen signature changed before any model can be loaded.
    plans, layouts, signature, _ = prepare()
    check = base.read(OUT / 'cpu_selfcheck.json')
    assert check['passed'] and check['code_sha256'] == base.sha(__file__)
    assert check['protocol_sha256'] == base.sha(PROTOCOL)
    assert shutil.disk_usage(OUT).free > 7 * 2**30
    lock = OUT / '.gpu_runner.lock'
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    model = None; started = time.perf_counter()
    try:
        existing = [cached(p, lo, signature) for p, lo in zip(plans, layouts)]
        if not all(existing):
            # Validate exact checkpoint bytes using the frozen download record;
            # do not call old prepare functions that write old manifests.
            download = base.read(ROOT / 'model_download_manifest.json')
            for entry in download['files']:
                assert base.sha(base.MODEL / entry['filename']) == entry['actual_sha256']
            model, _ = loader.load_nf4()
            gate = OUT / 'gpu_selfcheck.json'
            if gate.exists():
                g = base.read(gate)
                assert g['passed'] and g['signature_sha256'] == base.digest(signature)
            else:
                gate_rows = []
                for i in (0, len(plans)-1):
                    a = extract(model, plans[i], layouts[i]); b = extract(model, plans[i], layouts[i])
                    assert all(np.array_equal(a[k], b[k]) for k in a)
                    old_anchor(a, plans[i], signature)
                    existing[i] = save_row(a, plans[i], layouts[i], signature)
                    gate_rows.append({'response_id': plans[i]['response_id'], 'repeat_exact': True, 'old_anchor_exact': True})
                freeze(gate, {'passed': True, 'signature_sha256': base.digest(signature), 'rows': gate_rows})
        for i, (p, lo) in enumerate(zip(plans, layouts)):
            if existing[i] is None:
                existing[i] = save_row(extract(model, p, lo), p, lo, signature)
            if (i+1) % 10 == 0 or i+1 == len(plans):
                print('LOOKBACK_CONTROLS', i+1, '/', len(plans), 'seconds', round(time.perf_counter()-started, 1), flush=True)
        records = []
        for p, lo in zip(plans, layouts):
            meta = cached(p, lo, signature); assert meta is not None
            path = OUT / 'features' / f"{p['response_id']}.npz"
            records.append({**meta, 'npz': str(path), 'json': str(path.with_suffix('.json')),
                            'json_sha256': base.sha(path.with_suffix('.json'))})
        for path, h in signature['code_sha256'].items():
            assert base.sha(path) == h
        assert base.sha(PROTOCOL) == signature['protocol_sha256']
        manifest = {'complete': True, 'completed_count': len(records), 'records': records,
            'signature_sha256': base.digest(signature), 'variants': list(VARIANTS),
            'all_old_anchors_exact': True, 'test_read': False, 'labels_used': False}
        freeze(OUT / 'feature_manifest.json', manifest)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'cpu-check', 'run'])
    args = parser.parse_args()
    if args.action == 'run':
        run()
    elif args.action == 'cpu-check':
        cpu_check()
    else:
        _, _, _, report = prepare()
        print(json.dumps(report, ensure_ascii=False), flush=True)
