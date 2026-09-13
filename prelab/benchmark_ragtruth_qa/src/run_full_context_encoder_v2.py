"""BF16-forward variant of the frozen FP32 full-context token classifier.

Fresh generic ModernBERT weights, never a RAGTruth-finetuned release. The
unchanged original Llama BPE labels and evaluation are used after character
mapping. This is an additional semantic checker, not a generator-state probe.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_code
import torch.amp.autocast_mode as autocast_code
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForTokenClassification
from transformers.models.modernbert import modeling_modernbert as modernbert_code
from threadpoolctl import threadpool_limits
import run_development as q
import tail_finetune as mapping
import tail_finetune_all_docs as evaluation

ROOT = q.ROOT
OUT = ROOT / 'results/full_context_encoder_v2'
V1_OUT = ROOT / 'results/full_context_encoder_v1'
V1_SOURCE = Path(__file__).with_name('run_full_context_encoder.py')
MODEL = ROOT.parent / 'models/ModernBERT-base'
SEED = 20261005
NFIT = 3680
EPOCHS = 6


def protocol():
    return {
        'version': 'qa-full-context-modernbert-v2-bf16-forward',
        'source': 'https://arxiv.org/html/2502.17125v1',
        'architecture': 'ModernBERT-base AutoModelForTokenClassification equivalent, two output logits; all parameters trainable',
        'initialization': 'answerdotai/ModernBERT-base revision8949b909ec900327062f0ebf497f51aef5e6f0c8; generic masked-LM pretraining, not published LettuceDetect/RAGTruth finetuned weights',
        'input': 'Tokenizer default CLS/SEP around retrieved_passages + literal SEP + question + literal SEP + complete original_response. Question is a separate input segment, not asserted source evidence.',
        'context': 'Whole unmodified passages, question and answer in a single sequence, no sentence splitting, selection or truncation; fail if >8192.',
        'mapping': 'Per-encoder-token risk logit z1-z0 -> unchanged original raw BPE via nonwhitespace character overlap mean. Duplicate character coverage shares weight; pure whitespace logit0 and loss0.',
        'labels': 'All four original human span types unioned using unchanged original raw BPE risk labels. No new per-sentence or per-document label.',
        'weights': 'Exact expanded tail_finetune.token_weights; source groups equal, original/aux half base mass, fit-only class balancing and reequal group loss; target lexical mass560300.',
        'training': {'epochs': EPOCHS, 'seed': SEED, 'lr': 1e-5, 'weight_decay': .01,
                     'optimizer': 'AdamW', 'gradient_clip': 1., 'microbatch_answers': 1,
                     'effective_batch_answers': 8, 'scheduler': 'none',
                     'precision': 'CUDA-only BF16 autocast for model forward and native non-reentrant checkpoint recomputation; FP32 parameters, accumulated gradients and AdamW states. Cast both output logits to FP32 before difference, character mapping and BCE. CPU remains FP32 without autocast. No GradScaler; TF32 off.',
                     'attention': 'SDPA native torch; reference_compile disabled',
                     'gradient_checkpointing': 'non-reentrant per native transformer layer; torch checkpoint restores the original CUDA autocast context during recomputation'},
        'evaluation': 'Original159 calibration answers/42241 windows, 4 raw BPE stride1 lexical-max, answer max. Refusals retained. Epoch0 diagnostic only.',
        'selection': 'Per-epoch calibration-only separate risk-F1 thresholds, original precision/higher-cutoff tie rule. Select epoch1..6 by min(windowF1,answerF1), windowF1,windowPrecision,earlier epoch.',
        'scope': 'Same3680 fit answers/615 groups, same159 calibration/154 groups; official test stays sealed.',
        'comparison_limits': 'Changes encoder family, whole-input availability and number of finetuned layers together. Not an isolated question ablation or exact LettuceDetect reproduction. This run uses QA only, original BPE mapped/group-weighted loss, and current held-out development split.',
        'pretraining_limit': 'No RAGTruth task-finetuned checkpoint used; generic pretraining corpus independence is not established.',
        'artifacts': 'Preserve all epochs0..6 checkpoint/optimizer/RNG, complete predictions, thresholds, metrics and timing; no implicit resume/overwrite.',
        'gpu_gate': 'Actual longest-input repeated inference maxdiff<=2e-6, synthetic-zero-target finite nonzero backward, actual memory/runtime recorded before training. No gold labels in smoke objective.',
        'v1_control': 'Frozen v1 source and preparation are preserved. All3839 rendered inputs, weight arrays and all6 epoch orders must match v1 exactly. Initialization, optimizer and selection budget are unchanged.',
        'precision_limits': 'BF16 is an a-priori efficiency variant, not chosen by calibration performance. It can change predictions, gradients, optimization and final metrics versus FP32; numerical equivalence and faster runtime are not assumed. CUDA smoke and root decision are required before any real training.',
        'smoke_comparison': 'Same longest input and zero target as existing v1 smoke. Also record an untrained FP32-vs-BF16 logit difference descriptively, without using it to tune precision. Runtime comparison is a single-step diagnostic, not a throughput guarantee.',
    }


def source_files():
    _, _, paths = mapping.metadata()
    paths += [ROOT / 'fit_expansion/data/fit.jsonl', ROOT / 'data/calibration.jsonl',
              Path(__file__), Path(mapping.__file__), Path(evaluation.__file__), Path(q.__file__),
              ROOT / 'fit_expansion/data/export_freeze.json', ROOT / 'data/gold_manifest.json']
    paths += [MODEL / f for f in ['download_manifest.json', 'model.safetensors', 'config.json',
                                 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']]
    paths += [V1_SOURCE, V1_OUT / 'preparation_complete.json', V1_OUT / 'inputs.jsonl',
              V1_OUT / 'training_weights.npz', V1_OUT / 'answer_orders.npy',
              V1_OUT / 'GPU_SELFCHECK.json', Path(checkpoint_code.__file__),
              Path(autocast_code.__file__), Path(modernbert_code.__file__)]
    return {str(p.resolve()): q.sha(p) for p in paths}


def compare_v1_prepared():
    """No resplitting or retokenization drift is permitted for the precision change."""
    previous = q.read(V1_OUT / 'preparation_complete.json')
    for name, digest in previous['files_sha256'].items():
        assert q.sha(V1_OUT / name) == digest, ('v1_changed', name)
    assert q.sha(OUT / 'inputs.jsonl') == q.sha(V1_OUT / 'inputs.jsonl')
    assert np.array_equal(np.load(OUT / 'answer_orders.npy'), np.load(V1_OUT / 'answer_orders.npy'))
    with np.load(OUT / 'training_weights.npz') as new, np.load(V1_OUT / 'training_weights.npz') as old:
        assert set(new.files) == set(old.files)
        assert all(new[k].dtype == old[k].dtype and np.array_equal(new[k], old[k]) for k in new.files)
    return {'inputs_byte_identical': True, 'all_weight_arrays_exact': True,
            'all_six_epoch_orders_exact': True,
            'v1_preparation_sha256': q.sha(V1_OUT / 'preparation_complete.json'),
            'v1_source_sha256': q.sha(V1_SOURCE), 'v1_outputs_unchanged': True}


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'prepare_started.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    snap = source_files()
    q.save(OUT / 'prepare_started.json', {'time': time.time(), 'source_sha256': snap})
    answers, tokens, _ = mapping.metadata()
    text_rows = q.lines(ROOT / 'fit_expansion/data/fit.jsonl') + q.lines(ROOT / 'data/calibration.jsonl')
    assert [r['response_id'] for r in text_rows] == [a['response_id'] for a in answers]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert tokenizer.model_max_length == 8192
    rows = []
    for a, t, text in zip(answers, tokens, text_rows):
        assert text['original_response'] == a['original_response']
        prefix = text['retrieved_passages'] + tokenizer.sep_token + text['question'] + tokenizer.sep_token
        whole = prefix + text['original_response']
        enc = tokenizer(whole, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        offsets = np.asarray(enc['offset_mapping'], np.int64)
        begin = len(prefix)
        finish = len(whole)
        inside = (offsets[:, 1] > begin) & (offsets[:, 0] < finish)
        start = np.where(inside, np.maximum(offsets[:, 0] - begin, 0), -1)
        end = np.where(inside, np.minimum(offsets[:, 1] - begin, finish - begin), -1)
        char_map = mapping.character_map(a['original_response'], t['response_token_offsets'], start, end)
        assert len(enc['input_ids']) <= 8192
        assert all(start[j] >= 0 for j in char_map[1])
        sums = np.bincount(char_map[0], weights=char_map[2], minlength=t['token_count'])
        nonspace = np.array([any(not c.isspace() for c in a['original_response'][s:e])
                            for s, e in t['response_token_offsets']])
        assert np.max(np.abs(sums[nonspace] - 1)) < 2e-7 and not sums[~nonspace].any()
        rows.append({'response_id': a['response_id'], 'partition': a['partition'], 'group_id': a['group_id'],
                     'answer_sha256': a['answer_sha256'], 'input_ids': enc['input_ids'],
                     'answer_start_character': begin, 'answer_encoder_start': start.tolist(),
                     'answer_encoder_end': end.tolist(), 'raw_token_count': t['token_count'],
                     'mapping': [x.tolist() for x in char_map]})
    with (OUT / 'inputs.jsonl').open('w', encoding='utf-8') as stream:
        import json
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    weights = mapping.token_weights(answers, tokens)
    assert int(weights['target_mass']) == 560300
    np.savez_compressed(OUT / 'training_weights.npz', **weights)
    order = np.stack([np.random.default_rng(SEED + i).permutation(NFIT) for i in range(EPOCHS)])
    np.save(OUT / 'answer_orders.npy', order)
    q.save(OUT / 'V1_REUSE_AGREEMENT.json', compare_v1_prepared())
    q.save(OUT / 'protocol.json', protocol())
    assert snap == source_files()
    q.save(OUT / 'source_snapshot.json', {'files_sha256': snap, 'official_test_opened': False})
    paths = ['inputs.jsonl', 'training_weights.npz', 'answer_orders.npy', 'protocol.json', 'source_snapshot.json', 'V1_REUSE_AGREEMENT.json']
    q.save(OUT / 'preparation_complete.json', {
        'status': 'prepared_not_trained', 'answers': len(rows), 'fit_answers': NFIT,
        'max_input_tokens': max(len(x['input_ids']) for x in rows),
        'total_input_tokens': sum(len(x['input_ids']) for x in rows),
        'mapped_original_tokens': sum(x['raw_token_count'] for x in rows),
        'files_sha256': {name: q.sha(OUT / name) for name in paths},
        'GPU_used': False, 'official_test_opened': False})
    print('FULL_CONTEXT_PREPARED', len(rows), max(len(x['input_ids']) for x in rows), flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['answers'] == 3839 and q.read(OUT / 'protocol.json') == protocol()
    for name, digest in p['files_sha256'].items(): assert q.sha(OUT / name) == digest, name
    assert q.read(OUT / 'source_snapshot.json')['files_sha256'] == source_files()
    assert q.read(OUT / 'V1_REUSE_AGREEMENT.json') == compare_v1_prepared()
    return p


def load_model():
    torch.manual_seed(SEED)
    model = ModernBertForTokenClassification.from_pretrained(
        MODEL, local_files_only=True, use_safetensors=True, num_labels=2,
        torch_dtype=torch.float32, attn_implementation='sdpa', reference_compile=False)
    return model


def logits(model, row, device, *, fp32_reference=False):
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if torch.device(device).type == 'cuda' and not fp32_reference else nullcontext())
    with context:
        z = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    # Convert BEFORE subtraction: no BF16 difference, mapping or loss arithmetic.
    z = z.float()
    raw_map = (np.asarray(row['mapping'][0], np.int64), np.asarray(row['mapping'][1], np.int64),
               np.asarray(row['mapping'][2], np.float32))
    return mapping.mapped_logits(z[:, 1] - z[:, 0], raw_map, row['raw_token_count'])


def check_fp32_state(model, optimizer=None):
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in model.parameters())
    if optimizer is not None:
        assert all(not isinstance(v, torch.Tensor) or not v.is_floating_point() or v.dtype == torch.float32
                   for state in optimizer.state.values() for v in state.values())
    return {'parameters_fp32': True, 'parameter_gradients_fp32': True,
            'optimizer_floating_states_fp32': optimizer is not None}


def configure_gpu():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_bf16_supported(), 'CUDA BF16 support is required; no automatic precision fallback'


def gpu_smoke():
    check_prepared()
    assert (OUT / 'CPU_SELFCHECK.json').exists() and not (OUT / 'GPU_SELFCHECK.json').exists()
    configure_gpu()
    rows = q.lines(OUT / 'inputs.jsonl')
    row = max(rows, key=lambda x: len(x['input_ids']))
    v1_smoke = q.read(V1_OUT / 'GPU_SELFCHECK.json')
    assert (row['response_id'], len(row['input_ids'])) == (v1_smoke['response_id'], v1_smoke['input_tokens'])
    model = load_model().cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    autocast_calls = []
    def observe_layer(module, inputs):
        autocast_calls.append({'training': module.training,
                              'autocast_enabled': torch.is_autocast_enabled('cuda'),
                              'autocast_dtype': str(torch.get_autocast_dtype('cuda'))})
    observer = model.model.layers[0].register_forward_pre_hook(observe_layer)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        p = logits(model, row, 'cuda')
        p2 = logits(model, row, 'cuda')
        fp32 = logits(model, row, 'cuda', fp32_reference=True)
    fp32_difference = float((p - fp32).abs().max())
    difference = float((p - p2).abs().max())
    assert difference <= 2e-6 and torch.isfinite(p).all()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    z = logits(model, row, 'cuda')
    objective = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
    objective.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    optimizer.step()
    torch.cuda.synchronize()
    training_calls = [x for x in autocast_calls if x['training']]
    assert len(training_calls) >= 2, 'Checkpoint recomputation was not observed'
    assert all(x['autocast_enabled'] and x['autocast_dtype'] == 'torch.bfloat16' for x in training_calls)
    observer.remove()
    dtype_check = check_fp32_state(model, optimizer)
    q.save(OUT / 'GPU_SELFCHECK.json', {
        'passed': True, 'response_id': row['response_id'], 'input_tokens': len(row['input_ids']),
        'repeated_max_abs_difference': difference, 'synthetic_target_only': True,
        'gradient_norm_before_clip': float(grad_norm), 'step_seconds': time.perf_counter() - tick,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_scope': 'Synthetic BF16 training step after inference warmup; FP32 reference inference is excluded from peak reset',
        'parameters': sum(p.numel() for p in model.parameters()),
        'dtype_check': dtype_check, 'forward_autocast': 'cuda_bfloat16',
        'mapping_and_loss_dtype': 'float32',
        'untrained_fp32_vs_bf16_max_abs_logit_difference_descriptive': fp32_difference,
        'v1_reference_smoke_sha256': q.sha(V1_OUT / 'GPU_SELFCHECK.json'),
        'v1_reference_single_step_seconds': v1_smoke['step_seconds'],
        'same_input_and_zero_target_as_v1': True,
        'checkpoint_training_layer_calls': training_calls,
        'checkpoint_recomputation_bf16_autocast_confirmed': True,
        'preparation_sha256': q.sha(OUT / 'preparation_complete.json'), 'official_test_opened': False})
    print('FULL_CONTEXT_GPU_SMOKE_PASSED', flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def train():
    check_prepared()
    gate = q.read(OUT / 'GPU_SELFCHECK.json')
    assert gate['passed'] and gate['preparation_sha256'] == q.sha(OUT / 'preparation_complete.json')
    directory = OUT / 'full_finetune'
    directory.mkdir(exist_ok=True)
    assert not (directory / 'started.json').exists()
    configure_gpu()
    answers, tokens, _ = mapping.metadata()
    rows = q.lines(OUT / 'inputs.jsonl')
    with np.load(OUT / 'training_weights.npz') as z: weights = {k: z[k].copy() for k in z.files}
    mass = int(weights['target_mass'])
    order = np.load(OUT / 'answer_orders.npy')
    model = load_model().cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    q.save(directory / 'started.json', {'time': time.time(), 'preparation_sha256': q.sha(OUT / 'preparation_complete.json')})
    history = []
    best = None
    start = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        tick = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        online = 0.
        if epoch:
            model.train()
            for first in range(0, NFIT, 8):
                chosen = order[epoch-1, first:first+8]
                optimizer.zero_grad(set_to_none=True)
                for i in chosen:
                    z = logits(model, rows[i], 'cuda')
                    lo, hi = weights['bounds'][i]
                    y = torch.as_tensor(weights['y'][lo:hi], dtype=torch.float32, device='cuda')
                    w = torch.as_tensor(weights['loss'][lo:hi], dtype=torch.float32, device='cuda')
                    objective = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() * NFIT / (len(chosen) * mass)
                    assert torch.isfinite(objective)
                    objective.backward()
                    online += float(objective.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert torch.isfinite(norm)
                optimizer.step()
                if (first+len(chosen)) % 200 == 0:
                    print('FULL_CONTEXT_TRAIN', epoch, first+len(chosen), NFIT, round(time.perf_counter()-tick, 1), flush=True)
        model.eval()
        probabilities = {}
        fit_bce = 0.
        with torch.no_grad():
            for i, row in enumerate(rows):
                z = logits(model, row, 'cuda')
                probabilities[row['response_id']] = torch.sigmoid(z).cpu().numpy()
                if i < NFIT:
                    lo, hi = weights['bounds'][i]
                    y = torch.as_tensor(weights['y'][lo:hi], dtype=torch.float32, device='cuda')
                    w = torch.as_tensor(weights['loss'][lo:hi], dtype=torch.float32, device='cuda')
                    fit_bce += float((F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).double().sum())
                if (i+1) % 400 == 0: print('FULL_CONTEXT_EVAL', epoch, i+1, len(rows), flush=True)
        fit = evaluation.score_geometry(answers, tokens, probabilities, range(NFIT))
        cal = evaluation.score_geometry(answers, tokens, probabilities, range(NFIT, len(rows)))
        assert len(fit['window_scores']) == 653979 and len(cal['window_scores']) == 42241
        thresholds = {'window': q.choose_threshold(cal['window_labels'], cal['window_scores']),
                      'answer': q.choose_threshold(cal['answer_labels'], cal['answer_scores'])}
        key = [min(thresholds['window']['f1'], thresholds['answer']['f1']),
               thresholds['window']['f1'], thresholds['window']['precision'], -epoch]
        stem = f'epoch_{epoch:02d}'
        state = {'model_state_dict': evaluation.cpu_state(model.state_dict()),
                 'optimizer_state_dict': evaluation.cpu_state(optimizer.state_dict()),
                 'epoch': epoch, 'torch_rng_state': torch.get_rng_state(),
                 'cuda_rng_state': torch.cuda.get_rng_state().cpu(),
                 'preparation_sha256': q.sha(OUT / 'preparation_complete.json')}
        torch.save(state, directory / (stem + '.pt'))
        del state
        np.savez_compressed(directory / (stem + '_token_predictions.npz'), **probabilities)
        np.savez_compressed(directory / (stem + '_scores.npz'),
                            **{'fit_' + k: v for k, v in fit.items()}, **{'cal_' + k: v for k, v in cal.items()})
        entry = {'epoch': epoch, 'thresholds': thresholds, 'selection_key': key,
                 'fit_at_cal_thresholds': evaluation.metrics_for(fit, thresholds),
                 'calibration': evaluation.metrics_for(cal, thresholds),
                 'fit_weighted_bce': fit_bce / mass, 'online_objective_sum': online,
                 'seconds': time.perf_counter() - tick,
                 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'artifacts_sha256': {ext: q.sha(directory / (stem + ext)) for ext in ['.pt', '_token_predictions.npz', '_scores.npz']},
                 'official_test_opened': False}
        q.save(directory / (stem + '.json'), entry)
        history.append(entry)
        if epoch and (best is None or key > best['selection_key']): best = entry
        print('FULL_CONTEXT_EPOCH_COMPLETE', epoch, entry['calibration']['windows']['f1'], entry['calibration']['answers']['f1'], flush=True)
    q.save(directory / 'complete.json', {'status': 'complete_development_only', 'selected': best,
                                        'all_epochs': history, 'seconds': time.perf_counter()-start,
                                        'official_test_opened': False})


def cpu_test():
    check_prepared()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                             pad_token_id=0, num_labels=2, local_attention=16, reference_compile=False)
    config._attn_implementation = 'sdpa'
    torch.manual_seed(SEED)
    model = ModernBertForTokenClassification(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    cpu_autocast_states = []
    def observe_cpu(module, inputs):
        cpu_autocast_states.append(torch.is_autocast_enabled('cpu'))
    observer = model.model.layers[0].register_forward_pre_hook(observe_cpu)
    row = {'input_ids': [1, 3, 4, 5, 6, 7, 8, 2], 'raw_token_count': 3,
           'mapping': [[0, 0, 1, 2], [4, 5, 6, 7], [.5, .5, 1., 1.]]}
    model.train()
    z = logits(model, row, 'cpu')
    loss = F.binary_cross_entropy_with_logits(z, torch.tensor([1., 0., 1.]))
    loss.backward()
    assert z.dtype == loss.dtype == torch.float32
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.classifier.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    optimizer.step()
    dtype_check = check_fp32_state(model, optimizer)
    assert len(cpu_autocast_states) >= 2 and not any(cpu_autocast_states)
    observer.remove()
    model.eval()
    with torch.no_grad():
        original = logits(model, row, 'cpu')
        revised = dict(row, input_ids=[1, 13, 14, 15, 6, 7, 8, 2])
        other = logits(model, revised, 'cpu')
    assert not torch.equal(original, other)
    answers, tokens, _ = mapping.metadata()
    probs = {a['response_id']: np.random.default_rng(i).random(t['token_count']) for i, (a, t) in enumerate(zip(answers, tokens))}
    cal = evaluation.score_geometry(answers, tokens, probs, range(NFIT, len(answers)))
    old = q.metadata()
    expected = []
    for w in old['windows'][168123:]:
        expected.append(max(probs[w['response_id']][j] for j in w['lexical_token_indices']))
    assert np.array_equal(cal['window_scores'], expected)
    assert np.array_equal(cal['window_labels'], [w['label'] for w in old['windows'][168123:]])
    assert not torch.cuda.is_initialized()
    q.save(OUT / 'CPU_SELFCHECK.json', {'passed': True, 'tiny_real_architecture_gradient': True,
                                      'context_can_change_answer_logits': True, 'original_calibration_windows_exact': True,
                                      'cpu_autocast_disabled_in_forward_and_recompute': True,
                                      'mapping_and_BCE_fp32': True, 'dtype_check': dtype_check,
                                      'versions': {'torch': torch.__version__, 'numpy': np.__version__},
                                      'GPU_used': False, 'official_test_opened': False})
    print('FULL_CONTEXT_CPU_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cpu-test', 'gpu-smoke', 'train'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        try:
            {'prepare': prepare, 'cpu-test': cpu_test, 'gpu-smoke': gpu_smoke, 'train': train}[args.stage]()
        except Exception:
            OUT.mkdir(parents=True, exist_ok=True)
            q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json', {'traceback': traceback.format_exc(), 'stage': args.stage})
            raise
