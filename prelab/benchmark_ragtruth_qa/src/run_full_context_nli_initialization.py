"""Pinned NLI encoder-only initialization; matched generic token head and QA loop.

This is an exploratory additional semantic detector with incompletely traced
upstream NLI data. CPU preparation does not initialize CUDA or train on QA.
GPU smoke and six-epoch training require their own explicit scheduler slot.
"""
from pathlib import Path
import argparse
import importlib.util
import shutil
import time
import traceback
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForTokenClassification
from threadpoolctl import threadpool_limits
import run_full_context_encoder_v2 as base
import run_development as q
import tail_finetune as mapping

ROOT = q.ROOT
OUT = ROOT / 'results/full_context_nli_initialization_v1'
MODEL = ROOT.parent / 'models/ModernBERT-base-nli'
REVISION = 'de4ab7e77845098b7fab7f6ab9d370ddff27b19c'
MODEL_SHA = '86c32c52ce38b8f26e028ca959b06daee3a5f3f6947c63258bc8695dde88a465'
LIMIT = 2048
REUSED = ('inputs.jsonl', 'training_weights.npz', 'answer_orders.npy')
AUDITS = ROOT / 'results/nli_initialization_source_check'
ASSETS = ('README.md', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
          'special_tokens_map.json', 'model.safetensors', 'download_manifest.json')


def protocol():
    p = base.protocol()
    p.update({
        'version': 'qa-full-context-nli-encoder-initialization-v1',
        'initialization': {
            'encoder': 'tasksource/ModernBERT-base-nli', 'revision': REVISION,
            'encoder_loading': 'Copy only all134 model.* tensors into the same generic-base token detector; strict encoder state loading, all parameters trainable afterward.',
            'inherited_head': 'Keep generic answerdotai/ModernBERT-base head.dense.weight and head.norm.weight exactly as base v2.',
            'new_head': 'Same seed20261005 fresh binary classifier.weight/bias exactly as base v2; the entire prediction transform is not randomly reinitialized.',
            'discarded_nli': ['head.dense.weight', 'head.norm.weight', 'classifier.weight', 'classifier.bias'],
            'discarded_generic': 'Generic masked-LM decoder.bias is unused, exactly as in base v2.',
            'runtime_config': 'Generic-base token-detector configuration, except max_position_embeddings=2048. Compare all normalized architecture/forward config fields against pinned NLI; remaining differences must be task metadata only.',
        },
        'context': 'Whole unmodified passages, question and answer; no truncation. Explicit2048 cap from NLI model config, not the tokenizer unbounded sentinel. Current QA inputs must equal base v2 and max979.',
        'pretraining_limit': 'Exploratory NLI initialization with limited source traceability. The inspected task list has no direct RAGTruth task evidence; upstream independence is not proven. Some upstream tasks use synthetic text or silver labels, including TrueTeacher. Do not call this contamination-free or a RAGTruth-specialized pretrained detector.',
        'comparison_limits': 'Matched encoder-initialization comparison to base v2: same tokenizer output, complete input, inherited generic prediction transform, fresh classifier, supervision, optimizer, six epochs and selection. This still adds an offline semantic checker; it is not a pure generator-internal probe.',
        'v1_control': 'Original base v2 source/preparation unchanged. All3839 input IDs, full offsets, original-BPE maps, loss weights and six answer orders are checked before byte reuse.',
        'precision_limits': 'Unchanged BF16 CUDA forward / FP32 parameters, gradients, optimizer, mapping and loss. Actual GPU smoke not yet run; CPU preparation is not a trained result.',
    })
    return p


def source_files():
    paths = [Path(__file__), Path(base.__file__), Path(mapping.__file__), Path(q.__file__),
             base.OUT / 'preparation_complete.json', base.OUT / 'protocol.json',
             base.OUT / 'source_snapshot.json', ROOT / 'fit_expansion/data/fit.jsonl',
             ROOT / 'data/calibration.jsonl', ROOT / 'data/gold_manifest.json']
    paths += [base.OUT / n for n in REUSED]
    paths += [MODEL / n for n in ASSETS]
    paths += [AUDITS / n for n in ('TASK_NAME_COUNTS.md', 'TWO_TASK_SOURCE_NOTES.md')]
    return {str(p.resolve()): q.sha(p) for p in paths}


def config_agreement():
    generic = ModernBertConfig.from_pretrained(base.MODEL, local_files_only=True).to_dict()
    nli = ModernBertConfig.from_pretrained(MODEL, local_files_only=True).to_dict()
    differences = {k: [generic.get(k), nli.get(k)] for k in sorted(set(generic) | set(nli))
                   if generic.get(k) != nli.get(k)}
    metadata = {'_name_or_path', 'architectures', 'transformers_version', 'id2label',
                'label2id', 'problem_type', 'tasks', 'classifiers_size',
                'reference_compile', 'sparse_pred_ignore_index', 'sparse_prediction'}
    behavior = {k: v for k, v in differences.items() if k not in metadata}
    assert behavior == {'max_position_embeddings': [8192, LIMIT]}, behavior
    # ModernBertConfig consumes reference_compile without retaining it in
    # to_dict; both detectors explicitly override that runtime choice to False.
    fields = ['hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
              'vocab_size', 'hidden_activation', 'global_rope_theta', 'local_rope_theta',
              'local_attention', 'global_attn_every_n_layers', 'attention_bias', 'mlp_bias',
              'attention_dropout', 'embedding_dropout', 'classifier_dropout', 'classifier_bias',
              'classifier_activation', 'norm_eps', 'norm_bias', 'pad_token_id',
              'bos_token_id', 'eos_token_id', 'tie_word_embeddings']
    return {'all_normalized_config_keys_compared': True,
            'remaining_behavior_difference': behavior,
            'matched_forward_fields': {k: generic[k] for k in fields},
            'excluded_metadata_difference_keys': sorted(k for k in differences if k in metadata),
            'serialized_reference_compile': [q.read(base.MODEL / 'config.json').get('reference_compile'),
                                             q.read(MODEL / 'config.json').get('reference_compile')],
            'forced_runtime': {'reference_compile': False, 'attn_implementation': 'sdpa',
                               'max_position_embeddings': LIMIT, 'num_labels': 2},
            'rope_note': 'The cap changes rotary-cache capacity, not theta or attention geometry; all actual positions remain below979.'}


def load_model():
    config = ModernBertConfig.from_pretrained(base.MODEL, local_files_only=True)
    config.num_labels = 2
    config.max_position_embeddings = LIMIT
    config.reference_compile = False
    torch.manual_seed(base.SEED)
    model = ModernBertForTokenClassification.from_pretrained(
        base.MODEL, config=config, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation='sdpa', reference_compile=False)
    with safe_open(MODEL / 'model.safetensors', framework='pt', device='cpu') as data:
        encoder = {k[len('model.'):]: data.get_tensor(k) for k in data.keys() if k.startswith('model.')}
        assert len(encoder) == 134
        assert set(encoder) == set(model.model.state_dict())
        model.model.load_state_dict(encoder, strict=True)
    return model


def prepare():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'prepare_started.json').exists(), 'No overwrite/resume'
    OUT.mkdir(parents=True, exist_ok=True)
    base.check_prepared()
    assert q.sha(MODEL / 'model.safetensors') == MODEL_SHA
    manifest = q.read(MODEL / 'download_manifest.json')
    assert manifest['revision'] == REVISION
    snap = source_files()
    q.save(OUT / 'protocol.json', protocol())
    q.save(OUT / 'prepare_started.json', {'time': time.time(), 'source_sha256': snap})
    q.save(OUT / 'CONFIG_AGREEMENT.json', config_agreement())
    answers, tokens, _ = mapping.metadata()
    text_rows = q.lines(ROOT / 'fit_expansion/data/fit.jsonl') + q.lines(ROOT / 'data/calibration.jsonl')
    rows = q.lines(base.OUT / 'inputs.jsonl')
    assert len(rows) == len(text_rows) == len(answers) == 3839
    old = AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True)
    new = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert old.get_vocab() == new.get_vocab() and old.special_tokens_map == new.special_tokens_map
    total = 0
    max_tokens = 0
    for i, (answer, token, text, expected) in enumerate(zip(answers, tokens, text_rows, rows)):
        assert text['response_id'] == answer['response_id'] == expected['response_id']
        prefix = text['retrieved_passages'] + new.sep_token + text['question'] + new.sep_token
        whole = prefix + text['original_response']
        enc = new(whole, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        original = old(whole, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
        assert dict(enc) == dict(original), ('tokenizer_difference', answer['response_id'])
        assert enc['input_ids'] == expected['input_ids']
        assert len(enc['input_ids']) <= LIMIT
        offsets = np.asarray(enc['offset_mapping'], np.int64)
        begin, finish = len(prefix), len(whole)
        inside = (offsets[:, 1] > begin) & (offsets[:, 0] < finish)
        start = np.where(inside, np.maximum(offsets[:, 0] - begin, 0), -1)
        end = np.where(inside, np.minimum(offsets[:, 1] - begin, finish - begin), -1)
        assert begin == expected['answer_start_character']
        assert np.array_equal(start, expected['answer_encoder_start'])
        assert np.array_equal(end, expected['answer_encoder_end'])
        char_map = mapping.character_map(answer['original_response'], token['response_token_offsets'], start, end)
        assert all(np.array_equal(x, y) for x, y in zip(char_map, expected['mapping']))
        max_tokens = max(max_tokens, len(enc['input_ids']))
        total += len(enc['input_ids'])
        if (i + 1) % 400 == 0: print('NLI_CPU_INPUT_VERIFIED', i + 1, len(rows), flush=True)
    assert max_tokens == 979
    for name in REUSED:
        shutil.copyfile(base.OUT / name, OUT / name)
        assert q.sha(OUT / name) == q.sha(base.OUT / name)
    q.save(OUT / 'INPUT_AGREEMENT.json', {
        'answers': len(rows), 'fit_answers': 3680, 'calibration_answers': 159,
        'vocabulary_and_special_tokens_exact': True, 'all_full_input_ids_and_offsets_exact': True,
        'all_answer_offsets_and_raw_BPE_maps_exact': True, 'all_attention_masks_exact': True,
        'labels_weights_orders_byte_identical': True,
        'original_tokenizer_model_max_length': old.model_max_length,
        'nli_tokenizer_model_max_length_sentinel': new.model_max_length,
        'explicit_model_config_cap': LIMIT, 'max_input_tokens': max_tokens,
        'total_input_tokens': total, 'raw_BPE_tokens': sum(x['raw_token_count'] for x in rows),
        'reused_files_sha256': {n: q.sha(OUT / n) for n in REUSED},
        'GPU_used': False, 'official_test_opened': False})
    assert snap == source_files()
    q.save(OUT / 'source_snapshot.json', {'files_sha256': snap, 'official_test_opened': False})
    names = REUSED + ('protocol.json', 'CONFIG_AGREEMENT.json', 'INPUT_AGREEMENT.json', 'source_snapshot.json')
    q.save(OUT / 'preparation_complete.json', {
        'status': 'prepared_not_trained', 'answers': len(rows), 'fit_answers': 3680,
        'max_input_tokens': max_tokens, 'total_input_tokens': total,
        'files_sha256': {n: q.sha(OUT / n) for n in names},
        'GPU_used': False, 'official_test_opened': False})
    assert not torch.cuda.is_initialized()
    print('NLI_INITIALIZATION_ALL3839_PREPARED', flush=True)


def check_prepared():
    p = q.read(OUT / 'preparation_complete.json')
    assert p['answers'] == 3839 and q.read(OUT / 'protocol.json') == protocol()
    for name, digest in p['files_sha256'].items(): assert q.sha(OUT / name) == digest, name
    assert q.read(OUT / 'source_snapshot.json')['files_sha256'] == source_files()
    base.check_prepared()
    return p


def bound_core():
    # Independent module globals preserve the frozen base module and output path.
    spec = importlib.util.spec_from_file_location('_nli_matched_qa_core', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.load_model = load_model
    module.check_prepared = check_prepared
    module.protocol = protocol
    return module


def cpu_test():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    check_prepared()
    original = base.load_model()
    original_rng = torch.get_rng_state().clone()
    nli = load_model()
    assert torch.equal(original_rng, torch.get_rng_state())
    original_state, nli_state = original.state_dict(), nli.state_dict()
    head_keys = [k for k in original_state if not k.startswith('model.')]
    assert set(head_keys) == {'head.dense.weight', 'head.norm.weight', 'classifier.weight', 'classifier.bias'}
    assert all(torch.equal(original_state[k], nli_state[k]) for k in head_keys)
    with safe_open(MODEL / 'model.safetensors', framework='pt', device='cpu') as source:
        keys = list(source.keys())
        encoder_keys = [k for k in keys if k.startswith('model.')]
        assert all(torch.equal(source.get_tensor(k).float(), nli_state[k]) for k in encoder_keys)
        discarded = {k: list(source.get_slice(k).get_shape()) for k in keys if k not in encoder_keys}
    changed = [k for k in encoder_keys if not torch.equal(original_state[k], nli_state[k])]
    assert changed
    assert all(p.dtype == torch.float32 and p.requires_grad for p in nli.parameters())
    assert nli.config.max_position_embeddings == LIMIT
    # No real targets: verify the actual loaded encoder changes a tiny sequence,
    # remains finite, and is deterministic in evaluation on CPU.
    original.eval(); nli.eval()
    ids = torch.tensor([[50281, 15, 16, 17, 18, 19, 20, 50282]])
    with torch.no_grad():
        a = original(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        b = nli(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        c = nli(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    assert torch.isfinite(b).all() and torch.equal(b, c) and not torch.equal(a, b)
    detail = {'all134_encoder_tensors_exact_pinned_NLI': True,
              'encoder_tensors_different_from_generic': len(changed),
              'generic_prediction_head_exact': ['head.dense.weight', 'head.norm.weight'],
              'binary_classifier_exact_same_seed': True, 'post_initialization_CPU_RNG_exact': True,
              'all_nonencoder_keys_checked_exact': head_keys,
              'discarded_NLI_tensors': discarded,
              'parameters': sum(p.numel() for p in nli.parameters()),
              'actual_model_CPU_forward_finite_and_repeat_exact': True,
              'actual_model_CPU_logit_difference_vs_generic': float((a-b).abs().max()),
              'all_parameters_trainable_fp32': True, 'real_QA_loss_or_training': False,
              'config_agreement': config_agreement(), 'GPU_used': False}
    del a, b, c, original, nli, original_state, nli_state
    # Reuse the actual frozen training forward/mapping on a tiny synthetic model.
    bound_core().cpu_test()
    report = q.read(OUT / 'CPU_SELFCHECK.json')
    report['initialization'] = detail
    report['preparation_sha256'] = q.sha(OUT / 'preparation_complete.json')
    q.save(OUT / 'CPU_SELFCHECK.json', report)
    assert not torch.cuda.is_initialized()
    print('NLI_MATCHED_INITIALIZATION_CPU_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'check', 'cpu-test', 'gpu-smoke', 'train'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        try:
            if args.stage == 'prepare': prepare()
            elif args.stage == 'check': check_prepared(); print('NLI_PREPARED_CHECK_PASSED', flush=True)
            elif args.stage == 'cpu-test': cpu_test()
            else: getattr(bound_core(), args.stage.replace('-', '_'))()
        except Exception:
            OUT.mkdir(parents=True, exist_ok=True)
            q.save(OUT / f'FAILURE_{args.stage}_{time.time_ns()}.json',
                   {'traceback': traceback.format_exc(), 'stage': args.stage})
            raise
