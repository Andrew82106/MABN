"""Independent CPU review; never loads a checkpoint or official-test record."""
from pathlib import Path
import hashlib
import json
import sys
import time
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ghost_geometry as g
import feature_qa as q


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]


def run():
    started = time.time()
    assert not torch.cuda.is_initialized()
    frozen = q.read(OUT / 'preparation_complete.json')
    for name, expected in frozen['source_sha256'].items():
        assert q.sha(Path(name)) == expected, name
    old = readl(ROOT / 'data/feature_preparation/plans.jsonl')
    added = readl(ROOT / 'fit_expansion/data/new_token_plans.jsonl')
    inputs = readl(OUT / 'feature_inputs.jsonl')
    originals = old[:634] + added + old[634:]
    assert len(old) == 793 and len(added) == 3046 and len(inputs) == 3839
    assert len({r['response_id'] for r in inputs}) == 3839
    allowed = {'response_id', 'source_id', 'group_id', 'partition', 'input_ids',
               'answer_token_positions', 'answer_token_ids', 'response_token_offsets',
               'response_token_offsets_raw', 'old_plan_sha256', 'answer_sha256'}
    raw_count = crossing = 0
    for i, (r, source) in enumerate(zip(inputs, originals)):
        assert set(r) == allowed
        assert source['official_split'] == 'train' and source['labels_used'] is False
        expected = {k: source[k] for k in ('response_id', 'source_id', 'group_id', 'partition', 'answer_sha256')}
        expected.update({k: source['original'][k] for k in (
            'input_ids', 'answer_token_positions', 'answer_token_ids',
            'response_token_offsets', 'response_token_offsets_raw')})
        expected['old_plan_sha256'] = q.digest(source)
        assert r == expected
        assert r['partition'] == ('fit' if i < 3680 else 'calibration')
        positions = r['answer_token_positions']
        assert positions == list(range(positions[0], positions[-1] + 1))
        assert 0 < positions[0] and positions[-1] < len(r['input_ids'])
        assert [r['input_ids'][p] for p in positions] == r['answer_token_ids']
        assert len(positions) == len(r['response_token_offsets_raw']) == len(r['response_token_offsets'])
        raw_count += len(positions)
        crossing += int(r['response_token_offsets_raw'][0][0] < 0)
    fit_groups = {r['group_id'] for r in inputs[:3680]}
    cal_groups = {r['group_id'] for r in inputs[3680:]}
    assert len(fit_groups) == 615 and len(cal_groups) == 154 and fit_groups.isdisjoint(cal_groups)
    assert raw_count == 708506 and crossing == 3837

    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[1, 9, 11, 6, 7, 20, 31, 2, 27, 15, 25, 32, 12]])
    positions = torch.arange(5, 13)
    first, last = 2, 5
    initial_hooks = [set(layer._forward_hooks) for layer in model.model.layers]
    with torch.no_grad():
        actual = g.extract(model, ids, positions, first, last, logit_batch=3)
        dense = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      output_hidden_states=True, use_cache=False)
        states = torch.stack([h[0, positions - 1].float() for h in dense.hidden_states[first:last+1]])
        reference = dense.hidden_states[-1][0, positions - 1].float()
        turbulence = (1 - F.cosine_similarity(states[:-1], states[1:], dim=-1)).mean(0)
        stubbornness = F.cosine_similarity(states, reference.unsqueeze(0), dim=-1).mean(0)
        values, vocab = dense.logits[0, positions - 1].float().topk(10)
        p = values.softmax(-1)
        entropy = -(p * p.log()).sum(-1)
        emb = model.get_input_embeddings().weight[vocab].float()
        pair = F.cosine_similarity(emb[:, :, None], emb[:, None, :], dim=-1)
        offdiagonal = ~torch.eye(10, dtype=torch.bool)
        dispersion = (1 - pair[:, offdiagonal]).mean(-1)
        expected = torch.stack((turbulence, stubbornness, entropy, dispersion), -1)
        dense_error = float((expected - actual).abs().max())
        assert torch.allclose(actual, expected, atol=8e-7, rtol=0)
        # Intervene at the target AND all later positions, retaining identical shape.
        causal_errors = []
        for j in (0, 3, len(positions)-1):
            changed = ids.clone()
            cut = int(positions[j])
            changed[:, cut:] = (changed[:, cut:] + 8) % config.vocab_size
            intervened = g.extract(model, changed, positions, first, last, 3)
            error = float((actual[:j+1] - intervened[:j+1]).abs().max())
            assert torch.equal(actual[:j+1], intervened[:j+1])
            causal_errors.append(error)
        assert torch.equal(actual, g.extract(model, ids, positions, first, last, 3))
    assert [set(layer._forward_hooks) for layer in model.model.layers] == initial_hooks
    # Failure must remove its own hooks without deleting an independently installed hook.
    external = model.model.layers[0].register_forward_hook(lambda *args: None)
    def fail(*args):
        raise RuntimeError('deliberate independent hook-cleanup test')
    failing = model.model.layers[4].register_forward_hook(fail)
    preserved = [set(layer._forward_hooks) for layer in model.model.layers]
    try:
        try:
            g.extract(model, ids, positions, first, last, 3)
        except RuntimeError as exc:
            assert str(exc) == 'deliberate independent hook-cleanup test'
        else:
            raise AssertionError('failure injection did not run')
        assert [set(layer._forward_hooks) for layer in model.model.layers] == preserved
    finally:
        failing.remove()
        external.remove()
    assert not torch.cuda.is_initialized()
    report = {
        'status': 'passed', 'seconds': time.time() - started,
        'source_sha256': {str(Path(__file__).resolve()): q.sha(Path(__file__)),
                          str(OUT / 'preparation_complete.json'): q.sha(OUT / 'preparation_complete.json')},
        'all_frozen_source_hashes_unchanged': True, 'all_exported_records_exact_to_original_plans': True,
        'answers': 3839, 'fit': 3680, 'calibration': 159,
        'fit_groups': 615, 'calibration_groups': 154, 'groups_disjoint': True,
        'raw_answer_tokens_including_punctuation': raw_count,
        'boundary_crossing_first_tokens_preserved': crossing,
        'cpu_dense_oracle_max_abs_error': dense_error,
        'same_shape_target_and_future_intervention_errors': causal_errors,
        'same_path_repeat_exact': True, 'own_hooks_removed_on_success_and_exception': True,
        'external_hooks_preserved': True,
        'formula': ['mean of 26 adjacent cosine distances over HF states 3..29',
                    'mean cosine of those 27 states to final normalized hidden state',
                    'entropy of renormalized top-10 next-token probabilities',
                    'unweighted mean cosine distance of 90 ordered, off-diagonal input-embedding pairs'],
        'causality': 'All four features read complete causal forward at target position minus one.',
        'denominator': 'No gold, lexical selection, window geometry or classifier is consumed; all original raw BPE axes are retained.',
        'limitations': ['CPU tiny Llama proves indexing and implementation, not NF4 GPU numerical identity.',
                        'This is the declared pre-read/localization adaptation, not a claim of full author reproduction or accuracy gain.'],
        'GPU_used': False, 'pretrained_checkpoint_loaded': False, 'new_fits': 0, 'official_test_opened': False}
    q.save(OUT / 'INDEPENDENT_REVIEW.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
