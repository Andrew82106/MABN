"""Local silver-repair objectives; no model/data loader or training scheduler.

An author-proposed repair is a noisy preference, not proof of a correct answer.
Only supplied local target positions receive direct supervision. Neither side's
remaining words are declared correct. Dataset eligibility/group weights belong
to a separate, not-yet-implemented training protocol.
"""
from pathlib import Path
import hashlib
import json
import torch
import torch.nn.functional as F


def local_repair_loss(original_logits, repaired_logits, original_target,
                      repaired_target, ranking_weight=0., margin=1.):
    """Equal-sided local BCE, optionally plus a local mean-logit ranking term.

    The BCE labels encode the supplied silver editing preference. The paired
    control and ranking variant must consume identical inputs and target masks.
    Different numbers of target BPEs do not change the weight of either side.
    """
    assert original_logits.ndim == repaired_logits.ndim == 1
    assert ranking_weight >= 0 and margin >= 0
    selected = []
    for logits, indices in ((original_logits, original_target), (repaired_logits, repaired_target)):
        assert indices.ndim == 1 and indices.dtype == torch.long and indices.numel() > 0
        assert indices.device == logits.device and logits.dtype == torch.float32
        assert bool(torch.all(indices >= 0)) and bool(torch.all(indices < len(logits)))
        assert indices.unique().numel() == indices.numel()
        value = logits.index_select(0, indices)
        assert bool(torch.isfinite(value).all())
        selected.append(value)
    bad, repaired = selected
    bce = .5 * (F.binary_cross_entropy_with_logits(bad, torch.ones_like(bad)) +
                F.binary_cross_entropy_with_logits(repaired, torch.zeros_like(repaired)))
    rank = F.softplus(margin - (bad.mean() - repaired.mean()))
    return {'loss': bce + ranking_weight * rank, 'local_bce': bce, 'relative_rank': rank,
            'original_mean_logit': bad.mean(), 'repaired_mean_logit': repaired.mean()}


def cpu_selfcheck():
    from transformers import ModernBertConfig, ModernBertForTokenClassification
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    a = torch.tensor([-20., 2., 2., 100.], requires_grad=True)
    b = torch.tensor([60., -2., -400.], requires_grad=True)
    ai, bi = torch.tensor([1, 2]), torch.tensor([1])
    control = local_repair_loss(a, b, ai, bi, 0.)
    ranked = local_repair_loss(a, b, ai, bi, 1.)
    assert torch.equal(control['loss'], control['local_bce'])
    changed_a, changed_b = a.detach().clone(), b.detach().clone()
    changed_a[[0, 3]] += 1234
    changed_b[[0, 2]] -= 5432
    assert torch.equal(ranked['loss'], local_repair_loss(changed_a, changed_b, ai, bi, 1.)['loss'])
    duplicate_a = torch.tensor([2., 2., 2., 2.])
    assert torch.equal(ranked['loss'], local_repair_loss(duplicate_a, b, torch.arange(4), bi, 1.)['loss'])
    swapped = local_repair_loss(b, a, bi, ai, 1.)
    assert ranked['loss'] < swapped['loss']
    ranked['loss'].backward()
    assert torch.equal(a.grad[[0, 3]], torch.zeros(2)) and torch.equal(b.grad[[0, 2]], torch.zeros(2))
    assert bool(torch.all(a.grad[ai] < 0)) and bool(torch.all(b.grad[bi] > 0))

    config = ModernBertConfig(vocab_size=83, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             max_position_embeddings=128, global_attn_every_n_layers=2,
                             local_attention=32, num_labels=2, classifier_dropout=0.,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2,
                             cls_token_id=1, sep_token_id=2)
    config._attn_implementation = 'eager'
    model = ModernBertForTokenClassification(config).float().train()
    original_ids = torch.tensor([[1, 11, 13, 17, 19, 5, 23, 29, 31, 37, 41, 7]])
    repaired_ids = original_ids.clone()
    repaired_ids[0, 8] = 43
    za = model(input_ids=original_ids, attention_mask=torch.ones_like(original_ids)).logits[0].float()
    zb = model(input_ids=repaired_ids, attention_mask=torch.ones_like(repaired_ids)).logits[0].float()
    za.retain_grad()
    zb.retain_grad()
    target = torch.tensor([8])
    values = local_repair_loss(za[:, 1] - za[:, 0], zb[:, 1] - zb[:, 0], target, target, 1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    values['loss'].backward()
    assert torch.isfinite(values['loss'])
    outside = torch.arange(12) != 8
    assert torch.equal(za.grad[outside], torch.zeros_like(za.grad[outside]))
    assert torch.equal(zb.grad[outside], torch.zeros_like(zb.grad[outside]))
    # The source is not labelled; it still influences the contextual target.
    embedding_gradient = model.get_input_embeddings().weight.grad
    source_gradient = float(embedding_gradient[original_ids[0, 1:5]].abs().sum())
    assert source_gradient > 0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm) and norm > 0
    optimizer.step()
    assert not torch.cuda.is_initialized()
    out = Path(__file__).resolve().parents[1] / 'results/local_repair_pair_objective_v1'
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'CPU_objective_checked_not_real_data_trained',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'unchanged_unlabelled_logits_have_no_direct_loss': True,
              'equal_side_weight_despite_different_target_token_counts': True,
              'ranking_weight_zero_equals_paired_local_BCE': True,
              'preference_swap_increases_fixture_loss': True,
              'both_sides_get_finite_gradients_in_expected_direction': True,
              'real_tiny_encoder_backward_and_one_optimizer_step': True,
              'source_context_embedding_gradient_l1': source_gradient,
              'unlabelled_logits_direct_gradient_zero': True,
              'GPU_used': False, 'pretrained_weights_loaded': False,
              'real_dataset_read': False, 'official_test_opened': False,
              'training_protocol_or_new_model_comparison_completed': False}
    (out / 'CPU_CHECK.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    cpu_selfcheck()
