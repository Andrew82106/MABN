"""CPU integration checks; no real checkpoint, training, or test data."""
from pathlib import Path
import torch
from transformers import ModernBertConfig, ModernBertForTokenClassification
import modernbert_generation_bridge as bridge
from generation_residual_fusion import GenerationResidualFusion
import run_full_context_encoder_v2 as base
import run_development as q


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(20261006)
    cfg = ModernBertConfig(vocab_size=100, hidden_size=768, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=12, max_position_embeddings=128,
        local_attention=16, global_attn_every_n_layers=1, reference_compile=False,
        num_labels=2, attention_dropout=0., embedding_dropout=0., mlp_dropout=0.,
        classifier_dropout=0., pad_token_id=0, cls_token_id=1, sep_token_id=2)
    model = ModernBertForTokenClassification(cfg).eval()
    fusion = GenerationResidualFusion().eval()
    row = {'input_ids': [1, 5, 6, 7, 8, 9, 2], 'raw_token_count': 4,
           'mapping': [[0, 0, 1, 2, 3], [2, 3, 3, 4, 5], [.5, .5, 1., 1., 1.]]}
    lb, nll = torch.rand(4, 1024), torch.rand(4)
    expected = base.logits(model, row, 'cpu')
    actual, hidden = bridge.forward_aligned(model, row, 'cpu')
    output, original = bridge.forward_fused(model, fusion, row, lb, nll, 'cpu')
    assert torch.equal(expected, actual) and torch.equal(output, expected) and torch.equal(original, expected)
    assert hidden.shape == (4, 768)
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0., 1., 0., 1.])).backward()
    assert fusion.residual.weight.grad.abs().max() > 0
    assert model.model.embeddings.tok_embeddings.weight.grad.abs().max() > 0
    assert len(model.model._forward_hooks) == 0
    values = torch.arange(7 * 768, dtype=torch.float32).reshape(7, 768)
    oracle = torch.stack([(values[2] + values[3]) / 2, values[3], values[4], values[5]])
    assert torch.equal(bridge.map_hidden(values, row['mapping'], 4), oracle)
    report = {'passed': True, 'real_tiny_modernbert_architecture': True,
        'unchanged_baseline_forward_exact': True, 'zero_residual_fused_logits_exact': True,
        'independent_hidden_mapping_exact': True, 'backbone_and_residual_readout_gradients_nonzero': True,
        'hook_removed': True, 'GPU_used': False, 'training_performed': False,
        'scope': 'CPU integration only; no checkpoint/dataset/baseline changed',
        'source_sha256': q.sha(Path(bridge.__file__)), 'check_source_sha256': q.sha(Path(__file__))}
    q.save(q.ROOT / 'results/generation_residual_fusion_preparation/BRIDGE_CPU_SELFCHECK.json', report)
    print('BRIDGE_CPU_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()
