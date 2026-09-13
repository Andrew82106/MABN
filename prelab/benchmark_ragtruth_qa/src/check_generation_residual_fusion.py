"""Bounded CPU-only synthetic checks; no optimizer, real data or GPU calls."""
from pathlib import Path
import hashlib
import json

import torch

from generation_residual_fusion import GenerationResidualFusion


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(20261002)
    net = GenerationResidualFusion().cpu()
    hidden = torch.randn(2, 7, 768)
    baseline = torch.randn(2, 7)
    lb = torch.rand(2, 7, 1024)
    nll = torch.rand(2, 7, 1) * 8
    inputs = (hidden, baseline, lb, nll)
    parts = net.components(*inputs)
    assert parts['semantic'].shape == parts['generation'].shape == (2, 7, 32)
    assert parts['logit'].shape == parts['gate'].shape == (2, 7)
    assert torch.equal(parts['logit'], baseline)
    assert torch.equal(net(*(x.reshape(-1, *x.shape[2:]) for x in inputs)), baseline.flatten())
    assert torch.count_nonzero(parts['correction']) == 0
    assert torch.all((parts['gate'] > 0) & (parts['gate'] < 1))

    # Zero readout intentionally delays projection/gate gradients until it moves.
    net(*inputs).sum().backward()
    assert torch.count_nonzero(net.residual.weight.grad) > 0
    assert torch.count_nonzero(net.residual.bias.grad) > 0
    delayed = ('semantic.0.weight', 'lookback.0.weight', 'nll.0.weight', 'gate.weight')
    assert all(torch.count_nonzero(dict(net.named_parameters())[k].grad) == 0 for k in delayed)

    # Synthetic nonzero readout, NOT a fitted model or saved checkpoint.
    with torch.no_grad():
        net.residual.weight.copy_(torch.linspace(-.05, .07, 65).reshape(1, 65))
        net.residual.bias.fill_(.03)
    reference = net(*inputs).detach()
    changes = {}
    for i, name in enumerate(('semantic_hidden', 'baseline_logit', 'lookback', 'nll')):
        varied = [x.clone() for x in inputs]
        if name == 'baseline_logit':
            varied[i][0, 0] += .5
        else:
            varied[i][0, 0, 0] += .5
        other = net(*varied).detach()
        changes[name] = float((other - reference).abs().max())
        assert changes[name] > 0, name
        # The fusion head has no cross-token state/context or padding interaction.
        assert torch.equal(other.flatten()[1:], reference.flatten()[1:])

    net.zero_grad(set_to_none=True)
    trainable_inputs = [x.clone().requires_grad_() for x in inputs]
    net(*trainable_inputs).sum().backward()
    gradients = {}
    for name, value in net.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        gradients[name] = int(torch.count_nonzero(value.grad))
        assert gradients[name] > 0, name
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and
               torch.count_nonzero(x.grad) > 0 for x in trainable_inputs)
    try:
        net(hidden, baseline, lb, -torch.ones_like(nll))
    except ValueError:
        pass
    else:
        raise AssertionError('Negative NLL accepted')

    here = Path(__file__).resolve()
    module = here.with_name('generation_residual_fusion.py')
    out = here.parents[1] / 'results/generation_residual_fusion_preparation'
    out.mkdir(exist_ok=True, parents=True)
    dest = out / 'CPU_SELFCHECK.json'
    assert not dest.exists(), 'Do not overwrite a prior preparation check'
    report = {'passed': True, 'scope': 'synthetic CPU module checks only',
              'seed': 20261002, 'parameter_count': sum(p.numel() for p in net.parameters()),
              'shape_checks_passed': True, 'initial_baseline_exact': True,
              'initial_correction_exact_zero': True,
              'zero_readout_delays_branch_gradients_as_expected': True,
              'nonzero_test_readout_input_change_max_abs': changes,
              'nonzero_test_readout_all_parameter_and_input_gradients_nonzero_finite': True,
              'independent_per_token_no_added_context': True,
              'optimizer_or_training_performed': False, 'model_checkpoint_saved': False,
              'real_data_or_test_content_read': False, 'GPU_used': False,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (module, here)}}
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
