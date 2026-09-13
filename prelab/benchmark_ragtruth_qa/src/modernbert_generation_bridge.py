"""One unchanged ModernBERT forward plus token-aligned generation residuals.

No loading, training, window pooling, or threshold selection is done here.
Inputs are the already prepared whole-context rows and exact raw-BPE features.
"""
from contextlib import nullcontext
import torch
import tail_finetune as mapping


def map_hidden(hidden, raw_map, raw_count):
    rows = torch.as_tensor(raw_map[0], dtype=torch.long, device=hidden.device)
    cols = torch.as_tensor(raw_map[1], dtype=torch.long, device=hidden.device)
    weights = torch.as_tensor(raw_map[2], dtype=torch.float32, device=hidden.device)
    hidden = hidden.float()
    return hidden.new_zeros(raw_count, hidden.shape[-1]).index_add(
        0, rows, hidden.index_select(0, cols) * weights[:, None])


def forward_aligned(model, row, device, *, bf16_forward=True):
    """Capture final encoder state without changing the classifier's forward.

    The head contains nonlinear operations: map actual logits independently.
    Classifying a mapped/averaged state is not an equivalent baseline.
    """
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    captured = []

    def capture(_module, _args, output):
        captured.append(output.last_hidden_state[0])

    hook = model.model.register_forward_hook(capture)
    context = (torch.autocast('cuda', dtype=torch.bfloat16)
               if torch.device(device).type == 'cuda' and bf16_forward else nullcontext())
    try:
        with context:
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        hook.remove()
    assert len(captured) == 1
    logits = output.logits[0].float()
    raw_map = row['mapping']
    z = mapping.mapped_logits(logits[:, 1] - logits[:, 0], raw_map, row['raw_token_count'])
    h = map_hidden(captured[0], raw_map, row['raw_token_count'])
    return z, h


def forward_fused(model, fusion, row, lookback, nll, device, *, bf16_forward=True):
    z, h = forward_aligned(model, row, device, bf16_forward=bf16_forward)
    lb = torch.as_tensor(lookback, dtype=torch.float32, device=device)
    loss_probability = torch.as_tensor(nll, dtype=torch.float32, device=device)
    if loss_probability.ndim == 1:
        loss_probability = loss_probability[:, None]
    return fusion(h, z, lb, loss_probability), z
