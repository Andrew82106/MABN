"""Optional padded microbatch runtime; preserves the per-answer joint objective."""
from contextlib import nullcontext
import copy


def pair_inside_updates(order, lengths):
    """Reorder only within each existing eight-answer optimizer update."""
    result = []
    for begin in range(0, len(order), 8):
        group = order[begin:begin+8]
        result.extend(sorted(group, key=lambda index: lengths[index]))
    assert len(result) == len(order)
    assert all(sorted(result[i:i+8]) == sorted(order[i:i+8]) for i in range(0, len(order), 8))
    return result


def forward_many(torch, model, head, rows, pos_weight, train=True, lm_chunk=64, device="cuda"):
    from torch.utils.checkpoint import checkpoint
    core = model.get_base_model()
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), core.config.pad_token_id, device=device, dtype=torch.long)
    attention = torch.zeros_like(ids)
    lm_labels = torch.full_like(ids, -100)
    targets = torch.full((len(rows), width), -1., device=device, dtype=torch.float32)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device, dtype=torch.long)
        attention[i, :n] = 1
        lm_labels[i, :n] = torch.tensor(row["lm_labels"], device=device, dtype=torch.long)
        targets[i, :n] = torch.tensor(row["bce_labels"], device=device, dtype=torch.float32)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with context:
        out = core.model(input_ids=ids, attention_mask=attention, use_cache=False,
                         output_hidden_states=True, return_dict=True)
        logits = head(out.hidden_states[-16]).float().squeeze(-1)
        lm_head = core.get_output_embeddings()
        def ce(hidden, y):
            return torch.nn.functional.cross_entropy(lm_head(hidden).float(), y, reduction="sum")
        lms, bces = [], []
        for i, row in enumerate(rows):
            bce_mask = targets[i] >= 0
            bces.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[i, bce_mask], targets[i, bce_mask],
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)))
            y = lm_labels[i, 1:]
            mask = y != -100
            hidden, y = out.last_hidden_state[i, :-1][mask], y[mask]
            total = torch.zeros((), device=device, dtype=torch.float32)
            for start in range(0, len(y), lm_chunk):
                h, target = hidden[start:start+lm_chunk], y[start:start+lm_chunk]
                total = total + (checkpoint(ce, h, target, use_reentrant=False) if train else ce(h, target))
            lms.append(total / len(y))
        loss_lm, loss_bce = torch.stack(lms).mean(), torch.stack(bces).mean()
    loss = loss_lm + loss_bce
    assert bool(torch.isfinite(loss))
    return loss, loss_lm, loss_bce, logits


def cpu_selfcheck(outdir):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    import ragognizer_same_domain as main
    torch.set_num_threads(2)
    torch.manual_seed(20260928)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, pad_token_id=0, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    base = LlamaForCausalLM(cfg)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=2, lora_dropout=0., bias="none",
        task_type="CAUSAL_LM", target_modules=main.TARGETS))
    head = torch.nn.Sequential(torch.nn.Linear(32, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8),
                               torch.nn.ReLU(), torch.nn.Linear(8, 1, bias=False))
    batched_model, batched_head = copy.deepcopy(model), copy.deepcopy(head)
    rows = []
    for length, start in ((13, 4), (9, 5)):
        ids = torch.randint(1, 64, (length,)).tolist()
        rows.append({"input_ids": ids, "lm_labels": [-100]*start + ids[start:],
                     "bce_labels": [-1]*start + [i % 2 for i in range(length-start)]})
    # Independent, unpadded, nonchunked single-answer reference.
    reference_lm, reference_bce = [], []
    for row in rows:
        ids = torch.tensor([row["input_ids"]])
        core = model.get_base_model()
        result = core.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            output_hidden_states=True, return_dict=True)
        native = head(result.hidden_states[-16][0]).squeeze(-1)
        targets = torch.tensor(row["bce_labels"], dtype=torch.float32)
        mask = targets >= 0
        reference_bce.append(torch.nn.functional.binary_cross_entropy_with_logits(native[mask], targets[mask],
                              pos_weight=torch.tensor(2.6)))
        y = torch.tensor(row["lm_labels"])[1:]
        mask = y != -100
        reference_lm.append(torch.nn.functional.cross_entropy(core.get_output_embeddings()(result.last_hidden_state[0, :-1][mask]), y[mask]))
    ref_lm, ref_bce = torch.stack(reference_lm).mean(), torch.stack(reference_bce).mean()
    reference = ref_lm + ref_bce
    reference.backward()
    actual, lm, bce, _ = forward_many(torch, batched_model, batched_head, rows, 2.6,
                                    train=True, lm_chunk=3, device="cpu")
    actual.backward()
    assert torch.allclose(reference, actual, atol=3e-6, rtol=1e-5)
    assert torch.allclose(ref_lm, lm, atol=3e-6, rtol=1e-5)
    assert torch.allclose(ref_bce, bce, atol=3e-6, rtol=1e-5)
    worst, checked = 0., 0
    for left, right in ((model, batched_model), (head, batched_head)):
        for (n, a), (m, b) in zip(left.named_parameters(), right.named_parameters()):
            assert n == m and (a.grad is None) == (b.grad is None)
            if a.grad is None:
                continue
            delta = float((a.grad-b.grad).abs().max())
            worst = max(worst, delta)
            checked += 1
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=1e-4), (n, delta)
    result = {"status": "PASS", "model": "tiny32-layer Llama + seven-target LoRA +3-layer ReLU head",
        "input_lengths": [13, 9], "response_lengths": [9, 4], "right_padding": True,
        "objective": "per-answer LM and BCE means, then equal answer mean",
        "reference_joint_loss": float(reference), "batched_joint_loss": float(actual),
        "max_gradient_abs_difference": worst, "gradient_tensors_checked": checked,
        "effective_batch8_scaling": "microbatch_mean * microbatch_answers / actual_accumulation_block_answers",
        "default_microbatch_unchanged": 1, "gpu_used": False, "test_opened": False}
    main.save(outdir / "BATCH_CPU_SELFCHECK.json", result)
    print(result, flush=True)
