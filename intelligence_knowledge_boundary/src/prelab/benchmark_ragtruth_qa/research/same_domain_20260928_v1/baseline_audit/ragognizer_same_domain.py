"""Fresh RAGognizer architecture, NF4/BF16 joint LoRA + LM/BCE training.

Only fit and validation are accepted. No published detector weights are read.
prepare/selfcheck use CPU; smoke/train require --allow-gpu explicitly.
This is a disclosed precision/runtime adaptation, not the released checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
BENCH = RUN.parent.parent
BASE = BENCH.parent / "models/Llama-2-7b-chat-hf"
TOKENIZER = BENCH / "research/ragognizer_formal_baseline_v1/official_base_tokenizer_meta_f5db"
OFFICIAL_HEAD = BENCH / "third_party/transformer-heads/transformer_heads/model/head.py"
OFFICIAL_FT = BENCH / "third_party/RAGognizer/fine-tuning/ft.py"
HEAD = "hallu_head_neg_16"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl(path, values):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def recipe(microbatch_size=1, pair_within_update=False):
    return {
        "identity": "RAGognizer original architecture, same-domain NF4 precision adaptation",
        "base": str(BASE), "initialization": "pretrained generic base; fresh LoRA and fresh MLP",
        "published_detector_weights_loaded": False,
        "lora": {"r": 32, "alpha": 16, "dropout": 0., "bias": "none", "targets": TARGETS},
        "head": {"name": HEAD, "layer_hook": -16, "sizes": [4096, 1024, 1024, 1],
                 "hidden_activation": "ReLU", "biases": [True, True, False]},
        "loss": "response causal LM cross-entropy + response masked BCEWithLogits; lambda=1",
        "balanced": "positive BCE weight = (negative+positive)/positive, fit only",
        "optimizer": "torch AdamW; lr4e-5, betas(.9,.999), eps1e-8, weight_decay0",
        "epochs": 5, "batch_size": microbatch_size, "accumulation": 8 // microbatch_size,
        "effective_answers_per_update": 8, "loss_normalization": "per-answer LM/BCE then equal answer mean", "gradient_clip": 1.,
        "length_pairing": "sort only inside each preexisting8-answer update; preserve shuffle state and update membership" if pair_within_update else None,
        "schedule": "one cosine schedule across five epochs, warmup0.1",
        "precision": "NF4 base, double quantization, BF16 compute/frozen embeddings/LM projection; FP32 norms/trainable weights/Adam states",
        "memory": "nonreentrant gradient checkpointing; response-only chunked checkpointed LM loss",
        "selection": "primary: shared no-hint validation selection_key maximum; also retain minimum joint-loss checkpoint; ties earlier epoch",
        "pooling": "report native mean4 and shared lexical max4; choose validation selection_key, ties mean; freeze with thresholds",
        "training_code_gaps": ["official default script opens test during initialization",
            "official --ragtruth formatter lacks return for get_test=True",
            "official unbalanced branch prints undefined zero/one counters",
            "official lm_head target defaults to lm_head while dataset field is labels",
            "published exact checkpoint launch and allentries/balanced flags unavailable"],
        "disclosures": ["NF4 rather than released BF16; not a bitwise author rerun",
            "fresh jointly trained LoRA; this is not frozen-feature head-only fitting",
            "paper joint objective explicitly wired; one continuous schedule, not repeated trainer.train calls",
            "only legal spans (valid=true) supervise original native token coordinates",
            "no truncation, cropping, answer filtering, or early stopping; five fixed epochs"],
    }


def prompt_messages(prompt):
    match = re.match(r"^(?:<s>\s*)?\[INST\]", prompt)
    if not match:
        return [{"role": "user", "content": prompt}]
    if prompt.count("[INST]") != 1 or prompt.count("[/INST]") != 1 or not prompt.rstrip().endswith("[/INST]"):
        raise ValueError("Unsupported multi-turn prompt envelope")
    content = prompt[match.end():prompt.rfind("[/INST]")]
    if content.startswith(" "):
        content = content[1:]
    if content.endswith(" "):
        content = content[:-1]
    if content.startswith("<<SYS>>"):
        stop = content.index("<</SYS>>")
        return [{"role": "system", "content": content[7:stop].strip("\n")},
                {"role": "user", "content": content[stop + 8:].lstrip("\n")}]
    return [{"role": "user", "content": content}]


def make_native(tokenizer, answer, sequence):
    """Official chat template; only data-coordinate/valid-span adaptation."""
    chat = prompt_messages(answer["released_prompt"])
    response = answer["original_response"]
    text = tokenizer.apply_chat_template(chat + [{"role": "assistant", "content": response}], tokenize=False,
                                         add_generation_prompt=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = enc["input_ids"]
    expected = tokenizer.apply_chat_template(chat + [{"role": "assistant", "content": response}], tokenize=True,
                                             add_generation_prompt=False)
    assert ids == expected, "Template-tokenization mismatch"
    if len(ids) > 4096:
        raise ValueError(f"No silent truncation: {answer['response_id']} has {len(ids)} tokens")
    stripped = response.strip()
    if not stripped:
        raise ValueError("Empty response")
    start = text.rindex(stripped)
    leading = len(response) - len(response.lstrip())
    stop = start + len(stripped)
    offsets, positions = [], []
    bce = [-1] * len(ids)
    labels = [-100] * len(ids)
    spans = [s for s in answer["released_hallucinations"] if s.get("valid", True)]
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a < stop and b > start and b > a:
            lo, hi = max(0, a - start + leading), min(len(response), b - start + leading)
            positions.append(i)
            offsets.append([lo, hi])
            bce[i] = int(any(max(lo, s["start"]) < min(hi, s["end"]) for s in spans))
    assert positions
    # Match response-only causal LM training including the response-end/EOS token.
    for i in range(positions[0], len(ids)):
        labels[i] = ids[i]
    # Original template strips outside whitespace; carry it in external coordinates only.
    offsets[0][0] = 0
    offsets[-1][1] = len(response)
    return {"response_id": answer["response_id"], "group_id": answer["group_id"],
            "partition": answer["partition"], "input_ids": ids, "lm_labels": labels, "bce_labels": bce,
            "response_positions": positions, "response_offsets": offsets,
            "shared_offsets": sequence["response_token_offsets"],
            "shared_risk": sequence["risk_mask"], "shared_lexical": sequence["lexical_mask"],
            "answer_risk": answer["answer_risk"], "no_hint": not answer["unanswerability_hint_setting"]}


def prepare(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    # Author adds an unused distinct pad entry; batch size 1 means no actual padding.
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "PREPARED.json").exists():
        raise FileExistsError("Preparation already exists")
    counts, bindings, groups = {}, {}, {}
    for split in ("fit", "validation"):
        pa, ps = args.data / split / "answers.jsonl", args.data / split / "sequences.jsonl"
        aa, ss = rows(pa), rows(ps)
        assert len(aa) == len(ss) and all(a["response_id"] == s["response_id"] for a, s in zip(aa, ss))
        assert all(a["partition"] == split and a["official_split"] == "train" for a in aa)
        groups[split] = {a["group_id"] for a in aa}
        native = [make_native(tok, a, s) for a, s in zip(aa, ss)]
        positives = sum(sum(y == 1 for y in a["bce_labels"]) for a in native)
        negatives = sum(sum(y == 0 for y in a["bce_labels"]) for a in native)
        path = args.out / f"{split}.jsonl"
        jsonl(path, native)
        bindings[split] = {"answers": sha(pa), "sequences": sha(ps), "prepared": sha(path)}
        lengths = sorted(len(a["input_ids"]) for a in native)
        counts[split] = {"answers": len(native), "groups": len(groups[split]), "positive_tokens": positives,
                         "negative_tokens": negatives, "max_tokens": max(lengths),
                         "p95_tokens": lengths[math.ceil(.95 * len(lengths)) - 1], "total_tokens": sum(lengths),
                         "no_hint_answers": sum(a["no_hint"] for a in native)}
    assert not groups["fit"] & groups["validation"]
    fit = counts["fit"]
    assert fit["positive_tokens"] and fit["negative_tokens"]
    tok.save_pretrained(args.out / "tokenizer")
    save(args.out / "PREPARED.json", {"recipe": recipe(), "counts": counts, "bindings": bindings,
        "pos_weight": (fit["positive_tokens"] + fit["negative_tokens"]) / fit["positive_tokens"],
        "tokenizer_sha256": sha(TOKENIZER / "tokenizer.json"), "official_head_sha256": sha(OFFICIAL_HEAD),
        "official_training_source_sha256": sha(OFFICIAL_FT), "test_opened": False, "gpu_used": False})
    print(json.dumps(counts, indent=2), flush=True)


def create_head(torch):
    class DetectionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = torch.nn.ModuleList([torch.nn.Linear(4096, 1024), torch.nn.Linear(1024, 1024),
                                            torch.nn.Linear(1024, 1, bias=False)])

        def forward(self, x):
            return self.lins[2](torch.relu(self.lins[1](torch.relu(self.lins[0](x)))))
    return DetectionHead()


def selfcheck(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(17)
    tree = ast.parse(OFFICIAL_HEAD.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLPHead")
    namespace = {"torch": torch, "nn": torch.nn, "HeadConfig": object,
                 "activation_map": {"linear": torch.nn.Identity}}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(OFFICIAL_HEAD), "exec"), namespace)
    original = namespace["MLPHead"](HEAD, 4096, 1024, 3, "linear", output_bias=False)
    local = create_head(torch)
    local.load_state_dict(original.state_dict())
    x = torch.randn(5, 4096, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    p, q = original(x), local(z)
    p.square().sum().backward()
    q.square().sum().backward()
    assert torch.equal(p, q) and torch.equal(x.grad, z.grad)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(original.parameters(), local.parameters()))
    assert sum(p.numel() for p in local.parameters()) == 5245952
    # Response CE chunking is algebraically the same objective and gradient.
    layer = torch.nn.Linear(12, 29, bias=False)
    layer.requires_grad_(False)
    a = torch.randn(9, 12, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    yy = torch.arange(9) % 29
    loss1 = torch.nn.functional.cross_entropy(layer(a), yy)
    loss2 = sum(torch.nn.functional.cross_entropy(layer(b[k:k+3]), yy[k:k+3], reduction="sum")
                for k in range(0, 9, 3)) / 9
    loss1.backward(); loss2.backward()
    assert torch.allclose(loss1, loss2, atol=1e-6) and torch.allclose(a.grad, b.grad, atol=1e-7)
    save(args.out / "CPU_SELFCHECK.json", {"status": "PASS", "official_head_logits_and_gradients_exact": True,
          "chunked_lm_loss_and_gradients_close": True, "head_parameters": 5245952,
          "test_opened": False, "gpu_used": False})
    print("CPU fidelity checks PASS", flush=True)


def load_training_model(args, torch, checkpoint=None):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(args.prepared / "tokenizer", local_files_only=True)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, quantization_config=config,
              torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if len(tok) != base.config.vocab_size:
        base.resize_token_embeddings(len(tok), mean_resizing=False)
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
              gradient_checkpointing_kwargs={"use_reentrant": False})
    # PEFT preparation casts every nonquantized tensor to FP32. These two frozen
    # matrices came from BF16 weights and the official headed loader casts them
    # back to BF16 as well; keep norms and all learned adapter/head weights FP32.
    base.get_input_embeddings().to(dtype=torch.bfloat16)
    base.get_output_embeddings().to(dtype=torch.bfloat16)
    if checkpoint is None:
        model = get_peft_model(base, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.,
                 bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    else:
        model = PeftModel.from_pretrained(base, checkpoint / "lora", is_trainable=True, local_files_only=True)
    head = create_head(torch).cuda().float()
    if checkpoint:
        head.load_state_dict(torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True))
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert lora_names and all("lora_" in n for n in lora_names)
    assert all(sum(f".{name}.lora_A." in n for n in lora_names) == 32 for name in TARGETS)
    assert all(sum(f".{name}.lora_B." in n for n in lora_names) == 32 for name in TARGETS)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 79953920
    assert not model.get_base_model().get_output_embeddings().weight.requires_grad
    return model, head


def forward(torch, model, head, row, pos_weight, train=True, lm_chunk=64):
    """Identical network/response objective; avoid retaining full vocabulary logits."""
    from torch.utils.checkpoint import checkpoint
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
    bce_y = torch.tensor(row["bce_labels"], dtype=torch.float32, device="cuda")
    lm_y = torch.tensor(row["lm_labels"], dtype=torch.long, device="cuda")[1:]
    core = model.get_base_model()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = core.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                         output_hidden_states=True, return_dict=True)
        logits = head(out.hidden_states[-16][0]).float().view(-1)
        mask = bce_y >= 0
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], bce_y[mask],
                      pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device="cuda"))
        lm_mask = lm_y != -100
        hs = out.last_hidden_state[0, :-1][lm_mask]
        targets = lm_y[lm_mask]
        assert len(targets)
        total = torch.zeros((), device="cuda", dtype=torch.float32)
        lm_head = core.get_output_embeddings()
        def ce(hidden, targets):
            return torch.nn.functional.cross_entropy(lm_head(hidden).float(), targets, reduction="sum")
        for start in range(0, len(targets), lm_chunk):
            h, y = hs[start:start+lm_chunk], targets[start:start+lm_chunk]
            total = total + (checkpoint(ce, h, y, use_reentrant=False) if train else ce(h, y))
        loss_lm = total / len(targets)
    total_loss = loss_lm + loss_bce
    assert bool(torch.isfinite(total_loss))
    return total_loss, loss_lm, loss_bce, logits


def shared_tokens(native_prob, row):
    import numpy as np
    offsets = row["response_offsets"]
    token = []
    for a, b in row["shared_offsets"]:
        ix = [i for i, (x, y) in enumerate(offsets) if max(a, x) < min(b, y)]
        if not ix:
            raise ValueError(f"Unmapped shared token for {row['response_id']}: {a,b}")
        token.append(float(np.asarray(native_prob)[ix].mean()))
    return token


def shared_scores(native_prob, row, pooling="mean"):
    import numpy as np
    token = shared_tokens(native_prob, row)
    k = min(4, len(token))
    starts = [s for s in range(len(token)-k+1) if any(row["shared_lexical"][s:s+k])]
    if pooling == "mean":
        scores = [float(np.mean(token[s:s+k])) for s in starts]
    elif pooling == "max":
        scores = [float(max(p for p, lexical in zip(token[s:s+k], row["shared_lexical"][s:s+k]) if lexical)) for s in starts]
    else:
        raise ValueError(pooling)
    labels = [int(any(row["shared_risk"][s:s+k])) for s in starts]
    return scores, labels


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {"n": len(y), "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "ap": float(average_precision_score(y, p)) if sum(y) else None}


def evaluate(torch, model, head, values, pos_weight, output):
    model.eval(); head.eval()
    wy, wp, ay, ap, predictions, losses = [], [], [], [], [], []
    with torch.no_grad():
        for i, row in enumerate(values):
            loss, lm, bce, logits = forward(torch, model, head, row, pos_weight, train=False)
            probs = torch.sigmoid(logits[row["response_positions"]]).cpu().numpy()
            token_scores = shared_tokens(probs, row)
            scores, labels = shared_scores(probs, row)
            max_scores, max_labels = shared_scores(probs, row, pooling="max")
            assert labels == max_labels
            wy.extend(labels); wp.extend(scores); ay.append(row["answer_risk"]); ap.append(max(scores))
            losses.append([float(loss), float(lm), float(bce)])
            predictions.append({"response_id": row["response_id"], "group_id": row["group_id"],
                "native_probabilities": probs.tolist(), "native_offsets": row["response_offsets"],
                "token_scores": token_scores, "window_scores_by_pooling": {"mean": scores, "max": max_scores},
                "window_scores": scores, "answer_score": max(scores)})
            if (i+1) % 25 == 0:
                print(f"VALIDATION {i+1}/{len(values)}", flush=True)
    import numpy as np
    report = {"joint_loss": float(np.mean([x[0] for x in losses])),
              "lm_loss": float(np.mean([x[1] for x in losses])), "bce_loss": float(np.mean([x[2] for x in losses])),
              "window": metrics(wy, wp), "answer": metrics(ay, ap), "test_opened": False}
    sys.path.insert(0, str(RUN))
    from metrics import evaluate_partition
    tokens_by_response = {r["response_id"]: r["token_scores"] for r in predictions}
    pooling_evaluations = {pooling: evaluate_partition("validation", tokens_by_response, pooling=pooling)
                           for pooling in ("mean", "max")}
    selected_pooling = max(("mean", "max"), key=lambda p: tuple(pooling_evaluations[p]["selection_key"]))
    report["pooling_evaluations"] = pooling_evaluations
    report["selected_pooling"] = selected_pooling
    report["common_evaluation"] = pooling_evaluations[selected_pooling]
    for row in predictions:
        row["pooling"] = selected_pooling
        row["window_scores"] = row["window_scores_by_pooling"][selected_pooling]
        row["answer_score"] = max(row["window_scores"])
    for pooling, evaluation in pooling_evaluations.items():
        save(output / f"VALIDATION_{pooling.upper()}.json", evaluation)
    jsonl(output / "validation_predictions.jsonl", predictions)
    save(output / "VALIDATION.json", report)
    return report


def run_gpu(args, smoke=False):
    if not args.allow_gpu:
        raise RuntimeError("GPU run must be scheduled explicitly with --allow-gpu")
    if args.pair_within_update and args.microbatch_size != 2:
        raise ValueError("Length pairing is only available for microbatch2")
    import torch
    import numpy as np
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 compute unavailable; no automatic precision fallback")
    prepared = json.loads((args.prepared / "PREPARED.json").read_text(encoding="utf-8"))
    for split in ("fit", "validation"):
        assert sha(args.prepared / f"{split}.jsonl") == prepared["bindings"][split]["prepared"]
    fit, val = rows(args.prepared / "fit.jsonl"), rows(args.prepared / "validation.jsonl")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh run output; existing weights are never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "RECIPE.json", recipe(args.microbatch_size, args.pair_within_update))
    save(args.out / "RUN_BINDING.json", {"prepared_sha256": sha(args.prepared / "PREPARED.json"),
         "code_sha256": sha(__file__), "seed": args.seed, "smoke_only": smoke,
         "metrics_sha256": sha(RUN / "metrics.py"),
         "microbatch_size": args.microbatch_size,
         "pair_within_update": args.pair_within_update,
         "batch_runtime_sha256": sha(HERE / "ragognizer_batch_runtime.py") if args.microbatch_size == 2 else None,
         "software": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "bitsandbytes")}})
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, head = load_training_model(args, torch)
    model_load_seconds = time.time() - started
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=4e-5, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)
    steps_per_epoch = math.ceil(len(fit) / 8)
    total_steps = steps_per_epoch * 5
    warmup = math.ceil(total_steps * .1)
    def rate(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    if smoke:
        # A real update (not zero-LR warmup) checks optimizer memory and parameter changes.
        for group in optimizer.param_groups:
            group["lr"] = 4e-5
    pos_weight = prepared["pos_weight"]
    model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
    # Longest fit case exercises the worst known train length without opening held-out data.
    if smoke:
        order = sorted(range(len(fit)), key=lambda j: len(fit[j]["input_ids"]), reverse=True)[:2]
    else:
        order = list(range(len(fit)))
    history, best, best_shared = [], math.inf, None
    gradient_audited = False
    smoke_timings = []
    for epoch in range(1, (1 if smoke else 5) + 1):
        if not smoke:
            random.Random(args.seed + epoch).shuffle(order)
        execution_order = list(order)
        if args.pair_within_update:
            from ragognizer_batch_runtime import pair_inside_updates
            execution_order = pair_inside_updates(order, [len(row["input_ids"]) for row in fit])
        assert all(sorted(execution_order[i:i+8]) == sorted(order[i:i+8]) for i in range(0, len(order), 8))
        save(args.out / f"UPDATE_GROUPS_epoch_{epoch}.json", {
            "optimizer_update_members": [[fit[j]["response_id"] for j in order[i:i+8]] for i in range(0, len(order), 8)],
            "execution_order": [fit[j]["response_id"] for j in execution_order],
            "pair_within_update": args.pair_within_update})
        model.train(); head.train()
        running = []
        for i in range(0, len(order), args.microbatch_size):
            indices = execution_order[i:i+args.microbatch_size]
            consumed = i + len(indices)
            # Correct scaling of the final shorter accumulation block.
            block = min(8, len(order) - (i // 8) * 8)
            if smoke:
                torch.cuda.synchronize()
                tick = time.perf_counter()
            if args.microbatch_size == 1:
                loss, lm, bce, logits = forward(torch, model, head, fit[indices[0]], pos_weight, train=True)
            else:
                from ragognizer_batch_runtime import forward_many
                loss, lm, bce, logits = forward_many(torch, model, head, [fit[idx] for idx in indices], pos_weight, train=True)
            if smoke:
                torch.cuda.synchronize()
                middle = time.perf_counter()
            (loss * (len(indices) / block)).backward()
            if smoke:
                torch.cuda.synchronize()
                smoke_timings.append({"response_ids": [fit[idx]["response_id"] for idx in indices],
                    "tokens": sum(len(fit[idx]["input_ids"]) for idx in indices),
                    "padded_tokens": len(indices) * max(len(fit[idx]["input_ids"]) for idx in indices),
                    "forward_seconds": middle-tick, "backward_seconds": time.perf_counter()-middle})
            running.extend([[float(loss.detach()), float(lm.detach()), float(bce.detach())]] * len(indices))
            del loss, lm, bce, logits
            if consumed % 8 == 0 or consumed == len(order):
                if smoke or not gradient_audited:
                    active = {name: sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
                              for n, p in model.named_parameters() if f".{name}.lora_B." in n) for name in TARGETS}
                    if not all(n == 32 for n in active.values()) or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                        raise RuntimeError(f"Joint gradient audit failed: {active}")
                    save(args.out / "JOINT_GRADIENT_AUDIT.json", {"status": "PASS", "active_lora_B_by_projection": active,
                         "head_finite_gradients": True, "epoch": epoch, "answers": consumed})
                    gradient_audited = True
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if consumed % 10 == 0 or smoke:
                progress = {"epoch": epoch, "answers": consumed, "total": len(order), "last_losses": running[-1],
                    "microbatch_size": args.microbatch_size,
                    "elapsed_seconds": time.time()-started, "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved()}
                save(args.out / "PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
        if smoke:
            save(args.out / "GPU_SMOKE.json", {"status": "PASS", "longest_fit_lengths": [len(fit[j]["input_ids"]) for j in order],
                 "microbatch_size": args.microbatch_size,
                 "joint_lora_gradient_groups": active, "head_parameters": sum(p.numel() for p in head.parameters()),
                 "lora_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "optimizer_step_completed": True, "losses": running, "elapsed_seconds": time.time()-started,
                 "model_load_seconds": model_load_seconds, "per_answer_timings": smoke_timings,
                 "estimated_epoch_compute_seconds_linear_tokens": prepared["counts"]["fit"]["total_tokens"] /
                     sum(t["tokens"] for t in smoke_timings) * sum(t["forward_seconds"]+t["backward_seconds"] for t in smoke_timings),
                 "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
                 "test_opened": False})
            return
        destination = args.out / f"epoch_{epoch}"
        destination.mkdir()
        model.save_pretrained(destination / "lora", save_embedding_layers=True)
        torch.save(head.state_dict(), destination / "head.pt")
        report = evaluate(torch, model, head, val, pos_weight, destination)
        report.update(epoch=epoch, train_loss=np.mean(running, axis=0).tolist())
        history.append(report)
        if report["joint_loss"] < best:
            best = report["joint_loss"]
            save(args.out / "SELECTED_JOINT_LOSS.json", {"epoch": epoch, "checkpoint": str(destination),
                 "selection": "minimum validation joint loss; ties earlier epoch", "validation_loss": best,
                 "pooling": report["selected_pooling"], "thresholds": report["common_evaluation"]["thresholds"]})
        shared_key = tuple(report["common_evaluation"]["selection_key"])
        if best_shared is None or shared_key > best_shared:
            best_shared = shared_key
            selected = {"epoch": epoch, "checkpoint": str(destination),
                "selection": "maximum shared no-hint validation selection_key; ties earlier epoch",
                "selection_key": list(shared_key), "pooling": report["selected_pooling"],
                "thresholds": report["common_evaluation"]["thresholds"]}
            save(args.out / "SELECTED_SHARED_F1.json", selected)
            save(args.out / "SELECTED.json", selected)
        save(args.out / "HISTORY.json", history)
    save(args.out / "COMPLETE.json", {"status": "complete", "epochs": 5, "elapsed_seconds": time.time()-started,
         "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
         "recipe": recipe(args.microbatch_size, args.pair_within_update), "prepared_sha256": sha(args.prepared / "PREPARED.json"), "code_sha256": sha(__file__),
         "software": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "bitsandbytes")},
         "test_opened": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "selfcheck", "batch-selfcheck", "smoke", "train"])
    parser.add_argument("--data", type=Path, default=RUN / "data")
    parser.add_argument("--prepared", type=Path, default=HERE / "ragognizer_prepared_v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--pair-within-update", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.command in ("prepare", "selfcheck", "batch-selfcheck"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "prepare": prepare(args)
    elif args.command == "selfcheck": selfcheck(args)
    elif args.command == "batch-selfcheck":
        from ragognizer_batch_runtime import cpu_selfcheck
        cpu_selfcheck(args.out)
    else: run_gpu(args, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
