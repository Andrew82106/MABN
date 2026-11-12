"""Fixed-epoch fold-excluded refits; no validation selection or held labels.

Epoch count is copied from the completed full-fit validation selection. Each
fold starts from the same generic pretrained model, then predicts its held fit
groups with the LAST checkpoint. There is no held-fold evaluation or threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import train_full_context as base
from predict_full_context import encode_unlabeled, predict_records, FORBIDDEN

HERE = Path(__file__).resolve().parent


def verify_bindings(meta, out):
    """Bind the exact generic model and root-owned fold IDs before training."""
    fit_ids_path = Path(meta["protocol"]["fit_id_subset"])
    assert base.sha(fit_ids_path) == meta["source_sha256"][str(fit_ids_path.resolve())]
    pretrained = Path(meta["protocol"]["model"])
    for name, digest in meta["model_sha256"].items():
        assert base.sha(pretrained / name) == digest
    fit = base.rows(out / "fit_inputs.jsonl")
    held = base.rows(out / "held_inputs.jsonl")
    assert base.read_fit_ids(fit_ids_path) == {r["response_id"] for r in fit}
    held_ids_path = fit_ids_path.with_name(fit_ids_path.name.replace("_fit_ids", "_held_ids"))
    assert held_ids_path != fit_ids_path and held_ids_path.exists()
    assert base.read_fit_ids(held_ids_path) == {r["response_id"] for r in held}
    assert not {r["group_id"] for r in fit} & {r["group_id"] for r in held}
    return {"fit_ids": {"path": str(fit_ids_path.resolve()), "sha256": base.sha(fit_ids_path)},
        "held_ids": {"path": str(held_ids_path.resolve()), "sha256": base.sha(held_ids_path)},
        "pretrained_model": {"path": str(pretrained.resolve()), "files_sha256": meta["model_sha256"]}}


def prepare(args):
    assert not torch.cuda.is_initialized()
    assert not (args.out / "prepared.json").exists()
    full = base.prepared(argparse.Namespace(out=args.full_run))
    completed = base.read(args.full_run / "complete.json")
    epoch_count = completed["best"]["epoch"]
    assert 1 <= epoch_count <= 6
    retained = base.read_fit_ids(args.fit_ids)
    aa_path, ss_path = (args.data / "fit" / name for name in ("answers.jsonl", "sequences.jsonl"))
    answers, sequence_rows = base.rows(aa_path), base.rows(ss_path)
    sequences = {r["response_id"]: r for r in sequence_rows}
    assert retained and retained < {r["response_id"] for r in answers}
    assert set(sequences) == {r["response_id"] for r in answers}
    tokenizer = AutoTokenizer.from_pretrained(full["protocol"]["model"], local_files_only=True)
    fit, held = [], []
    full_prepared = {r["response_id"]: r for r in base.rows(args.full_run / "fit_inputs.jsonl")}
    for answer in answers:
        rid = answer["response_id"]
        sequence = sequences[rid]
        # Whitelist input fields. Held targets never enter encoder or output.
        text = {k: answer[k] for k in ("response_id", "question", "retrieved_passages", "original_response")}
        geometry = {k: sequence[k] for k in ("response_token_offsets", "lexical_mask")}
        record = encode_unlabeled(text, geometry, tokenizer, group_id=answer["group_id"])
        assert record["input_ids"] == full_prepared[rid]["input_ids"]
        assert record["mapping"] == full_prepared[rid]["mapping"]
        if rid in retained:
            record["risk_mask"] = sequence["risk_mask"]
            assert len(record["risk_mask"]) == record["raw_token_count"]
            assert all(not y or x for x, y in zip(record["lexical_mask"], record["risk_mask"]))
            fit.append(record)
        else:
            assert not FORBIDDEN & set(record)
            held.append(record)
    fit_groups, held_groups = ({r["group_id"] for r in subset} for subset in (fit, held))
    assert not fit_groups & held_groups
    args.out.mkdir(parents=True, exist_ok=True)
    base.write_rows(args.out / "fit_inputs.jsonl", fit)
    base.write_rows(args.out / "held_inputs.jsonl", held)
    protocol = dict(full["protocol"])
    protocol.update(epochs=epoch_count, selection="none; last epoch only; fixed full-fit-selected epoch count",
        threshold="none; held fit predictions only", fit_id_subset=str(args.fit_ids.resolve()))
    base.save(args.out / "prepared.json", {"protocol": protocol,
        "fit_answers": len(fit), "fit_groups": len(fit_groups), "held_answers": len(held), "held_groups": len(held_groups),
        "source_sha256": {str(p.resolve()): base.sha(p) for p in
                          (aa_path, ss_path, args.fit_ids, args.full_run / "complete.json")},
        "prepared_sha256": {name: base.sha(args.out / name) for name in ("fit_inputs.jsonl", "held_inputs.jsonl")},
        "model_sha256": full["model_sha256"], "full_run": str(args.full_run.resolve()),
        "held_labels_used": False, "validation_opened": False, "test_labels_read": False, "GPU_used": False})
    print("CROSSFIT_PREPARED", len(fit), len(held), "fixed_epochs", epoch_count, flush=True)


def cpu_check(args):
    meta = base.prepared(args)
    bindings = verify_bindings(meta, args.out)
    assert base.read(Path(meta["full_run"]) / "cpu_check.json")["passed"]
    fit, held = (base.rows(args.out / name) for name in ("fit_inputs.jsonl", "held_inputs.jsonl"))
    assert not {r["group_id"] for r in fit} & {r["group_id"] for r in held}
    assert all(not FORBIDDEN & set(r) for r in held)
    weights = base.group_weights(fit)
    mass = Counter()
    for record, weight in zip(fit, weights):
        mass[record["group_id"]] += float(weight.sum())
        assert not weight[~np.asarray(record["lexical_mask"], bool)].any()
    assert max(abs(v - 1 / len(mass)) for v in mass.values()) < 1e-8
    assert not torch.cuda.is_initialized()
    base.save(args.out / "cpu_check.json", {"passed": True,
        "fit_and_held_groups_disjoint": True, "held_inputs_have_no_targets": True,
        "all_encodings_identical_to_full_fit": True, "group_loss_mass_equal": True,
        "exact_model_and_root_fold_ids_verified": bindings,
        "model_gradient_path_reuses_passed_full_run_CPU_check": True,
        "prepared_sha256": base.sha(args.out / "prepared.json"), "GPU_used": False})
    print("CROSSFIT_CPU_CHECK_PASSED", flush=True)


def train(args):
    meta = base.prepared(args)
    gate = base.read(args.out / "cpu_check.json")
    assert gate["passed"] and gate["prepared_sha256"] == base.sha(args.out / "prepared.json")
    full_run = Path(meta["full_run"])
    assert base.read(full_run / "gpu_smoke.json")["passed"]
    assert not (args.out / "train_started.json").exists()
    bindings = verify_bindings(meta, args.out)
    base.configure_gpu()
    fit, held = (base.rows(args.out / name) for name in ("fit_inputs.jsonl", "held_inputs.jsonl"))
    assert all(not FORBIDDEN & set(r) for r in held)
    weights = base.group_weights(fit)
    protocol = meta["protocol"]
    model = base.load_model(meta).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=.01, foreach=False)
    base.save(args.out / "train_started.json", {"started_at": time.time(), "prepared_sha256": base.sha(args.out / "prepared.json")})
    history, started = [], time.perf_counter()
    for epoch in range(1, protocol["epochs"] + 1):
        model.train(); torch.cuda.reset_peak_memory_stats()
        tick, online = time.perf_counter(), 0.
        order = np.random.default_rng(protocol["seed"] + epoch).permutation(len(fit))
        batch = protocol["effective_batch_answers"]
        for first in range(0, len(fit), batch):
            chosen = order[first:first + batch]
            optimizer.zero_grad(set_to_none=True)
            for i in chosen:
                record = fit[i]
                z = base.logits(model, record, "cuda")
                y = torch.tensor(record["risk_mask"], dtype=torch.float32, device="cuda")
                weight = torch.as_tensor(weights[i], dtype=torch.float32, device="cuda")
                objective = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * weight).sum() * len(fit) / len(chosen)
                assert torch.isfinite(objective)
                objective.backward(); online += float(objective.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            if (first // batch) % 25 == 0:
                print("CROSSFIT_TRAIN", args.out.name, epoch, first + len(chosen), len(fit), flush=True)
        entry = {"epoch": epoch, "seconds": time.perf_counter() - tick,
            "online_objective_sum": online, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
        history.append(entry)
        base.save(args.out / f"epoch_{epoch:02d}.json", entry)
        print("CROSSFIT_EPOCH_COMPLETE", epoch, flush=True)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": protocol["epochs"], "protocol": protocol,
        "prepared_sha256": base.sha(args.out / "prepared.json")}, args.out / "last.pt")
    del optimizer
    predictions = predict_records(model, held, "cuda")
    base.write_rows(args.out / "held_fit_predictions.jsonl", predictions)
    prediction_manifest = {"schema": "same-domain-held-fit-prediction-binding-v1",
        "fixed_epochs": protocol["epochs"], "fit_answers": len(fit), "held_answers": len(held),
        **bindings,
        "prepared": {"path": str((args.out / "prepared.json").resolve()), "sha256": base.sha(args.out / "prepared.json")},
        "fit_inputs": {"path": str((args.out / "fit_inputs.jsonl").resolve()), "sha256": base.sha(args.out / "fit_inputs.jsonl")},
        "held_inputs": {"path": str((args.out / "held_inputs.jsonl").resolve()), "sha256": base.sha(args.out / "held_inputs.jsonl")},
        "checkpoint": {"path": str((args.out / "last.pt").resolve()), "sha256": base.sha(args.out / "last.pt")},
        "predictions": {"path": str((args.out / "held_fit_predictions.jsonl").resolve()), "sha256": base.sha(args.out / "held_fit_predictions.jsonl")},
        "runner_sha256": {str(Path(__file__).resolve()): base.sha(__file__),
                          str(Path(base.__file__).resolve()): base.sha(base.__file__)},
        "held_groups_excluded_from_training": True, "held_labels_used": False,
        "validation_opened": False, "test_labels_read": False}
    base.save(args.out / "held_prediction_manifest.json", prediction_manifest)
    base.save(args.out / "complete.json", {"status": "fixed_epoch_fold_fit_and_held_predictions_complete",
        "history": history, "seconds": time.perf_counter() - started,
        "fit_answers": len(fit), "held_answers": len(held),
        "last_checkpoint_sha256": base.sha(args.out / "last.pt"),
        "predictions_sha256": base.sha(args.out / "held_fit_predictions.jsonl"),
        "held_prediction_manifest": {"path": str((args.out / "held_prediction_manifest.json").resolve()),
                                     "sha256": base.sha(args.out / "held_prediction_manifest.json")},
        "held_labels_used": False, "held_metrics_computed": False,
        "validation_opened": False, "test_labels_read": False})
    print("CROSSFIT_COMPLETE", args.out.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "cpu-check", "train"])
    parser.add_argument("--data", type=Path, default=HERE.parent / "data")
    parser.add_argument("--full-run", type=Path, default=HERE / "modernbert_base_v1")
    parser.add_argument("--fit-ids", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        assert args.fit_ids is not None
    torch.set_num_threads(4)
    {"prepare": prepare, "cpu-check": cpu_check, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
