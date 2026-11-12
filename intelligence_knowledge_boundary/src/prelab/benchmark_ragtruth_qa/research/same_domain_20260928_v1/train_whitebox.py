"""CPU sequence probe over frozen white-box signals; test prediction is label-free."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F

import metrics as common_metrics

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SIGNALS = HERE / "signals"
SEED = 20260928
CONFIG = {"lb_width": 1024, "probability_width": 3, "hidden_width": 64,
          "pca_components_per_layer": 32, "samples_per_answer": 16,
          "branch_widths": [32, 16, 32], "sequence_width": 64,
          "conv_dilations": [1, 2], "conv_kernel_size": 3, "dropout": 0.2,
          "epochs": 24, "batch_size": 8, "learning_rate": 0.001,
          "weight_decay": 0.01, "cpu_threads": 4}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_predictions(path, scores, fit_ids):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    fit_ids = set(fit_ids)
    with temporary.open("w", encoding="utf-8") as handle:
        for rid, values in scores.items():
            handle.write(json.dumps({"response_id": rid, "token_scores": values.tolist(),
                                     "training_member": rid in fit_ids}, separators=(",", ":")) + "\n")
    temporary.replace(path)


def parse_ids(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = line.strip()
            value.append(row["response_id"] if isinstance(row, dict) else row)
    if isinstance(value, dict):
        keys = [key for key in ("response_ids", "fit_ids", "train_ids") if key in value]
        if len(keys) != 1:
            raise ValueError("ID manifest must contain one of response_ids/fit_ids/train_ids")
        value = value[keys[0]]
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a nonempty ID list")
    result = [row["response_id"] if isinstance(row, dict) else str(row) for row in value]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate IDs")
    return set(result)


def fit_members(path=None):
    members = [row for row in readl(DATA / "split_assignments.jsonl") if row["split"] == "fit"]
    wanted = parse_ids(path) if path else {row["response_id"] for row in members}
    if not wanted <= {row["response_id"] for row in members}:
        raise ValueError("--fit-ids may contain only official experiment fit IDs")
    grouped = defaultdict(set)
    for row in members:
        grouped[row["group_id"]].add(row["response_id"])
    for group, ids in grouped.items():
        if ids & wanted and not ids <= wanted:
            raise ValueError(f"--fit-ids splits source group {group}")
    return [row for row in members if row["response_id"] in wanted]


def selected_target_rows(path, wanted):
    """Decode targets only for selected fit IDs, never for held-out fit rows."""
    selected = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            # Canonical sequence exports put response_id near the line start.
            match = re.search(r'"response_id"\s*:\s*"([^"]+)"', line[:400])
            if match is None:
                raise ValueError("Missing response_id in canonical sequence header")
            rid = match.group(1)
            if rid in wanted:
                assert rid not in selected
                selected[rid] = json.loads(line)
    assert set(selected) == set(wanted)
    return selected


class SignalReader:
    """Load only geometry and frozen arrays; no answer/sealed target file access."""
    def __init__(self, partitions):
        self.geometry = {}
        for partition in partitions:
            self.geometry.update({row["response_id"]: row for row in readl(DATA / partition / "geometry.jsonl")})
        self.receipts = {}

    def __call__(self, rid):
        path = SIGNALS / f"{rid}.npz"
        receipt = read_json(SIGNALS / f"{rid}.json")
        if rid not in self.receipts:
            assert sha(path) == receipt["npz_sha256"], rid
            self.receipts[rid] = {key: receipt[key] for key in
                                  ("npz_sha256", "source_sha256", "input_sha256")}
        else:
            assert all(receipt[key] == value for key, value in self.receipts[rid].items())
        geometry = self.geometry[rid]
        assert receipt["input_sha256"] == geometry["input_sha256"], rid
        with np.load(path, allow_pickle=False) as archive:
            signal = {key: archive[key] for key in (
                "lb_prefix_pre_header", "nll", "entropy", "selected_vs_best_logp",
                "hidden_mid", "hidden_last", "token_ids", "answer_token_positions",
                "response_token_offsets", "lexical_mask")}
        n = len(geometry["response_token_ids"])
        assert signal["lb_prefix_pre_header"].shape == (n, 1024)
        assert signal["hidden_mid"].shape == signal["hidden_last"].shape == (n, 4096)
        for key, geom_key in (("token_ids", "response_token_ids"),
                              ("answer_token_positions", "answer_token_positions"),
                              ("response_token_offsets", "response_token_offsets"),
                              ("lexical_mask", "lexical_mask")):
            assert np.array_equal(signal[key], geometry[geom_key]), (rid, key)
        assert all(len(value) == n and np.isfinite(value).all() for value in signal.values())
        assert signal["lexical_mask"].any()
        return signal


def sample_positions(lexical_mask):
    available = np.flatnonzero(lexical_mask)
    if not len(available):
        raise ValueError("An answer has no lexical tokens")
    # Always 16 slots per answer. Very short answers repeat evenly spaced slots.
    return available[np.rint(np.linspace(0, len(available) - 1, 16)).astype(np.int64)]


def probability_block(signal):
    return np.stack([signal[key] for key in ("nll", "entropy", "selected_vs_best_logp")], axis=1)


def fit_preprocessor(fit_ids, reader):
    samples = {key: [] for key in ("lb", "probability", "mid", "last")}
    for rid in fit_ids:
        signal = reader(rid)
        indices = sample_positions(signal["lexical_mask"])
        for key, value in (("lb", signal["lb_prefix_pre_header"]),
                           ("probability", probability_block(signal)),
                           ("mid", signal["hidden_mid"]), ("last", signal["hidden_last"])):
            samples[key].append(value[indices].astype(np.float32, copy=True))
    samples = {key: np.concatenate(value) for key, value in samples.items()}
    if min(samples["mid"].shape) < 32 or min(samples["last"].shape) < 32:
        raise ValueError("PCA32 needs at least 32 samples and hidden dimensions")
    state = {"fit_ids": list(fit_ids), "fit_ids_sha256": digest(list(fit_ids)),
             "samples_per_answer": 16, "sample_axis": "uniform lexical tokens, repeating if fewer than 16",
             "lb_scaler": StandardScaler().fit(samples["lb"]),
             "probability_scaler": StandardScaler().fit(samples["probability"])}
    hidden = []
    for name in ("mid", "last"):
        pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED)
        pca.fit(samples[name])
        hidden.append(pca.transform(samples[name]).astype(np.float32))
        state[f"pca_{name}"] = pca
        del samples[name]
    state["hidden_scaler"] = StandardScaler().fit(np.concatenate(hidden, axis=1))
    return state


def transform(state, signal):
    hidden = np.concatenate([state[f"pca_{key}"].transform(signal[f"hidden_{key}"])
                             for key in ("mid", "last")], axis=1)
    result = np.concatenate((state["lb_scaler"].transform(signal["lb_prefix_pre_header"]),
                             state["probability_scaler"].transform(probability_block(signal)),
                             state["hidden_scaler"].transform(hidden)), axis=1).astype(np.float32)
    assert np.isfinite(result).all()
    return result


def training_weights(rows):
    """Base mass 1/group, then 1/answer within group, then 1/lexical token."""
    counts = Counter(row["group_id"] for row in rows)
    groups = len(counts)
    weights = {}
    for row in rows:
        lexical = np.asarray(row["lexical_mask"], bool)
        if not lexical.any():
            raise ValueError("No lexical target")
        weights[row["response_id"]] = lexical.astype(np.float64) / (
            groups * counts[row["group_id"]] * lexical.sum())
    positive = sum(float(np.dot(weights[row["response_id"]], row["risk_mask"])) for row in rows)
    if not 0 < positive < 1:
        raise ValueError("Selected fit must contain both target classes")
    classes = {"negative": 0.5 / (1 - positive), "positive": 0.5 / positive,
               "weighted_positive_fraction": positive}
    assert np.isclose(sum(value.sum() for value in weights.values()), 1)
    return weights, classes


class ResidualConv(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation, padding=dilation)
        self.dropout = nn.Dropout(0.2)

    def forward(self, value, mask):
        normalized = self.norm(value) * mask
        delta = self.conv(normalized.transpose(1, 2)).transpose(1, 2)
        return (value + self.dropout(F.gelu(delta))) * mask


class WhiteboxProbe(nn.Module):
    def __init__(self, lb_width=1024):
        super().__init__()
        self.lb_width = lb_width
        self.lb = nn.Linear(lb_width, 32)
        self.probability = nn.Linear(3, 16)
        self.hidden = nn.Linear(64, 32)
        self.project = nn.Linear(80, 64)
        self.dropout = nn.Dropout(0.2)
        self.blocks = nn.ModuleList([ResidualConv(1), ResidualConv(2)])
        self.output = nn.Linear(64, 1)

    def forward(self, features, valid_tokens):
        mask = valid_tokens[:, :, None].to(features.dtype)
        # Mask before each convolution, including after normalization biases.
        pieces = [F.gelu(self.lb(features[:, :, :self.lb_width])),
                  F.gelu(self.probability(features[:, :, self.lb_width:self.lb_width + 3])),
                  F.gelu(self.hidden(features[:, :, self.lb_width + 3:]))]
        value = self.dropout(F.gelu(self.project(torch.cat(pieces, dim=-1)))) * mask
        for block in self.blocks:
            value = block(value, mask)
        return self.output(value).squeeze(-1) * valid_tokens


def collate(rows, weights=None):
    n, longest, width = len(rows), max(len(row["features"]) for row in rows), rows[0]["features"].shape[1]
    x = torch.zeros(n, longest, width)
    mask = torch.zeros(n, longest, dtype=torch.bool)
    y, w = torch.zeros(n, longest), torch.zeros(n, longest)
    for i, row in enumerate(rows):
        length = len(row["features"])
        x[i, :length] = torch.from_numpy(row["features"])
        mask[i, :length] = True
        if weights is not None:
            y[i, :length] = torch.as_tensor(row["risk_mask"], dtype=torch.float32)
            w[i, :length] = torch.as_tensor(weights[row["response_id"]], dtype=torch.float32)
    return x, mask, y, w


@torch.inference_mode()
def predict(model, rows):
    model.eval()
    scores = {}
    for begin in range(0, len(rows), CONFIG["batch_size"]):
        batch = rows[begin:begin + CONFIG["batch_size"]]
        x, mask, _, _ = collate(batch)
        probabilities = model(x, mask).sigmoid().numpy()
        for i, row in enumerate(batch):
            scores[row["response_id"]] = probabilities[i, :len(row["features"])].copy()
    return scores


def prepare_features(ids, reader, preprocessor, labels=None):
    prepared = []
    for rid in ids:
        signal = reader(rid)
        row = {"response_id": rid, "features": transform(preprocessor, signal),
               "lexical_mask": signal["lexical_mask"].astype(bool)}
        if labels is not None:
            target = labels[rid]
            assert np.array_equal(row["lexical_mask"], target["lexical_mask"])
            row.update(group_id=target["group_id"], risk_mask=target["risk_mask"])
        prepared.append(row)
    return prepared


def provenance(reader, ids):
    receipts = {rid: reader.receipts[rid] for rid in ids}
    sources = {value["source_sha256"] for value in receipts.values()}
    assert len(sources) == 1, "Mixed frozen-signal implementation versions"
    return {"source_sha256": next(iter(sources)), "signal_receipts_sha256": digest(receipts),
            "receipts": receipts}


def epoch_validation(model, rows, fixed_final):
    if fixed_final:
        return None, None
    scores = predict(model, rows)
    return scores, common_metrics.evaluate_partition("validation", scores)


def train(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN.json").exists() or (out / "best.pt").exists():
        raise RuntimeError("Use a fresh --out for each fit/fold; training does not overwrite a run")
    members = fit_members(args.fit_ids)
    ids = [row["response_id"] for row in members]
    validation_ids = [] if args.fixed_final else [row["response_id"] for row in readl(DATA / "validation/geometry.jsonl")]
    reader = SignalReader(("fit",) if args.fixed_final else ("fit", "validation"))
    started = time.perf_counter()
    print(json.dumps({"stage": "fit_preprocessor", "fit_answers": len(ids), "validation_answers": len(validation_ids)}), flush=True)
    preprocessor = fit_preprocessor(ids, reader)
    with (out / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    target_rows = selected_target_rows(DATA / "fit/sequences.jsonl", set(ids))
    fit_rows = prepare_features(ids, reader, preprocessor, target_rows)
    validation_rows = prepare_features(validation_ids, reader, preprocessor)
    weights, classes = training_weights(fit_rows)
    signal_source = provenance(reader, ids + validation_ids)
    save(out / "FIT_IDS.json", ids)
    config = dict(CONFIG, epochs=args.epochs)
    selection = ("fixed final epoch; no validation or held-out target use" if args.fixed_final else
                 "validation no_hint, common metrics selection_key; ties keep earliest epoch")
    run = {"config": config, "seed": SEED, "fit_ids": ids,
           "fit_ids_sha256": digest(ids), "fit_groups": len({row["group_id"] for row in members}),
           "fit_ids_file_sha256": sha(out / "FIT_IDS.json"), "fixed_final": args.fixed_final,
           "fit_subset_group_complete": True, "fit_only_class_weights": classes,
           "preprocessor_sha256": sha(out / "preprocessor.pkl"),
           "data_manifest_sha256": sha(DATA / "MANIFEST.json"),
           "source_sha256": sha(Path(__file__)), "metrics_sha256": sha(Path(common_metrics.__file__)),
           "signal_source": signal_source, "test_labels_read": False,
           "validation_used_for_preprocessing_or_class_balance": False,
           "validation_targets_used_for_selection": not args.fixed_final,
           "heldout_fit_targets_decoded": False, "selection": selection,
           "fit_predictions_are_not_oof": True}
    save(out / "RUN.json", run)
    model = WhiteboxProbe().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = np.random.default_rng(SEED)
    history, best_key, best_record = [], None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(fit_rows))
        total_loss, batches = 0., 0
        for begin in range(0, len(order), 8):
            batch = [fit_rows[int(i)] for i in order[begin:begin + 8]]
            x, mask, targets, base_weight = collate(batch, weights)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            class_weight = torch.where(targets > 0.5, classes["positive"], classes["negative"])
            # Uniform answer batches estimate the group-balanced full objective.
            loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") *
                    base_weight * class_weight).sum() * (len(fit_rows) / len(batch))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()); batches += 1
        scores, evaluation = epoch_validation(model, validation_rows, args.fixed_final)
        record = {"epoch": epoch, "mean_batch_loss": total_loss / batches,
                  "elapsed_seconds": time.perf_counter() - started, "evaluation": evaluation}
        history.append(record)
        save(out / ("training_history.json" if args.fixed_final else "validation_history.json"), history)
        key = tuple(evaluation["selection_key"]) if evaluation else ()
        select = epoch == args.epochs if args.fixed_final else (best_key is None or key > best_key)
        if select:
            best_key = key
            best_record = record
            checkpoint = {"state_dict": model.state_dict(), "config": config,
                          "epoch": epoch, "thresholds": evaluation["thresholds"] if evaluation else None,
                          "fixed_final": args.fixed_final, "selection": selection,
                          "selection_key": list(key), "fit_ids": ids,
                          "preprocessor_sha256": run["preprocessor_sha256"],
                          "run_manifest_sha256": sha(out / "RUN.json")}
            temporary = out / "best.pt.pending"
            torch.save(checkpoint, temporary)
            temporary.replace(out / "best.pt")
            save(out / "SELECTED.json", dict(record, selection=selection, fixed_final=args.fixed_final))
            if not args.fixed_final:
                save(out / "best_validation_metrics.json", record)
                write_predictions(out / "best_validation_token_predictions.jsonl", scores, ids)
        print(json.dumps({"epoch": epoch, "validation_key": list(key), "best_key": list(best_key) if best_key is not None else None,
                          "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    save(out / "COMPLETE.json", {"epochs": args.epochs, "best": best_record, "fixed_final": args.fixed_final,
                                 "checkpoint_sha256": sha(out / "best.pt"), "test_labels_read": False})


def predict_partition(args):
    out = args.out.resolve()
    run = read_json(out / "RUN.json")
    assert sha(DATA / "MANIFEST.json") == run["data_manifest_sha256"]
    assert sha(out / "preprocessor.pkl") == run["preprocessor_sha256"]
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["run_manifest_sha256"] == sha(out / "RUN.json")
    assert checkpoint["preprocessor_sha256"] == run["preprocessor_sha256"]
    with (out / "preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    ids = [row["response_id"] for row in readl(DATA / args.predict_partition / "geometry.jsonl")]
    if args.predict_ids:
        wanted = parse_ids(args.predict_ids)
        if not wanted <= set(ids):
            raise ValueError("Prediction IDs do not belong to requested partition")
        ids = [rid for rid in ids if rid in wanted]
    reader = SignalReader((args.predict_partition,))
    rows = prepare_features(ids, reader, preprocessor)
    signal_source = provenance(reader, ids)
    assert signal_source["source_sha256"] == run["signal_source"]["source_sha256"]
    model = WhiteboxProbe().cpu()
    model.load_state_dict(checkpoint["state_dict"])
    scores = predict(model, rows)
    target = out / f"token_predictions_{args.predict_partition}.jsonl"
    write_predictions(target, scores, run["fit_ids"])
    save(target.with_suffix(".manifest.json"), {
        "partition": args.predict_partition, "responses": len(ids), "ids_sha256": digest(ids),
        "checkpoint_sha256": sha(out / "best.pt"), "prediction_sha256": sha(target),
        "training_run_sha256": sha(out / "RUN.json"), "fit_ids_sha256": run["fit_ids_sha256"],
        "fit_ids_file_sha256": run["fit_ids_file_sha256"], "fixed_final": run["fixed_final"],
        "thresholds": checkpoint["thresholds"],
        "training_member_count": len(set(ids) & set(run["fit_ids"])),
        "all_predictions_out_of_fit": not bool(set(ids) & set(run["fit_ids"])),
        "labels_read": False, "test_labels_read": False, "signal_source": signal_source})
    print(json.dumps({"predictions": str(target), "responses": len(ids), "labels_read": False}), flush=True)


def cpu_test(args):
    assert epoch_validation(None, None, fixed_final=True) == (None, None)
    fixture_path = args.out / "CPU_SELECTION_FIXTURE.jsonl"
    try:
        fixture_path.write_text(
            '{"response_id":"chosen","risk_mask":[1,0]}\n'
            '{"response_id":"heldout","risk_mask":INVALID_MUST_NOT_BE_DECODED}\n', encoding="utf-8")
        assert selected_target_rows(fixture_path, {"chosen"}) == {
            "chosen": {"response_id": "chosen", "risk_mask": [1, 0]}}
    finally:
        fixture_path.unlink(missing_ok=True)
    model = WhiteboxProbe().eval()
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(2, 9, 1091, generator=generator)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.inference_mode():
        original = model(x, mask)
        wider = torch.cat((x, 100 * torch.randn(2, 8, 1091, generator=generator)), dim=1)
        wide_mask = torch.cat((mask, torch.zeros(2, 8, dtype=torch.bool)), dim=1)
        padded = model(wider, wide_mask)
        individual = model(x[:1, :5], torch.ones(1, 5, dtype=torch.bool))
    assert torch.allclose(original[mask], padded[:, :9][mask], atol=1e-6, rtol=1e-6)
    assert torch.allclose(original[0, :5], individual[0], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(padded[~wide_mask]) == 0
    fixture = [
        {"response_id": "a", "group_id": "g1", "lexical_mask": [1, 1, 0], "risk_mask": [0, 1, 0]},
        {"response_id": "b", "group_id": "g2", "lexical_mask": [1, 1], "risk_mask": [1, 0]},
        {"response_id": "c", "group_id": "g2", "lexical_mask": [1, 0, 1, 1], "risk_mask": [0, 0, 0, 1]},
    ]
    weights, classes = training_weights(fixture)
    assert np.isclose(weights["a"].sum(), 0.5)
    assert np.isclose(weights["b"].sum(), 0.25) and np.isclose(weights["c"].sum(), 0.25)
    assert np.isclose(classes["positive"] * classes["weighted_positive_fraction"], 0.5)
    assert np.isclose(classes["negative"] * (1 - classes["weighted_positive_fraction"]), 0.5)
    rng = np.random.default_rng(SEED)
    sources = {}
    for rid in ("fit_a", "fit_b", "fit_c", "fit_d", "validation_FORBIDDEN"):
        sources[rid] = {"lb_prefix_pre_header": rng.normal(size=(21, 1024)).astype(np.float32),
                        "nll": rng.normal(size=21).astype(np.float32),
                        "entropy": rng.normal(size=21).astype(np.float32),
                        "selected_vs_best_logp": rng.normal(size=21).astype(np.float32),
                        "hidden_mid": rng.normal(size=(21, 48)).astype(np.float32),
                        "hidden_last": rng.normal(size=(21, 48)).astype(np.float32),
                        "lexical_mask": np.ones(21, bool)}
    calls, ids = [], ["fit_a", "fit_b", "fit_c", "fit_d"]
    def fit_only_reader(rid):
        assert rid in ids, "Validation entered preprocessing"
        calls.append(rid)
        return sources[rid]
    state = fit_preprocessor(ids, fit_only_reader)
    assert calls == ids
    sampled = np.concatenate([sources[rid]["lb_prefix_pre_header"][sample_positions(sources[rid]["lexical_mask"])] for rid in ids])
    assert np.allclose(state["lb_scaler"].mean_, sampled.mean(axis=0, dtype=np.float64))
    state_hash = hashlib.sha256(pickle.dumps(state)).hexdigest()
    for value in sources["validation_FORBIDDEN"].values():
        if value.dtype != bool:
            value[:] = 1000000
    transform(state, sources["validation_FORBIDDEN"])
    assert hashlib.sha256(pickle.dumps(state)).hexdigest() == state_hash
    for row in fixture:
        row["features"] = rng.normal(size=(len(row["lexical_mask"]), 1091)).astype(np.float32)
    model.train()
    x, mask, targets, weight = collate(fixture, weights)
    loss = (F.binary_cross_entropy_with_logits(model(x, mask), targets, reduction="none") * weight).sum()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    save(args.out / "CPU_TEST.json", {"passed": True,
         "padding_and_batch_length_invariance": True, "group_answer_lexical_equal_mass": True,
         "fit_only_pca_and_scaler": True, "validation_transform_does_not_mutate_preprocessor": True,
         "finite_training_gradients": True, "GPU_used": False, "real_signals_loaded": False,
         "fixed_final_skips_validation_prediction_and_targets": True,
         "unselected_fit_target_payloads_not_decoded": True,
         "source_sha256": sha(Path(__file__))})
    print("WHITEBOX_CPU_TEST_PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu-test", "train", "predict"))
    parser.add_argument("--fit-ids", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "whitebox_model")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--fixed-final", action="store_true",
                        help="Use the fixed final epoch; do not load validation features/targets or select thresholds")
    parser.add_argument("--predict-partition", choices=("fit", "validation", "test"), default="validation")
    parser.add_argument("--predict-ids", type=Path)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        {"cpu-test": cpu_test, "train": train, "predict": predict_partition}[args.action](args)


if __name__ == "__main__":
    main()
