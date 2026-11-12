"""CPU-only hierarchical relation/scope localizer over frozen QA caches.

The claim-level evidence model answers whether an automatic claim is risky.
This readout keeps that score as context but predicts each original 4-BPE
window separately.  Cheap lexical slot indicators tell the readout where a
relation can change (source/step, subject scope, condition, order, quantity,
negation, comparison, and repetition).  All fit-side learned upstream scores
are source-group OOF.  Formal baselines and the official test are untouched.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_development as q  # noqa: E402


OUT = ROOT / "results/relation_scope_localizer_v1"
PREPARED = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
OOF_WINDOW = ROOT / "results/group_crossfit_lb_large_v1/oof_input_scores.npy"
GENERATION = ROOT / "results/development_v1/matrices/base.npy"
LOCAL_NLI = ROOT / "results/nli_local_signal_cuda_scoring_v1/window_nli_features.npy"
CITATION = ROOT / "results/citation_alignment_v1/window_features.npy"
CLAIM_FEATURES = ROOT / "results/retrieved_evidence_nli_citation_v2/claim_features.npy"
CLAIM_SCORES = ROOT / "results/claim_evidence_meta_v1/scores.npz"
CLAIM_FEATURE_NAMES = ROOT / "results/retrieved_evidence_nli_citation_v2/feature_names.json"
INCUMBENT = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4.json"
GROUP_OOF_CONTROL = ROOT / "results/group_crossfit_lb_large_v1/group_oof.json"
CLAIM_CONTROL = ROOT / "results/claim_evidence_meta_v1/summary.json"

SEED = 20261018
FOLDS = 5
C = 0.01


TRIGGERS = {
    "source_or_step": re.compile(
        r"\b(?:passage|source|document|article|step|stage|phase)\s*(?:#|no\.?\s*)?\d*\b|\[[123]\]",
        re.I,
    ),
    "number": re.compile(r"\b\d+(?:[.,:/-]\d+)*\b|[%$€£]"),
    "negation": re.compile(
        r"\b(?:no|not|never|none|neither|without|lack(?:s|ed|ing)?|unable|cannot|can't|"
        r"doesn't|didn't|isn't|aren't|wasn't|weren't)\b",
        re.I,
    ),
    "quantifier": re.compile(
        r"\b(?:all|both|only|every|each|any|always|entire|either|neither|solely|exactly)\b",
        re.I,
    ),
    "condition": re.compile(r"\b(?:if|unless|when|whenever|while|once|provided|depending)\b", re.I),
    "order_or_repeat": re.compile(
        r"\b(?:before|after|then|next|first|second|third|finally|until|previously|again|another|repeat(?:ed)?)\b",
        re.I,
    ),
    "comparison": re.compile(
        r"\b(?:more|less|higher|lower|than|same|different|better|worse|finer|coarser)\b|[<>]",
        re.I,
    ),
    "modal": re.compile(r"\b(?:may|might|could|should|must|can|would|possibly|probably|likely)\b", re.I),
    "attribution": re.compile(
        r"\b(?:according|states?|mentions?|says?|reports?|indicates?|provides?|suggests?)\b",
        re.I,
    ),
    "evidence_absence": re.compile(
        r"\b(?:unable to answer|do(?:es)? not provide|not mentioned|no information|lacks? information|"
        r"insufficient information)\b",
        re.I,
    ),
    "scope_punctuation": re.compile(r"[:;()[\]{}\"']"),
}


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def trigger_vector(text: str):
    return np.asarray([float(bool(pattern.search(text))) for pattern in TRIGGERS.values()], dtype=np.float32)


def scope_geometry(window, claim, answer_chars):
    claim_start, claim_end = int(claim["char_start"]), int(claim["char_end"])
    start, end = int(window["char_start"]), int(window["char_end"])
    length = max(1, claim_end - claim_start)
    local_start = np.clip((start - claim_start) / length, 0.0, 1.0)
    local_end = np.clip((end - claim_start) / length, 0.0, 1.0)
    center = (local_start + local_end) / 2
    overlap = max(0, min(end, claim_end) - max(start, claim_start))
    return np.asarray([
        local_start,
        local_end,
        center,
        min(center, 1 - center),
        float(local_start <= .2),
        float(local_end >= .8),
        overlap / max(1, end - start),
        np.log1p(length),
        start / max(1, answer_chars),
        end / max(1, answer_chars),
    ], dtype=np.float32)


GEOMETRY_NAMES = (
    "claim_local_start", "claim_local_end", "claim_local_center", "claim_edge_distance",
    "claim_first_fifth", "claim_last_fifth", "claim_overlap_fraction", "log_claim_chars",
    "answer_start_fraction", "answer_end_fraction",
)


def build_features(meta, prepared):
    assert len(prepared) == 793
    assert [row["response_id"] for row in prepared] == [answer["response_id"] for answer in meta["answers"]]
    oof = np.load(OOF_WINDOW, mmap_mode="r")
    generation = np.load(GENERATION, mmap_mode="r")
    local_nli = np.load(LOCAL_NLI, mmap_mode="r")
    citation = np.load(CITATION, mmap_mode="r")
    claim_features = np.load(CLAIM_FEATURES, mmap_mode="r")
    with np.load(CLAIM_SCORES, allow_pickle=False) as cache:
        claim_scores = cache["claim_scores"].astype(np.float64)
    assert oof.shape == (210364, 2)
    assert generation.shape == (210364, 1025)
    assert local_nli.shape == (210364, 12)
    assert citation.shape == (210364, 8)
    assert claim_features.shape == (8845, 70)
    assert claim_scores.shape == (8845,)
    claim_names = read(CLAIM_FEATURE_NAMES)["combined"]
    assert len(claim_names) == 70

    base = np.column_stack((oof, np.asarray(generation[:, -1]), local_nli, citation)).astype(np.float32)
    base_names = (["oof_lookback", "oof_large", "generation_nll"] +
                  [f"local_nli_{i}" for i in range(12)] +
                  [f"citation_{i}" for i in range(8)])
    width = (len(base_names) + 4 + len(claim_names) + len(GEOMETRY_NAMES) +
             3 * len(TRIGGERS) + len(GEOMETRY_NAMES))
    x = np.empty((len(meta["windows"]), width), dtype=np.float32)
    selected_claim = np.empty(len(meta["windows"]), dtype=np.int32)
    relation_masks = np.empty((len(meta["windows"]), len(TRIGGERS)), dtype=np.int8)

    cursor = 0
    for answer_index, (answer, row) in enumerate(zip(meta["answers"], prepared)):
        assert answer["response_id"] == row["response_id"]
        local_claims = row["claims"]
        offset = cursor
        cursor += len(local_claims)
        owners = np.asarray(row["lexical_token_claim"], dtype=np.int32)
        assert len(owners) == answer["token_count"]
        answer_text = answer["original_response"]
        for wi in meta["answer_windows"][answer["response_id"]]:
            window = meta["windows"][wi]
            local_ids = sorted({int(owners[token]) for token in window["lexical_token_indices"] if owners[token] >= 0})
            assert local_ids
            global_ids = np.asarray([offset + cid for cid in local_ids], dtype=np.int64)
            scores = claim_scores[global_ids]
            chosen_position = int(np.argmax(scores))
            gid = int(global_ids[chosen_position])
            cid = local_ids[chosen_position]
            claim = local_claims[cid]
            selected_claim[wi] = gid
            local_text = answer_text[window["char_start"]:window["char_end"]]
            local_triggers = trigger_vector(local_text)
            claim_triggers = trigger_vector(claim["text"])
            relation_masks[wi] = local_triggers.astype(np.int8)
            geometry = scope_geometry(window, claim, len(answer_text))
            score_stats = np.asarray([scores.max(), scores.mean(), scores.min(), scores.max() - scores.min()], dtype=np.float32)
            parts = (
                base[wi],
                score_stats,
                np.asarray(claim_features[gid], dtype=np.float32),
                geometry,
                local_triggers,
                claim_triggers,
                local_triggers * float(scores.max()),
                geometry * float(scores.max()),
            )
            x[wi] = np.concatenate(parts)
        if len(meta["answer_windows"][answer["response_id"]]) and (answer_index + 1) % 100 == 0:
            print("RELATION_SCOPE_FEATURES", answer_index + 1, 793, flush=True)
    assert cursor == 8845 and np.isfinite(x).all()
    names = (base_names + ["claim_score_max", "claim_score_mean", "claim_score_min", "claim_score_gap"] +
             [f"claim_nli__{name}" for name in claim_names] + list(GEOMETRY_NAMES) +
             [f"window_has__{name}" for name in TRIGGERS] +
             [f"claim_has__{name}" for name in TRIGGERS] +
             [f"claim_score_x_window__{name}" for name in TRIGGERS] +
             [f"claim_score_x__{name}" for name in GEOMETRY_NAMES])
    assert len(names) == x.shape[1]
    return x, names, selected_claim, relation_masks


def active_weights(meta, active):
    active = np.asarray(active, dtype=np.int64)
    tree = defaultdict(lambda: defaultdict(list))
    for local, global_index in enumerate(active):
        row = meta["windows"][int(global_index)]
        tree[row["group_id"]][row["answer_id"]].append(local)
    base = np.empty(len(active), dtype=np.float64)
    for answers in tree.values():
        for ids in answers.values():
            base[ids] = 1 / (len(answers) * len(ids))
    base /= base.mean()
    y = np.asarray([meta["windows"][int(i)]["label"] for i in active], dtype=np.int8)
    mass = np.bincount(y, weights=base, minlength=2)
    loss = base * (mass.sum() / (2 * mass))[y]
    for answers in tree.values():
        ids = [i for values in answers.values() for i in values]
        loss[ids] *= (len(active) / len(tree)) / loss[ids].sum()
    loss *= len(active) / loss.sum()
    return base, loss, y


def fit_model(x, y, active, base, loss):
    scaler = StandardScaler()
    scaler.fit(x[active], sample_weight=base)
    z = scaler.transform(x[active]).astype(np.float32)
    model = LogisticRegression(C=C, solver="liblinear", max_iter=3000, random_state=SEED)
    model.fit(z, y, sample_weight=loss)
    assert int(model.n_iter_.max()) < 3000
    return scaler, model


def predict(model_pair, x, active):
    scaler, model = model_pair
    return model.predict_proba(scaler.transform(x[active]).astype(np.float32))[:, 1]


def type_recall(meta, scores, threshold, partition):
    left, right = meta["bounds"][partition]
    by_response = {answer["response_id"]: answer for answer in meta["answers"]}
    totals, hits = defaultdict(int), defaultdict(int)
    for wi in range(left, right):
        window = meta["windows"][wi]
        if not window["label"]:
            continue
        labels = by_response[window["response_id"]]["original_labels"]
        kinds = set()
        for label in labels:
            if any(max(span["start"], label["start"]) < min(span["end"], label["end"])
                   for span in window["risk_character_spans"]):
                kinds.add(label["label_type"])
        for kind in kinds:
            totals[kind] += 1
            hits[kind] += int(scores[wi] >= threshold)
    return {kind: {"positive_windows": totals[kind], "hits": hits[kind],
                   "recall": hits[kind] / totals[kind]}
            for kind in sorted(totals)}


def run():
    assert not OUT.exists() or not any(OUT.iterdir()), f"Preserve existing run: {OUT}"
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    meta = q.metadata()
    prepared = rows(PREPARED)
    x, names, selected_claim, relation_masks = build_features(meta, prepared)
    fit_left, fit_right = meta["bounds"]["fit"]
    cal_left, cal_right = meta["bounds"]["calibration"]
    fit_indices = np.arange(fit_left, fit_right, dtype=np.int64)
    groups = np.asarray([window["group_id"] for window in meta["windows"][:fit_right]])
    fit_y = np.asarray([window["label"] for window in meta["windows"][:fit_right]], dtype=np.int8)
    splitter = GroupKFold(FOLDS)
    oof = np.full(len(meta["windows"]), np.nan, dtype=np.float64)
    folds = []
    fold_models = []
    with threadpool_limits(limits=4):
        for fold, (train_local, held_local) in enumerate(splitter.split(fit_indices, fit_y, groups)):
            train = fit_indices[train_local]
            held = fit_indices[held_local]
            base, loss, y = active_weights(meta, train)
            pair = fit_model(x, y, train, base, loss)
            oof[held] = predict(pair, x, held)
            fold_models.append(pair)
            folds.append({"fold": fold, "train_windows": len(train), "held_windows": len(held),
                          "train_groups": len(set(groups[train])), "held_groups": len(set(groups[held])),
                          "iterations": pair[1].n_iter_.tolist()})
            print("RELATION_SCOPE_FOLD", fold, len(train), len(held), flush=True)
    assert np.isfinite(oof[:fit_right]).all()

    base, loss, y = active_weights(meta, fit_indices)
    full_pair = fit_model(x, y, fit_indices, base, loss)
    cal_indices = np.arange(cal_left, cal_right, dtype=np.int64)
    oof[cal_indices] = predict(full_pair, x, cal_indices)
    assert np.isfinite(oof).all()

    answers = q.answer_scores(meta, oof)
    fit_answer_indices = np.asarray([i for i, answer in enumerate(meta["answers"])
                                     if answer["partition"] == "fit"], dtype=np.int64)
    fit_answer_y = np.asarray([meta["answers"][i]["label"] for i in fit_answer_indices], dtype=np.int8)
    thresholds = {
        "window": q.choose_threshold(fit_y, oof[:fit_right]),
        "answer": q.choose_threshold(fit_answer_y, answers[fit_answer_indices]),
    }
    result_metrics = q.metrics(meta, oof, thresholds)
    window_threshold = thresholds["window"]["threshold"]
    diagnostics = {
        "type_recall": {
            partition: type_recall(meta, oof, window_threshold, partition)
            for partition in ("fit", "calibration")
        },
        "relation_trigger_windows": {
            partition: {
                "positive": int(np.count_nonzero(
                    fit_y[fit_left:fit_right] & relation_masks[fit_left:fit_right].any(1)))
                if partition == "fit" else int(np.count_nonzero(
                    np.asarray([w["label"] for w in meta["windows"][cal_left:cal_right]], dtype=bool) &
                    relation_masks[cal_left:cal_right].any(1))),
            }
            for partition in ("fit", "calibration")
        },
    }
    for partition, (left, right) in {"fit": (fit_left, fit_right), "calibration": (cal_left, cal_right)}.items():
        mask = relation_masks[left:right].any(1)
        labels = np.asarray([w["label"] for w in meta["windows"][left:right]], dtype=bool)
        pred = oof[left:right] >= window_threshold
        diagnostics["relation_trigger_windows"][partition].update({
            "hits": int(np.count_nonzero(labels & mask & pred)),
            "recall": float(np.count_nonzero(labels & mask & pred) / max(1, np.count_nonzero(labels & mask))),
            "false_positives": int(np.count_nonzero(~labels & mask & pred)),
        })

    protocol = {
        "version": "relation-scope-localizer-v1",
        "role": "standalone development method candidate; formal baselines unchanged",
        "inputs": {
            "window": "group-held-out Lookback and ModernBERT-large probabilities, frozen NLL, local NLI and citation features",
            "relation": "group-held-out claim_evidence_meta score plus frozen 70-column retrieved-evidence E/N/C and citation features",
            "scope": list(TRIGGERS) + list(GEOMETRY_NAMES),
        },
        "training": {
            "model": "StandardScaler + L2 LogisticRegression",
            "C": C, "seed": SEED, "folds": FOLDS,
            "crossfit": "GroupKFold by source-connected group; every fit prediction is downstream OOF and every learned upstream fit score is upstream OOF",
            "weights": "Within each fold: equal group, then answer/window; binary balance; restore equal group loss mass",
        },
        "readout": "One probability per unchanged eligible 4-original-BPE stride-one window; answer is max window probability",
        "thresholds": "Window and answer thresholds selected only from fit downstream OOF; calibration reporting only",
        "limitations": [
            "Lexical slot cues locate possible relation boundaries but do not parse full predicate-argument structure.",
            "Claim NLI is still based on fixed sentence retrieval and an automatic claim splitter.",
            "The same calibration split has been repeatedly inspected elsewhere and is development evidence only.",
        ],
        "official_test_opened": False,
    }
    source_paths = [PREPARED, OOF_WINDOW, GENERATION, LOCAL_NLI, CITATION, CLAIM_FEATURES,
                    CLAIM_SCORES, CLAIM_FEATURE_NAMES, INCUMBENT, GROUP_OOF_CONTROL, CLAIM_CONTROL]
    save(OUT / "protocol.json", protocol)
    save(OUT / "feature_names.json", names)
    np.savez_compressed(OUT / "scores.npz", window_scores=oof, answer_scores=answers,
                        selected_claim=selected_claim, relation_masks=relation_masks)
    (OUT / "model.pkl").write_bytes(pickle.dumps({"full": full_pair, "folds": fold_models,
                                                   "feature_names": names}, protocol=5))
    summary = {
        "status": "development_only_complete",
        "method": "hierarchical claim relation gate plus 4-BPE scope localizer",
        "features": len(names), "folds": folds,
        "thresholds_from_fit_OOF": thresholds,
        "metrics": result_metrics,
        "diagnostics": diagnostics,
        "controls_read_only": {
            "incumbent": read(INCUMBENT)["metrics"]["calibration"],
            "clean_group_oof_lb_large": read(GROUP_OOF_CONTROL)["metrics"]["calibration"],
            "claim_broadcast": read(CLAIM_CONTROL)["metrics"]["calibration"],
        },
        "source_sha256": {str(path.relative_to(ROOT)): q.sha(path) for path in source_paths},
        "seconds": time.perf_counter() - started,
        "calibration_used_for_model_or_threshold_selection": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
        "final_test_claim": False,
    }
    save(OUT / "summary.json", summary)
    fitm = result_metrics["fit"]
    calm = result_metrics["calibration"]
    report = [
        "# Relation-scope localizer v1", "",
        "冻结 claim 证据关系分数作为上下文，再由独立 4-BPE 窗口读出定位否定、条件、顺序、数量、来源/步骤及主体范围。所有上游 fit 学习分数和本层输出均按来源组 OOF；阈值只由 fit OOF 决定。", "",
        "| 方法 | fit OOF窗口F1 | cal窗口F1 | fit OOF整答F1 | cal整答F1 |", "|---|---:|---:|---:|---:|",
        f"| relation_scope_localizer | {fitm['windows']['f1']:.6f} | {calm['windows']['f1']:.6f} | {fitm['answers']['f1']:.6f} | {calm['answers']['f1']:.6f} |", "",
        f"cal 窗口 AUROC/AP：{calm['windows']['auroc']:.6f}/{calm['windows']['average_precision']:.6f}；TP/FP/FN={calm['windows']['tp']}/{calm['windows']['fp']}/{calm['windows']['fn']}。", "",
        "这是独立开发候选；正式基线、原4-BPE标签和 official test 均未改。词面槽位只能近似作用域，不能代替真正的谓词—论元解析。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("RELATION_SCOPE_COMPLETE", calm["windows"]["f1"], calm["answers"]["f1"], flush=True)


if __name__ == "__main__":
    run()
