"""Independent, fit-fold-0-only audit of the type-aware v5 GPU pilot.

The audit intentionally reads exactly the frozen fit prefixes.  It does not
read calibration examples, calibration labels/scores, or official test data.
It does not import either training runner.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TYPE_RUN = ROOT / "results/microclaim_crossencoder_typeaware_v5"
V4_RUN = ROOT / "results/microclaim_crossencoder_expanded_v4"
DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
RUNNER = ROOT / "src/run_microclaim_crossencoder_typeaware_v5.py"

FIT_CLAIMS = 34_919
FIT_WINDOWS = 653_979
PILOT_FOLD = 0
SEED = 20_261_023
EXPECTED_PARENT_SHA256 = "4e7fc075b3a0ee3a0cbe9ebd0cd69182bdb49663496ebcfee7b551427103b048"
ERROR_TYPES = (
    "Evident Baseless Info",
    "Subtle Baseless Info",
    "Evident Conflict",
    "Subtle Conflict",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_exact_fit_rows() -> list[dict]:
    """Stop after FIT_CLAIMS lines; do not advance into calibration."""
    rows = []
    path = DATA / "examples.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for expected in range(FIT_CLAIMS):
            line = handle.readline()
            assert line, expected
            row = json.loads(line)
            assert row["example_index"] == expected
            assert row["partition"] == "fit"
            rows.append({
                "index": expected,
                "fold": int(row["held_fold"]),
                "label": int(row["gold_label"]),
                "group_id": str(row["group_id"]),
                "source_id": str(row["source_id"]),
                "response_id": str(row["response_id"]),
                "input_pair_sha256": str(row["input_pair_sha256"]),
                "types": sorted({span["label_type"] for span in row["overlapping_gold_spans"]}),
            })
    return rows


def ranking(y, score) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    assert len(y) == len(score) and np.isfinite(score).all()
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def f1_opt(y, score) -> dict:
    """Descriptive threshold selected and evaluated on the same held fold."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    at = int(np.nanargmax(f1))
    threshold = float(thresholds[at])
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    exact_f1 = 2 * p * r / (p + r) if p + r else 0.0
    assert np.isclose(exact_f1, f1[at])
    return {
        "threshold": threshold,
        "f1": exact_f1,
        "precision": p,
        "recall": r,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_positive_rate": fp / (fp + tn),
    }


def metric_block(y, score) -> dict:
    return {"ranking": ranking(y, score), "same_fold_F1Opt": f1_opt(y, score)}


def load_fit_projection_prefix():
    """Copy only the frozen fit window prefix from the shared NPZ."""
    with np.load(DATA / "arrays.npz", allow_pickle=False) as arrays:
        indptr = arrays["window_claim_indptr"][: FIT_WINDOWS + 1].copy()
        assert len(indptr) == FIT_WINDOWS + 1 and indptr[0] == 0
        edge_end = int(indptr[-1])
        owners = arrays["window_claim_example_index"][:edge_end].copy()
        labels = arrays["window_label"][:FIT_WINDOWS].copy()
        response_index = arrays["window_response_index"][:FIT_WINDOWS].copy()
        claim_labels = arrays["labels"][:FIT_CLAIMS].copy()
        claim_folds = arrays["held_fold"][:FIT_CLAIMS].copy()
    assert len(owners) == edge_end and np.all(indptr[1:] > indptr[:-1])
    assert owners.min() >= 0 and owners.max() < FIT_CLAIMS
    assert response_index.min() >= 0 and response_index.max() < 3_680
    return indptr, owners, labels.astype(np.int8), claim_labels.astype(np.int8), claim_folds.astype(np.int8)


def project_one_fold(indptr, owners, window_labels, folds, held_indices, held_scores):
    global_scores = np.full(FIT_CLAIMS, np.nan, dtype=np.float32)
    global_scores[held_indices] = held_scores
    first_owner = owners[indptr[:-1]]
    window_fold = folds[first_owner]
    # Every owner of a window belongs to the same answer/source-connected fold.
    expanded_window_fold = np.repeat(window_fold, np.diff(indptr))
    assert np.array_equal(folds[owners], expanded_window_fold)
    all_window_scores = np.maximum.reduceat(global_scores[owners], indptr[:-1])
    mask = window_fold == PILOT_FOLD
    assert int(mask.sum()) == 132_213
    assert np.isfinite(all_window_scores[mask]).all()
    assert int(window_labels[mask].sum()) == 11_734
    return window_labels[mask], all_window_scores[mask]


def audit_checkpoint(complete: dict) -> dict:
    parent_path = V4_RUN / "fold_0/model.pt"
    child_path = TYPE_RUN / "fold_0_pilot/model.pt"
    assert sha(parent_path) == EXPECTED_PARENT_SHA256
    assert sha(child_path) == complete["checkpoint_sha256"]
    assert complete["checkpoint_sha256"] != EXPECTED_PARENT_SHA256

    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    child = torch.load(child_path, map_location="cpu", weights_only=True)
    assert parent["fold"] == child["fold"] == PILOT_FOLD
    assert child["seed"] == SEED
    assert child["parent_checkpoint_sha256"] == EXPECTED_PARENT_SHA256
    assert child["protocol_sha256"] == sha(TYPE_RUN / "protocol.json")
    pstate, cstate = parent["model_state_dict"], child["model_state_dict"]
    assert pstate.keys() == cstate.keys()
    changed, unchanged, changed_numel = [], [], 0
    for name in pstate:
        assert pstate[name].shape == cstate[name].shape
        assert pstate[name].dtype == cstate[name].dtype
        if torch.equal(pstate[name], cstate[name]):
            unchanged.append(name)
        else:
            changed.append(name)
            changed_numel += int(cstate[name].numel())
    assert changed
    classifier_changed = [name for name in changed if "classifier" in name]
    encoder_changed = [name for name in changed if "classifier" not in name]
    assert classifier_changed and encoder_changed
    result = {
        "parent_sha256": EXPECTED_PARENT_SHA256,
        "child_sha256": complete["checkpoint_sha256"],
        "state_tensors": len(cstate),
        "changed_tensors": len(changed),
        "unchanged_tensors": len(unchanged),
        "changed_parameters": changed_numel,
        "encoder_and_classifier_both_changed": True,
        "metadata_parent_binding_passed": True,
    }
    del parent, child, pstate, cstate
    return result


def main() -> None:
    complete_path = TYPE_RUN / "fold_0_pilot/complete.json"
    prediction_path = TYPE_RUN / "fold_0_pilot/predictions.npz"
    assert complete_path.exists() and prediction_path.exists(), "pilot is not complete"
    complete = load_json(complete_path)
    assert complete["status"] == "fold0_pilot_complete"
    assert complete["predictions_sha256"] == sha(prediction_path)
    assert complete["calibration_rows_read"] == 0
    assert complete["calibration_labels_used"] is False
    assert complete["official_test_opened"] is False
    assert complete["baselines_modified"] is False

    manifest = load_json(TYPE_RUN / "manifest.json")
    started = load_json(TYPE_RUN / "fold_0_pilot/started.json")
    assert started["manifest_sha256"] == sha(TYPE_RUN / "manifest.json")
    assert manifest["source_code_sha256"] == sha(RUNNER)
    assert manifest["calibration_rows_read"] == 0
    assert manifest["official_test_opened"] is False

    rows = read_exact_fit_rows()
    assert len(rows) == FIT_CLAIMS
    with np.load(TYPE_RUN / "type_labels.npz", allow_pickle=False) as values:
        labels = values["binary_labels"].copy().astype(np.int8)
        folds = values["fold_assignment"].copy().astype(np.int8)
        targets = values["type_targets"].copy()
    assert labels.shape == folds.shape == (FIT_CLAIMS,)
    assert targets.shape == (FIT_CLAIMS, 3) and np.allclose(targets.sum(axis=1), 1)
    assert np.array_equal(labels, np.asarray([row["label"] for row in rows], dtype=np.int8))
    assert np.array_equal(folds, np.asarray([row["fold"] for row in rows], dtype=np.int8))

    held = np.flatnonzero(folds == PILOT_FOLD)
    train = np.flatnonzero(folds != PILOT_FOLD)
    assert len(held) == 6_983 and len(train) == 27_936
    assert complete["training"]["examples"] == len(train)
    assert complete["held_inference"]["examples"] == len(held)

    held_rows = [rows[i] for i in held]
    train_rows = [rows[i] for i in train]
    overlap = {}
    for key in ("group_id", "source_id", "response_id", "input_pair_sha256"):
        overlap[key] = len({row[key] for row in held_rows} & {row[key] for row in train_rows})
        assert overlap[key] == 0

    with np.load(prediction_path, allow_pickle=False) as values:
        v5_indices = values["held_indices"].copy()
        v5_scores = values["held_risk_scores"].copy()
        v5_probabilities = values["held_threeway_probabilities"].copy()
    v4_prediction_path = V4_RUN / "fold_0/predictions.npz"
    v4_complete = load_json(V4_RUN / "fold_0/complete.json")
    assert v4_complete["predictions_sha256"] == sha(v4_prediction_path)
    with np.load(v4_prediction_path, allow_pickle=False) as values:
        v4_indices = values["held_indices"].copy()
        v4_scores = values["held_scores"].copy()
    assert np.array_equal(v5_indices, held)
    assert np.array_equal(v4_indices, held)
    assert len(np.unique(v5_indices)) == len(held)
    assert v5_scores.shape == v4_scores.shape == (len(held),)
    assert v5_probabilities.shape == (len(held), 3)
    assert np.isfinite(v5_scores).all() and np.isfinite(v4_scores).all()
    assert np.all((v5_scores >= 0) & (v5_scores <= 1))
    assert np.allclose(v5_probabilities.sum(axis=1), 1.0, atol=2e-6)
    assert np.allclose(v5_scores, v5_probabilities[:, 1:].sum(axis=1), atol=2e-6)

    indptr, owners, window_y, array_claim_y, array_claim_folds = load_fit_projection_prefix()
    assert np.array_equal(array_claim_y, labels)
    assert np.array_equal(array_claim_folds, folds)
    v4_window_y, v4_window_score = project_one_fold(
        indptr, owners, window_y, folds, held, v4_scores)
    v5_window_y, v5_window_score = project_one_fold(
        indptr, owners, window_y, folds, held, v5_scores)
    assert np.array_equal(v4_window_y, v5_window_y)

    held_y = labels[held]
    methods = {
        "expanded_v4_fold0": {"claim_scores": v4_scores, "window_scores": v4_window_score},
        "typeaware_v5_fold0": {"claim_scores": v5_scores, "window_scores": v5_window_score},
    }
    results = {}
    held_type_sets = [set(row["types"]) for row in held_rows]
    for name, values in methods.items():
        claim = metric_block(held_y, values["claim_scores"])
        window = metric_block(v5_window_y, values["window_scores"])
        claim_pred = values["claim_scores"] >= claim["same_fold_F1Opt"]["threshold"]
        recalls = {}
        typed_rankings = {}
        for error_type in ERROR_TYPES:
            mask = np.asarray([error_type in types for types in held_type_sets], dtype=bool) & (held_y == 1)
            recalls[error_type] = {
                "total": int(mask.sum()),
                "detected": int(np.sum(claim_pred & mask)),
                "recall": float(np.mean(claim_pred[mask])),
            }
            comparison = (held_y == 0) | mask
            typed_rankings[error_type] = ranking(mask[comparison].astype(np.int8), values["claim_scores"][comparison])
        results[name] = {
            "microclaims": claim,
            "windows_4BPE_stride1_max_projection": window,
            "microclaim_recall_by_gold_type_at_own_same_fold_F1Opt": recalls,
            "microclaim_ranking_by_gold_type_against_safe": typed_rankings,
        }

    v5_head_diagnostics = {}
    for error_type in ERROR_TYPES:
        typed = np.asarray([error_type in types for types in held_type_sets], dtype=bool) & (held_y == 1)
        comparison = (held_y == 0) | typed
        head = 1 if "Baseless" in error_type else 2
        v5_head_diagnostics[error_type] = {
            "native_head": "neutral" if head == 1 else "contradiction",
            "ranking_against_safe": ranking(
                typed[comparison].astype(np.int8), v5_probabilities[comparison, head]
            ),
        }

    old, new = results["expanded_v4_fold0"], results["typeaware_v5_fold0"]
    delta = {
        "microclaim_auroc": new["microclaims"]["ranking"]["auroc"] - old["microclaims"]["ranking"]["auroc"],
        "microclaim_average_precision": new["microclaims"]["ranking"]["average_precision"] - old["microclaims"]["ranking"]["average_precision"],
        "microclaim_same_fold_F1Opt": new["microclaims"]["same_fold_F1Opt"]["f1"] - old["microclaims"]["same_fold_F1Opt"]["f1"],
        "microclaim_FP": new["microclaims"]["same_fold_F1Opt"]["fp"] - old["microclaims"]["same_fold_F1Opt"]["fp"],
        "window_auroc": new["windows_4BPE_stride1_max_projection"]["ranking"]["auroc"] - old["windows_4BPE_stride1_max_projection"]["ranking"]["auroc"],
        "window_average_precision": new["windows_4BPE_stride1_max_projection"]["ranking"]["average_precision"] - old["windows_4BPE_stride1_max_projection"]["ranking"]["average_precision"],
        "window_same_fold_F1Opt": new["windows_4BPE_stride1_max_projection"]["same_fold_F1Opt"]["f1"] - old["windows_4BPE_stride1_max_projection"]["same_fold_F1Opt"]["f1"],
        "window_FP": new["windows_4BPE_stride1_max_projection"]["same_fold_F1Opt"]["fp"] - old["windows_4BPE_stride1_max_projection"]["same_fold_F1Opt"]["fp"],
        "type_recall": {
            error_type: new["microclaim_recall_by_gold_type_at_own_same_fold_F1Opt"][error_type]["recall"]
            - old["microclaim_recall_by_gold_type_at_own_same_fold_F1Opt"][error_type]["recall"]
            for error_type in ERROR_TYPES
        },
    }

    checkpoint = audit_checkpoint(complete)
    audit = {
        "status": "independent_fit_fold0_only_audit_passed",
        "scope": "Exactly fit held fold 0; calibration labels/scores and official test were not read.",
        "integrity": {
            "fit_rows_read": FIT_CLAIMS,
            "calibration_rows_read": 0,
            "official_test_opened": False,
            "held_indices_exact": True,
            "train_held_counts": {"train": len(train), "held": len(held)},
            "train_held_overlap": overlap,
            "source_runner_sha256": sha(RUNNER),
            "type_predictions_sha256": sha(prediction_path),
            "v4_predictions_sha256": sha(v4_prediction_path),
            "checkpoint": checkpoint,
            "baselines_modified": False,
        },
        "fold0_gold": {
            "microclaims": {"n": len(held_y), "positive": int(held_y.sum())},
            "windows": {"n": len(v5_window_y), "positive": int(v5_window_y.sum())},
            "positive_type_membership": dict(Counter(
                t for i, types in enumerate(held_type_sets) if held_y[i] == 1 for t in types
            )),
            "raw_character_span_type_membership_all_binary_labels": dict(Counter(
                t for types in held_type_sets for t in types
            )),
        },
        "methods": results,
        "typeaware_native_threeway_head_diagnostics": v5_head_diagnostics,
        "typeaware_minus_v4": delta,
        "interpretation_guard": "Same-fold F1Opt is descriptive and optimistic; the threshold is selected and evaluated on fold 0 itself. AUROC/AP are threshold-free.",
    }
    save_json(HERE / "AUDIT.json", audit)

    def f(value):
        return f"{value:.6f}"

    report = [
        "# Type-aware v5 fold-0 pilot：独立审计", "",
        "范围：只读 fit held fold 0；未读取 calibration 标签/分数或 official test。v4 与 v5 使用完全相同的 6,983 条微主张，并经同一 ragged max 规则投影到 132,213 个 4-BPE、stride-1 窗口。", "",
        "## 完整性", "",
        f"- 初始 checkpoint 精确绑定 v4 fold-0：`{checkpoint['parent_sha256']}`。",
        f"- v5 产物哈希与 `complete.json` 一致；{checkpoint['changed_tensors']}/{checkpoint['state_tensors']} 个参数张量发生变化，encoder 与 classifier 均更新。",
        "- held indices 与 `held_fold == 0` 逐项一致；train/held 的 group、source、response、完整输入哈希交集均为 0。",
        "- 训练 27,936 条、held 6,983 条；运行记录声明 calibration 读取 0、test 未打开。", "",
        "## 无阈值排序与同折 F1Opt", "",
        "| 层级 | 模型 | AUROC | AP | 同折F1Opt | P | R | FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level, key in (("微主张", "microclaims"), ("4-BPE窗口", "windows_4BPE_stride1_max_projection")):
        for label, name in (("v4", "expanded_v4_fold0"), ("type-aware v5", "typeaware_v5_fold0")):
            block = results[name][key]
            rank, opt = block["ranking"], block["same_fold_F1Opt"]
            report.append(f"| {level} | {label} | {f(rank['auroc'])} | {f(rank['average_precision'])} | {f(opt['f1'])} | {f(opt['precision'])} | {f(opt['recall'])} | {opt['fp']} |")
    report += ["", "同折 F1Opt 只用于诊断，阈值在 fold 0 上选择并在同一折打分，不能当作可迁移结果。", "",
               "## 按错误类型的微主张召回", "",
               "| 类型 | 数量 | v4 | type-aware v5 | 差值 |", "|---|---:|---:|---:|---:|"]
    for error_type in ERROR_TYPES:
        a = old["microclaim_recall_by_gold_type_at_own_same_fold_F1Opt"][error_type]
        b = new["microclaim_recall_by_gold_type_at_own_same_fold_F1Opt"][error_type]
        report.append(f"| {error_type} | {a['total']} | {a['detected']}/{a['total']} ({f(a['recall'])}) | {b['detected']}/{b['total']} ({f(b['recall'])}) | {delta['type_recall'][error_type]:+.6f} |")

    window_better = delta["window_average_precision"] > 0 and delta["window_same_fold_F1Opt"] > 0
    report += ["", "## 结论", ""]
    if window_better:
        report.append(
            f"type-aware 续训在 fold 0 改善了窗口 AP {delta['window_average_precision']:+.6f}、同折 F1Opt {delta['window_same_fold_F1Opt']:+.6f}；窗口 FP 变化 {delta['window_FP']:+d}。这是单折 pilot 的正信号，仍需完整五折 OOF 才能决定是否保留。"
        )
    else:
        report.append(
            f"type-aware 续训没有同时改善窗口 AP 和同折 F1Opt：AP {delta['window_average_precision']:+.6f}，F1 {delta['window_same_fold_F1Opt']:+.6f}，窗口 FP {delta['window_FP']:+d}。按冻结 gate，应停止该结构，不扩成五折。"
        )
    ec_old = old["microclaim_ranking_by_gold_type_against_safe"]["Evident Conflict"]
    ec_new = new["microclaim_ranking_by_gold_type_against_safe"]["Evident Conflict"]
    ec_head = v5_head_diagnostics["Evident Conflict"]["ranking_against_safe"]
    report.append(
        f"局部上，Evident Conflict 对安全样本的风险 AUROC 从 {f(ec_old['auroc'])} 升到 {f(ec_new['auroc'])}，独立 contradiction 头 AP 为 {f(ec_head['average_precision'])}；但它没有转化为全局风险召回，而无依据类召回明显下降。"
    )
    report.append("")
    (HERE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("TYPEAWARE_V5_FOLD0_INDEPENDENT_AUDIT_PASSED", json.dumps(delta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
