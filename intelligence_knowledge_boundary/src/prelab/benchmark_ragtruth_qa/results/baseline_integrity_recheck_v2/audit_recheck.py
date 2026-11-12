"""CPU-only independent audit of formal baseline identity and common evaluation.

This script reads calibration artifacts only.  It never loads a generator,
trains a model, rewrites a baseline artifact, or opens the sealed test split.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRELAB = ROOT.parent
PROJECT = PRELAB.parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row]


def verify_file_manifest(folder: Path, manifest_name: str = "complete.json") -> dict:
    manifest = read(folder / manifest_name)
    checked = {}
    for name, expected in manifest.get("files_sha256", {}).items():
        actual = sha(folder / name)
        assert actual == expected, (folder.name, name, expected, actual)
        checked[name] = actual
    return checked


def best_threshold(labels, scores) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    candidates = [np.nextafter(s.max(), np.inf), *np.unique(s)]
    choices = []
    for threshold in candidates:
        prediction = s >= threshold
        tp = int(np.sum(prediction & (y == 1)))
        fp = int(np.sum(prediction & (y == 0)))
        fn = int(np.sum(~prediction & (y == 1)))
        tn = int(np.sum(~prediction & (y == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        choices.append((f1, precision, float(threshold), tp, fp, fn, tn, recall))
    best = max(choices)
    return {
        "threshold": best[2], "f1": best[0], "precision": best[1],
        "recall": best[7], "tp": best[3], "fp": best[4],
        "fn": best[5], "tn": best[6],
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
    }


def fitted_calls(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"fit", "partial_fit", "fit_transform"}):
            result.append({"call": node.func.attr, "line": node.lineno})
    return result


def common_calibration():
    answers = lines(ROOT / "data/answers_calibration.jsonl")
    tokens = lines(ROOT / "data/tokens_calibration.jsonl")
    windows = lines(ROOT / "data/windows_k4_calibration.jsonl")
    assert len(answers) == len(tokens) == 159
    assert len(windows) == 42_241
    assert len({row["group_id"] for row in answers}) == 154
    assert sum(row["label"] for row in answers) == 100
    assert sum(row["label"] for row in windows) == 5_984
    assert [row["response_id"] for row in answers] == [row["response_id"] for row in tokens]
    return answers, tokens, windows


def audit_lookback(answers, tokens, windows):
    baseline = ROOT / "results/lookback_official_span_v1"
    adapter = ROOT / "results/lookback_k4_evaluation_adapter_v1"
    ledger = read(baseline / "answer_order.json")
    with np.load(baseline / "span_geometry.npz", allow_pickle=False) as source:
        offsets = source["answer_window_offsets"].copy()
        starts = source["token_start"].copy()
        ends = source["token_end"].copy()
    with np.load(baseline / "scores.npz", allow_pickle=False) as source:
        native_scores = source["window_scores"].astype(np.float64)
    with np.load(adapter / "adapted_scores.npz", allow_pickle=False) as source:
        saved_windows = source["window_scores"].copy()
        saved_answers = source["answer_scores"].copy()
        assert source["window_ids"].tolist() == [row["window_id"] for row in windows]
        assert source["answer_ids"].tolist() == [row["answer_id"] for row in answers]
    assert [row["response_id"] for row in ledger[3680:]] == [row["response_id"] for row in answers]

    positions = defaultdict(list)
    for index, row in enumerate(windows):
        positions[row["response_id"]].append(index)
    rebuilt_windows = np.empty(len(windows), dtype=np.float64)
    rebuilt_answers = np.empty(len(answers), dtype=np.float64)
    for local_index, (answer, token) in enumerate(zip(answers, tokens)):
        base_index = 3680 + local_index
        left, right = map(int, offsets[base_index])
        token_count = token["token_count"]
        totals = np.zeros(token_count, dtype=np.float64)
        owners = np.zeros(token_count, dtype=np.int16)
        for score, start, end in zip(native_scores[left:right], starts[left:right], ends[left:right]):
            totals[start:end] += float(score)
            owners[start:end] += 1
        assert np.all((owners >= 1) & (owners <= 8))
        token_score = totals / owners
        answer_positions = positions[answer["response_id"]]
        for window_index in answer_positions:
            rebuilt_windows[window_index] = token_score[windows[window_index]["token_indices"]].mean()
        rebuilt_answers[local_index] = rebuilt_windows[answer_positions].max()

    with (baseline / "lookback_lr.pkl").open("rb") as handle:
        model = pickle.load(handle)["model"]
    parameters = model.get_params()
    assert model.n_features_in_ == 1024
    assert parameters["C"] == 1.0 and parameters["solver"] == "lbfgs"
    assert parameters["class_weight"] is None and parameters["max_iter"] == 1000
    return {
        "mapping": "frozen 8-BPE span -> covering-span token mean -> 4-BPE mean -> answer max",
        "window_max_abs_error": float(np.max(np.abs(rebuilt_windows - saved_windows))),
        "answer_max_abs_error": float(np.max(np.abs(rebuilt_answers - saved_answers))),
        "native_model": {
            "class": type(model).__module__ + "." + type(model).__name__,
            "features": int(model.n_features_in_), "parameters": parameters,
        },
        "window": best_threshold([row["label"] for row in windows], saved_windows),
        "answer": best_threshold([row["label"] for row in answers], saved_answers),
    }


def audit_ghost(answers, windows):
    baseline = ROOT / "results/ghost_official_answer_rf_v1"
    adapter = ROOT / "results/ghost_k4_evaluation_adapter_v1"
    order = read(baseline / "answer_order.json")
    with np.load(baseline / "answer_scores.npz", allow_pickle=False) as source:
        native = source["answer_scores"].astype(np.float64)
    lookup = dict(zip([row["response_id"] for row in order], native))
    rebuilt_answers = np.asarray([lookup[row["response_id"]] for row in answers])
    rebuilt_windows = np.asarray([lookup[row["response_id"]] for row in windows])
    with np.load(adapter / "adapted_scores.npz", allow_pickle=False) as source:
        saved_answers = source["answer_scores"].copy()[-159:]
        saved_windows = source["window_scores"].copy()[-len(windows):]
        assert source["response_ids"].tolist()[-159:] == [row["response_id"] for row in answers]
        assert source["window_ids"].tolist()[-len(windows):] == [row["window_id"] for row in windows]
    with (baseline / "answer_rf.pkl").open("rb") as handle:
        model = pickle.load(handle)["model"]
    parameters = model.get_params()
    assert model.n_features_in_ == 4 and len(model.estimators_) == 750
    expected = {
        "n_estimators": 750, "max_depth": 40, "min_samples_split": 8,
        "min_samples_leaf": 2, "max_features": "sqrt",
        "class_weight": "balanced_subsample", "random_state": 42,
    }
    assert all(parameters[key] == value for key, value in expected.items())
    return {
        "mapping": "frozen whole-answer probability copied unchanged to every 4-BPE window",
        "window_exact": bool(np.array_equal(rebuilt_windows, saved_windows)),
        "answer_exact": bool(np.array_equal(rebuilt_answers, saved_answers)),
        "native_localization": False,
        "native_model": {
            "class": type(model).__module__ + "." + type(model).__name__,
            "features": int(model.n_features_in_), "parameters": parameters,
        },
        "window": best_threshold([row["label"] for row in windows], saved_windows),
        "answer": best_threshold([row["label"] for row in answers], saved_answers),
    }


def audit_lumina(answers, tokens, windows):
    feature_folder = ROOT / "results/lumina_qa_features_v1/features"
    feature_manifest = read(ROOT / "results/lumina_qa_features_v1/feature_manifest.json")
    score_path = ROOT / "results/lumina_qa_scoring_v1/lumina_scores.npz"
    with np.load(score_path, allow_pickle=False) as source:
        saved_windows = source["window_scores"].copy()[-len(windows):]
        saved_answers = source["answermax_scores"].copy()[-159:]

    positions = defaultdict(list)
    for index, row in enumerate(windows):
        positions[row["response_id"]].append(index)
    rebuilt_windows = np.empty(len(windows), dtype=np.float64)
    rebuilt_answers = np.empty(len(answers), dtype=np.float64)
    cal_feature_hashes = 0
    for local_index, (answer, token) in enumerate(zip(answers, tokens)):
        global_index = 3680 + local_index
        path = feature_folder / f"{global_index:05d}.npz"
        entry = feature_manifest["entries"][global_index]
        assert entry["response_id"] == answer["response_id"] and sha(path) == entry["npz_sha256"]
        cal_feature_hashes += 1
        with np.load(path, allow_pickle=False) as source:
            values = source["lumina_features"]
            assert np.array_equal(values[:, 2], .5 * values[:, 0] - .5 * values[:, 1])
            token_score = values[:, 2].astype(np.float64)
        assert len(token_score) == token["token_count"]
        answer_positions = positions[answer["response_id"]]
        for window_index in answer_positions:
            rebuilt_windows[window_index] = token_score[windows[window_index]["token_indices"]].mean()
        rebuilt_answers[local_index] = rebuilt_windows[answer_positions].max()
    return {
        "mapping": "fixed token score 0.5*IPR-0.5*MMD^2 -> 4-BPE mean -> answer max",
        "new_fits": 0,
        "calibration_feature_files_hash_checked": cal_feature_hashes,
        "window_max_abs_error": float(np.max(np.abs(rebuilt_windows - saved_windows))),
        "answer_max_abs_error": float(np.max(np.abs(rebuilt_answers - saved_answers))),
        "window": best_threshold([row["label"] for row in windows], saved_windows),
        "answer": best_threshold([row["label"] for row in answers], saved_answers),
    }


def main():
    implementations = {
        "lookback_runner": (ROOT / "src/run_lookback_official_span_v1.py", "a4d1a7feb10974a5707749b9da6a3781882e6653b18580f92c43f01227d7d99b"),
        "lookback_extractor": (ROOT / "src/feature_lookback_controls_v2.py", "b49629aa5370444f74afe9e313b01330bf31e6210156e8c0069088aabb84426f"),
        "lookback_author_step03": (ROOT / "results/lookback_official_span_v1/author_step03_lookback_lens.py", "8039386cc1d43eec1752d88184e62ab1969532b2b05c8ea2e5eb1a3084915c93"),
        "lookback_adapter": (ROOT / "src/run_lookback_k4_evaluation_adapter_v1.py", "b726eaf9b015ff672d0588606877c9fc28090b770ae3b0a3094a0bb106603e8c"),
        "lookback_verifier": (ROOT / "src/verify_lookback_k4_evaluation_adapter_v1.py", "991a6f76d8f480b86ba0c96e79776c4b86f93bf0f8fb835959e7ed517882cbf7"),
        "ghost_runner": (ROOT / "src/run_ghost_official_answer_rf_v1.py", "e2cb7bfc6466392b83c1e285230c56a1944f473a2e90c7c5711e6b5e5019572a"),
        "ghost_geometry": (ROOT / "src/ghost_geometry.py", "2d3976a0c22c122e4ea95c2b68b9122d984b1dca8b4bbd516ce476553d98b17e"),
        "ghost_extractor": (ROOT / "src/run_ghost_feature_extraction_v1.py", "55898518b3d0f7fc9f9bf068e09c7631532bc0785d03d200339cd1b1e3bd6a57"),
        "ghost_adapter": (ROOT / "src/run_ghost_k4_evaluation_adapter_v1.py", "a1eb4fd383027b0ecf34ef69a0eddddcbf9cb14d3d8bcab9f624c1c28c74bfdf"),
        "ghost_verifier": (ROOT / "src/verify_ghost_k4_evaluation_adapter_v1.py", "9f63751499c602824f1830cd6ad011be7e3e2c2be8a0454c8301d781dd78f190"),
        "lumina_extractor": (ROOT / "src/run_lumina_qa_extraction_v1.py", "b7038aefefe82de948955d25d4bd616805aa104e451413887f28ee324a03d76d"),
        "lumina_adapter": (ROOT / "src/score_lumina_qa_v1.py", "e5fcdb74d3c07290330235aa8928ac7984dcaf9b8c3b4d7bc6ce881e0a36f2b9"),
        "lumina_native_aggregation": (ROOT / "src/score_lumina_official_answer_v1.py", "891ec560d6be46e569f4e4a58679f0011959b3e58ad2c6c304dae9f4e8ad8cf5"),
        "lumina_formula": (ROOT / "src/lumina_qa_signals.py", "15910f994e66ebc88bd5f826c9081b292fd13c4259ce35bd7214d17987983626"),
        "lumina_memory_port": (PRELAB / "round7_evidence_grounding/src/lumina7.py", "ea1d1e45e22a510450a4428335a5b3f3366d09459f41db40aaf83ea98278aab8"),
        "lumina_author_source": (PRELAB / "round7_evidence_grounding/references/lumina_official.py", "1b5ecd8c08982b40dae59f299e580219074ee83673768feb53cf15612b7d8a5f"),
    }
    ref_root = ROOT / "third_party/reference/refchecker_1df1b25"
    refchecker = {
        "extractor": (ref_root / "refchecker/extractor/llm_extractor.py", "33df974bbf575e914f43b690b414d2c877ed39c98551a947902663a5fde78aff"),
        "nli": (ref_root / "refchecker/checker/nli_checker.py", "7c676393d1a21d41a096b3c1aa504eebf87104c7cfc19ca128222b1013601fae"),
        "repc": (ref_root / "refchecker/checker/repc/repc_checker.py", "563d2dfbdd2f1005516f0cebba23387469ccf76031142530b90e6c8998c69630"),
        "localizer": (ref_root / "refchecker/localizer/embed_localizer.py", "aefb1180969bae072f1c0dbc63ddbeb03e56f909732eaf34fbab774caeb96fb3"),
        "feasibility": (ROOT / "research/refchecker_baseline_feasibility_r32_v1/FEASIBILITY_CHECKLIST.json", "8129f42b54536411f172d6237e22d205503da4460887fe0f8a08083b0a51da90"),
    }
    hashes = {}
    for name, (path, expected) in {**implementations, **{f"refchecker_{k}": v for k, v in refchecker.items()}}.items():
        actual = sha(path)
        assert actual == expected, (name, expected, actual)
        hashes[name] = {"path": str(path.relative_to(PROJECT)), "sha256": actual, "match": True}

    artifact_sets = {}
    for name in (
        "lookback_official_span_v1", "lookback_k4_evaluation_adapter_v1",
        "ghost_official_answer_rf_v1", "ghost_k4_evaluation_adapter_v1",
        "lumina_qa_scoring_v1", "lumina_official_answer_v1",
    ):
        folder = ROOT / "results" / name
        artifact_sets[name] = {"files_checked": len(verify_file_manifest(folder)), "passed": True}
    artifact_sets["lookback_official_preparation"] = {
        "files_checked": len(verify_file_manifest(ROOT / "results/lookback_official_span_v1", "preparation_complete.json")), "passed": True}
    artifact_sets["ghost_official_preparation"] = {
        "files_checked": len(verify_file_manifest(ROOT / "results/ghost_official_answer_rf_v1", "preparation_complete.json")), "passed": True}
    artifact_sets["ghost_adapter_preparation"] = {
        "files_checked": len(verify_file_manifest(ROOT / "results/ghost_k4_evaluation_adapter_v1", "preparation_complete.json")), "passed": True}

    repo_head = subprocess.check_output(["git", "-C", str(ref_root), "rev-parse", "HEAD"], text=True).strip()
    repo_tree = subprocess.check_output(["git", "-C", str(ref_root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    repo_status = subprocess.check_output(["git", "-C", str(ref_root), "status", "--porcelain=v1"], text=True).strip()
    assert repo_head == "1df1b25cee792ba2b171302e31ca4f768bd67703"
    assert repo_tree == "9f2e810cc91abaeeceb105cf4b8027eb463580f8" and not repo_status

    adapters = {
        "lookback": ROOT / "src/run_lookback_k4_evaluation_adapter_v1.py",
        "ghost": ROOT / "src/run_ghost_k4_evaluation_adapter_v1.py",
        "lumina": ROOT / "src/score_lumina_qa_v1.py",
    }
    adapter_training_calls = {name: fitted_calls(path) for name, path in adapters.items()}
    assert all(not calls for calls in adapter_training_calls.values())

    answers, tokens, windows = common_calibration()
    scores = {
        "lookback": audit_lookback(answers, tokens, windows),
        "ghost": audit_ghost(answers, windows),
        "lumina": audit_lumina(answers, tokens, windows),
    }
    assert scores["lookback"]["window_max_abs_error"] <= 5e-16
    assert scores["lookback"]["answer_max_abs_error"] <= 5e-16
    assert scores["ghost"]["window_exact"] and scores["ghost"]["answer_exact"]
    assert scores["lumina"]["window_max_abs_error"] == 0
    assert scores["lumina"]["answer_max_abs_error"] == 0

    audit = {
        "schema_version": "baseline-integrity-recheck-v2",
        "audit_time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "CPU-only read of fit/calibration artifacts; no GPU, training, baseline mutation, or sealed test read.",
        "policy": {
            "evaluation_owned_by_project": True,
            "common_units": "same cal159, 154 groups, 42241 eligible 4-raw-BPE stride-1 windows, same answer/window gold",
            "common_threshold_rule": "separate cal F1 -> precision -> higher-threshold tie-break; score >= threshold",
            "baseline_method_owned_by_authors": "preserve declared features, architecture, training recipe, and native score",
            "allowed_mapping": "deterministic, parameter-free, label-independent output-coordinate mapping only",
        },
        "overall": {
            "completed_common_eval_pass": True,
            "no_learned_or_label_conditioned_adapter": True,
            "no_deliberate_baseline_weakening_found": True,
            "frozen_hashes_checked": len(hashes),
            "all_frozen_hashes_match": True,
            "completed_formal_baselines": ["Lookback Lens structure transfer", "LUMINA official-formula transfer", "GHOST final-structure/parameter transfer"],
            "not_scored_as_formal_baselines": ["ReDeEP", "MVA", "RefChecker"],
        },
        "common_evaluation": {
            "answers": len(answers), "groups": len({row["group_id"] for row in answers}),
            "windows": len(windows), "positive_answers": sum(row["label"] for row in answers),
            "positive_windows": sum(row["label"] for row in windows),
        },
        "adapter_training_calls": adapter_training_calls,
        "scores": scores,
        "hashes": hashes,
        "artifact_sets": artifact_sets,
        "refchecker_checkout": {"head": repo_head, "tree": repo_tree, "clean": not bool(repo_status)},
        "material_caveats": [
            {
                "method": "GHOST", "severity": "strict-author-training-failure",
                "finding": "The final published 750-tree RF parameters are used, but the paper's 50-configuration five-fold search was not rerun and seed 42 is local. Call it a final-structure/parameter transfer, not unchanged author training or official reproduction.",
            },
            {
                "method": "Lookback Lens", "severity": "transfer-limit",
                "finding": "The classifier recipe is preserved, but attention extraction is local NF4 teacher-forced replay rather than an end-to-end author trace reproduction.",
            },
            {
                "method": "LUMINA", "severity": "transfer-limit",
                "finding": "The official formula is preserved exactly; local evidence replacement, prompt convention, NF4 replay, and 4-BPE/answer-max aggregation are disclosed task adaptations.",
            },
            {
                "method": "ReDeEP/MVA/RefChecker", "severity": "not-comparable-yet",
                "finding": "No complete formal common-eval scores exist. Historical local ReDeEP variants and RefChecker-backbone ablations must remain excluded from the formal baseline ranking.",
            },
        ],
        "documentation_finding": {
            "severity": "corrected",
            "file": "CURRENT_STATUS.md",
            "finding": "The prior blanket phrase that all baselines preserve author training overstated GHOST. It was corrected to the accurate final-structure/parameter-transfer terminology.",
        },
        "official_test_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 正式基线共同评测复核 v2

本轮只读校准数据、冻结代码与结果；未用 GPU、未训练、未读取封存 test，也未改动任何基线。

## 结论

**当前评测方向符合用户要求：考卷由我们定，模型由作者方法定。** 三项已完成方法都在同一 159 答、154 材料组、42,241 个 4-BPE 窗口和同一标签/阈值/指标规则下计分。适配器只做零参数、标签无关的坐标映射；AST 检查三项均无任何训练调用。

| 正式方法身份 | 4-BPE F1 | 整答 F1 | 映射复核 |
|---|---:|---:|---|
| Lookback Lens 结构迁移 | {scores['lookback']['window']['f1']:.6f} | {scores['lookback']['answer']['f1']:.6f} | 独立重建最大误差 {scores['lookback']['window_max_abs_error']:.2e} |
| LUMINA 作者公式迁移 | {scores['lumina']['window']['f1']:.6f} | {scores['lumina']['answer']['f1']:.6f} | 159 个特征文件逐哈希，映射逐值一致 |
| GHOST 最终结构/参数迁移 | {scores['ghost']['window']['f1']:.6f} | {scores['ghost']['answer']['f1']:.6f} | 冻结整答概率逐值广播一致 |

16 个三项实现哈希和 5 个 RefChecker 冻结哈希全部匹配；六组已完成结果清单均通过。没有发现给基线删特征、减训练、增加学习型后处理、按金标选映射位置，或为了压低分数而改写输出。

## 必须保留的边界

1. **GHOST 不满足“作者训练流程完全不变”的严格说法。** 当前保留论文最终 750 树 RF 结构与参数，但没有复跑论文 50 组五折搜索，seed 42 也是本地约定；只能叫“论文最终结构/参数迁移”。
2. Lookback 保留原 8-BPE、1024 特征和默认 LR 配方，但上游是本地 NF4 强制重放；LUMINA 保留作者公式，但输入干预、骨干精度及 4-BPE/answer-max 是场景与评测适配。三者都不能写成端到端官方复现。
3. ReDeEP、MVA、RefChecker 尚无完整正式共同评测分数，必须保留 N/A；旧 `redeep_tuned` 和 RefChecker 骨干替换实验只能列本地适配/消融。
4. 已修正 `CURRENT_STATUS.md` 中对 GHOST 训练保真度过强的笼统表述；正式冻结表和结果表继续使用“迁移”命名。

机器复核结果见 `AUDIT.json`；复跑命令：`prelab/.venv/Scripts/python.exe prelab/benchmark_ragtruth_qa/results/baseline_integrity_recheck_v2/audit_recheck.py`。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("BASELINE_INTEGRITY_RECHECK_V2_PASSED", len(hashes), len(windows))


if __name__ == "__main__":
    main()
