"""CPU-only evidence-union extension of the direct semantic-window v2 scorer.

``fit`` is the only stage that may use fit labels.  It freezes the feature
variant, C, window/answer thresholds, and full-fit model.  ``evaluate`` is
single-use per output directory and is the only stage that opens calibration.
There is no official-test path in this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
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

import run_semantic_window_v2 as v2  # noqa: E402
import semantic_window_v2_features as feature_v2  # noqa: E402
import semantic_window_v3_features as feature_v3  # noqa: E402


VERSION = "semantic-window-v3-evidence-union-native634"
DEFAULT_OUT = ROOT / "results/semantic_window_v3"
PROTOCOL_DOCUMENT = ROOT / "research/semantic_window_v3/PROTOCOL.md"
V2_OUT = ROOT / "results/semantic_window_v2"
EVIDENCE_OUT = ROOT / "results/evidence_union_nli_v2"
EVIDENCE_RUNNER = HERE / "run_evidence_union_nli_v2.py"
EVIDENCE_NAMES = EVIDENCE_OUT / "aggregate_feature_names.json"

EXPECTED_WINDOWS = {"fit": 168123, "calibration": 42241}
EXPECTED_ANSWERS = {"fit": 634, "calibration": 159}
EXPECTED_CLAIMS = {"fit": 9055, "calibration": 2267}
EXPECTED_GROUPS = {"fit": 615, "calibration": 154}
COHORT = {"fit": "fit_native", "calibration": "calibration"}
V2_FEATURES = {
    "fit": V2_OUT / "fit_window_features.npz",
    "calibration": V2_OUT / "calibration_window_features.npz",
}

FOLDS = 5
THREADS = 4
SEED = 20260913
C_VALUES = (0.001, 0.01, 0.1)
VARIANTS = tuple(feature_v3.VARIANT_BLOCKS)

V2_STRICT = {"windows": 0.6539950721576909, "answers": 0.8545454545454545}
V2_F1OPT = {"windows": 0.6604922390875568, "answers": 0.8708133971291866}
LOOKBACK_REFERENCE = {"windows": 0.6008821240162587, "answers": 0.8454545454545455}
INCUMBENT_REFERENCE = {"windows": 0.6902813989031736, "answers": 0.8910891089108911}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def frozen_json(path: Path, value) -> None:
    if path.exists():
        assert read_json(path) == value, f"Frozen JSON changed: {path}"
    else:
        atomic_json(path, value)


def atomic_npz(path: Path, **arrays) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    pending.replace(path)


def atomic_pickle(path: Path, value) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(pickle.dumps(value, protocol=5))
    pending.replace(path)


def evidence_feature_names() -> tuple[str, ...]:
    payload = read_json(EVIDENCE_NAMES)
    assert payload["width"] == feature_v3.EVIDENCE_UNION_WIDTH == 44
    assert payload["labels_used"] is False
    assert payload["official_test_opened"] is False
    names = tuple(map(str, payload["names"]))
    assert len(names) == 44 and len(set(names)) == 44
    return names


def protocol() -> dict:
    evidence_names = evidence_feature_names()
    names = feature_v3.feature_names(evidence_names)
    return {
        "version": VERSION,
        "scope": {
            "fit": "Native 634-answer fit cohort only; 615 source-connected groups.",
            "calibration": "159 answers, evaluated once after the full fit freeze.",
            "expanded_fit": "Not used in this version.",
            "official_test": "No path or loader exists in this runner.",
        },
        "features": {
            "union_width": len(names),
            "backbone": {
                "width": 26,
                "source": "Exact frozen semantic-window v2 whitebox12+geometry14 columns.",
                "names": [names[index] for index in range(26)],
            },
            "evidence_union": {
                "width": 44,
                "source": "Frozen evidence_union_nli_v2 per-claim aggregate interface.",
                "claim_names": list(evidence_names),
                "window_mapping": "Lexical-token ownership weighted mean; no claim-risk max projection.",
            },
            "attribution": {
                "width": 36,
                "source": "Exact frozen semantic-window v2 four-band attribution columns.",
                "names": list(feature_v2.ATTRIBUTION_NAMES),
            },
            "labels_used_to_build_features": False,
        },
        "candidates": {
            "variants_in_order": list(VARIANTS),
            "variant_blocks": {
                name: list(feature_v3.VARIANT_BLOCKS[name]) for name in VARIANTS
            },
            "widths": {
                name: int(len(feature_v3.variant_indices(name))) for name in VARIANTS
            },
            "C": list(C_VALUES),
            "model": "Hierarchy-weighted StandardScaler plus L2 logistic regression/liblinear.",
            "target": "Native any-error four-BPE window label.",
            "backbone_only_selectable": False,
        },
        "fit": {
            "crossfit": "Five-fold GroupKFold on source-connected group_id.",
            "weights": "Group equal, answer equal within group, window equal within answer, then one binary-class balance step.",
            "thresholds": "Separate fit-OOF window and answer thresholds: F1, precision, then higher cutoff.",
            "selection": "Maximize min(window F1, answer F1), window F1, answer F1, window AP, answer AP, fewer dimensions, smaller C, earlier preregistered variant.",
            "full_fit": "Refit only the selected variant and C after fit-only selection.",
        },
        "evaluation": {
            "strict": "Apply frozen fit-OOF thresholds once to calibration.",
            "diagnostic": "Calibration F1-opt thresholds are reported only as diagnostics.",
            "references": {
                "semantic_window_v2_strict": V2_STRICT,
                "semantic_window_v2_cal_F1Opt": V2_F1OPT,
                "Lookback_common_cal_F1Opt": LOOKBACK_REFERENCE,
                "historical_incumbent_cal_selected": INCUMBENT_REFERENCE,
            },
        },
        "disabled": [
            "conflict head", "conflict-max", "add-gate",
            "claim-risk max projection", "learned layer/head selection",
        ],
        "seed": SEED,
        "cpu_threads": THREADS,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def formal_baseline_snapshot() -> dict[str, str]:
    return v2.baseline_snapshot()


def fit_source_files() -> tuple[Path, ...]:
    return (
        PROTOCOL_DOCUMENT,
        Path(v2.__file__), Path(feature_v2.__file__),
        V2_OUT / "fit_complete.json", V2_FEATURES["fit"],
        EVIDENCE_RUNNER, EVIDENCE_OUT / "protocol.json",
        EVIDENCE_OUT / "manifest_fit_native.json",
        EVIDENCE_OUT / "diagnostics_fit_native.json",
        EVIDENCE_OUT / "fit_rule_freeze.json",
        EVIDENCE_NAMES, EVIDENCE_OUT / "claim_aggregates_fit_native.npz",
    )


def source_snapshot() -> dict:
    return {
        "files_sha256": {str(path.resolve()): sha(path) for path in fit_source_files()},
        "formal_baseline_files_sha256": formal_baseline_snapshot(),
        "calibration_artifacts_opened": False,
        "calibration_labels_opened": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }


def validate_v2_fit_source() -> dict:
    fit_done = read_json(V2_OUT / "fit_complete.json")
    assert fit_done["status"] == "model_features_C_and_thresholds_frozen_before_calibration"
    assert fit_done["selected"]["variant"] == "whitebox_geometry"
    assert fit_done["selected"]["width"] == 26
    assert fit_done["calibration_labels_opened"] is False
    assert sha(V2_FEATURES["fit"]) == fit_done["artifacts_sha256"]["fit_window_features.npz"]
    assert sha(Path(v2.__file__)) == fit_done["artifacts_sha256"]["runner"]
    assert sha(Path(feature_v2.__file__)) == fit_done["artifacts_sha256"]["feature_module"]
    return {
        "selected_variant": fit_done["selected"]["variant"],
        "backbone_width": 26,
        "fit_feature_sha256": sha(V2_FEATURES["fit"]),
        "calibration_labels_opened_by_v2_fit": False,
    }


def validate_evidence_interface(partition: str) -> dict:
    cohort = COHORT[partition]
    manifest_path = EVIDENCE_OUT / f"manifest_{cohort}.json"
    diagnostic_path = EVIDENCE_OUT / f"diagnostics_{cohort}.json"
    aggregate_path = EVIDENCE_OUT / f"claim_aggregates_{cohort}.npz"
    manifest = read_json(manifest_path)
    diagnostic = read_json(diagnostic_path)
    expected_status = "fit_rule_frozen" if partition == "fit" else "frozen_rule_diagnostic"
    assert diagnostic["status"] == expected_status
    assert diagnostic["claims"] == EXPECTED_CLAIMS[partition]
    assert diagnostic["answers"] == EXPECTED_ANSWERS[partition]
    interface = diagnostic["aggregate_interface"]
    assert interface["rows"] == EXPECTED_CLAIMS[partition]
    assert interface["width"] == 44
    assert interface["features_sha256"] == sha(aggregate_path)
    assert interface["feature_names_sha256"] == sha(EVIDENCE_NAMES)
    assert manifest["missing_cache_instances"] == 0
    assert manifest["existing_cache_hit_instances"] == manifest["candidate_instances"]
    assert manifest["labels_used"] is False and diagnostic["labels_used"] is False
    assert manifest["official_test_opened"] is False
    assert diagnostic["official_test_opened"] is False
    if partition == "calibration":
        freeze_path = EVIDENCE_OUT / "fit_rule_freeze.json"
        assert diagnostic["fit_rule_freeze_sha256"] == sha(freeze_path)
        assert diagnostic["rule_selected_or_changed_on_calibration"] is False
    return {
        "cohort": cohort,
        "answers": diagnostic["answers"],
        "claims": diagnostic["claims"],
        "width": interface["width"],
        "features_sha256": interface["features_sha256"],
        "feature_names_sha256": interface["feature_names_sha256"],
        "candidate_instances": manifest["candidate_instances"],
        "existing_cache_hit_instances": manifest["existing_cache_hit_instances"],
        "missing_cache_instances": 0,
        "labels_used": False,
    }


def load_evidence_claims(partition: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    source_audit = validate_evidence_interface(partition)
    path = EVIDENCE_OUT / f"claim_aggregates_{COHORT[partition]}.npz"
    expected_keys = {
        "features", "response_index", "claim_id", "microclaim_index",
        "hypothesis_identity_sha256", "feature_names_sha256",
    }
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == expected_keys
        features = loaded["features"].copy()
        response_index = loaded["response_index"].copy()
        claim_ids = loaded["claim_id"].copy()
        microclaim_indices = loaded["microclaim_index"].copy()
        hypothesis_ids = loaded["hypothesis_identity_sha256"].copy()
        assert loaded["feature_names_sha256"].item() == sha(EVIDENCE_NAMES)
    assert features.shape == (EXPECTED_CLAIMS[partition], 44)
    assert features.dtype == np.float32 and np.isfinite(features).all()
    assert hypothesis_ids.shape == (EXPECTED_CLAIMS[partition], 32)

    response_ids, cursor = [], 0
    for answer_index, row in enumerate(rows):
        assert [int(claim["claim_id"]) for claim in row["claims"]] == list(
            range(len(row["claims"]))
        )
        for claim in row["claims"]:
            assert int(response_index[cursor]) == answer_index
            assert int(claim_ids[cursor]) == int(claim["claim_id"])
            assert int(microclaim_indices[cursor]) == int(claim["microclaim_index"])
            assert hypothesis_ids[cursor].tobytes().hex() == claim["hypothesis_sha256"]
            response_ids.append(row["response_id"])
            cursor += 1
    assert cursor == EXPECTED_CLAIMS[partition]
    source_audit.update({
        "claim_rows_identity_exact": True,
        "response_index_exact": True,
        "claim_id_exact": True,
        "microclaim_index_exact": True,
        "hypothesis_sha256_exact": True,
    })
    return features, np.asarray(response_ids), source_audit


def load_v2_window_union(partition: str, meta: dict) -> tuple[np.ndarray, dict]:
    path = V2_FEATURES[partition]
    if partition == "fit":
        source = read_json(V2_OUT / "fit_complete.json")
        expected_sha = source["artifacts_sha256"]["fit_window_features.npz"]
    else:
        source = read_json(V2_OUT / "summary.json")
        expected_sha = source["artifacts_sha256"]["calibration_window_features.npz"]
    assert sha(path) == expected_sha
    with np.load(path, allow_pickle=False) as loaded:
        union = loaded["union"].copy()
        assert np.array_equal(loaded["feature_names"], feature_v2.UNION_NAMES)
        identity = {
            name: loaded[name].astype(str).copy()
            for name in ("window_ids", "response_ids", "group_ids", "answer_ids")
        }
    expected = {
        "window_ids": np.asarray([row["window_id"] for row in meta["windows"]]),
        "response_ids": np.asarray([row["response_id"] for row in meta["windows"]]),
        "group_ids": np.asarray([row["group_id"] for row in meta["windows"]]),
        "answer_ids": np.asarray([row["answer_id"] for row in meta["windows"]]),
    }
    for name in expected:
        assert np.array_equal(identity[name], expected[name]), name
    assert union.shape == (EXPECTED_WINDOWS[partition], len(feature_v2.UNION_NAMES))
    assert len(np.unique(identity["window_ids"])) == EXPECTED_WINDOWS[partition]
    return union, {
        "partition": partition,
        "rows": len(union),
        "unique_window_ids": len(union),
        "v2_feature_sha256": expected_sha,
        "identity_exact": True,
    }


def build_partition_design(partition: str) -> tuple[np.ndarray, list[dict], dict, dict]:
    rows, _ = v2.source_v1.partition_rows_and_layouts(partition)
    meta = v2.source_v1.load_partition_meta(partition)
    assert len(rows) == EXPECTED_ANSWERS[partition]
    v2_union, backbone_audit = load_v2_window_union(partition, meta)
    claim_features, response_ids, evidence_audit = load_evidence_claims(partition, rows)
    matrix = feature_v3.compose_window_features(
        v2_union, claim_features, response_ids, rows, meta
    )
    assert matrix.shape == (EXPECTED_WINDOWS[partition], 106)
    return matrix, rows, meta, {
        "backbone": backbone_audit,
        "evidence_union": evidence_audit,
        "feature_builder_labels_used": False,
    }


def prepare(out: Path) -> None:
    assert not out.exists() or not any(out.iterdir()), f"Use a fresh output directory: {out}"
    out.mkdir(parents=True, exist_ok=True)
    frozen_json(out / "protocol.json", protocol())
    v2_source = validate_v2_fit_source()
    evidence_source = validate_evidence_interface("fit")
    rows, _ = v2.source_v1.partition_rows_and_layouts("fit")
    _, _, identity = load_evidence_claims("fit", rows)
    snapshot = source_snapshot()
    atomic_json(out / "source_snapshot.json", snapshot)
    selfcheck = feature_v3.synthetic_selfcheck()
    atomic_json(out / "CPU_SELFCHECK.json", {
        **selfcheck,
        "v2_source": v2_source,
        "evidence_source": evidence_source,
        "claim_identity": identity,
        "fit_labels_opened": False,
        "calibration_artifacts_opened": False,
        "calibration_labels_opened": False,
        "GPU_used": False,
    })
    atomic_json(out / "PREPARATION.json", {
        "status": "prepared_fit_sources_label_free_only",
        "version": VERSION,
        "protocol_sha256": sha(out / "protocol.json"),
        "protocol_document_sha256": sha(PROTOCOL_DOCUMENT),
        "source_snapshot_sha256": sha(out / "source_snapshot.json"),
        "fit_claims_identity_checked": EXPECTED_CLAIMS["fit"],
        "fit_labels_opened": False,
        "calibration_artifacts_opened": False,
        "calibration_labels_opened": False,
        "model_trained": False,
        "GPU_used": False,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    })
    print("SEMANTIC_WINDOW_V3_PREPARED", 106, flush=True)


def check_prepared(out: Path) -> None:
    prepared = read_json(out / "PREPARATION.json")
    assert prepared["status"] == "prepared_fit_sources_label_free_only"
    assert read_json(out / "protocol.json") == protocol()
    assert prepared["protocol_sha256"] == sha(out / "protocol.json")
    assert prepared["protocol_document_sha256"] == sha(PROTOCOL_DOCUMENT)
    assert prepared["source_snapshot_sha256"] == sha(out / "source_snapshot.json")
    snapshot = read_json(out / "source_snapshot.json")
    for path, expected in snapshot["files_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert formal_baseline_snapshot() == snapshot["formal_baseline_files_sha256"]


def candidate_selection_key(record: dict, prereg_index: int) -> tuple:
    windows = record["fit_OOF"]["windows"]
    answers = record["fit_OOF"]["answers"]
    return (
        min(windows["f1"], answers["f1"]),
        windows["f1"], answers["f1"],
        windows["average_precision"], answers["average_precision"],
        -record["width"], -record["C"], -prereg_index,
    )


def crossfit_grid(union: np.ndarray, meta: dict) -> tuple[np.ndarray, list[str], list[dict], list[dict]]:
    labels = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    groups = np.asarray([row["group_id"] for row in meta["windows"]])
    answer_ids = np.asarray([row["answer_id"] for row in meta["windows"]])
    indices = np.arange(len(labels))
    splits = list(GroupKFold(FOLDS).split(indices, labels, groups))
    names = [f"{variant}__C{c_value:g}" for variant in VARIANTS for c_value in C_VALUES]
    lookup = {name: index for index, name in enumerate(names)}
    oof = np.full((len(names), len(labels)), np.nan, dtype=np.float64)
    fold_log = []
    with threadpool_limits(limits=THREADS):
        for fold, (train, held) in enumerate(splits):
            assert not set(groups[train]) & set(groups[held])
            base, loss, final_mass = v2.window_hierarchy_weights(
                groups, answer_ids, labels, train
            )
            for variant in VARIANTS:
                columns = feature_v3.variant_indices(variant)
                x = union[:, columns]
                scaler = StandardScaler()
                scaler.fit(x[train], sample_weight=base[train])
                z_train, z_held = scaler.transform(x[train]), scaler.transform(x[held])
                for c_value in C_VALUES:
                    model = LogisticRegression(
                        C=c_value, solver="liblinear", penalty="l2",
                        max_iter=3000, random_state=SEED,
                    )
                    model.fit(z_train, labels[train], sample_weight=loss[train])
                    name = f"{variant}__C{c_value:g}"
                    oof[lookup[name], held] = model.predict_proba(z_held)[:, 1]
            fold_log.append({
                "fold": fold,
                "train_windows": len(train), "held_windows": len(held),
                "train_answers": len(set(answer_ids[train])),
                "held_answers": len(set(answer_ids[held])),
                "train_groups": len(set(groups[train])),
                "held_groups": len(set(groups[held])),
                "group_overlap": 0,
                "train_positive": int(labels[train].sum()),
                "held_positive": int(labels[held].sum()),
                "final_loss_mass_by_class": final_mass.tolist(),
            })
            print("SEMANTIC_WINDOW_V3_FOLD", fold + 1, FOLDS, flush=True)
    assert np.isfinite(oof).all()
    table = []
    for index, name in enumerate(names):
        variant, c_text = name.rsplit("__C", 1)
        thresholds = v2.choose_thresholds(meta, oof[index])
        table.append({
            "candidate": name,
            "variant": variant,
            "blocks": list(feature_v3.VARIANT_BLOCKS[variant]),
            "width": int(len(feature_v3.variant_indices(variant))),
            "C": float(c_text),
            "fit_OOF_thresholds": thresholds,
            "fit_OOF": v2.metric_pair(meta, oof[index], thresholds),
        })
    return oof, names, table, fold_log


def fit(out: Path) -> None:
    check_prepared(out)
    assert not (out / "fit_complete.json").exists()
    assert not (out / "summary.json").exists()
    started = time.perf_counter()
    union, rows, meta, feature_audit = build_partition_design("fit")
    upstream_oof = v2.assert_fit_upstream_source_group_oof(meta)
    whitebox = v2.assert_whitebox_pooling_from_local_signals("fit", rows, meta)
    labels = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    groups = np.asarray([row["group_id"] for row in meta["windows"]])
    answer_ids = np.asarray([row["answer_id"] for row in meta["windows"]])
    assert len(set(groups)) == EXPECTED_GROUPS["fit"]
    names = feature_v3.feature_names(evidence_feature_names())
    feature_path = out / "fit_window_features.npz"
    atomic_npz(
        feature_path,
        union=union, feature_names=np.asarray(names),
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["windows"]]),
        group_ids=groups, answer_ids=answer_ids,
    )
    oof, candidate_names, table, fold_log = crossfit_grid(union, meta)
    selected_index = max(
        range(len(table)),
        key=lambda index: candidate_selection_key(table[index], index),
    )
    selected = table[selected_index]
    columns = feature_v3.variant_indices(selected["variant"])
    active = np.arange(len(labels))
    base, loss, final_mass = v2.window_hierarchy_weights(
        groups, answer_ids, labels, active
    )
    scaler, model = v2.fit_scaled_lr(
        union[:, columns], labels, active, base, loss, selected["C"]
    )
    model_path = out / "model.pkl"
    atomic_pickle(model_path, {
        "version": VERSION,
        "variant": selected["variant"], "C": selected["C"],
        "columns": columns,
        "feature_names": [names[index] for index in columns],
        "scaler": scaler, "model": model,
    })
    scores_path = out / "fit_scores.npz"
    atomic_npz(
        scores_path,
        candidate_names=np.asarray(candidate_names), candidate_scores=oof,
        selected_scores=oof[selected_index], labels=labels,
        selected_index=np.asarray(selected_index, dtype=np.int32),
        answer_scores=v2.answer_scores(meta, oof[selected_index]),
    )
    assert formal_baseline_snapshot() == read_json(out / "source_snapshot.json")[
        "formal_baseline_files_sha256"
    ]
    artifacts = {
        "model.pkl": sha(model_path), "fit_scores.npz": sha(scores_path),
        "fit_window_features.npz": sha(feature_path),
        "protocol.json": sha(out / "protocol.json"),
        "source_snapshot.json": sha(out / "source_snapshot.json"),
        "runner": sha(Path(__file__)),
        "feature_module": sha(HERE / "semantic_window_v3_features.py"),
    }
    fit_complete = {
        "status": "model_features_C_and_thresholds_frozen_before_calibration",
        "selected": {key: selected[key] for key in (
            "candidate", "variant", "blocks", "width", "C",
            "fit_OOF_thresholds", "fit_OOF",
        )},
        "selection_rule": protocol()["fit"]["selection"],
        "candidate_table": table,
        "crossfit": fold_log,
        "full_fit": {
            "windows": len(labels), "positive_windows": int(labels.sum()),
            "answers": len(meta["answers"]), "groups": len(set(groups)),
            "final_loss_mass_by_class": final_mass.tolist(),
            "n_iter": int(model.n_iter_[0]),
        },
        "feature_audit": feature_audit,
        "upstream_OOF_audit": upstream_oof,
        "whitebox_reconstruction_audit": whitebox,
        "artifacts_sha256": artifacts,
        "fit_labels_opened": True,
        "calibration_artifacts_opened": False,
        "calibration_labels_opened": False,
        "calibration_used_for_model_feature_C_or_threshold_selection": False,
        "conflict_head_enabled": False, "add_gate_enabled": False,
        "claim_score_max_projection_used": False,
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(out / "fit_complete.json", fit_complete)
    print(
        "SEMANTIC_WINDOW_V3_FIT_FROZEN", selected["candidate"],
        selected["fit_OOF"]["windows"]["f1"],
        selected["fit_OOF"]["answers"]["f1"], flush=True,
    )


def format_metric_row(scope: str, metric: dict, threshold: float) -> str:
    return (
        f"| {scope} | {metric['n']} | {metric['tp']}/{metric['fp']}/{metric['fn']}/{metric['tn']} | "
        f"{metric['precision']:.6f} | {metric['recall']:.6f} | {metric['f1']:.6f} | "
        f"{metric['auroc']:.6f} | {metric['average_precision']:.6f} | {threshold:.9f} |"
    )


def render_report(fit_done: dict, summary: dict) -> str:
    selected = fit_done["selected"]
    strict = summary["calibration_strict_fit_thresholds"]
    diagnostic = summary["calibration_F1Opt_diagnostic"]
    fit_thresholds = selected["fit_OOF_thresholds"]
    cal_thresholds = summary["calibration_F1Opt_thresholds"]
    lines = [
        "# Semantic window v3 — evidence-union native-634 pilot", "",
        "v3 保持 v2 的 whitebox12+geometry14 主干和直接 4-BPE 窗口监督，只增加冻结的 44 维 evidence-union claim 统计；第二个变体再加入 v2 的 36 维归因统计。逐 claim 特征按窗口内真实 lexical token 归属求均值，没有 conflict-max、add-gate 或 claim 风险 max 铺开。", "",
        f"fit OOF 选择 `{selected['candidate']}`，宽度 {selected['width']}。两种变体、三个 C、两级阈值和选择规则均在读取 calibration 前冻结。", "",
        "## Fit OOF 候选", "",
        "| 候选 | 维度 | 窗口F1 | 窗口AP | 整答F1 | 整答AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in fit_done["candidate_table"]:
        wm, am = record["fit_OOF"]["windows"], record["fit_OOF"]["answers"]
        lines.append(
            f"| `{record['candidate']}` | {record['width']} | {wm['f1']:.6f} | "
            f"{wm['average_precision']:.6f} | {am['f1']:.6f} | {am['average_precision']:.6f} |"
        )
    fm = selected["fit_OOF"]
    lines += [
        "", "选择只使用 fit OOF：依次比较两级 F1 的较小值、窗口 F1、整答 F1、两级 AP、维度、C 和预注册顺序。", "",
        "## 选中模型完整指标", "",
        "| 口径 | n | TP/FP/FN/TN | P | R | F1 | AUROC | AP | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        format_metric_row("fit OOF 窗口", fm["windows"], fit_thresholds["window"]["threshold"]),
        format_metric_row("fit OOF 整答", fm["answers"], fit_thresholds["answer"]["threshold"]),
        format_metric_row("cal 严格窗口", strict["windows"], fit_thresholds["window"]["threshold"]),
        format_metric_row("cal 严格整答", strict["answers"], fit_thresholds["answer"]["threshold"]),
        format_metric_row("cal F1-opt 窗口诊断", diagnostic["windows"], cal_thresholds["window"]["threshold"]),
        format_metric_row("cal F1-opt 整答诊断", diagnostic["answers"], cal_thresholds["answer"]["threshold"]),
        "", "## 与冻结参照比较", "",
        "| 方法/口径 | 窗口F1 | 整答F1 |",
        "|---|---:|---:|",
        f"| v3 严格 fit 阈值 | {strict['windows']['f1']:.6f} | {strict['answers']['f1']:.6f} |",
        f"| v3 cal F1-opt 诊断 | {diagnostic['windows']['f1']:.6f} | {diagnostic['answers']['f1']:.6f} |",
        f"| v2 严格 fit 阈值 | {V2_STRICT['windows']:.6f} | {V2_STRICT['answers']:.6f} |",
        f"| v2 cal F1-opt 诊断 | {V2_F1OPT['windows']:.6f} | {V2_F1OPT['answers']:.6f} |",
        f"| Lookback 统一 cal F1-opt | {LOOKBACK_REFERENCE['windows']:.6f} | {LOOKBACK_REFERENCE['answers']:.6f} |",
        f"| 历史 incumbent（cal 选型） | {INCUMBENT_REFERENCE['windows']:.6f} | {INCUMBENT_REFERENCE['answers']:.6f} |",
        "",
        f"v3 严格结果相对 v2 严格为窗口 {strict['windows']['f1'] - V2_STRICT['windows']:+.6f}、整答 {strict['answers']['f1'] - V2_STRICT['answers']:+.6f}；相对 Lookback 为 {strict['windows']['f1'] - LOOKBACK_REFERENCE['windows']:+.6f}/{strict['answers']['f1'] - LOOKBACK_REFERENCE['answers']:+.6f}；相对 incumbent 为 {strict['windows']['f1'] - INCUMBENT_REFERENCE['windows']:+.6f}/{strict['answers']['f1'] - INCUMBENT_REFERENCE['answers']:+.6f}。", "",
        "## 边界与审计", "",
        "- fit 五折按 source-connected group 隔离，每个 fit 窗口恰有一个外折预测。",
        "- evidence-union 的 claim 行通过 response index、claim id、microclaim index 和 hypothesis SHA-256 四重绑定；原生 fit/cal 候选缓存均 100% 命中。",
        "- whitebox 的 fit Lookback/large 输入已从上游三折 held artifacts 逐窗重建，确认为 source-group OOF。",
        "- calibration 只在完整 fit freeze 后执行一次；cal-F1Opt 只作诊断。",
        "- 正式 baseline 前后哈希一致，official test 未打开，GPU 未使用。",
        "- 本轮仍只有 native 634 答，不能解释为与 3,680-fit 正式方法同预算比较，也不自动替换 incumbent。",
        "- calibration 已被项目反复用于开发；本报告不是独立测试结论。", "",
        "可复跑入口：`python src/run_semantic_window_v3.py run-all --output-dir results/<fresh-directory>`。",
    ]
    return "\n".join(lines) + "\n"


def evaluate(out: Path) -> None:
    check_prepared(out)
    fit_done = read_json(out / "fit_complete.json")
    assert fit_done["status"] == "model_features_C_and_thresholds_frozen_before_calibration"
    assert not (out / "summary.json").exists(), "Calibration evaluation is single-use per output directory"
    for name in (
        "model.pkl", "fit_scores.npz", "fit_window_features.npz",
        "protocol.json", "source_snapshot.json",
    ):
        assert sha(out / name) == fit_done["artifacts_sha256"][name]
    assert sha(Path(__file__)) == fit_done["artifacts_sha256"]["runner"]
    assert sha(HERE / "semantic_window_v3_features.py") == fit_done["artifacts_sha256"]["feature_module"]
    started = time.perf_counter()

    union, _, meta, feature_audit = build_partition_design("calibration")
    payload = pickle.loads((out / "model.pkl").read_bytes())
    assert payload["version"] == VERSION
    assert payload["variant"] == fit_done["selected"]["variant"]
    assert payload["C"] == fit_done["selected"]["C"]
    columns = np.asarray(payload["columns"], dtype=np.int32)
    assert np.array_equal(columns, feature_v3.variant_indices(payload["variant"]))
    scores = payload["model"].predict_proba(
        payload["scaler"].transform(union[:, columns])
    )[:, 1]
    assert scores.shape == (EXPECTED_WINDOWS["calibration"],)
    assert np.isfinite(scores).all()

    fit_thresholds = fit_done["selected"]["fit_OOF_thresholds"]
    strict = v2.metric_pair(meta, scores, fit_thresholds)
    cal_thresholds = v2.choose_thresholds(meta, scores)
    diagnostic = v2.metric_pair(meta, scores, cal_thresholds)
    names = feature_v3.feature_names(evidence_feature_names())
    feature_path = out / "calibration_window_features.npz"
    atomic_npz(
        feature_path, union=union, feature_names=np.asarray(names),
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["windows"]]),
        group_ids=np.asarray([row["group_id"] for row in meta["windows"]]),
        answer_ids=np.asarray([row["answer_id"] for row in meta["windows"]]),
    )
    scores_path = out / "calibration_scores.npz"
    atomic_npz(
        scores_path, window_scores=scores,
        answer_scores=v2.answer_scores(meta, scores),
        window_ids=np.asarray([row["window_id"] for row in meta["windows"]]),
        response_ids=np.asarray([row["response_id"] for row in meta["windows"]]),
    )
    v2_summary = read_json(V2_OUT / "summary.json")
    assert v2_summary["calibration_strict_fit_thresholds"]["windows"]["f1"] == V2_STRICT["windows"]
    assert v2_summary["calibration_strict_fit_thresholds"]["answers"]["f1"] == V2_STRICT["answers"]
    assert v2_summary["calibration_F1Opt_diagnostic"]["windows"]["f1"] == V2_F1OPT["windows"]
    assert v2_summary["calibration_F1Opt_diagnostic"]["answers"]["f1"] == V2_F1OPT["answers"]
    baseline_before = read_json(out / "source_snapshot.json")["formal_baseline_files_sha256"]
    baseline_after = formal_baseline_snapshot()
    assert baseline_before == baseline_after
    evidence_cal_files = (
        EVIDENCE_OUT / "manifest_calibration.json",
        EVIDENCE_OUT / "diagnostics_calibration.json",
        EVIDENCE_OUT / "claim_aggregates_calibration.npz",
        EVIDENCE_NAMES,
    )
    summary = {
        "status": "development_calibration_evaluated_once",
        "selected_candidate": fit_done["selected"]["candidate"],
        "fit_OOF": fit_done["selected"]["fit_OOF"],
        "fit_OOF_thresholds": fit_thresholds,
        "calibration_strict_fit_thresholds": strict,
        "calibration_F1Opt_diagnostic": diagnostic,
        "calibration_F1Opt_thresholds": cal_thresholds,
        "references": protocol()["evaluation"]["references"],
        "strict_deltas": {
            "versus_semantic_window_v2": {
                key: strict[key]["f1"] - V2_STRICT[key] for key in ("windows", "answers")
            },
            "versus_Lookback": {
                key: strict[key]["f1"] - LOOKBACK_REFERENCE[key] for key in ("windows", "answers")
            },
            "versus_historical_incumbent": {
                key: strict[key]["f1"] - INCUMBENT_REFERENCE[key] for key in ("windows", "answers")
            },
        },
        "feature_audit": feature_audit,
        "calibration_source_files_sha256": {
            str(path.resolve()): sha(path) for path in evidence_cal_files
        },
        "calibration_evaluations": 1,
        "calibration_used_for_model_feature_C_or_fit_threshold_selection": False,
        "fit_complete_sha256": sha(out / "fit_complete.json"),
        "artifacts_sha256": {
            "calibration_scores.npz": sha(scores_path),
            "calibration_window_features.npz": sha(feature_path),
        },
        "formal_baseline_files_unchanged": True,
        "incumbent_replaced": False,
        "calibration_labels_opened": True,
        "conflict_head_enabled": False, "add_gate_enabled": False,
        "claim_score_max_projection_used": False,
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(out / "summary.json", summary)
    report_path = out / "REPORT.md"
    report_path.write_text(render_report(fit_done, summary), encoding="utf-8")
    atomic_json(out / "complete.json", {
        "status": "complete_native634_development_only",
        "fit_complete_sha256": sha(out / "fit_complete.json"),
        "summary_sha256": sha(out / "summary.json"),
        "report_sha256": sha(report_path),
        "calibration_scores_sha256": sha(scores_path),
        "calibration_evaluations": 1,
        "incumbent_replaced": False,
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    })
    print(
        "SEMANTIC_WINDOW_V3_EVALUATED",
        strict["windows"]["f1"], strict["answers"]["f1"],
        diagnostic["windows"]["f1"], diagnostic["answers"]["f1"], flush=True,
    )


def status(out: Path) -> None:
    print(json.dumps({
        "output": str(out.resolve()),
        "prepared": (out / "PREPARATION.json").is_file(),
        "fit_frozen": (out / "fit_complete.json").is_file(),
        "calibration_evaluated": (out / "summary.json").is_file(),
        "complete": (out / "complete.json").is_file(),
        "GPU_used": False, "formal_baselines_modified": False,
        "official_test_opened": False,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "run-all", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(out)
    elif args.stage == "fit":
        fit(out)
    elif args.stage == "evaluate":
        evaluate(out)
    elif args.stage == "run-all":
        prepare(out); fit(out); evaluate(out)
    else:
        status(out)


if __name__ == "__main__":
    main()
