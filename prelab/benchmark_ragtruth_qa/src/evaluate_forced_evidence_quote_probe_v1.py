"""Fail-closed CPU evaluator for the forced-evidence quote probe.

The real entrypoint validates a separately frozen, label-free feature bundle
before it opens any fit gold or P0 data.  Until that bundle exists, only the
synthetic self-test is executable.  No calibration/test path is defined here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
OUT = ROOT / "results/forced_evidence_quote_probe_v1"
PLAN_PATH = RESEARCH / "PLAN.json"
PROTOCOL_PATH = RESEARCH / "PROTOCOL.md"
CPU_MANIFEST_PATH = OUT / "MANIFEST.json"
INPUT_PATH = OUT / "label_free_inputs.jsonl"

# A future extractor must write and seal this exact two-file feature contract.
FEATURE_DIR = OUT / "frozen_features"
FEATURE_FREEZE_PATH = FEATURE_DIR / "FEATURE_FREEZE.json"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "FEATURE_MANIFEST.json"

# These gold-bearing fit files are referenced only inside load_real_gold_geometry,
# which requires a validated FrozenFeatureReceipt.
FIT_ANSWERS_PATH = ROOT / "fit_expansion/data/answers_fit.jsonl"
FIT_TOKENS_PATH = ROOT / "fit_expansion/data/tokens_fit.jsonl"
FIT_WINDOWS_PATH = ROOT / "fit_expansion/data/windows_k4_fit.jsonl"

SALT = "forced_evidence_quote_probe_v1"
SEED = 20_261_023
FOLDS = (0, 1, 2, 3, 4)
EXPECTED_CLAIMS = 3_776
EXPECTED_ANSWERS = 256
EXPECTED_GROUPS = 256
EXPECTED_INPUT_SHA = "ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777"
EXPECTED_PROTOCOL_SHA = "78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa"
EXPECTED_PLAN_SHA = "6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c"
EXPECTED_DIMS = {"P1": 21, "P2": 549, "P3": 549}
CONDITION_C = {"P0": 1e-4, "P1": 1e-3, "P2": 1e-3, "P3": 1e-3}
TYPE_CODES = {
    "Evident Conflict": "EC",
    "Subtle Conflict": "SC",
    "Evident Baseless Info": "EBI",
    "Subtle Baseless Info": "SBI",
}
OUTPUT_FIELDS = (
    "response_id", "source_id", "group_id", "model", "question",
    "passage_1", "passage_2", "passage_3", "microclaim_id",
    "microclaim_index", "claim_start", "claim_end", "claim_text_raw",
    "claim_prompt_text",
)
FORBIDDEN_FEATURE_KEYS = frozenset((
    "labels", "gold_label", "risk_mask", "risk_bpe_indices",
    "risk_bpe_fraction", "label_type", "answer_risk", "existing_scores",
    "original_response", "quality", "answer_generator",
))
FORBIDDEN_GOLD_PATH = re.compile(r"(?i)(calibration|official[_-]?test|test150)")
REAL_GOLD_OPEN_COUNT = 0


class ProtocolFailure(RuntimeError):
    pass


class FrozenFeaturesMissing(ProtocolFailure):
    pass


def assert_runtime_contract() -> None:
    """Fail before opening gold when the frozen scoring runtime changed."""
    if sklearn.__version__ != "1.6.1":
        raise ProtocolFailure(
            f"scikit-learn version changed: {sklearn.__version__} != 1.6.1")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_lines(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ProtocolFailure(f"invalid JSONL: {path}:{line_number}") from error


def write_json_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refuse overwrite: {path}")
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def fold_for(group_id: str) -> int:
    raw = hashlib.sha256(f"{SALT}\0fold\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % 5


def row_key_digest(rows: list[dict]) -> str:
    serial = "".join(
        f"{row['response_id']}\t{row['microclaim_id']}\t{int(row['microclaim_index'])}\n"
        for row in rows
    )
    return digest_text(serial)


def reject_feature_taint(value) -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_FEATURE_KEYS & set(value)
        if overlap:
            raise ProtocolFailure(f"gold/answer payload in feature artifact: {sorted(overlap)}")
        for child in value.values():
            reject_feature_taint(child)
    elif isinstance(value, list):
        for child in value:
            reject_feature_taint(child)


def assert_safe_gold_path(path: Path) -> None:
    if FORBIDDEN_GOLD_PATH.search(str(path)):
        raise ProtocolFailure(f"forbidden non-fit path: {path}")


def load_clean_rows() -> list[dict]:
    if sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA or sha(PLAN_PATH) != EXPECTED_PLAN_SHA:
        raise ProtocolFailure("protocol or plan changed")
    cpu_manifest = read_json(CPU_MANIFEST_PATH)
    if cpu_manifest.get("sole_GPU_input") != INPUT_PATH.name:
        raise ProtocolFailure("CPU manifest does not name the sole clean input")
    if cpu_manifest["files_sha256"].get(INPUT_PATH.name) != EXPECTED_INPUT_SHA:
        raise ProtocolFailure("CPU manifest input lock changed")
    if sha(INPUT_PATH) != EXPECTED_INPUT_SHA:
        raise ProtocolFailure("clean input hash mismatch")
    rows = list(json_lines(INPUT_PATH))
    if len(rows) != EXPECTED_CLAIMS:
        raise ProtocolFailure("clean input row count mismatch")
    for row in rows:
        if tuple(row) != OUTPUT_FIELDS:
            raise ProtocolFailure("clean input schema/order mismatch")
        reject_feature_taint(row)
        if row["model"] != "llama-2-7b-chat":
            raise ProtocolFailure("non-native answer in cohort")
    if len({row["response_id"] for row in rows}) != EXPECTED_ANSWERS:
        raise ProtocolFailure("answer count mismatch")
    if len({row["group_id"] for row in rows}) != EXPECTED_GROUPS:
        raise ProtocolFailure("group count mismatch")
    return rows


@dataclass(frozen=True)
class FrozenFeatureReceipt:
    feature_dir: Path
    freeze_sha256: str
    manifest_sha256: str
    array_sha256: str
    row_keys_sha256: str
    arrays: dict[str, np.ndarray]


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_frozen_features(feature_dir: Path = FEATURE_DIR) -> tuple[FrozenFeatureReceipt, list[dict]]:
    """Validate the full feature freeze before any caller may open fit gold."""
    feature_dir = Path(feature_dir)
    freeze_path = feature_dir / FEATURE_FREEZE_PATH.name
    manifest_path = feature_dir / FEATURE_MANIFEST_PATH.name
    if not freeze_path.is_file() or not manifest_path.is_file():
        raise FrozenFeaturesMissing("frozen feature manifest is absent; fit gold remains closed")
    rows = load_clean_rows()
    freeze = read_json(freeze_path)
    expected_freeze_keys = {
        "schema_version", "status", "feature_manifest", "feature_manifest_sha256",
        "label_free_inputs_sha256", "protocol_sha256", "plan_sha256", "rows",
        "labels_used", "gold_read", "calibration_read", "official_test_read",
    }
    if set(freeze) != expected_freeze_keys:
        raise ProtocolFailure("feature freeze schema mismatch")
    if freeze["status"] != "complete_frozen_before_gold":
        raise ProtocolFailure("feature freeze is incomplete")
    if any(bool(freeze[key]) for key in
           ("labels_used", "gold_read", "calibration_read", "official_test_read")):
        raise ProtocolFailure("tainted feature freeze attestation")
    if (freeze["label_free_inputs_sha256"], freeze["protocol_sha256"],
            freeze["plan_sha256"], int(freeze["rows"])) != (
            EXPECTED_INPUT_SHA, EXPECTED_PROTOCOL_SHA, EXPECTED_PLAN_SHA, EXPECTED_CLAIMS):
        raise ProtocolFailure("feature freeze source lock mismatch")
    declared_manifest = (feature_dir / str(freeze["feature_manifest"])).resolve()
    if declared_manifest != manifest_path.resolve() or not _inside(declared_manifest, feature_dir):
        raise ProtocolFailure("feature manifest path escaped frozen feature directory")
    if sha(manifest_path) != freeze["feature_manifest_sha256"]:
        raise ProtocolFailure("feature manifest hash mismatch")

    manifest = read_json(manifest_path)
    reject_feature_taint(manifest)
    expected_manifest_keys = {
        "schema_version", "status", "rows", "row_keys_sha256",
        "label_free_inputs_sha256", "protocol_sha256", "plan_sha256",
        "arrays", "labels_used", "gold_read", "calibration_read",
        "official_test_read",
    }
    if set(manifest) != expected_manifest_keys:
        raise ProtocolFailure("feature manifest schema mismatch")
    if manifest["status"] != "complete_label_free_features":
        raise ProtocolFailure("feature manifest incomplete")
    if any(bool(manifest[key]) for key in
           ("labels_used", "gold_read", "calibration_read", "official_test_read")):
        raise ProtocolFailure("tainted feature manifest attestation")
    keys_digest = row_key_digest(rows)
    if (int(manifest["rows"]), manifest["row_keys_sha256"],
            manifest["label_free_inputs_sha256"], manifest["protocol_sha256"],
            manifest["plan_sha256"]) != (
            EXPECTED_CLAIMS, keys_digest, EXPECTED_INPUT_SHA,
            EXPECTED_PROTOCOL_SHA, EXPECTED_PLAN_SHA):
        raise ProtocolFailure("feature row/source identity mismatch")
    spec = manifest["arrays"]
    if set(spec) != {"path", "sha256", "keys", "dtypes", "shapes"}:
        raise ProtocolFailure("feature array spec mismatch")
    array_path = (feature_dir / str(spec["path"])).resolve()
    if not _inside(array_path, feature_dir) or not array_path.is_file():
        raise ProtocolFailure("feature array path invalid")
    if sha(array_path) != spec["sha256"]:
        raise ProtocolFailure("feature array hash mismatch")
    if list(spec["keys"]) != ["P1", "P2", "P3"]:
        raise ProtocolFailure("feature array key order mismatch")
    expected_shapes = {name: [EXPECTED_CLAIMS, width]
                       for name, width in EXPECTED_DIMS.items()}
    if spec["shapes"] != expected_shapes or spec["dtypes"] != {
            name: "float32" for name in EXPECTED_DIMS}:
        raise ProtocolFailure("feature array declared shape/dtype mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        if archive.files != ["P1", "P2", "P3"]:
            raise ProtocolFailure("feature NPZ key order mismatch")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, width in EXPECTED_DIMS.items():
        value = arrays[name]
        if value.shape != (EXPECTED_CLAIMS, width) or value.dtype != np.float32:
            raise ProtocolFailure(f"{name} actual shape/dtype mismatch")
        if not np.isfinite(value).all():
            raise ProtocolFailure(f"{name} contains non-finite values")
    if not np.array_equal(arrays["P1"], arrays["P2"][:, :21]):
        raise ProtocolFailure("P2 does not preserve the exact P1 surface block")
    receipt = FrozenFeatureReceipt(
        feature_dir=feature_dir.resolve(), freeze_sha256=sha(freeze_path),
        manifest_sha256=sha(manifest_path), array_sha256=sha(array_path),
        row_keys_sha256=keys_digest, arrays=arrays,
    )
    return receipt, rows


@dataclass
class GoldGeometry:
    claim_y: np.ndarray
    claim_response: np.ndarray
    claim_folds: np.ndarray
    window_y: np.ndarray
    window_response: np.ndarray
    window_folds: np.ndarray
    window_to_claims: list[np.ndarray]
    answer_ids: np.ndarray
    answer_y: np.ndarray
    answer_folds: np.ndarray
    answer_to_windows: list[np.ndarray]
    window_type_masks: dict[str, np.ndarray]
    answer_type_masks: dict[str, np.ndarray]
    p0_features: np.ndarray
    source_hashes: dict[str, str]


def _locked_tokens_hash(plan: dict) -> str:
    lock = plan["input_locks"]["original_probe_protocol"]
    path = ROOT / lock["path"].replace("prelab/benchmark_ragtruth_qa/", "")
    if sha(path) != lock["sha256"]:
        raise ProtocolFailure("original probe protocol changed")
    protocol = read_json(path)
    hits = [value for key, value in protocol["files_sha256"].items()
            if key.replace("\\", "/").endswith(
                "/prelab/benchmark_ragtruth_qa/fit_expansion/data/tokens_fit.jsonl")]
    if len(hits) != 1:
        raise ProtocolFailure("tokens_fit lineage not unique")
    return hits[0]


def load_real_gold_geometry(receipt: FrozenFeatureReceipt,
                            clean_rows: list[dict]) -> GoldGeometry:
    """Open fit gold/P0 only after a cryptographically validated receipt."""
    global REAL_GOLD_OPEN_COUNT
    if not isinstance(receipt, FrozenFeatureReceipt):
        raise ProtocolFailure("validated feature receipt required before gold")
    if receipt.row_keys_sha256 != row_key_digest(clean_rows):
        raise ProtocolFailure("receipt no longer matches clean rows")
    REAL_GOLD_OPEN_COUNT += 1
    plan = read_json(PLAN_PATH)
    p0 = plan["input_locks"]["P0_raw_lineage"]
    paths = {
        "preparation_manifest": ROOT / p0["preparation_manifest"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "completed_fit_manifest": ROOT / p0["completed_fit_manifest"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "raw_window_matrix": ROOT / p0["raw_window_matrix"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "token_index": ROOT / p0["token_index"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "fit_window_index": ROOT / p0["fit_window_index"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "fit_answer_index": ROOT / p0["fit_answer_index"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
        "producer": ROOT / p0["producer"]["path"].replace(
            "prelab/benchmark_ragtruth_qa/", ""),
    }
    for path in (*paths.values(), FIT_TOKENS_PATH):
        assert_safe_gold_path(path)
    expected = {name: p0[name]["sha256"] for name in paths}
    expected["tokens_fit"] = _locked_tokens_hash(plan)
    for name, path in {**paths, "tokens_fit": FIT_TOKENS_PATH}.items():
        if sha(path) != expected[name]:
            raise ProtocolFailure(f"P0/gold lineage changed: {name}")

    selected_ids = {row["response_id"] for row in clean_rows}
    by_clean = defaultdict(list)
    group_by_response = {}
    for claim_index, row in enumerate(clean_rows):
        by_clean[row["response_id"]].append((claim_index, row))
        old = group_by_response.setdefault(row["response_id"], row["group_id"])
        if old != row["group_id"]:
            raise ProtocolFailure("one response maps to multiple groups")
    answers = {}
    answer_rows_seen = 0
    for row in json_lines(FIT_ANSWERS_PATH):
        answer_rows_seen += 1
        if row["response_id"] in selected_ids:
            answers[row["response_id"]] = row
    if answer_rows_seen != int(p0["fit_answer_index"]["rows"]) or len(answers) != EXPECTED_ANSWERS:
        raise ProtocolFailure("fit answer geometry mismatch")
    tokens = {row["response_id"]: row for row in json_lines(FIT_TOKENS_PATH)
              if row["response_id"] in selected_ids}
    if len(tokens) != EXPECTED_ANSWERS:
        raise ProtocolFailure("selected token gold missing")
    windows = []
    window_global_indices = []
    total_windows = 0
    for global_index, row in enumerate(json_lines(FIT_WINDOWS_PATH)):
        total_windows += 1
        if row["response_id"] in selected_ids:
            windows.append(row); window_global_indices.append(global_index)
    if total_windows != int(p0["fit_window_index"]["rows"]):
        raise ProtocolFailure("fit window count mismatch")

    claim_y = np.zeros(EXPECTED_CLAIMS, dtype=np.int8)
    window_to_claims = []
    window_y = np.zeros(len(windows), dtype=np.int8)
    window_response = np.asarray([row["response_id"] for row in windows], dtype=str)
    window_folds = np.asarray([fold_for(group_by_response[rid])
                               for rid in window_response], dtype=np.int8)
    window_type_masks = {code: np.zeros(len(windows), dtype=bool)
                         for code in TYPE_CODES.values()}
    answer_type_sets = defaultdict(set)
    owner_by_response = {}
    span_types_by_response = {}
    for response_id in sorted(selected_ids):
        token = tokens[response_id]
        text = token["original_response"]
        local_claims = by_clean[response_id]
        clean_identity = {
            (claim["response_id"], claim["source_id"], claim["group_id"])
            for _claim_index, claim in local_claims
        }
        if clean_identity != {(response_id, answers[response_id]["source_id"],
                               answers[response_id]["group_id"])}:
            raise ProtocolFailure("clean/answer response-source-group identity mismatch")
        if (token["response_id"], token["source_id"], token["group_id"]) != (
                response_id, answers[response_id]["source_id"],
                answers[response_id]["group_id"]):
            raise ProtocolFailure("token/answer response-source-group identity mismatch")
        for _claim_index, claim in local_claims:
            left, right = int(claim["claim_start"]), int(claim["claim_end"])
            if not (0 <= left < right <= len(text)):
                raise ProtocolFailure("microclaim character bounds invalid")
            if text[left:right] != claim["claim_text_raw"]:
                raise ProtocolFailure("microclaim text/coordinate binding mismatch")
        offsets = [tuple(map(int, item)) for item in token["response_token_offsets"]]
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        risk = np.asarray(token["risk_mask"], dtype=bool)
        if not (len(offsets) == len(lexical) == len(risk) == int(token["token_count"])):
            raise ProtocolFailure("token geometry length mismatch")
        owners = []
        inverse = defaultdict(list)
        for token_index, (left, right) in enumerate(offsets):
            chars = [position for position in range(left, right)
                     if 0 <= position < len(text) and text[position].isalnum()]
            if bool(chars) != bool(lexical[token_index]):
                raise ProtocolFailure("lexical mask disagrees with text offsets")
            active = [claim_index for claim_index, claim in local_claims
                      if any(int(claim["claim_start"]) <= position <
                             int(claim["claim_end"]) for position in chars)]
            owners.append(active)
            for claim_index in active:
                inverse[claim_index].append(token_index)
        for claim_index, _claim in local_claims:
            indices = inverse[claim_index]
            if not indices:
                raise ProtocolFailure("eligible microclaim owns no lexical BPE")
            claim_y[claim_index] = int(bool(risk[indices].any()))
        span_types = defaultdict(set)
        labels = token["original_labels"]
        for mapping in token["span_token_mapping"]:
            span_index = int(mapping["span_index"])
            label_type = labels[span_index]["label_type"]
            if label_type not in TYPE_CODES:
                raise ProtocolFailure(f"unknown label type: {label_type}")
            code = TYPE_CODES[label_type]
            answer_type_sets[response_id].add(code)
            for token_index in mapping["risk_token_indices"]:
                span_types[int(token_index)].add(code)
        owner_by_response[response_id] = owners
        span_types_by_response[response_id] = span_types
        if int(answers[response_id]["label"]) != int(token["answer_risk"]):
            raise ProtocolFailure("answer label mismatch")

    for window_index, row in enumerate(windows):
        response_id = row["response_id"]
        token = tokens[response_id]
        if (row["response_id"], row["source_id"], row["group_id"]) != (
                token["response_id"], token["source_id"], token["group_id"]):
            raise ProtocolFailure("window/token response-source-group identity mismatch")
        indices = list(map(int, row["token_indices"]))
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        risk = np.asarray(token["risk_mask"], dtype=bool)
        if not indices or not lexical[indices].any():
            raise ProtocolFailure("noneligible row in fit window index")
        expected_label = int(bool(risk[indices].any()))
        if int(row["label"]) != expected_label:
            raise ProtocolFailure("window label mismatch")
        window_y[window_index] = expected_label
        active = sorted({claim_index for token_index in indices
                         for claim_index in owner_by_response[response_id][token_index]})
        window_to_claims.append(np.asarray(active, dtype=np.int32))
        for token_index in indices:
            for code in span_types_by_response[response_id][token_index]:
                window_type_masks[code][window_index] = True

    answer_ids = np.asarray(sorted(selected_ids), dtype=str)
    answer_y = np.asarray([int(answers[rid]["label"]) for rid in answer_ids], dtype=np.int8)
    answer_folds = np.asarray([fold_for(group_by_response[rid]) for rid in answer_ids], dtype=np.int8)
    answer_to_windows = [np.flatnonzero(window_response == rid).astype(np.int32)
                         for rid in answer_ids]
    if any(len(indices) == 0 for indices in answer_to_windows):
        raise ProtocolFailure("selected answer has no eligible window")
    answer_type_masks = {
        code: np.asarray([code in answer_type_sets[rid] for rid in answer_ids], dtype=bool)
        for code in TYPE_CODES.values()
    }
    claim_response = np.asarray([row["response_id"] for row in clean_rows], dtype=str)
    claim_folds = np.asarray([fold_for(row["group_id"]) for row in clean_rows], dtype=np.int8)
    raw_spec = p0["raw_window_matrix"]
    raw = np.load(paths["raw_window_matrix"], mmap_mode="r")
    if raw.shape != tuple(raw_spec["shape"]) or str(raw.dtype) != raw_spec["dtype"]:
        raise ProtocolFailure("P0 raw matrix shape/dtype mismatch")
    p0_features = np.asarray(raw[np.asarray(window_global_indices, dtype=np.int64)],
                             dtype=np.float32)
    if not np.isfinite(p0_features).all():
        raise ProtocolFailure("nonfinite P0 raw features")
    return GoldGeometry(
        claim_y=claim_y, claim_response=claim_response, claim_folds=claim_folds,
        window_y=window_y, window_response=window_response,
        window_folds=window_folds, window_to_claims=window_to_claims,
        answer_ids=answer_ids, answer_y=answer_y, answer_folds=answer_folds,
        answer_to_windows=answer_to_windows,
        window_type_masks=window_type_masks, answer_type_masks=answer_type_masks,
        p0_features=p0_features, source_hashes=expected,
    )


@dataclass
class ConditionData:
    name: str
    X: np.ndarray
    y: np.ndarray
    response_ids: np.ndarray
    folds: np.ndarray
    window_to_units: list[np.ndarray]


def answer_normalized_base_weights(response_ids: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(response_ids), dtype=np.float64)
    for response_id in np.unique(response_ids):
        indices = np.flatnonzero(response_ids == response_id)
        if len(indices) == 0:
            raise ProtocolFailure("empty answer in training subset")
        weights[indices] = 1.0 / len(indices)
    return weights


def fit_predict(condition: ConditionData, train: np.ndarray, valid: np.ndarray,
                c_value: float) -> tuple[np.ndarray, dict]:
    if np.any(train & valid) or not train.any() or not valid.any():
        raise ProtocolFailure("invalid train/valid masks")
    X_train = np.asarray(condition.X[train], dtype=np.float64)
    X_valid = np.asarray(condition.X[valid], dtype=np.float64)
    y_train = np.asarray(condition.y[train], dtype=np.int8)
    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise ProtocolFailure("nonfinite classifier input")
    if set(np.unique(y_train)) != {0, 1}:
        raise ProtocolFailure("single-class training subset")
    response_train = condition.response_ids[train]
    base = answer_normalized_base_weights(response_train)
    scaler = StandardScaler()
    scaler.fit(X_train, sample_weight=base)
    masses = np.bincount(y_train, weights=base, minlength=2).astype(np.float64)
    if np.any(masses <= 0):
        raise ProtocolFailure("zero weighted class mass")
    factors = masses.sum() / (2.0 * masses)
    sample_weight = base * factors[y_train]
    model = LogisticRegression(
        penalty="l2", C=float(c_value), solver="liblinear", max_iter=2000,
        random_state=SEED, class_weight=None,
    )
    model.fit(scaler.transform(X_train), y_train, sample_weight=sample_weight)
    if int(model.n_iter_[0]) >= 2000:
        raise ProtocolFailure("liblinear reached max_iter")
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    scores = model.predict_proba(scaler.transform(X_valid))[:, positive_column]
    if not np.isfinite(scores).all():
        raise ProtocolFailure("nonfinite probabilities")
    audit = {
        "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "train_answers": int(len(np.unique(response_train))),
        "base_weight_sum": float(base.sum()),
        "class_mass_0": float(masses[0]), "class_mass_1": float(masses[1]),
        "loss_weight_sum_0": float(sample_weight[y_train == 0].sum()),
        "loss_weight_sum_1": float(sample_weight[y_train == 1].sum()),
        "train_folds": sorted(map(int, np.unique(condition.folds[train]))),
        "valid_folds": sorted(map(int, np.unique(condition.folds[valid]))),
    }
    return scores.astype(np.float64), audit


def project_to_windows(condition: ConditionData, unit_scores: np.ndarray,
                       window_mask: np.ndarray) -> np.ndarray:
    output = np.full(len(condition.window_to_units), np.nan, dtype=np.float64)
    for window_index in np.flatnonzero(window_mask):
        indices = condition.window_to_units[window_index]
        if len(indices) == 0:
            # Frozen protocol: an eligible window uncovered by any claim has
            # the fixed, parameter-free risk score zero.
            output[window_index] = 0.0
            continue
        values = unit_scores[indices]
        if not np.isfinite(values).all():
            raise ProtocolFailure("window references missing unit score")
        output[window_index] = float(values.max())
    return output


def f1_precision(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool); prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(y & prediction)); n_pred = int(prediction.sum()); n_pos = int(y.sum())
    denominator = n_pred + n_pos
    f1 = 2.0 * tp / denominator if denominator else 0.0
    precision = tp / n_pred if n_pred else 0.0
    return float(f1), float(precision)


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(y) != len(scores) or not np.isfinite(scores).all():
        raise ProtocolFailure("invalid threshold inputs")
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
    best = None
    for threshold in candidates:
        prediction = scores >= threshold
        f1, precision = f1_precision(y, prediction)
        key = (f1, precision, float(threshold))
        if best is None or key > best[0]:
            best = (key, {"threshold": float(threshold), "F1": f1,
                          "precision": precision})
    assert best is not None
    return best[1]


def fixed_ap(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    if positives == len(y):
        return 1.0
    return float(average_precision_score(y, scores))


def aggregate_answers(window_scores: np.ndarray, geometry: GoldGeometry) -> np.ndarray:
    output = np.asarray([float(window_scores[indices].max())
                         for indices in geometry.answer_to_windows], dtype=np.float64)
    if not np.isfinite(output).all():
        raise ProtocolFailure("nonfinite answer score")
    return output


def run_nested_cv(condition: ConditionData, geometry: GoldGeometry) -> dict:
    if condition.name not in CONDITION_C:
        raise ProtocolFailure("unknown condition")
    if not (len(condition.X) == len(condition.y) == len(condition.response_ids)
            == len(condition.folds)):
        raise ProtocolFailure("condition unit arrays disagree")
    n_windows = len(geometry.window_y)
    if len(condition.window_to_units) != n_windows:
        raise ProtocolFailure("window projection length mismatch")
    unit_oof = np.full(len(condition.y), np.nan, dtype=np.float64)
    window_oof = np.full(n_windows, np.nan, dtype=np.float64)
    window_binary = np.zeros(n_windows, dtype=bool)
    thresholds = {}
    fit_audit = []
    for outer in FOLDS:
        outer_train_folds = [fold for fold in FOLDS if fold != outer]
        inner_unit = np.full(len(condition.y), np.nan, dtype=np.float64)
        for inner_valid in outer_train_folds:
            inner_train_folds = [fold for fold in outer_train_folds if fold != inner_valid]
            train = np.isin(condition.folds, inner_train_folds)
            valid = condition.folds == inner_valid
            scores, audit = fit_predict(condition, train, valid, CONDITION_C[condition.name])
            audit.update({"stage": "inner", "outer_fold": outer,
                          "inner_valid_fold": inner_valid})
            if audit["train_folds"] != inner_train_folds or audit["valid_folds"] != [inner_valid]:
                raise ProtocolFailure("inner fold leakage")
            inner_unit[valid] = scores; fit_audit.append(audit)
        inner_window_mask = np.isin(geometry.window_folds, outer_train_folds)
        inner_windows = project_to_windows(condition, inner_unit, inner_window_mask)
        threshold_row = choose_threshold(
            geometry.window_y[inner_window_mask], inner_windows[inner_window_mask])
        thresholds[str(outer)] = threshold_row

        train = np.isin(condition.folds, outer_train_folds)
        valid = condition.folds == outer
        scores, audit = fit_predict(condition, train, valid, CONDITION_C[condition.name])
        audit.update({"stage": "outer", "outer_fold": outer})
        if audit["train_folds"] != outer_train_folds or audit["valid_folds"] != [outer]:
            raise ProtocolFailure("outer fold leakage")
        unit_oof[valid] = scores; fit_audit.append(audit)
        outer_window_mask = geometry.window_folds == outer
        projected = project_to_windows(condition, unit_oof, outer_window_mask)
        window_oof[outer_window_mask] = projected[outer_window_mask]
        window_binary[outer_window_mask] = (
            projected[outer_window_mask] >= threshold_row["threshold"])
    if not np.isfinite(unit_oof).all() or not np.isfinite(window_oof).all():
        raise ProtocolFailure("OOF coverage incomplete")
    answer_oof = aggregate_answers(window_oof, geometry)
    answer_binary = np.zeros(len(geometry.answer_ids), dtype=bool)
    per_fold = {}
    for outer in FOLDS:
        answer_mask = geometry.answer_folds == outer
        window_mask = geometry.window_folds == outer
        threshold = thresholds[str(outer)]["threshold"]
        answer_binary[answer_mask] = answer_oof[answer_mask] >= threshold
        wf1, wp = f1_precision(geometry.window_y[window_mask], window_binary[window_mask])
        af1, aprec = f1_precision(geometry.answer_y[answer_mask], answer_binary[answer_mask])
        per_fold[str(outer)] = {
            "threshold": float(threshold),
            "window_AP": fixed_ap(geometry.window_y[window_mask], window_oof[window_mask]),
            "window_F1": wf1, "window_precision": wp,
            "answer_AP": fixed_ap(geometry.answer_y[answer_mask], answer_oof[answer_mask]),
            "answer_F1": af1, "answer_precision": aprec,
        }
    window_f1, window_precision = f1_precision(geometry.window_y, window_binary)
    answer_f1, answer_precision = f1_precision(geometry.answer_y, answer_binary)
    return {
        "condition": condition.name,
        "C": CONDITION_C[condition.name],
        "window_scores": window_oof,
        "window_binary": window_binary,
        "answer_scores": answer_oof,
        "answer_binary": answer_binary,
        "metrics": {
            "pooled_outer_oof_window_AP": fixed_ap(geometry.window_y, window_oof),
            "pooled_outer_oof_window_F1": window_f1,
            "pooled_outer_oof_window_precision": window_precision,
            "pooled_outer_oof_answer_AP": fixed_ap(geometry.answer_y, answer_oof),
            "pooled_outer_oof_answer_F1": answer_f1,
            "pooled_outer_oof_answer_precision": answer_precision,
        },
        "thresholds": thresholds,
        "per_fold": per_fold,
        "fit_audit": fit_audit,
    }


def make_conditions(receipt: FrozenFeatureReceipt, geometry: GoldGeometry) -> dict[str, ConditionData]:
    p0_mapping = [np.asarray([index], dtype=np.int32)
                  for index in range(len(geometry.window_y))]
    conditions = {
        "P0": ConditionData("P0", geometry.p0_features, geometry.window_y,
                            geometry.window_response, geometry.window_folds, p0_mapping),
    }
    for name in ("P1", "P2", "P3"):
        conditions[name] = ConditionData(
            name, receipt.arrays[name], geometry.claim_y,
            geometry.claim_response, geometry.claim_folds,
            geometry.window_to_claims,
        )
    return conditions


def type_diagnostics(result: dict, geometry: GoldGeometry,
                     conflict_sufficient: bool) -> dict:
    output = {}
    for code in ("EC", "SC", "EBI", "SBI"):
        wy = geometry.window_type_masks[code].astype(np.int8)
        ay = geometry.answer_type_masks[code].astype(np.int8)
        if code in ("EC", "SC") and not conflict_sufficient:
            output[code] = {"status": "N/A_insufficient_combined_conflict_support",
                            "positive_windows": int(wy.sum()),
                            "positive_answers": int(ay.sum())}
            continue
        wpred = result["window_binary"]
        apred = result["answer_binary"]
        output[code] = {
            "status": "reported",
            "positive_windows": int(wy.sum()), "positive_answers": int(ay.sum()),
            "window_recall": float(np.sum(wy.astype(bool) & wpred) / max(1, int(wy.sum()))),
            "window_AP": fixed_ap(wy, result["window_scores"]),
            "answer_recall": float(np.sum(ay.astype(bool) & apred) / max(1, int(ay.sum()))),
            "answer_AP": fixed_ap(ay, result["answer_scores"]),
        }
    return output


def compute_advance_gate(results: dict[str, dict], p3_features: np.ndarray,
                         geometry: GoldGeometry) -> dict:
    parse_valid = p3_features[:, 0]
    source_exact = p3_features[:, 1]
    stop_close = p3_features[:, 3]
    for column in (parse_valid, source_exact, stop_close):
        if not np.isin(column, (0.0, 1.0)).all():
            raise ProtocolFailure("P3 exact-rate state columns are not binary")
    exact_count = int(np.sum((parse_valid == 1) & (source_exact == 1) & (stop_close == 1)))
    exact_rate = exact_count / EXPECTED_CLAIMS
    p0, p2, p3 = (results[name] for name in ("P0", "P2", "P3"))
    positive_folds = sum(
        p3["per_fold"][str(fold)]["window_AP"] >
        p2["per_fold"][str(fold)]["window_AP"] for fold in FOLDS)
    conflict_windows = geometry.window_type_masks["EC"] | geometry.window_type_masks["SC"]
    conflict_answers = geometry.answer_type_masks["EC"] | geometry.answer_type_masks["SC"]
    checks = {
        "source_exact_rate_at_least_0.95": exact_rate >= 0.95,
        "window_AP_at_least_P0_plus_0.01": (
            p3["metrics"]["pooled_outer_oof_window_AP"] >=
            p0["metrics"]["pooled_outer_oof_window_AP"] + 0.01),
        "window_AP_at_least_P2_plus_0.01": (
            p3["metrics"]["pooled_outer_oof_window_AP"] >=
            p2["metrics"]["pooled_outer_oof_window_AP"] + 0.01),
        "P3_minus_P2_positive_in_at_least_4_folds": positive_folds >= 4,
        "answer_AP_no_more_than_0.01_below_P2": (
            p3["metrics"]["pooled_outer_oof_answer_AP"] >=
            p2["metrics"]["pooled_outer_oof_answer_AP"] - 0.01),
    }
    return {
        "exact_count": exact_count, "exact_denominator": EXPECTED_CLAIMS,
        "exact_rate": exact_rate, "P3_minus_P2_positive_window_AP_folds": positive_folds,
        "combined_conflict_positive_windows": int(conflict_windows.sum()),
        "combined_conflict_positive_answers": int(conflict_answers.sum()),
        "conflict_diagnostics_sufficient": bool(
            conflict_windows.sum() >= 50 and conflict_answers.sum() >= 10),
        "checks": checks, "advance": bool(all(checks.values())),
        "failure_action": "stop_without_retuning_on_pilot",
    }


def serializable_result(result: dict) -> dict:
    return {key: value for key, value in result.items()
            if key not in ("window_scores", "window_binary", "answer_scores", "answer_binary")}


def pairwise_fold_deltas(results: dict[str, dict]) -> dict:
    """Fixed P3-minus-comparator diagnostics; never used for fitting."""
    output = {}
    for fold in FOLDS:
        p3 = results["P3"]["per_fold"][str(fold)]
        output[str(fold)] = {}
        for comparator in ("P0", "P1", "P2"):
            other = results[comparator]["per_fold"][str(fold)]
            output[str(fold)][f"P3_minus_{comparator}"] = {
                metric: float(p3[metric] - other[metric])
                for metric in ("window_AP", "window_F1", "answer_AP", "answer_F1")
            }
    return output


def evaluate_real() -> None:
    assert_runtime_contract()
    # This call is the hard taint gate. Nothing below may run if it fails.
    receipt, clean_rows = load_frozen_features()
    geometry = load_real_gold_geometry(receipt, clean_rows)
    conditions = make_conditions(receipt, geometry)
    results = {name: run_nested_cv(condition, geometry)
               for name, condition in conditions.items()}
    gate = compute_advance_gate(results, receipt.arrays["P3"], geometry)
    diagnostics = {
        name: type_diagnostics(result, geometry, gate["conflict_diagnostics_sufficient"])
        for name, result in results.items()
    }
    diagnostics["per_fold_pairwise_deltas"] = pairwise_fold_deltas(results)
    diagnostics["P3_quote"] = {
        "source_exact_rate": gate["exact_rate"],
        "invalid_quote_rate": float(1.0 - receipt.arrays["P3"][:, 0].mean()),
    }
    output = {
        "status": "fit_only_nested_OOF_complete",
        "conditions": {name: serializable_result(value) for name, value in results.items()},
        "diagnostics": diagnostics, "advance_gate": gate,
        "feature_receipt": {
            "freeze_sha256": receipt.freeze_sha256,
            "manifest_sha256": receipt.manifest_sha256,
            "array_sha256": receipt.array_sha256,
            "row_keys_sha256": receipt.row_keys_sha256,
        },
        "gold_source_hashes": geometry.source_hashes,
        "evaluator_sha256": sha(Path(__file__)),
        "sklearn_version": sklearn.__version__,
        "calibration_read": False, "official_test_read": False,
        "fit_only": True, "GPU_used_by_evaluator": False,
    }
    write_json_new(OUT / "EVALUATION.json", output)


def synthetic_geometry(seed: int = SEED) -> tuple[GoldGeometry, dict[str, ConditionData]]:
    rng = np.random.default_rng(seed)
    answer_ids, answer_y, answer_folds = [], [], []
    claim_y, claim_response, claim_folds = [], [], []
    window_y, window_response, window_folds, window_to_claims = [], [], [], []
    answer_to_windows = []
    window_types = {code: [] for code in TYPE_CODES.values()}
    answer_types = {code: [] for code in TYPE_CODES.values()}
    for fold in FOLDS:
        for answer_local in range(4):
            response_id = f"f{fold}_a{answer_local}"
            answer_ids.append(response_id); answer_folds.append(fold)
            pattern = ((0, 0), (1, 0), (0, 1), (1, 1))[answer_local]
            claim_start = len(claim_y)
            for value in pattern:
                claim_y.append(value); claim_response.append(response_id); claim_folds.append(fold)
            local_windows = []
            for mapping in ((0,), (0, 1), (1,)):
                indices = np.asarray([claim_start + item for item in mapping], dtype=np.int32)
                local_windows.append(len(window_y)); window_to_claims.append(indices)
                label = int(any(claim_y[index] for index in indices))
                window_y.append(label); window_response.append(response_id); window_folds.append(fold)
                active_code = "EC" if label and answer_local % 2 else "EBI" if label else None
                for code in TYPE_CODES.values():
                    window_types[code].append(code == active_code)
            answer_to_windows.append(np.asarray(local_windows, dtype=np.int32))
            answer_y.append(int(any(pattern)))
            for code in TYPE_CODES.values():
                answer_types[code].append(any(window_types[code][index] for index in local_windows))
    claim_y_array = np.asarray(claim_y, dtype=np.int8)
    window_y_array = np.asarray(window_y, dtype=np.int8)
    def features(labels, width, strength):
        labels = np.asarray(labels, dtype=np.float64)
        value = rng.normal(0, 0.35, size=(len(labels), width))
        value[:, 0] += strength * (2 * labels - 1)
        return value.astype(np.float32)
    geometry = GoldGeometry(
        claim_y=claim_y_array, claim_response=np.asarray(claim_response),
        claim_folds=np.asarray(claim_folds, dtype=np.int8),
        window_y=window_y_array, window_response=np.asarray(window_response),
        window_folds=np.asarray(window_folds, dtype=np.int8),
        window_to_claims=window_to_claims, answer_ids=np.asarray(answer_ids),
        answer_y=np.asarray(answer_y, dtype=np.int8),
        answer_folds=np.asarray(answer_folds, dtype=np.int8),
        answer_to_windows=answer_to_windows,
        window_type_masks={code: np.asarray(value, dtype=bool)
                           for code, value in window_types.items()},
        answer_type_masks={code: np.asarray(value, dtype=bool)
                           for code, value in answer_types.items()},
        p0_features=features(window_y_array, 7, 0.9), source_hashes={},
    )
    p0_map = [np.asarray([index], dtype=np.int32) for index in range(len(window_y))]
    conditions = {
        "P0": ConditionData("P0", geometry.p0_features, window_y_array,
                            geometry.window_response, geometry.window_folds, p0_map),
        "P1": ConditionData("P1", features(claim_y_array, 6, 0.5), claim_y_array,
                            geometry.claim_response, geometry.claim_folds, window_to_claims),
        "P2": ConditionData("P2", features(claim_y_array, 8, 0.8), claim_y_array,
                            geometry.claim_response, geometry.claim_folds, window_to_claims),
        "P3": ConditionData("P3", features(claim_y_array, 9, 1.1), claim_y_array,
                            geometry.claim_response, geometry.claim_folds, window_to_claims),
    }
    return geometry, conditions


def synthetic_selftest() -> None:
    global REAL_GOLD_OPEN_COUNT
    assert_runtime_contract()
    before = REAL_GOLD_OPEN_COUNT
    missing_gate = False
    synthetic_missing = OUT / "__absent_evaluator_selftest_features__"
    if synthetic_missing.exists():
        raise ProtocolFailure("synthetic missing-feature sentinel unexpectedly exists")
    try:
        load_frozen_features(synthetic_missing)
    except FrozenFeaturesMissing:
        missing_gate = True
    if not missing_gate or REAL_GOLD_OPEN_COUNT != before:
        raise ProtocolFailure("missing features did not fail before gold access")

    geometry, conditions = synthetic_geometry()
    results = {name: run_nested_cv(condition, geometry)
               for name, condition in conditions.items()}
    for name, result in results.items():
        if len(result["fit_audit"]) != 25:
            raise ProtocolFailure(f"{name}: expected 20 inner and 5 outer fits")
        for row in result["fit_audit"]:
            expected_fold_count = 3 if row["stage"] == "inner" else 4
            if len(row["train_folds"]) != expected_fold_count or len(row["valid_folds"]) != 1:
                raise ProtocolFailure("synthetic nested fold audit failed")
            if not np.isclose(row["base_weight_sum"], row["train_answers"]):
                raise ProtocolFailure("per-answer base weights are not normalized")
            if not np.isclose(row["loss_weight_sum_0"], row["loss_weight_sum_1"]):
                raise ProtocolFailure("local class balancing failed")
    threshold = choose_threshold(np.asarray([1, 0, 0]), np.asarray([0.9, 0.8, 0.2]))
    if threshold != {"threshold": 0.9, "F1": 1.0, "precision": 1.0}:
        raise ProtocolFailure("threshold rule mismatch")
    if fixed_ap(np.zeros(3, dtype=int), np.asarray([0.1, 0.2, 0.3])) != 0.0:
        raise ProtocolFailure("no-positive AP rule mismatch")
    if fixed_ap(np.ones(3, dtype=int), np.asarray([0.1, 0.2, 0.3])) != 1.0:
        raise ProtocolFailure("all-positive AP rule mismatch")
    mapping_probe = ConditionData(
        "P1", np.zeros((3, 1), dtype=np.float32), np.zeros(3, dtype=np.int8),
        np.asarray(["a", "a", "a"]), np.zeros(3, dtype=np.int8),
        [np.asarray([0, 2], dtype=np.int32), np.asarray([], dtype=np.int32)],
    )
    mapped = project_to_windows(
        mapping_probe, np.asarray([0.2, 0.3, 0.8]), np.asarray([True, True]))
    if not np.array_equal(mapped, np.asarray([0.8, 0.0])):
        raise ProtocolFailure("claim max or uncovered-window-zero rule mismatch")
    for name, result in results.items():
        for fold in FOLDS:
            mask = geometry.answer_folds == fold
            expected = (result["answer_scores"][mask] >=
                        result["thresholds"][str(fold)]["threshold"])
            if not np.array_equal(result["answer_binary"][mask], expected):
                raise ProtocolFailure(f"{name}: answer did not reuse window threshold")
    fake = {
        "P0": {"metrics": {"pooled_outer_oof_window_AP": 0.50,
                             "pooled_outer_oof_answer_AP": 0.60},
               "per_fold": {str(f): {"window_AP": 0.50, "window_F1": 0.50,
                                      "answer_AP": 0.60, "answer_F1": 0.60}
                            for f in FOLDS}},
        "P2": {"metrics": {"pooled_outer_oof_window_AP": 0.55,
                             "pooled_outer_oof_answer_AP": 0.60},
               "per_fold": {str(f): {"window_AP": 0.55, "window_F1": 0.55,
                                      "answer_AP": 0.60, "answer_F1": 0.60}
                            for f in FOLDS}},
        "P3": {"metrics": {"pooled_outer_oof_window_AP": 0.57,
                             "pooled_outer_oof_answer_AP": 0.59},
               "per_fold": {str(f): {"window_AP": 0.56, "window_F1": 0.56,
                                      "answer_AP": 0.59, "answer_F1": 0.59}
                            for f in FOLDS}},
    }
    p3 = np.zeros((EXPECTED_CLAIMS, 4), dtype=np.float32)
    exact = int(np.ceil(0.95 * EXPECTED_CLAIMS)); p3[:exact, (0, 1, 3)] = 1.0
    # A small synthetic geometry cannot satisfy the registered support counts;
    # gate metric inequalities and exact denominator are tested independently.
    gate = compute_advance_gate(fake, p3, geometry)
    if not all(gate["checks"].values()) or gate["exact_denominator"] != EXPECTED_CLAIMS:
        raise ProtocolFailure("advance-gate arithmetic mismatch")
    deltas = pairwise_fold_deltas({**fake, "P1": fake["P0"]})
    if any(not np.isclose(deltas[str(f)]["P3_minus_P2"]["window_AP"], 0.01)
           for f in FOLDS):
        raise ProtocolFailure("per-fold pairwise delta mismatch")
    report = {
        "status": "passed_synthetic_CPU_only_no_real_scoring",
        "script_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(PROTOCOL_PATH), "plan_sha256": sha(PLAN_PATH),
        "clean_input_sha256": sha(INPUT_PATH),
        "sklearn_version": sklearn.__version__,
        "checks": {
            "missing_feature_freeze_failed_before_gold": True,
            "real_gold_open_count": REAL_GOLD_OPEN_COUNT,
            "conditions_exercised": sorted(results),
            "nested_models_per_condition": 25,
            "inner_models_per_condition": 20,
            "outer_models_per_condition": 5,
            "fold_local_answer_base_weights": True,
            "fold_local_weighted_scaler": True,
            "fold_local_class_balance": True,
            "fixed_C_and_liblinear": True,
            "threshold_candidates_and_ties": True,
            "claim_to_4BPE_window_max_and_uncovered_zero": True,
            "answer_max_and_same_fold_threshold": True,
            "pooled_AP_and_F1": True,
            "advance_gate": True,
            "per_fold_pairwise_deltas": True,
        },
        "synthetic_primary_metrics": {
            name: result["metrics"] for name, result in results.items()
        },
        "real_feature_freeze_present": FEATURE_FREEZE_PATH.is_file(),
        "real_evaluation_run": False,
        "fit_gold_opened": False,
        "calibration_read": False, "official_test_read": False,
        "GPU_used": False, "baseline_modified": False,
        "real_entrypoint_status": "fail_closed_until_frozen_features_exist",
        "feature_contract": {
            "directory": str(FEATURE_DIR.resolve()),
            "freeze": FEATURE_FREEZE_PATH.name,
            "manifest": FEATURE_MANIFEST_PATH.name,
            "array_keys": ["P1", "P2", "P3"],
            "array_shapes": {name: [EXPECTED_CLAIMS, width]
                             for name, width in EXPECTED_DIMS.items()},
        },
    }
    write_json_new(OUT / "CPU_EVALUATOR_SELFTEST.json", report)
    print("FORCED_QUOTE_EVALUATOR_SYNTHETIC_SELFTEST_PASSED", flush=True)


def preflight() -> None:
    try:
        receipt, _rows = load_frozen_features()
    except FrozenFeaturesMissing as error:
        print(json.dumps({"status": "blocked_before_gold", "reason": str(error),
                          "fit_gold_open_count": REAL_GOLD_OPEN_COUNT}, indent=2))
        raise SystemExit(2)
    print(json.dumps({"status": "frozen_features_valid_gold_still_unopened",
                      "feature_manifest_sha256": receipt.manifest_sha256,
                      "fit_gold_open_count": REAL_GOLD_OPEN_COUNT}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("selftest", "preflight", "evaluate"))
    arguments = parser.parse_args()
    {"selftest": synthetic_selftest, "preflight": preflight,
     "evaluate": evaluate_real}[arguments.stage]()
