"""Blind human-annotation packets and agreement statistics for MAS traces.

The annotation packet is a one-way projection: labels, hidden fields,
evaluator outputs, and model scores are removed before a trace reaches an
annotator.  Observable evidence receives stable ``evidence_id`` values so an
annotator can cite a first violating event without seeing the source id.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_FORBIDDEN_EXACT = {
    "label", "labels", "external_effect", "hidden_fields", "label_source",
    "scenario_family", "mechanism", "evaluation", "evaluator",
    "ground_truth", "groundtruth", "outcome", "prediction", "predictions",
    "risk_score", "monitor_score", "model_score", "hierarchical_fusion",
    "runtime_bn", "runtime_logistic", "workflow_bn", "workflow_logistic",
}
_FORBIDDEN_TOKENS = (
    "groundtruth", "ground_truth", "hidden", "label_source", "scenario_family",
    "model_score", "monitor_score", "prediction", "evaluator", "evaluation",
    "risk_score",
)
_KEY_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_CLASSES = (
    "none", "unauthorized_access", "privilege_escalation",
    "tainted_data_to_privileged_tool", "unexpected_delegation",
    "cross_agent_collusion", "other",
)


def _normal_key(key: Any) -> str:
    return _KEY_RE.sub("_", str(key).lower()).strip("_")


def is_forbidden_key(key: Any) -> bool:
    normal = _normal_key(key)
    return normal in _FORBIDDEN_EXACT or any(token in normal for token in _FORBIDDEN_TOKENS)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL, a JSON list, or ``{"records": [...]}``."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            rows = value["records"]
        elif isinstance(value, list):
            rows = value
        else:
            rows = [value]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source} must contain JSON objects")
    return rows


def _safe_value(value: Any, *, drop_ids: bool = False) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if is_forbidden_key(key):
                continue
            if drop_ids and _normal_key(key) in {"event_id", "id"}:
                continue
            result[str(key)] = _safe_value(child)
        return result
    if isinstance(value, list):
        return [_safe_value(child) for child in value]
    if isinstance(value, tuple):
        return [_safe_value(child) for child in value]
    return value


def _source_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("episode_id", "trace_id", "case_id", "run_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"row-{index:06d}"


def _annotation_id(source_id: str, index: int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{index}\0{source_id}".encode("utf-8")).hexdigest()
    return f"mas-{digest[:20]}"


def _events_projection(events: Any, annotation_id: str) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    projected: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        evidence_id = f"{annotation_id}:event:{index:04d}"
        clean = _safe_value(event, drop_ids=True)
        if not isinstance(clean, dict):
            clean = {}
        if "sequence" not in clean:
            clean["sequence"] = index
        projected.append({"evidence_id": evidence_id, **clean})
    return projected


def _trace_projection(row: Mapping[str, Any], annotation_id: str) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    if isinstance(row.get("agents"), list):
        trace["agents"] = _safe_value(row["agents"])
    trace["events"] = _events_projection(row.get("events", []), annotation_id)
    workflow = row.get("workflow_dag")
    if isinstance(workflow, Mapping):
        clean_workflow = _safe_value(workflow)
        if isinstance(clean_workflow, dict):
            trace["workflow_dag"] = clean_workflow
    for key in ("records", "observations", "failed_request_ids", "topology"):
        if key in row and not is_forbidden_key(key):
            trace[key] = _safe_value(row[key])
    return trace


def _projection_for_row(projection: Mapping[str, Any] | None) -> dict[str, Any]:
    if not projection:
        return {}
    clean = _safe_value(projection)
    if not isinstance(clean, dict):
        return {}
    for key in ("episode_id", "trace_id", "case_id", "run_id", "id"):
        clean.pop(key, None)
    return clean


def _index_projections(projections: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in projections:
        source_id = _source_id(row, len(indexed))
        if source_id in indexed:
            raise ValueError(f"duplicate projection source id: {source_id}")
        indexed[source_id] = row
    return indexed


def build_annotation_rows(
    episodes: Sequence[Mapping[str, Any]],
    projections: Sequence[Mapping[str, Any]] | None = None,
    *,
    salt: str = "paperalpha-mas-blind-v1",
    shuffle_seed: int | None = 0,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build paired blank rows and a non-sensitive source index."""
    if not episodes:
        raise ValueError("at least one episode is required")
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError("classes must be a non-empty sequence of unique names")
    projection_index = _index_projections(projections or [])
    rows: list[dict[str, Any]] = []
    source_index: list[dict[str, str]] = []
    for index, episode in enumerate(episodes):
        source_id = _source_id(episode, index)
        annotation_id = _annotation_id(source_id, index, salt)
        trace = _trace_projection(episode, annotation_id)
        row = {
            "annotation_id": annotation_id,
            "trace": trace,
            "monitor_projection": _projection_for_row(projection_index.get(source_id)),
            "annotation": {
                "system_violation": None,
                "violation_class": None,
                "first_violation_event_id": None,
                "evidence_event_ids": [],
                "rationale": "",
            },
        }
        rows.append(row)
        digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        source_index.append({"annotation_id": annotation_id, "row_index": str(index), "trace_sha256": digest})

    if shuffle_seed is not None:
        import random

        order = list(range(len(rows)))
        random.Random(shuffle_seed).shuffle(order)
        rows = [rows[index] for index in order]
        source_index = [source_index[index] for index in order]
    return rows, source_index


def validate_annotation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        annotation_id = row.get("annotation_id")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("each annotation row needs a non-empty annotation_id")
        if annotation_id in seen:
            raise ValueError(f"duplicate annotation_id: {annotation_id}")
        seen.add(annotation_id)
        annotation = row.get("annotation")
        if not isinstance(annotation, Mapping):
            raise ValueError(f"{annotation_id}: annotation must be an object")
        violation = annotation.get("system_violation")
        if violation not in (None, True, False, 0, 1):
            raise ValueError(f"{annotation_id}: system_violation must be bool or null")
        event_id = annotation.get("first_violation_event_id")
        if event_id is not None and not isinstance(event_id, str):
            raise ValueError(f"{annotation_id}: first_violation_event_id must be string or null")
        evidence = annotation.get("evidence_event_ids", [])
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise ValueError(f"{annotation_id}: evidence_event_ids must be a list of strings")
        if not isinstance(annotation.get("rationale", ""), str):
            raise ValueError(f"{annotation_id}: rationale must be a string")


def _normal_category(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return value


def cohen_kappa(values_a: Sequence[Any], values_b: Sequence[Any]) -> dict[str, Any]:
    """Return pairwise agreement and Cohen's kappa, excluding missing pairs."""
    if len(values_a) != len(values_b):
        raise ValueError("agreement vectors must have equal length")
    pairs = [(_normal_category(a), _normal_category(b)) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    missing_a = sum(a is None for a in values_a)
    missing_b = sum(b is None for b in values_b)
    n = len(pairs)
    if not n:
        return {"n": 0, "missing_a": missing_a, "missing_b": missing_b, "observed_agreement": None, "expected_agreement": None, "cohens_kappa": None, "confusion": {}}
    observed = sum(a == b for a, b in pairs) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    categories = sorted(set(a_counts) | set(b_counts), key=lambda item: str(item))
    expected = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else ((observed - expected) / (1.0 - expected) if expected != 1.0 else 0.0)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        confusion[str(a)][str(b)] += 1
    return {"n": n, "missing_a": missing_a, "missing_b": missing_b, "observed_agreement": observed, "expected_agreement": expected, "cohens_kappa": kappa, "confusion": {a: dict(b) for a, b in confusion.items()}}


def evidence_set_agreement(values_a: Sequence[Sequence[str]], values_b: Sequence[Sequence[str]]) -> dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("evidence vectors must have equal length")
    jaccards: list[float] = []
    exact = 0
    for left, right in zip(values_a, values_b):
        a, b = set(left or ()), set(right or ())
        union = a | b
        jaccards.append(1.0 if not union else len(a & b) / len(union))
        exact += int(a == b)
    return {"n": len(jaccards), "exact_set_agreement": (exact / len(jaccards) if jaccards else None), "mean_jaccard": (sum(jaccards) / len(jaccards) if jaccards else None)}


def agreement_report(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_annotation_rows(rows_a)
    validate_annotation_rows(rows_b)
    by_a = {str(row["annotation_id"]): row for row in rows_a}
    by_b = {str(row["annotation_id"]): row for row in rows_b}
    if set(by_a) != set(by_b):
        raise ValueError("annotation ids differ")
    ids = sorted(by_a)
    ann_a = [by_a[item]["annotation"] for item in ids]
    ann_b = [by_b[item]["annotation"] for item in ids]
    return {
        "n_items": len(ids),
        "annotation_ids_match": True,
        "system_violation": cohen_kappa([row.get("system_violation") for row in ann_a], [row.get("system_violation") for row in ann_b]),
        "violation_class": cohen_kappa([row.get("violation_class") for row in ann_a], [row.get("violation_class") for row in ann_b]),
        "first_violation_event_id": cohen_kappa([row.get("first_violation_event_id") for row in ann_a], [row.get("first_violation_event_id") for row in ann_b]),
        "evidence_event_ids": evidence_set_agreement([row.get("evidence_event_ids", []) for row in ann_a], [row.get("evidence_event_ids", []) for row in ann_b]),
    }


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
