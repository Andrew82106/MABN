"""Measure which evidence can affect the current BN, without outcome labels.

Text replacement is an input-use diagnostic, not a new MAS execution or an
attack experiment. Zero score sensitivity is not a safety certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.journal import RuntimeEvidenceProjector
from paperalpha_monitor.compact_action_bn import CompactActionCompiler, CompactFilter, CompactCapabilityBN
from paperalpha_monitor.capability_action_bn import PARAMETER_NAMES
from paperalpha_monitor.tool_capabilities import DEFAULT_PATH, load_tool_registry
import evaluate_prefix_forecast_v1 as protocol


def project(records):
    projector = RuntimeEvidenceProjector()
    events = []
    for record in records:
        event = projector.push(record)
        if event is not None:
            events.append(event)
    if not events:
        raise ValueError("no model events to diagnose")
    return events, projector.report()


def evaluate(events, theta, registry, *, use_messages=True):
    compiler = CompactActionCompiler(tool_registry=registry)
    engine = CompactFilter(theta, use_messages=use_messages, derivatives=True)
    scores, gradients, info_maxima, operations = [], [], [], Counter()
    for event in events:
        step = compiler.push(event)
        scores.append(engine.apply(step, compiler.variables))
        gradients.append(engine.gradient.copy())
        explanation = engine.explain(step, compiler.variables)
        info_maxima.append(max((v["marginal"] for v in explanation["variables"]
                               if v["kind"] == "information_state"), default=0.))
        operations.update(op.evidence["template"] for op in step.operations)
    if not scores:
        raise ValueError("a nonempty event sequence is required")
    return {"scores": np.asarray(scores), "score_gradients": np.asarray(gradients),
            "information_state_maxima": np.asarray(info_maxima),
            "operation_counts": dict(operations), "last_explanation": explanation}


def replace_content(events):
    """Preserve all structured metadata, identity and order; replace only text."""
    changed = copy.deepcopy(events)
    for event in changed:
        if "content" in event:
            event["content"] = "DIAGNOSTIC REPLACEMENT: unrelated text, not a recorded execution."
    return changed


def diagnose(events, parameter_vectors, registry):
    vectors = np.asarray(parameter_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != len(PARAMETER_NAMES) or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("nonempty finite parameter vectors with capability-v4 layout required")
    changed = replace_content(events)
    original, no_messages, text_changed, gradients, maxima = [], [], [], [], []
    for theta in vectors:
        baseline = evaluate(events, theta, registry)
        original.append(baseline["scores"])
        gradients.append(baseline["score_gradients"])
        maxima.append(baseline["information_state_maxima"])
        no_messages.append(evaluate(events, theta, registry, use_messages=False)["scores"])
        text_changed.append(evaluate(changed, theta, registry)["scores"])
    original, no_messages, text_changed = map(np.asarray, (original, no_messages, text_changed))
    gradients, maxima = np.asarray(gradients), np.asarray(maxima)
    sensitivity = np.max(np.abs(gradients), axis=(0, 1))
    names = list(PARAMETER_NAMES)
    information_parameters = [i for i, name in enumerate(names)
                              if name.startswith(("source:", "adoption:", "transmission:"))]
    actions = baseline["last_explanation"]["actions"]
    report = {
        "model": CompactCapabilityBN.model_name,
        "model_events": len(events), "parameter_vectors": len(vectors),
        "event_counts": dict(Counter(e["event_type"] for e in events)),
        "operation_counts": baseline["operation_counts"],
        "origin_variables": sum(v["kind"] == "origin_hypothesis" for v in baseline["last_explanation"]["variables"]),
        "roles": dict(Counter(v.get("role", "unknown") for v in baseline["last_explanation"]["variables"]
                              if v["kind"] == "information_state")),
        "maximum_information_state_probability": float(maxima.max()),
        "disable_message_propagation_max_score_difference": float(np.max(np.abs(original - no_messages))),
        "replace_content_max_score_difference": float(np.max(np.abs(original - text_changed))),
        "information_parameter_count": len(information_parameters),
        "information_parameters_max_score_gradient": float(sensitivity[information_parameters].max()),
        "parameters_with_nonzero_score_gradient": [name for name, value in zip(names, sensitivity) if value != 0.],
        "parameters_max_abs_score_gradient": dict(zip(names, sensitivity.tolist())),
        "action_permission_statuses": dict(Counter(a["permission_status"] for a in actions)),
        "action_capabilities": dict(Counter(a["capability_profile"]["capability"] for a in actions)),
        "action_scope_complete": sum(a["scope_complete"] for a in actions),
        "labels_used": False, "parameter_fit_performed": False,
        "scope": "Evidence-use and score-derivative diagnostic; no detection accuracy, new execution, calibration or causal claim.",
        "limits": [
            "Numerical checks apply to the supplied events and parameter vectors; zero numerical sensitivity alone is not a universal proof.",
            "Disabling latent propagation retains visible message counts/features; equal scores do not mean all MAS information is ignored.",
            "Replaced content leaves structured evidence unchanged; this diagnoses this adapter/model, not every possible text extractor.",
            "Score gradients are not training-loss gradients: regularization could move parameters without learning from these observations.",
        ],
    }
    arrays = {"theta": vectors, "scores": original, "no_message_scores": no_messages,
              "replaced_content_scores": text_changed, "score_gradients": gradients,
              "information_state_maxima": maxima}
    return report, arrays


def run(input_path, output, registry_path=DEFAULT_PATH):
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an evidence diagnostic")
    # Capture exact input bytes before reading; do not diagnose changing logs.
    raw = input_path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    events, journal = project(records)
    if not journal["run_finished"] or journal["open_invocations"]:
        raise ValueError("this offline diagnostic requires finalized capture; preserve unfinished runs separately")
    output.mkdir(parents=True)
    (output / "runtime_events.jsonl").write_bytes(raw)
    snapshots = protocol.snapshot_sources(output, registry_path)
    for path in [Path(__file__), *(ROOT / 'code/src/paperalpha_runtime').glob("*.py")]:
        relative = path.resolve().relative_to(ROOT.resolve())
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshots[relative.as_posix()] = {"original": str(path), "sha256": protocol.digest(target)}
    seed = 20260919
    rng = np.random.default_rng(seed)
    # One untrained default plus eight dense vectors; not selected by scores.
    vectors = np.vstack([CompactCapabilityBN().theta, rng.normal(0., .7, (8, len(PARAMETER_NAMES)))])
    report, arrays = diagnose(events, vectors, load_tool_registry(registry_path))
    report.update(journal=journal, parameter_probe_seed=seed,
                  parameter_probe_description="untrained default plus eight independent Normal(0,0.7) vectors; no labels or optimization")
    np.savez_compressed(output / "NUMERICS.npz", **arrays)
    protocol.save_json(output / "REPORT.json", report)
    protocol.save_json(output / "manifest.json", {
        "source_snapshot": snapshots,
        "inputs": {str(input_path): protocol.digest(output / "runtime_events.jsonl"),
                   str(registry_path): protocol.digest(registry_path)},
        "artifacts": {name: protocol.digest(output / name) for name in ("runtime_events.jsonl", "REPORT.json", "NUMERICS.npz")},
    })
    return {key: report[key] for key in ("model_events", "parameter_vectors", "origin_variables",
        "maximum_information_state_probability", "disable_message_propagation_max_score_difference",
        "replace_content_max_score_difference", "information_parameters_max_score_gradient")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tool-registry", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out, args.tool_registry), ensure_ascii=False, indent=2))
