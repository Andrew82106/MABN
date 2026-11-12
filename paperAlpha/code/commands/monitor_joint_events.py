"""Passive JSON-lines adapter for an already fitted joint risk monitor.

stdin: one structured runtime event per line. stdout: one accepted estimate
or explicit rejection per line. No LLM calls, actions, blocking or retraining.
The caller supplies audited pre-run context and a separately chosen threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor import DynamicJointBN, JointRiskStream
from paperalpha_monitor.dynamic_joint_bn import PARAMETER_NAMES, KNOWLEDGE


def load_model(path, method="joint_bn"):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("name") not in {"DynamicJointBN-v1", "TypedJointBN-v2", "ActionScopedJointBN-v3"}:
        if method not in record or "bn" not in record[method]:
            raise ValueError("selected method is not a saved joint BN")
        record = record[method]["bn"]
    if record.get("name") == "ActionScopedJointBN-v3":
        from paperalpha_monitor.action_joint_bn import ActionScopedJointBN, PARAMETER_NAMES as action_names, SPEC, SPEC_SHA256
        if record.get("parameter_names") != action_names or record.get("action_registry") != {"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256}:
            raise ValueError("action parameter layout or knowledge specification does not match this monitor")
        model = ActionScopedJointBN(use_messages=record["use_messages"], use_latent_head=record["use_latent_head"])
    elif record.get("name") == "TypedJointBN-v2":
        from paperalpha_monitor.typed_joint_bn import TypedJointBN, PARAMETER_NAMES as typed_names, SPEC, SPEC_SHA256
        if record.get("parameter_names") != typed_names or record.get("typed_registry") != {"id": SPEC["registry_id"], "specification_sha256": SPEC_SHA256}:
            raise ValueError("typed parameter layout or knowledge specification does not match this monitor")
        model = TypedJointBN(typed_head=record["typed_head"], use_messages=record["use_messages"], use_latent_head=record["use_latent_head"])
    elif record.get("name") == "DynamicJointBN-v1" and record.get("parameter_names") == PARAMETER_NAMES:
        model = DynamicJointBN(use_messages=record["use_messages"], use_latent_head=record["use_latent_head"])
    else:
        raise ValueError("model name or parameter layout does not match this monitor")
    binding = record.get("knowledge_registry")
    if record.get("name") in {"TypedJointBN-v2", "ActionScopedJointBN-v3"} and binding is None:
        raise ValueError("typed/action model requires the base knowledge specification binding")
    if binding is not None and binding != {"id": KNOWLEDGE.registry_id, "specification_sha256": KNOWLEDGE.fingerprint}:
        raise ValueError("saved model knowledge specification differs from the active registry")
    model.theta = np.asarray(record["theta"], dtype=float)
    model.optimizer = record.get("optimizer", {})
    model.regularization = record.get("regularization", model.regularization)
    model.max_iter = record.get("max_iter", model.max_iter)
    # JointRiskStream checks shape/finiteness before accepting any events.
    return model


def monitor_lines(stream, lines, *, explain=False):
    """Yield an error, not a safe score, when an input cannot be consumed."""
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            result = stream.push(event)
            if explain:
                result = stream.explain()
            yield {"status": "accepted", "line": number, **result}
        except (ValueError, TypeError, KeyError) as exc:
            yield {"status": "rejected", "line": number, "error": str(exc),
                   "risk_probability": None, "alarm": None,
                   "interpretation": "this event was not accepted; missing evidence is not safety"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-parameters", type=Path, required=True)
    parser.add_argument("--method", default="joint_bn")
    parser.add_argument("--context", type=Path, help="pre-run workflow/roles/allowed_tools/permission_mode JSON; action model also accepts trusted_checkers")
    parser.add_argument("--threshold", type=float, help="separately validated threshold; omitted means probability only")
    parser.add_argument("--max-variables", type=int, help="default: 10 for original BN, 12 for typed/action BN")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else {}
    if not isinstance(context, dict) or set(context) - {"workflow", "roles", "allowed_tools", "permission_mode", "trusted_checkers"}:
        raise ValueError("context accepts only pre-run workflow, roles, allowed_tools, permission_mode and trusted_checkers")
    model = load_model(args.model_parameters, args.method)
    name = model.parameters()["name"]
    if "trusted_checkers" in context and name != "ActionScopedJointBN-v3":
        raise ValueError("trusted_checkers are specific to the action-scoped input protocol")
    if name == "ActionScopedJointBN-v3":
        from paperalpha_monitor import ActionRiskStream
        stream_type, default_limit = ActionRiskStream, 12
    elif name == "TypedJointBN-v2":
        from paperalpha_monitor import TypedRiskStream
        stream_type, default_limit = TypedRiskStream, 12
    else:
        stream_type, default_limit = JointRiskStream, 10
    limit = default_limit if args.max_variables is None else args.max_variables
    stream = stream_type(model, threshold=args.threshold, max_variables=limit, **context)
    rejected = False
    for result in monitor_lines(stream, sys.stdin, explain=args.explain):
        rejected |= result["status"] == "rejected"
        result["training_optimizer_success"] = model.optimizer.get("success")
        result["deployment_validation"] = "not established by loading this development model"
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
