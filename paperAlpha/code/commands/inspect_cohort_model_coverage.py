"""Replay current v7 on new raw journals to measure representation coverage.

No semantic API calls, labels, training, thresholds or efficacy scores. Missing
text annotations remain missing; initial user prompts are not silently injected
from the audit task. Include failed trajectories for engineering replay only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from harnessaudit_cohort import ROOT, load_plan, sha, write_new

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.knowledge_scoped_bn import JournalKnowledgeCompiler, KnowledgeScopedBN
from paperalpha_monitor.tool_capabilities import load_tool_registry


def inspect_rows(rows):
    model = KnowledgeScopedBN()
    compiler = JournalKnowledgeCompiler(tool_registry=load_tool_registry())
    engine = model._engine(model.theta)
    steps, revisions, unavailable = 0, 0, 0
    factors = Counter()
    for row in rows:
        step = compiler.push(row)
        if step is None:
            continue
        engine.apply(step, compiler.variables)
        steps += 1
        factors.update(f.template_id for f in step.mechanism_factors)
        if step.revision is not None:
            revisions += 1
            unavailable += "extraction_unavailable" in step.revision.evidence["feature_names"]
    if not steps:
        raise ValueError("no usable model events")
    explanation = engine.explain(compiler.last_step, compiler.variables)
    actions = explanation["actions"]
    return {"model": model.model_name, "parameter_count": len(model.theta), "model_events": steps,
            "process_nodes": len(compiler.variables), "retained_joint_states": explanation["retained_joint_states"],
            "actions": len(actions), "actions_with_unknown_role": sum(a["role"] == "unknown" for a in actions),
            "actions_with_unknown_permission": sum(a["permission_status"] == "unknown" for a in actions),
            "actions_with_incomplete_scope": len(explanation["coverage"]["incomplete_scope_actions"]),
            "actions_with_unknown_capability": len(explanation["coverage"]["unknown_capability_actions"]),
            "instantiated_factors": dict(factors), "text_records": revisions, "text_extraction_unavailable": unavailable,
            "mechanism_registry_sha256": model.registry.sha256,
            "model_parameters": "unchanged untrained defaults; no effectiveness scores reported",
            "interpretation": "Acceptance/coverage only. Structural replay of a failed run cannot make its endpoint label eligible."}


def inspect_cohort(plan_path, output):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = load_plan(plan_path)
    if output.exists():
        raise ValueError("new coverage directory required")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for task in plan["tasks"]:
        path = plan_path.parent / "runs" / task["task_id"] / "runtime_events.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            result = {"status": "replayed", **inspect_rows(rows)}
        except (ValueError, KeyError, TypeError) as exc:
            result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        results.append({"task_id": task["task_id"], "input_sha256": sha(path), **result})
    sources = [Path(__file__), *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
               *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json"),
               ROOT / 'knowledge/risks/risk_catalog.json']
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        hashes[relative.as_posix()] = sha(target)
    report = {"plan_sha256": plan["plan_sha256"], "runs": results, "source_sha256": hashes,
              "new_api_calls": 0, "trained": False, "labels_used_by_model": False,
              "scope": "Engineering coverage on new captured executions, not safety performance or fixed-DAG evaluation."}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_cohort(args.plan, args.out)
    print(json.dumps(report["runs"], ensure_ascii=False, indent=2))
