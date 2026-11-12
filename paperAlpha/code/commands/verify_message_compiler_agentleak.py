"""Replay existing development prefixes through the generic message compiler.

All graph structures/observations are compared; predictions and gradients are
additionally compared on the first ID per task template, without label-based
selection. This is input/compiler equivalence, NOT a new performance test.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, write, sha
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.task_field_semantics import context_from_initial, bind_task_cues
from paperalpha_monitor.agentleak_event_projection import project
from paperalpha_monitor.message_workflow_compiler import MessageWorkflowCompiler
from paperalpha_monitor.agentleak_message_compiler import compile_prefix, rebind_existing_policy
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN, SemanticArrivalWorkflowRiskBN


def run(experiment, destination):
    plan, fit = read(experiment / "PLAN.json"), read(experiment / "FIT.json")
    for name, expected in plan["inputs_sha256"].items():
        assert sha(experiment / "inputs" / name) == expected
    for name in ('code/src/paperalpha_monitor/agentleak_event_projection.py', 'code/src/paperalpha_monitor/task_cue_message.py', 'code/src/paperalpha_monitor/message_hurdle.py'):
        assert sha(ROOT / name) == plan["source_sha256"][name]
    rows = read(experiment / "inputs/train.json") + read(experiment / "inputs/validation.json")
    provenance = {r["id"]:r for r in read(experiment / "inputs/provenance.json")}
    contexts = {r["id"]:r["context_sha256"] for r in read(experiment / "TASK_PROJECTION.json")}
    selected = {}; rows.sort(key=lambda r:r["id"])
    for row in rows: selected.setdefault(row["template_id"], row["id"])
    selected_ids = set(selected.values())
    models = {}
    for name, cls in (("counts_bn_0.001", TaskCueMessageWorkflowRiskBN), ("task_bn_0.001", TaskCueMessageWorkflowRiskBN), ("arrival_bn_0.001", SemanticArrivalWorkflowRiskBN)):
        path = experiment / "models" / (name + ".json")
        assert sha(path) == fit["model_file_sha256"][path.name]
        models[name] = cls.restore(read(path))
    destination.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/verify_message_compiler_agentleak.py', 'code/tests/test_message_workflow_compiler.py',
        'code/commands/run_agentleak_event_dag.py', 'code/commands/inspect_agentleak_release.py', 'code/commands/harnessaudit_cohort.py']
    for name in names:
        dest = destination / "source" / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    sources = {n:sha(destination / "source" / n) for n in names}
    write(destination / "PLAN.json", {"schema":"message-compiler-prefix-replay-v1", "experiment":str(experiment),
        "experiment_plan_sha256":sha(experiment / "PLAN.json"), "source_sha256":sources,
        "selected_ids_by_template":selected, "selection":"First lexicographic ID per template; all graphs compared, selected predictions/gradients compared.",
        "uses_final_output":False, "new_fit":False, "new_api_calls":0, "old_test_read":False})
    compiler = MessageWorkflowCompiler(); comparisons, prediction_rows = [], []; changed_routes = 0
    max_probability = max_gradient = 0.
    for i,row in enumerate(rows):
        prov = provenance[row["id"]]; path = Path(plan["upstream"]) / prov["source_path"]
        assert sha(path) == prov["source_sha256"]
        raw = read(path); initial = raw["input"]
        def prefix_call(index):
            return {k:raw["llm_calls"][index][k] for k in ("call_id", "system_prompt", "user_prompt", "response")}
        coordinator, worker = prefix_call(1), prefix_call(2)
        for phase in ("early", "late"):
            dag, manifest = compile_prefix(initial, coordinator=coordinator, worker=worker if phase == "late" else None, compiler=compiler)
            old = project(initial, coordinator_text=coordinator["response"], coordinator_record_id=coordinator["call_id"],
                **({"worker_text":worker["response"], "worker_record_id":worker["call_id"]} if phase == "late" else {}))
            previous = old.event_dag
            assert dag.workflow == previous.workflow and dag.events == previous.events and dag.observations == previous.observations
            assert dag.order == previous.order and dag.parents == previous.parents and dag.missing_past == previous.missing_past
            assert [(e["source"],e["target"],e["basis"]) for e in dag.edges] == [(e["source"],e["target"],e["basis"]) for e in previous.edges]
            changed = sum(a["evidence"] != b["evidence"] for a,b in zip(dag.edges,previous.edges)); changed_routes += changed
            new = rebind_existing_policy(old, dag)
            assert new.authority.specification() == old.authority.specification() and new.queries == old.queries
            comparisons.append({"id":row["id"], "phase":phase, "source_sha256":prov["source_sha256"],
                "previous_sample_sha256":old.sha256, "compiled_sample_sha256":new.sha256, "manifest_sha256":digest(manifest),
                "events":len(dag.ids), "route_evidence_updated":changed})
            if row["id"] in selected_ids:
                annotation = read(experiment / "extraction/classifications" / (contexts[row["id"]] + ".json"))["annotation"]
                context, _ = context_from_initial(initial["request"], initial["vault"])
                assert digest(context) == contexts[row["id"]]
                for name,model in models.items():
                    a,b = (old,new) if name.startswith("counts") else tuple(bind_task_cues(s,context,annotation) for s in (old,new))
                    p,q = [model.predict_known_authorization_union(s, derivatives=True) for s in (a,b)]
                    error = abs(p["probabilities"]["system:any"]-q["probabilities"]["system:any"])
                    gradient = float(np.max(np.abs(p["gradients"]["system:any"]-q["gradients"]["system:any"])))
                    assert error <= 1e-12 and gradient <= 1e-12
                    max_probability=max(max_probability,error); max_gradient=max(max_gradient,gradient)
                    prediction_rows.append({"id":row["id"], "phase":phase, "model":name, "model_file_sha256":fit["model_file_sha256"][name+".json"],
                        "previous":p["probabilities"]["system:any"], "compiled":q["probabilities"]["system:any"], "maximum_gradient_difference":gradient})
                sample_dir = destination / "examples" / (row["id"] + "_" + phase)
                write(sample_dir / "manifest.json", manifest); write(sample_dir / "event_dag.json", dag.specification())
        if (i+1)%500 == 0: print(json.dumps({"flows_compared":i+1}), flush=True)
    for name,expected in sources.items(): assert sha(ROOT / name) == expected
    write(destination / "COMPARISONS.json", comparisons); write(destination / "PREDICTIONS.json", prediction_rows)
    report = {"flows":len(rows), "compiled_prefixes":len(comparisons), "event_nodes_compared":sum(r["events"] for r in comparisons),
        "route_evidence_updated":changed_routes, "prediction_gradient_pairs":len(prediction_rows),
        "maximum_probability_difference":max_probability, "maximum_gradient_difference":max_gradient,
        "scope":"All existing development prefix event structures and observations; selected frozen predictions and gradients. Not new detection accuracy, automatically discovered workflows or new API data.",
        "knowledge_sha256":compiler.knowledge_sha256, "compiler_sha256":compiler.sha256,
        "comparisons_sha256":sha(destination / "COMPARISONS.json"), "predictions_sha256":sha(destination / "PREDICTIONS.json"),
        "new_fit":False, "new_api_calls":0, "old_test_read":False,
        "evidence_change":"Past deliveries now cite actual destination API inputs at the completed-prefix stage, not initial planned-route text; future routes remain declared. Historical artifacts unchanged."}
    write(destination / "REPORT.json", report); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment",type=Path,required=True); parser.add_argument("--destination",type=Path,required=True)
    args = parser.parse_args(); run(args.experiment.resolve(), args.destination.resolve())
