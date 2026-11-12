"""Label-free frozen numerical worker; invoked BEFORE either buyer request."""
from pathlib import Path
import argparse
from itertools import product
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read, write, sha
from evaluate_choice_attributes import reference_worlds
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN, attribute_rows
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input
from paperalpha_runtime.crossed_purchase import independent_features
from paperalpha_runtime.mas_purchase_study import reference_verdict
from scalar_binary_graph import evaluate


def policy_truth(value, sample, states):
    answer = []
    for bits in states:
        packet = {"task_id": value["task_id"], "principal": value["principal"],
                  "actions": [p["binding"] for p, bit in zip(sample.proposals, bits) if bit],
                  "authority_records": value["authority_records"], "cutoff": value["cutoff"],
                  "grant_references": {b["actor"]: "child-" + b["actor"][-1] for b in value["branches"]},
                  "single_limits": {b["resource"]: b["single_limit"] for b in value["branches"]},
                  "contracts": [{**c, "task_id": value["task_id"], "principal": value["principal"], "currency": "USD"} for c in value["contracts"]]}
        answer.append(int(not reference_verdict(packet)["allowed"]))
    return np.asarray(answer)


def score(value, models):
    sample = compile_attribute_input(value)
    x = np.asarray(independent_features(value, sample)); actual, evidence = attribute_rows(sample)
    if not np.array_equal(x, actual): raise ValueError("independent prefix feature extraction differs")
    states = np.asarray(list(product((0, 1), repeat=len(sample.action_ids))))
    truth = policy_truth(value, sample, states); result = {}
    audit = {"graphs": 0, "max_graph_error": 0., "max_model_joint_error": 0.}
    def save(name, p):
        if min(p) < 0 or abs(p.sum() - 1) > 1e-12: raise ValueError("improper choice distribution")
        slots = []
        for g in sample.choice_groups:
            ix = [sample.action_ids.index(a) for a in g["action_ids"]]
            if float(p[states[:, ix].sum(axis=1) > 1].sum()) != 0: raise ValueError("impossible multiple-choice mass")
            slots.append({"id": g["id"], "action_ids": g["action_ids"],
                          "probabilities": [float(p[states[:, ix].sum(axis=1) == 0].sum())]
                                           + [float(p[states[:, i] == 1].sum()) for i in ix]})
        result[name] = {"joint": p.tolist(), "risk_probability": float(p @ truth), "slots": slots}
    for path in sorted(models.glob("*.json")):
        saved = read(path); name = path.stem
        if name.startswith("attribute_bn_"):
            model = AttributeChoiceWorkflowRiskBN.restore(saved)
            worlds = model._worlds(sample, model.theta); p = np.exp(worlds["log_mass"]).sum(axis=1)
            independently = reference_worlds(sample, x, model.theta[:10])[1]
            error = float(max(abs(independently - p))); audit["max_model_joint_error"] = max(audit["max_model_joint_error"], error)
            if error > 1e-12: raise ValueError("frozen BN/reference joint differs")
            prediction = model.predict(sample)
            for query, probability in prediction["probabilities"].items():
                error = abs(evaluate(model.probability_graph(sample, output=query)) - probability)
                audit["graphs"] += 1; audit["max_graph_error"] = max(audit["max_graph_error"], error)
                if error > 1e-12: raise ValueError("exported BN differs")
            if abs(prediction["probabilities"]["system:any"] - float(p @ truth)) > 1e-12:
                raise ValueError("policy oracle differs from model query")
            save(name, p)
        elif name.startswith("standalone_"):
            save(name, reference_worlds(sample, x, np.asarray(saved["theta"]))[1])
        else:
            raise ValueError("only preselected attribute methods may be loaded")
    # Predeclared transparent comparators get the same price/warranty/priority.
    # They are forecasting heuristics, not native MAS monitor reproductions.
    priority = json.loads(value["actual_planner_text"])["priority"]
    for rule in ("lowest_cost", "longest_warranty", "priority_value"):
        chosen = np.zeros(len(sample.action_ids), int)
        for g in sample.choice_groups:
            ix = [sample.action_ids.index(a) for a in g["action_ids"]]
            def key(i):
                cost, warranty = x[i, 3], x[i, 4]
                mode = ("lowest_cost" if priority == "economy" else "longest_warranty" if priority == "quality" else "value") if rule == "priority_value" else rule
                if mode == "lowest_cost": return (cost, -warranty, i)
                if mode == "longest_warranty": return (-warranty, cost, i)
                return (-warranty / cost, cost, i)
            chosen[min(ix, key=key)] = 1
        p = np.zeros(len(states)); p[np.flatnonzero(np.all(states == chosen, axis=1))[0]] = 1.
        save("rule_" + rule, p)
    return {"schema": "crossed-choice-frozen-forecast-v1", "input_sha256": sample.sha256,
            "action_ids": list(sample.action_ids), "policy_truth": truth.tolist(), "methods": result,
            "attribute_evidence": evidence, "numerical_audit": audit,
            "target": "Future local simulated dispatch risk conditional on prepared input delivered unchanged and valid command protocol completed; not API/parse failures or real financial effects.",
            "model_sha256": {p.name: sha(p) for p in sorted(models.glob("*.json"))}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); write(a.out, score(read(a.prefix)["value"], a.models))
