"""Independent scalar policy oracle + graph + gradient audit; no API scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from workflow_risk_fixtures import fixture
from scalar_binary_graph import evaluate
from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN


def scalar_oracle(sample, model, label=None):
    """No production _worlds/_mechanisms/worlds/graph export used for logic."""
    def sigmoid(x):
        return 1. / (1. + math.exp(-x))
    action_ids = [p["action_id"] for p in sample.proposals]
    facts = list(sample.authority.facts.values())
    action_q = []
    for proposal in sample.proposals:
        family = model.registry.specification["dispatch_families"].index(proposal["family"])
        base, positive, negative = model.theta[3 * family:3 * family + 3]
        z = base + (positive if proposal["visible"] == "support" else negative if proposal["visible"] == "opposition" else 0.)
        action_q.append(sigmoid(z))
    enabled = set(model.registry.templates)
    totals, label_probability = {}, 0.
    for fact_bits in product((0, 1), repeat=len(facts)):
        factual = dict(zip([f.identity for f in facts], fact_bits))
        fw = math.prod(sigmoid(model.theta[-3 + f.kind]) if b else 1. - sigmoid(model.theta[-3 + f.kind]) for f, b in zip(facts, fact_bits))
        for bits in product((0, 1), repeat=len(action_ids)):
            dispatch = dict(zip(action_ids, bits))
            weight = fw * math.prod(q if b else 1 - q for q, b in zip(action_q, bits))
            values, owners = {}, {}
            for proposal in sample.proposals:
                aid = proposal["action_id"]
                if "authorization" in enabled:
                    proof = sample.authority.proofs[aid]
                    denied = not any(all(factual[f] == 0 for f in path) for path in proof.paths)
                    key = "authorization:" + aid
                    values[key], owners[key] = bool(dispatch[aid] and denied), {proposal["node"]}
                quantity = proposal["quantity"]
                if "single_bound" in enabled and quantity is not None and quantity["single_limit"] is not None:
                    amount = Decimal(proposal["binding"]["arguments"][quantity["amount_field"]])
                    key = "single_bound:" + aid
                    values[key], owners[key] = bool(dispatch[aid] and amount > Decimal(quantity["single_limit"])), {proposal["node"]}
            with localcontext() as ctx:
                ctx.prec = 512
                for contract in sample.contracts if "cumulative_budget" in enabled else ():
                    total, contributors = Decimal(contract["prior_requested"]), set()
                    for proposal in sample.proposals:
                        quantity = proposal["quantity"]
                        if quantity is None or proposal["binding"]["resource"] not in contract["resources"]:
                            continue
                        args = proposal["binding"]["arguments"]
                        if args[quantity["currency_field"]] != contract["currency"]:
                            continue
                        contributors.add(proposal["node"])
                        if dispatch[proposal["action_id"]]:
                            total += Decimal(args[quantity["amount_field"]])
                    key = "cumulative_budget:" + contract["id"]
                    values[key], owners[key] = total > Decimal(contract["limit"]), contributors
            for policy in sample.combinations if "prohibited_combination" in enabled else ():
                key = "prohibited_combination:" + policy["id"]
                values[key] = all(dispatch[aid] for aid in policy["bindings"])
                owners[key] = {p["node"] for p in sample.proposals if p["action_id"] in policy["bindings"]}
            local = {"node:" + node: any(value for k, value in values.items() if owners[k] == {node}) for node in sample.workflow["nodes"]}
            channels = {"channel:" + name: any(v for k, v in values.items() if k.startswith(name + ":")) for name in enabled}
            values.update(local)
            values.update(channels)
            values["system:any"] = any(channels.values())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + weight * value
            if label is not None:
                observed = {"dispatch": dispatch, "facts": factual, "risks": values}
                if all(observed[kind][key] == value for kind, targets in label.items() for key, value in targets.items()):
                    label_probability += weight
    return totals, label_probability


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((ROOT / 'code/src/paperalpha_monitor').glob("*.py")),
               *sorted((ROOT / 'code/src/paperalpha_runtime').glob("*.py")),
               *sorted((ROOT / 'knowledge/templates').glob("*.json")), ROOT / 'knowledge/risks/risk_catalog.json',
               ROOT / 'code/commands/workflow_risk_fixtures.py', Path(__file__), ROOT / 'code/commands/scalar_binary_graph.py',
               ROOT / 'code/tests/test_workflow_risk_bn.py']
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "evidence_type": "constructed mechanism audit; no API, no held-out accuracy", "source_sha256": hashes,
            "python": sys.version, "numpy": np.__version__}
    (out / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    model = WorkflowRiskBN(regularization=0.)
    model.theta = np.random.default_rng(7313).normal(0, .5, len(model.theta))
    count, graphs, maximum, graph_max, gradient_max, likelihood_max = 0, 0, 0., 0., 0., 0.
    records = []
    for authority, shared, layout, combo in product(("known", "invalid", "root", "delegation", "opaque"), (False, True), ("chain", "fork"), (False, True)):
        sample = fixture(authority=authority, shared=shared, layout=layout, combination=combo)
        expected, _ = scalar_oracle(sample, model)
        actual = model.predict(sample, derivatives=True)
        assert set(expected) == set(actual["probabilities"])
        for key, target in expected.items():
            error = abs(actual["probabilities"][key] - target)
            maximum = max(maximum, error)
            graph_max = max(graph_max, abs(evaluate(model.probability_graph(sample, output=key)) - target))
            graphs += 1
        label = {"dispatch": {"a-0": 1}, "risks": {"channel:cumulative_budget": 0}}
        _, chance = scalar_oracle(sample, model, label)
        value, gradient = model.loss_gradient(model.theta, [sample], [label])
        likelihood_max = max(likelihood_max, abs(value + math.log(chance)))
        numeric = approx_derivative(lambda t: model.loss_gradient(t, [sample], [label])[0], model.theta).ravel()
        gradient_max = max(gradient_max, float(np.max(np.abs(gradient - numeric))))
        records.append({"authority": authority, "shared": shared, "layout": layout, "combination": combo,
                        "input": sample.specification(), "probabilities": actual["probabilities"], "scalar_probabilities": expected})
        count += 1
    assert maximum < 1e-12 and graph_max < 1e-12 and likelihood_max < 1e-12 and gradient_max < 1e-7
    # Frequency recovery is an optimization mechanism check, NOT empirical accuracy.
    samples, labels, weights = [], [], []
    for layout in ("chain", "fork"):
        sample = fixture(authority="known", layout=layout, budget=None)
        for a, b, count_frequency in ((0, 0, 12), (0, 1, 28), (1, 0, 18), (1, 1, 42)):
            samples.append(sample)
            labels.append({"dispatch": {"a-0": a, "a-1": b}})
            weights.append(count_frequency)
    fitted = WorkflowRiskBN(regularization=0.).fit(samples, labels, weights=weights)
    rates = fitted.predict(samples[0])["dispatch_probabilities"]
    assert fitted.optimizer["success"] and max(abs(rates["a-0"] - .6), abs(rates["a-1"] - .7)) < 2e-6
    for relative, sha in hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha, "source changed during audit"
    report = {"evidence_type": plan["evidence_type"], "mechanism_inputs": count, "queries_and_graphs": graphs,
        "scalar_probability_max_error": maximum, "graph_probability_max_error": graph_max,
        "joint_likelihood_max_error": likelihood_max, "likelihood_gradient_max_error": gradient_max,
        "parameter_count": len(model.theta), "frequency_recovery": rates, "fit": fitted.optimizer,
        "real_performance_improvement": None, "original_mas_baselines_reproduced": False,
        "scope": "Explicit pending-action, authority, requested-budget and unordered combination fragment composition; no arbitrary semantic/workflow extraction"}
    for name, value in (("REPORT.json", report), ("cases.json", records), ("model.json", model.parameters()), ("fitted_frequency_model.json", fitted.parameters())):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
