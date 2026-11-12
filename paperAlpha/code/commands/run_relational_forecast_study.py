"""Freeze, collect and fit a relation-varying API development pilot.

Only development cases are collected. New test topics remain unrun until a
separate frozen testing protocol is reviewed. No earlier held-out files read.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime import relational_forecast_study as study
from paperalpha_runtime.privacy_transfer import delivered_text


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(out, numerical_python):
    out.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/run_relational_forecast_study.py', 'code/commands/scalar_binary_graph.py',
              'code/commands/verify_relational_forecast_study.py', 'code/commands/audit_relational_event_meaning.py',
              'code/tests/test_relational_forecast_study.py']
    for name in names:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    check = subprocess.run([str(numerical_python), "-X", "utf8", "-B", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from paperalpha_runtime import relational_forecast_study as s; "
        "from paperalpha_monitor.observable_event_learning import fit_event_traces; "
        "print(len([s.compile_input(s.prefix(c, {'source': {'record_id': 'source', 'sequence': 1, 'text': c['source']}}, 2)) for c in s.cases()]))",
        str(out / "source/code/src")], capture_output=True, text=True, timeout=60)
    write(out / "IMPORT_CHECK.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    if check.returncode:
        raise RuntimeError("frozen standalone numerical preflight failed; no API calls made")
    plan = {"schema": "relational-forecast-study-v1", "cases": study.cases(),
        "source_sha256": {name: sha(out / "source" / name) for name in sorted(names)},
        "endpoint": "http://localhost:58661/v1", "model": "gpt-5.5", "effort": "low", "transport_retries": 0,
        "max_output_tokens": 2400, "timeout_seconds": 60, "concurrency": 3,
        "collect_splits": ["train", "validation"], "planned_development_calls": 84,
        "regularization": [.001, .03], "initialization": "zero", "max_iter": 500,
        "numerical_python": str(numerical_python.resolve()),
        "training": "One early-prefix likelihood per workflow. Full future intermediate+terminal labels, and terminal-only supervision control. No duplication of the late prefix in training.",
        "comparators": ["event DAG BN", "same fitted marginals made independent (dependency ablation)",
            "same labelled local conditional logistic models (equivalence audit)", "direct 4-outcome linear softmax using prefix and topology", "preserve-or-remove rule", "always absent rule"],
        "metrics": ["4-outcome joint log loss", "4-outcome Brier", "any unauthorized literal appearance Brier and F1", "both appearances Brier"],
        "limits": ["Authored fictional tasks, not a native public benchmark.", "12 train and 12 validation workflows in two task templates: not evidence of broad generalization.",
            "No native MAS-method superiority claim; current-action guards are not relabelled as future forecasters.",
            "Literal canary appearance is a study policy violation, not semantic leakage or real-world harm.",
            "BN full-trace fit has intermediate supervision; terminal-only control and local logistic receive their explicitly matched supervision.",
            "Reserved-test cases are not generated or read by this development experiment; no tuning on old held-out sets."],
        "docs": "https://developers.openai.com/api/reference/cli/resources/responses/methods/create"}
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "source_files": len(names), "plan_sha256": sha(out / "PLAN.json"), "planned_development_calls": 84}))


def frozen_plan(out):
    plan = read(out / "PLAN.json")
    if ROOT != (out / "source").resolve():
        raise ValueError("use the frozen script, not the live worktree")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen source changed: " + name)
    if study.cases() != plan["cases"]:
        raise ValueError("frozen cases changed")
    return plan


async def collect_case(out, case, api):
    directory = out / "runs" / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    def record(kind, **value):
        result = {"sequence": len(records) + 1, "monotonic_ns": time.monotonic_ns(), "kind": kind, **value}
        write(directory / f"record_{result['sequence']:04d}.json", result)
        records.append(result)
        return result
    count, failure = 0, None
    source = record("source", text=case["source"])
    completed = {"source": {"record_id": "record_0001.json", "sequence": 1, "text": source["text"]}}
    messages = {"source": case["source"]}
    replies = {}
    early = study.prefix(case, completed, len(records) + 1)
    record("prefix", phase="early", packet=early)
    write(directory / "EARLY.json", early)
    try:
        w = study.workflow(case)
        for node in w["nodes"][1:]:
            if node == "publish-a":
                late = study.prefix(case, completed, len(records) + 1)
                record("prefix", phase="late", packet=late)
                write(directory / "LATE.json", late)
            request = study.request(case, node, messages)
            record("request", stage=node, request=request)
            count += 1
            raw = (await api.responses.create(**request)).model_dump(mode="json")
            public = {k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage")}
            public["output"] = [m for m in raw.get("output", []) if m.get("type") == "message"]
            response = record("response", stage=node, response=public)
            reply = delivered_text(public)
            if reply["native_refusal"]:
                raise ValueError("native refusal; preserve as collection failure")
            item = {"record_id": f"record_{response['sequence']:04d}.json", "sequence": response["sequence"], "text": reply["text"]}
            replies[node], messages[node] = item, item["text"]
            if node.startswith("review-"):
                completed[node] = item
        labels = study.outcome(case, replies)
        write(directory / "OUTCOME.json", labels)
        record("outcome", outcome_sha256=sha(directory / "OUTCOME.json"))
    except Exception as exc:
        # Do not serialize exception bodies or request headers containing keys.
        failure = {"error_type": type(exc).__name__}
        record("failure", **failure)
    report = {"id": case["id"], "split": case["split"], "completed": failure is None, "failure": failure, "api_calls": count,
        "plan_sha256": sha(out / "PLAN.json"), "files_sha256": {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    write(directory / "REPORT.json", report)
    print(json.dumps({k: report[k] for k in ("id", "completed", "failure", "api_calls")}), flush=True)
    return report


async def collect(out):
    from paperalpha_runtime.quadsentinel_adapter import create_api_client
    plan = frozen_plan(out)
    client = create_api_client(plan["endpoint"], os.environ.get("PAPERALPHA_API_KEY"))
    semaphore = asyncio.Semaphore(plan["concurrency"])
    async def bounded(case):
        async with semaphore:
            return await collect_case(out, case, client)
    try:
        results = await asyncio.gather(*(bounded(c) for c in plan["cases"] if c["split"] in plan["collect_splits"]))
    finally:
        await client.close()
    write(out / "COLLECTION.json", {"cases": len(results), "completed": sum(r["completed"] for r in results),
        "api_calls": sum(r["api_calls"] for r in results), "failed_ids": [r["id"] for r in results if not r["completed"]],
        "reserved_test_calls": 0, "plan_sha256": sha(out / "PLAN.json")})


def verified_data(out):
    """Reconstruct actual delivery and both cutoffs from retained public output."""
    plan = frozen_plan(out)
    from paperalpha_monitor.observable_event_learning import validated_labels
    data, checked, requests = [], 0, 0
    for case in plan["cases"]:
        if case["split"] not in plan["collect_splits"]:
            if (out / "runs" / case["id"]).exists():
                raise ValueError("reserved test was unexpectedly collected")
            continue
        directory = out / "runs" / case["id"]
        report = read(directory / "REPORT.json")
        if report["plan_sha256"] != sha(out / "PLAN.json"):
            raise ValueError("collection plan changed")
        for name, expected in report["files_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError("collection evidence changed")
            checked += 1
        if not report["completed"]:
            continue
        records = [read(p) for p in sorted(directory.glob("record_*.json"))]
        assert [r["sequence"] for r in records] == list(range(1, len(records) + 1))
        assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(records, records[1:]))
        messages = {"source": case["source"]}
        assert records[0]["kind"] == "source" and records[0]["text"] == messages["source"]
        completed = {"source": {"record_id": "record_0001.json", "sequence": 1, "text": messages["source"]}}
        replies, prefix_packets = {}, {}
        expected_stages = study.workflow(case)["nodes"][1:]
        for i, record in enumerate(records):
            if record["kind"] == "prefix":
                assert record["phase"] not in prefix_packets
                assert record["packet"] == study.prefix(case, completed, record["sequence"])
                assert record["packet"] == read(directory / (record["phase"].upper() + ".json"))
                prefix_packets[record["phase"]] = record["packet"]
            elif record["kind"] == "request":
                node = expected_stages[len(replies)]
                assert record["stage"] == node and record["request"] == study.request(case, node, messages)
                if node.startswith("publish-"):
                    assert "late" in prefix_packets
                else:
                    assert "early" in prefix_packets
                requests += 1
            elif record["kind"] == "response":
                previous = records[i - 1]
                assert previous["kind"] == "request" and previous["stage"] == record["stage"]
                reply = delivered_text(record["response"])
                assert not reply["native_refusal"]
                node = record["stage"]
                item = {"record_id": f"record_{record['sequence']:04d}.json", "sequence": record["sequence"], "text": reply["text"]}
                messages[node], replies[node] = item["text"], item
                if node.startswith("review-"):
                    completed[node] = item
        labels = study.outcome(case, replies)
        assert labels == read(directory / "OUTCOME.json") and set(prefix_packets) == {"early", "late"}
        samples = {phase: study.compile_input(packet) for phase, packet in prefix_packets.items()}
        validated_labels(samples["early"].event_dag, labels)
        validated_labels(samples["late"].event_dag, {n: labels[n] for n in ("publish-a", "publish-b")})
        data.append({"case": case, "samples": samples, "labels": labels})
    return data, {"files_verified": checked, "actual_requests_reconstructed": requests, "complete_workflows": len(data), "reserved_test_calls": 0}


def features(row, phase):
    """Public prefix-only direct joint baseline; no future observation values."""
    sample = row["samples"][phase]
    dag = sample.event_dag
    reviewers = [e for e in dag.events if e["family"] == "review"]
    observed = [dag.observations[e["id"]]["value"] for e in reviewers if e["id"] in dag.observations]
    return [1., dag.observations["source"]["value"], *reviewers[0]["cues"], int(len(reviewers) == 1),
            int(phase == "late"), float(sum(observed) / len(observed)) if observed else 0.,
            int(bool(observed) and all(observed)), int(bool(observed) and any(observed))]


def metrics(records):
    import numpy as np
    p = np.asarray([r["joint"] for r in records]); y = np.asarray([r["label"] for r in records], int)
    expected = np.eye(4)[y]
    any_p, any_y = 1 - p[:, 0], y != 0
    calls = any_p >= .5
    tp, fp, fn = int(np.sum(calls & any_y)), int(np.sum(calls & ~any_y)), int(np.sum(~calls & any_y))
    return {"n": len(y), "positives": int(any_y.sum()), "outcome_counts": np.bincount(y, minlength=4).tolist(),
        "joint_brier": float(np.mean(np.sum((p - expected) ** 2, axis=1))),
        "joint_log_loss_clipped_1e_12": float(np.mean(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))),
        "any_brier": float(np.mean((any_p - any_y) ** 2)), "both_brier": float(np.mean((p[:, 3] - (y == 3)) ** 2)),
        "tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def fit(out):
    import numpy as np
    from itertools import product
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from sklearn.linear_model import LogisticRegression
    from paperalpha_monitor.observable_event_learning import fit_event_traces
    from paperalpha_monitor.observable_event_dag import ObservableEventDAG
    plan = frozen_plan(out)
    data, verification = verified_data(out)
    if len(data) != 24:
        raise ValueError("incomplete development collection; retain failures instead of fitting a selected subset")
    train = [r for r in data if r["case"]["split"] == "train"]
    dags = [r["samples"]["early"].event_dag for r in train]
    target_names = ("publish-a", "publish-b")
    models, predictions, equivalence = {}, {}, {}
    for supervision in ("all_future", "terminal_only"):
        labels = [r["labels"] if supervision == "all_future" else {n: r["labels"][n] for n in target_names} for r in train]
        for penalty in plan["regularization"]:
            model = fit_event_traces(dags, labels, regularization=penalty, max_iter=plan["max_iter"])
            name = f"dag_{supervision}_{penalty}"
            models[name] = model.parameters()
            records, independent = [], []
            for row in data:
                for phase, sample in row["samples"].items():
                    joint = model.event_law.conditional(sample.event_dag, target_names, model.theta[:25])
                    probabilities = np.exp(joint["log_probabilities"])
                    marginal = probabilities @ joint["states"]
                    factored = np.prod(np.where(joint["states"] == 1, marginal, 1 - marginal), axis=1)
                    label = 2 * row["labels"][target_names[0]]["value"] + row["labels"][target_names[1]]["value"]
                    base = {"id": row["case"]["id"], "split": row["case"]["split"], "layout": row["case"]["layout"],
                        "phase": phase, "input_sha256": sample.sha256, "label": label}
                    policy = model.predict(sample)["probabilities"]
                    assert abs(policy["system:any"] - (1 - probabilities[0])) < 1e-12
                    records.append({**base, "joint": probabilities.tolist(), "policy": policy})
                    independent.append({**base, "joint": factored.tolist()})
            predictions[name], predictions[name + "_independent"] = records, independent
            print(json.dumps({"fit": name, "optimizer": model.optimizer}), flush=True)
            if supervision == "all_future":
                x, y = [], []
                for row in train:
                    dag = row["samples"]["early"].event_dag
                    values = {n: v["value"] for n, v in {**dag.observations, **row["labels"]}.items()}
                    for n in dag.order:
                        if n not in dag.observations:
                            x.append(model.event_law.design(dag.lookup[n], [values[p] for p in dag.parents[n]])); y.append(values[n])
                x = np.asarray(x)
                control = LogisticRegression(C=1 / (2 * penalty * len(train)), fit_intercept=False, tol=1e-10, max_iter=1000).fit(x, y)
                difference = 0.
                for row in data:
                    for sample in row["samples"].values():
                        p = model.event_law.conditional(sample.event_dag, target_names, control.coef_[0])
                        q = model.event_law.conditional(sample.event_dag, target_names, model.theta[:25])
                        difference = max(difference, float(np.max(np.abs(np.exp(p["log_probabilities"]) - np.exp(q["log_probabilities"])))))
                equivalence[name] = {"max_joint_prediction_difference": difference, "design_shape": list(x.shape),
                    "design_rank": int(np.linalg.matrix_rank(x)), "local_parent_values": sorted(set(float(v) for row in train for n in row["labels"] for v in
                        [({**row['samples']['early'].event_dag.observations, **row['labels']})[p]['value'] for p in row['samples']['early'].event_dag.parents[n]])),
                    "coefficient": control.coef_[0].tolist(), "meaning": "Same local probability model trained with an independent sklearn optimizer; not an independent algorithmic innovation."}
    # A direct joint classifier is trained separately at each available prefix.
    # Terminal targets match terminal-only BN; full-trace BN has extra labels.
    for penalty in plan["regularization"]:
        name = f"direct_joint_softmax_{penalty}"
        records, saved = [], {}
        for phase in ("early", "late"):
            x = np.asarray([features(r, phase) for r in train])
            y = np.asarray([2 * r["labels"]["publish-a"]["value"] + r["labels"]["publish-b"]["value"] for r in train])
            def objective(flat):
                coef = flat.reshape(x.shape[1], 4)
                logits = x @ coef; logs = logits - logsumexp(logits, axis=1, keepdims=True)
                grad = x.T @ (np.exp(logs) - np.eye(4)[y]) / len(y) + 2 * penalty * coef
                return float(-logs[np.arange(len(y)), y].mean() + penalty * np.sum(coef ** 2)), grad.ravel()
            opt = minimize(objective, np.zeros(x.shape[1] * 4), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            saved[phase] = {"coef": opt.x.reshape(x.shape[1], 4).tolist(), "loss": float(opt.fun), "success": bool(opt.success),
                "feature_rank": int(np.linalg.matrix_rank(x)), "train_prefix_count": len(y)}
            for row in data:
                logits = np.asarray(features(row, phase)) @ np.asarray(saved[phase]["coef"])
                label = 2 * row["labels"]["publish-a"]["value"] + row["labels"]["publish-b"]["value"]
                records.append({"id": row["case"]["id"], "split": row["case"]["split"], "layout": row["case"]["layout"], "phase": phase,
                    "joint": np.exp(logits - logsumexp(logits)).tolist(), "label": label})
        models[name], predictions[name] = saved, records
    for rule in ("preserve_unless_removed", "always_absent"):
        records = []
        for row in data:
            for phase, sample in row["samples"].items():
                dag = sample.event_dag
                bits = []
                for target in target_names:
                    parent = dag.parents[target][0]
                    value = dag.observations[parent]["value"] if parent in dag.observations else dag.observations["source"]["value"] * (1 - dag.lookup[parent]["cues"][1])
                    bits.append(0 if rule == "always_absent" else value)
                probs = [float(i == 2 * bits[0] + bits[1]) for i in range(4)]
                label = 2 * row["labels"]["publish-a"]["value"] + row["labels"]["publish-b"]["value"]
                records.append({"id": row["case"]["id"], "split": row["case"]["split"], "layout": row["case"]["layout"], "phase": phase, "joint": probs, "label": label})
        predictions[rule] = records
    for name, value in models.items():
        write(out / "fit/models" / (name + ".json"), value)
    write(out / "fit/PREDICTIONS.json", predictions)
    report = {"metrics": {name: {f"{split}/{phase}": metrics([r for r in records if r["split"] == split and r["phase"] == phase])
        for split in ("train", "validation") for phase in ("early", "late")} for name, records in predictions.items()},
        "local_logistic_equivalence": equivalence, "collection_verification": verification,
        "optimizer": {n: v["optimizer"] for n, v in models.items() if n.startswith("dag_")},
        "models_sha256": {p.name: sha(p) for p in sorted((out / "fit/models").glob("*.json"))},
        "predictions_sha256": sha(out / "fit/PREDICTIONS.json"), "plan_sha256": sha(out / "PLAN.json"),
        "new_default_selected": False, "reserved_test_calls": 0, "old_heldout_read": False,
        "scope": "Controlled 12/12 development, prefix-correct offline fits. Not held-out superiority or novel causal learning."}
    write(out / "fit/REPORT.json", report)
    print(json.dumps({"fitted_models": len(models), "methods": len(predictions), "local_equivalence": {n: e["max_joint_prediction_difference"] for n, e in equivalence.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "collect", "fit", "verify-collection"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--numerical-python", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "prepare": prepare(out, args.numerical_python)
    elif args.mode == "collect": asyncio.run(collect(out))
    elif args.mode == "fit": fit(out)
    else: print(json.dumps(verified_data(out)[1]))
