"""Separate scalar/count-polynomial probability and raw cue-binding audit."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from itertools import product

import joblib
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, write, sha, frozen_plan, load_data, metric
from run_forecast_alarm_study import bind_partition, split_templates
from run_task_cue_message_study import load_views
from verify_forecast_alarm_study import independent_threshold, audit_decisions
from paperalpha_runtime.canonical import digest
from paperalpha_monitor.message_hurdle import MessageHurdleWorkflowRiskBN
from paperalpha_monitor.task_cue_message import TaskCueMessageWorkflowRiskBN, SemanticArrivalWorkflowRiskBN


def independent_features(sample, phase, with_task):
    d = sample.event_dag; n = len(sample.action_ids); counts = np.zeros(8); counts[0] = n; hist = np.zeros(72)
    for j in range(n):
        stem = f"value-{j}:"
        v = d.observations[stem + "vault"]["value"]; c = d.observations[stem + "prepare"]["value"]
        w = d.observations[stem + "work"]["value"] if phase == "late" else 0
        counts[1:] += [v, c, v*c, w, v*w, c*w, v*c*w]
        a, b = [1 if d.lookup[stem + p]["cues"][0] else 2 if d.lookup[stem + p]["cues"][1] else 0 for p in ("work", "publish")]
        hist[(((v*2+c)*2+w)*3+a)*3+b] += 1
    return np.r_[counts, hist] if with_task else counts


def independent_probability(saved, sample, phase):
    t = dict(zip(saved["names"], saved["theta"])); d = sample.event_dag; n = len(sample.action_ids)
    assert d.workflow["edges"] == [["vault", "prepare"], ["vault", "work"], ["prepare", "work"], ["work", "publish"]]
    v = np.array([d.observations[f"value-{j}:vault"]["value"] for j in range(n)])
    c = np.array([d.observations[f"value-{j}:prepare"]["value"] for j in range(n)])
    cues = {p: np.array([d.lookup[f"value-{j}:" + p]["cues"] for j in range(n)]) for p in ("work", "publish")}
    def message(p, mean, any_parent, supported=0., opposed=0.):
        return expit(t[f"message:{p}:bias"] + t[f"message:{p}:log_group_size"] * np.log1p(n)
            + t[f"message:{p}:parent_fraction"] * mean + t[f"message:{p}:any_parent"] * any_parent
            + t.get(f"message:{p}:cue_support_fraction", 0.) * cues[p][:, 0].mean()
            + t.get(f"message:{p}:cue_opposition_fraction", 0.) * cues[p][:, 1].mean()
            + t.get(f"message:{p}:supported_arrival_fraction", 0.) * supported
            + t.get(f"message:{p}:opposed_arrival_fraction", 0.) * opposed)
    def local(p, mean, all_parents):
        return expit(t[f"event:{p}:bias"] + t[f"event:{p}:visible_support"] * cues[p][:, 0]
            + t[f"event:{p}:visible_opposition"] * cues[p][:, 1]
            + t[f"event:{p}:parent_mean"] * mean + t[f"event:{p}:multiple_parents_all"] * all_parents)
    counts = np.arange(n+1); q_final = message("publish", counts/n, (counts > 0).astype(float))
    if phase == "early":
        p = local("work", (v+c)/2, v*c)
        q = float(message("work", (v.sum()+c.sum())/(2*n), float(v.any() or c.any()),
            np.dot(cues["work"][:,0], (v+c)/2)/n, np.dot(cues["work"][:,1], (v+c)/2)/n))
        if "message:publish:supported_arrival_fraction" in t:
            # Independent exhaustive binary worlds. Unlike the production DAG
            # factors, this spells out all future Worker messages explicitly.
            states = np.asarray(list(product((0.,1.), repeat=n)))
            mass = np.prod(np.where(states == 1, p, 1-p), axis=1)
            mass[1:] *= q/mass[1:].sum(); mass[0] = 1-q
            q_output = message("publish", states.mean(axis=1), states.any(axis=1).astype(float),
                states @ cues["publish"][:,0]/n, states @ cues["publish"][:,1]/n)
            return float(mass @ q_output)
        law = np.array([1.])
        for item in p: law = np.convolve(law, [1-item, item])
        law[1:] *= q/law[1:].sum(); law[0] = 1-q
        return float(law @ q_final)
    w = np.asarray([d.observations[f"value-{j}:work"]["value"] for j in range(n)])
    return float(message("publish", w.mean(), float(w.any()), np.dot(w, cues["publish"][:,0])/n, np.dot(w, cues["publish"][:,1])/n))


def audit_cue_binding(original, updated, annotation):
    lookup = {f["field_name"]: f for f in annotation["fields"]}; checked = 0
    assert original.authority.specification() == updated.authority.specification()
    assert original.queries == updated.queries and original.event_dag.observations == updated.event_dag.observations
    for q in updated.queries.values():
        stem = q["event_id"].rsplit(":", 1)[0]
        for phase, column in (("work", "internal_use"), ("publish", "outward_form")):
            votes = []
            for name in q["field_literals"]:
                category = lookup[name][column]
                votes.append((1,0) if category in ("needed", "verbatim") else (0,0) if category == "unclear" else (0,1))
            expected = list(votes[0]) if len(set(votes)) == 1 else [0,0]
            actual = updated.event_dag.lookup[stem + ":" + phase]
            assert actual["cues"] == expected
            source = json.loads(actual["cue_evidence"]["text"])
            assert source["hypotheses"] == annotation and source["grants_permission"] is False
            assert actual["cue_evidence"]["sequence"] == 1
            checked += 1
    return checked


def run(out, destination):
    plan = frozen_plan(out); config = plan["config"]; report = read(out / "REPORT.json")
    assert sha(out / "PLAN.json") == report["plan_sha256"]
    fit = read(out / "FIT.json"); cal = read(out / "CALIBRATION.json"); start = read(out / "VALIDATION_START.json")
    assert sha(out / "CALIBRATION.json") == report["calibration_sha256"] == start["calibration_sha256"]
    assert fit["saved_at_utc"] <= cal["saved_at_utc"] <= start["started_at_utc"] <= report["completed_at_utc"]
    for name, expected in fit["model_file_sha256"].items(): assert sha(out / "models" / name) == expected
    for key in ("prediction_sha256", "decision_sha256", *(["raw_prediction_sha256"] if config.get("probability_digits") is not None else [])):
        for name, expected in report[key].items(): assert sha(out / name) == expected
    data, projection = load_data(out, verbose=False); assert projection == read(out / "PROJECTION.json")
    views, task_projection = load_views(out, data); assert task_projection == read(out / "TASK_PROJECTION.json")
    parts = bind_partition(data, plan)
    assignment, ordered = split_templates(read(out / "inputs/train.json"), config)
    assert ordered == plan["ordered_original_train_templates"]
    assert all(plan["partition_by_id"][i] == p for i,p in assignment.items())
    assert set(fit["fit_ids"]) == {r["id"] for r in parts["fit"]}
    assert set(cal["calibration_ids"]) == {r["id"] for r in parts["calibration"]}
    negatives = {r["id"] for r in parts["calibration"] if not r["y"]}
    assert negatives == set(cal["calibration_negative_ids"])
    contexts = {r["id"]: r["context_sha256"] for r in task_projection}; cue_count = 0
    for row in data:
        annotation = read(out / "extraction/classifications" / (contexts[row["id"]] + ".json"))["annotation"]
        for phase in config["phases"]: cue_count += audit_cue_binding(row[phase], views["task"][row["id"]][phase], annotation)
    max_error, gradient_error, probability_count, gradients, thresholds, decisions = 0., 0., 0, 0, 0, 0
    quantization_error, equivalent_inputs = 0., 0
    for name, meta in fit["models"].items():
        saved = read(out / "models" / (name + ".json")); assert digest(saved) == meta["fingerprint"]
        if config.get("probability_digits") is not None:
            assert saved["score_transform"] == {"mode": "round_nearest_even_decimal", "probability_digits": config["probability_digits"]}
        estimators = {p: joblib.load(out / "models" / (name + "_" + p + ".joblib")) for p in config["phases"]} if meta["kind"] != "bn" else None
        neg_episodes = [e for e in read(out / "predictions/calibration" / (name + ".json")) if e["episode_id"] in negatives]
        for rule, entry in cal["models"][name].items():
            policy = entry["policy"]
            assert policy["model_fingerprint"] == meta["fingerprint"] and policy["target_id"] == config["target_id"]
            for phase in config["phases"]:
                values = [max(s["probability"] for s in e["prefixes"] if rule == "episode_max" or s["phase"] == phase) for e in neg_episodes]
                threshold = .5 if rule == "fixed_half" else independent_threshold(values, config["total_empirical_fpr_budget"] if rule == "episode_max" else config["fixed_phase_budgets"][phase])
                assert policy["thresholds"][phase] == threshold; thresholds += 1
        signature_scores = {}
        for partition in ("calibration", "validation"):
            episodes = read(out / "predictions" / partition / (name + ".json")); stored = read(out / "decisions" / partition / (name + ".json"))
            raw_episodes = read(out / "raw_predictions" / partition / (name + ".json")) if config.get("probability_digits") is not None else episodes
            assert len(raw_episodes) == len(episodes)
            assert {e["episode_id"] for e in episodes} == {r["id"] for r in parts[partition]}
            arrays = {p: estimators[p].predict_proba(np.array([independent_features(views[meta["view"]][e["episode_id"]][p], p, meta["view"] == "task") for e in episodes]))[:,1]
                for p in config["phases"]} if estimators else None
            for i, e in enumerate(episodes):
                assert e["model_fingerprint"] == meta["fingerprint"] and e["target_id"] == config["target_id"]
                raw_e = raw_episodes[i]
                assert {k:v for k,v in raw_e.items() if k != "prefixes"} == {k:v for k,v in e.items() if k != "prefixes"}
                assert [s["phase"] for s in raw_e["prefixes"]] == [s["phase"] for s in e["prefixes"]] == config["phases"]
                row = views[meta["view"]][e["episode_id"]]
                for prefix in e["prefixes"]:
                    phase = prefix["phase"]; sample = row[phase]
                    assert prefix["input_sha256"] == sample.sha256 and prefix["sequence"] == sample.event_dag.prefix_sequence
                    p = float(arrays[phase][i]) if arrays is not None else independent_probability(saved, sample, phase)
                    raw_prefix = next(s for s in raw_e["prefixes"] if s["phase"] == phase)
                    assert {k:v for k,v in raw_prefix.items() if k != "probability"} == {k:v for k,v in prefix.items() if k != "probability"}
                    raw_p = raw_prefix["probability"]
                    max_error = max(max_error, abs(p-raw_p)); probability_count += 1
                    if config.get("probability_digits") is not None:
                        assert prefix["probability"] == round(raw_p, config["probability_digits"])
                        quantization_error = max(quantization_error, abs(raw_p-prefix["probability"]))
                        signature = (phase, *independent_features(sample, phase, meta["view"] != "counts"))
                        if signature in signature_scores:
                            assert signature_scores[signature] == prefix["probability"], "mathematically equal feature inputs split after score resolution"
                            equivalent_inputs += 1
                        signature_scores[signature] = prefix["probability"]
                    if meta["kind"] == "bn" and partition == "validation" and i == 0:
                        cls = MessageHurdleWorkflowRiskBN if meta["view"] == "local" else SemanticArrivalWorkflowRiskBN if meta["view"] == "arrival" else TaskCueMessageWorkflowRiskBN
                        model = cls.restore(saved); actual = model.predict_known_authorization_union(sample, derivatives=True)
                        for k in range(model.event_count):
                            value = saved["theta"][k]
                            saved["theta"][k] = value+1e-5; plus = independent_probability(saved, sample, phase)
                            saved["theta"][k] = value-1e-5; minus = independent_probability(saved, sample, phase)
                            saved["theta"][k] = value
                            gradient_error = max(gradient_error, abs((plus-minus)/2e-5-actual["gradients"]["system:any"][k])); gradients += 1
            labels = {r["id"]: r["y"] for r in parts[partition]}
            for phase in config["phases"]:
                expected = metric([labels[e["episode_id"]] for e in episodes], [next(s["probability"] for s in e["prefixes"] if s["phase"] == phase) for e in episodes])
                assert expected == report["phase_probability_metrics"][partition + "/" + name][phase]
            for rule, entry in cal["models"][name].items():
                summary, n = audit_decisions(episodes, labels, entry["policy"], stored[rule]); decisions += n
                assert summary == report["metrics"][partition + "/" + name][rule]
                if partition == "calibration" and rule != "fixed_half": assert summary["any_alarm_fpr"] <= .1 + 1e-15
    assert max_error < 1e-10 and gradient_error < 2e-8
    if config.get("probability_digits") is not None: assert quantization_error <= .5*10**(-config["probability_digits"]) + 1e-15
    destination.mkdir(parents=True, exist_ok=False); shutil.copyfile(Path(__file__), destination / "AUDITOR.py")
    result = {"reconstructed_flows": len(data), "bound_cues_checked": cue_count, "probabilities_recomputed": probability_count,
        "maximum_probability_error": max_error, "gradient_directions": gradients, "maximum_gradient_error": gradient_error,
        "thresholds_recomputed": thresholds, "stream_decisions_recomputed": decisions, "new_fit": False, "new_api_calls": 0,
        "semantic_truth_verified": False, "plan_sha256": sha(out / "PLAN.json"), "report_sha256": sha(out / "REPORT.json"), "auditor_sha256": sha(destination / "AUDITOR.py")}
    if config.get("probability_digits") is not None:
        result.update(probability_digits=config["probability_digits"], maximum_operational_quantization_error=quantization_error,
            equivalent_feature_inputs_checked=equivalent_inputs, all_checked_equivalent_scores_tied=True)
    write(destination / "REPORT.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(); run(args.out.resolve(), args.destination.resolve())
