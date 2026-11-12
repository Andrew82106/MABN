"""Fit-only check: the observable message OR gate is ordinary logistic.

This control does not claim that a task-level classifier reproduces early
joint inference or the within-message field distribution. No validation or
calibration outcomes are read; no primary model or result is replaced.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, write, sha
from paperalpha_monitor.privacy_flow_data import protected_groups
from paperalpha_runtime.task_field_semantics import alias_cue


def audit(out, destination):
    plan, fit = read(out / "PLAN.json"), read(out / "FIT.json")
    for name in ("train.json", "provenance.json"):
        assert sha(out / "inputs" / name) == plan["inputs_sha256"][name]
    provenance = {r["id"]: r for r in read(out / "inputs/provenance.json")}
    originals = {r["id"]: r for r in read(out / "inputs/train.json")}
    projections = {r["id"]: r for r in read(out / "TASK_PROJECTION.json")}
    designs = {(view, phase): [] for view in ("counts", "task", "arrival") for phase in ("work", "publish")}
    labels = {p: [] for p in ("work", "publish")}
    for identity in fit["fit_ids"]:
        assert plan["partition_by_id"][identity] == "fit"
        source = provenance[identity]; path = Path(plan["upstream"]) / source["source_path"]
        assert sha(path) == source["source_sha256"]
        raw = read(path); initial = raw["input"]; groups = protected_groups(initial); n = len(groups)
        actual = [json.dumps(initial["vault"], indent=2).lower(), raw["channel_messages"][1]["content"].lower(), raw["channel_messages"][2]["content"].lower()]
        v, c, w = [np.asarray([int(g["literal"] in text) for g in groups], float) for text in actual]
        labels["work"].append(int(w.any()))
        final = int(any(g["literal"] in raw["channel_messages"][4]["content"].lower() for g in groups))
        assert final == originals[identity]["y"]; labels["publish"].append(final)
        name = projections[identity]["context_sha256"] + ".json"
        annotation = read(out / "extraction/classifications" / name)["annotation"]
        mapping = {f["field_name"]: f for f in annotation["fields"]}
        for phase, parent, any_parent in (("work", (v+c)/2, bool(v.any() or c.any())), ("publish", w, bool(w.any()))):
            cues = np.asarray([alias_cue([mapping[f] for f in g["field_names"]], phase) for g in groups], float)
            base = [1., np.log1p(n), parent.mean(), float(any_parent)]
            designs["counts", phase].append([*base, 0., 0.])
            designs["task", phase].append([*base, *cues.mean(axis=0)])
            designs["arrival", phase].append([*base, *cues.mean(axis=0), *(cues*parent[:,None]).mean(axis=0)])
    result = {"scope": "Fit-only gate/logistic equivalence; not a new task-level MAS baseline or a validation performance gain.",
        "fit_flows": len(fit["fit_ids"]), "plan_sha256": sha(out / "PLAN.json"), "fit_sha256": sha(out / "FIT.json"), "controls": {}}
    for name, meta in fit["models"].items():
        if meta["kind"] != "bn": continue
        assert sha(out / "models" / (name + ".json")) == fit["model_file_sha256"][name + ".json"]
        saved = read(out / "models" / (name + ".json")); penalty = saved["regularization"]
        for phase in ("work", "publish"):
            keys = [k for k in saved["names"] if k.startswith("message:" + phase + ":")]
            theta = np.array([saved["theta"][saved["names"].index(k)] for k in keys])
            x = np.asarray(designs[meta["view"], phase]); y = np.array(labels[phase]); n = len(y)
            assert x.shape[1] == len(theta)
            control = LogisticRegression(C=1/(2*penalty*n), fit_intercept=False, max_iter=3000, tol=1e-12).fit(x,y)
            alternative = control.coef_[0]
            def objective(t):
                z = x@t
                return float(np.mean(np.logaddexp(0,z)-y*z) + penalty*(t@t))
            result["controls"][name + "/" + phase] = {"columns": keys, "input_rank": int(np.linalg.matrix_rank(x)),
                "bn_gate_objective": objective(theta), "logistic_objective": objective(alternative),
                "maximum_training_logit_difference": float(np.max(np.abs(x@(theta-alternative)))),
                "maximum_absolute_logistic_coefficient": float(np.max(np.abs(alternative))),
                "bn_coefficient": theta.tolist(), "logistic_coefficient": alternative.tolist(), "iterations": int(control.n_iter_[0])}
    destination.mkdir(parents=True, exist_ok=False); write(destination / "REPORT.json", result)
    print(json.dumps({"controls": len(result["controls"]), "fit_flows": result["fit_flows"],
        "maximum_gate_logit_difference": max(r["maximum_training_logit_difference"] for r in result["controls"].values()),
        "maximum_gate_objective_difference": max(abs(r["bn_gate_objective"]-r["logistic_objective"]) for r in result["controls"].values())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(); audit(args.out.resolve(), args.destination.resolve())
