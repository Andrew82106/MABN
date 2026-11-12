"""Post-hoc numerical diagnostic; preserves all frozen models and decisions."""
from pathlib import Path
import argparse
from collections import defaultdict
import json
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from run_agentleak_event_dag import read, sha, write


def audit(out, destination):
    report, fit, cal = (read(out / name) for name in ("REPORT.json", "FIT.json", "CALIBRATION.json"))
    plan = read(out / "PLAN.json")
    for name, expected in plan["inputs_sha256"].items(): assert sha(out / "inputs" / name) == expected
    for name, expected in fit["model_file_sha256"].items(): assert sha(out / "models" / name) == expected
    for name, expected in report["prediction_sha256"].items(): assert sha(out / name) == expected
    original = {r["id"]: r for split in ("train", "validation") for r in read(out / "inputs" / (split + ".json"))}
    context_ids = {r["id"]: r["context_sha256"] for r in read(out / "TASK_PROJECTION.json")}
    classifications = {c: {f["field_name"]: f["outward_form"] for f in read(out / "extraction/classifications" / (c + ".json"))["annotation"]["fields"]}
        for c in set(context_ids.values())}
    results = {}
    for name, meta in fit["models"].items():
        if meta["kind"] != "bn": continue
        saved = read(out / "models" / (name + ".json")); t = dict(zip(saved["names"], saved["theta"]))
        for partition in ("calibration", "validation"):
            episodes = read(out / "predictions" / partition / (name + ".json"))
            buckets = defaultdict(list); actual, canonical, labels = [], [], []
            for e in episodes:
                row = original[e["episode_id"]]; groups = row["prefixes"]["2"]["groups"]
                n, w = len(groups), sum(g["worker_present"] for g in groups); support, opposition = 0, 0
                if meta["view"] == "task":
                    mapping = classifications[context_ids[row["id"]]]
                    for g in groups:
                        votes = {1 if mapping[f] == "verbatim" else 0 if mapping[f] == "unclear" else 2 for f in g["field_names"]}
                        support += votes == {1}; opposition += votes == {2}
                value = float(expit(t["message:publish:bias"] + t["message:publish:log_group_size"]*np.log1p(n)
                    + t["message:publish:parent_fraction"]*w/n + t["message:publish:any_parent"]*(w > 0)
                    + t.get("message:publish:cue_support_fraction", 0)*support/n
                    + t.get("message:publish:cue_opposition_fraction", 0)*opposition/n))
                p = next(s["probability"] for s in e["prefixes"] if s["phase"] == "late")
                buckets[n,w,support,opposition].append(p); actual.append(p); canonical.append(value); labels.append(row["y"])
            spans = [max(values)-min(values) for values in buckets.values()]
            splits = {}
            for rule, entry in cal["models"][name].items():
                threshold = entry["policy"]["thresholds"]["late"]
                splits[rule] = sum(min(values) < threshold <= max(values) for values in buckets.values())
            results[partition + "/" + name] = {"flows": len(episodes), "mathematically_distinct_input_groups": len(buckets),
                "groups_with_floating_point_spread": sum(s > 0 for s in spans), "maximum_within_group_spread": max(spans),
                "maximum_difference_from_direct_gate": float(np.max(np.abs(np.array(actual)-canonical))),
                "auroc_stored": float(roc_auc_score(labels, actual)), "auroc_direct_gate": float(roc_auc_score(labels, canonical)),
                "late_threshold_splits_within_equal_gate_input": splits}
    destination.mkdir(parents=True, exist_ok=False)
    result = {"model_report_sha256": sha(out / "REPORT.json"), "results": results,
        "scope": "Post-hoc numerical tie diagnostic only. No refit, threshold replacement, altered saved scores or new performance selection. Direct gate is valid only for the checked fully observed-parent, whole-message, known-authorization late query."}
    write(destination / "REPORT.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(); audit(args.out.resolve(), args.destination.resolve())
