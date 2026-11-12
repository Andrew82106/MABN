"""Post-hoc shortcut diagnostics; original v8 experiment is not overwritten.

These controls were added AFTER discovering every high positive belongs to the
coordinator. They are diagnostic, not a prospectively registered comparison.
Features are exact physical actor/tool names already visible at dispatch. They
use training labels only, no benchmark policy, workflow, outcome or text judge.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from train_typed_action_development import load_data, metrics, sha, write_new, now


def run(source, output):
    if output.exists():
        raise ValueError("new diagnostic output directory required")
    output.mkdir(parents=True)
    plan, data = load_data(source)
    write_new(output / "PLAN.json", {"created_at": now(), "source_plan_sha256": sha(source / "PLAN.json"),
        "methods": ["actor_only_counts", "actor_tool_counts"], "prior": "add-one smoothing, absent-key pooled training fallback",
        "post_hoc": True, "reason": "All five high-severity positives were found to use the same physical actor.",
        "threshold": .5, "folds": plan["folds"], "script_sha256": sha(Path(__file__))})
    bindings = {key: {a["action_id"]: a["binding"] for a in row[0].steps[-1].base.actions} for key, row in data.items()}
    all_rows, fits = [], []
    for fold in plan["folds"]:
        for method in ("actor_only_counts", "actor_tool_counts"):
            def cell(binding):
                return (binding["actor"],) if method == "actor_only_counts" else (binding["actor"], binding["tool"])
            groups = defaultdict(list)
            for key in fold["train"]:
                for aid, target in data[key][1].actions.items():
                    groups[cell(bindings[key][aid])].append(target)
            pooled = [label for labels in groups.values() for label in labels]
            def probability(labels):
                nonhigh = [v for v in labels if v.forbidden == 0]
                high = (sum(v.forbidden for v in labels) + 1) / (len(labels) + 2)
                low = (sum(v.unnecessary for v in nonhigh) + 1) / (len(nonhigh) + 2)
                return {"forbidden": high, "unnecessary_given_not_forbidden": low,
                        "unnecessary": (1 - high) * low, "no_tool_tier_violation": (1 - high) * (1 - low)}
            table = {k: probability(v) for k, v in groups.items()}
            fits.append({"fold": fold["evaluation_task"], "method": method,
                "table": [{"key": list(k), "training_count": len(groups[k]), "probabilities": v} for k, v in table.items()],
                "fallback": probability(pooled)})
            for key in fold["evaluate"]:
                for aid, target in data[key][1].actions.items():
                    lookup = cell(bindings[key][aid])
                    all_rows.append({"fold": fold["evaluation_task"], "run_id": key, "action_id": aid, "method": method,
                        "key_seen_in_training": lookup in table, "prediction": table.get(lookup, probability(pooled)),
                        "targets": {"forbidden": target.forbidden, "unnecessary": target.unnecessary}})
    write_new(output / "models.json", fits)
    write_new(output / "predictions.json", all_rows)
    report = {"post_hoc_diagnostic": True, "plan_sha256": sha(output / "PLAN.json"),
        "metrics": {m: metrics([r for r in all_rows if r["method"] == m]) for m in ("actor_only_counts", "actor_tool_counts")},
        "meaning": "Actor identity memorization can detect a dataset shortcut; no claim of general safety or cross-team generalization.",
        "files_sha256": {name: sha(output / name) for name in ("models.json", "predictions.json")}}
    write_new(output / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.out.resolve())
