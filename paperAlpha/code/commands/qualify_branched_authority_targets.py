"""Post-acquisition scope audit; original online predictions stay unchanged."""
import argparse
import json
from pathlib import Path

from collect_branched_authority_api import load, MODELS
from harnessaudit_cohort import ROOT, sha, write_new
from verify_branched_authority_api import lines
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor
from paperalpha_monitor.authority_risk_queries import query_reference_risk, qualify_two_action_endpoint
from paperalpha_runtime.branched_authority_pilot import monitor_configuration


def qualify(run, out):
    if out.exists():
        raise ValueError("new derived output required")
    plan = load(run / "PLAN.json")
    models = [json.loads((run / name).read_text(encoding="utf-8")) for name in MODELS]
    prefixes, endpoints, prior_pairs = [], [], []
    for case in plan["cases"]:
        folder = run / "runs" / case["id"]
        manifest = json.loads((folder / "RUN.json").read_text(encoding="utf-8"))
        monitor = BranchedAuthorityMonitor(monitor_configuration(case), *models)
        last = {}
        for event in lines(folder / "monitor_events.jsonl"):
            output = monitor.push(event)
            for view, data in output["views"].items():
                for row in data["dispatches"]:
                    graph = monitor.graphs[view]
                    query = query_reference_risk(graph, monitor.reference.theta[monitor.reference.prior_slice], at_least=2)
                    assert query["model_probability"] == row["reference_two_failures"]
                    qualified = qualify_two_action_endpoint(len(graph.proofs), row["model_point_probability"],
                        row["graph_logistic_two_failures"], run_completed=False)
                    assert qualified["scores"] is None  # An entry callback is not episode completion.
                    prefixes.append({"case_id": case["id"], "view": view, "sequence": event["sequence"],
                        "reference_query": query, "endpoint_heads": qualified})
                    last[view] = row
        for view, row in last.items():
            graph = monitor.graphs[view]
            qualified = qualify_two_action_endpoint(len(graph.proofs), row["model_point_probability"],
                row["graph_logistic_two_failures"], run_completed=manifest["status"] == "completed")
            endpoints.append({"case_id": case["id"], "view": view, "endpoint_heads": qualified})
        if not last:
            endpoints.append({"case_id": case["id"], "view": "both", "endpoint_heads": None,
                "reason": "No action entries; refusal/no-action remains in the cohort, not scored as monitoring success."})
    # Outcome-distinct cases with equal monitor features/scores must remain
    # distinguishable in the evidence report, never split by hidden treatment.
    for layout, ids in (("shared", ("branch-01", "branch-03", "branch-04")),
                        ("separate", ("branch-05", "branch-07", "branch-08"))):
        scores = [next(r for r in endpoints if r["case_id"] == identity and r["view"] == "root_updates_missing")
            ["endpoint_heads"]["raw_diagnostic_scores"] for identity in ids]
        assert all(s == scores[0] for s in scores)
        prior_pairs.append({"layout": layout, "cases": ids, "same_partial_scores": scores[0],
            "meaning": "These different true outcomes have equal current monitor scores; missing facts were not recovered."})
    out.mkdir(parents=True)
    write_new(out / "prefix_scope.json", prefixes)
    write_new(out / "endpoint_scope.json", endpoints)
    write_new(out / "same_score_opposite_outcome.json", prior_pairs)
    for path in (Path(__file__), ROOT / 'code/src/paperalpha_monitor/authority_risk_queries.py', ROOT / 'code/tests/test_authority_risk_queries.py'):
        with (out / path.name).open("xb") as handle:
            handle.write(path.read_bytes())
    report = {"source_report_sha256": sha(run / "REPORT.json"), "prefix_view_count": len(prefixes),
        "prefixes_with_less_than_two_actions_and_nonzero_raw_system_score": sum(
            r["reference_query"]["target"]["action_count"] < 2 and r["endpoint_heads"]["raw_diagnostic_scores"]["system_bn"] > 0 for r in prefixes),
        "qualified_prefix_system_forecasts": 0,
        "scope": "Post-acquisition qualification only, not the original online output; neither scores nor labels changed.",
        "artifacts_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_new(out / "REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run.resolve(), args.out.resolve())
