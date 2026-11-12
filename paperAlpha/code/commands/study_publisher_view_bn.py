"""Freeze dual-view mechanism learning and replay prior API input snapshots.

No new API calls, no fitting on the viewed real cohort, no claimed accuracy gain.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import sha, write, now
from publisher_view_fixtures import fixture
from scalar_binary_graph import evaluate
from paperalpha_monitor.policy_gated_privacy import PolicyGatedPrivacyBN
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN, PublisherInput, ViewSeparatedRelease
from paperalpha_monitor.scoped_privacy_monitor import ScopedPrivacyMonitor, compile_observation

SOURCES = ('code/commands/study_publisher_view_bn.py', 'code/commands/verify_publisher_view_bn.py', 'code/commands/publisher_view_fixtures.py',
    'code/commands/collect_privacy_transfer_api.py', 'code/commands/scalar_binary_graph.py', 'code/tests/test_publisher_view_bn.py',
    'code/src/paperalpha_runtime/__init__.py', 'code/src/paperalpha_runtime/canonical.py', 'code/src/paperalpha_runtime/privacy_transfer.py',
    'code/src/paperalpha_runtime/scoped_privacy_api.py', 'code/src/paperalpha_monitor/publisher_view_bn.py',
    'code/src/paperalpha_monitor/policy_gated_privacy.py', 'knowledge/risks/risk_catalog.json')


def mechanism_rows():
    rows = []
    for visible, probability in (("supported", .8), ("contradicted", .1), ("missing", .4)):
        for wi in range(4):
            w = [wi & 1, (wi >> 1) & 1]
            for fi in range(4):
                f = [fi & 1, (fi >> 1) & 1]
                weight = .25 * np.prod([probability if bit else 1 - probability for bit in f])
                rows.append({"fixture": {"current": "revoked", "visible": visible, "w": w}, "f": f, "a": [1, 1], "weight": float(weight)})
    return rows


def training_data(rows):
    return ([fixture(**r["fixture"]) for r in rows], [r["f"] for r in rows], [r["a"] for r in rows], [r["weight"] for r in rows])


def zero_coeff_model(p=.4):
    model = PublisherViewPrivacyBN(regularization=0.)
    model.theta[:] = 0
    model.theta[model.presence.d:model.presence.d + 4] = logit(p)
    return model


def replay(parent, out):
    """Reconstruct at final REQUEST availability, later than old monitor cutoff."""
    collection = json.loads((parent / "COLLECTION.json").read_text(encoding="utf-8"))
    for name, expected in collection["artifacts_sha256"].items():
        assert sha(parent / name) == expected, name
    events = [json.loads(line) for line in (parent / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    old = ScopedPrivacyMonitor(parent / "bundle")
    records, regression_errors, graph_errors, cue_rows = [], [], [], []
    for event in events:
        if event["kind"] != "prediction":
            continue
        request = next(e for e in events if e["case_id"] == event["case_id"] and e["kind"] == "api_request" and e["stage"] == "coordinator_finalize")
        response = next(e for e in events if e["kind"] == "api_response" and e["request_seq"] == request["seq"])
        assert event["seq"] < request["seq"] < response["seq"]
        payload = json.loads(request["request"]["input"][0]["content"])
        assert payload["earlier_agent_message"] == event["observation"]["worker_text"]
        source = PublisherInput.capture(payload, transport_observer="api-request-journal", captured_journal_sequence=request["seq"], prefix_journal_sequence=request["seq"])
        sample = ViewSeparatedRelease(compile_observation(event["observation"]), source, trusted_transport_observers=["api-request-journal"])
        cue_rows.extend(sample.visible_features.tolist())
        old_replay = old.score(event["observation"])
        # This refactor retains original model results and complete output data.
        assert json.loads(json.dumps(old_replay)) == event["result"]
        models = []
        for name, old_model in old.models(.5).items():
            new = PublisherViewPrivacyBN(old_model.presence, regularization=old_model.regularization)
            new.theta[:new.n + 3] = old_model.theta[:old_model.n + 3]
            results = []
            for k in (1, 2):
                result = new.predict(sample, at_least=k)
                old_probability = event["result"]["results_by_assumed_prior"]["0.5"]["scores"][str(k)][name]
                regression_errors.append(abs(result["model_probability"] - old_probability))
                graph = new.probability_graph(sample, at_least=k)
                graph_errors.append(abs(evaluate(graph) - result["model_probability"]))
                results.append({"target": k, "result": result, "graph": graph})
            models.append({"name": name, "parameters": new.parameters(), "results": results})
        records.append({"case_id": event["case_id"], "view": event["view"], "old_prediction_sequence": event["seq"],
            "new_input_available_sequence": request["seq"], "response_sequence": response["seq"], "observation": event["observation"],
            "publisher_input": source.__dict__, "models": models})
    write(out / "replayed_inputs.json", records)
    design = np.c_[np.ones(len(cue_rows)), np.asarray(cue_rows)]
    return {"input_views": len(records), "executable_graphs": len(graph_errors), "max_graph_error": max(graph_errors),
        "max_zero_coefficient_old_score_error": max(regression_errors), "old_complete_score_records_exact": True,
        "visible_status_counts": {"supported": int(np.sum(design[:, 1])), "contradicted": int(np.sum(design[:, 2])),
            "unknown": int(np.sum(design[:, 1:].sum(axis=1) == 0))}, "visible_design_rank_with_intercept": int(np.linalg.matrix_rank(design)),
        "visible_coefficients_identified_from_real_cohort": False, "new_real_scores_were_online": False,
        "scope": "Offline reconstruction at actual final request availability; new coefficients ZERO, no refit or new performance claim"}


def run(parent, out):
    if out.exists():
        raise ValueError("new study directory required")
    source_parent = parent / "source_snapshot"
    files = {p.relative_to(source_parent).as_posix(): p for p in (source_parent / 'code/src/paperalpha_monitor').glob("*.py")}
    files.update({s: ROOT / s for s in SOURCES})
    for name, source in files.items():
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write(out / "PLAN.json", {"created_at": now(), "parent_api_run": str(parent), "parent_collection_sha256": sha(parent / "COLLECTION.json"),
        "source_sha256": {s: sha(out / "source_snapshot" / s) for s in files},
        "scope": "Analytic frequency learning + offline real-input integration; not an independent performance trial",
        "model": "44 parameters: 39 presence, 3 current authority priors, 2 publisher-visible cue coefficients",
        "real_fitting": False, "api_calls": 0, "old_results_untouched": True})
    rows = mechanism_rows()
    write(out / "frequency_rows.json", rows)
    samples, f, a, weights = training_data(rows)
    partial = zero_coeff_model()
    partial.fit(samples, f, a, weights=weights, trainable_names=list(partial.EMISSION_NAMES), max_iter=300)
    expected = [float(logit(.8) - logit(.4)), float(logit(.1) - logit(.4))]
    assert partial.optimizer["success"] and np.max(abs(partial.theta[-2:] - expected)) < 1e-5
    write(out / "visible_coefficients_fit.json", partial.parameters())
    control_rows = [{"fixture": {"current": "revoked", "visible": v, "w": w}, "f": f, "a": [1, 1], "weight": 1.}
        for v, w, f in (("supported", [1, 1], [1, 1]), ("contradicted", [1, 0], [0, 0]),
            ("unresolved", [0, 1], [1, 0]), ("missing", [0, 0], [0, 1]))]
    write(out / "joint_control_rows.json", control_rows)
    xs, fs, auth, ws = training_data(control_rows)
    joint = PublisherViewPrivacyBN(regularization=.003)
    before = joint.loss_gradient(joint.theta, xs, fs, auth, weights=ws)[0]
    joint.fit(xs, fs, auth, weights=ws, max_iter=400)
    assert joint.optimizer["success"] and joint.optimizer["loss"] < before
    write(out / "joint_fit.json", joint.parameters())
    examples, errors = [], []
    for current in ("valid", "revoked", "root_unknown", "delegation_unknown", "opaque"):
        for visible in ("supported", "contradicted", "unresolved", "missing"):
            for stage in (1, 2):
                if stage == 1 and visible != "missing":
                    continue
                sample = fixture(current=current, visible=visible, stage=stage)
                for target in (1, 2):
                    prediction = partial.predict(sample, at_least=target)
                    graph = partial.probability_graph(sample, at_least=target)
                    errors.append(abs(evaluate(graph) - prediction["model_probability"]))
                    examples.append({"fixture": {"current": current, "visible": visible, "stage": stage}, "target": target,
                        "prediction": prediction, "graph": graph})
    write(out / "graph_examples.json", examples)
    real = replay(parent, out)
    report = {"model": "PublisherViewPrivacyBN-v1", "parameter_count": len(joint.theta), "frequency_rows": len(rows),
        "frequency_total_weight": sum(weights), "not_independent_trajectories": True, "expected_visible_coefficients": expected,
        "recovered_visible_coefficients": partial.theta[-2:].tolist(), "partial_optimizer": partial.optimizer,
        "joint_control_initial_loss": before, "joint_control_final_loss": joint.optimizer["loss"], "joint_optimizer": joint.optimizer,
        "mechanism_graph_count": len(examples), "max_mechanism_graph_error": max(errors), "real_input_replay": real,
        "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}}
    write(out / "REPORT.json", report)
    print(json.dumps({k: report[k] for k in ("parameter_count", "recovered_visible_coefficients", "mechanism_graph_count", "real_input_replay")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.out.resolve())
