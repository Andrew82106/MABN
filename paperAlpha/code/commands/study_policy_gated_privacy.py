"""Frozen mechanism/learning study, explicitly not API detection evaluation."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from policy_privacy_fixtures import fixture
from scalar_binary_graph import evaluate
from paperalpha_monitor.policy_gated_privacy import PolicyGatedPrivacyBN
from paperalpha_monitor.privacy_model_independent import ModelIndependentPrivacyBN

SOURCES = ('code/commands/study_policy_gated_privacy.py', 'code/commands/policy_privacy_fixtures.py', 'code/commands/scalar_binary_graph.py',
    'code/commands/verify_policy_privacy_study.py',
    'code/src/paperalpha_monitor/policy_gated_privacy.py', 'code/src/paperalpha_monitor/privacy_model_independent.py',
    'code/src/paperalpha_monitor/privacy_stage_bn.py', 'code/src/paperalpha_monitor/privacy_flow_data.py',
    'code/src/paperalpha_monitor/privacy_flow_bn.py', 'code/src/paperalpha_monitor/privacy_supervision_controls.py',
    'code/src/paperalpha_monitor/shared_authority_graph.py', 'code/src/paperalpha_monitor/delegation_evidence.py',
    'code/src/paperalpha_runtime/canonical.py',
    'code/src/paperalpha_runtime/__init__.py', 'knowledge/risks/risk_catalog.json', 'code/tests/test_policy_gated_privacy.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def constant_model(p=.6, regularization=0.):
    model = PolicyGatedPrivacyBN(regularization=regularization)
    model.theta[:model.n] = 0.
    model.theta[model.presence.d:model.presence.d + 4] = logit(p)
    return model


def frequency_rows():
    """Analytic population masses, not sampled or independent experimental runs.

    Generate authority truth explicitly, without calling graph.worlds/predict.
    Six constructed profiles each have total weight one; W bits are fair coins.
    F probability is .8 when supported, .6 when reference policy fails.
    """
    definitions = [({"shared": True}, "shared", .2), ({"shared": False}, "independent", .2),
        ({"unknown_components": "delegation"}, "independent", .3),
        ({"status": "missing_root"}, "independent", .4), ({"status": "valid"}, "known", 0.),
        ({"status": "revoked"}, "known", 1.)]
    rows = []
    for spec, structure, q in definitions:
        authorities = ([[[0, 0], 1 - q], [[1, 1], q]] if structure == "shared" else
            [[[int(q), int(q)], 1.]] if structure == "known" else
            [[list(a), float(np.prod([q if z else 1 - q for z in a]))] for a in product((0, 1), repeat=2)])
        for authority, mass in authorities:
            for w, f in product(product((0, 1), repeat=2), repeat=2):
                p = [.6 if failed else .8 for failed in authority]
                weight = .25 * mass * float(np.prod([r if y else 1 - r for r, y in zip(p, f)]))
                rows.append({"fixture": {**spec, "w": list(w)}, "final_labels": list(f), "authority_labels": authority,
                    "weight": weight, "system_label": int(any(x and a for x, a in zip(f, authority)))})
    return rows


def dataset(rows):
    return ([fixture(**r["fixture"]) for r in rows], [r["final_labels"] for r in rows],
        [r["authority_labels"] for r in rows], [r["weight"] for r in rows])


def run(out):
    if out.exists():
        raise ValueError("new study output required")
    out.mkdir(parents=True)
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write(out / "PLAN.json", {"schema": "policy-gated-privacy-mechanism-study-v1",
        "source_sha256": {s: sha(out / "source_snapshot" / s) for s in SOURCES},
        "scope": "Synthetic algebraic and population-frequency checks; no API calls, performance holdout or original MAS baseline",
        "learning": "Population recovery trains only three new fact logits and authorized-emission coefficient, with a fixed known presence generator. A separate four-row control trains all 43 parameters.",
        "no_legacy_changes": "No old models, predictions or test labels are opened or overwritten."})
    catalog = json.loads((ROOT / 'knowledge/risks/risk_catalog.json').read_text(encoding="utf-8"))
    ids = {"RB009", "RB010", "RB011", "RB012", "RB013", "RB014"}
    entries = [e for e in catalog["entries"] if e["id"] in ids]
    assert {e["id"] for e in entries} == ids
    source_ids = {s for e in entries for s in e["source_ids"]}
    write(out / "KNOWLEDGE_BINDING.json", {"entries": entries, "sources": [s for s in catalog["sources"] if s["id"] in source_ids],
        "mapping": {"RB009": "Existing parent/child scope-narrowing checks", "RB010": "Recipient/purpose/field digest binding",
            "RB011": "Task/principal/actor binding", "RB012": "Observed revocation/expiry at prefix, NOT future revocation forecast",
            "RB013": "Final literal occurrence AND reference-policy failure", "RB014": "Authorized internal use is a normal control; this final-stage fragment does not score every internal recipient"},
        "not_established": "Purpose string binding does not establish task necessity; U/V and cross-component independence remain statistical assumptions."})
    rows = frequency_rows()
    write(out / "population_frequency_rows.json", rows)
    samples, fs, authorities, weights = dataset(rows)
    population = constant_model()
    initial = population.parameters()
    population.fit(samples, fs, authorities, weights=weights, trainable_names=population.names[-4:], max_iter=200)
    assert population.optimizer["success"]
    expected = [.2, .3, .4]
    actual = expit(population.theta[population.n:population.n + 3])
    assert np.max(np.abs(actual - expected)) < 1e-5
    assert abs(population.theta[-1] - (logit(.8) - logit(.6))) < 1e-5
    write(out / "population_initial.json", initial)
    write(out / "population_fit.json", population.parameters())
    print(json.dumps({"population_recovery": actual.tolist(), "beta": float(population.theta[-1]), "iterations": population.optimizer["iterations"]}), flush=True)
    small = [{"fixture": {"status": s, "w": w}, "final_labels": f, "authority_labels": a, "weight": 1.}
        for s, w, f, a in (("valid", [0, 1], [1, 0], [0, 0]), ("revoked", [1, 0], [0, 1], [1, 1]),
            ("uncertain", [1, 1], [1, 1], [1, 1]), ("mixed", [0, 0], [1, 0], [1, 0]))]
    write(out / "joint_control_rows.json", small)
    joint = PolicyGatedPrivacyBN(ModelIndependentPrivacyBN(seed=1), regularization=.01)
    write(out / "joint_initial.json", joint.parameters())
    ss, ff, aa, ww = dataset(small)
    joint.fit(ss, ff, aa, weights=ww)
    assert joint.optimizer["success"]
    write(out / "joint_fit.json", joint.parameters())
    examples, max_error = [], 0.
    for shared, status, stage, threshold in product((False, True), ("uncertain", "valid", "revoked", "recipient_mismatch", "mixed", "missing_root"), (1, 2), (1, 2)):
        spec = {"shared": shared, "status": status, "stage": stage}
        sample = fixture(**spec)
        graph = population.probability_graph(sample, at_least=threshold)
        result = population.predict(sample, at_least=threshold)
        numeric = evaluate(graph)
        error = abs(numeric - result["model_probability"])
        max_error = max(max_error, error)
        assert error < 1e-12
        examples.append({"fixture": spec, "at_least": threshold, "result": result, "graph": graph, "scalar_probability": numeric})
    write(out / "graph_examples.json", examples)
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT / name) == value for name, value in plan["source_sha256"].items())
    write(out / "REPORT.json", {"scope": "Mechanism and learnability checks, not generalization or safety performance",
        "population_weight_rows": len(rows), "population_profiles": 6, "frequency_weight_sum": sum(weights),
        "recovered_fact_probabilities": actual.tolist(), "expected_fact_probabilities": expected,
        "recovered_authorized_emission_coefficient": float(population.theta[-1]), "expected_coefficient": float(logit(.8) - logit(.6)),
        "population_optimizer": population.optimizer, "joint_optimizer": joint.optimizer,
        "executable_graph_examples": len(examples), "max_graph_probability_error": max_error,
        "parameter_count": len(joint.theta), "artifacts_sha256": {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}})
    print(json.dumps({"saved": str(out), "graphs": len(examples), "max_graph_error": max_error,
        "joint_loss": joint.optimizer["loss"], "joint_initial_loss": joint.optimizer["initial_loss"]}), flush=True)


def verify(folder, out):
    if out.exists():
        raise ValueError("new verification output required")
    report = json.loads((folder / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        assert sha(folder / name) == expected, name
    # This command must execute the frozen script, hence imports its own snapshot.
    if ROOT != folder / "source_snapshot":
        raise ValueError("run verification using the source_snapshot script")
    loss_errors, gradient_errors, graph_errors = [], [], []
    for prefix, rows_file in (("population", "population_frequency_rows.json"), ("joint", "joint_control_rows.json")):
        model = PolicyGatedPrivacyBN.restore(json.loads((folder / f"{prefix}_fit.json").read_text(encoding="utf-8")))
        rows = json.loads((folder / rows_file).read_text(encoding="utf-8"))
        samples, fs, authorities, weights = dataset(rows)
        loss, grad = model.loss_gradient(model.theta, samples, fs, authorities, weights=weights)
        loss_errors.append(abs(loss - model.optimizer["loss"]))
        assert loss_errors[-1] < 1e-10
        for seed in range(3):
            direction = np.random.default_rng(seed).normal(size=len(model.theta))
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            a = model.loss_gradient(model.theta + eps * direction, samples, fs, authorities, weights=weights)[0]
            b = model.loss_gradient(model.theta - eps * direction, samples, fs, authorities, weights=weights)[0]
            gradient_errors.append(abs((a - b) / (2 * eps) - grad @ direction))
            assert gradient_errors[-1] < 1e-7
    population = PolicyGatedPrivacyBN.restore(json.loads((folder / "population_fit.json").read_text(encoding="utf-8")))
    for record in json.loads((folder / "graph_examples.json").read_text(encoding="utf-8")):
        sample = fixture(**record["fixture"])
        result = population.predict(sample, at_least=record["at_least"])
        assert json.loads(json.dumps(result)) == record["result"]
        graph = population.probability_graph(sample, at_least=record["at_least"])
        # JSON converts tuples to arrays, so compare the serialized structure.
        assert json.loads(json.dumps(graph)) == record["graph"]
        scalar = evaluate(record["graph"])
        graph_errors.append(abs(scalar - result["model_probability"]))
        assert graph_errors[-1] < 1e-12
    write(out / "REPORT.json", {"parent_report_sha256": sha(folder / "REPORT.json"), "artifact_hashes_checked": len(report["artifacts_sha256"]),
        "max_objective_error": max(loss_errors), "max_directional_gradient_error": max(gradient_errors),
        "max_executable_graph_error": max(graph_errors), "graphs_recomputed": len(graph_errors),
        "scope": "Frozen synthetic mechanism study verified, not a performance study"})
    print(json.dumps({"verified": str(out), "graphs": len(graph_errors), "max_gradient_error": max(gradient_errors)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.out.resolve())
    else:
        verify(args.run.resolve(), args.out.resolve())
