"""Constructed capability/correctness checks, not measured API performance."""
import copy
import json

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit

from paperalpha_runtime.canonical import digest
from paperalpha_monitor.agentleak_event_projection import project, later_labels
from paperalpha_monitor.observable_event_dag import ObservableEventDAG, ObservableEventLaw, DAGWorkflowRiskBN, DAGWorkflowInput
from paperalpha_monitor.event_elimination import EliminatingDAGWorkflowRiskBN
from paperalpha_monitor.relational_event_law import (RelationalEventLaw, RELATIONAL_FEATURES,
    make_relational_context)
from paperalpha_monitor.observable_event_learning import complete_trace_design, trace_loss_gradient, fit_event_traces
from paperalpha_monitor.grouped_event_learning import GroupedEventTrace, grouped_complete_design, grouped_loss_gradient, fit_grouped_event_traces
from paperalpha_monitor.partial_grouped_event_learning import (PartialGroupedEventTrace,
    partial_grouped_design, partial_grouped_loss_gradient, fit_partial_grouped_event_traces)
from paperalpha_monitor.delegation_evidence import DelegationLedger
from paperalpha_monitor.shared_authority_graph import AuthorityProofGraph, AuthorityWindow
from scalar_binary_graph import evaluate


def fixture(*, request="Summarize Alice Green's visit.", records=None, literal="TOKEN_1234", scope="case"):
    initial = {"vault": {"secret": literal}, "request": request, "allowed_set": {"fields": []}}
    sample = project(initial, coordinator_text="Prepare a short summary.", coordinator_record_id=scope+":coordinator")
    records = records or [{"record_id": "person-a", "fields": {"name": "Alice Green", "secret": literal}}]
    def source(name, text):
        return {"record_id": scope+":"+name, "sequence": 1, "source": "constructed-visible-api-snapshot", "text": text}
    templates = {"vault": "configured_source", "prepare": "prepare_message", "work": "work_message", "publish": "publish_message"}
    context = make_relational_context(sample.event_dag, request=request, records=records,
        reference_fields={"name": "individual"}, visibility_basis="initial_api_request",
        initial_evidence=source("initial", json.dumps({"request": request, "records": records}, ensure_ascii=False)),
        stage_definitions={node: {"template": template, "evidence": source("stage:"+node, "Declared workflow template: "+template)}
                           for node, template in templates.items()})
    spec = sample.event_dag.specification(); spec["relational_context"] = context
    sample.event_dag = ObservableEventDAG(**spec)
    for proposal in sample.proposals:
        proposal["evidence"]["event_dag_sha256"] = sample.event_dag.sha256
    sample.sha256 = digest(sample.specification()); sample.validate()
    labels = later_labels(initial, worker_text=str(literal), worker_record_id=scope+":worker",
                          final_text="No protected value.", final_record_id=scope+":final")
    return sample, labels


def constructor(**options):
    return EliminatingDAGWorkflowRiskBN(event_law=RelationalEventLaw(), **options)


def group_fixture(scope="group", request="Summarize Alice Green's visit."):
    sample, labels = fixture(scope=scope, request=request)
    second = copy.deepcopy(labels["value-0:publish"])
    second.update(record_id=scope+":second-final", sequence=8, text="TOKEN_1234", value=1)
    return sample, GroupedEventTrace(sample.event_dag, {"value-0:work": labels["value-0:work"]},
        [{"value-0:publish": labels["value-0:publish"]}, {"value-0:publish": second}])


def test_optional_context_preserves_old_spec_hash_and_design():
    sample, _ = fixture()
    spec = sample.event_dag.specification(); spec.pop("relational_context")
    old = ObservableEventDAG(**spec)
    assert old.specification() == spec and old.sha256 == digest(spec)
    assert "relational_context" not in old.specification()
    law = ObservableEventLaw()
    assert np.array_equal(law.design(old.lookup["value-0:work"], [1, 0]),
                          law.node_design(old, "value-0:work", [1, 0]))
    with pytest.raises(ValueError, match="context"):
        RelationalEventLaw().conditional(old, ["value-0:publish"], np.zeros(70))


def test_context_hash_and_derived_features_are_revalidated():
    sample, _ = fixture(); dag = sample.event_dag
    assert dag.relational_features["value-0:work"] == [1, 0, 0, 0, 0, 0]
    changed = copy.deepcopy(dag)
    changed.relational_context["request"] = "Another task"
    with pytest.raises(ValueError): changed.validate()
    changed = copy.deepcopy(dag)
    changed.relational_features["value-0:work"][0] = 0
    with pytest.raises(ValueError): changed.validate()


@pytest.mark.parametrize("mutation", ["future_initial", "future_stage", "features", "wrong_snapshot", "missing_alias", "duplicate_alias", "unknown_stage", "wrong_family", "unseen_source"])
def test_forged_incomplete_or_future_context_rejected(mutation):
    sample, _ = fixture(); spec = sample.event_dag.specification(); context = spec["relational_context"]
    if mutation == "future_initial": context["initial_evidence"]["sequence"] = 4
    elif mutation == "future_stage": context["stage_definitions"]["work"]["evidence"]["sequence"] = 4
    elif mutation == "features": context["features"] = {"individual_reference": 1}
    elif mutation == "wrong_snapshot": context["records"][0]["fields"]["name"] = "Forged Person"
    elif mutation == "missing_alias": context["aliases"]["value-0:work"] = []
    elif mutation == "duplicate_alias": context["aliases"]["value-0:work"] *= 2
    elif mutation == "unknown_stage": context["stage_definitions"]["work"]["template"] = "guess-from-node-name"
    elif mutation == "wrong_family": context["stage_definitions"]["work"]["template"] = "prepare_message"
    else: context["visibility_basis"] = "hidden_backend_database"
    with pytest.raises(ValueError): ObservableEventDAG(**spec)


def test_equal_literal_aliases_are_one_event_with_explicit_object_conflict():
    records = [{"record_id": "a", "fields": {"name": "Alice Green", "secret": "TOKEN_1234", "alias": "TOKEN_1234"}},
               {"record_id": "b", "fields": {"name": "Bob Brown", "secret": "token_1234"}}]
    sample, _ = fixture(records=records)
    dag = sample.event_dag
    assert len(dag.ids) == 4
    assert len(dag.relational_context["aliases"]["value-0:work"]) == 3
    assert dag.relational_features["value-0:work"] == [0, 0, 0, 0, 1, 0]


def test_string_numeric_aliases_follow_upstream_literal_rule():
    records = [{"record_id": "a", "fields": {"name": "Alice Green", "amount": 10000, "text": "10000"}}]
    sample, _ = fixture(records=records, literal="10000")
    assert len(sample.event_dag.relational_context["aliases"]["value-0:work"]) == 2
    assert sample.event_dag.relational_features["value-0:work"] == [1, 0, 0, 0, 0, 0]


def test_unbound_literal_is_unknown_not_other_person_or_unauthorized():
    sample, _ = fixture(records=[{"record_id": "a", "fields": {"name": "Alice Green"}}])
    assert sample.event_dag.relational_features["value-0:work"] == [0, 0, 0, 0, 0, 1]


def test_context_rejects_duplicate_same_literal_event_in_one_node():
    sample, _ = fixture(); spec = sample.event_dag.specification()
    event = copy.deepcopy(spec["events"][-1]); event["id"] = "duplicate-value"
    spec["events"].append(event)
    spec["relational_context"]["aliases"][event["id"]] = copy.deepcopy(spec["relational_context"]["aliases"]["value-0:publish"])
    with pytest.raises(ValueError, match="independent events"):
        ObservableEventDAG(**spec)


def test_context_rejects_casefold_only_observation_outside_lower_oracle():
    sample, _ = fixture(literal="StraßeX"); spec = sample.event_dag.specification()
    spec["observations"]["value-0:vault"].update(value=1, text="STRASSEX")
    with pytest.raises(ValueError, match="lower oracle"):
        ObservableEventDAG(**spec)


def test_source_relay_swap_same_mean_can_have_different_probability():
    sample, _ = fixture(); dag = sample.event_dag; law = RelationalEventLaw()
    assert dag.parents["value-0:work"] == ("value-0:vault", "value-0:prepare")
    first, second = [law.node_design(dag, "value-0:work", bits) for bits in ([1, 0], [0, 1])]
    theta = np.zeros(len(law.names))
    theta[law.names.index("event:work:parent_count:configured_source")] = 2.
    theta[law.names.index("event:work:parent_count:prepare_message")] = -2.
    assert expit(first @ theta) > .8 and expit(second @ theta) < .2
    legacy = ObservableEventLaw()
    assert np.array_equal(legacy.node_design(dag, "value-0:work", [1, 0]), legacy.node_design(dag, "value-0:work", [0, 1]))
    assert "parent_mean" not in RELATIONAL_FEATURES


def test_record_and_actor_identity_renaming_cannot_change_features():
    first, _ = fixture()
    second, _ = fixture(records=[{"record_id": "unseen-record-999", "fields": {"name": "Alice Green", "secret": "TOKEN_1234"}}])
    spec = second.event_dag.specification()
    for node, actors in spec["workflow"]["actors"].items():
        spec["workflow"]["actors"][node] = ["renamed:"+a for a in actors]
    for event in spec["events"]: event["actor"] = "renamed:"+event["actor"]
    renamed = ObservableEventDAG(**spec); law = RelationalEventLaw()
    assert np.array_equal(law.node_design(first.event_dag, "value-0:work", [1, 0]),
                          law.node_design(renamed, "value-0:work", [1, 0]))


def test_elimination_joint_scores_match_enumeration_and_finite_difference():
    sample, _ = fixture(); dag = sample.event_dag; law = RelationalEventLaw()
    theta = np.random.default_rng(716).normal(0, .2, len(law.names))
    targets = ["value-0:work", "value-0:publish"]
    actual = law.conditional(dag, targets, theta)
    enumerated = ObservableEventLaw.conditional(law, dag, targets, theta)
    assert np.allclose(actual["log_probabilities"], enumerated["log_probabilities"], atol=1e-12)
    assert np.allclose(actual["scores"], enumerated["scores"], atol=1e-12)
    numeric = approx_derivative(lambda t: law.conditional(dag, targets, t)["log_probabilities"], theta)
    assert np.allclose(actual["scores"], numeric, atol=1e-8)


def test_policy_risk_exported_cpds_gradients_and_restore_agree():
    sample, _ = fixture(); model = constructor()
    model.theta = np.random.default_rng(84).normal(0, .2, len(model.names))
    expected = model.predict(sample, derivatives=True)
    numeric = approx_derivative(lambda t: model.predict(sample, theta=t)["probabilities"]["system:any"], model.theta).ravel()
    assert np.allclose(expected["gradients"]["system:any"], numeric, atol=1e-8)
    graph = model.probability_graph(sample)
    assert evaluate(graph) == pytest.approx(expected["probabilities"]["system:any"], abs=1e-12)
    nodes = [n for n in graph["nodes"] if "relational_evidence" in n]
    assert len(nodes) == 4 and all(not n["relational_evidence"]["grants_permission"] for n in nodes)
    restored = EliminatingDAGWorkflowRiskBN.restore(model.parameters())
    assert isinstance(restored.event_law, RelationalEventLaw)
    assert restored.predict(sample)["probabilities"] == expected["probabilities"]
    plain = DAGWorkflowRiskBN(event_law=RelationalEventLaw())
    assert isinstance(DAGWorkflowRiskBN.restore(plain.parameters()).event_law, RelationalEventLaw)


def test_complete_grouped_and_partial_learning_use_same_context_design():
    sample, group = group_fixture()
    law = RelationalEventLaw(); labels = {**group.shared_labels, **group.continuations[0]}
    x, y = complete_trace_design(law, [group.dag], [labels])
    one = GroupedEventTrace(group.dag, group.shared_labels, group.continuations[:1])
    gx, gy, audit = grouped_complete_design(law, [one])
    assert x.shape == (2, 70) and np.array_equal(x, gx) and np.array_equal(y, gy)
    partial = PartialGroupedEventTrace(group.dag, tuple(group.shared_labels), group.shared_labels, group.continuations)
    px, py, pa = partial_grouped_design(law, [partial])
    all_x, all_y, ga = grouped_complete_design(law, [group])
    assert np.array_equal(px, all_x) and np.array_equal(py, all_y)
    theta = np.random.default_rng(26).normal(0, .2, 70)
    expected = grouped_loss_gradient(law, theta, [group], .02)
    actual = partial_grouped_loss_gradient(law, theta, [partial], .02)
    assert actual[0] == expected[0] and np.array_equal(actual[1], expected[1])
    numeric = approx_derivative(lambda t: partial_grouped_loss_gradient(law, t, [partial], .02)[0], theta).ravel()
    assert np.allclose(actual[1], numeric, atol=1e-8)
    for fitted in (
            fit_event_traces([group.dag], [labels], model_class=constructor, regularization=.03),
            fit_grouped_event_traces([group], model_class=constructor, regularization=.03),
            fit_partial_grouped_event_traces([partial], model_class=constructor, regularization=.03)):
        assert fitted.optimizer["feature_coverage"]["event:review:bias"]["nonzero_rows"] == 0
        saved = fitted.parameters(); serialized = json.dumps(saved)
        assert "Alice Green" not in serialized and "TOKEN_1234" not in serialized
        assert "person-a" not in serialized and "group:worker" not in serialized
        assert "training_audit" not in saved["optimizer"]
        restored = EliminatingDAGWorkflowRiskBN.restore(saved)
        assert restored.predict(sample)["probabilities"] == fitted.predict(sample)["probabilities"]


@pytest.mark.parametrize("mutation", ["features", "schema", "case_registry"])
def test_restore_refuses_changed_vocabulary_or_case_registry(mutation):
    saved = constructor().parameters()
    if mutation == "features": saved["event_law"]["features"].append("raw_actor_id")
    elif mutation == "schema": saved["event_law"]["schema"] = "guess-new-law"
    else: saved["event_law"]["case_registry"] = {"case": "training-data"}
    with pytest.raises(ValueError): EliminatingDAGWorkflowRiskBN.restore(saved)


def large_fixture():
    """45 distinct literal groups x four stages; constructed, not API data."""
    base, _ = fixture(); base_dag = base.event_dag
    spec = base_dag.specification(); spec.pop("relational_context")
    spec.update(events=[], edges=[], observations={})
    values = {f"field-{j}": f"PRIVATE_LITERAL_{j:04d}" for j in range(45)}
    source_text = json.dumps(values)
    for j, literal in enumerate(values.values()):
        for event in base_dag.events:
            item = copy.deepcopy(event)
            item.update(id=event["id"].replace("value-0:", f"value-{j}:"), subject=literal)
            spec["events"].append(item)
            if event["id"] in base_dag.observations:
                observation = copy.deepcopy(base_dag.observations[event["id"]])
                if event["node"] == "vault": observation.update(text=source_text, value=1)
                spec["observations"][item["id"]] = observation
        for edge in base_dag.edges:
            spec["edges"].append({**edge, "source": edge["source"].replace("value-0:", f"value-{j}:"),
                "target": edge["target"].replace("value-0:", f"value-{j}:")})
    dag = ObservableEventDAG(**spec)
    request = "Summarize Alice Green's records."
    records = [{"record_id": "large-object", "fields": {"name": "Alice Green", **values}}]
    source = {"record_id": "large-initial", "sequence": 1, "source": "constructed-visible-api-snapshot",
              "text": json.dumps({"request": request, "records": records})}
    context = make_relational_context(dag, request=request, records=records, reference_fields={"name": "individual"},
        visibility_basis="initial_api_request", initial_evidence=source,
        stage_definitions=base_dag.relational_context["stage_definitions"])
    dag = ObservableEventDAG(**dag.specification(), relational_context=context)
    ledger = DelegationLedger(root_principals=["benchmark-user"], trusted_observers=["benchmark-allowlist-adapter"], coverage_complete=True)
    authority = AuthorityProofGraph(AuthorityWindow("large-static-policy", 0, 200, ()))
    proposals, queries = [], {}
    for j, (field, literal) in enumerate(values.items()):
        proposal = copy.deepcopy(base.proposals[0]); aid = f"release-{j}"
        proposal["action_id"] = aid; proposal["binding"]["action_id"] = aid
        proposal["binding"]["arguments"]["field_names"] = [field]
        authority.add(ledger.assess(proposal["binding"], sequence=10+j, task_id="constructed-large", grant_ids=[]))
        proposal["evidence"] = {"event_dag_sha256": dag.sha256, "event_id": f"value-{j}:publish",
            "available_journal_sequence": 3, "prefix_journal_sequence": 3}
        proposals.append(proposal)
        queries[aid] = {"event_id": f"value-{j}:publish", "binding_sha256": digest(proposal["binding"]),
                        "field_literals": {field: literal}}
    return DAGWorkflowInput(event_dag=dag, queries=queries, workflow=dag.workflow, authority=authority,
        proposals=proposals, cutoff=100, horizon=101, event_semantics={p["action_id"]: "literal_appearance" for p in proposals})


def test_180_event_low_width_union_preserves_all_fields_and_finite_gradient():
    sample = large_fixture(); model = constructor(); dag = sample.event_dag
    assert len(dag.ids) == 180 and len(sample.action_ids) == 45
    for name, value in {"event:work:bias": -.5, "event:work:parent_count:configured_source": 1.,
                        "event:publish:bias": -4., "event:publish:parent_count:work_message": 1.5}.items():
        model.theta[model.names.index(name)] = value
    result = model.predict_known_authorization_union(sample, derivatives=True)
    worker = expit(model.event_law.node_design(dag, "value-0:work", [1, 0]) @ model.theta[:70])
    final = [(expit(model.event_law.node_design(dag, "value-0:publish", [bit]) @ model.theta[:70])) for bit in (0, 1)]
    marginal = (1-worker)*final[0] + worker*final[1]
    expected = -np.expm1(45*np.log1p(-marginal))
    assert result["probabilities"]["system:any"] == pytest.approx(expected, abs=1e-11)
    assert len(result["invalid_authorization_events"]) == 45 and not result["approximation"]
    assert result["inference"]["prefix"]["peak_factor_variables"] <= 2
    gradient = result["gradients"]["system:any"]
    assert np.isfinite(gradient).all()
    for name in ("event:work:bias", "event:work:parent_count:configured_source", "event:publish:bias", "event:publish:individual_reference"):
        index = model.names.index(name); values = []
        for change in (-1e-5, 1e-5):
            shifted = copy.deepcopy(model); shifted.theta[index] += change
            values.append(shifted.predict_known_authorization_union(sample)["probabilities"]["system:any"])
        assert gradient[index] == pytest.approx((values[1]-values[0])/2e-5, abs=2e-8)
    with pytest.raises(ValueError, match="joint.*capacity"):
        model.predict(sample)  # Never silently replace a 45-output joint by marginals.
    spec = dag.specification(); spec["events"] += [copy.deepcopy(spec["events"][0])] * 77
    with pytest.raises(ValueError, match="256"):
        ObservableEventDAG(**spec)
