"""Inspectable object references and typed source/message parents in one BN.

The context is an input, never a model-held case registry. It records a monitor
adapter's visible initial-request snapshot and declared workflow templates.
Anchoring proves consistency with that snapshot, not authenticity of an API
log, semantic identity, relevance, permission, or causal influence.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from paperalpha_runtime.canonical import digest
from paperalpha_runtime.record_references import FEATURES as REFERENCE_FEATURES, bind_record_references
from .event_elimination import EliminatingEventLaw
from .observable_event_dag import FAMILIES, FEATURES as LEGACY_FEATURES


STAGE_FAMILIES = {
    "configured_source": "prepare", "prepare_message": "prepare",
    "work_message": "work", "review_message": "review", "publish_message": "publish",
}
PARENT_KINDS = tuple(STAGE_FAMILIES)
RELATIONAL_FEATURES = (
    "bias", "visible_support", "visible_opposition",
    *REFERENCE_FEATURES, "alias_conflict", "alias_unbound",
    *("parent_count:" + kind for kind in PARENT_KINDS),
)
CONTEXT_KEYS = {"schema", "request", "records", "reference_fields", "visibility_basis",
                "initial_evidence", "stage_definitions", "aliases"}


def _exact(value, keys, description):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(description + " requires its exact declared fields")


def _aliases(dag, records):
    # This is the upstream literal oracle, including numeric/boolean values.
    # Two fields with the same representation are aliases of ONE content event.
    return {identity: sorted(
        [{"record_id": record["record_id"], "field_name": field}
         for record in records for field, value in record["fields"].items()
         if str(value).lower() == dag.lookup[identity]["subject"].lower()],
        key=lambda row: (row["record_id"], row["field_name"]))
        for identity in dag.ids}


def make_relational_context(dag, *, request, records, reference_fields,
                            visibility_basis, initial_evidence, stage_definitions):
    """Build input bindings; the collector must attest the visible source.

    ``initial_evidence.text`` is the adapter's JSON snapshot containing exactly
    ``request`` and ``records``, copied from a visible API request or explicit
    monitor setup, not an unobserved backend database. Its record ID and clock
    must refer to the collector's prefix journal. Raw API evidence is retained
    by that adapter; this module does not authenticate transport logs.
    """
    dag.validate()
    bind_record_references(request, records, reference_fields)
    result = copy.deepcopy({"schema": "observable-relational-context",
        "request": request, "records": records, "reference_fields": reference_fields,
        "visibility_basis": visibility_basis, "initial_evidence": initial_evidence,
        "stage_definitions": stage_definitions, "aliases": _aliases(dag, records)})
    validate_relational_context(dag, result)
    return result


def validate_relational_context(dag, context):
    """Recompute references/alias features; externally supplied scores rejected."""
    _exact(context, CONTEXT_KEYS, "relational context")
    if context["schema"] != "observable-relational-context" or context["visibility_basis"] not in {
            "initial_api_request", "explicit_monitor_setup"}:
        raise ValueError("declared visible initial source required")
    source = context["initial_evidence"]
    dag._evidence(source)
    if source["sequence"] < 1:
        raise ValueError("visible initial source requires a positive journal sequence")
    try:
        snapshot = json.loads(source["text"])
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("initial evidence must be a JSON visible-input snapshot") from error
    expected = {"request": context["request"], "records": context["records"]}
    if snapshot != expected:
        raise ValueError("records and request must match the bound initial source snapshot")
    references = bind_record_references(context["request"], context["records"], context["reference_fields"])
    for record in context["records"]:
        _exact(record, ("record_id", "fields"), "visible record")
        if any(not isinstance(field, str) or not field for field in record["fields"]):
            raise ValueError("named visible record fields required")
    stages = context["stage_definitions"]
    _exact(stages, dag.workflow["nodes"], "stage definitions")
    for node, stage in stages.items():
        _exact(stage, ("template", "evidence"), "declared stage")
        if stage["template"] not in STAGE_FAMILIES:
            raise ValueError("known declared message/source template required")
        dag._evidence(stage["evidence"])
        # Template labels are declared architecture, not inferred from IDs.
        for event in dag.events:
            if event["node"] == node and event["family"] != STAGE_FAMILIES[stage["template"]]:
                raise ValueError("declared template and event family disagree")
            if event["node"] == node and stage["template"] == "configured_source" and event["id"] not in set(dag.observations) | set(dag.missing_past):
                raise ValueError("configured source cannot be a guessed future output")
    if any(event["semantics"] != "literal_appearance" for event in dag.events):
        raise ValueError("relational context currently supports literal appearances only")
    node_literals = [(event["node"], event["subject"].lower()) for event in dag.events]
    if len(set(node_literals)) != len(node_literals):
        raise ValueError("same literal aliases cannot become independent events in one node")
    aliases = _aliases(dag, context["records"])
    if context["aliases"] != aliases:
        raise ValueError("complete exact literal alias sets required, without duplicates or omissions")
    for identity, observation in dag.observations.items():
        if observation["value"] != int(dag.lookup[identity]["subject"].lower() in observation["text"].lower()):
            raise ValueError("observed literal must also match the upstream lower oracle")
    features = {}
    for identity, candidates in aliases.items():
        states = {tuple(references["features"][alias["record_id"]][name] for name in REFERENCE_FEATURES)
                  for alias in candidates}
        conflict = len(states) > 1
        row = list(next(iter(states))) if len(states) == 1 else [0] * len(REFERENCE_FEATURES)
        features[identity] = row + [int(conflict), int(not candidates)]
    return features


class RelationalEventLaw(EliminatingEventLaw):
    """Same exact inference, with named object/parent features in each CPD.

    Parents of the same declared kind share a coefficient; this is an explicit
    exchangeability assumption, not causal discovery. Review/dispatch weights
    with no training coverage are not empirical findings. No raw identities,
    record values, domains, benchmark template IDs or permissions are features.
    """
    def __init__(self, specification=None, *, max_factor_variables=16):
        if specification is None:
            super().__init__(max_factor_variables=max_factor_variables)
            knowledge = copy.deepcopy(self.specification["knowledge"])
            specification = {"schema": "relational-observable-event-law",
                "families": list(FAMILIES), "features": list(RELATIONAL_FEATURES),
                "stage_families": dict(STAGE_FAMILIES), "literal_rule": "str(value).lower()",
                "knowledge": knowledge}
        else:
            _exact(specification, ("schema", "families", "features", "stage_families", "literal_rule", "knowledge"), "relational law")
            if specification["schema"] != "relational-observable-event-law" or specification["families"] != list(FAMILIES) or specification["features"] != list(RELATIONAL_FEATURES) or specification["stage_families"] != STAGE_FAMILIES or specification["literal_rule"] != "str(value).lower()":
                raise ValueError("fixed relational feature definitions required")
            legacy = {"schema": "observable-event-dag-v1", "families": list(FAMILIES),
                      "features": list(LEGACY_FEATURES), "knowledge": specification["knowledge"]}
            super().__init__(legacy, max_factor_variables=max_factor_variables)
        self.specification = copy.deepcopy(specification)
        self.names = tuple(f"event:{family}:{feature}" for family in FAMILIES for feature in RELATIONAL_FEATURES)
        self.sha256 = digest(self.specification)

    def design(self, event, parents):
        raise ValueError("relational law requires node_design with a bound DAG context")

    def node_design(self, dag, identity, parents):
        if dag.relational_context is None or dag.relational_features is None:
            raise ValueError("explicit prefix-bound relational context required")
        if identity not in dag.lookup or len(parents) != len(dag.parents[identity]) or any(value not in (0, 1) for value in parents):
            raise ValueError("one binary state per named parent required")
        event = dag.lookup[identity]
        counts = dict.fromkeys(PARENT_KINDS, 0.)
        for parent, value in zip(dag.parents[identity], parents):
            stage = dag.relational_context["stage_definitions"][dag.lookup[parent]["node"]]
            counts[stage["template"]] += float(value)
        vector = [1., *event["cues"], *dag.relational_features[identity], *(counts[k] for k in PARENT_KINDS)]
        x = np.zeros(len(self.names))
        start = FAMILIES.index(event["family"]) * len(RELATIONAL_FEATURES)
        x[start:start + len(RELATIONAL_FEATURES)] = vector
        return x

    def graph_nodes(self, dag, theta, mapping):
        dag.validate()
        nodes = super().graph_nodes(dag, theta, mapping)
        for node, identity in zip(nodes, dag.order):
            node["relational_evidence"] = {
                "context_sha256": digest(dag.relational_context),
                "aliases": copy.deepcopy(dag.relational_context["aliases"][identity]),
                "reference_features": dict(zip((*REFERENCE_FEATURES, "alias_conflict", "alias_unbound"), dag.relational_features[identity])),
                "parent_templates": {mapping[p]: dag.relational_context["stage_definitions"][dag.lookup[p]["node"]]["template"] for p in dag.parents[identity]},
                "grants_permission": False}
        return nodes

    def feature_coverage(self, design):
        x = np.asarray(design, float).reshape((-1, len(self.names)))
        return {name: {"nonzero_rows": int(np.count_nonzero(x[:, i])),
                       "distinct_values": np.unique(x[:, i]).tolist(),
                       "empirical_effect_identified": False}
                for i, name in enumerate(self.names)}

    @staticmethod
    def optimizer_summary(optimizer):
        """Keep diagnostics/coverage, not training-case or journal identities."""
        if optimizer is None:
            return None
        allowed = {"success", "message", "iterations", "loss", "max_gradient", "supervision",
                   "fitted_parameter_names", "authority_priors_fitted", "normalization_groups",
                   "trace_count", "label_count", "objective_evaluation", "assumptions",
                   "unimplemented", "calibrated", "target", "feature_coverage"}
        return {key: copy.deepcopy(value) for key, value in optimizer.items() if key in allowed}
