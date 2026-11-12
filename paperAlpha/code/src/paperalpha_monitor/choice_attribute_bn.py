"""Prepared-input-bound catalogue attributes inside the choice/policy BN.

The conditional law is ordinary attribute-based logit. Slot identity, SKU,
agent name and global policy are not score features. Static field extraction
does not discover a DAG or recover intent. Multiple/no warranty mentions are
explicitly unknown; the extractor does not prove a supplier's claim is true.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import logsumexp

from paperalpha_runtime.canonical import digest
from .exclusive_dispatch_bn import ExclusiveDispatchInput, ExclusiveDispatchWorkflowRiskBN
from .shared_authority_graph import PRIOR_NAMES
from .workflow_risk_bn import WorkflowFragmentRegistry

from ._resources import static_resource

SPEC_PATH = static_resource('knowledge/templates/catalogue_choice_attributes_v1.json')
FEATURES = ("dispatch_bias", "priority_economy", "priority_quality", "cost_fraction",
            "claimed_warranty", "warranty_unknown", "economy_cost", "quality_cost",
            "economy_warranty", "quality_warranty")
WORDS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
WARRANTY = re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)[ -]year warranty", re.I)
QUALIFIER = re.compile(r"\b(no|not|without|excluded|except|unless|if|may|might|could|optional|optionally|separately|conditional|subject|additional|extra|purchased|registered|registration)\b", re.I)


def warranty_claim(text, *, assignment_need=None):
    if not isinstance(text, str):
        raise ValueError("catalogue description must be text")
    # Deliberately restricted grammar. A regex mention anywhere in prose is
    # not enough to establish an affirmative attribute (e.g. "No three-year
    # warranty"). Unsupported/qualified language remains unknown.
    clause = text.strip()
    if ";" in text:
        # Only the known *source grammar* permits a preamble. A free preceding
        # sentence may qualify a later phrase even without a keyword on our
        # list ("Eligibility determines coverage; three-year warranty").
        preambles = () if not isinstance(assignment_need, str) else (
            f"Basic {assignment_need}; meets essential requirements;",
            f"Enhanced {assignment_need}; sturdier construction and convenience features;")
        matching = [p for p in preambles if text.startswith(p)]
        if len(matching) != 1:
            return {"years": None, "span": None, "status": "unsupported_or_ambiguous_catalogue_clause"}
        clause = text[len(matching[0]):].strip()
    if clause.endswith("."): clause = clause[:-1].rstrip()
    match = WARRANTY.fullmatch(clause)
    if match is None or len(re.findall(r"\bwarranty\b", text, re.I)) != 1 or QUALIFIER.search(text):
        return {"years": None, "span": None, "status": "unsupported_or_ambiguous_catalogue_clause"}
    token = match.group(1).lower()
    years = float(WORDS[token]) if token in WORDS else float(token)
    if not np.isfinite(years) or not 0 <= years <= 100:
        raise ValueError("bounded finite advertised warranty years required")
    start = text.rfind(clause)
    return {"years": years, "span": [start, start + len(clause)], "status": "plain_catalogue_assertion_only"}


def _positive_decimal(value, *, allow_zero=False):
    if not isinstance(value, str):
        raise ValueError("decimal quantities must retain source strings")
    try:
        number = Decimal(value)
    except Exception as exc:
        raise ValueError("finite source decimal required") from exc
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        raise ValueError("positive denominator/nonnegative price required")
    return number


class AttributedDispatchInput(ExclusiveDispatchInput):
    """Exact prepared assignment, not proof of already-completed delivery."""
    def __init__(self, *, attribute_evidence, **kwargs):
        self.attribute_evidence = copy.deepcopy(attribute_evidence)
        super().__init__(**kwargs)
        self._check_attributes()

    def specification(self):
        return {**super().specification(), "attribute_evidence": copy.deepcopy(self.attribute_evidence)}

    def validate(self):
        super().validate()
        self._check_attributes()

    @classmethod
    def from_input(cls, sample, attribute_evidence):
        return cls(workflow=sample.workflow, authority=sample.authority, proposals=sample.proposals,
                   contracts=sample.contracts, combinations=sample.combinations, cutoff=sample.cutoff,
                   horizon=sample.horizon, event_semantics=sample.event_semantics,
                   choice_groups=sample.choice_groups, attribute_evidence=attribute_evidence)

    def _check_attributes(self):
        evidence = self.attribute_evidence
        if not isinstance(evidence, dict) or set(evidence) != {"schema", "planner_text", "assignments"} or evidence["schema"] != "prepared-catalogue-assignments-v1":
            raise ValueError("explicit catalogue assignment evidence required; no labels")
        try:
            planner = json.loads(evidence["planner_text"])
        except (TypeError, ValueError) as exc:
            raise ValueError("actual planner JSON evidence required") from exc
        if not isinstance(planner, dict) or set(planner) != {"priority", "work_order"} or planner["priority"] not in {"economy", "balanced", "quality"}:
            raise ValueError("known visible planner priority required")
        if not isinstance(planner["work_order"], str) or not planner["work_order"].strip():
            raise ValueError("nonempty actual work order required")
        actors = {p["binding"]["actor"] for p in self.proposals}
        assignments = evidence["assignments"]
        if not isinstance(assignments, dict) or set(assignments) != actors:
            raise ValueError("one buyer-visible assignment per actor required")
        for actor, record in assignments.items():
            if not isinstance(record, dict) or set(record) != {"record_id", "sequence", "source", "intended_recipient", "delivery_status", "assignment"}:
                raise ValueError("strict prepared assignment record required")
            if record["intended_recipient"] != actor or record["delivery_status"] != "prepared_for_next_call" or type(record["sequence"]) is not int or not 0 <= record["sequence"] <= self.cutoff:
                raise ValueError("prepared next-call input required; not proof of delivery before cutoff")
            if any(not isinstance(record[k], str) or not record[k] for k in ("source", "record_id")):
                raise ValueError("assignment provenance required")
            branch = record["assignment"]
            if not isinstance(branch, dict) or set(branch) != {"actor", "node", "resource", "need", "single_limit", "catalogue"} or branch["actor"] != actor:
                raise ValueError("exact buyer assignment required")
            _positive_decimal(branch["single_limit"])
            items = branch["catalogue"]
            if not isinstance(items, list) or not items:
                raise ValueError("nonempty finite catalogue required")
            seen = set()
            for item in items:
                if not isinstance(item, dict) or set(item) != {"sku", "price", "description"} or not isinstance(item["sku"], str) or not item["sku"] or item["sku"] in seen:
                    raise ValueError("unique exact catalogue items required")
                seen.add(item["sku"]); _positive_decimal(item["price"], allow_zero=True)
                warranty_claim(item["description"], assignment_need=branch["need"])
            proposals = [p for p in self.proposals if p["binding"]["actor"] == actor]
            groups = [g for g in self.choice_groups if any(a in {p["action_id"] for p in proposals} for a in g["action_ids"])]
            if len(groups) != 1 or set(groups[0]["action_ids"]) != {p["action_id"] for p in proposals}:
                raise ValueError("one actor catalogue must map to exactly one exclusive slot")
            if len(proposals) != len(items) or {p["binding"]["arguments"].get("sku") for p in proposals} != seen:
                raise ValueError("exact complete candidate/catalogue coverage required")
            for p in proposals:
                item = next(i for i in items if i["sku"] == p["binding"]["arguments"]["sku"])
                if (p["family"] != "payment" or p["node"] != branch["node"] or p["binding"]["resource"] != branch["resource"]
                        or p["binding"]["arguments"] != {"sku": item["sku"], "amount": item["price"], "currency": "USD"}
                        or p["quantity"]["single_limit"] != branch["single_limit"]):
                    raise ValueError("attribute record must bind the exact purchase candidate and visible limit")
                if p["evidence"]["text"] != evidence["planner_text"]:
                    raise ValueError("choice and attribute models must use the same planner evidence")


def attribute_rows(sample):
    """Ten dimensionless score features and independently inspectable evidence."""
    if not isinstance(sample, AttributedDispatchInput):
        raise ValueError("bound catalogue attributes required")
    sample.validate()
    priority = json.loads(sample.attribute_evidence["planner_text"])["priority"]
    economy, quality = float(priority == "economy"), float(priority == "quality")
    rows, audit = [], []
    for p in sample.proposals:
        record = sample.attribute_evidence["assignments"][p["binding"]["actor"]]
        branch = record["assignment"]
        item_index = next(i for i, v in enumerate(branch["catalogue"]) if v["sku"] == p["binding"]["arguments"]["sku"])
        item = branch["catalogue"][item_index]
        cost = float(_positive_decimal(item["price"], allow_zero=True) / _positive_decimal(branch["single_limit"]))
        claim = warranty_claim(item["description"], assignment_need=branch["need"])
        warranty = 0. if claim["years"] is None else claim["years"] / 3.
        features = [1., economy, quality, cost, warranty, float(claim["years"] is None),
                    economy * cost, quality * cost, economy * warranty, quality * warranty]
        if not np.isfinite(features).all() or cost > 1e6:
            raise ValueError("bounded finite feature values required")
        rows.append(features)
        audit.append({"action_id": p["action_id"], "action_sha256": digest(p["binding"]),
                      "record_id": record["record_id"], "record_sha256": digest(record),
                      "catalogue_path": ["assignment", "catalogue", item_index],
                      "price": item["price"], "visible_limit": branch["single_limit"],
                      "warranty_claim": claim, "features": dict(zip(FEATURES, features))})
    return np.asarray(rows), audit


class AttributeChoiceWorkflowRiskBN(ExclusiveDispatchWorkflowRiskBN):
    MODEL = "AttributeChoiceWorkflowRiskBN-v1"

    def __init__(self, *, registry=None, regularization=.001, max_binary_variables=16):
        super().__init__(registry=registry, regularization=regularization, max_binary_variables=max_binary_variables)
        self.attribute_spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        if (self.attribute_spec.get("schema") != "catalogue-choice-attributes-v1"
                or self.attribute_spec["features"] != list(FEATURES) or self.attribute_spec["warranty_scale_years"] != 3
                or self.attribute_spec.get("warranty_parser") != "bound-catalogue-grammar-v2"
                or self.attribute_spec.get("input_timing") != "prepared_for_next_call"
                or self.attribute_spec.get("priority_reference") != "balanced"
                or self.attribute_spec.get("choice_reference") != "abstention, all feature values zero"):
            raise ValueError("supported frozen attribute program required")
        self.attribute_sha256 = digest(self.attribute_spec)
        self.names = tuple("catalogue_choice:" + n for n in FEATURES) + PRIOR_NAMES
        self.theta = np.zeros(len(self.names))

    def _choice_parts(self, sample, theta):
        if digest(self.attribute_spec) != self.attribute_sha256:
            raise ValueError("attribute program mutated")
        raw, _ = attribute_rows(sample)
        x = np.pad(raw, ((0, 0), (0, len(PRIOR_NAMES))))
        parts = []
        for group in sample.choice_groups:
            indices = [sample.action_ids.index(a) for a in group["action_ids"]]
            logits = np.r_[0., (x @ theta)[indices]]
            parts.append((group, indices, logits - logsumexp(logits)))
        return x, parts

    def predict(self, sample, *, theta=None, derivatives=False, event_conditions=None):
        result = super().predict(sample, theta=theta, derivatives=derivatives, event_conditions=event_conditions)
        _, evidence = attribute_rows(sample)
        result["attribute_program_sha256"] = self.attribute_sha256
        result["attribute_evidence"] = evidence
        result["forecast_conditions"] = {
            "input_delivery": "Prepared assignments and planner text are delivered unchanged before the corresponding decisions; not already-delivered evidence.",
            "outcome_protocol": "Each buyer returns a valid finite command or abstention; the application dispatches every accepted command once to a local simulator.",
            "not_modeled": ["API delivery/response failure", "parse failure", "approval rejection", "post-choice cancellation", "real-world payment success"]}
        result["limitations"] += [
            "Attribute-based conditional logit is standard and requires an equal-information choice-model comparator.",
            "Warranty is an explicit catalogue claim, not actual product quality or a causal explanation of agent intent.",
            "Priority/attribute interactions do not capture free-text work_order nuance, future common planner states or general risk semantics.",
            "Distinct feature vectors do not prove statistical identification; correlated attributes require separate empirical checks."]
        return result

    def probability_graph(self, sample, *, output="system:any", event_conditions=None):
        graph = super().probability_graph(sample, output=output, event_conditions=event_conditions)
        graph.update({"model": self.MODEL, "attribute_program_sha256": self.attribute_sha256,
                      "parameter_snapshot_sha256": digest({"names": list(self.names), "theta": self.theta.tolist()}),
                      "feature_names": list(FEATURES), "attribute_evidence_sha256": digest(sample.attribute_evidence),
                      "forecast_input_timing": "prepared_for_next_call",
                      "categorical_edge_meaning": "Sequential CPT factorization only, not time order or a causal influence between candidates."})
        return graph

    def parameters(self):
        return {**super().parameters(), "attribute_specification": copy.deepcopy(self.attribute_spec),
                "attribute_program_sha256": self.attribute_sha256}

    @classmethod
    def restore(cls, saved):
        if saved["model"] != cls.MODEL:
            raise ValueError("wrong model")
        model = cls(registry=WorkflowFragmentRegistry(saved["registry"]), regularization=saved["regularization"],
                    max_binary_variables=saved["max_binary_variables"])
        if (model.registry.sha256 != saved["registry_sha256"] or list(model.names) != saved["names"]
                or model.attribute_spec != saved["attribute_specification"]
                or model.attribute_sha256 != saved["attribute_program_sha256"]):
            raise ValueError("parameter, policy or attribute program identity mismatch")
        model.theta = np.asarray(saved["theta"], float)
        if model.theta.shape != (len(model.names),) or not np.isfinite(model.theta).all():
            raise ValueError("finite compatible parameters required")
        model.optimizer = copy.deepcopy(saved["optimizer"])
        return model
