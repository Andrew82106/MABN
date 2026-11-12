"""Executable conditional-table layouts tied to bounded literature claims.

The current interpreter supports four event bindings, not arbitrary discovery
of new risk mechanisms. The JSON owns table axes, names, initial parameters,
state-update semantics and knowledge links; runtime facts select table rows.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .template_registry import _text, _unique

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = static_resource('knowledge/templates/joint_risk_templates_v1.json')
DEFAULT_CATALOG = static_resource('knowledge/risks/risk_catalog.json')
SUPPORTED_BINDINGS = {
    "source": ("origin_once", "seed_or", ["input_received"], ["trust"]),
    "adoption": ("origin_to_node_once", "carrier_or", ["input_received"], ["role", "trust"]),
    "authorization": ("privileged_action_once", "seed_or", ["tool_proposed", "tool_call"], ["role", "permission"]),
    "transmission": ("actual_message", "carrier_or", ["message_sent", "delegation"], ["trust", "expected", "delegation"]),
}
SUPPORTED_OBSERVABLES = ("bias", "log_observed_nodes", "log_messages", "untrusted_message_fraction",
    "unknown_message_fraction", "workflow_known", "unexpected_message_fraction", "log_observed_origins",
    "privileged_action_seen", "permission_check_seen", "permission_mismatch_seen", "capability_manifest_known", "privileged_capability_fraction")
SUPPORTED_LATENT_SUMMARIES = ("node_risk_fraction", "joint_pair_risk_fraction", "capable_node_risk_fraction", "workflow_terminal_risk_fraction")


class JointKnowledgeRegistry:
    def __init__(self, specification, catalog):
        spec = copy.deepcopy(specification)
        if spec.get("schema_version") != "1.0":
            raise ValueError("unsupported joint knowledge schema")
        _text(spec.get("registry_id"), "registry_id")
        self.registry_id = spec["registry_id"]
        self._spec = spec
        self._axes = spec.get("axes", {})
        if set(self._axes) != {"trust", "role", "permission", "expected", "delegation"}:
            raise ValueError("unsupported axis inventory")
        for name, axis in self._axes.items():
            states = axis.get("states")
            if not isinstance(states, list) or not states or any(not isinstance(s, str) or not s for s in states) or len(set(states)) != len(states):
                raise ValueError("invalid axis states: " + name)
            if not isinstance(axis.get("unknown"), list) or not set(axis["unknown"]) <= set(states):
                raise ValueError("invalid unknown states")
            _text(axis.get("meaning"), "axis meaning")
        # Observation adapters use these explicit categories. New semantics
        # require a deliberate adapter version, not silently reordered rows.
        expected_axes = {"trust": ["clean", "untrusted", "unknown"], "permission": ["unknown", "no_mismatch", "mismatch"],
                         "expected": ["unknown", "expected", "unexpected"], "delegation": ["0", "1"]}
        if any(self._axes[k]["states"] != v for k, v in expected_axes.items()) or self._axes["role"]["states"][0] != "unknown":
            raise ValueError("axis states/order do not match the observation adapter")
        self._templates = _unique(spec.get("templates"), "id", "joint template id")
        if tuple(self._templates) != tuple(SUPPORTED_BINDINGS):
            raise ValueError("current interpreter requires the four supported bindings in declared order")
        self._entries = _unique(catalog.get("entries"), "id", "catalog risk id")
        self._sources = _unique(catalog.get("sources"), "id", "source id")
        self.offsets, self.parameter_names, self.initial_values, self._cells = {}, [], [], {}
        for name, template in self._templates.items():
            binding, update, events, axes = SUPPORTED_BINDINGS[name]
            if (template.get("binding"), template.get("update"), template.get("event_types"), template.get("axes")) != (binding, update, events, axes):
                raise ValueError("unsupported event binding/update layout: " + name)
            self.offsets[name] = len(self.parameter_names)
            for field in ("interpretation", "scope", "parameter_pattern"):
                _text(template.get(field), field)
            for field in ("benign_controls", "evidence_gaps"):
                if not isinstance(template.get(field), list) or not template[field]:
                    raise ValueError(field + " must be nonempty")
                for value in template[field]:
                    _text(value, field)
            links = _unique(template.get("knowledge_links"), "risk_id", "knowledge risk link")
            for risk_id, link in links.items():
                if risk_id not in self._entries:
                    raise ValueError("unknown linked risk: " + risk_id)
                ids = link.get("source_ids")
                if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
                    raise ValueError("unique nonempty source IDs required")
                if any(isinstance(i, bool) or not isinstance(i, int) or i not in self._sources or i not in self._entries[risk_id]["source_ids"] for i in ids):
                    raise ValueError("source ID does not belong to linked catalog risk")
                _text(link.get("coverage"), "knowledge coverage boundary")
            rows = list(itertools.product(*(self.states(axis) for axis in axes)))
            initial = template.get("initial_logits")
            if not isinstance(initial, list) or len(initial) != len(rows) or not np.isfinite(np.asarray(initial, dtype=float)).all():
                raise ValueError("initial logit count/finiteness mismatch")
            for values, prior in zip(rows, initial):
                parents = dict(zip(axes, values))
                parameter = template["parameter_pattern"].format(**parents)
                index = len(self.parameter_names)
                self._cells[index] = {"template": name, "parents": parents, "update": update, "binding": binding}
                self.parameter_names.append(parameter)
                self.initial_values.append(float(prior))
        confidence = spec["confidence"]
        _text(confidence.get("parameter_name"), "confidence parameter name")
        self.confidence_offset = len(self.parameter_names)
        self.parameter_names.append(confidence["parameter_name"])
        self.initial_values.append(float(confidence["initial_value"]))
        self.head_offset = len(self.parameter_names)
        head = spec["head"]
        self.observables, self.latent_summaries = tuple(head["observables"]), tuple(head["latent_summaries"])
        if self.observables != SUPPORTED_OBSERVABLES or self.latent_summaries != SUPPORTED_LATENT_SUMMARIES:
            raise ValueError("head features/order do not match the current observation adapter")
        names = self.observables + self.latent_summaries
        if len(names) != len(set(names)) or any(not isinstance(v, str) or not v for v in names):
            raise ValueError("invalid head feature names")
        if len(head["initial_coefficients"]) != len(names):
            raise ValueError("head coefficient count mismatch")
        self.parameter_names += ["head:" + name for name in names]
        self.initial_values += [float(v) for v in head["initial_coefficients"]]
        if len(set(self.parameter_names)) != len(self.parameter_names) or len(self.parameter_names) != spec.get("expected_parameters"):
            raise ValueError("parameter names/count mismatch")
        if not np.isfinite(self.initial_values).all():
            raise ValueError("nonfinite parameters")
        self.parameter_names = tuple(self.parameter_names)
        self.initial_values = tuple(self.initial_values)
        self.fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def states(self, axis):
        return tuple(self._axes[axis]["states"])

    def cell_index(self, template_id, parents):
        template = self._templates[template_id]
        if set(parents) != set(template["axes"]):
            raise ValueError("parent inventory differs from template")
        index = 0
        for axis in template["axes"]:
            states = self.states(axis)
            if parents[axis] not in states:
                raise ValueError("unknown parent state: " + axis)
            index = index * len(states) + states.index(parents[axis])
        return self.offsets[template_id] + index

    def metadata(self, template_id=None):
        templates = list(self._templates.values()) if template_id is None else [self._templates[template_id]]
        rows = copy.deepcopy(templates)
        for template in rows:
            for link in template["knowledge_links"]:
                link["catalog_entry"] = copy.deepcopy(self._entries[link["risk_id"]])
                link["sources"] = [copy.deepcopy(self._sources[i]) for i in link["source_ids"]]
        return {"registry_id": self.registry_id, "specification_sha256": self.fingerprint,
                "scope": self._spec["scope"], "templates": rows}

    def describe_operation(self, operation):
        if operation.parameter not in self._cells:
            raise ValueError("operation does not select a template parameter")
        cell = self._cells[operation.parameter]
        if operation.evidence.get("template") != cell["template"]:
            raise ValueError("operation template and selected row disagree")
        template = self._templates[cell["template"]]
        if operation.evidence.get("event_type") not in template["event_types"]:
            raise ValueError("event type does not instantiate selected template")
        if (operation.sender is None) != (cell["update"] == "seed_or"):
            raise ValueError("operation carrier does not match knowledge update rule")
        if operation.is_message != (cell["template"] == "transmission"):
            raise ValueError("message flag does not match selected template")
        for axis, value in cell["parents"].items():
            if axis == "permission":
                observed = self.states(axis)[operation.evidence["permission_status"]]
            elif axis == "delegation":
                observed = str(int(operation.evidence["event_type"] == "delegation"))
            else:
                observed = operation.evidence.get(axis)
            if observed != value:
                raise ValueError("operation evidence disagrees with table parent: " + axis)
        return {**copy.deepcopy(cell), "parameter": self.parameter_names[operation.parameter],
                "cell": operation.parameter, "knowledge_links": copy.deepcopy(template["knowledge_links"]),
                "interpretation": template["interpretation"], "evidence_gaps": list(template["evidence_gaps"])}


def load_joint_registry(spec_path=DEFAULT_SPEC, catalog_path=DEFAULT_CATALOG):
    return JointKnowledgeRegistry(json.loads(Path(spec_path).read_text(encoding="utf-8")),
                                  json.loads(Path(catalog_path).read_text(encoding="utf-8")))
