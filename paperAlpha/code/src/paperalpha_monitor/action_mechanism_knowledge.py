"""Literature-linked, executable dispatch-time local BN fragments.

Selectors control factor presence, not only explanatory labels. Numeric
mechanisms have their own latent histories and cannot change authorization
factors. Parameter identity is shared by template, not by agent name.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from paperalpha_runtime.semantic_evidence import digest
from .scoped_quantities import FEATURE_NAMES as QUANTITY_FEATURES
from .text_conditioned_bn import PARAMETER_NAMES as TEXT_PARAMETERS

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = static_resource('knowledge/templates/action_mechanism_templates_v1.json')
SELECTORS = {"every_dispatch", "profiled_payment", "scoped_cumulative_comparison"}


@dataclass(frozen=True)
class MechanismFactor:
    action_id: str
    node: int
    channel: int
    template_id: str
    parameter_indices: tuple[int, ...]
    values: tuple[float, ...]
    evidence: dict


class ActionMechanismRegistry:
    def __init__(self, specification, catalog):
        self._spec = copy.deepcopy(specification)
        if self._spec.get("schema_version") != "action-mechanism-templates-v1":
            raise ValueError("unsupported action mechanism specification")
        for key in ("registry_id", "scope"):
            if not isinstance(self._spec.get(key), str) or not self._spec[key]:
                raise ValueError("missing registry identity/scope")
        entries = {e["id"]: e for e in catalog["entries"]}
        sources = {s["id"] for s in catalog["sources"]}
        mechanisms = self._spec.get("mechanisms")
        if not isinstance(mechanisms, list) or not 1 <= len(mechanisms) <= 3:
            raise ValueError("current exact engine supports one to three mechanisms")
        names, initial = list(TEXT_PARAMETERS), []
        self.numeric_slices, self.head_indices = {}, {}
        ids = []
        for i, item in enumerate(mechanisms):
            identity = item.get("id")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError("unique mechanism IDs required")
            ids.append(identity)
            features = item.get("features")
            if not isinstance(features, list) or len(features) != len(set(features)) or not set(features) <= set(QUANTITY_FEATURES):
                raise ValueError("unsupported or duplicate mechanism features")
            if item.get("selector") not in SELECTORS:
                raise ValueError("unknown mechanism selector")
            if i == 0:
                if (identity, item.get("selector"), item.get("factor"), features) != (
                        "authorization", "every_dispatch", "capability_role_check_cell", []):
                    raise ValueError("first channel must retain the independent authorization factor")
            else:
                if item.get("factor") != "logistic_observed_features" or item["selector"] == "every_dispatch":
                    raise ValueError("numeric mechanisms need explicit observation applicability")
                bias = item.get("initial_bias")
                if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not np.isfinite(bias):
                    raise ValueError("finite numeric factor bias required")
                self.numeric_slices[identity] = tuple(range(len(names), len(names) + len(features) + 1))
                names += [f"mechanism:{identity}:{f}" for f in ("bias", *features)]
                initial += [float(bias), *([0.] * len(features))]
            for field in ("interpretation",):
                if not isinstance(item.get(field), str) or not item[field]:
                    raise ValueError("mechanism interpretation required")
            for field in ("benign_controls", "limitations"):
                if not isinstance(item.get(field), list) or not item[field] or any(not isinstance(v, str) or not v for v in item[field]):
                    raise ValueError("mechanism controls and limitations required")
            if not item.get("knowledge_links"):
                raise ValueError("mechanism needs literature linkage")
            for link in item["knowledge_links"]:
                risk = entries.get(link.get("risk_id"))
                cited = link.get("source_ids")
                if risk is None or not isinstance(cited, list) or not cited or len(cited) != len(set(cited)) or any(
                        isinstance(n, bool) or not isinstance(n, int) or n not in sources or n not in risk["source_ids"] for n in cited):
                    raise ValueError("invalid catalog/source linkage")
                if not isinstance(link.get("coverage"), str) or not link["coverage"]:
                    raise ValueError("knowledge coverage boundary required")
        for identity in ids[1:]:
            self.head_indices[identity] = (len(names), len(names) + 1)
            names += [f"head:{identity}:any", f"head:{identity}:process_overlap"]
            initial += [0., 0.]
        self.templates = tuple(copy.deepcopy(mechanisms))
        self.channels, self.parameter_names = tuple(ids), tuple(names)
        self.additional_initial = np.asarray(initial)
        self.sha256 = digest(self._spec)

    def specification(self):
        return copy.deepcopy(self._spec)

    def factors(self, action, quantity, sequence):
        result, decisions = [], []
        for channel, template in enumerate(self.templates):
            selector = template["selector"]
            if selector == "every_dispatch":
                applicable = True
            elif selector == "profiled_payment":
                applicable = quantity is not None and action["capability_profile"]["capability"] == "payment_effect"
            else:
                applicable = quantity is not None and action["capability_profile"]["capability"] == "payment_effect" and bool(quantity["cumulative_contracts"])
            decisions.append({"template": template["id"], "selector": selector, "instantiated": applicable,
                "reason": "required observation available" if applicable else "outside declared coverage or required context unavailable"})
            if not applicable:
                continue
            if channel == 0:
                indices, values = (action["authorization_parameter"],), (1.,)
                evidence = {"permission_status": action["permission_status"], "accepted_check": copy.deepcopy(action["accepted_check"])}
            else:
                indices = self.numeric_slices[template["id"]]
                values = (1., *(float(quantity["features"][name]) for name in template["features"]))
                evidence = {"quantity_observation": copy.deepcopy(quantity), "feature_names": ["bias", *template["features"]]}
            evidence.update(sequence=sequence, actor=action["binding"]["actor"], workflow_node=action["node"],
                capability=action["capability_profile"]["capability"],
                knowledge_links=copy.deepcopy(template["knowledge_links"]), interpretation=template["interpretation"],
                limitations=list(template["limitations"]), registry_sha256=self.sha256)
            result.append(MechanismFactor(action["action_id"], action["information_variable"], channel,
                template["id"], indices, values, evidence))
        return tuple(result), tuple(decisions)


def load_action_mechanisms(path=DEFAULT_SPEC):
    return ActionMechanismRegistry(json.loads(Path(path).read_text(encoding="utf-8")),
        json.loads(static_resource('knowledge/risks/risk_catalog.json').read_text(encoding="utf-8")))
