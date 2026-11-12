"""Reviewed public tool descriptions, not benchmark policy or runtime trust.

Annotations choose a primary capability, NOT whether an action is permitted.
An exact supporting quote is an audit trail, not semantic verification.
The initial registry is manual; there is no automatic LLM classifier here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


CAPABILITIES = ("information_read", "information_processing", "external_communication",
                "state_change", "payment_effect", "unknown")
from ._resources import static_resource

DEFAULT_PATH = static_resource('knowledge/templates/harnessaudit_public_capabilities_v1.json')


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ToolCapabilityRegistry:
    def __init__(self, specification):
        spec = copy.deepcopy(specification)
        if not isinstance(spec, dict) or set(spec) != {"schema_version", "registry_id", "source", "tools"}:
            raise ValueError("registry requires only schema_version, registry_id, source and tools")
        if spec["schema_version"] != "1" or not isinstance(spec["registry_id"], str) or not spec["registry_id"]:
            raise ValueError("unsupported or missing registry version")
        if not isinstance(spec["source"], dict) or set(spec["source"]) != {"url", "revision", "selection"}:
            raise ValueError("public source provenance required")
        if any(not isinstance(v, str) or not v for v in spec["source"].values()):
            raise ValueError("source provenance must be nonempty text")
        if not isinstance(spec["tools"], list):
            raise ValueError("tools must be a list of public descriptions and reviewed annotations")
        entries = {}
        for item in spec["tools"]:
            if not isinstance(item, dict) or set(item) != {"name", "description", "capability", "supporting_quote", "annotation_method"}:
                raise ValueError("tool profile contains missing or non-public fields")
            if any(not isinstance(v, str) or not v for v in item.values()):
                raise ValueError("tool profile fields must be nonempty text")
            if item["name"] in entries or item["capability"] not in CAPABILITIES:
                raise ValueError("duplicate tool or unsupported capability")
            if item["supporting_quote"] not in item["description"]:
                raise ValueError("supporting quote must occur in the public description")
            if item["annotation_method"] != "manual_review":
                raise ValueError("only explicitly reviewed manual annotations are currently supported")
            entries[item["name"]] = item
        self._spec, self._entries = spec, entries
        self.registry_id, self.sha256 = spec["registry_id"], fingerprint(spec)

    def specification(self):
        return copy.deepcopy(self._spec)

    def resolve(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be nonempty text")
        entry = self._entries.get(name)
        return {"tool": name, "capability": "unknown" if entry is None else entry["capability"],
                "resolution": "unregistered" if entry is None else "reviewed_public_description",
                "description": None if entry is None else entry["description"],
                "supporting_quote": None if entry is None else entry["supporting_quote"],
                "annotation_method": None if entry is None else entry["annotation_method"],
                "registry_id": self.registry_id, "registry_sha256": self.sha256}


def load_tool_registry(path=DEFAULT_PATH):
    return ToolCapabilityRegistry(json.loads(Path(path).read_text(encoding="utf-8")))
