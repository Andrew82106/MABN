"""Knowledge-selected message subgraphs, independent of benchmark stage names.

The caller supplies a declared DAG and bound prefix records, not future labels.
Each stage owns one complete output and an explicit tracking set. A route is
not proof that the tracked value occurred; only completed text sets 0 or 1.
This compiler does not infer roles, routes, permissions, or factual cue truth.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from paperalpha_runtime.canonical import digest
from paperalpha_runtime.fixed_workflow import topological_order
from .observable_event_dag import ObservableEventDAG

from ._resources import static_resource

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = static_resource('knowledge/templates/message_workflow_templates_v1.json')
SUPPORTED = {
    "configured_source": ("prepare", True),
    "prepare_message": ("prepare", False),
    "work_message": ("work", False),
    "review_message": ("review", False),
    "publish_message": ("publish", False),
}
CUES = {"support": [1, 0], "opposition": [0, 1], "unknown": [0, 0]}


def _fields(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(description + " has unsupported or missing fields")


def _identity(value):
    if not isinstance(value, str) or not value.strip() or ":" in value:
        raise ValueError("nonempty identity without ':' required")
    return value


def _evidence(value, cutoff):
    _fields(value, ("record_id", "sequence", "source", "text"), "prefix evidence")
    if type(value["sequence"]) is not int or not 0 <= value["sequence"] <= cutoff:
        raise ValueError("future evidence cannot enter a prefix")
    if any(not isinstance(value[k], str) or not value[k] for k in ("record_id", "source", "text")):
        raise ValueError("nonempty record identity, source and evidence text required")


class MessageWorkflowCompiler:
    def __init__(self, specification=None, catalog=None):
        spec = json.loads(TEMPLATES.read_text(encoding="utf-8")) if specification is None else copy.deepcopy(specification)
        catalog = json.loads(static_resource('knowledge/risks/risk_catalog.json').read_text(encoding="utf-8")) if catalog is None else copy.deepcopy(catalog)
        _fields(spec, ("schema", "scope", "knowledge", "templates"), "template registry")
        if spec["schema"] != "message-workflow-templates-v1" or not isinstance(spec["scope"], str) or not spec["scope"]:
            raise ValueError("supported message compiler schema and scope required")
        risks = {r["id"]: r for r in catalog["entries"]}
        sources = {r["id"]: r for r in catalog["sources"]}
        knowledge = spec["knowledge"]
        _fields(knowledge, ("risk_ids", "source_ids", "meaning"), "knowledge")
        rids, sids = knowledge["risk_ids"], knowledge["source_ids"]
        if not rids or not sids or not set(rids) <= set(risks) or not set(sids) <= set(sources):
            raise ValueError("knowledge must resolve to the existing risk catalogue")
        if not set(sids) <= {s for r in rids for s in risks[r]["source_ids"]} or not knowledge["meaning"]:
            raise ValueError("knowledge sources must support the selected risk entries")
        self.knowledge = {**copy.deepcopy(knowledge),
            "risks": [copy.deepcopy(risks[r]) for r in rids],
            "sources": [copy.deepcopy(sources[s]) for s in sids]}
        self.templates = {}
        for item in spec["templates"]:
            _fields(item, ("id", "family", "requires_past", "meaning"), "message template")
            name = item["id"]
            if name in self.templates or name not in SUPPORTED or type(item["requires_past"]) is not bool:
                raise ValueError("unique supported message templates required")
            if (item["family"], item["requires_past"]) != SUPPORTED[name] or not item["meaning"]:
                raise ValueError("template semantics cannot be silently redefined")
            self.templates[name] = copy.deepcopy(item)
        if not self.templates:
            raise ValueError("at least one message template required")
        self.specification, self.sha256 = spec, digest(spec)
        self.knowledge_sha256 = digest(self.knowledge)

    def compile(self, *, workflow, subjects, stages, deliveries, records, prefix_sequence, cues=()):
        """Return a normal trainable ObservableEventDAG and a compilation manifest.

        `records`: node -> {kind: complete|missing, evidence: prefix record}.
        Absent nodes are future; missing nodes are past but not observed as 0.
        `deliveries`: explicit routes and the subject IDs tracked on each route.
        `stages`: template ID, actor, subject IDs and source-bound definition.
        `cues`: optional node/subject hypotheses with their own prefix evidence.
        """
        if digest(self.specification) != self.sha256 or digest(self.knowledge) != self.knowledge_sha256:
            raise ValueError("compiler configuration or knowledge changed")
        if self.templates != {t["id"]: t for t in self.specification["templates"]}:
            raise ValueError("compiled template lookup changed")
        if type(prefix_sequence) is not int or prefix_sequence < 1:
            raise ValueError("positive prefix sequence required")
        payload = copy.deepcopy({"workflow": workflow, "subjects": subjects, "stages": stages,
            "deliveries": deliveries, "records": records, "prefix_sequence": prefix_sequence, "cues": list(cues)})
        workflow, subjects, stages, deliveries, records, cues = (payload[k] for k in ("workflow", "subjects", "stages", "deliveries", "records", "cues"))
        _fields(workflow, ("nodes", "edges", "actors"), "workflow")
        topological_order(workflow)
        if not isinstance(workflow["actors"], dict) or set(workflow["actors"]) != set(workflow["nodes"]):
            raise ValueError("complete functional actor ownership required")
        for node in workflow["nodes"]:
            _identity(node)
        if not isinstance(subjects, list) or not subjects or not isinstance(stages, list) or not isinstance(deliveries, list):
            raise ValueError("explicit subjects, stages and deliveries required")
        literals = {}; seen_literals = set()
        for item in subjects:
            _fields(item, ("id", "literal"), "tracked subject")
            identity = _identity(item["id"]); literal = item["literal"]
            if identity in literals or not isinstance(literal, str) or not literal:
                raise ValueError("unique subject identity and nonempty exact value required")
            if literal.casefold() in seen_literals:
                raise ValueError("equal literal aliases must be one tracked subject, not independent events")
            literals[identity] = literal; seen_literals.add(literal.casefold())
        by_node = {}
        for stage in stages:
            _fields(stage, ("node", "actor", "template", "subjects", "definition"), "stage")
            node = stage["node"]
            if node not in workflow["nodes"] or node in by_node or stage["actor"] not in workflow["actors"][node]:
                raise ValueError("one declared owned message per functional stage required")
            if stage["template"] not in self.templates:
                raise ValueError("no matching knowledge template; do not fall back to a guessed role")
            tracked = stage["subjects"]
            if not isinstance(tracked, list) or not tracked or len(set(tracked)) != len(tracked) or not set(tracked) <= set(literals):
                raise ValueError("explicit nonempty unique stage tracking set required")
            _evidence(stage["definition"], prefix_sequence); by_node[node] = stage
        if set(by_node) != set(workflow["nodes"]) or set(literals) != {s for stage in stages for s in stage["subjects"]}:
            raise ValueError("all workflow stages and declared tracked subjects require coverage")
        if not isinstance(records, dict) or not set(records) <= set(by_node):
            raise ValueError("only declared prefix stage records permitted")
        record_ids = set()
        for node, record in records.items():
            _fields(record, ("kind", "evidence"), "stage record")
            if record["kind"] not in ("complete", "missing"):
                raise ValueError("record must be complete or explicitly missing past content")
            _evidence(record["evidence"], prefix_sequence)
            key = record["evidence"]["record_id"]
            if key in record_ids:
                raise ValueError("one output record cannot masquerade as multiple stage outputs")
            record_ids.add(key)
        hypotheses = {}
        for cue in cues:
            _fields(cue, ("node", "subject", "category", "evidence"), "task cue")
            key = cue["subject"], cue["node"]
            if cue["node"] not in by_node or cue["subject"] not in by_node[cue["node"]]["subjects"] or key in hypotheses:
                raise ValueError("unique cue for an existing stage subject required")
            if cue["category"] not in CUES:
                raise ValueError("explicit support, opposition or unknown cue required")
            _evidence(cue["evidence"], prefix_sequence); hypotheses[key] = cue
        events, observations, missing, manifest_nodes = [], {}, [], []
        # Subject-major order is stable and preserves the historical projection.
        for subject, literal in literals.items():
            for node in workflow["nodes"]:
                stage = by_node[node]
                if subject not in stage["subjects"]:
                    continue
                template = self.templates[stage["template"]]; identity = subject + ":" + node
                if template["requires_past"] and node not in records:
                    raise ValueError("configured source must be observed or declared missing, not a guessed future value")
                cue = hypotheses.get((subject, node))
                events.append({"id": identity, "node": node, "actor": stage["actor"], "family": template["family"],
                    "semantics": "literal_appearance", "subject": literal,
                    "cues": CUES[cue["category"]].copy() if cue else [0, 0],
                    "cue_evidence": cue["evidence"] if cue else stage["definition"]})
                if node in records:
                    record = records[node]
                    if record["kind"] == "missing":
                        missing.append(identity)
                    else:
                        e = record["evidence"]
                        observations[identity] = {**e, "kind": "complete_message", "value": int(literal.casefold() in e["text"].casefold())}
                manifest_nodes.append({"event_id": identity, "subject_id": subject, "template_id": template["id"],
                    "family": template["family"], "stage_definition_sha256": digest(stage["definition"]),
                    "cue_sha256": digest(cue) if cue else None,
                    "record_sha256": digest(records[node]) if node in records else None})
        edges, seen_routes = [], set()
        for route in deliveries:
            _fields(route, ("source", "target", "subjects", "basis", "evidence"), "delivery route")
            a, b = route["source"], route["target"]
            if [a, b] not in workflow["edges"] or (a, b) in seen_routes:
                raise ValueError("one explicit delivery descriptor per declared functional edge required")
            tracked = route["subjects"]
            if not isinstance(tracked, list) or len(set(tracked)) != len(tracked) or not set(tracked) <= set(by_node[a]["subjects"]) & set(by_node[b]["subjects"]):
                raise ValueError("delivery may only connect the same explicitly tracked subjects")
            if route["basis"] not in ("observed_delivery", "declared_delivery"):
                raise ValueError("delivery cannot assert causal or inferred adoption")
            _evidence(route["evidence"], prefix_sequence); seen_routes.add((a, b))
            if b in records and a not in records:
                raise ValueError("past recipient requires past input stages; record missing input content explicitly")
            if route["basis"] == "observed_delivery" and (a not in records or records[a]["evidence"]["sequence"] > route["evidence"]["sequence"]):
                raise ValueError("observed delivery must follow its source output")
            if route["basis"] == "observed_delivery" and b in records and route["evidence"]["sequence"] > records[b]["evidence"]["sequence"]:
                raise ValueError("observed delivery cannot follow the recipient output")
            if a in records and b in records and records[a]["evidence"]["sequence"] > records[b]["evidence"]["sequence"]:
                raise ValueError("past input routes cannot reverse message time")
        if seen_routes != {tuple(e) for e in workflow["edges"]}:
            raise ValueError("every workflow edge needs a descriptor, including empty tracked content")
        for subject in literals:
            for route in deliveries:
                if subject in route["subjects"]:
                    edges.append({"source": subject + ":" + route["source"], "target": subject + ":" + route["target"],
                        "basis": route["basis"], "evidence": route["evidence"]})
        dag = ObservableEventDAG(workflow=workflow, events=events, edges=edges, observations=observations,
            prefix_sequence=prefix_sequence, missing_past=missing)
        manifest = {"schema": "compiled-message-workflow-v1", "compiler_sha256": self.sha256,
            "knowledge": copy.deepcopy(self.knowledge), "knowledge_sha256": self.knowledge_sha256,
            "input_sha256": digest(payload), "event_dag_sha256": dag.sha256, "nodes": manifest_nodes,
            "stage_templates": {node: by_node[node]["template"] for node in workflow["nodes"]},
            "delivery_evidence_sha256": [digest(route) for route in deliveries],
            "scope": self.specification["scope"],
            "limitations": ["Roles/routes are declared inputs, not discovered facts.",
                "Tracking sets must be supplied completely; omitted values and semantic paraphrases are not covered.",
                "Shared family parameters are modeling assumptions, not claims established by literature.",
                "Template expansion adds no hidden intent, authenticated authorization or GNN."]}
        return dag, manifest
