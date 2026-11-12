"""Ground initial API context to the existing v7 role-conditioned CPTs.

No new probabilities or weights. Role categories remain classifier outputs;
resource mention comparisons are explanatory observations, not permissions or
new scoring features. Initial context does not create a normative workflow.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from paperalpha_runtime.initial_task_context import from_initial_request, validate_roles, action_references
from paperalpha_runtime.semantic_evidence import digest
from .knowledge_scoped_bn import JournalKnowledgeCompiler
from .dynamic_joint_bn import JointTrace


class ContextualKnowledgeCompiler(JournalKnowledgeCompiler):
    def __init__(self, *, initial_request, run_start, role_annotation=None, role_mode="unknown", **context):
        if role_mode not in {"unknown", "predicted"}:
            raise ValueError("role_mode must explicitly select unknown or predicted")
        if "roles" in context or (context.get("workflow") or {}).get("roles"):
            raise ValueError("role classifier and external role maps cannot silently override one another")
        self.initial_request, self.initial_run_start = copy.deepcopy(initial_request), copy.deepcopy(run_start)
        self.initial_context = from_initial_request(initial_request, run_start)
        self.role_mode = role_mode
        self.classification = validate_roles(self.initial_context, role_annotation) if role_annotation is not None else None
        if role_mode == "predicted" and self.classification is None:
            raise ValueError("predicted mode requires a quote-bound classification")
        actors = [a["actor"] for a in self.initial_context["actors"]]
        roles = self.classification["roles"] if role_mode == "predicted" else {a: "unknown" for a in actors}
        aliases = (context.get("workflow") or {}).get("agent_to_node", {})
        grouped = defaultdict(set)
        for actor, role in roles.items():
            grouped[aliases.get(actor, actor)].add(role)
        # Mixed functions in one workflow node are not forced into one role.
        self.effective_roles = {node: next(iter(values)) if len(values) == 1 else "unknown" for node, values in grouped.items()}
        self.grouped_role_ambiguities = [node for node, values in grouped.items() if len(values) > 1]
        self.reference_observations = {}
        super().__init__(roles=self.effective_roles, **context)

    def _push(self, row, annotation):
        if row.get("sequence") == 1 and digest(row) != digest(self.initial_run_start):
            raise ValueError("initial context does not match this run-start record")
        step = super()._push(row, annotation)
        if row["kind"] == "tool_handler_started" and isinstance(row["payload"].get("arguments"), dict):
            action = row["payload"]
            self.reference_observations[action["invocation_id"]] = action_references(
                self.initial_context, self.initial_request, action["tool"], action["arguments"])
        return step

    def explain_context(self):
        return {"initial_context": copy.deepcopy(self.initial_context), "role_mode": self.role_mode,
                "role_classification": copy.deepcopy(self.classification), "effective_node_roles": dict(self.effective_roles),
                "grouped_role_ambiguities": list(self.grouped_role_ambiguities),
                "action_resource_references": copy.deepcopy(self.reference_observations),
                "scoring_effect": "Only selected role categories enter existing v7 CPTs; raw goal/IDs and literal reference comparisons do not directly enter scores.",
                "timing": "Role classification obtained offline; not a live early-warning result.",
                "authorization_effect": "None: unknown permissions and incomplete authenticated bindings remain unchanged."}


def compile_contextual_journal(rows, annotations=None, **context):
    compiler, steps = ContextualKnowledgeCompiler(**context), []
    annotations, seen = ({} if annotations is None else annotations), set()
    for row in rows:
        identity = f"journal-{row['sequence']}"
        if identity in annotations:
            seen.add(identity)
        step = compiler.push(row, annotations.get(identity))
        if step is not None:
            steps.append(step)
    if set(annotations) != seen or not steps:
        raise ValueError("nonempty prefix and no future text annotations required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), tuple(steps), compiler.workflow), compiler.explain_context()
