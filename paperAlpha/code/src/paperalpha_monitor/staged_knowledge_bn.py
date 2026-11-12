"""Stage-aware input/compiler for the existing 345-parameter v7 factors.

Process variables refer to functional stages. Tool bindings retain physical
actors. Text windows retain original actor/quotes but restrict prior access to
the actual stage. No new probabilities, permissions or safety labels arise
from graph expansion. Existing frozen compilers and results stay unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from paperalpha_runtime.stage_workflow import validate_stage_workflow, StageBindings, StageTextWindow
from .compact_action_bn import CompactActionCompiler, _LogicalCompiler
from .dynamic_joint_bn import JointTrace
from .knowledge_scoped_bn import JournalKnowledgeCompiler, KnowledgeStep
from .text_conditioned_bn import TextRevision, CHANNELS, grouped_features


class _StageLogicalCompiler(_LogicalCompiler):
    def _binding(self, event):
        physical = {**event, "actor": event["runtime_actor"]}
        return super()._binding(physical)

    def _action_metadata(self, event, bound):
        return {**super()._action_metadata(event, bound), "stage_id": event["actor"]}


class StageCompactCompiler(CompactActionCompiler):
    def __init__(self, *, max_information_variables=12, **context):
        self._logical = _StageLogicalCompiler(max_variables=max_information_variables, **context)
        self.variables, self._indices, self.last_step = [], {}, None
        self._sync()

    def push_stage(self, event, stage, target_stage):
        staged = {**copy.deepcopy(event), "actor": stage, "runtime_actor": event["actor"]}
        if event["event_type"] in {"message_sent", "delegation"}:
            staged["target"] = target_stage
        # Base graph events already use stage IDs. This per-event alias is
        # needed ONLY when action creation reads the preserved physical binding.
        self._logical._base.aliases = {event["actor"]: stage}
        step = super().push(staged)
        operations = []
        for op in step.operations:
            evidence = {**op.evidence, "actor": event["actor"], "stage_id": stage}
            if op.is_message:
                evidence.update(target=event["target"], target_stage_id=target_stage)
            operations.append(replace(op, evidence=evidence))
        self.last_step = replace(step, operations=tuple(operations))
        return copy.deepcopy(self.last_step)


class StageKnowledgeCompiler(JournalKnowledgeCompiler):
    """Dispatch-only stage v1 protocol, fresh contexts, v7 mechanism model.

    All journal events must carry logger-bound stages. This deliberately does
    not silently reinterpret an old hub/spoke trace as a discovered DAG.
    """
    def __init__(self, *, stage_workflow, tool_registry, available_tools=None,
                 quantity_profiles=None, budget_contracts=(), mechanism_registry=None,
                 max_information_variables=12):
        validate_stage_workflow(stage_workflow)
        self.stage_workflow = copy.deepcopy(stage_workflow)
        self.bindings = StageBindings(stage_workflow)
        workflow = {k: copy.deepcopy(stage_workflow[k]) for k in ("nodes", "edges", "roles")}
        if available_tools is not None:
            if set(available_tools) - set(stage_workflow["owners"].values()):
                raise ValueError("tool availability must use physical actor IDs")
            available_tools = {s: copy.deepcopy(available_tools[a]) for s, a in stage_workflow["owners"].items()
                               if a in available_tools}
        context = dict(workflow=workflow, tool_registry=tool_registry, available_tools=available_tools,
                       max_information_variables=max_information_variables)
        super().__init__(quantity_profiles=quantity_profiles, budget_contracts=budget_contracts,
                         mechanism_registry=mechanism_registry, **context)
        self.base = StageCompactCompiler(**context)
        self.window = StageTextWindow()

    def _push(self, row, annotation):
        event = self.projector.push(row)
        stage, target_stage = self.bindings.observe(row)
        packet = self.window.push(row)
        if packet is None and annotation is not None:
            raise ValueError("text annotations require a text record")
        if event is None:
            return None
        base = self.base.push_stage(event, stage, target_stage)
        revision = None
        if packet is not None:
            features, evidence = grouped_features(packet, annotation)
            index = next(i for i, v in enumerate(self.variables) if v["kind"] == "information_state" and v["id"] == stage)
            revision = TextRevision(index, CHANNELS.index(packet["current"]["kind"]), features, evidence)
        self.quantities.observe(row)  # physical binding and original arguments
        observations = {identity: copy.deepcopy(pair[1]) for identity, pair in self.quantities.calls.items()}
        if base.pending:
            raise ValueError("stage v1 supports dispatch observations, not pending-factor revision")
        actions = {a["action_id"]: a for a in base.actions}
        factors, decisions = [], []
        for dispatch in base.dispatches:
            action = actions[dispatch.action_id]
            compiled, selected = self.mechanisms.factors(action, observations.get(dispatch.action_id), base.sequence)
            factors.extend(compiled)
            decisions.append({"action_id": dispatch.action_id, "node": action["node"],
                              "actor": action["binding"]["actor"], "templates": list(selected)})
        self.last_step = KnowledgeStep(base, revision, observations, tuple(factors), tuple(decisions), self.mechanisms.sha256)
        return copy.deepcopy(self.last_step)


def compile_stage_journal(rows, annotations=None, **context):
    compiler, steps, seen = StageKnowledgeCompiler(**context), [], set()
    annotations = {} if annotations is None else annotations
    for row in rows:
        identity = f"journal-{row['sequence']}"
        if identity in annotations:
            seen.add(identity)
        step = compiler.push(row, annotations.get(identity))
        if step is not None:
            steps.append(step)
    if set(annotations) != seen or not steps:
        raise ValueError("nonempty visible prefix with no future annotations required")
    return JointTrace(tuple(copy.deepcopy(compiler.variables)), tuple(steps), compiler.workflow)
