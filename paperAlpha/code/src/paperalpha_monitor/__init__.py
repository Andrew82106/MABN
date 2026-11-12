"""Explicit, lazy entry points for prefix-only MAS monitoring components.

Importing the package does not select a default model, load a knowledge file,
or import scientific dependencies. Historical exports remain available on
demand; the named current components are building blocks, not a winning model.
"""
from importlib import import_module


_EXPORTS = {
    "MessageWorkflowCompiler": "message_workflow_compiler",
    "ObservableEventDAG": "observable_event_dag",
    "ObservableEventLaw": "observable_event_dag",
    "DAGWorkflowInput": "observable_event_dag",
    "DAGWorkflowRiskBN": "observable_event_dag",
    "EliminatingEventLaw": "event_elimination",
    "EliminatingDAGWorkflowRiskBN": "event_elimination",
    "WorkflowRiskInput": "workflow_risk_bn",
    "WorkflowRiskBN": "workflow_risk_bn",
    "WorkflowFragmentRegistry": "workflow_risk_bn",
    "AuthorityProofGraph": "shared_authority_graph",
    "AuthorityWindow": "shared_authority_graph",
    "bind_message_policy": "message_policy_binding",
    "GroupedEventTrace": "grouped_event_learning",
    "ForecastAlarmStream": "forecast_alarm",
    # Compatibility only: importing paperalpha_monitor alone loads none of these.
    "EvidenceStream": "streaming_bn",
    "TemplateBN": "streaming_bn",
    "replay": "streaming_bn",
    "DynamicJointBN": "dynamic_joint_bn",
    "JointRiskStream": "joint_stream",
    "TypedJointBN": "typed_joint_bn",
    "TypedRiskStream": "typed_stream",
    "ActionScopedJointBN": "action_joint_bn",
    "ActionRiskStream": "action_joint_bn",
    "CapabilityActionBN": "capability_action_bn",
    "CapabilityRiskStream": "capability_action_bn",
    "CompactCapabilityBN": "compact_action_bn",
    "CompactCapabilityStream": "compact_action_bn",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
