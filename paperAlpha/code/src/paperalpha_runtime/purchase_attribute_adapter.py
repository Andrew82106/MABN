"""Extract catalogue-choice evidence from a frozen pre-buyer prefix only."""
import copy

from .canonical import digest
from .exclusive_purchase import compile_input
from paperalpha_monitor.choice_attribute_bn import AttributedDispatchInput


def compile_attribute_input(prefix, *, authority_complete=True):
    # No labels, buyer replies or global policy become choice-score features.
    sample = compile_input(prefix, authority_complete=authority_complete)
    assignments = {}
    for branch in prefix["branches"]:
        assignments[branch["actor"]] = {
            "record_id": "prepared-buyer-assignment:" + digest(branch),
            "sequence": prefix["cutoff"], "source": "pre-buyer-assignment",
            "intended_recipient": branch["actor"], "delivery_status": "prepared_for_next_call",
            "assignment": copy.deepcopy(branch)}
    return AttributedDispatchInput.from_input(sample, {
        "schema": "prepared-catalogue-assignments-v1", "planner_text": prefix["actual_planner_text"],
        "assignments": assignments})
