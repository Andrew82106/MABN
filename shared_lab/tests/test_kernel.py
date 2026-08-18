from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hooks.defense_actions import cut_edge, isolate, revoke, rollback
from hooks.message_gate import MessageGate
from hooks.observation_noise import ObservationNoise
from lab_kernel.kernel import Kernel
from lab_kernel.ledger import LedgerIntegrityError, iter_events, verify_ledger
from lab_kernel.models import Action
from lab_kernel.provenance import (
    ProvenanceError,
    build_provenance,
    load_receipt,
    verify_receipt,
    write_receipt,
)
from lab_kernel.replay import ReplayError, replay
from lab_kernel.scenario import load_scenario
from lab_kernel.scheduler import Scheduler
from lab_kernel.state import state_hash
from lab_kernel.testing import StubAgent


SCENARIO_ROOT = ROOT / "scenarios" / "supplier_review_8agent"
CODE_ROOT = ROOT / "src"


class KernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.scenario = load_scenario(SCENARIO_ROOT)

    def kernel(self, hooks=()) -> Kernel:
        return Kernel(self.scenario, self.temp / "ledger.jsonl", "run-test", hooks=hooks)

    def test_offline_stub_scheduler_and_deterministic_replay(self) -> None:
        kernel = self.kernel()
        action = Action("intake", "state.write", "set_value", {"path": ["task_status"], "value": "completed"})
        Scheduler(kernel, {"intake": StubAgent([[action]])}).run()
        rebuilt = replay(
            kernel.ledger.path, kernel.initial_state, expected_run_id="run-test",
            expected_head=kernel.ledger.head_hash,
            expected_final_state_hash=state_hash(kernel.state),
        )
        self.assertEqual(rebuilt, kernel.state)

    def test_first_scenario_runs_end_to_end_with_stub_agents(self) -> None:
        kernel = self.kernel()
        agents = {
            agent["id"]: StubAgent([[]]) for agent in self.scenario.config["agents"]
        }
        report = {
            "supplier_id": "SUP-001", "recommendation": "approve", "risk_level": "low"
        }
        agents["report_publisher"] = StubAgent([[
            Action(
                "report_publisher", "state.write", "set_value",
                {"path": ["outputs", "public_report"], "value": report},
            ),
            Action(
                "report_publisher", "state.write", "set_value",
                {"path": ["task_status"], "value": "completed"},
            ),
        ]])
        Scheduler(kernel, agents).run()
        self.assertEqual(
            kernel.evaluation, {"task_success": True, "hazard_detected": False}
        )
        self.assertEqual(replay(kernel.ledger.path, kernel.initial_state), kernel.state)
        events = list(iter_events(kernel.ledger.path))
        self.assertEqual(events[-2]["event_type"], "run_evaluated")

    def test_permission_denial_is_recorded_before_execution(self) -> None:
        kernel = self.kernel()
        before = copy.deepcopy(kernel.state)
        denied = Action("intake", "report.publish", "set_value", {"path": ["outputs", "public_report"], "value": {"bad": True}})
        result = kernel.dispatch(denied)
        kernel.finish()
        self.assertFalse(result.accepted)
        self.assertEqual(kernel.state, before)
        events = list(iter_events(kernel.ledger.path))
        self.assertEqual(events[1]["event_type"], "action_denied")
        self.assertIn("requires capability", events[1]["payload"]["reason"])

    def test_every_action_kind_rejects_other_owned_capability(self) -> None:
        kernel = self.kernel()
        snapshot = copy.deepcopy(kernel.state)
        cases = (
            ("set_value", "message.send", {"path": ["task_status"], "value": "completed"}),
            ("append_value", "message.send", {"path": ["messages"], "value": {"x": 1}}),
            ("send_message", "state.write", {"message": {"edge_id": "e3", "recipient": "independent_verifier", "content": "x"}}),
            ("isolate_agent", "state.write", {"agent_id": "risk_analyst"}),
            ("cut_edge", "state.write", {"edge_id": "e1"}),
            ("revoke_permission", "state.write", {"agent_id": "intake", "capability": "message.send"}),
            ("rollback_state", "state.write", {"snapshot": snapshot, "snapshot_hash": state_hash(snapshot)}),
        )
        for kind, other_capability, params in cases:
            with self.subTest(kind=kind, claimed_capability=other_capability):
                before = kernel.state
                result = kernel.dispatch(Action("coordinator", other_capability, kind, params))
                self.assertFalse(result.accepted)
                self.assertEqual(kernel.state, before)
                self.assertIn("requires capability", result.reason)
        kernel.finish()
        denied = [
            event for event in iter_events(kernel.ledger.path)
            if event["event_type"] == "action_denied"
        ]
        self.assertEqual(len(denied), len(cases))

    def test_defense_kinds_cannot_be_spoofed_with_state_write(self) -> None:
        kernel = self.kernel()
        snapshot = copy.deepcopy(kernel.state)
        actions = (
            Action("coordinator", "state.write", "isolate_agent", {"agent_id": "risk_analyst"}),
            Action("coordinator", "state.write", "cut_edge", {"edge_id": "e1"}),
            Action("coordinator", "state.write", "revoke_permission", {"agent_id": "intake", "capability": "message.send"}),
            Action("coordinator", "state.write", "rollback_state", {"snapshot": snapshot, "snapshot_hash": state_hash(snapshot)}),
        )
        for action in actions:
            with self.subTest(kind=action.kind):
                before = kernel.state
                self.assertFalse(kernel.dispatch(action).accepted)
                self.assertEqual(kernel.state, before)
        kernel.finish()
        events = [
            event for event in iter_events(kernel.ledger.path)
            if event["event_type"] == "action_denied"
        ]
        self.assertEqual([event["payload"]["action"]["kind"] for event in events], [
            "isolate_agent", "cut_edge", "revoke_permission", "rollback_state"
        ])

    def test_state_property_cannot_bypass_dispatch(self) -> None:
        kernel = self.kernel()
        exposed = kernel.state
        exposed["task_status"] = "bypassed"
        self.assertNotEqual(kernel.state["task_status"], "bypassed")
        kernel.finish()

    def test_message_must_use_live_configured_edge(self) -> None:
        kernel = self.kernel()
        action = Action("intake", "message.send", "send_message", {"message": {"edge_id": "missing", "recipient": "coordinator", "content": "x"}})
        self.assertFalse(kernel.dispatch(action).accepted)
        kernel.finish()

    def test_append_only_ledger_refuses_reopen(self) -> None:
        kernel = self.kernel()
        kernel.finish()
        with self.assertRaises(FileExistsError):
            Kernel(self.scenario, kernel.ledger.path, "other")

    def test_ledger_tampering_fails_even_if_event_hash_is_recomputed(self) -> None:
        kernel = self.kernel()
        kernel.finish()
        trusted_head = kernel.ledger.head_hash
        trusted_file = verify_ledger(kernel.ledger.path)["file_hash"]
        lines = kernel.ledger.path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[0])
        event["payload"]["scenario_tree_hash"] = "tampered"
        event.pop("event_hash")
        from lab_kernel.canonical import content_hash
        event["event_hash"] = content_hash(event)
        lines[0] = json.dumps(event, sort_keys=True, separators=(",", ":"))
        kernel.ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(LedgerIntegrityError):
            verify_ledger(kernel.ledger.path, expected_head=trusted_head, expected_file_hash=trusted_file)

    def test_replay_rejects_state_hash_tampering(self) -> None:
        kernel = self.kernel()
        kernel.dispatch(Action("intake", "state.write", "set_value", {"path": ["task_status"], "value": "completed"}))
        kernel.finish()
        lines = kernel.ledger.path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[1])
        event["payload"]["state_after_hash"] = "0" * 64
        event.pop("event_hash")
        from lab_kernel.canonical import content_hash
        event["event_hash"] = content_hash(event)
        lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
        kernel.ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ReplayError):
            replay(kernel.ledger.path, kernel.initial_state)


class HookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.scenario = load_scenario(SCENARIO_ROOT)

    def message(self, content="private token", round_number=0) -> Action:
        return Action("intake", "message.send", "send_message", {"message": {"edge_id": "e1", "recipient": "dossier_extractor", "content": content, "round": round_number}})

    def test_message_gate_replace_drop_and_safe_rewrite(self) -> None:
        replace = Kernel(self.scenario, self.temp / "replace.jsonl", "replace", hooks=[MessageGate([{"edge_id": "e1", "mode": "replace", "content": "replacement"}])])
        replace.dispatch(self.message())
        self.assertEqual(replace.state["messages"][0]["content"], "replacement")
        replace.finish()
        drop = Kernel(self.scenario, self.temp / "drop.jsonl", "drop", hooks=[MessageGate([{"edge_id": "e1", "mode": "drop"}])])
        self.assertTrue(drop.dispatch(self.message()).accepted)
        self.assertEqual(drop.state["messages"], [])
        drop.finish()
        safe = Kernel(self.scenario, self.temp / "safe.jsonl", "safe", hooks=[MessageGate([{"edge_id": "e1", "mode": "safe_rewrite", "replacements": {"private": "redacted"}}])])
        safe.dispatch(self.message())
        self.assertEqual(safe.state["messages"][0]["content"], "redacted token")
        safe.finish()

    def test_observation_noise_missing_and_false_label(self) -> None:
        noise = ObservationNoise([{"agent_id": "intake", "mode": "drop_field", "field": "state"}, {"agent_id": "intake", "mode": "replace_value", "field": "detector_alert", "value": True}])
        kernel = Kernel(self.scenario, self.temp / "noise.jsonl", "noise", hooks=[noise])
        observation = kernel.observation_for("intake")
        self.assertNotIn("state", observation)
        self.assertTrue(observation["detector_alert"])
        kernel.finish()

    def test_all_defense_actions_are_permission_checked_and_replayable(self) -> None:
        kernel = Kernel(self.scenario, self.temp / "defense.jsonl", "defense")
        snapshot = copy.deepcopy(kernel.state)
        self.assertTrue(kernel.dispatch(cut_edge("coordinator", "e1")).accepted)
        self.assertTrue(kernel.dispatch(revoke("coordinator", "intake", "state.write")).accepted)
        self.assertTrue(kernel.dispatch(isolate("coordinator", "risk_analyst")).accepted)
        self.assertTrue(kernel.dispatch(rollback("coordinator", snapshot)).accepted)
        kernel.finish()
        rebuilt = replay(kernel.ledger.path, kernel.initial_state)
        self.assertEqual(rebuilt, kernel.state)

    def test_unauthorized_defense_is_denied(self) -> None:
        kernel = Kernel(self.scenario, self.temp / "denied.jsonl", "denied")
        self.assertFalse(kernel.dispatch(isolate("intake", "risk_analyst")).accepted)
        self.assertIn("risk_analyst", kernel.state["active_agents"])
        kernel.finish()

    def test_hook_cannot_change_action_kind_or_authority(self) -> None:
        class EscalatingHook:
            def before_action(self, action, context):
                del context
                return Action(action.actor, action.capability, "isolate_agent", {"agent_id": "risk_analyst"})

            def transform_observation(self, agent_id, observation, context):
                del agent_id, context
                return observation

        kernel = Kernel(self.scenario, self.temp / "escalation.jsonl", "escalation", hooks=[EscalatingHook()])
        benign = Action("intake", "state.write", "set_value", {"path": ["task_status"], "value": "completed"})
        self.assertFalse(kernel.dispatch(benign).accepted)
        self.assertIn("risk_analyst", kernel.state["active_agents"])
        kernel.finish()

    def test_hook_produced_message_edge_is_rechecked(self) -> None:
        class RedirectingHook:
            def before_action(self, action, context):
                del context
                message = dict(action.params["message"])
                message["edge_id"] = "missing"
                return Action(action.actor, action.capability, action.kind, {"message": message})

            def transform_observation(self, agent_id, observation, context):
                del agent_id, context
                return observation

        kernel = Kernel(
            self.scenario, self.temp / "edge-recheck.jsonl", "edge-recheck",
            hooks=[RedirectingHook()],
        )
        self.assertFalse(kernel.dispatch(self.message()).accepted)
        self.assertEqual(kernel.state["messages"], [])
        kernel.finish()


class ProvenanceAndScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)

    def test_scenario_is_declarative_and_programmatically_evaluated(self) -> None:
        scenario = load_scenario(SCENARIO_ROOT)
        self.assertEqual(len(scenario.config["agents"]), 8)
        evaluator = scenario.evaluator()
        state = scenario.initial_state()
        state["task_status"] = "completed"
        state["outputs"]["public_report"] = {"supplier_id": "SUP-001", "recommendation": "approve", "risk_level": "low"}
        self.assertEqual(evaluator(state, scenario.config), {"task_success": True, "hazard_detected": False})
        state["outputs"]["public_report"]["notes"] = state["records"]["private"]["canary"]
        self.assertTrue(evaluator(state, scenario.config)["hazard_detected"])

    def test_scenario_evaluator_rejects_empty_canary(self) -> None:
        scenario = load_scenario(SCENARIO_ROOT)
        state = scenario.initial_state()
        state["records"]["private"]["canary"] = ""
        with self.assertRaisesRegex(ValueError, "non-empty"):
            scenario.evaluator()(state, scenario.config)

    def test_receipt_verifies_config_code_environment_and_ledger(self) -> None:
        scenario = load_scenario(SCENARIO_ROOT)
        kernel = Kernel(
            scenario, self.temp / "ledger.jsonl", "provenance-run",
            code_roots=[CODE_ROOT, ROOT / "hooks"],
        )
        kernel.finish()
        self.assertTrue(kernel.receipt_path.is_file())
        receipt = load_receipt(kernel.receipt_path)
        result = verify_receipt(receipt, scenario_root=SCENARIO_ROOT, code_roots=[CODE_ROOT, ROOT / "hooks"], ledger_path=kernel.ledger.path, check_current_environment=True)
        self.assertTrue(result["valid"])
        tampered = copy.deepcopy(receipt)
        tampered["provenance"]["environment"]["python_version"] = "0.0"
        with self.assertRaises(ProvenanceError):
            verify_receipt(tampered, scenario_root=SCENARIO_ROOT, code_roots=[CODE_ROOT, ROOT / "hooks"], ledger_path=kernel.ledger.path)

    def test_configuration_tampering_fails_against_receipt(self) -> None:
        copied = self.temp / "scenario"
        import shutil
        shutil.copytree(SCENARIO_ROOT, copied)
        scenario = load_scenario(copied)
        kernel = Kernel(scenario, self.temp / "ledger.jsonl", "config-run")
        kernel.finish()
        provenance = build_provenance(copied, [CODE_ROOT, ROOT / "hooks"])
        receipt_path = self.temp / "receipt.json"
        receipt = write_receipt(receipt_path, provenance, run_id="config-run", ledger_path=kernel.ledger.path, ledger_head=kernel.ledger.head_hash, event_count=kernel.ledger.count, final_state=kernel.state)
        prompt = copied / "prompts" / "task.txt"
        prompt.write_text(prompt.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
        with self.assertRaises(ProvenanceError):
            verify_receipt(receipt, scenario_root=copied, code_roots=[CODE_ROOT, ROOT / "hooks"], ledger_path=kernel.ledger.path)

    def test_code_tampering_fails_against_receipt(self) -> None:
        import shutil
        copied_code = self.temp / "code"
        copied_hooks = self.temp / "hooks"
        shutil.copytree(CODE_ROOT, copied_code)
        shutil.copytree(ROOT / "hooks", copied_hooks)
        scenario = load_scenario(SCENARIO_ROOT)
        kernel = Kernel(
            scenario, self.temp / "ledger.jsonl", "code-run",
            code_roots=[copied_code, copied_hooks],
        )
        kernel.finish()
        receipt = load_receipt(kernel.receipt_path)
        target = copied_code / "lab_kernel" / "canonical.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
        with self.assertRaises(ProvenanceError):
            verify_receipt(
                receipt, scenario_root=SCENARIO_ROOT,
                code_roots=[copied_code, copied_hooks], ledger_path=kernel.ledger.path,
            )

    def test_kernel_source_has_no_first_scenario_constants(self) -> None:
        forbidden = ("supplier", "vendor", "canary", "8agent", "假机密", "供应商")
        for path in CODE_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token.lower(), text, f"{token!r} leaked into {path}")


if __name__ == "__main__":
    unittest.main()
