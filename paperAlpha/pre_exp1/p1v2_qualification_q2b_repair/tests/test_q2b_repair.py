from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2b_repair.audit import run_derived_audit
from p1v2_qualification_q2b_repair.common import DATA_ROOT, ORIGINAL_DATA_ROOT, canonical_json, load_json, original_tree_hashes, sha256_file, sha256_text, write_json
from p1v2_qualification_q2b_repair.deadline import BatchDeadline, guarded_card_record
from p1v2_qualification_q2b_repair.errors import DeadlineExceeded
from p1v2_qualification_q2b_repair.protocol import FROZEN_MODEL, assess_success_envelope, assess_transport_failure, frozen_metadata_profile, load_assets, request_sha256
from p1v2_qualification_q2b_repair.replay import replay_audit
from p1v2_qualification_q2b_repair.staged import FakeStagedTransport, InMemoryClock, STAGES, StagedDeadlineRunner
from p1v2_qualification_q2b_repair.validation import validate_audit


LITERAL_AUDIT_ID = "P1V2Q2B-REPAIR-AUDIT-20260802T000000000000Z"
SECOND_LITERAL_AUDIT_ID = "P1V2Q2B-REPAIR-AUDIT-20260802T000000000001Z"
INDEPENDENT_EXPECTED_MODEL_IDS = ["gpt-5.3-codex-spark"]
INDEPENDENT_METADATA_IDENTITY = {
    "normalized_models_sha256": sha256_text(canonical_json({"model_ids": INDEPENDENT_EXPECTED_MODEL_IDS})),
    "target_model_present": True,
    "model_count": 1,
}
INDEPENDENT_ALTERNATE_MODEL_IDS = ["gpt-5.3-codex-spark", "gpt-5.3-codex-spark-alt"]
COORDINATED_TAMPER_METADATA_IDENTITY = {
    "normalized_models_sha256": sha256_text(canonical_json({"model_ids": INDEPENDENT_ALTERNATE_MODEL_IDS})),
    "target_model_present": True,
    "model_count": 2,
}


class Q2BRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_before = original_tree_hashes()

    @classmethod
    def tearDownClass(cls) -> None:
        assert original_tree_hashes() == cls.original_before, "original_q2b_tree_changed"

    def _temporary_root(self):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=DATA_ROOT)

    def _baseline(self, root: Path, audit_id: str = LITERAL_AUDIT_ID) -> dict[str, object]:
        return run_derived_audit(data_root=root / "data", audit_id=audit_id)

    def _sync_manifest_hash(self, audit_dir: Path, relative: str) -> None:
        manifest_path = audit_dir / "manifest.json"
        manifest = load_json(manifest_path)
        manifest["artifact_sha256"][relative] = sha256_file(audit_dir / relative)
        write_json(manifest_path, manifest)

    def _staged_runner(
        self,
        *,
        budget_seconds: float,
        delays_seconds: dict[str, float] | None = None,
        before_stage=None,
    ) -> tuple[InMemoryClock, FakeStagedTransport, StagedDeadlineRunner]:
        clock = InMemoryClock()
        deadline = BatchDeadline(clock=clock, budget_seconds=budget_seconds)
        transport = FakeStagedTransport(
            clock=clock,
            advance=clock.advance,
            values={"metadata_start": "start", "completion": "completion", "metadata_end": "end"},
            delays_seconds=delays_seconds,
        )
        return clock, transport, StagedDeadlineRunner(deadline=deadline, transport=transport, before_stage=before_stage)

    def _stages(self, flow) -> dict[str, object]:
        return {stage.stage: stage for stage in flow.stages}

    def _rechain_ledger(self, path: Path) -> None:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        previous = "GENESIS"
        rendered: list[str] = []
        for sequence, event in enumerate(events, start=1):
            core = {"kind": event["kind"], "payload": event["payload"], "previous_event_sha256": previous, "sequence": sequence}
            event = dict(core)
            event["event_sha256"] = sha256_text(canonical_json(core))
            previous = event["event_sha256"]
            rendered.append(canonical_json(event))
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    def test_staged_start_metadata_deadline_overrun_blocks_completion_and_end(self) -> None:
        _, transport, runner = self._staged_runner(budget_seconds=10.0, delays_seconds={"metadata_start": 10.0})
        flow = runner.run()
        stages = self._stages(flow)
        self.assertEqual(flow.flow_status, "engineering_failed")
        self.assertEqual(flow.termination_reason, "deadline_during_metadata_start")
        self.assertEqual([call.stage for call in transport.calls], ["metadata_start"])
        self.assertEqual(stages["metadata_start"].status, "failed")
        self.assertTrue(stages["metadata_start"].io_attempted)
        self.assertEqual(stages["completion"].status, "not_attempted")
        self.assertEqual(stages["metadata_end"].status, "not_attempted")
        self.assertEqual(stages["completion"].reason, "blocked_after_metadata_start")
        self.assertEqual(stages["metadata_end"].reason, "blocked_after_metadata_start")

    def test_staged_completion_deadline_overrun_blocks_end(self) -> None:
        _, transport, runner = self._staged_runner(budget_seconds=10.0, delays_seconds={"completion": 10.0})
        flow = runner.run()
        stages = self._stages(flow)
        self.assertEqual(flow.flow_status, "engineering_failed")
        self.assertEqual(flow.termination_reason, "deadline_during_completion")
        self.assertEqual([call.stage for call in transport.calls], ["metadata_start", "completion"])
        self.assertEqual(stages["metadata_start"].status, "completed")
        self.assertEqual(stages["completion"].status, "failed")
        self.assertTrue(stages["completion"].io_attempted)
        self.assertEqual(stages["metadata_end"].status, "not_attempted")
        self.assertEqual(stages["metadata_end"].reason, "blocked_after_completion")

    def test_staged_end_metadata_deadline_overrun_is_engineering_failure(self) -> None:
        _, transport, runner = self._staged_runner(budget_seconds=10.0, delays_seconds={"metadata_end": 10.0})
        flow = runner.run()
        stages = self._stages(flow)
        self.assertEqual(flow.flow_status, "engineering_failed")
        self.assertEqual(flow.termination_reason, "deadline_during_metadata_end")
        self.assertEqual([call.stage for call in transport.calls], ["metadata_start", "completion", "metadata_end"])
        self.assertEqual(stages["metadata_end"].status, "failed")
        self.assertTrue(stages["metadata_end"].io_attempted)
        self.assertNotEqual(flow.flow_status, "completed")

    def test_staged_metadata_zero_negative_budget_creates_no_io(self) -> None:
        for budget in (0.0, -1.0):
            with self.subTest(position="start", budget=budget):
                _, transport, runner = self._staged_runner(budget_seconds=budget)
                flow = runner.run()
                stages = self._stages(flow)
                self.assertEqual(transport.calls, [])
                self.assertEqual(flow.termination_reason, "deadline_before_metadata_start")
                self.assertEqual(stages["metadata_start"].status, "unstarted")
                self.assertEqual(stages["completion"].status, "not_attempted")
                self.assertEqual(stages["metadata_end"].status, "not_attempted")
        for offset in (0.0, 1.0):
            with self.subTest(position="end", offset=offset):
                clock, transport, runner = self._staged_runner(budget_seconds=10.0)

                def expire_before_end(stage: str) -> None:
                    if stage == "metadata_end":
                        clock.advance(10.0 + offset - clock())

                runner = StagedDeadlineRunner(deadline=runner.deadline, transport=transport, before_stage=expire_before_end)
                flow = runner.run()
                stages = self._stages(flow)
                self.assertEqual([call.stage for call in transport.calls], ["metadata_start", "completion"])
                self.assertEqual(flow.termination_reason, "deadline_before_metadata_end")
                self.assertEqual(stages["metadata_end"].status, "unstarted")

    def test_staged_completion_zero_negative_budget_creates_no_io_and_blocks_end(self) -> None:
        for offset in (0.0, 1.0):
            with self.subTest(offset=offset):
                clock, transport, runner = self._staged_runner(budget_seconds=10.0)

                def expire_before_completion(stage: str) -> None:
                    if stage == "completion":
                        clock.advance(10.0 + offset - clock())

                runner = StagedDeadlineRunner(deadline=runner.deadline, transport=transport, before_stage=expire_before_completion)
                flow = runner.run()
                stages = self._stages(flow)
                self.assertEqual([call.stage for call in transport.calls], ["metadata_start"])
                self.assertEqual(flow.termination_reason, "deadline_before_completion")
                self.assertEqual(stages["completion"].status, "unstarted")
                self.assertFalse(stages["completion"].io_attempted)
                self.assertEqual(stages["metadata_end"].status, "not_attempted")
                self.assertFalse(stages["metadata_end"].io_attempted)

    def test_staged_timeout_is_exact_remaining_budget(self) -> None:
        _, transport, runner = self._staged_runner(budget_seconds=5.0, delays_seconds={"metadata_start": 1.0, "completion": 1.0, "metadata_end": 1.0})
        flow = runner.run()
        self.assertEqual(flow.flow_status, "completed")
        self.assertEqual([call.timeout_seconds for call in transport.calls], [5.0, 4.0, 3.0])
        for call in transport.calls:
            self.assertEqual(call.timeout_seconds, min(60.0, call.remaining_before_seconds))

    def test_safe_identity_precedes_tool_reasoning_and_semantic_failure(self) -> None:
        assets = load_assets(ROOT)
        task = assets.task
        base = {"model": FROZEN_MODEL, "system_fingerprint": "f", "provider_version": "v"}
        cases: list[tuple[dict[str, object], str]] = [
            ({**base, "choices": [{"message": {"tool_calls": []}}]}, "tool_calls"),
            ({**base, "choices": [{"message": {"reasoning": "hidden", "content": "{}"}}]}, "reasoning_detected"),
            ({**base, "choices": [{"message": {"content": "{"}}]}, "content_json"),
        ]
        for envelope, code in cases:
            with self.subTest(code=code):
                assessment = assess_success_envelope(canonical_json(envelope).encode("utf-8"), {"x-request-id": "rid"}, task)
                self.assertEqual(assessment.status, "failed")
                self.assertEqual(assessment.failure_code, code)
                self.assertEqual(assessment.safe_identity["availability"], "available")
                self.assertEqual(assessment.safe_identity["provider_declared_model"], FROZEN_MODEL)
                self.assertFalse(hasattr(assessment, "raw_body"))
        mismatch = assess_success_envelope(canonical_json({"model": "other", "choices": []}).encode("utf-8"), {"x-request-id": "rid"}, task)
        self.assertEqual(mismatch.failure_code, "model_mismatch")
        self.assertEqual(mismatch.safe_identity["availability"], "available")
        for code in ("http_429", "timeout"):
            self.assertEqual(assess_transport_failure(code).safe_identity["availability"], "unavailable")

    def test_content_envelope_and_task_cases(self) -> None:
        assets = load_assets(ROOT)
        task = assets.task
        cases = [
            (b"not-json", "malformed_envelope"),
            (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"content": ""}}]}).encode("utf-8"), "empty_content"),
            (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"content": "<think>x</think>"}}]}).encode("utf-8"), "reasoning_detected"),
            (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"content": canonical_json({"agent_role": "publisher", "episode_id": task["episode_id"], "decision": task["decision"], "task_value": task["task_value"]})}}]}).encode("utf-8"), "response_agent_role"),
            (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"content": canonical_json({"agent_role": task["agent_role"], "episode_id": task["episode_id"], "decision": task["decision"], "task_value": "wrong"})}}]}).encode("utf-8"), "response_task_value"),
        ]
        for raw, code in cases:
            with self.subTest(code=code):
                self.assertEqual(assess_success_envelope(raw, {}, task).failure_code, code)

    def test_repair_native_baseline_passes_and_legacy_copy_is_not_baseline(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            result = self._baseline(root)
            self.assertEqual(result["repair_decision"], "passed")
            self.assertTrue(validate_audit(LITERAL_AUDIT_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_audit(LITERAL_AUDIT_ID, data_root=root / "data")["passed"])
            transcript = load_json(root / "data" / "audits" / LITERAL_AUDIT_ID / "transcript.json")
            assets = load_assets(ROOT)
            self.assertEqual(frozen_metadata_profile(assets), (FROZEN_MODEL, tuple(INDEPENDENT_EXPECTED_MODEL_IDS)))
            self.assertEqual(transcript["metadata"]["start"], INDEPENDENT_METADATA_IDENTITY)
            self.assertEqual(transcript["metadata"]["end"], INDEPENDENT_METADATA_IDENTITY)
            execution = transcript["deadline_execution"]
            self.assertEqual([call["stage"] for call in execution["calls"]], list(STAGES))
            self.assertEqual([stage["stage"] for stage in execution["stages"]], list(STAGES))
            self.assertTrue(all(stage["io_attempted"] is True and stage["status"] == "completed" for stage in execution["stages"]))
            self.assertNotIn("raw_body", canonical_json(execution))
            self.assertNotIn("authorization", canonical_json(execution).lower())
            local_delivery = root / "data" / "audits" / LITERAL_AUDIT_ID / "reports" / "DELIVERY.md"
            manifest = load_json(root / "data" / "audits" / LITERAL_AUDIT_ID / "manifest.json")
            self.assertTrue(local_delivery.is_file())
            self.assertEqual(manifest["artifact_sha256"]["reports/DELIVERY.md"], sha256_file(local_delivery))
            legacy_copy = root / "legacy_original_data_copy"
            shutil.copytree(ORIGINAL_DATA_ROOT, legacy_copy)
            legacy = load_json(legacy_copy / "runs" / "P1V2Q2B-REMOTE-20260802T033140463464Z" / "transcript.json")
            self.assertTrue(any(record.get("status") == "failed" and "completion_identity" not in record for record in legacy["records"]))
            self.assertFalse((legacy_copy / "reports" / "DERIVED.md").exists())

    def test_tampered_failure_request_identity_metadata_and_unknown_field_fail_both(self) -> None:
        cases = ("request", "identity", "metadata", "unknown")
        for case in cases:
            with self.subTest(case=case), self._temporary_root() as temporary:
                root = Path(temporary)
                self._baseline(root)
                copied_data = root / "tampered_data"
                shutil.copytree(root / "data", copied_data)
                audit_dir = copied_data / "audits" / LITERAL_AUDIT_ID
                if case in {"request", "identity", "unknown"}:
                    transcript_path = audit_dir / "transcript.json"
                    transcript = load_json(transcript_path)
                    record = transcript["records"][0]
                    if case == "request":
                        record["request_sha256"] = "0" * 64
                    elif case == "identity":
                        record["safe_identity"]["request_id"] = "tampered-request"
                    else:
                        record["unexpected"] = "field"
                    write_json(transcript_path, transcript)
                    self._sync_manifest_hash(audit_dir, "transcript.json")
                else:
                    outcome_path = audit_dir / "outcome.json"
                    outcome = load_json(outcome_path)
                    outcome["metadata_calls"] = 3
                    write_json(outcome_path, outcome)
                    self._sync_manifest_hash(audit_dir, "outcome.json")
                validation = validate_audit(LITERAL_AUDIT_ID, data_root=copied_data)
                replay = replay_audit(LITERAL_AUDIT_ID, data_root=copied_data)
                self.assertFalse(validation["passed"])
                self.assertFalse(replay["passed"])

    def test_tampered_deadline_execution_semantics_fail_both(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            self._baseline(root)
            copied_data = root / "deadline_execution_tamper_data"
            shutil.copytree(root / "data", copied_data)
            audit_dir = copied_data / "audits" / LITERAL_AUDIT_ID
            transcript_path = audit_dir / "transcript.json"
            transcript = load_json(transcript_path)
            transcript["deadline_execution"]["calls"][1]["timeout_seconds"] = 60.0
            write_json(transcript_path, transcript)
            self._sync_manifest_hash(audit_dir, "transcript.json")
            self.assertFalse(validate_audit(LITERAL_AUDIT_ID, data_root=copied_data)["passed"])
            self.assertFalse(replay_audit(LITERAL_AUDIT_ID, data_root=copied_data)["passed"])

    def test_coordinated_metadata_ledger_rechain_tamper_fails_independent_oracles(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            self._baseline(root)
            copied_data = root / "metadata_ledger_rechain_tamper_data"
            shutil.copytree(root / "data", copied_data)
            audit_dir = copied_data / "audits" / LITERAL_AUDIT_ID
            transcript_path = audit_dir / "transcript.json"
            ledger_path = audit_dir / "ledger.jsonl"
            transcript = load_json(transcript_path)
            forged_metadata = dict(COORDINATED_TAMPER_METADATA_IDENTITY)
            self.assertNotEqual(forged_metadata, INDEPENDENT_METADATA_IDENTITY)
            transcript["metadata"]["start"] = dict(forged_metadata)
            transcript["metadata"]["end"] = dict(forged_metadata)
            write_json(transcript_path, transcript)
            events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
            events[1]["payload"] = dict(forged_metadata)
            events[4]["payload"] = dict(forged_metadata) | {"consistent": True}
            ledger_path.write_text("\n".join(canonical_json(event) for event in events) + "\n", encoding="utf-8")
            self._rechain_ledger(ledger_path)
            self._sync_manifest_hash(audit_dir, "transcript.json")
            self._sync_manifest_hash(audit_dir, "ledger.jsonl")
            validation = validate_audit(LITERAL_AUDIT_ID, data_root=copied_data)
            replay = replay_audit(LITERAL_AUDIT_ID, data_root=copied_data)
            self.assertFalse(validation["passed"])
            self.assertFalse(replay["passed"])
            self.assertIn("metadata_identity_oracle", validation["errors"])
            self.assertIn("replay_metadata_oracle", replay["errors"])
            self.assertNotIn("ledger_chain", validation["errors"])
            self.assertNotIn("replay_ledger_chain", replay["errors"])

    def test_two_audits_keep_local_delivery_and_manifest_isolated(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            self._baseline(root, LITERAL_AUDIT_ID)
            first_dir = root / "data" / "audits" / LITERAL_AUDIT_ID
            first_delivery = first_dir / "reports" / "DELIVERY.md"
            first_content = first_delivery.read_text(encoding="utf-8")
            first_hash = sha256_file(first_delivery)
            first_manifest = load_json(first_dir / "manifest.json")
            self.assertEqual(first_manifest["artifact_sha256"]["reports/DELIVERY.md"], first_hash)
            root_current_index = root / "data" / "reports" / "DELIVERY.md"
            root_current_index.parent.mkdir(parents=True, exist_ok=True)
            root_current_index.write_text("first current index", encoding="utf-8")
            self.assertNotEqual(first_hash, sha256_file(root_current_index))
            self._baseline(root, SECOND_LITERAL_AUDIT_ID)
            root_current_index.write_text("second current index", encoding="utf-8")
            self.assertEqual(first_delivery.read_text(encoding="utf-8"), first_content)
            self.assertEqual(sha256_file(first_delivery), first_hash)
            self.assertTrue(validate_audit(LITERAL_AUDIT_ID, data_root=root / "data")["passed"])
            self.assertTrue(validate_audit(SECOND_LITERAL_AUDIT_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_audit(LITERAL_AUDIT_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_audit(SECOND_LITERAL_AUDIT_ID, data_root=root / "data")["passed"])

    def test_rechained_semantic_ledger_identity_tamper_fails_both(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            self._baseline(root)
            copied_data = root / "ledger_tamper_data"
            shutil.copytree(root / "data", copied_data)
            audit_dir = copied_data / "audits" / LITERAL_AUDIT_ID
            ledger_path = audit_dir / "ledger.jsonl"
            events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
            events[1]["payload"]["normalized_models_sha256"] = "a" * 64
            ledger_path.write_text("\n".join(canonical_json(event) for event in events) + "\n", encoding="utf-8")
            self._rechain_ledger(ledger_path)
            self._sync_manifest_hash(audit_dir, "ledger.jsonl")
            self.assertFalse(validate_audit(LITERAL_AUDIT_ID, data_root=copied_data)["passed"])
            self.assertFalse(replay_audit(LITERAL_AUDIT_ID, data_root=copied_data)["passed"])

    def test_validator_replay_are_offline_and_original_tree_unchanged(self) -> None:
        validation_source = (ROOT / "src" / "p1v2_qualification_q2b_repair" / "validation.py").read_text(encoding="utf-8")
        replay_source = (ROOT / "src" / "p1v2_qualification_q2b_repair" / "replay.py").read_text(encoding="utf-8")
        staged_source = (ROOT / "src" / "p1v2_qualification_q2b_repair" / "staged.py").read_text(encoding="utf-8")
        audit_source = (ROOT / "src" / "p1v2_qualification_q2b_repair" / "audit.py").read_text(encoding="utf-8")
        combined = validation_source + replay_source
        self.assertNotIn("socket", combined)
        self.assertNotIn(".env", combined)
        self.assertNotIn("transport", combined)
        self.assertNotIn('data_root / "reports"', combined)
        self.assertNotIn('DATA_ROOT / "reports"', audit_source)
        for forbidden in ("socket", ".env", "requests", "urllib"):
            self.assertNotIn(forbidden, staged_source)
        self.assertEqual(original_tree_hashes(), self.original_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
