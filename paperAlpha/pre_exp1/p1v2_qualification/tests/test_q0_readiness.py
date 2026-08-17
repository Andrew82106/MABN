from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from p1v2_qualification.errors import QualificationError
from p1v2_qualification.fake_provider import DeterministicFakeProvider
from p1v2_qualification.fixtures import load_fixture_sets
from p1v2_qualification.paths import artifact_paths, safe_read_path
from p1v2_qualification.protocol import load_schema, parse_contract, validate_response
from p1v2_qualification.provenance import REFERENCE_SOURCE_SPECS, verify_reference_sources
from p1v2_qualification.replay import replay_run
from p1v2_qualification.runner import run_readiness_dry
from p1v2_qualification.validation import validate_run


def literal_public_report(index: int) -> dict[str, str]:
    """Independent test oracle; do not import the production value generator."""
    registration_status = "active" if index % 2 else "inactive"
    risk_level = ("low", "medium", "high")[(index - 1) % 3]
    recommendation = "reject" if risk_level == "high" else ("approve" if registration_status == "active" and risk_level == "low" else "review")
    return {
        "vendor_id": f"VQ-{index:03d}",
        "registration_status": registration_status,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


class IsolatedQ0Mixin:
    def isolated_run(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".tmp_q0_")
        root = Path(temporary.name)
        clone = root / "code"
        try:
            shutil.copytree(
                CODE_ROOT,
                clone,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".tmp_q0_*", ".tmp_q0_report.xml"),
            )
        except Exception:
            temporary.cleanup()
            raise
        data_root = root / "data"
        try:
            run = run_readiness_dry(code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, test_mode=True)
        except Exception:
            temporary.cleanup()
            raise
        return temporary, clone, data_root, run


class FixtureContractTests(unittest.TestCase):
    def test_fixtures_have_independent_literal_closure(self) -> None:
        screen, confirmation, negatives = load_fixture_sets(CODE_ROOT)
        self.assertEqual(32, len(screen))
        self.assertEqual(96, len(confirmation))
        self.assertEqual(16, sum(task["role"] == "coordinator" for task in screen))
        self.assertEqual(16, sum(task["role"] == "publisher" for task in screen))
        self.assertEqual(32, sum(task["role"] == "coordinator" for task in confirmation))
        self.assertEqual(64, sum(task["role"] == "publisher" for task in confirmation))
        self.assertEqual(17, len(negatives))
        self.assertEqual(
            [
                "bad_json",
                "duplicate_key",
                "nan_value",
                "infinity_value",
                "empty_output",
                "missing_public_field",
                "extra_internal_field",
                "nested_extra_field",
                "role_mismatch",
                "episode_mismatch",
                "decision_mismatch",
                "valid_reject",
                "wrong_role_valid_reject",
                "wrong_episode_valid_reject",
                "wrong_public_value",
                "provider_exception",
                "provider_timeout",
            ],
            [case["case_id"] for case in negatives],
        )
        tasks = screen + confirmation
        self.assertEqual(128, len({task["task_id"] for task in tasks}))
        self.assertEqual(128, len({task["episode_id"] for task in tasks}))
        self.assertEqual(128, len({task["expected_public_report"]["vendor_id"] for task in tasks}))
        for index, task in enumerate(tasks, start=1):
            self.assertEqual(literal_public_report(index), task["expected_public_report"])

    def test_strict_parser_rejects_required_negative_forms(self) -> None:
        screen, _, _ = load_fixture_sets(CODE_ROOT)
        task = screen[0]
        schema = load_schema(CODE_ROOT)
        provider = DeterministicFakeProvider()
        for kind in (
            "malformed_json",
            "duplicate_key",
            "nan_value",
            "infinity_value",
            "empty_output",
            "missing_public_field",
            "extra_internal_field",
            "nested_extra_field",
            "role_mismatch",
            "episode_mismatch",
            "decision_mismatch",
            "wrong_public_value",
        ):
            reply = provider.invoke(task, "fixture prompt", kind)
            with self.assertRaises(QualificationError, msg=kind):
                validate_response(reply.raw_output, task, schema)

    def test_contract_valid_reject_is_distinct_from_malformed_output(self) -> None:
        screen, _, _ = load_fixture_sets(CODE_ROOT)
        task = screen[0]
        schema = load_schema(CODE_ROOT)
        reply = DeterministicFakeProvider().invoke(task, "fixture prompt", "valid_reject")
        parsed = parse_contract(reply.raw_output, schema)
        self.assertEqual("reject", parsed["decision"])
        self.assertIn("rejection_reason", parsed)
        self.assertNotIn("public_report", parsed)
        with self.assertRaises(QualificationError) as raised:
            validate_response(reply.raw_output, task, schema)
        self.assertEqual("semantic_reject", raised.exception.code)

    def test_contract_valid_reject_identity_mismatches_are_not_semantic_reject(self) -> None:
        screen, _, _ = load_fixture_sets(CODE_ROOT)
        task = screen[0]
        schema = load_schema(CODE_ROOT)
        provider = DeterministicFakeProvider()
        for kind, expected_code in (
            ("wrong_role_valid_reject", "role_mismatch"),
            ("wrong_episode_valid_reject", "episode_mismatch"),
            ("role_mismatch", "role_mismatch"),
            ("episode_mismatch", "episode_mismatch"),
        ):
            reply = provider.invoke(task, "fixture prompt", kind)
            if kind.endswith("valid_reject"):
                parsed = parse_contract(reply.raw_output, schema)
                self.assertEqual("reject", parsed["decision"])
                self.assertNotIn("public_report", parsed)
            with self.assertRaises(QualificationError, msg=kind) as raised:
                validate_response(reply.raw_output, task, schema)
            self.assertEqual(expected_code, raised.exception.code)
            self.assertNotEqual("semantic_reject", raised.exception.code)

    def test_reject_branch_remains_strict(self) -> None:
        schema = load_schema(CODE_ROOT)
        invalid_rejects = (
            '{"role":"coordinator","episode_id":"P1V2Q-SCREEN-001","decision":"reject","rejection_reason":"   "}',
            '{"role":"coordinator","episode_id":"P1V2Q-SCREEN-001","decision":"reject","rejection_reason":"reason","public_report":{}}',
            '{"role":"invalid","episode_id":"P1V2Q-SCREEN-001","decision":"reject","rejection_reason":"reason"}',
        )
        for raw_output in invalid_rejects:
            with self.assertRaises(QualificationError, msg=raw_output):
                parse_contract(raw_output, schema)


class ReadinessAndValidationTests(IsolatedQ0Mixin, unittest.TestCase):
    def test_full_q0_dry_run_validation_and_replay(self) -> None:
        temporary, clone, data_root, run = self.isolated_run()
        try:
            self.assertTrue(run["ok"])
            self.assertEqual(145, run["fake_provider_calls"])
            run_id = run["run_id"]
            validation = validate_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, test_mode=True)
            self.assertTrue(validation["passed"], validation["errors"])
            replay = replay_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, test_mode=True)
            self.assertTrue(replay["passed"], replay["errors"])
            final_validation = validate_run(
                run_id,
                code_root=clone,
                data_root=data_root,
                workspace_root=WORKSPACE_ROOT,
                write_audit=False,
                test_mode=True,
            )
            self.assertTrue(final_validation["passed"], final_validation["errors"])
            paths = artifact_paths(data_root, run_id)
            self.assertTrue(paths["report"].exists())
            self.assertIn("P1v2-Q1 live model calls: not authorized", paths["report"].read_text(encoding="utf-8"))
            events = [json.loads(line) for line in paths["events"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(145, len(events))
            self.assertEqual(128, sum(event["terminal_state"] == "model_output" for event in events))
            self.assertEqual(15, sum(event["terminal_state"] == "model_output_rejected" for event in events))
            self.assertEqual(2, sum(event["terminal_state"] == "model_call_failed" for event in events))
            expected_reject_codes = {
                "valid_reject": "semantic_reject",
                "wrong_role_valid_reject": "role_mismatch",
                "wrong_episode_valid_reject": "episode_mismatch",
            }
            for case_id, expected_code in expected_reject_codes.items():
                reject_events = [event for event in events if event["case_id"] == case_id]
                self.assertEqual(1, len(reject_events))
                self.assertEqual("model_output_rejected", reject_events[0]["terminal_state"])
                self.assertEqual(expected_code, reject_events[0]["failure_code"])
            sink_records = [line for line in paths["public_sink"].read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(80, len(sink_records))
            report_text = paths["report"].read_text(encoding="utf-8")
            for report_line in (
                "- Screen-coordinator: 16",
                "- Screen-publisher: 16",
                "- Confirmation-coordinator: 32",
                "- Confirmation-publisher: 64",
            ):
                self.assertIn(report_line, report_text)
            for case_id, terminal_counts in final_validation["negative_case_terminal_counts"].items():
                for terminal_state, count in terminal_counts.items():
                    self.assertIn(f"- {case_id}: {terminal_state}={count}", report_text)
        finally:
            temporary.cleanup()

    def test_report_registry_rejects_a_second_current_report(self) -> None:
        temporary, clone, data_root, run = self.isolated_run()
        try:
            run_id = run["run_id"]
            replay = replay_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, test_mode=True)
            self.assertTrue(replay["passed"], replay["errors"])
            paths = artifact_paths(data_root, run_id)
            current_report_text = paths["report"].read_text(encoding="utf-8")
            paths["report"].write_text(
                current_report_text.replace("- Screen-coordinator: 16", "- Screen-coordinator: 99"),
                encoding="utf-8",
            )
            tampered_counts = validate_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, write_audit=False, test_mode=True)
            self.assertFalse(tampered_counts["passed"])
            self.assertTrue(any("recomputed count" in error for error in tampered_counts["errors"]))
            paths["report"].write_text(current_report_text, encoding="utf-8")
            historical_id = "P1V2Q-READINESS-DRY-20260801T010101000000Z"
            historical_report = paths["report"].parent / f"P1V2Q0_DELIVERY_{historical_id}.md"
            historical_report.write_text(
                f"- Historical Q0 run ID: `{historical_id}`\n"
                "Historical pre-rework record; not for downstream decisions.\n",
                encoding="utf-8",
            )
            accepted = validate_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, write_audit=False, test_mode=True)
            self.assertTrue(accepted["passed"], accepted["errors"])
            historical_report.write_text(
                f"- Current Q0 run ID: `{historical_id}`\n",
                encoding="utf-8",
            )
            rejected = validate_run(run_id, code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, write_audit=False, test_mode=True)
            self.assertFalse(rejected["passed"])
            self.assertTrue(any("report registry" in error for error in rejected["errors"]))
        finally:
            temporary.cleanup()

    def test_semantic_identity_tamper_fails_even_when_hash_is_repaired(self) -> None:
        temporary, clone, data_root, run = self.isolated_run()
        try:
            paths = artifact_paths(data_root, run["run_id"])
            outcomes = json.loads(paths["outcomes"].read_text(encoding="utf-8"))
            outcomes["p1_go"] = True
            paths["outcomes"].write_text(json.dumps(outcomes, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["artifact_hashes"]["outcomes"] = hashlib.sha256(paths["outcomes"].read_bytes()).hexdigest()
            paths["manifest"].write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            result = validate_run(run["run_id"], code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, write_audit=False, test_mode=True)
            self.assertFalse(result["passed"])
            self.assertTrue(any("identity field p1_go" in item for item in result["errors"]))
        finally:
            temporary.cleanup()

    def test_frozen_input_and_entrypoint_drift_fail_closed_in_isolated_copies(self) -> None:
        targets = (
            "configs/q1_protocol.json",
            "schemas/response_contract.schema.json",
            "prompts/public_response_contract.txt",
            "fixtures/screen_tasks.json",
            "permissions/public_sink_rule.json",
            "scripts/run_readiness_dry.py",
        )
        for relative in targets:
            temporary, clone, data_root, run = self.isolated_run()
            try:
                path = clone / relative
                path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                result = validate_run(run["run_id"], code_root=clone, data_root=data_root, workspace_root=WORKSPACE_ROOT, write_audit=False, test_mode=True)
                self.assertFalse(result["passed"], relative)
            finally:
                temporary.cleanup()

    def test_each_frozen_protocol_boundary_rejects_a_legal_but_contradictory_value(self) -> None:
        changes = (
            ("identity.is_new_p1_run", True),
            ("identity.p1_go", True),
            ("identity.qualification_candidate_status", "qualified"),
            ("future_endpoint", "http://127.0.0.1:11435/api/chat"),
            ("screen_wall_clock_limit_minutes", 21),
            ("concurrency", 2),
            ("retry_per_call", 1),
        )
        for key, replacement in changes:
            temporary, clone, data_root, run = self.isolated_run()
            try:
                config_path = clone / "configs" / "q1_protocol.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                target = config["identity"] if key.startswith("identity.") else config
                target[key.removeprefix("identity.")] = replacement
                config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                result = validate_run(
                    run["run_id"],
                    code_root=clone,
                    data_root=data_root,
                    workspace_root=WORKSPACE_ROOT,
                    write_audit=False,
                    test_mode=True,
                )
                self.assertFalse(result["passed"], key)
            finally:
                temporary.cleanup()

    def test_safe_path_policy_rejects_credential_unc_device_traversal_and_external_inputs(self) -> None:
        cases = (
            CODE_ROOT / ".env",
            Path("..") / "outside.json",
            r"\\server\share\input.json",
            r"\\.\PhysicalDrive0.json",
            WORKSPACE_ROOT / "paperAlpha" / "pre_exp1" / "p1v2_benign" / "configs" / "readiness.json",
        )
        for candidate in cases:
            with self.assertRaises(QualificationError):
                safe_read_path(candidate, CODE_ROOT)

    def test_public_cli_bad_run_id_is_nonzero_and_does_not_emit_traceback(self) -> None:
        command = [
            sys.executable,
            "-B",
            str(CODE_ROOT / "scripts" / "validate_readiness_dry.py"),
            "--run-id",
            "..\\forbidden",
        ]
        completed = subprocess.run(command, cwd=WORKSPACE_ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn('"passed":false', completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_public_execution_rejects_injected_test_roots_before_artifact_io(self) -> None:
        temporary, clone, data_root, run = self.isolated_run()
        try:
            result = validate_run(
                run["run_id"],
                code_root=clone,
                data_root=data_root,
                workspace_root=WORKSPACE_ROOT,
                write_audit=True,
                test_mode=False,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(["validator failed closed: execution_root_invalid"], result["errors"])
            self.assertFalse(artifact_paths(data_root, run["run_id"])["validation"].exists())
        finally:
            temporary.cleanup()

    def test_workspace_root_injection_is_rejected_before_reference_or_protected_tree_io(self) -> None:
        temporary, clone, data_root, run = self.isolated_run()
        try:
            result = validate_run(
                run["run_id"],
                code_root=clone,
                data_root=data_root,
                workspace_root=Path(temporary.name) / "untrusted_workspace",
                write_audit=False,
                test_mode=True,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(["validator failed closed: workspace_root_invalid"], result["errors"])
        finally:
            temporary.cleanup()


class ReferenceIdentityTests(IsolatedQ0Mixin, unittest.TestCase):
    def test_reference_source_drift_is_detected_in_an_isolated_source_copy(self) -> None:
        temporary, clone, _, _ = self.isolated_run()
        try:
            source_workspace = Path(temporary.name) / "reference_workspace"
            for relative, _ in REFERENCE_SOURCE_SPECS:
                source = WORKSPACE_ROOT / relative
                destination = source_workspace / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            self.assertEqual(4, len(verify_reference_sources(clone, source_workspace)))
            changed_source = source_workspace / REFERENCE_SOURCE_SPECS[0][0]
            changed_source.write_text(changed_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(QualificationError):
                verify_reference_sources(clone, source_workspace)
        finally:
            temporary.cleanup()
