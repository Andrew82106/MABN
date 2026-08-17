from __future__ import annotations

import json

import pytest

from pre_exp1.provenance import (
    EXPECTED_EXECUTABLE,
    EXPECTED_PYTHON_VERSION,
    capture_python_environment,
    relative_hashes,
    source_file_hashes,
)
from pre_exp1.runtime_config import RuntimeConfig
from pre_exp1.state import stable_hash
from pre_exp1.templates import TemplateConfigError, TemplateRegistry


def test_runtime_manifest_inputs_cover_configs_and_shared_files(runtime):
    config_hashes = relative_hashes(
        runtime.config_paths,
        runtime.paths.paper_alpha_root,
    )
    shared_hashes = relative_hashes(
        runtime.shared_input_paths,
        runtime.paths.paper_alpha_root,
    )
    assert set(config_hashes) == {
        "pre_exp1/configs/agents.json",
        "pre_exp1/configs/experiment.json",
        "pre_exp1/configs/graph.json",
        "pre_exp1/configs/treatments.json",
    }
    assert {
        "data/shared/fixtures/vendor_records.json",
        "data/shared/task_templates/vendor_review.json",
        "data/shared/treatment_templates/message_treatments.json",
        "data/shared/schemas/event.schema.json",
        "data/shared/schemas/outcome.schema.json",
        "data/shared/schemas/replay.schema.json",
        "data/shared/schemas/run_manifest.schema.json",
    } == set(shared_hashes)
    assert all(len(value) == 64 for value in config_hashes.values())
    assert all(len(value) == 64 for value in shared_hashes.values())


def test_template_content_change_changes_manifest_input_hash(tamper_project):
    paths, _ = tamper_project
    before_runtime = RuntimeConfig.load(paths)
    before = relative_hashes(
        before_runtime.shared_input_paths,
        paths.paper_alpha_root,
    )
    template_path = (
        paths.shared_data_dir
        / "treatment_templates"
        / "message_treatments.json"
    )
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["templates"]["safe"]["content"]["instruction"] += " Changed."
    payload["templates"]["safe"]["content_hash"] = stable_hash(
        payload["templates"]["safe"]["content"]
    )
    template_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    after_runtime = RuntimeConfig.load(paths)
    after = relative_hashes(
        after_runtime.shared_input_paths,
        paths.paper_alpha_root,
    )
    key = "data/shared/treatment_templates/message_treatments.json"
    assert before[key] != after[key]
    assert (
        before_runtime.templates.hashes()["safe"]
        != after_runtime.templates.hashes()["safe"]
    )


def test_wrong_declared_template_hash_prevents_startup(tamper_project):
    paths, _ = tamper_project
    template_path = (
        paths.shared_data_dir
        / "treatment_templates"
        / "message_treatments.json"
    )
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["templates"]["drop"]["content"]["reason"] = "TAMPERED"
    template_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TemplateConfigError, match="hash mismatch"):
        TemplateRegistry.from_file(template_path)
    with pytest.raises(TemplateConfigError, match="hash mismatch"):
        RuntimeConfig.load(paths)


def test_python_environment_is_captured_from_required_interpreter():
    environment = capture_python_environment(enforce=True)
    assert environment["sys_executable"] == str(EXPECTED_EXECUTABLE)
    assert environment["python_version"] == EXPECTED_PYTHON_VERSION
    assert environment["pytest_version"] == "8.4.2"
    assert environment["source_version"] == "0.2.0"
    assert environment["installed_distribution_version"] == "0.2.0"
    assert "paperalpha-pre-exp1==0.2.0" in environment["dependency_inventory"]
    assert len(environment["dependency_inventory_hash"]) == 64


def test_source_file_hashes_cover_runtime_source_and_execution_scripts(runtime):
    hashes = source_file_hashes(runtime.paths)
    assert "pre_exp1/src/pre_exp1/validation.py" in hashes
    assert "pre_exp1/scripts/run_p0_smoke.py" in hashes
    assert "pre_exp1/scripts/validate_run.py" in hashes
    assert all(path.endswith(".py") for path in hashes)
    assert not any("/tests/" in path for path in hashes)
    assert all(len(value) == 64 for value in hashes.values())
