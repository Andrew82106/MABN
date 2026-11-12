import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
import evaluate_dynamic_joint_bn_v1 as experiment
from paperalpha_monitor.dynamic_joint_bn import DynamicJointBN, HEAD, OBSERVABLES


SETTING = {"workflow": "runtime_only", "permission": "observed"}


def trace(name="example", mismatch=True):
    return {"episode_id": name, "workflow_dag": {"nodes": ["a", "b"], "edges": [["a", "b"]]}, "events": [
        {"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"},
        {"sequence": 1, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted"},
        {"sequence": 2, "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": mismatch},
    ]}


def test_flat_control_contains_exact_observables_and_chronological_templates():
    episode = experiment.prepare_trace(trace(), SETTING)
    assert episode.flat.shape == (3, 62)
    np.testing.assert_equal(episode.flat[:, :13], [s.observables for s in episode.joint.steps])
    np.testing.assert_equal(episode.flat[:, 13:], [s.vector() for s in episode.templates])


def test_all_feature_paths_mask_permissions_and_unavailable_workflow():
    first, changed = trace(), trace(mismatch=False)
    changed["events"][1]["edge_kind"] = "unexpected"
    setting = {**SETTING, "permission": "masked"}
    a, b = [experiment.prepare_trace(t, setting) for t in (first, changed)]
    np.testing.assert_equal(a.flat, b.flat)
    np.testing.assert_equal(*experiment.predict(DynamicJointBN(), [a, b]))


def test_future_events_do_not_change_flat_or_joint_prefixes():
    full = trace()
    first = {**full, "events": full["events"][:1]}
    a, b = [experiment.prepare_trace(t, SETTING) for t in (first, full)]
    np.testing.assert_equal(a.flat, b.flat[:1])
    np.testing.assert_allclose(experiment.predict(DynamicJointBN(), [a])[0], experiment.predict(DynamicJointBN(), [b])[0][:1], atol=1e-12)


def test_role_metadata_cannot_silently_give_only_joint_model_more_information():
    value = trace()
    value["workflow_dag"]["roles"] = {"a": "reader"}
    with pytest.raises(ValueError, match="matched baseline"):
        experiment.prepare_trace(value, {**SETTING, "workflow": "workflow_known"})


def test_no_latent_head_is_exactly_observable_only_not_hidden_propagation():
    episode = experiment.prepare_trace(trace(), SETTING)
    model = DynamicJointBN(use_latent_head=False)
    model.theta[:HEAD] = np.linspace(-8, 8, HEAD)
    expected = expit(episode.flat[:, :len(OBSERVABLES)] @ model.theta[HEAD:HEAD + len(OBSERVABLES)])
    np.testing.assert_allclose(experiment.predict(model, [episode])[0], expected, atol=1e-12)
    model.theta[:HEAD] *= -1
    np.testing.assert_allclose(experiment.predict(model, [episode])[0], expected, atol=1e-12)


def test_no_messages_disables_state_transport_but_not_visible_message_features():
    episode = experiment.prepare_trace(trace(), SETTING)
    full, local = DynamicJointBN(), DynamicJointBN(use_messages=False)
    assert experiment.predict(full, [episode])[0][1] > experiment.predict(local, [episode])[0][1]
    assert episode.flat[1, 2] > 0  # message count remains observed in both.


def test_small_fixture_fits_all_six_methods_with_saved_probability_shapes():
    config = json.loads((ROOT / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json').read_text())
    config["training"]["max_iter"] = 3
    config["baselines"]["histgb_flat_prefix"].update(max_iter=2, min_samples_leaf=1)
    episodes = [experiment.prepare_trace(trace(str(i), bool(i % 2)), SETTING) for i in range(8)]
    fitted = list(experiment.fit_estimators(episodes, np.arange(8) % 2, config, 123))
    assert [name for name, _, _ in fitted] == list(experiment.METHODS)
    for _, model, seconds in fitted:
        assert seconds >= 0
        scores = experiment.predict(model, episodes)
        assert len(scores) == 8 and all(len(s) == 3 for s in scores)
        assert all(np.isfinite(s).all() for s in scores)
    metadata = experiment.training_diagnostics(episodes)
    assert metadata["prefix_weight_sum"] == 8
    assert metadata["fit_prefix_n"] == 24
    assert metadata["declared_node_roles"] == {"unknown": 16}


def test_frozen_snapshot_includes_evaluator_protocol_core_and_knowledge(tmp_path):
    inventory = experiment.snapshot_sources(tmp_path, ROOT / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json')
    required = {'code/commands/evaluate_dynamic_joint_bn_v1.py', 'code/commands/evaluate_prefix_forecast_v1.py',
                'code/src/paperalpha_monitor/dynamic_joint_bn.py', 'code/src/paperalpha_monitor/template_registry.py',
                'knowledge/templates/risk_templates_v1.json', 'knowledge/templates/joint_risk_templates_v1.json', 'knowledge/risks/risk_catalog.json'}
    assert required <= set(inventory)
    for relative in required:
        assert experiment.protocol.digest(tmp_path / "source_snapshot" / relative) == inventory[relative]["sha256"]


def test_tiny_cli_and_frozen_auditor_replay_all_six_estimators(tmp_path):
    # A fully isolated miniature repository; no changes to the real experiment.
    for relative in ('code/src/paperalpha_monitor',):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    for relative in ('code/commands/evaluate_dynamic_joint_bn_v1.py', 'code/commands/evaluate_prefix_forecast_v1.py',
                     'code/commands/audit_dynamic_joint_bn_run.py', 'code/commands/audit_prefix_forecast_run.py',
                     'code/commands/continue_joint_optimization.py', 'experiments/model_mechanisms/configs/joint_optimizer_continuation_v1.json',
                     'code/commands/evaluate_phase_alarm_policy.py', 'experiments/runtime_alarm/configs/phase_alarm_policy_v1.json',
                     'code/commands/audit_joint_followups.py',
                     'code/commands/evaluate_typed_joint_bn_v2.py', 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json', 'knowledge/templates/typed_joint_bn_v2.json',
                     'code/commands/audit_typed_joint_run.py',
                     'knowledge/templates/risk_templates_v1.json', 'knowledge/templates/joint_risk_templates_v1.json', 'knowledge/risks/risk_catalog.json'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    data = tmp_path / "results/fixture_data"
    data.mkdir(parents=True)
    traces, labels, parts = [], [], {}
    for part, size, nodes in (("fit", 8, ["a", "b"]), ("validation", 4, ["a", "b", "c"]),
                              ("test", 4, ["a", "b", "c", "d"])):
        selected = []
        workflow = {"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))}
        for i in range(size):
            value = trace(part + str(i), bool(i % 2))
            value["workflow_dag"] = workflow
            traces.append(value)
            labels.append({"episode_id": value["episode_id"], "label": i % 2})
            selected.append(value["episode_id"])
        parts[part] = {"episode_ids": selected, "groups": [experiment.protocol.topology_group(workflow)]}
    for name, rows in (("traces_public.jsonl", traces), ("labels.jsonl", labels)):
        (data / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    config = json.loads((ROOT / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json').read_text())
    config.update(data="results/fixture_data", reference_manifest="results/reference.json", settings=[SETTING], expected_seeds=[123])
    config["training"]["max_iter"] = 2
    config["baselines"]["histgb_flat_prefix"].update(max_iter=2, min_samples_leaf=1)
    experiment.protocol.save_json(tmp_path / 'experiments/model_mechanisms/configs/dynamic_joint_bn_v1.json', config)
    experiment.protocol.save_json(tmp_path / config["reference_manifest"], {
        "config": {"data": config["data"], "settings": config["settings"]}, "splits": {"123": parts},
        "input_sha256": {p.relative_to(tmp_path).as_posix(): experiment.protocol.digest(p) for p in data.iterdir()},
    })
    output = tmp_path / "results/run"
    command = [sys.executable, "-B", str(tmp_path / 'code/commands/evaluate_dynamic_joint_bn_v1.py'), "--out", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit_command = [sys.executable, "-B", str(tmp_path / 'code/commands/audit_dynamic_joint_bn_run.py'), "--run", str(output)]
    audited = subprocess.run(audit_command, text=True, capture_output=True, timeout=40)
    assert audited.returncode == 0, audited.stdout + audited.stderr
    result = json.loads((output / "AUDIT.json").read_text())
    assert result["status"] == "passed"
    assert result["checked_estimators"] == 6
    assert result["max_absolute_score_error"] == 0
    continued = tmp_path / "results/continued"
    continuation_config = json.loads((tmp_path / 'experiments/model_mechanisms/configs/joint_optimizer_continuation_v1.json').read_text())
    continuation_config["max_additional_iterations"] = 12
    experiment.protocol.save_json(tmp_path / 'experiments/model_mechanisms/configs/joint_optimizer_continuation_v1.json', continuation_config)
    extended = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/continue_joint_optimization.py'),
                               "--run", str(output), "--out", str(continued)],
                              text=True, capture_output=True, timeout=40)
    assert extended.returncode == 0, extended.stdout + extended.stderr
    decisions = json.loads((continued / "training_decisions.json").read_text())
    assert decisions["complete"] and decisions["checked_n"] == 3
    assert all(row["fit_episode_ids"] == parts["fit"]["episode_ids"] for row in decisions["records"])
    assert all(row["after"]["fit_objective"] <= row["before"]["fit_objective"] for row in decisions["records"])
    frozen = json.loads((continued / "manifest.json").read_text())
    assert frozen["continuation"]["training_decisions_sha256"] == experiment.protocol.digest(continued / "training_decisions.json")
    audited_continuation = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/audit_dynamic_joint_bn_run.py'),
                                           "--run", str(continued)], text=True, capture_output=True, timeout=40)
    assert audited_continuation.returncode == 0, audited_continuation.stdout + audited_continuation.stderr
    typed_config_path = tmp_path / 'experiments/model_mechanisms/configs/typed_joint_evaluation_v2.json'
    typed_config = json.loads(typed_config_path.read_text())
    typed_config.update(reference_run=str(continued), expected_seeds=[123])
    typed_config["training"]["max_iter"] = 3
    experiment.protocol.save_json(typed_config_path, typed_config)
    typed_out = tmp_path / "results/typed"
    typed_run = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/evaluate_typed_joint_bn_v2.py'),
                                "--out", str(typed_out)], text=True, capture_output=True, timeout=40)
    assert typed_run.returncode == 0, typed_run.stdout + typed_run.stderr
    typed_metrics = json.loads((typed_out / "metrics.json").read_text())
    assert typed_metrics["complete"] and len(typed_metrics["runs"][0]["metrics"]) == 7
    assert typed_metrics["runs"][0]["collapsed_head_vs_source_max_probability_error"] < 1e-12
    typed_audit = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/audit_typed_joint_run.py'),
                                  "--run", str(typed_out)], text=True, capture_output=True, timeout=40)
    assert typed_audit.returncode == 0, typed_audit.stdout + typed_audit.stderr
    phase_config = json.loads((tmp_path / 'experiments/runtime_alarm/configs/phase_alarm_policy_v1.json').read_text())
    phase_config["source_run"] = str(continued)
    experiment.protocol.save_json(tmp_path / 'experiments/runtime_alarm/configs/phase_alarm_policy_v1.json', phase_config)
    phase_out = tmp_path / "results/phase"
    phase_run = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/evaluate_phase_alarm_policy.py'),
                                "--out", str(phase_out)], text=True, capture_output=True, timeout=40)
    assert phase_run.returncode == 0, phase_run.stdout + phase_run.stderr
    phase_metrics = json.loads((phase_out / "metrics.json").read_text())
    assert phase_metrics["complete"] and len(phase_metrics["records"]) == 6
    assert not phase_metrics["probability_scores_changed"]
    for mode, run_directory in (("optimization", continued), ("phase", phase_out)):
        followed = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/audit_joint_followups.py'),
                                   mode, "--run", str(run_directory)], text=True, capture_output=True, timeout=40)
        assert followed.returncode == 0, followed.stdout + followed.stderr
    # Verify that an altered calibration artifact is rejected without a report.
    calibration = phase_out / "calibration.json"
    calibration.write_text(calibration.read_text() + "\n", encoding="utf-8")
    failed_audit = subprocess.run([sys.executable, "-B", str(tmp_path / 'code/commands/audit_joint_followups.py'),
                                   "phase", "--run", str(phase_out), "--out", str(phase_out / "BAD_AUDIT.json")],
                                  text=True, capture_output=True, timeout=40)
    assert failed_audit.returncode != 0 and "output hashes differ" in failed_audit.stderr
    assert not (phase_out / "BAD_AUDIT.json").exists()
    again = subprocess.run(command, text=True, capture_output=True, timeout=40)
    assert again.returncode != 0 and "Refusing to overwrite" in again.stderr
    # Changing a frozen source is detected before loading estimator artifacts.
    source = output / "source_snapshot/code/src/paperalpha_monitor/dynamic_joint_bn.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# audit tamper test\n", encoding="utf-8")
    tampered = subprocess.run(audit_command + ["--out", str(output / "AUDIT_tampered.json")], text=True, capture_output=True, timeout=40)
    assert tampered.returncode != 0 and "frozen source changed" in tampered.stderr
    assert not (output / "AUDIT_tampered.json").exists()
