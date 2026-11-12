import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from evaluate_prefix_forecast_v1 import (  # noqa: E402
    choose_alarm_threshold, evaluate_scores, fit_models, flatten_training,
    predict_trajectories, resolve_manifest_splits,
)
from paperalpha_monitor.streaming_bn import replay, topology_group  # noqa: E402
import evaluate_prefix_forecast_v1 as experiment  # noqa: E402


def fixture_trace(name, extra=0, mismatch=False):
    events = [{"sequence": 0, "actor": "a", "event_type": "input_received", "data_class": "untrusted"}]
    events += [{"sequence": k + 1, "actor": "a", "target": "b", "event_type": "message_sent", "data_class": "untrusted"} for k in range(extra)]
    events += [{"sequence": len(events), "actor": "b", "event_type": "tool_call", "tool": "shell.exec", "permission_mismatch": mismatch}]
    return {"episode_id": name, "events": events}


def test_prefix_weights_are_equal_per_episode_not_per_event():
    trajectories = [replay(fixture_trace("short")), replay(fixture_trace("long", 5))]
    rows, y, weight = flatten_training(trajectories, [0, 1])
    cut = len(trajectories[0])
    assert len(rows) == 9
    assert weight[:cut].sum() == pytest.approx(1)
    assert weight[cut:].sum() == pytest.approx(1)
    np.testing.assert_equal(y, [0] * cut + [1] * len(trajectories[1]))


@pytest.mark.parametrize("maxima, alpha", [([.2, .2, .2, .8], .25), ([.4] * 10, .1), ([0, 1], 0), ([.1, .2], 1)])
def test_threshold_is_lowest_feasible_float_and_handles_ties(maxima, alpha):
    selected = choose_alarm_threshold(maxima, alpha)
    threshold = selected["threshold"]
    assert np.mean(np.asarray(maxima) >= threshold) <= alpha
    if threshold > 0:
        assert np.mean(np.asarray(maxima) >= np.nextafter(threshold, -np.inf)) > alpha


def test_all_negative_threshold_above_one_is_a_decision_not_probability():
    selected = choose_alarm_threshold([1.0] * 3, 0.10)
    assert selected["threshold"] == np.nextafter(1.0, np.inf)
    assert selected["validation_negative_alarm_n"] == 0


@pytest.mark.parametrize("maxima", [[], [np.nan], [-.1], [1.1]])
def test_invalid_validation_maxima_rejected(maxima):
    with pytest.raises(ValueError, match="maxima"):
        choose_alarm_threshold(maxima)


def test_episode_brier_and_strict_warning_not_terminal_alarm():
    traces = [fixture_trace("negative"), fixture_trace("positive", 2, True)]
    scores = [np.array([.1, .1]), np.array([.2, .2, .2, .9])]
    result = evaluate_scores([0, 1], scores, traces, .8)
    assert result["prefix_brier_episode_equal"] == pytest.approx((.01 + (.64 * 3 + .01) / 4) / 2)
    assert result["any_alarm_false_positive_rate"] == 0
    assert result["any_alarm_recall"] == 1
    assert result["warning_proxies"]["first_tool_call"]["recall"] == 0
    assert result["actual_violation_lead_time"] is None
    scores[1][2] = .9
    assert evaluate_scores([0, 1], scores, traces, .8)["warning_proxies"]["first_tool_call"]["recall"] == 1


def test_early_negative_alarm_counts_even_if_terminal_score_is_low():
    trace = fixture_trace("negative")
    result = evaluate_scores([0], [np.array([.9, .1])], [trace], .8)
    assert result["any_alarm_false_positive_rate"] == 1
    assert result["terminal"]["f1"] == 0


def test_source_manifest_exact_ids_and_topology_isolation():
    traces, parts = [], {}
    for i, name in enumerate(("fit", "validation", "test")):
        nodes = [f"a{k}" for k in range(i + 2)]
        workflow = {"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))}
        traces.append({"episode_id": name, "workflow_dag": workflow})
        parts[name] = {"episode_ids": [name], "groups": [topology_group(workflow)]}
    manifest = {"splits": {"123": parts}}
    splits, _ = resolve_manifest_splits(manifest, list(reversed(traces)))
    assert splits[123]["fit"].tolist() == [2]
    parts["test"]["episode_ids"] = ["fit"]
    with pytest.raises(ValueError, match="disjoint"):
        resolve_manifest_splits(manifest, traces)


def test_small_fixture_trains_all_four_methods_with_balanced_prefixes():
    traces = [fixture_trace(str(i), i % 3, bool(i % 2)) for i in range(8)]
    trajectories = [replay(t) for t in traces]
    config = {"training": {"regularization": .01}, "baselines": {
        "logistic_prefix": {"C": 1., "max_iter": 100},
        "histgb_prefix": {"max_iter": 2, "min_samples_leaf": 1, "early_stopping": False},
    }}
    models, training = fit_models(trajectories, np.arange(8) % 2, config, 123)
    assert set(models) == {"terminal_bn", "prefix_bn", "logistic_prefix", "histgb_prefix"}
    assert training["prefix_weight_sum"] == pytest.approx(8)
    scaler = models["logistic_prefix"].named_steps["standardscaler"]
    assert scaler.n_samples_seen_ == pytest.approx(8)
    for model in models.values():
        predictions = predict_trajectories(model, trajectories[:2])
        assert [len(p) for p in predictions] == [len(t) for t in trajectories[:2]]
        assert all(np.isfinite(p).all() for p in predictions)


def test_tiny_cli_preserves_splits_predictions_sources_and_refuses_overwrite(tmp_path, monkeypatch):
    original_root = experiment.ROOT
    shutil.copytree(original_root / 'code/src/paperalpha_monitor', tmp_path / 'code/src/paperalpha_monitor')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"reference_manifest": "reference.json", "data": "data", "expected_seeds": [123],
              "settings": [{"workflow": "runtime_only", "permission": "observed"}],
              "forecast_target": "terminal outcome given prefix", "terminal_bn_target": "terminal only",
              "training": {"regularization": .01}, "baselines": {
                  "logistic_prefix": {"C": 1., "max_iter": 100},
                  "histgb_prefix": {"max_iter": 2, "min_samples_leaf": 1, "early_stopping": False}},
              "alarm_threshold": {"validation_negative_any_alarm_fpr": .1}}
    traces, label_rows, parts = [], [], {}
    for size, part in enumerate(("fit", "validation", "test"), start=2):
        nodes = [f"a{k}" for k in range(size)]
        workflow = {"nodes": nodes, "edges": list(zip(nodes, nodes[1:]))}
        ids = []
        for label in (0, 1):
            name = f"{part}-{label}"
            trace = fixture_trace(name, label, bool(label))
            trace["workflow_dag"] = workflow
            traces.append(trace)
            label_rows.append({"episode_id": name, "label": label})
            ids.append(name)
        parts[part] = {"episode_ids": ids, "groups": [topology_group(workflow)]}
    for name, records in (("traces_public.jsonl", traces), ("labels.jsonl", label_rows)):
        (data_dir / name).write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    reference = {"config": config, "splits": {"123": parts},
                 "input_sha256": {f"data/{name}": experiment.digest(data_dir / name)
                                  for name in ("traces_public.jsonl", "labels.jsonl")}}
    experiment.save_json(tmp_path / "reference.json", reference)
    config_file = tmp_path / "config.json"
    experiment.save_json(config_file, config)
    output = tmp_path / "run"
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluate_prefix_forecast_v1.py", "--config", str(config_file), "--out", str(output)])
    experiment.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"] == reference["splits"]
    assert manifest["reference_manifest_sha256"] == experiment.digest(tmp_path / "reference.json")
    assert (output / "source_snapshot/code/src/paperalpha_monitor/streaming_bn.py").is_file()
    run_dir = output / "runtime_only__observed/123"
    records = experiment.read_jsonl(run_dir / "predictions.jsonl")
    assert [r["episode_id"] for r in records] == ["validation-0", "test-0", "test-1"]
    assert all(set(record["prefix_scores"]) == {"terminal_bn", "prefix_bn", "logistic_prefix", "histgb_prefix"} for record in records)
    assert len(list(run_dir.glob("*.joblib"))) == 4
    with pytest.raises(FileExistsError, match="overwrite"):
        experiment.main()
