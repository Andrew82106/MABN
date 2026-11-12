"""Independent frozen dual-view replay, fit gradients and executable graphs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify(run, out):
    if out.exists():
        raise ValueError("new verifier directory required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from collect_privacy_transfer_api import sha, write
    from study_publisher_view_bn import training_data
    from publisher_view_fixtures import fixture
    from scalar_binary_graph import evaluate
    import paperalpha_monitor.publisher_view_bn as numerical
    from paperalpha_monitor.scoped_privacy_monitor import compile_observation
    assert Path(numerical.__file__).resolve() == run / "source_snapshot/code/src/paperalpha_monitor/publisher_view_bn.py"
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        assert sha(run / name) == expected, name
    objective_errors, gradients, graph_errors = [], [], []
    models = {}
    for model_file, data_file in (("visible_coefficients_fit.json", "frequency_rows.json"), ("joint_fit.json", "joint_control_rows.json")):
        model = numerical.PublisherViewPrivacyBN.restore(json.loads((run / model_file).read_text(encoding="utf-8")))
        models[model_file] = model
        xs, fs, a, w = training_data(json.loads((run / data_file).read_text(encoding="utf-8")))
        loss, grad = model.loss_gradient(model.theta, xs, fs, a, weights=w)
        objective_errors.append(abs(loss - model.optimizer["loss"]))
        for seed in range(4):
            direction = np.random.default_rng(seed).normal(size=len(model.theta))
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            plus = model.loss_gradient(model.theta + eps * direction, xs, fs, a, weights=w)[0]
            minus = model.loss_gradient(model.theta - eps * direction, xs, fs, a, weights=w)[0]
            gradients.append(abs((plus - minus) / (2 * eps) - grad @ direction))
    for saved in json.loads((run / "graph_examples.json").read_text(encoding="utf-8")):
        model = models["visible_coefficients_fit.json"]
        sample = fixture(**saved["fixture"])
        result = model.predict(sample, at_least=saved["target"])
        assert normalized(result) == saved["prediction"]
        graph = model.probability_graph(sample, at_least=saved["target"])
        assert normalized(graph) == saved["graph"]
        graph_errors.append(abs(evaluate(graph) - result["model_probability"]))
    replayed, instances = 0, 0
    for saved in json.loads((run / "replayed_inputs.json").read_text(encoding="utf-8")):
        source = numerical.PublisherInput(**saved["publisher_input"])
        sample = numerical.ViewSeparatedRelease(compile_observation(saved["observation"]), source, trusted_transport_observers=["api-request-journal"])
        assert saved["old_prediction_sequence"] < saved["new_input_available_sequence"] < saved["response_sequence"]
        assert source.captured_journal_sequence == source.prefix_journal_sequence == saved["new_input_available_sequence"]
        for entry in saved["models"]:
            model = numerical.PublisherViewPrivacyBN.restore(entry["parameters"])
            assert np.all(model.theta[-2:] == 0) and model.optimizer is None
            for output in entry["results"]:
                result = model.predict(sample, at_least=output["target"])
                graph = model.probability_graph(sample, at_least=output["target"])
                assert normalized(result) == output["result"] and normalized(graph) == output["graph"]
                graph_errors.append(abs(evaluate(graph) - result["model_probability"]))
                replayed += 1
        instances += 1
    assert max(objective_errors) < 1e-10 and max(gradients) < 1e-7 and max(graph_errors) < 1e-12
    result = {"report_sha256": sha(run / "REPORT.json"), "auditor_sha256": sha(Path(__file__)),
        "artifacts_checked": len(report["artifacts_sha256"]), "max_objective_error": max(objective_errors),
        "directional_gradients": len(gradients), "max_directional_gradient_error": max(gradients),
        "graphs_recomputed": len(graph_errors), "max_executable_graph_error": max(graph_errors),
        "real_input_views_recomputed": instances, "real_input_scores_recomputed": replayed,
        "normalized_records_exact": True, "scope": "Frozen controlled fitting and offline input replay; no new API, refit or claimed detection improvement"}
    write(out / "REPORT.json", result)
    shutil.copyfile(Path(__file__), out / "auditor_source.py")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
