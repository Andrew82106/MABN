"""Post-fit development diagnosis, never model selection or new API evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np


def run(fit, out):
    if out.exists():
        raise ValueError("new diagnostic directory required")
    sys.path[:0] = [str(fit / "source_snapshot/code/src"), str(fit / "source_snapshot/code/commands")]
    from collect_privacy_transfer_api import sha, write
    from train_branch_publisher import load_data, restore_bn
    report = json.loads((fit / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        assert sha(fit / name) == expected
    selected = json.loads((fit / "SELECTED.json").read_text(encoding="utf-8"))["entries"]["branch"]
    model = restore_bn(fit / selected["file"])
    rows, samples = load_data(fit)
    all_samples = samples["train"] + samples["validation"]
    before = np.array([model.predict(s)["violation_count_probabilities"] for s in all_samples])
    final_boost = float(model.theta[model.presence.boost_start + 1])
    # Hold all final-emission and authority parameters fixed. Perturb only
    # worker priors/emissions, nonowner shift and U -> V association. These
    # artificial vectors are mechanism checks, not re-fitted predictors.
    changes = []
    if final_boost == 0:
        for seed in range(4):
            altered = copy.deepcopy(model)
            rng = np.random.default_rng(seed)
            d, k, boost = model.presence.d, model.presence.k, model.presence.boost_start
            positions = [*range(d), *range(2 * d, 2 * d + 2 * k), boost, model.n - 1, model.n + 5]
            altered.theta[positions] = rng.uniform(-4, 4, len(positions))
            altered.theta[boost] = 3.
            after = np.array([altered.predict(s)["violation_count_probabilities"] for s in all_samples])
            changes.append({"seed": seed, "changed_parameter_count": len(positions),
                            "max_prediction_difference": float(np.max(abs(before - after)))})
        assert max(c["max_prediction_difference"] for c in changes) < 1e-12
    facts = [len(s.release.graph.facts) for s in all_samples]
    result = {"fit_report_sha256": sha(fit / "REPORT.json"), "diagnostic_source_sha256": sha(Path(__file__)),
        "selected_model": selected["id"], "selected_final_state_boost": final_boost,
        "worker_to_publisher_association": float(model.theta[model.n - 1]),
        "state_route_closed_at_selected_fit": final_boost == 0,
        "observed_worker_matrices": sorted({tuple(map(int, s.worker_presence.ravel())) for s in all_samples}),
        "authority_fact_counts": facts, "unseen_authority_priors_identified_here": False,
        "perturbation_checks": changes, "real_development_runs": len(all_samples),
        "distinct_template_count": len({r["template_id"] for subset in rows.values() for r in subset}),
        "interpretation": "Selected final-state boost is exactly zero: latent worker/publisher states do not affect final appearance at this fit. One observed branch arrangement and deterministic cache-following behavior do not establish branch-model benefit. Known authority does not train unknown-fact priors.",
        "next_evidence_requirements": [
            "Independent real workflow/behavior variation, not solely the same two diagonal occurrence rows",
            "Predeclared cases with unchanged local pooled evidence but meaningful cross-agent provenance/permission relations",
            "Evaluate generalization and false alarms against same-evidence rules/classifiers; do not seek only tasks engineered for BN",
            "Arbitrary workflow and multi-risk template composition remain project requirements beyond this privacy component"],
        "test_outputs_used": False, "parameters_replaced": False, "new_api_calls": 0,
        "scope": "Post-selection explanation using development inputs and constructed parameter perturbations only"}
    write(out / "REPORT.json", result)
    write(out / "source.json", {"source": Path(__file__).read_text(encoding="utf-8")})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.fit.resolve(), args.out.resolve())
