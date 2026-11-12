"""Frozen fixed-branch scoring service; accepts prior evidence, never outcomes."""
import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import sha
from paperalpha_monitor.branch_benchmark import sample_from_prefix, classifier_distribution, rule_distribution
from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
from paperalpha_monitor.publisher_view_bn import PublisherViewPrivacyBN

EXPECTED_FAMILIES = {"branch", "branch_uncoupled", "pooled", "frozen_source_pooled", "logistic", "histgb",
                     "persistence_shared_gate", "visible_compliance_shared_gate"}

class FrozenBranchPilot:
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "FROZEN.json").read_text(encoding="utf-8"))
        if self.manifest["schema"] != "fixed-branch-pilot-monitor-v1":
            raise ValueError("frozen branch pilot bundle required")
        if set(self.manifest["selected"]) != EXPECTED_FAMILIES or set(self.manifest["entries"]) != EXPECTED_FAMILIES:
            raise ValueError("all eight frozen comparison methods required")
        for name, expected in self.manifest["files_sha256"].items():
            if sha(self.bundle / name) != expected:
                raise ValueError("frozen artifact changed")
        self.models = {}
        for family, entry in self.manifest["entries"].items():
            if entry["family"] != family or entry["id"] != self.manifest["selected"][family]:
                raise ValueError("selected family binding differs")
            filename = entry["file"]
            if filename is None:
                if family not in {"persistence_shared_gate", "visible_compliance_shared_gate"}:
                    raise ValueError("unknown frozen rule")
                self.models[family] = None
            elif filename.endswith(".joblib"):
                self.models[family] = joblib.load(self.bundle / filename)
            else:
                saved = json.loads((self.bundle / filename).read_text(encoding="utf-8"))
                cls = BranchPublisherPrivacyBN if saved["model"] == BranchPublisherPrivacyBN.MODEL_NAME else PublisherViewPrivacyBN
                self.models[family] = cls.restore(saved)

    def score(self, message):
        if set(message) != {"observation", "prefix_sequence"} or type(message["prefix_sequence"]) is not int:
            raise ValueError("strict prior-only request required")
        sample = sample_from_prefix({"observation": message["observation"], "seq": message["prefix_sequence"]})
        distributions = {}
        for family, model in self.models.items():
            filename = self.manifest["entries"][family]["file"]
            if model is None:
                dist = rule_distribution(sample, family).tolist()
            elif filename.endswith(".joblib"):
                dist = classifier_distribution(sample, model).tolist()
            else:
                bound = sample if isinstance(model, BranchPublisherPrivacyBN) else sample.release
                dist = model.predict(bound)["violation_count_probabilities"]
            distributions[family] = dist
        return {"schema": "frozen-branch-prefix-score-v1", "bundle_sha256": sha(self.bundle / "FROZEN.json"),
            "selected": self.manifest["selected"], "count_distributions": distributions,
            "scores": {str(k): {name: float(sum(dist[k:])) for name, dist in distributions.items()} for k in (1, 2)},
            "available_through_prefix": message["prefix_sequence"], "automatic_blocking": False,
            "calibrated": False, "scope": "Frozen two-field, two-worker final-release forecast; no post-prefix inputs or refitting"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    monitor = FrozenBranchPilot(args.bundle)
    print(json.dumps({"ready": True, "selected": monitor.manifest["selected"]}), flush=True)
    for line in sys.stdin:
        try:
            print(json.dumps({"ok": True, "result": monitor.score(json.loads(line))}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)


if __name__ == "__main__":
    main()
