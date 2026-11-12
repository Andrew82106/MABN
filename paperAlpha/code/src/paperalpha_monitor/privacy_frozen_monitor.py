"""Same-information, frozen scoring bundle for new identity-free prefixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .privacy_model_independent import ModelIndependentPrivacyBN, graph_features, direct_features, BASELINE_NAMES, validate
from .privacy_supervision_controls import object_predictions


class FrozenPrivacyMonitor:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "FROZEN.json").read_text(encoding="utf-8"))
        if self.manifest["schema"] != "frozen-model-independent-privacy-v1" or self.manifest["baseline_names"] != list(BASELINE_NAMES):
            raise ValueError("frozen input/estimator specification mismatch")
        for name, expected in self.manifest["model_files_sha256"].items():
            if hashlib.sha256((self.folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("frozen model changed")
        self.bn = {name: ModelIndependentPrivacyBN.restore(json.loads((self.folder / file).read_text(encoding="utf-8")))
            for name, file in self.manifest["selected"]["bn"].items()}
        if any(not m.optimizer or not m.optimizer["success"] for m in self.bn.values()):
            raise ValueError("trained estimators required for runtime scoring")
        self.classifiers = {s: {name: joblib.load(self.folder / file) for name, file in items.items()}
            for s, items in self.manifest["selected"]["classifiers"].items()}

    def score(self, packet):
        validate(packet)
        stage = str(packet["stage"])
        scores, explanations = {}, {}
        for name, model in self.bn.items():
            detail = model.predict_details([packet])
            scores[name] = float(detail["probabilities"][0])
            explanations[name] = {"state_pairs": detail["state_pairs"], "state_weights": detail["state_weights"][0].tolist(),
                "state_conditional_risk": detail["state_conditional_risk"][0].tolist(), "field_marginals": detail["field_marginals"][0]}
        x = graph_features([packet])
        for name, estimator in self.classifiers[stage].items():
            if name == "field_histgb":
                data = direct_features([packet])
                q = np.zeros(data["mask"].shape)
                q[data["mask"]] = estimator.predict_proba(data["design"][data["mask"]])[:, 1]
                scores[name] = float(object_predictions(q, data["mask"])[0])
            else:
                scores[name] = float(estimator.predict_proba(x)[0, 1])
        scores["training_prevalence"] = self.manifest["training_prevalence"]
        latest = "coordinator_present" if stage == "1" else "worker_present"
        scores["current_presence"] = float(any(g[latest] for g in packet["groups"]))
        return {"cutoff": int(stage), "scores": scores, "explanations": explanations, "model_identity_used": False,
            "target": "Later publisher response contains a source-policy-prohibited literal value", "calibrated": False,
            "automatic_blocking": False, "scope": "Fixed phase structure and configured initial vault/allowlist; not general safety."}
