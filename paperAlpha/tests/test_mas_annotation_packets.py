import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_mas_annotation_packets_v1.py"
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_mas_annotation_packets_v1 import agreement, prepare  # noqa: E402


def test_prepare_removes_evaluator_fields_and_agreement(tmp_path):
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps({"episode_id": "e1", "events": [{"event_id": "x", "label": 1}],
                                  "label": 1, "scenario_role": "attack", "public": {"role": "worker"}}) + "\n", encoding="utf-8")
    out = tmp_path / "packets"
    prepare(source, out, seed=7)
    a = json.loads((out / "annotator_a.jsonl").read_text(encoding="utf-8"))
    observable = json.dumps(a["observable_trace"])
    assert "label" not in observable
    assert "scenario_role" not in observable
    assert "episode_id" not in observable
    assert "request_id" not in observable
    assert "scenario_id" not in observable
    assert "episode_id" not in a
    b_path = tmp_path / "b.jsonl"
    row_b = dict(a)
    row_b["annotation"] = {"risk_label": 1, "first_violation_index": 0,
                            "violation_class": "scope", "evidence_event_ids": ["x"],
                            "rationale": "The evidence shows a scope violation."}
    a["annotation"] = dict(row_b["annotation"])
    (out / "annotator_a.jsonl").write_text(json.dumps(a) + "\n", encoding="utf-8")
    row_b["annotation"]["risk_label"] = 0
    (b_path).write_text(json.dumps(row_b) + "\n", encoding="utf-8")
    report = agreement(out / "annotator_a.jsonl", b_path, tmp_path / "agreement.json")
    assert report["n"] == 1
    assert report["raw_agreement"] == 0.0
    assert report["cohen_kappa"] == 0.0


def test_agreement_rejects_unlabeled_packets(tmp_path):
    row = {"packet_id": "x", "annotation": {}}
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(row) + "\n", encoding="utf-8")
    b.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        agreement(a, b, tmp_path / "out.json")
