"""Explicit annotation binding regressions; temporary synthetic text only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/"src/annotation9.py"
spec = importlib.util.spec_from_file_location("annotation9_test_module", PATH)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnnotationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(a, "ROOT", self.root)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.row = {"row_id": "synthetic", "question_id": "q", "group_id": "q", "split": "train", "condition": "partial"}
        self.item = {"item_id": "synthetic__1", "text": "Bob is 20.", "start": 3, "end": 13, "parse_ok": True}
        self.generated = {"row_id": "synthetic", "response": "1. Bob is 20.", "items": [self.item]}
        (self.root/"data/generation_records").mkdir(parents=True)
        (self.root/"data/inputs.jsonl").write_text(json.dumps(self.row)+"\n", encoding="utf-8")
        self.gp = self.root/"data/generation_records/synthetic.json"
        self.gp.write_text(json.dumps(self.generated), encoding="utf-8")
        self.decision = {"item_id": "synthetic__1", "stance": "asserted", "evidence_relation": "unsupported",
                         "risk": 1, "rationale": "Synthetic source lacks age.", "spans": ["20"]}
        self.dp = self.root/"data/decisions.json"
        a.add(self.dp, [self.decision], "synthetic")

    def test_add_persists_identity_without_mutating_caller(self):
        stored = json.loads(self.dp.read_text(encoding="utf-8"))["decisions"][0]
        self.assertNotIn("source_generation_sha256", self.decision)
        self.assertEqual(stored["source_generation_sha256"], a.sha(self.gp))
        for key in ("text", "start", "end"):
            self.assertEqual(stored[key], self.item[key])
        a.serialize([self.dp], self.root/"data/output.jsonl")
        output = a.readl(self.root/"data/output.jsonl")[0]
        self.assertEqual(output["risk_spans"][0]["text"], "20")
        self.assertEqual(output["risk_spans"][0]["start"], 10)

    def test_same_text_but_old_generation_hash_rejected(self):
        self.generated["metadata_changed"] = True
        self.gp.write_text(json.dumps(self.generated), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "generation changed"):
            a.serialize([self.dp], self.root/"data/output.jsonl")
        self.assertFalse((self.root/"data/output.jsonl").exists())

    def test_changed_item_with_same_risk_quote_rejected(self):
        self.item.update(text="Bob is 20 and tall.", end=22)
        self.generated["response"] = "1. Bob is 20 and tall."
        self.gp.write_text(json.dumps(self.generated), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "original answer"):
            a.serialize([self.dp], self.root/"data/output.jsonl")

    def test_supported_item_still_requires_exact_range(self):
        _, rows = a.load_rows()
        d = json.loads(self.dp.read_text(encoding="utf-8"))["decisions"][0]
        d.update(risk=0, evidence_relation="supported", spans=[])
        rows["synthetic__1"][1]["response"] = "1. BAD TEXT!!"
        with self.assertRaises(AssertionError):
            a.record(d, "synthetic", rows)

    def test_repeated_quote_requires_explicit_occurrence(self):
        with self.assertRaisesRegex(AssertionError, "Quote repeated"):
            a.locate("20 and 20", "20")
        self.assertEqual(a.locate("20 and 20", "20", 1), 7)

    def test_duplicate_decisions_not_merged(self):
        with self.assertRaises(AssertionError):
            a.add(self.dp, [self.decision], "synthetic")
        with self.assertRaisesRegex(AssertionError, "Overlapping"):
            a.serialize([self.dp, self.dp], self.root/"data/output.jsonl")


if __name__ == "__main__":
    unittest.main()
