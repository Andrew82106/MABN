import asyncio
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
import extract_runtime_semantics as extraction
import audit_semantic_extraction as audit
import revalidate_semantic_items as item_derivation


@pytest.fixture(autouse=True)
def isolate_frozen_package():
    yield
    for name in list(sys.modules):
        if name == "paperalpha_semantics_audit_frozen" or name.startswith("paperalpha_semantics_audit_frozen."):
            sys.modules.pop(name)


def execute(tmp_path, monkeypatch, mode="valid", protocol="v1"):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0 and kwargs["base_url"] == "http://localhost:58661/v1"
            self.responses = self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def create(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            if mode == "api_error":
                raise RuntimeError("do not record arbitrary SDK exception text")
            value = {"status": "no_supported_claims", "more_possible": False, "observations": []}
            if mode == "bad_quote":
                value = {"status": "observations", "more_possible": False, "observations": [{
                    "kind": "authorization_claim", "presentation": "direct", "polarity": "affirmed",
                    "quote": "Invented proof of approval.", "occurrence": 0, "references": []}]}
            if mode == "v2_partial":
                current = json.loads(kwargs["input"][1]["content"])["current"]["content"]
                good = {"predicate": "action_requested", "presentation": "direct", "quote": current,
                        "occurrence": 0, "condition_quotes": [], "quantity_quote": None, "references": []}
                bad = {**good, "predicate": "permission_claimed_conditional"}
                value = {"status": "observations", "more_possible": False, "observations": [good, bad]}
            result = {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5},
                      "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
            return SimpleNamespace(model_dump=lambda **_: copy.deepcopy(result))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=Client))
    monkeypatch.setenv("PAPERALPHA_API_BASE", "http://localhost:58661/v1")
    monkeypatch.setenv("PAPERALPHA_API_KEY", "test-fixture-not-a-real-key")
    monkeypatch.setattr(extraction.importlib.metadata, "version", lambda _: "test-double")
    args = SimpleNamespace(input=ROOT / 'code/tests/fixtures/semantic_controls_v1.jsonl', out=tmp_path / "run", limit=2, protocol=protocol)
    result = asyncio.run(extraction.extract(args))
    return args, result, calls


@pytest.mark.parametrize("mode,expected", [("valid", "validated_observation"), ("bad_quote", "invalid_observation"), ("api_error", "api_error")])
def test_saved_request_and_error_protocol_reconstructs(tmp_path, monkeypatch, mode, expected):
    args, result, calls = execute(tmp_path, monkeypatch, mode)
    assert len(calls) == 2 and result["outcome_counts"] == {expected: 2}
    report = audit.audit(args.out, tmp_path / "audit.json")
    assert report["status"] == "passed" and report["reconstructed_prefix_packets"] == 2
    manifest = json.loads((args.out / "RUN.json").read_text(encoding="utf-8"))
    assert manifest["available_text_records"] == 6 and len(manifest["unselected_record_ids"]) == 4
    if mode == "api_error":
        assert result["input_tokens"] == result["output_tokens"] == 0
        assert "do not record arbitrary SDK exception text" not in (args.out / "api_responses.jsonl").read_text()
    with pytest.raises(FileExistsError):
        asyncio.run(extraction.extract(args))
    with pytest.raises(FileExistsError):
        audit.audit(args.out, tmp_path / "audit.json")


def test_hash_modified_output_is_not_accepted(tmp_path, monkeypatch):
    args, _, _ = execute(tmp_path, monkeypatch)
    with (args.out / "extractions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(AssertionError, match="artifact mismatch"):
        audit.audit(args.out, tmp_path / "audit.json")


def test_rehashed_future_leak_in_input_packet_is_detected(tmp_path, monkeypatch):
    args, _, _ = execute(tmp_path, monkeypatch)
    path = args.out / "input_packets.jsonl"
    values = audit.read_rows(path)
    values[0]["prior_records"].append(copy.deepcopy(values[1]["current"]))
    path.write_text("\n".join(json.dumps(v) for v in values) + "\n", encoding="utf-8")
    manifest = json.loads((args.out / "RUN.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["input_packets.jsonl"] = audit.sha(path)
    (args.out / "RUN.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="prefix input packets"):
        audit.audit(args.out, tmp_path / "audit.json")


def test_v2_pipeline_records_protocol_and_uses_frozen_v2_validator(tmp_path, monkeypatch):
    args, result, calls = execute(tmp_path, monkeypatch, protocol="v2")
    assert result["outcome_counts"] == {"validated_observation": 2}
    assert calls[0]["max_output_tokens"] == 3200
    manifest = json.loads((args.out / "RUN.json").read_text(encoding="utf-8"))
    assert manifest["semantic_protocol"] == "v2"
    report = audit.audit(args.out, tmp_path / "audit.json")
    assert report["status"] == "passed"


@pytest.mark.parametrize("mode", ["v2_partial", "api_error"])
def test_saved_item_derivation_no_new_calls_and_preserves_original_run(tmp_path, monkeypatch, mode):
    args, _, calls = execute(tmp_path, monkeypatch, protocol="v2", mode=mode)
    original_hash = audit.sha(args.out / "RUN.json")
    report = item_derivation.derive(args.out, tmp_path / "items")
    assert len(calls) == 2 and report["new_api_calls"] == 0
    assert audit.sha(args.out / "RUN.json") == original_hash == report["source_manifest_sha256"]
    assert report["accepted_items"] == (2 if mode == "v2_partial" else 0)
    assert report["rejected_items"] == (2 if mode == "v2_partial" else 0)
    assert report["retained_from_originally_rejected_records"] == report["accepted_items"]
    for relative, expected in report["artifact_sha256"].items():
        assert audit.sha(tmp_path / "items" / relative) == expected
    with pytest.raises(FileExistsError):
        item_derivation.derive(args.out, tmp_path / "items")


def test_item_derivation_does_not_silently_reinterpret_v1(tmp_path, monkeypatch):
    args, _, _ = execute(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="only typed v2"):
        item_derivation.derive(args.out, tmp_path / "items")


def test_v5_runtime_verifier_consumes_frozen_v2_without_new_api_calls(tmp_path, monkeypatch):
    import verify_text_conditioned_runtime as verifier
    args, _, calls = execute(tmp_path, monkeypatch, protocol="v2", mode="v2_partial")
    report = verifier.verify(args.out, tmp_path / "v5")
    assert len(calls) == 2 and report["new_api_calls"] == 0
    assert report["text_revisions"] == 6  # Four unselected records remain explicit missing annotations.
    assert report["accepted_candidate_items"] == 2
    assert report["default_feature_mask_score_change"] == 0
    assert report["max_no_text_vs_v4_error"] < 2e-14
    assert report["all_parameter_terminal_score_gradient_max_error"] < 4e-8
    with pytest.raises(FileExistsError):
        verifier.verify(args.out, tmp_path / "v5")
