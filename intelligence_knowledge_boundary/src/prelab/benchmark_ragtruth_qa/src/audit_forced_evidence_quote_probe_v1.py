"""Static and artifact audit for forced-evidence quote CPU preparation.

It does not open fit/calibration/test data.  The gold-bearing fit source is
checked only by its byte hash; the sanitizer's allowlist access is inspected in
the Python AST.  The only row payload opened here is the sanitized manifest.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results/forced_evidence_quote_probe_v1"
RESEARCH = ROOT / "research/forced_evidence_quote_probe_v1"
PREP_CODE = HERE / "prepare_forced_evidence_quote_probe_v1.py"
RUNNER_CODE = HERE / "run_forced_evidence_quote_probe_v1.py"
PLAN_PATH = RESEARCH / "PLAN.json"
PROTOCOL_PATH = RESEARCH / "PROTOCOL.md"

OUTPUT_FIELDS = (
    "response_id", "source_id", "group_id", "model", "question",
    "passage_1", "passage_2", "passage_3", "microclaim_id",
    "microclaim_index", "claim_start", "claim_end", "claim_text_raw",
    "claim_prompt_text",
)
READ_ALLOWLIST = frozenset((
    "response_id", "source_id", "group_id", "partition", "official_split",
    "model", "question", "retrieved_passages",
))
FORBIDDEN = frozenset((
    "original_response", "quality", "labels", "gold_label", "risk_mask",
    "risk_bpe_indices", "risk_bpe_fraction", "label_type", "answer_risk",
    "existing_scores",
))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ("frozenset", "tuple"):
                    return ast.literal_eval(call.args[0])
                return ast.literal_eval(call)
    raise AssertionError(name)


def function_node(tree, name):
    return next(node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def recursively_reject(value):
    if isinstance(value, dict):
        assert not (FORBIDDEN & set(value))
        for child in value.values():
            recursively_reject(child)
    elif isinstance(value, list):
        for child in value:
            recursively_reject(child)


def main():
    prep_text = PREP_CODE.read_text(encoding="utf-8")
    runner_text = RUNNER_CODE.read_text(encoding="utf-8")
    prep_tree, runner_tree = ast.parse(prep_text), ast.parse(runner_text)
    assert frozenset(literal_assignment(prep_tree, "FIT_READ_ALLOWLIST")) == READ_ALLOWLIST
    assert tuple(literal_assignment(prep_tree, "OUTPUT_FIELDS")) == OUTPUT_FIELDS
    assert frozenset(literal_assignment(prep_tree, "FORBIDDEN")) == FORBIDDEN

    read_fit = function_node(prep_tree, "read_and_select_fit")
    raw_subscripts = [ast.unparse(node) for node in ast.walk(read_fit)
                      if isinstance(node, ast.Subscript) and
                      isinstance(node.value, ast.Name) and node.value.id == "raw"]
    assert raw_subscripts == ["raw[key]"], raw_subscripts
    visible_comprehensions = [node for node in ast.walk(read_fit)
                              if isinstance(node, ast.DictComp) and
                              ast.unparse(node.value) == "raw[key]"]
    assert len(visible_comprehensions) == 1
    assert ast.unparse(visible_comprehensions[0].generators[0].iter) == "FIT_READ_ALLOWLIST"
    raw_name_contexts = Counter(type(node.ctx).__name__ for node in ast.walk(read_fit)
                                if isinstance(node, ast.Name) and node.id == "raw")
    assert raw_name_contexts == {"Store": 1, "Load": 1, "Del": 1}, raw_name_contexts

    cli = function_node(runner_tree, "cpu_selfcheck")
    assert "torch.cuda" in ast.unparse(cli)
    skeleton = next(node for node in runner_tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "GPUExecutionSkeleton")
    assert {node.name for node in skeleton.body if isinstance(node, ast.FunctionDef)} == {
        "gpu_smoke", "extract"}
    assert all(any(isinstance(child, ast.Raise) for child in ast.walk(node))
               for node in skeleton.body if isinstance(node, ast.FunctionDef))
    cli_choices = [node for node in ast.walk(runner_tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
    assert any("cpu-selfcheck" in ast.unparse(node) and "describe" in ast.unparse(node)
               and "gpu" not in ast.unparse(node) for node in cli_choices)
    assert "post_generation_full_sequence_no_cache_replay" in runner_text
    contract = function_node(runner_tree, "full_no_cache_replay_contract")
    assert "'use_cache': False" in ast.unparse(contract)

    plan = read(PLAN_PATH)
    manifest = read(OUT / "MANIFEST.json")
    preparation = read(OUT / "PREPARATION.json")
    selfcheck = read(OUT / "CPU_SELFCHECK.json")
    assert preparation["protocol_sha256"] == sha(PROTOCOL_PATH)
    assert preparation["plan_sha256"] == sha(PLAN_PATH)
    assert manifest["files_sha256"]["label_free_inputs.jsonl"] == sha(
        OUT / "label_free_inputs.jsonl")
    for name, expected in manifest["files_sha256"].items():
        assert sha(OUT / name) == expected, name
    assert selfcheck["status"] == "passed_CPU_only"
    assert selfcheck["whitebox_extraction_mode"] == "post_generation_full_sequence_no_cache_replay"
    assert not any((selfcheck[key] for key in
                    ("GPU_used", "model_weights_loaded", "calibration_read",
                     "official_test_read",
                     "gold_bearing_source_JSON_opened_by_selfcheck",
                     "sanitized_manifest_contains_gold_or_label_fields",
                     "scoring_run")))
    source_scope = preparation["scope"]
    assert source_scope["source_JSON_fully_parsed_and_contains_gold_fields"] is True
    assert source_scope["gold_fields_referenced_copied_or_used_in_selection_or_features"] is False

    rows = 0; answers = set(); groups = set()
    with (OUT / "label_free_inputs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); rows += 1
            assert tuple(row) == OUTPUT_FIELDS
            recursively_reject(row)
            assert row["model"] == "llama-2-7b-chat"
            answers.add(row["response_id"]); groups.add(row["group_id"])
    assert (rows, len(answers), len(groups)) == (3776, 256, 256)
    assert manifest["counts"] == {"groups": 256, "answers": 256, "claims": 3776}
    assert plan["scope"]["selected_atomic_claims"] == rows

    result = {
        "status": "passed_CPU_implementation_audit",
        "counts": {"groups": len(groups), "answers": len(answers), "claims": rows},
        "AST_allowlist_proof": {
            "fit_source_full_JSON_parse_disclosed": True,
            "only_raw_subscript_in_sanitizer": raw_subscripts,
            "raw_is_immediately_projected_through_fixed_allowlist_then_deleted": True,
            "output_schema_exact": list(OUTPUT_FIELDS),
            "forbidden_payload_absent_recursively": True,
        },
        "runner": {
            "executable_CLI_stages": ["cpu-selfcheck", "describe"],
            "GPU_methods_raise": True,
            "whitebox_contract": "generation token IDs followed by full-sequence no-cache replay",
            "use_cache_in_whitebox_replay": False,
        },
        "files_sha256": {
            "prepare_code": sha(PREP_CODE), "runner_code": sha(RUNNER_CODE),
            "protocol": sha(PROTOCOL_PATH), "plan": sha(PLAN_PATH),
            "label_free_inputs": sha(OUT / "label_free_inputs.jsonl"),
            "preparation": sha(OUT / "PREPARATION.json"),
            "resource_estimate": sha(OUT / "RESOURCE_ESTIMATE.json"),
            "CPU_SELFCHECK": sha(OUT / "CPU_SELFCHECK.json"),
        },
        "files_opened": [str(path.resolve()) for path in (
            PREP_CODE, RUNNER_CODE, PLAN_PATH, PROTOCOL_PATH,
            OUT / "MANIFEST.json", OUT / "PREPARATION.json",
            OUT / "CPU_SELFCHECK.json", OUT / "label_free_inputs.jsonl")],
        "fit_payload_opened": False, "gold_payload_opened": False,
        "calibration_or_test_opened": False, "model_loaded": False,
        "GPU_used": False, "scoring_run": False, "baseline_modified": False,
    }
    target = OUT / "CPU_IMPLEMENTATION_AUDIT.json"
    assert not target.exists(), target
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("FORCED_QUOTE_CPU_IMPLEMENTATION_AUDIT_PASSED", rows, flush=True)


if __name__ == "__main__":
    main()
