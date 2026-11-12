"""Static and CPU-artifact audit for the SDPA-only v2 GPU runner.

The audit never imports either runner and never initializes a model or CUDA.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results/auxiliary_conflict_token_pilot_v1"
V2_OUT = DATA / "gpu_runs_v2"
V1 = HERE / "run_auxiliary_conflict_token_pilot_v1.py"
V2 = HERE / "run_auxiliary_conflict_token_pilot_v2.py"
FAILURE = DATA / "FAILURE_HISTORY.md"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def assignments(tree):
    output = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            output[node.targets[0].id] = node.value
    return output


def canonical(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class NormalizeInitialize(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            node.keywords = [kw for kw in node.keywords
                             if kw.arg not in ("attn_implementation", "reference_compile")]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = [statement for statement in node.body
                     if not (isinstance(statement, ast.Assert)
                             and "model.config" in ast.unparse(statement.test))]
        return node


class NormalizeOutputRoot(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "OUT":
            node.id = "DATA"
        return node


def normalize_cpu_check(node):
    node = NormalizeOutputRoot().visit(copy.deepcopy(node))
    cleaned = []
    for statement in node.body:
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "loader_config"
                        for target in statement.targets)):
            continue
        if isinstance(statement, ast.Assert) and "loader_config" in ast.unparse(statement.test):
            continue
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "result"
                        for target in statement.targets)
                and isinstance(statement.value, ast.Dict)):
            kept = [(key, value) for key, value in zip(statement.value.keys, statement.value.values)
                    if not (isinstance(key, ast.Constant)
                            and key.value in ("loader_config_attn_implementation",
                                              "loader_config_reference_compile"))]
            statement.value.keys = [key for key, _ in kept]
            statement.value.values = [value for _, value in kept]
        cleaned.append(statement)
    node.body = cleaned
    ast.fix_missing_locations(node)
    return node


def assigned_value(function, name):
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError((function.name, name))


def main():
    v1_source = V1.read_text(encoding="utf-8")
    v2_source = V2.read_text(encoding="utf-8")
    v1_tree, v2_tree = ast.parse(v1_source), ast.parse(v2_source)
    f1, f2 = functions(v1_tree), functions(v2_tree)
    a1, a2 = assignments(v1_tree), assignments(v2_tree)

    unchanged_constants = (
        "HERE", "ROOT", "DATA", "MODEL", "V4_CHECKPOINT", "SEED", "PAD_ID",
        "TOKEN_BUDGET", "MAX_EXAMPLES", "ACCUM", "EVAL_TOKEN_BUDGET",
        "EVAL_MAX_EXAMPLES", "LR", "WEIGHT_DECAY", "CLIP_NORM",
        "MIN_FREE_GPU_BYTES", "STAGE_CODES",
    )
    for name in unchanged_constants:
        assert canonical(a1[name]) == canonical(a2[name]), name
    assert ast.literal_eval(a2["OUT"].right) == "gpu_runs_v2"
    assert ast.literal_eval(a1["OUT"].right) == "gpu_runs"

    unchanged_functions = (
        "sha", "read_json", "save_json", "load_data", "make_batches",
        "batch_tensors", "token_logits", "conflict_logit", "exact_span_loss",
        "batch_plan", "plan_stats", "grouped_train", "held_inference",
        "gpu_preflight", "train_arm", "choose_threshold", "count", "top_k",
    )
    for name in unchanged_functions:
        assert canonical(f1[name]) == canonical(f2[name]), name

    normalized_initialize = NormalizeInitialize().visit(copy.deepcopy(f2["initialize_model"]))
    ast.fix_missing_locations(normalized_initialize)
    assert canonical(f1["initialize_model"]) == canonical(normalized_initialize)
    load_calls = [node for node in ast.walk(f2["initialize_model"])
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "from_pretrained"]
    assert len(load_calls) == 1
    keywords = {kw.arg: kw.value for kw in load_calls[0].keywords}
    assert ast.literal_eval(keywords["attn_implementation"]) == "sdpa"
    assert ast.literal_eval(keywords["reference_compile"]) is False
    initialize_text = ast.unparse(f2["initialize_model"])
    assert "model.config._attn_implementation == 'sdpa'" in initialize_text
    assert "model.config.reference_compile is False" in initialize_text

    output_only_changes = ("estimate", "gpu_smoke")
    for name in output_only_changes:
        normalized = NormalizeOutputRoot().visit(copy.deepcopy(f2[name]))
        ast.fix_missing_locations(normalized)
        assert canonical(f1[name]) == canonical(normalized), name
    assert canonical(f1["cpu_check"]) == canonical(normalize_cpu_check(f2["cpu_check"]))
    cpu_check_text = ast.unparse(f2["cpu_check"])
    assert "AutoConfig.from_pretrained" in cpu_check_text
    assert "attn_implementation='sdpa'" in cpu_check_text
    assert "reference_compile=False" in cpu_check_text

    # Existing window metrics and the pilot gate retain the same expressions.
    assert canonical(assigned_value(f1["finalize"], "gate")) == canonical(
        assigned_value(f2["finalize"], "gate"))
    for exact in (
        'combined = np.maximum(v4, conflict)',
        'matched = top_k(combined, budget)',
        'high = arrays["held_window_max_evidence_coverage"] >= 0.5',
        'base_conflict_recall = float(np.count_nonzero(v4_pred & (conflict_y == 1)) / conflict_y.sum())',
    ):
        assert exact in v2_source

    helper = ast.unparse(f2["answer_max_from_windows"])
    assert "np.maximum.at(result, answer_index, scores)" in helper
    assert "np.isfinite(result).all()" in helper
    finalize_text = ast.unparse(f2["finalize"])
    for exact in (
        "v4_answer = answer_max_from_windows(v4, answer_index, len(answer_y))",
        "conflict_answer = answer_max_from_windows(conflict, answer_index, len(answer_y))",
        "combined_answer = answer_max_from_windows(combined, answer_index, len(answer_y))",
        "'answer_max_window': v4_answer_metrics",
        "'conflict_only_answer_max_window': conflict_answer_metrics",
        "'combined_answer_max_window': combined_answer_metrics",
    ):
        assert exact in finalize_text, exact

    assert 'save_json(DATA / "GPU_' not in v2_source
    assert 'save_json(DATA / "CPU_' not in v2_source
    assert 'ROOT / "calibration' not in v2_source and 'ROOT / "test' not in v2_source

    core_manifest = load_json(DATA / "manifest.json")
    for name, expected in core_manifest["files_sha256"].items():
        assert sha(DATA / name) == expected, name
    cpu_seal = load_json(DATA / "CPU_REVIEW_COMPLETE.json")
    v1_sealed = [value for name, value in cpu_seal["files_sha256"].items()
                 if name.endswith("src\\run_auxiliary_conflict_token_pilot_v1.py")]
    assert v1_sealed == [sha(V1)]

    cpu = load_json(V2_OUT / "CPU_CHECK.json")
    plan = load_json(V2_OUT / "GPU_PLAN.json")
    v1_cpu = load_json(DATA / "CPU_CHECK.json")
    v1_plan = load_json(DATA / "GPU_PLAN.json")
    assert plan == v1_plan
    for key, value in v1_cpu.items():
        assert cpu[key] == value, key
    assert cpu["loader_config_attn_implementation"] == "sdpa"
    assert cpu["loader_config_reference_compile"] is False
    assert cpu["status"] == "passed" and cpu["model_checkpoint_loaded"] is False
    assert cpu["GPU_used"] is False and plan["GPU_used"] is False
    assert plan["gpu_smoke_not_run"] is True

    assert FAILURE.exists()
    assert not (DATA / "GPU_SMOKE.json").exists()
    assert not (DATA / "gpu_runs").exists()
    assert not (V2_OUT / "GPU_SMOKE.json").exists()
    assert not (V2_OUT / "candidate").exists() and not (V2_OUT / "control").exists()
    assert not (V2_OUT / "summary.json").exists()

    result = {
        "status": "passed",
        "scope": "Static runner diff plus CPU artifacts only; no model load, GPU, calibration, or test.",
        "hashes": {
            "v1_runner_sha256": sha(V1),
            "v2_runner_sha256": sha(V2),
            "failure_history_sha256": sha(FAILURE),
            "v2_cpu_check_sha256": sha(V2_OUT / "CPU_CHECK.json"),
            "v2_gpu_plan_sha256": sha(V2_OUT / "GPU_PLAN.json"),
            "core_manifest_sha256": sha(DATA / "manifest.json"),
            "protocol_sha256": sha(DATA / "PROTOCOL.md"),
            "audit_script_sha256": sha(Path(__file__)),
        },
        "intentional_differences": {
            "backend": "from_pretrained(attn_implementation='sdpa', reference_compile=False) plus config assertions",
            "output_directory": "gpu_runs_v2",
            "reporting_only": "answer score=max window; F1/AP added for v4, candidate, and control",
        },
        "invariants": {
            "unchanged_constant_assignments": list(unchanged_constants),
            "unchanged_function_asts": list(unchanged_functions),
            "initialization_equal_after_removing_backend_keywords_and_assertions": True,
            "estimate_and_smoke_equal_after_output_root_normalization": True,
            "cpu_check_equal_after_backend_config_diagnostic_and_output_normalization": True,
            "window_readout_and_gate_unchanged": True,
            "v2_gpu_plan_bitwise_equal_to_v1": True,
            "v2_cpu_gradient_check_equal_to_v1": True,
            "core_data_manifest_valid": True,
            "v1_runner_matches_pre_failure_cpu_seal": True,
            "published_baselines_modified": False,
        },
        "failure_preservation": {
            "v1_success_smoke_artifact_exists": False,
            "v1_gpu_run_directory_exists": False,
            "history_written": True,
        },
        "execution": {
            "v2_cpu_check": "passed",
            "v2_loader_config_attn_implementation": cpu["loader_config_attn_implementation"],
            "v2_loader_config_reference_compile": cpu["loader_config_reference_compile"],
            "v2_gpu_plan_reproduced": True,
            "v2_gpu_smoke_run": False,
            "model_checkpoint_loaded": False,
            "GPU_used": False,
            "calibration_rows_read": 0,
            "test_rows_read": 0,
        },
    }
    target = V2_OUT / "V2_DIFF_AUDIT.json"
    pending = target.with_suffix(target.suffix + ".pending")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
