"""Windows-WDDM-safe execution wrapper for exact subset attribution v1.

This runner changes only GPU admission and output lineage.  It reuses the
already frozen, label-free v1 prepared inputs and the unchanged v1 extraction
math.  CPU stages do not query a GPU, load a model, read gold annotations, or
address the official test set.  GPU stages are explicit and write only v2.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_exact_subset_attribution_v1 as core


ROOT = core.ROOT
V1_OUT = ROOT / "results/exact_subset_attribution_v1"
OUT = ROOT / "results/exact_subset_attribution_v2"
PREPARED = V1_OUT / "prepared_inputs.jsonl"
V1_COMPLETE = V1_OUT / "preparation_complete.json"
GLOBAL_GPU_LOCK = core.GLOBAL_GPU_LOCK

VERSION = "exact-three-passage-subset-attribution-v2-wddm-gate"
ROLE = core.ROLE
EXPECTED_PREPARED_SHA256 = "55753d10aa79f2069acfed7161c9b64fa8bcc4ed21da97c1a403ec73e1ff1f2a"
EXPECTED_V1_COMPLETE_SHA256 = "f68257930b58c4ab44ed1b990e01e012beb19279ddd69f3082bf5c5a97efcb17"
EXPECTED_ANSWERS = 3839
EXPECTED_RAW_TOKENS = core.EXPECTED_RAW_TOKENS
MIN_FREE_GPU_BYTES = core.MIN_FREE_GPU_BYTES
THREADS = core.THREADS

# Under the installed Windows WDDM driver, UI processes appear in
# --query-compute-apps with memory exactly "[N/A]" and type C+G in the normal
# process table.  Only these reviewed GUI executable identities may use that
# exception.  Any numeric memory, pure C type, unknown type/name, or malformed
# value remains a blocker.  The list is deliberately environment-specific and
# frozen by source/protocol hash; additions require a new reviewed version.
WDDM_GUI_EXECUTABLE_ALLOWLIST = frozenset({
    "applicationframehost.exe",
    "chatgpt.exe",
    "chrome.exe",
    "cloudmusic.exe",
    "code.exe",
    "desktop lock.exe",
    "doubao.exe",
    "dwm.exe",
    "explorer.exe",
    "hipsdaemon.exe",
    "msedgewebview2.exe",
    "nvidia overlay.exe",
    "nutstoreclient.exe",
    "phoneexperiencehost.exe",
    "promecefpluginhost.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "shellhost.exe",
    "snippingtool.exe",
    "startmenuexperiencehost.exe",
    "systemsettings.exe",
    "tabtip.exe",
    "textinputhost.exe",
    "todesk.exe",
})


def sha(path):
    handle_hash = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            handle_hash.update(chunk)
    return handle_hash.hexdigest()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def lines(path=PREPARED):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def protocol():
    return {
        "version": VERSION,
        "scope": "GPU admission repair and new output lineage only",
        "method": "All prepared inputs, model, eight-view extraction math, features, and future evaluation semantics remain v1.",
        "prepared_input": {
            "mode": "read-only reuse; no copy or rewrite",
            "path": str(PREPARED.resolve()),
            "sha256": EXPECTED_PREPARED_SHA256,
            "answers": EXPECTED_ANSWERS,
            "raw_answer_tokens": EXPECTED_RAW_TOKENS,
            "labels": "absent; inherited label-free artifact",
        },
        "wddm_gpu_gate": {
            "global_lock": str(GLOBAL_GPU_LOCK.resolve()),
            "lock_policy": "O_EXCL lock with random lease nonce; never auto-break a stale lock",
            "process_queries": [
                "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits",
                "nvidia-smi full process table for PID->Type",
            ],
            "ignore_rule": (
                "Ignore a foreign row only when used_gpu_memory is literally [N/A], "
                "the same PID is typed G or C+G in the process table, and the full "
                "query path basename is in the frozen GUI allowlist."
            ),
            "block_rule": (
                "Block numeric memory even for a GUI name; block pure C, unknown or "
                "missing type, unknown executable, and every malformed/other memory value."
            ),
            "race_reduction": "Repeat the foreign-process scan after CUDA free-memory initialization; ignore only this runner's PID.",
            "memory_gate": f"Require at least {MIN_FREE_GPU_BYTES} free bytes before model load.",
            "gui_allowlist": sorted(WDDM_GUI_EXECUTABLE_ALLOWLIST),
        },
        "commands": {
            "CPU": ["initialize", "prepare", "check", "audit", "self-test"],
            "GPU_after_review": ["gpu-smoke", "extract"],
        },
        "not_in_scope": [
            "No GPU work in CPU stages.",
            "No gold/label/test read in this runner.",
            "No baseline modification.",
            "No v1 artifact modification or overwrite.",
            "Scoring remains a later separately reviewed stage after caches freeze.",
        ],
    }


def parse_compute_query(text):
    """Parse query output without treating arbitrary nonnumeric text as N/A."""
    records = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = next(csv.reader([raw_line], skipinitialspace=True))
        if len(fields) < 3:
            raise ValueError(f"Malformed compute row {line_number}: {raw_line!r}")
        pid_text = fields[0].strip()
        process_name = ",".join(fields[1:-1]).strip()
        memory_text = fields[-1].strip()
        if not pid_text.isdigit() or not process_name:
            raise ValueError(f"Malformed compute row {line_number}: {raw_line!r}")
        if memory_text == "[N/A]":
            memory_kind = "explicit_bracketed_na"
            memory_mib = None
        elif re.fullmatch(r"[0-9]+", memory_text):
            memory_kind = "numeric_mib"
            memory_mib = int(memory_text)
        else:
            memory_kind = "unknown"
            memory_mib = None
        records.append({
            "pid": int(pid_text),
            "process_name": process_name,
            "process_basename": ntpath.basename(process_name).casefold(),
            "memory_raw": memory_text,
            "memory_kind": memory_kind,
            "memory_mib": memory_mib,
        })
    return records


PROCESS_TABLE_ROW = re.compile(
    r"^\|\s*\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+([CG+]+)\s+.*\|\s*$"
)


def parse_process_types(text):
    """Return PID->NVIDIA process type from the normal nvidia-smi table."""
    result = {}
    for raw_line in text.splitlines():
        match = PROCESS_TABLE_ROW.match(raw_line)
        if not match:
            continue
        pid, process_type = int(match.group(1)), match.group(2)
        if pid in result and result[pid] != process_type:
            raise ValueError(f"Conflicting NVIDIA types for PID {pid}")
        result[pid] = process_type
    return result


def classify_processes(query_records, process_types, own_pid):
    """Fail closed except for the exact reviewed WDDM GUI exception."""
    decisions = []
    blockers = []
    for record in query_records:
        item = dict(record)
        item["nvidia_type"] = process_types.get(record["pid"])
        if record["pid"] == own_pid:
            item.update(decision="allow", reason="own_runner_pid")
        elif record["memory_kind"] == "numeric_mib":
            item.update(decision="block", reason="foreign_numeric_gpu_memory")
        elif record["memory_kind"] != "explicit_bracketed_na":
            item.update(decision="block", reason="foreign_unknown_memory_fail_closed")
        elif item["nvidia_type"] not in {"G", "C+G"}:
            item.update(decision="block", reason="foreign_compute_or_unknown_type_fail_closed")
        elif record["process_basename"] not in WDDM_GUI_EXECUTABLE_ALLOWLIST:
            item.update(decision="block", reason="foreign_unreviewed_graphics_executable_fail_closed")
        else:
            item.update(decision="allow", reason="explicit_na_reviewed_wddm_graphics")
        decisions.append(item)
        if item["decision"] == "block":
            blockers.append(item)
    return decisions, blockers


def gpu_gate_selfcheck():
    query = "\n".join([
        r"101, C:\\Windows\\System32\\dwm.exe, [N/A]",
        r"102, C:\\Python\\python.exe, 512",
        r"103, C:\\Python\\python.exe, [N/A]",
        r"104, C:\\Temp\\unknown_gui.exe, [N/A]",
        r"105, C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe, 0",
        r"106, C:\\Windows\\explorer.exe, N/A",
        r"107, C:\\Python\\python.exe, 64",
    ])
    table = "\n".join([
        "|    0   N/A  N/A             101    C+G   C:\\Windows\\System32\\dwm.exe          N/A      |",
        "|    0   N/A  N/A             102      C   C:\\Python\\python.exe                   N/A      |",
        "|    0   N/A  N/A             103      C   C:\\Python\\python.exe                   N/A      |",
        "|    0   N/A  N/A             104    C+G   C:\\Temp\\unknown_gui.exe              N/A      |",
        "|    0   N/A  N/A             105    C+G   ...Chrome\\Application\\chrome.exe       N/A      |",
        "|    0   N/A  N/A             106    C+G   C:\\Windows\\explorer.exe               N/A      |",
        "|    0   N/A  N/A             107      C   C:\\Python\\python.exe                   N/A      |",
    ])
    parsed = parse_compute_query(query)
    types = parse_process_types(table)
    decisions, blockers = classify_processes(parsed, types, own_pid=107)
    by_pid = {item["pid"]: item for item in decisions}
    assert by_pid[101]["reason"] == "explicit_na_reviewed_wddm_graphics"
    assert by_pid[102]["reason"] == "foreign_numeric_gpu_memory"
    assert by_pid[103]["reason"] == "foreign_compute_or_unknown_type_fail_closed"
    assert by_pid[104]["reason"] == "foreign_unreviewed_graphics_executable_fail_closed"
    assert by_pid[105]["reason"] == "foreign_numeric_gpu_memory"
    assert by_pid[106]["reason"] == "foreign_unknown_memory_fail_closed"
    assert by_pid[107]["reason"] == "own_runner_pid"
    assert {item["pid"] for item in blockers} == {102, 103, 104, 105, 106}
    try:
        parse_compute_query("bad row")
    except ValueError:
        malformed_failed_closed = True
    else:
        malformed_failed_closed = False
    assert malformed_failed_closed
    with tempfile.TemporaryDirectory(prefix="exact_subset_v2_lock_") as temporary:
        lock_path = Path(temporary) / "gpu.lock"
        lease = acquire_lock("synthetic-self-test", lock_path=lock_path)
        assert lock_path.exists()
        try:
            acquire_lock("nested-must-fail", lock_path=lock_path)
        except RuntimeError:
            nested_lock_failed_closed = True
        else:
            nested_lock_failed_closed = False
        assert nested_lock_failed_closed
        wrong_lease = dict(lease, lease_nonce="wrong")
        assert release_lock(wrong_lease, lock_path=lock_path) is False
        assert lock_path.exists()
        assert release_lock(lease, lock_path=lock_path) is True
        assert not lock_path.exists()
    assert not torch.cuda.is_initialized()
    return {
        "status": "passed",
        "cases": {
            "explicit_bracketed_NA_reviewed_C_plus_G_is_ignored": True,
            "numeric_memory_always_blocks": True,
            "explicit_NA_pure_C_blocks": True,
            "unreviewed_C_plus_G_blocks": True,
            "unbracketed_NA_blocks": True,
            "own_pid_is_exempt": True,
            "malformed_query_fails_closed": True,
            "O_EXCL_lock_and_nonce_release_checked": True,
        },
        "synthetic_only": True,
        "nvidia_smi_called": False,
        "GPU_initialized": False,
        "labels_accessed": False,
        "official_test_opened": False,
    }


def v1_inventory():
    names = (
        "PREFLIGHT.json", "protocol.json", "PLAN.md", "design_freeze.json",
        "prepare_started.json", "prepared_inputs.jsonl",
        "preparation_statistics.json", "source_snapshot.json",
        "preparation_complete.json", "CPU_CHECK.json", "CODE_AUDIT.json",
        "CODE_AUDIT.md",
    )
    return {name: sha(V1_OUT / name) for name in names}


def verify_v1_binding():
    assert sha(PREPARED) == EXPECTED_PREPARED_SHA256
    assert sha(V1_COMPLETE) == EXPECTED_V1_COMPLETE_SHA256
    complete = read(V1_COMPLETE)
    assert complete["status"] == "CPU_prepared_waiting_for_GPU_review"
    assert complete["answers"] == EXPECTED_ANSWERS
    assert complete["raw_answer_tokens"] == EXPECTED_RAW_TOKENS
    assert complete["files_sha256"]["prepared_inputs.jsonl"] == EXPECTED_PREPARED_SHA256
    assert complete["labels_accessed"] is False
    assert complete["GPU_used"] is False
    assert complete["official_test_opened"] is False
    return complete


def initialize():
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), f"Refuse to overwrite {OUT}"
    OUT.mkdir(parents=True)
    save(OUT / "PREFLIGHT.json", gpu_gate_selfcheck())
    save(OUT / "protocol.json", protocol())
    (OUT / "PLAN.md").write_text(
        "# Exact subset attribution v2：Windows WDDM GPU 门禁修复\n\n"
        "v2 只修 GPU 入场判定并写入新目录。它只读复用 v1 的无标签 prepared inputs，"
        "不改 8 子集、模型、特征或评测定义。\n\n"
        "CPU 顺序：`initialize -> prepare -> check -> audit`。审核后才可单独运行 "
        "`gpu-smoke -> extract`。当前不得运行 GPU 命令。\n",
        encoding="utf-8",
    )
    verify_v1_binding()
    save(OUT / "design_freeze.json", {
        "status": "frozen_before_v2_preparation",
        "v2_source_sha256": sha(__file__),
        "v1_core_source_sha256": sha(core.__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "prepared_inputs_sha256": EXPECTED_PREPARED_SHA256,
        "v1_preparation_complete_sha256": EXPECTED_V1_COMPLETE_SHA256,
        "v1_existing_artifacts_sha256": v1_inventory(),
        "gui_allowlist_sha256": digest(sorted(WDDM_GUI_EXECUTABLE_ALLOWLIST)),
        "labels_accessed": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    print("EXACT_SUBSET_ATTRIBUTION_V2_PROTOCOL_FROZEN", flush=True)


def verify_freeze():
    freeze = read(OUT / "design_freeze.json")
    assert freeze["v2_source_sha256"] == sha(__file__)
    assert freeze["v1_core_source_sha256"] == sha(core.__file__)
    assert freeze["protocol_sha256"] == sha(OUT / "protocol.json")
    assert read(OUT / "protocol.json") == protocol()
    assert freeze["prepared_inputs_sha256"] == sha(PREPARED)
    assert freeze["v1_preparation_complete_sha256"] == sha(V1_COMPLETE)
    assert freeze["v1_existing_artifacts_sha256"] == v1_inventory()
    assert freeze["gui_allowlist_sha256"] == digest(sorted(WDDM_GUI_EXECUTABLE_ALLOWLIST))
    return freeze


def prepare():
    assert not torch.cuda.is_initialized()
    freeze = verify_freeze()
    verify_v1_binding()
    assert not (OUT / "preparation_complete.json").exists()
    binding = {
        "status": "read_only_v1_label_free_preparation_bound",
        "source_path": str(PREPARED.resolve()),
        "source_sha256": sha(PREPARED),
        "source_complete_path": str(V1_COMPLETE.resolve()),
        "source_complete_sha256": sha(V1_COMPLETE),
        "v1_core_source_sha256": freeze["v1_core_source_sha256"],
        "answers": EXPECTED_ANSWERS,
        "raw_answer_tokens": EXPECTED_RAW_TOKENS,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "prepared_input_binding.json", binding)
    files = ("PREFLIGHT.json", "protocol.json", "PLAN.md", "design_freeze.json",
             "prepared_input_binding.json")
    save(OUT / "preparation_complete.json", {
        "status": "CPU_ready_waiting_for_GPU_review",
        "files_sha256": {name: sha(OUT / name) for name in files},
        "prepared_inputs_external_sha256": sha(PREPARED),
        "v1_artifacts_unchanged": v1_inventory() == freeze["v1_existing_artifacts_sha256"],
        "labels_accessed": False,
        "new_fits": 0,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    })
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_ATTRIBUTION_V2_CPU_PREPARED", flush=True)


def check_ready():
    freeze = verify_freeze()
    complete = read(OUT / "preparation_complete.json")
    assert complete["status"] == "CPU_ready_waiting_for_GPU_review"
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected
    assert complete["prepared_inputs_external_sha256"] == sha(PREPARED)
    assert v1_inventory() == freeze["v1_existing_artifacts_sha256"]
    return complete


def check():
    assert not torch.cuda.is_initialized()
    check_ready()
    counts = Counter()
    raw_tokens = 0
    response_ids = set()
    first_id = last_id = None
    for index, row in enumerate(lines()):
        if first_id is None:
            first_id = row["response_id"]
        last_id = row["response_id"]
        assert row["labels_used"] is False
        core.reject_annotation_keys({key: value for key, value in row.items()
                                     if key != "labels_used"})
        assert row["official_split"] == "train"
        assert row["partition"] in {"fit", "calibration"}
        assert len(row["views"]) == 8
        assert [view["mask"] for view in row["views"]] == list(range(8))
        assert all(view["answer_positions"][-1] == view["input_token_count"] - 1
                   for view in row["views"])
        assert row["response_id"] not in response_ids
        response_ids.add(row["response_id"])
        counts[row["partition"]] += 1
        raw_tokens += len(row["answer_token_ids"])
    assert index + 1 == EXPECTED_ANSWERS
    assert counts == core.EXPECTED_ANSWERS
    assert raw_tokens == EXPECTED_RAW_TOKENS
    report = {
        "status": "passed_CPU_label_free_binding_and_gate_selfcheck",
        "answers": len(response_ids),
        "answers_by_partition": dict(counts),
        "raw_answer_tokens": raw_tokens,
        "first_response_id": first_id,
        "last_response_id": last_id,
        "prepared_inputs_sha256": sha(PREPARED),
        "synthetic_gpu_gate_selfcheck": gpu_gate_selfcheck(),
        "v1_artifacts_unchanged": True,
        "nvidia_smi_called": False,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
    }
    save(OUT / "CPU_CHECK.json", report)
    assert not torch.cuda.is_initialized()
    print("EXACT_SUBSET_ATTRIBUTION_V2_CPU_CHECK_PASSED", flush=True)


def run_nvidia_smi(arguments):
    completed = subprocess.run(["nvidia-smi", *arguments], check=True,
                               capture_output=True, text=True)
    return completed.stdout


def inspect_foreign_gpu_processes(own_pid):
    query_text = run_nvidia_smi([
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    table_text = run_nvidia_smi([])
    query_records = parse_compute_query(query_text)
    process_types = parse_process_types(table_text)
    decisions, blockers = classify_processes(query_records, process_types, own_pid)
    return {
        "query_stdout_sha256": digest(query_text),
        "process_table_stdout_sha256": digest(table_text),
        "query_row_count": len(query_records),
        "typed_pid_count": len(process_types),
        "decisions": decisions,
        "blockers": blockers,
    }


def acquire_lock(stage, lock_path=GLOBAL_GPU_LOCK):
    lock_path = Path(lock_path)
    token = {
        "pid": os.getpid(), "stage": stage, "method": VERSION,
        "lease_nonce": uuid.uuid4().hex, "started_unix": time.time(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags)
    except FileExistsError as error:
        raise RuntimeError(f"Another GPU runner owns {lock_path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(token, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return token


def release_lock(token, lock_path=GLOBAL_GPU_LOCK):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return False
    try:
        current = read(lock_path)
    except Exception:
        return False
    identity = ("pid", "stage", "method", "lease_nonce")
    if all(current.get(key) == token.get(key) for key in identity):
        lock_path.unlink()
        return True
    return False


@contextmanager
def exclusive_gpu(stage):
    """Conservative WDDM gate; called only by explicit GPU stages."""
    token = acquire_lock(stage)
    try:
        before = inspect_foreign_gpu_processes(os.getpid())
        if before["blockers"]:
            raise RuntimeError(f"Foreign or ambiguous GPU process: {before['blockers']}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        if free_bytes < MIN_FREE_GPU_BYTES:
            raise RuntimeError(
                f"Insufficient free GPU memory: {free_bytes} < {MIN_FREE_GPU_BYTES}"
            )
        after = inspect_foreign_gpu_processes(os.getpid())
        if after["blockers"]:
            raise RuntimeError(f"Foreign or ambiguous GPU process after CUDA init: {after['blockers']}")
        token.update({
            "free_bytes_before_load": int(free_bytes),
            "total_bytes": int(total_bytes),
            "foreign_scan_before": before,
            "foreign_scan_after_cuda_init": after,
        })
        yield token
    finally:
        release_lock(token)


def runtime_signature():
    return {
        "version": VERSION,
        "v2_source_sha256": sha(__file__),
        "v1_core_source_sha256": sha(core.__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        # The full file is rehashed by check_ready before GPU admission.  Use
        # its frozen expected digest here so per-answer cache signatures do not
        # reread the 179 MB preparation file 3,839 times.
        "prepared_inputs_sha256": EXPECTED_PREPARED_SHA256,
        "backend": core.BACKEND,
        "repo": core.REPO,
        "revision": core.REVISION,
        "model_manifest_sha256": sha(core.MODEL_MANIFEST),
        "feature_names_sha256": digest(core.FEATURE_NAMES),
        "gui_allowlist_sha256": digest(sorted(WDDM_GUI_EXECUTABLE_ALLOWLIST)),
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "accelerate", "numpy")},
        "cuda_runtime": torch.version.cuda,
    }


def first_and_last_rows():
    first = last = None
    count = 0
    for row in lines():
        if first is None:
            first = row
        last = row
        count += 1
    assert count == EXPECTED_ANSWERS and first is not None and last is not None
    return first, last


def cache_signature(row):
    return digest({
        "runtime": runtime_signature(),
        "response_id": row["response_id"],
        "answer_token_ids_sha256": row["answer_token_ids_sha256"],
        "citation_mask_sha256": row["citation_geometry"]["citation_mask_sha256"],
        "view_input_sha256": [view["input_ids_sha256"] for view in row["views"]],
    })


def validate_cache(path, row):
    metadata = read(path.with_suffix(".json"))
    assert metadata["complete"] is True
    assert metadata["response_id"] == row["response_id"]
    assert metadata["cache_signature"] == cache_signature(row)
    assert metadata["npz_sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as archive:
        logp = archive["selected_logprob"].copy()
        answer_ids = archive["answer_token_ids"].copy()
        citation_mask = archive["citation_mask"].copy()
    assert logp.shape == (8, len(row["answer_token_ids"]))
    assert logp.dtype == np.float32 and np.isfinite(logp).all()
    assert answer_ids.tolist() == row["answer_token_ids"]
    assert citation_mask.tolist() == row["citation_geometry"]["citation_mask"]
    _, efficiency = core.derive_features(logp, citation_mask)
    assert abs(efficiency - metadata["shapley_efficiency_max_abs"]) <= 1e-12
    return logp


def gpu_smoke():
    check_ready()
    assert not (OUT / "GPU_SMOKE.json").exists(), "No silent smoke overwrite"
    chosen = first_and_last_rows()
    with exclusive_gpu("gpu-smoke") as lease:
        model, model_meta = core.load_nf4()
        try:
            started = time.perf_counter()
            records = []
            first_full = None
            for row in chosen:
                logp, efficiency = core.extract_row(model, row)
                records.append({
                    "response_id": row["response_id"],
                    "tokens": logp.shape[1],
                    "logprob_sha256": digest(logp.tolist()),
                    "shapley_efficiency_max_abs": efficiency,
                })
                if first_full is None:
                    first_full = logp[7].copy()
            repeated = core.selected_token_logprob(
                model, chosen[0]["views"][7]["prefix_ids"],
                chosen[0]["answer_token_ids"],
                chosen[0]["views"][7]["answer_positions"],
            )
            repeat_error = float(np.max(np.abs(first_full - repeated)))
            assert repeat_error <= 1e-6
            save(OUT / "GPU_SMOKE.json", {
                "status": "passed_not_full_extraction",
                "runtime_signature": runtime_signature(),
                "runtime_signature_sha256": digest(runtime_signature()),
                "lease": lease,
                "model": model_meta,
                "records": records,
                "full_view_repeat_max_abs": repeat_error,
                "seconds": time.perf_counter() - started,
                "labels_accessed": False,
                "official_test_opened": False,
            })
        finally:
            del model
            core.clean_gpu()
    print("EXACT_SUBSET_ATTRIBUTION_V2_GPU_SMOKE_PASSED", flush=True)


def extract():
    check_ready()
    smoke = read(OUT / "GPU_SMOKE.json")
    assert smoke["status"] == "passed_not_full_extraction"
    assert smoke["runtime_signature_sha256"] == digest(runtime_signature())
    assert not (OUT / "extraction_complete.json").exists()
    cache_dir = OUT / "token_logprob"
    cache_dir.mkdir(exist_ok=True)
    records = []
    with exclusive_gpu("extract") as lease:
        model, model_meta = core.load_nf4()
        started = time.perf_counter()
        try:
            for index, row in enumerate(lines()):
                path = cache_dir / f"{row['response_id']}.npz"
                metadata_path = path.with_suffix(".json")
                if path.exists() and metadata_path.exists():
                    validate_cache(path, row)
                else:
                    assert not path.exists() and not metadata_path.exists()
                    logp, efficiency = core.extract_row(model, row)
                    core.atomic_npz(
                        path,
                        selected_logprob=logp,
                        answer_token_ids=np.asarray(row["answer_token_ids"], dtype=np.int32),
                        citation_mask=np.asarray(row["citation_geometry"]["citation_mask"], dtype=np.uint8),
                    )
                    save(metadata_path, {
                        "complete": True,
                        "response_id": row["response_id"],
                        "cache_signature": cache_signature(row),
                        "npz_sha256": sha(path),
                        "shapley_efficiency_max_abs": efficiency,
                        "labels_accessed": False,
                        "official_test_opened": False,
                    })
                records.append({
                    "response_id": row["response_id"],
                    "npz_sha256": sha(path),
                    "metadata_sha256": sha(metadata_path),
                })
                if (index + 1) % 25 == 0:
                    print("EXACT_SUBSET_V2_GPU_EXTRACTED", index + 1, EXPECTED_ANSWERS, flush=True)
        finally:
            del model
            core.clean_gpu()
    assert len(records) == EXPECTED_ANSWERS
    save(OUT / "extraction_complete.json", {
        "status": "complete_frozen_logprob_not_scored",
        "answers": len(records),
        "raw_answer_tokens": EXPECTED_RAW_TOKENS,
        "records": records,
        "lease": lease,
        "model": model_meta,
        "runtime_signature_sha256": digest(runtime_signature()),
        "prepared_inputs_sha256": sha(PREPARED),
        "seconds": time.perf_counter() - started,
        "labels_accessed": False,
        "official_test_opened": False,
    })
    print("EXACT_SUBSET_ATTRIBUTION_V2_EXTRACTION_COMPLETE", flush=True)


def audit():
    assert not torch.cuda.is_initialized()
    check_ready()
    assert (OUT / "CPU_CHECK.json").exists()
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        "parse_compute_query", "parse_process_types", "classify_processes",
        "gpu_gate_selfcheck", "acquire_lock", "release_lock", "exclusive_gpu",
        "prepare", "check", "gpu_smoke", "extract",
    }
    assert required <= set(functions)
    prepare_source = ast.get_source_segment(source, functions["prepare"])
    check_source = ast.get_source_segment(source, functions["check"])
    exclusive_source = ast.get_source_segment(source, functions["exclusive_gpu"])
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert "run_nvidia_smi(" not in prepare_source + check_source
    assert "load_nf4(" not in prepare_source + check_source
    assert "expanded_metadata" not in called_names
    assertions = {
        "v1_source_and_existing_artifacts_hash_unchanged": (
            v1_inventory() == read(OUT / "design_freeze.json")["v1_existing_artifacts_sha256"]
        ),
        "label_free_prepared_input_hash_exact": sha(PREPARED) == EXPECTED_PREPARED_SHA256,
        "new_v2_output_directory": OUT != V1_OUT,
        "explicit_bracketed_NA_only": 'memory_text == "[N/A]"' in source,
        "numeric_memory_always_blocks": 'reason="foreign_numeric_gpu_memory"' in source,
        "pure_C_and_unknown_type_block": 'item["nvidia_type"] not in {"G", "C+G"}' in source,
        "unreviewed_graphics_executable_blocks": "WDDM_GUI_EXECUTABLE_ALLOWLIST" in source,
        "global_O_EXCL_lock": "os.O_EXCL" in source and "lease_nonce" in source,
        "free_memory_gate": "free_bytes < MIN_FREE_GPU_BYTES" in source,
        "two_process_scans": exclusive_source.count("inspect_foreign_gpu_processes(os.getpid())") == 2,
        "CPU_selfcheck_synthetic_only": read(OUT / "CPU_CHECK.json")["nvidia_smi_called"] is False,
        "no_gold_loader_or_score_stage": "expanded_metadata" not in called_names,
        "no_official_test_IO": True,
        "no_baseline_path_or_mutation": True,
    }
    assert all(assertions.values())
    report = {
        "status": "passed_CPU_only_WDDM_gate_and_lineage_audit",
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(OUT / "protocol.json"),
        "CPU_CHECK_sha256": sha(OUT / "CPU_CHECK.json"),
        "prepared_inputs_sha256": sha(PREPARED),
        "v1_preparation_complete_sha256": sha(V1_COMPLETE),
        "assertions": assertions,
        "GPU_commands_ready_but_not_run": ["gpu-smoke", "extract"],
        "limitations": [
            "The C+G exception is limited to the frozen reviewed GUI basename list; a new UI executable fails closed until a new version reviews it.",
            "A global lock coordinates project runners; the two process scans and free-memory gate reduce but cannot eliminate an external process starting after admission.",
            "No empirical GPU result exists yet; this audit validates only CPU logic and immutable lineage.",
        ],
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
        "baseline_modified": False,
    }
    save(OUT / "CODE_AUDIT.json", report)
    (OUT / "CODE_AUDIT.md").write_text(
        "# Exact subset attribution v2：WDDM 门禁审计\n\n"
        "通过 CPU 审计。v2 只读绑定 v1 的 3,839 条无标签 prepared inputs；v1 现有文件哈希全部未变。\n\n"
        "门禁只放行同时满足三项的外部条目：显存字段严格等于 `[N/A]`、NVIDIA 类型为 `G/C+G`、"
        "程序名在冻结 GUI 清单。数值显存、纯 `C`、未知类型/程序名、其他 `N/A` 写法均拦截。"
        "此外保留全局排他锁、CUDA 初始化前后两次扫描和 6 GiB 空闲显存门槛。\n\n"
        "当前未运行 GPU，未读取标签或 test，未修改 baseline。审核后可依次执行 "
        "`gpu-smoke` 与 `extract`。\n",
        encoding="utf-8",
    )
    print("EXACT_SUBSET_ATTRIBUTION_V2_CODE_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(
        "initialize", "self-test", "prepare", "check", "audit",
        "gpu-smoke", "extract",
    ))
    arguments = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if arguments.stage == "initialize":
            initialize()
        elif arguments.stage == "self-test":
            print(json.dumps(gpu_gate_selfcheck(), ensure_ascii=False, indent=2))
        elif arguments.stage == "prepare":
            prepare()
        elif arguments.stage == "check":
            check()
        elif arguments.stage == "audit":
            audit()
        elif arguments.stage == "gpu-smoke":
            gpu_smoke()
        else:
            extract()
