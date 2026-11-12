"""Authorized one-shot GPU handoff, runtime smoke, five-epoch fit, frozen replay.

Only the parent GPU-release receipt permits subprocesses to start. This wrapper
does not load any partition annotations; test replay uses visible input geometry.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def runtime_stats(receipt):
    compute = sum(item["forward_seconds"] + item["backward_seconds"]
                  for item in receipt["per_answer_timings"])
    elapsed = receipt["elapsed_seconds"] - receipt["model_load_seconds"]
    tokens = sum(receipt["longest_fit_lengths"])
    return {"post_load_seconds": elapsed, "forward_backward_seconds": compute,
            "valid_tokens_per_post_load_second": tokens / elapsed,
            "valid_tokens_per_compute_second": tokens / compute,
            "peak_allocated_bytes": receipt["peak_allocated"],
            "peak_reserved_bytes": receipt["peak_reserved"]}


def select_runtime(old, new, smoke_exit_code):
    before = runtime_stats(old)
    after = runtime_stats(new) if new is not None else None
    successful = (smoke_exit_code == 0 and new is not None and new["status"] == "PASS"
                  and new["optimizer_step_completed"]
                  and sorted(new["longest_fit_lengths"]) == sorted(old["longest_fit_lengths"])
                  and new["lora_parameters"] == old["lora_parameters"]
                  and new["head_parameters"] == old["head_parameters"]
                  and new["joint_lora_gradient_groups"] == old["joint_lora_gradient_groups"])
    faster = bool(successful and after["post_load_seconds"] < before["post_load_seconds"]
                  and after["forward_backward_seconds"] < before["forward_backward_seconds"])
    return {"microbatch_size": 2 if faster else 1, "pair_within_update": faster,
            "smoke_exit_code": smoke_exit_code, "microbatch2_successful": bool(successful),
            "microbatch2_faster": faster, "microbatch1": before, "microbatch2": after,
            "rule": "Use2 only if the unique real-Adam longest-two smoke passes and both post-load total and forward/backward time improve; otherwise use1",
            "same_objective": "per-answer LM+BCE; equal answer mean; effective batch8; same original update membership",
            "test_labels_read": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-dir", type=Path, default=HERE / "ragognizer_pipeline")
    parser.add_argument("--out", type=Path, default=HERE / "ragognizer_run_v1")
    args = parser.parse_args()
    status_file = args.status_dir / "STATUS.json"
    if status_file.exists() or args.out.exists():
        raise FileExistsError("One-shot pipeline requires fresh status and run directories")
    smoke_out = HERE / "ragognizer_gpu_smoke_mb2_v1"
    predictions = {part: HERE / f"ragognizer_{part}_predictions_v1" for part in ("validation", "test")}
    if smoke_out.exists() or any(path.exists() for path in predictions.values()):
        raise FileExistsError("Smoke and prediction destinations must be fresh")
    started = time.time()
    status = {"stage": "waiting_for_gpu_handoff", "pid": os.getpid(), "started_at": started,
              "poll_seconds": 50, "run_dir": str(args.out), "test_labels_read": False}
    children = []
    sources = {name: sha(HERE / name) for name in
               ("ragognizer_same_domain.py", "ragognizer_batch_runtime.py", "predict_frozen_baselines.py")}

    def update(stage, **extra):
        status.update(stage=stage, updated_at=time.time(), **extra)
        save(status_file, status)

    def execute(stage, command, allow_failure=False):
        assert all(sha(HERE / name) == fingerprint for name, fingerprint in sources.items())
        full = [sys.executable, "-u", "-B", "-X", "utf8", *map(str, command)]
        update(stage, command=full, child_pid=None)
        with (args.status_dir / f"{stage}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            update(stage, child_pid=child.pid)
            code = child.wait()
        update(stage, child_pid=None, last_child_exit_code=code)
        if code and not allow_failure:
            raise subprocess.CalledProcessError(code, full)
        return code

    try:
        update("waiting_for_gpu_handoff", source_sha256=sources)
        release_file = RUN / "GPU_READY_FOR_RAGOGNIZER.json"
        while not release_file.exists():
            upstream = RUN / "prediction_pipeline/GPU_STATUS.json"
            if upstream.exists() and read(upstream).get("stage") == "failed":
                raise RuntimeError("Upstream GPU prediction pipeline failed; no GPU work started")
            time.sleep(50)
        release = read(release_file)
        if release.get("all_gpu_children_exited") is not True:
            raise RuntimeError("GPU handoff receipt does not confirm subprocess exit")
        save(args.status_dir / "GPU_HANDOFF.json", {"receipt": release, "sha256": sha(release_file),
                                                   "accepted_at": time.time()})
        old_path = HERE / "ragognizer_gpu_smoke_v1/GPU_SMOKE.json"
        old = read(old_path)
        code = execute("microbatch2_smoke", [HERE / "ragognizer_same_domain.py", "smoke",
                       "--out", smoke_out, "--microbatch-size", "2", "--pair-within-update", "--allow-gpu"],
                       allow_failure=True)
        new_path = smoke_out / "GPU_SMOKE.json"
        new = read(new_path) if new_path.exists() else None
        decision = select_runtime(old, new, code)
        decision.update(microbatch1_receipt_sha256=sha(old_path),
                        microbatch2_receipt_sha256=sha(new_path) if new_path.exists() else None,
                        smoke_log=str(args.status_dir / "microbatch2_smoke.log"))
        save(args.status_dir / "RUNTIME_SELECTION.json", decision)
        update("runtime_selected", runtime_selection=decision)
        print(json.dumps(decision), flush=True)
        flags = ["--microbatch-size", str(decision["microbatch_size"])]
        if decision["pair_within_update"]:
            flags.append("--pair-within-update")
        execute("five_epoch_training", [HERE / "ragognizer_same_domain.py", "train", "--out", args.out,
                                      *flags, "--allow-gpu"])
        completed = read(args.out / "COMPLETE.json")
        assert completed["status"] == "complete" and completed["epochs"] == 5
        save(args.out / "RUNTIME_SELECTION.json", decision)
        selected = read(args.out / "SELECTED.json")
        receipts = {}
        for partition, output in predictions.items():
            execute(f"frozen_{partition}_prediction", [HERE / "predict_frozen_baselines.py", "predict",
                    "--method", "ragognizer", "--partition", partition, "--run-dir", args.out,
                    "--out", output, "--allow-gpu"])
            receipt = read(output / "PREDICTION_RECEIPT.json")
            assert receipt["responses"] == (271 if partition == "validation" else 279)
            assert receipt["fit_called"] is False and receipt["threshold_selection_called"] is False
            receipts[partition] = {"path": str(output / "PREDICTION_RECEIPT.json"),
                                   "sha256": sha(output / "PREDICTION_RECEIPT.json"),
                                   "predictions": str(output / "predictions.jsonl")}
        assert all(child.poll() is not None for child in children)
        done = {"status": "complete", "run_dir": str(args.out), "selected": selected,
                "runtime_selection": decision, "prediction_receipts": receipts,
                "all_gpu_children_exited": True, "GPU_released": True, "test_labels_read": False,
                "elapsed_seconds": time.time()-started, "finished_at": time.time()}
        save(RUN / "RAG_PIPELINE_COMPLETE.json", done)
        update("complete_gpu_released", completion=done)
        print(json.dumps(done), flush=True)
    except BaseException:
        update("failed", traceback=traceback.format_exc(),
               all_gpu_children_exited=all(child.poll() is not None for child in children))
        raise


if __name__ == "__main__":
    main()
