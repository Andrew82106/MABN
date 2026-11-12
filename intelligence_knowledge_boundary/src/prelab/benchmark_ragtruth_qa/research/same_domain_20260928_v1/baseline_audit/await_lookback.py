"""Wait for the shared extraction once, then execute the frozen CPU LR grid."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "lookback_run_v2")
    parser.add_argument("--status-dir", type=Path, default=HERE / "lookback_pipeline")
    args = parser.parse_args()
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_dir / "STATUS.json"
    if status_path.exists():
        raise FileExistsError("Pipeline was already started; inspect its saved PID/status")
    ids = []
    for partition in ("fit", "validation"):
        with (RUN / "data" / partition / "geometry.jsonl").open(encoding="utf-8") as f:
            ids.extend(json.loads(line)["response_id"] for line in f if line.strip())
    assert len(ids) == len(set(ids)) == 1563
    started = time.time()
    status = {"state": "waiting_for_shared_signals", "pid": os.getpid(), "started_unix": started,
              "poll_seconds": 50, "responses_required": len(ids), "out": str(args.out.resolve()),
              "test_labels_opened": False, "gpu_used": False}
    save(status_path, status)
    print("Waiting for COMPLETE_fit_validation.json and all1563 caches", flush=True)
    complete = RUN / "signals/COMPLETE_fit_validation.json"
    while True:
        if complete.exists():
            manifest = json.loads(complete.read_text(encoding="utf-8"))
            if manifest["count"] != len(ids):
                raise ValueError("Unexpected extraction completion count")
            missing = [rid for rid in ids if not (RUN / "signals" / f"{rid}.npz").exists()
                       or not (RUN / "signals" / f"{rid}.json").exists()]
            if not missing:
                break
        time.sleep(50)
    status.update(state="training_C_grid", training_started_unix=time.time())
    save(status_path, status)
    command = [sys.executable, "-X", "utf8", str(HERE / "lookback_same_domain.py"), "--out", str(args.out)]
    log = args.status_dir / "training.log"
    print("Shared signals complete; starting four-C CPU LR grid", flush=True)
    with log.open("w", encoding="utf-8") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        status["training_pid"] = child.pid
        save(status_path, status)
        code = child.wait()
    status.update(state="complete" if code == 0 else "failed", exit_code=code,
                  finished_unix=time.time(), total_seconds=time.time()-started, log=str(log.resolve()))
    save(status_path, status)
    print(json.dumps(status), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
