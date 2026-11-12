"""Read-only development support diagnostics, never fit or call an API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_repeated_message_study import verify, sha
from paperalpha_monitor.repeated_message_diagnostics import diagnose_verified_collection


def diagnose(out):
    out = Path(out).resolve()
    verified = verify(out)
    return {**diagnose_verified_collection(verified), "plan_sha256": sha(out / "PLAN.json"),
            "collection_sha256": sha(out / "COLLECTION.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if ROOT != args.out.resolve() / "source":
        raise ValueError("run diagnostics from this collection's frozen source")
    print(json.dumps(diagnose(args.out), ensure_ascii=False, sort_keys=True, allow_nan=False))
