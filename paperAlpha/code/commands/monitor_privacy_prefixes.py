"""Isolated stdin/stdout monitor; accepts no future labels or API responses."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.privacy_frozen_monitor import FrozenPrivacyMonitor
from paperalpha_monitor.privacy_model_independent import make_packet


def observation_packet(request):
    if set(request) != {"initial", "vertical", "coordinator_text", "worker_text"}:
        raise ValueError("only initial policy and completed earlier messages may be projected")
    initial = request["initial"]
    if set(initial) != {"vault", "request", "allowed_set"} or set(initial["allowed_set"]) != {"fields"}:
        raise ValueError("strict initial policy fields required")
    return make_packet(initial, request["vertical"], request["coordinator_text"], request["worker_text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--observations", action="store_true", help="Project strict raw prefix envelopes inside the numerical process")
    args = parser.parse_args()
    monitor = FrozenPrivacyMonitor(args.models)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if args.observations:
            packet = observation_packet(request)
        else:
            if set(request) != {"packet"}:
                raise ValueError("only a visible prefix packet may be scored")
            packet = request["packet"]
        print(json.dumps({"ok": True, "packet": packet, "result": monitor.score(packet)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
