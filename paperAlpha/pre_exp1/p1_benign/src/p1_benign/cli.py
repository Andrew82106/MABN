"""Copyable P1 Phase A commands with non-zero safety failures."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .configuration import load_static_inputs, provider_catalog_entry
from .core import DEFAULT_CONTEXT, LIVE_MODEL, LOCAL_MODEL_SHAKEDOWN, TEST_DOUBLE
from .providers import (
    DeterministicTestProvider,
    OllamaLocalProvider,
    OpenAICompatibleProvider,
)
from .readiness import check_live_readiness
from .replay import replay_run
from .runner import run_p1
from .validation import validate_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p1-benign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser(
        "run-p1-dry",
        help="run deterministic non-scientific P1 engineering episodes",
    )
    dry.add_argument("--run-id")
    dry.add_argument("--episodes", type=int, default=2)

    local = subparsers.add_parser(
        "run-p1-local-shakedown",
        help="run at most one local qwen3:8b engineering episode",
    )
    local.add_argument("--run-id")
    local.add_argument("--model", default="qwen3:8b")
    local.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )
    local.add_argument(
        "--allow-local-shakedown",
        action="store_true",
    )

    validate = subparsers.add_parser(
        "validate-p1-run",
        help="validate one P1 run and return non-zero on failure",
    )
    validate.add_argument("--run-id", required=True)

    replay = subparsers.add_parser(
        "replay-p1-run",
        help="replay recorded outputs without a provider call",
    )
    replay.add_argument("--run-id", required=True)

    readiness = subparsers.add_parser(
        "check-p1-live-readiness",
        help="evaluate the default-deny live gate without a request",
    )
    readiness.add_argument("--provider-id")
    readiness.add_argument("--model-id")
    readiness.add_argument("--endpoint")
    readiness.add_argument("--run-id")
    readiness.add_argument("--allow-live", action="store_true")

    live = subparsers.add_parser(
        "run-p1-live",
        help="run only after every frozen live gate passes",
    )
    live.add_argument("--provider-id")
    live.add_argument("--model-id")
    live.add_argument("--endpoint")
    live.add_argument("--run-id")
    live.add_argument("--episodes", type=int, default=20)
    live.add_argument("--allow-live", action="store_true")
    return parser


def _print(value: dict) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    context=DEFAULT_CONTEXT,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-p1-dry":
            result = run_p1(
                context=context,
                provider=DeterministicTestProvider(),
                execution_mode=TEST_DOUBLE,
                episode_count=args.episodes,
                run_id=args.run_id,
                eligible_for_scientific_analysis=False,
            )
            _print(result)
            return 0 if result["passed"] else 1

        if args.command == "run-p1-local-shakedown":
            if not args.allow_local_shakedown:
                _print(
                    {
                        "passed": False,
                        "error": "explicit_local_shakedown_allow_required",
                        "network_request_attempted": False,
                    }
                )
                return 2
            if args.model != "qwen3:8b":
                _print(
                    {
                        "passed": False,
                        "error": "phase_a_model_must_be_qwen3:8b",
                        "model_download_attempted": False,
                    }
                )
                return 2
            provider = OllamaLocalProvider(
                model_id=args.model,
                endpoint=args.endpoint,
            )
            result = run_p1(
                context=context,
                provider=provider,
                execution_mode=LOCAL_MODEL_SHAKEDOWN,
                episode_count=1,
                run_id=args.run_id,
                eligible_for_scientific_analysis=False,
                allow_local_shakedown=True,
            )
            _print(result)
            return 0 if result["passed"] else 1

        if args.command == "validate-p1-run":
            result = validate_run(
                args.run_id,
                context=context,
                write_outputs=True,
            )
            _print(result)
            return 0 if result["passed"] else 1

        if args.command == "replay-p1-run":
            result = replay_run(
                args.run_id,
                context=context,
                write_output=True,
            )
            _print(result)
            return 0 if result["passed"] else 1

        if args.command == "check-p1-live-readiness":
            result = check_live_readiness(
                context=context,
                allow_live=args.allow_live,
                provider_id=args.provider_id,
                model_id=args.model_id,
                run_id=args.run_id,
                endpoint=args.endpoint,
            )
            _print(result)
            return 0 if result["ready"] else 2

        readiness = check_live_readiness(
            context=context,
            allow_live=args.allow_live,
            provider_id=args.provider_id,
            model_id=args.model_id,
            run_id=args.run_id,
            endpoint=args.endpoint,
        )
        if not readiness["ready"]:
            _print(readiness)
            return 2
        static = load_static_inputs(context)
        catalog = provider_catalog_entry(static, args.provider_id)
        assert catalog is not None
        provider = OpenAICompatibleProvider(
            provider_id=args.provider_id,
            model_id=args.model_id,
            endpoint=args.endpoint or catalog["endpoint"],
            credential_env=catalog["credential_env"],
            decoding=static.experiment["live_freeze"]["decoding"],
        )
        result = run_p1(
            context=context,
            provider=provider,
            execution_mode=LIVE_MODEL,
            episode_count=args.episodes,
            run_id=args.run_id,
            eligible_for_scientific_analysis=True,
        )
        _print(result)
        return 0 if result["passed"] else 1
    except Exception as exc:
        _print(
            {
                "passed": False,
                "error_type": type(exc).__name__,
                "credential_value_recorded": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
