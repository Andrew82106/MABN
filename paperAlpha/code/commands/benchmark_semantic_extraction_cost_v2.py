"""Bounded structured-output semantic extraction cost benchmark.

This is the submission-facing follow-up to v1.  It keeps the extractor outside
the risk evaluator, requests JSON mode when the endpoint supports it, and
allows one explicit retry only for an empty/invalid response.  Failed attempts
remain in the records and a final failure is an abstention; no result is joined
to labels or used for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_semantic_extraction_cost_v1 import (
    EXTRACTION_SCHEMA, _extract_json, _message, _percentile, _usage,
    digest, load_public_records, prompt_for, validate_endpoint,
)


def api_call(endpoint: str, key: str, model: str, prompt: str, timeout: float,
             max_tokens: int, json_object: bool, thinking_disabled: bool):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if json_object:
        body["response_format"] = {"type": "json_object"}
    if thinking_disabled:
        body["thinking"] = {"type": "disabled"}
    request = Request(endpoint + "/chat/completions",
                      data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + key,
                               "Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), time.perf_counter() - started, int(response.status)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"http_{exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"transport_{type(exc).__name__}: {exc}") from exc


def run(source: Path, output: Path, endpoint: str, model: str, key: str,
        timeout: float, max_tokens: int, limit: int | None,
        json_object: bool, thinking_disabled: bool, retry_empty: bool) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    endpoint = validate_endpoint(endpoint)
    rows = load_public_records(source, limit)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "paperalpha-semantic-extraction-cost-v2",
        "source": str(source), "source_records": len(rows),
        "source_sha256": digest([{k: row[k] for k in ("source_id", "message_sha256", "source_usage")} for row in rows]),
        "endpoint": endpoint, "model": model, "prompt_schema": EXTRACTION_SCHEMA,
        "limits": {"timeout_seconds": timeout, "max_tokens": max_tokens,
                    "retry_empty_or_invalid": bool(retry_empty), "temperature": 0,
                    "json_object": bool(json_object), "thinking_disabled": bool(thinking_disabled)},
        "not_for": ["risk scoring", "classification metrics", "training", "label generation"],
    }
    (output / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attempts = completed = failed = parsed_first = parsed_final = parse_failed_first = 0
    latencies, tokens = [], []
    records = []
    for row in rows:
        prompt = prompt_for(row)
        base = {"source_id": row["source_id"], "episode": row["episode"],
                "message_sha256": row["message_sha256"], "request_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "attempts": []}
        final_output = None
        first_parse_ok = False
        for attempt_no in range(1, 3 if retry_empty else 2):
            attempts += 1
            started = time.time()
            try:
                payload, elapsed, status = api_call(endpoint, key, model, prompt, timeout,
                                                    max_tokens, json_object, thinking_disabled)
                completed += 1
                usage = _usage(payload); tokens.append(usage); latencies.append(elapsed)
                item = {"attempt": attempt_no, "status": "completed", "http_status": status,
                        "elapsed_seconds": elapsed, "usage": usage,
                        "response_model": payload.get("model"), "started_at": started}
                try:
                    text = _message(payload)
                    item["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    item["output"] = _extract_json(text)
                    item["parse_status"] = "parsed"
                    if attempt_no == 1:
                        parsed_first += 1; first_parse_ok = True
                    final_output = item["output"]; base["attempts"].append(item)
                    break
                except Exception as exc:
                    item.update(parse_status="parse_failed", parse_error=str(exc)[:300])
                    if attempt_no == 1:
                        parse_failed_first += 1
                    base["attempts"].append(item)
            except Exception as exc:
                failed += 1
                base["attempts"].append({"attempt": attempt_no, "status": "api_failed",
                                         "error": str(exc)[:300], "started_at": started})
            if attempt_no == 1 and not retry_empty:
                break
        if final_output is not None:
            parsed_final += 1
        base["final_status"] = "parsed" if final_output is not None else "abstain"
        base["output"] = final_output
        records.append(base)
        with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(base, ensure_ascii=False, sort_keys=True) + "\n")
    input_tokens = [v["input_tokens"] for v in tokens]; output_tokens = [v["output_tokens"] for v in tokens]
    report = {
        "schema": "paperalpha-semantic-extraction-cost-v2-report",
        "plan": plan,
        "counts": {"source_records": len(rows), "requests_started": attempts,
                    "api_completed": completed, "api_failed": failed,
                    "parsed_first": parsed_first, "parse_failed_first": parse_failed_first,
                    "parsed_final": parsed_final, "final_abstain": len(rows) - parsed_final,
                    "first_parse_failure_rate": parse_failed_first / len(rows) if rows else None,
                    "final_abstain_rate": (len(rows) - parsed_final) / len(rows) if rows else None},
        "latency_seconds": {"p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95)},
        "tokens": {"input": sum(input_tokens), "output": sum(output_tokens), "total": sum(input_tokens) + sum(output_tokens)},
        "scope": "Online structured extraction cost only; no semantic accuracy or risk metric.",
    }
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(
        "# Online semantic extraction cost benchmark v2\n\n" +
        f"- Source records: **{len(rows)}**\n- Requests: **{attempts}** (completed {completed}, failed {failed})\n" +
        f"- First strict parse failures: **{parse_failed_first}/{len(rows)}**\n" +
        f"- Final parsed: **{parsed_final}/{len(rows)}**; final abstain rate **{report['counts']['final_abstain_rate']}**\n" +
        f"- Latency p50/p95: **{report['latency_seconds']['p50']} / {report['latency_seconds']['p95']} s**\n" +
        f"- Tokens input/output: **{sum(input_tokens)} / {sum(output_tokens)}**\n\n" +
        "A failed or invalid response is an explicit abstention. Results are not joined to labels or risk metrics.\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--key-env", default="PAPERALPHA_API_KEY")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-json-object", action="store_true")
    parser.add_argument("--thinking-disabled", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    args = parser.parse_args()
    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"{args.key_env} is required in the process environment")
    report = run(args.source, args.output, args.endpoint, args.model, key, args.timeout,
                 args.max_tokens, args.limit, not args.no_json_object,
                 args.thinking_disabled, not args.no_retry)
    print(json.dumps({"output": str(args.output.resolve()), "counts": report["counts"],
                      "latency_seconds": report["latency_seconds"], "tokens": report["tokens"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
