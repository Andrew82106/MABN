"""Benchmark online semantic extraction cost on public MAS API responses.

This benchmark is deliberately separate from risk evaluation.  It reads only
the public ``api_responses.jsonl`` message text from a completed unified
topology run and asks a low-cost model to produce a small event schema.  It
records transport latency, token usage and JSON parse failures; it never reads
evaluator files, labels or hidden prompts, and it never writes a risk score.

Credentials are read from the process environment and are never persisted.
The output directory is expected to be ignored by git.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_SCHEMA = "paperalpha-semantic-cost-source-v1"
REPORT_SCHEMA = "paperalpha-semantic-extraction-cost-v1"
EXTRACTION_SCHEMA = {
    "event_type": "one of observation, decision, handoff, tool_call, review, other",
    "action": "short normalized action phrase",
    "target": "short object or recipient phrase, or null",
    "risk_signal": "one of none, authorization, scope, privacy, resource, workflow, uncertainty, other",
    "evidence_quote": "an exact short quote from the supplied response, or null",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def validate_endpoint(value: str) -> str:
    parsed = urlparse(value)
    local = (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
             and parsed.port == 58661 and parsed.path.rstrip("/") == "/v1")
    lanyun = (parsed.scheme == "https" and parsed.hostname == "maas-api.lanyun.net"
              and parsed.port is None and parsed.path.rstrip("/") == "/v1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not (local or lanyun):
        raise ValueError("endpoint must be localhost:58661/v1 or maas-api.lanyun.net/v1")
    if lanyun and os.environ.get("PAPERALPHA_ALLOW_EXTERNAL_ENDPOINT") != "1":
        raise ValueError("set PAPERALPHA_ALLOW_EXTERNAL_ENDPOINT=1 for Lanyun")
    return value.rstrip("/")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    low, high = int(index), min(int(index) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def _usage(payload: dict) -> dict[str, int]:
    usage = payload.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    try:
        input_tokens, output_tokens = int(input_tokens), int(output_tokens)
    except (TypeError, ValueError):
        input_tokens = output_tokens = 0
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _message(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item)
                           for item in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has empty message content")
    return content.strip()


def _extract_json(text: str) -> dict:
    """Parse strict JSON while accepting a single markdown code fence."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        # A model occasionally adds one sentence around the object.  Extracting
        # the first balanced object is a parse aid, not a semantic correction.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in extraction response")
        value = json.loads(candidate[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("extraction output must be a JSON object")
    required = set(EXTRACTION_SCHEMA)
    if set(value) != required:
        raise ValueError("extraction keys differ from frozen schema")
    if value["event_type"] not in {"observation", "decision", "handoff", "tool_call", "review", "other"}:
        raise ValueError("invalid event_type")
    if value["risk_signal"] not in {"none", "authorization", "scope", "privacy", "resource", "workflow", "uncertainty", "other"}:
        raise ValueError("invalid risk_signal")
    if not isinstance(value["action"], str) or not value["action"].strip():
        raise ValueError("action must be a nonempty string")
    for key in ("target", "evidence_quote"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"{key} must be string or null")
    return value


def load_public_records(source: Path, limit: int | None) -> list[dict]:
    """Load completed response messages only; evaluator artifacts are forbidden."""
    source = source.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(source)
    rows: list[dict] = []
    for path in sorted(source.glob("episodes/*/api_responses.jsonl")):
        episode = path.parent.name
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") != "completed" or not isinstance(item.get("message"), str):
                # Failed API attempts remain visible in the report as source
                # misses but do not become synthetic semantic observations.
                continue
            rows.append({
                "source_id": f"{episode}:{line_number}",
                "episode": episode,
                "source_path": str(path.relative_to(source)),
                "message": item["message"],
                "message_sha256": hashlib.sha256(item["message"].encode("utf-8")).hexdigest(),
                "source_usage": _usage(item),
            })
    rows.sort(key=lambda row: row["source_id"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no completed public API responses found")
    return rows


def prompt_for(row: dict) -> str:
    # Deliberately no variant, evaluator score, expected label or hidden prompt.
    return (
        "Extract one observable event from the public multi-agent response below. "
        "Return ONLY one JSON object with exactly these keys: "
        f"{json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, sort_keys=True)}. "
        "Do not infer hidden intent, truth, safety labels or probabilities. "
        "Use evidence_quote only when the quote appears verbatim in the response.\n"
        f"Public response:\n{row['message']}"
    )


def api_call(endpoint: str, key: str, model: str, prompt: str, timeout: float, max_tokens: int) -> tuple[dict, float, int]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    request = Request(endpoint + "/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, time.perf_counter() - started, int(response.status)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"http_{exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"transport_{type(exc).__name__}: {exc}") from exc


def run(source: Path, output: Path, endpoint: str, model: str, key: str,
        timeout: float, max_tokens: int, limit: int | None) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    endpoint = validate_endpoint(endpoint)
    rows = load_public_records(source, limit)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": PUBLIC_SCHEMA,
        "created_at": utc_now(),
        "source": str(source),
        "source_scope": "public completed API response messages only; evaluator files and labels excluded",
        "source_records": len(rows),
        "source_sha256": digest([{k: row[k] for k in ("source_id", "message_sha256", "source_usage")} for row in rows]),
        "endpoint": endpoint,
        "model": model,
        "prompt_schema": EXTRACTION_SCHEMA,
        "max_tokens": max_tokens,
        "limits": {"timeout_seconds": timeout, "retries": 0, "temperature": 0},
        "not_for": ["risk scoring", "classification metrics", "training", "label generation"],
    }
    (output / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    requests = completed = failed = parsed = parse_failed = 0
    latencies: list[float] = []
    token_values: list[dict[str, int]] = []
    records: list[dict] = []
    for row in rows:
        requests += 1
        prompt = prompt_for(row)
        record = {"source_id": row["source_id"], "episode": row["episode"],
                  "message_sha256": row["message_sha256"], "status": "not_started",
                  "request_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  "input_chars": len(prompt), "output": None}
        started_at = utc_now()
        try:
            payload, elapsed, http_status = api_call(endpoint, key, model, prompt, timeout, max_tokens)
            completed += 1
            usage = _usage(payload)
            latencies.append(elapsed)
            token_values.append(usage)
            record.update(status="completed", http_status=http_status, elapsed_seconds=elapsed,
                          usage=usage, response_model=payload.get("model"), started_at=started_at,
                          received_at=utc_now())
            try:
                text = _message(payload)
                record["raw_output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record["output"] = _extract_json(text)
                record["parse_status"] = "parsed"
                parsed += 1
            except Exception as exc:
                record.update(parse_status="parse_failed", parse_error_type=type(exc).__name__,
                              parse_error=str(exc)[:300])
                parse_failed += 1
        except Exception as exc:
            failed += 1
            record.update(status="api_failed", started_at=started_at, received_at=utc_now(),
                          error_type=type(exc).__name__, error=str(exc)[:300])
        records.append(record)
        json_line(output / "records.jsonl", record)

    def summary(values: list[float]) -> dict:
        return {"n": len(values), "mean": statistics.fmean(values) if values else None,
                "p50": _percentile(values, .5), "p95": _percentile(values, .95)}

    input_tokens = [item["input_tokens"] for item in token_values]
    output_tokens = [item["output_tokens"] for item in token_values]
    total_tokens = [item["total_tokens"] for item in token_values]
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": utc_now(),
        "plan": plan,
        "counts": {"source_records": len(rows), "requests_started": requests,
                    "api_completed": completed, "api_failed": failed,
                    "parsed": parsed, "parse_failed": parse_failed,
                    "parse_failure_rate_among_completed": parse_failed / completed if completed else None},
        "latency_seconds": summary(latencies),
        "tokens": {"input": summary(input_tokens), "output": summary(output_tokens),
                   "total": summary(total_tokens), "sum": {"input": sum(input_tokens),
                   "output": sum(output_tokens), "total": sum(total_tokens)}},
        "scope": "Online semantic extraction cost only; no semantic accuracy or risk metric; output is not joined to evaluation.",
        "limitations": [
            "The source is a small API integration cohort, not an independent confirmation set.",
            "Latency includes semantic-extractor API transport and generation, but not target-agent generation.",
            "No accuracy is inferred from parsing success; evidence is recorded for cost and observability only.",
        ],
    }
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Online semantic extraction cost benchmark", "",
             f"- Source public responses: **{len(rows)}**",
             f"- API calls: **{requests}** started, **{completed}** completed, **{failed}** failed",
             f"- JSON parsing: **{parsed}** parsed, **{parse_failed}** failed ({report['counts']['parse_failure_rate_among_completed']})",
             f"- Latency p50/p95: **{report['latency_seconds']['p50']} / {report['latency_seconds']['p95']} s**",
             f"- Total input/output tokens: **{sum(input_tokens)} / {sum(output_tokens)}**", "",
             "This is a cost/observability benchmark only. It is not joined to the risk evaluator and does not support a safety-accuracy claim."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--key-env", default="PAPERALPHA_API_KEY")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"{args.key_env} is required in the process environment")
    report = run(args.source, args.output, args.endpoint, args.model, key,
                 args.timeout, args.max_tokens, args.limit)
    print(json.dumps({"output": str(args.output.resolve()), "counts": report["counts"],
                      "latency_seconds": report["latency_seconds"], "tokens": report["tokens"]["sum"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
