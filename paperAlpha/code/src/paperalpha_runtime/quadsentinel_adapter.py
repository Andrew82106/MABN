"""Optional transport adapter for the pinned author QuadSentinel implementation.

This is an event-time verdict, NOT a forecast probability. No guard verdict is
fed back to a protected agent. The full-predicate profile uses the author's
public k=-1 option; it does not reproduce embedding retrieval or paper scores.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

from .privacy_transfer import delivered_text

UPSTREAM_COMMIT = "1d4e29443284385a2505acb1e7986d9d80dc97f1"
PROFILE = "quadsentinel-author-full-predicates-gpt55-monitor-only-v1"


def inspect_checkout(path):
    path = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("unexpected author revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("author tracked source was modified")
    names = git("ls-files").splitlines()
    return {"commit": UPSTREAM_COMMIT, "repository": "https://github.com/yyiliu/QuadSentinel",
            "files_sha256": {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in names},
            "license_file_present": any("license" in n.lower() for n in names)}


def validate_policy_book(book):
    """Only an explicitly compiled boolean policy book may reach upstream eval.

    Do not execute arbitrary model-generated Python. This admission check does
    not replace, fix, or reinterpret the author's propositional evaluator.
    IMPLIES is excluded from this initial profile (its parser needs a separate
    compatibility audit); equivalent trusted AND/OR/NOT policies are supported.
    """
    if not isinstance(book, list) or not book:
        raise ValueError("nonempty compiled policy book required")
    names, descriptions = {}, set()
    for rule in book:
        if set(rule) != {"description", "predicates", "logic"} or not isinstance(rule["description"], str):
            raise ValueError("explicit policy description/predicates/logic required")
        if rule["description"] in descriptions:
            raise ValueError("duplicate policy description")
        descriptions.add(rule["description"])
        for entry in rule["predicates"]:
            if not isinstance(entry, list) or len(entry) != 4:
                raise ValueError("four-part predicate definition required")
            name, meaning, keywords, value = entry
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z_0-9]*", name):
                raise ValueError("snake-case predicate names required")
            if not isinstance(meaning, str) or not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords) or type(value) is not bool:
                raise ValueError("typed predicate definition required")
            if name in names and names[name] != entry:
                raise ValueError("conflicting predicate definitions")
            names[name] = entry
    for rule in book:
        logic = rule["logic"]
        if not isinstance(logic, str) or not logic.strip():
            raise ValueError("boolean expression required")
        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[()]", logic)
        if "".join(tokens) != re.sub(r"\s+", "", logic):
            raise ValueError("only boolean policy tokens allowed")
        if not set(tokens) <= set(names) | {"AND", "OR", "NOT", "(", ")"}:
            raise ValueError("unknown predicate or unsupported boolean operator")
        # Validate the grammar without executing even the trusted expression.
        import ast
        translated = " ".join({"AND": "and", "OR": "or", "NOT": "not"}.get(t, t) for t in tokens)
        tree = ast.parse(translated, mode="eval")
        allowed = (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Name, ast.Load, ast.And, ast.Or, ast.Not)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise ValueError("unsupported policy grammar")
    return copy.deepcopy(book)


def response_request(messages):
    converted = []
    for message in messages:
        role = {"SystemMessage": "system", "UserMessage": "user"}.get(type(message).__name__)
        if role is None or not isinstance(message.content, str):
            raise ValueError("the guard profile supports textual system/user prompts only")
        converted.append({"role": role, "content": message.content})
    return {"model": "gpt-5.5", "reasoning": {"effort": "low"}, "input": converted,
            "max_output_tokens": 3200, "store": False}


class ResponsesGuardClient:
    """Duck-typed model client used by the unmodified author agents."""
    def __init__(self, client, record, max_calls=48):
        self.client, self.record, self.max_calls = client, record, max_calls
        self.calls = 0

    async def create(self, *, messages, json_output=None, cancellation_token=None, **kwargs):
        if kwargs or json_output is not None:
            raise ValueError("unexpected author request mode")
        if self.calls >= self.max_calls:
            raise RuntimeError("guard API call budget exhausted")
        from autogen_core.models import CreateResult, RequestUsage
        request = response_request(messages)
        self.calls += 1
        identity = self.calls
        self.record({"kind": "api_request", "id": identity, "request": request, "monotonic_ns": time.monotonic_ns()})
        try:
            raw = (await self.client.responses.create(**request)).model_dump(mode="json")
            # Retain public response evidence, but no hidden reasoning or credentials.
            response = {k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage")}
            response["output"] = [m for m in raw.get("output", []) if m.get("type") == "message"]
            self.record({"kind": "api_response", "id": identity, "response": response, "monotonic_ns": time.monotonic_ns()})
            reply = delivered_text(response)
            if reply["native_refusal"]:
                raise RuntimeError("guard model refusal; not a safe verdict")
        except Exception as exc:
            self.record({"kind": "api_failure", "id": identity, "error_type": type(exc).__name__})
            # Prevent upstream JSON retries from converting truncation to another attempt.
            raise RuntimeError("guard request did not produce an eligible reply") from None
        usage = raw.get("usage") or {}
        return CreateResult(content=reply["text"], finish_reason="stop", cached=False,
                            usage=RequestUsage(prompt_tokens=usage.get("input_tokens", 0), completion_tokens=usage.get("output_tokens", 0)))


def create_api_client(endpoint, key):
    u = urlparse(endpoint)
    if u.scheme != "http" or u.hostname not in {"localhost", "127.0.0.1"} or u.port != 58661 or u.path.rstrip("/") != "/v1" or u.query or u.fragment or u.username:
        raise ValueError("this profile only permits the user-designated local endpoint")
    if not key:
        raise ValueError("runtime API credential required")
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=endpoint, api_key=key, timeout=60., max_retries=0)


class FullPredicateEmbedding:
    """Storage placeholder ONLY: no retrieval calls are permitted by the profile."""
    def __call__(self, input):
        return [[1., 0.] for _ in input]

    def name(self):
        return "unused-full-predicate-storage"


async def create_author_guard(checkout, model_client, book):
    inspect_checkout(checkout)
    book = validate_policy_book(book)
    sys.path.insert(0, str(Path(checkout).resolve() / "src"))
    from quadsentinel import Guard
    import inspect
    if not Path(inspect.getfile(Guard)).resolve().is_relative_to(Path(checkout).resolve()):
        raise ValueError("another QuadSentinel package was already loaded")
    guard = Guard(model_client, model_client, model_client, FullPredicateEmbedding())
    await guard.init()
    guard.add_policy_from_dict(book)
    guard.add_message_policy_from_dict(book)
    return guard


async def observe_event(guard, event):
    """Mirror author intervention ordering, without dispatch or agent feedback."""
    if event["kind"] == "message":
        return await guard.handle_message(event["text"], k=-1, sender=event.get("sender"), recipient=event.get("recipient"))
    if event["kind"] != "tool_proposal":
        raise ValueError("only messages and actual pre-dispatch proposals are supported")
    from autogen_core import FunctionCall
    call = FunctionCall(id=event["call_id"], name=event["tool"], arguments=json.dumps(event["arguments"], ensure_ascii=False))
    # Same order as utils/intervention.py for ToolCallRequestEvent: first
    # handle_message(str(content), k=-1), then handle_action if still allowed.
    allowed, reason = await guard.handle_message(str([call]), k=-1, sender=event.get("sender"), recipient=event.get("recipient"))
    if not allowed:
        return allowed, reason
    return await guard.handle_action(call.name, call.arguments, description=event["description"], sender=event.get("sender"))
