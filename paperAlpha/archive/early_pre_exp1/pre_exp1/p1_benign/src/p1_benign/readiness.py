"""Default-deny live readiness checks with provider-usage enforcement."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .configuration import load_static_inputs, provider_catalog_entry
from .core import (
    LIVE_MODEL,
    ProjectContext,
    new_run_id,
    validate_run_id,
)


def check_live_readiness(
    *,
    context: ProjectContext,
    allow_live: bool,
    provider_id: str | None,
    model_id: str | None,
    run_id: str | None = None,
    endpoint: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness without opening `.env` or making any request."""

    static = load_static_inputs(context)
    catalog = (
        provider_catalog_entry(static, provider_id)
        if provider_id
        else None
    )
    candidate_run_id = run_id or new_run_id("P1-BENIGN-LIVE-")
    try:
        validate_run_id(candidate_run_id)
        run_id_valid = True
    except ValueError:
        run_id_valid = False
    artifact_paths = (
        context.outputs.artifact_paths(candidate_run_id)
        if run_id_valid
        else {}
    )
    outputs_unique = bool(artifact_paths) and not any(
        path.exists() for path in artifact_paths.values()
    )

    usage_allowed = bool(
        catalog
        and catalog.get("allowed_for_automated_experiment") is True
    )
    is_test = bool(catalog and catalog.get("provider_type") == "test_double")
    is_local = bool(catalog and catalog.get("provider_type") == "local")
    credential_env = (
        catalog.get("credential_env") if catalog else None
    )
    credential_checked = False
    credential_present = False
    local_endpoint_accessible = False

    # Usage-restricted providers are rejected before credential inspection.
    if catalog and not usage_allowed:
        credential_checked = False
    elif is_local:
        local_endpoint_accessible = bool(endpoint or catalog.get("endpoint"))
    elif isinstance(credential_env, str) and credential_env:
        credential_checked = True
        values = environment if environment is not None else os.environ
        credential_present = credential_env in values

    freeze = static.experiment["live_freeze"]
    budget = static.experiment["live_budget"]
    allowed_models = catalog.get("allowed_models", []) if catalog else []
    model_allowed = bool(
        catalog
        and model_id
        and (not allowed_models or model_id in allowed_models)
    )
    endpoint_configured = bool(
        endpoint or (catalog and catalog.get("endpoint"))
    )

    checks = {
        "explicit_allow_live": allow_live is True,
        "run_id_valid": run_id_valid,
        "run_id_unique_and_outputs_empty": outputs_unique,
        "provider_present": bool(provider_id),
        "provider_known": catalog is not None,
        "provider_not_test_double": bool(catalog) and not is_test,
        "provider_allowed_for_automated_experiment": usage_allowed,
        "provider_frozen": (
            freeze.get("provider_frozen") is True
            and freeze.get("provider_id") == provider_id
        ),
        "model_present": bool(model_id),
        "model_allowed_for_provider": model_allowed,
        "model_frozen": (
            freeze.get("model_frozen") is True
            and freeze.get("model_id") == model_id
        ),
        "decoding_frozen": freeze.get("decoding_frozen") is True,
        "budget_frozen": freeze.get("budget_frozen") is True,
        "max_episodes_configured": (
            isinstance(budget.get("max_episodes"), int)
            and budget["max_episodes"] > 0
        ),
        "max_calls_configured": (
            isinstance(budget.get("max_model_calls_per_episode"), int)
            and budget["max_model_calls_per_episode"] > 0
            and isinstance(budget.get("max_total_model_calls"), int)
            and budget["max_total_model_calls"] > 0
        ),
        "retry_cap_configured": (
            isinstance(budget.get("max_retries_per_call"), int)
            and budget["max_retries_per_call"] >= 0
        ),
        "token_cap_configured": (
            isinstance(budget.get("max_output_tokens_per_call"), int)
            and budget["max_output_tokens_per_call"] > 0
        ),
        "endpoint_configured": endpoint_configured,
        "credential_or_local_endpoint": (
            local_endpoint_accessible or credential_present
        ),
        "manifest_budget_evidence_configured": all(
            key in budget
            for key in (
                "max_episodes",
                "max_model_calls_per_episode",
                "max_total_model_calls",
                "max_retries_per_call",
                "max_output_tokens_per_call",
                "resume_policy",
            )
        ),
        "live_execution_enabled": freeze.get("live_enabled") is True,
    }
    reasons = [
        name for name, passed in checks.items() if not passed
    ]
    if catalog and catalog.get("usage_scope") == "interactive_coding_only":
        reason = (
            "provider_usage_scope_interactive_coding_only;"
            "automated_experiment_prohibited"
        )
        if reason not in reasons:
            reasons.insert(0, reason)
    ready = all(checks.values())
    return {
        "ready": ready,
        "execution_mode": LIVE_MODEL,
        "provider_id": provider_id,
        "model_id": model_id,
        "run_id": candidate_run_id,
        "provider_usage_scope": (
            catalog.get("usage_scope") if catalog else None
        ),
        "allowed_for_automated_experiment": (
            catalog.get("allowed_for_automated_experiment")
            if catalog
            else None
        ),
        "credential_env": credential_env,
        "credential_checked": credential_checked,
        "credential_present": credential_present,
        "checks": checks,
        "reasons": reasons,
        "network_request_attempted": False,
    }

