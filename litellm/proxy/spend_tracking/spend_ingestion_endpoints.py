import math
import secrets
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import ProxyUpdateSpend
from litellm.types.utils import CallTypes
from litellm.utils import get_end_user_id_for_cost_tracking

router = APIRouter()

SUPPORTED_SPEND_INGEST_CALL_TYPES = {
    CallTypes.arealtime.value,
}


class SpendIngestionRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    api_key_hash: str = Field(..., min_length=1)
    call_type: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    spend: float = Field(...)
    start_time: datetime
    end_time: datetime
    usage: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    custom_llm_provider: Optional[str] = None
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _get_configured_master_key() -> Optional[str]:
    from litellm.proxy.proxy_server import master_key

    return master_key


async def _require_actual_master_key(
    authorization: Optional[str] = Header(default=None),
) -> None:
    master_key = _get_configured_master_key()
    if not isinstance(master_key, str) or master_key == "":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LiteLLM master key is required for spend ingestion",
        )

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing LiteLLM master key",
        )

    received_key = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(received_key, master_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid LiteLLM master key",
        )


def _validate_call_type(call_type: str) -> str:
    try:
        validated_call_type = CallTypes(call_type).value
    except ValueError as exc:
        supported_call_types = sorted(call_type.value for call_type in CallTypes)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Unsupported call_type: {call_type}",
                "supported_call_types": supported_call_types,
            },
        ) from exc
    if validated_call_type not in SUPPORTED_SPEND_INGEST_CALL_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Unsupported spend ingestion call_type: {call_type}",
                "supported_call_types": sorted(SUPPORTED_SPEND_INGEST_CALL_TYPES),
            },
        )
    return validated_call_type


def _validate_spend(spend: float) -> float:
    if not math.isfinite(spend) or spend < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="spend must be a finite non-negative number",
        )
    return spend


def _get_usage(payload: SpendIngestionRequest) -> Dict[str, Any]:
    usage = dict(payload.usage or {})
    # Keep usage provider-neutral. This endpoint does not normalize native
    # provider fields such as input_tokens/output_tokens into LiteLLM's
    # prompt_tokens/completion_tokens columns. Ingesters that need those spend
    # log columns populated must send prompt_tokens, completion_tokens, and
    # total_tokens explicitly.
    usage.setdefault("prompt_tokens", 0)
    usage.setdefault("completion_tokens", 0)
    usage.setdefault("total_tokens", 0)
    return usage


def _require_prisma_client() -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database is required for spend ingestion",
        )


async def _get_user_api_key_auth(api_key_hash: str) -> Any:
    from litellm.proxy._types import ProxyErrorTypes, ProxyException
    from litellm.proxy.auth.auth_checks import get_key_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    try:
        return await get_key_object(
            hashed_token=api_key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except ProxyException as exc:
        if exc.type == ProxyErrorTypes.token_not_found_in_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="api_key_hash was not found",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to lookup api_key_hash",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to lookup api_key_hash",
        ) from exc


def _merge_key_level_spend_metadata(
    metadata: Dict[str, Any], user_api_key_auth: Any
) -> None:
    key_metadata = getattr(user_api_key_auth, "metadata", None)
    if not isinstance(key_metadata, dict):
        return

    if key_metadata.get("tags") is not None:
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

        metadata["tags"] = LiteLLMProxyRequestSetup._merge_tags(
            request_tags=metadata.get("tags"),
            tags_to_add=key_metadata.get("tags"),
        )

    key_spend_logs_metadata = key_metadata.get("spend_logs_metadata")
    if not isinstance(key_spend_logs_metadata, dict):
        return

    existing_spend_logs_metadata = metadata.get("spend_logs_metadata")
    if isinstance(existing_spend_logs_metadata, dict):
        for key, value in key_spend_logs_metadata.items():
            if key not in existing_spend_logs_metadata:
                existing_spend_logs_metadata[key] = value
    else:
        metadata["spend_logs_metadata"] = dict(key_spend_logs_metadata)


def _build_metadata(
    payload: SpendIngestionRequest,
    user_api_key_auth: Any,
    api_key_hash: str,
) -> Dict[str, Any]:
    # Keep caller metadata as-is. Ingesters must use LiteLLM-native metadata
    # keys such as `tags` and `spend_logs_metadata` when they need downstream
    # spend-log behavior; this endpoint only adds the auth attribution fields.
    metadata = dict(payload.metadata or {})
    _merge_key_level_spend_metadata(
        metadata=metadata,
        user_api_key_auth=user_api_key_auth,
    )
    metadata.update(
        {
            "user_api_key": api_key_hash,
            "user_api_key_hash": api_key_hash,
            "user_api_key_alias": getattr(user_api_key_auth, "key_alias", None),
            "user_api_key_user_id": getattr(user_api_key_auth, "user_id", None),
            "user_api_key_team_id": getattr(user_api_key_auth, "team_id", None),
            "user_api_key_team_alias": getattr(user_api_key_auth, "team_alias", None),
            "user_api_key_org_id": getattr(user_api_key_auth, "org_id", None),
            "user_api_key_model_max_budget": getattr(
                user_api_key_auth, "model_max_budget", None
            ),
            "model_group": payload.model,
            "spend_ingestion": True,
        }
    )
    end_user_model_max_budget = getattr(
        user_api_key_auth, "end_user_model_max_budget", None
    )
    if end_user_model_max_budget is not None:
        metadata["user_api_key_end_user_model_max_budget"] = end_user_model_max_budget
    if payload.session_id is not None:
        metadata["session_id"] = payload.session_id
    key_agent_id = getattr(user_api_key_auth, "agent_id", None)
    existing_agent_id = metadata.get("agent_id")
    metadata["agent_id"] = key_agent_id or existing_agent_id
    if payload.model_id is not None:
        model_info = metadata.get("model_info") if isinstance(metadata, dict) else None
        if not isinstance(model_info, dict):
            model_info = {}
        model_info["id"] = payload.model_id
        metadata["model_info"] = model_info
    return metadata


def _get_active_api_key_hash(
    payload: SpendIngestionRequest, user_api_key_auth: Any
) -> str:
    # get_key_object can resolve a deprecated key hash to the active key row.
    # Spend must follow the resolved row because normal budget checks use token.
    return getattr(user_api_key_auth, "token", None) or payload.api_key_hash


async def _warm_active_api_key_cache(
    payload: SpendIngestionRequest,
    user_api_key_auth: Any,
    api_key_hash: str,
) -> None:
    if api_key_hash == payload.api_key_hash:
        return

    from litellm.proxy.auth.auth_checks import _cache_key_object
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _cache_key_object(
        hashed_token=api_key_hash,
        user_api_key_obj=user_api_key_auth,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


def _build_litellm_kwargs(
    payload: SpendIngestionRequest,
    call_type: str,
    user_api_key_auth: Any,
    api_key_hash: str,
    response_obj: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = _build_metadata(
        payload=payload,
        user_api_key_auth=user_api_key_auth,
        api_key_hash=api_key_hash,
    )
    litellm_params: Dict[str, Any] = {
        "custom_llm_provider": payload.custom_llm_provider,
        "metadata": metadata,
        "proxy_server_request": {},
    }
    if isinstance(metadata.get("model_info"), dict):
        litellm_params["model_info"] = metadata["model_info"]
    litellm_kwargs = {
        "call_type": call_type,
        "model": payload.model,
        "custom_llm_provider": payload.custom_llm_provider,
        "litellm_call_id": payload.request_id,
        "litellm_trace_id": payload.session_id or payload.request_id,
        "response_cost": payload.spend,
        "completion_start_time": payload.end_time,
        "cache_hit": False,
        "litellm_params": litellm_params,
    }
    litellm_kwargs["standard_logging_object"] = _build_standard_logging_object(
        payload=payload,
        call_type=call_type,
        metadata=metadata,
        response_obj=response_obj,
    )
    return litellm_kwargs


def _get_request_tags(metadata: Dict[str, Any]) -> list:
    tags = metadata.get("tags")
    return tags if isinstance(tags, list) else []


def _get_response_time_seconds(payload: SpendIngestionRequest) -> float:
    return max((payload.end_time - payload.start_time).total_seconds(), 0.0)


def _build_standard_logging_object(
    payload: SpendIngestionRequest,
    call_type: str,
    metadata: Dict[str, Any],
    response_obj: Dict[str, Any],
) -> Dict[str, Any]:
    usage = response_obj.get("usage") or {}
    if usage and "usage_object" not in metadata:
        metadata["usage_object"] = usage
    trace_id = payload.session_id or payload.request_id
    end_user_id = get_end_user_id_for_cost_tracking({"metadata": metadata})

    return {
        "id": payload.request_id,
        "litellm_call_id": payload.request_id,
        "trace_id": trace_id,
        "call_type": call_type,
        "stream": False,
        "response_cost": payload.spend,
        "cost_breakdown": None,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "status_fields": {"llm_api_status": "success"},
        "custom_llm_provider": payload.custom_llm_provider,
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "startTime": payload.start_time.timestamp(),
        "endTime": payload.end_time.timestamp(),
        "completionStartTime": payload.end_time.timestamp(),
        "response_time": _get_response_time_seconds(payload),
        "model_map_information": {
            "model_map_key": "",
            "model_map_value": None,
        },
        "model": payload.model,
        "model_id": payload.model_id,
        "model_group": payload.model,
        "api_base": "",
        "metadata": metadata,
        "cache_hit": False,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": _get_request_tags(metadata),
        "end_user": end_user_id or "",
        "requester_ip_address": metadata.get("requester_ip_address"),
        "user_agent": metadata.get("user_agent"),
        "messages": None,
        "response": response_obj,
        "error_str": None,
        "error_information": None,
        "model_parameters": {},
        "hidden_params": {
            "model_id": payload.model_id,
            "cache_key": None,
            "api_base": None,
            "response_cost": None,
            "litellm_overhead_time_ms": None,
            "additional_headers": None,
            "batch_models": None,
            "litellm_model_name": None,
            "usage_object": usage,
        },
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }


def _get_router_budget_callbacks(model_max_budget_limiter: Any) -> list:
    import litellm
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

    callbacks = []
    seen_callback_ids = set()
    for callback in litellm.logging_callback_manager._get_all_callbacks():
        if callback is model_max_budget_limiter:
            continue
        if not isinstance(callback, RouterBudgetLimiting):
            continue
        callback_id = id(callback)
        if callback_id in seen_callback_ids:
            continue
        seen_callback_ids.add(callback_id)
        callbacks.append(callback)
    return callbacks


def _get_session_budget_callbacks() -> list:
    import litellm
    from litellm.proxy.hooks.max_budget_per_session_limiter import (
        _PROXY_MaxBudgetPerSessionHandler,
    )

    callbacks = []
    seen_callback_ids = set()
    for callback in litellm.logging_callback_manager._get_all_callbacks():
        if not isinstance(callback, _PROXY_MaxBudgetPerSessionHandler):
            continue
        callback_id = id(callback)
        if callback_id in seen_callback_ids:
            continue
        seen_callback_ids.add(callback_id)
        callbacks.append(callback)
    return callbacks


async def _run_litellm_spend_callbacks(
    litellm_kwargs: Dict[str, Any],
    response_obj: Dict[str, Any],
    start_time: datetime,
    end_time: datetime,
) -> None:
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.proxy.proxy_server import model_max_budget_limiter

    callbacks = [
        ("proxy_db_logger", _ProxyDBLogger().async_log_success_event),
        ("model_max_budget_limiter", model_max_budget_limiter.async_log_success_event),
    ]
    callbacks.extend(
        (
            "max_budget_per_session_limiter",
            session_budget_callback.async_log_success_event,
        )
        for session_budget_callback in _get_session_budget_callbacks()
    )
    if litellm_kwargs.get("litellm_params", {}).get("custom_llm_provider") is not None:
        callbacks.extend(
            (
                "router_budget_limiter",
                router_budget_callback.async_log_success_event,
            )
            for router_budget_callback in _get_router_budget_callbacks(
                model_max_budget_limiter=model_max_budget_limiter,
            )
        )
    for callback_name, callback in callbacks:
        try:
            await callback(
                kwargs=litellm_kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception:
            verbose_proxy_logger.exception(
                "Spend ingestion callback failed: %s", callback_name
            )


@router.post(
    "/internal/spend/ingest",
    tags=["Internal"],
    include_in_schema=False,
)
async def ingest_spend(
    payload: SpendIngestionRequest,
    _: None = Depends(_require_actual_master_key),
):
    # Require the literal configured master key, not a PROXY_ADMIN virtual key
    # or no-auth dev-mode token. This endpoint can mutate spend for any key hash.
    # This ingestion API is intentionally non-idempotent in v1. `request_id`
    # is the spend-log event id, not a retry key. Internal senders must provide
    # a unique request_id per billable event; reposting the same request_id can
    # increment aggregate spend/counters more than once even though the later
    # spend-log flush may skip duplicate rows.
    call_type = _validate_call_type(payload.call_type)
    payload.spend = _validate_spend(payload.spend)
    _require_prisma_client()
    user_api_key_auth = await _get_user_api_key_auth(api_key_hash=payload.api_key_hash)
    api_key_hash = _get_active_api_key_hash(
        payload=payload,
        user_api_key_auth=user_api_key_auth,
    )

    usage = _get_usage(payload)
    response_obj = {
        "id": payload.request_id,
        "object": "spend.ingestion",
        "model": payload.model,
        "usage": usage,
    }
    litellm_kwargs = _build_litellm_kwargs(
        payload=payload,
        call_type=call_type,
        user_api_key_auth=user_api_key_auth,
        api_key_hash=api_key_hash,
        response_obj=response_obj,
    )

    if ProxyUpdateSpend.disable_spend_updates() is True:
        return {
            "status": "skipped",
            "reason": "spend updates are disabled",
            "request_id": payload.request_id,
            "call_type": call_type,
        }

    await _warm_active_api_key_cache(
        payload=payload,
        user_api_key_auth=user_api_key_auth,
        api_key_hash=api_key_hash,
    )
    # This endpoint acknowledges receipt, not durable spend-log persistence.
    # Match LiteLLM's best-effort cost callback behavior: callback failures are
    # logged, but do not turn the ingestion request into a retryable error.
    await _run_litellm_spend_callbacks(
        litellm_kwargs=litellm_kwargs,
        response_obj=response_obj,
        start_time=payload.start_time,
        end_time=payload.end_time,
    )

    return {
        "status": "accepted",
        "request_id": payload.request_id,
        "call_type": call_type,
    }
