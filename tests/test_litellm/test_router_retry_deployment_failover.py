from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.types.llms.openai import ResponsesAPIResponse


def _router(
    *,
    model_name: str = "responses-model",
    litellm_model: str = "openai/gpt-4.1-mini",
    enable_retry_deployment_failover: bool = False,
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": litellm_model,
                    "api_key": "test-key-a",
                    "num_retries": 1,
                    "cooldown_time": 0,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": litellm_model,
                    "api_key": "test-key-b",
                    "num_retries": 1,
                    "cooldown_time": 0,
                },
                "model_info": {"id": "B"},
            },
        ],
        num_retries=0,
        disable_cooldowns=True,
        optional_pre_call_checks=["deployment_affinity"],
        enable_retry_deployment_failover=enable_retry_deployment_failover,
    )


def _rate_limit_error(
    *,
    model: str = "openai/gpt-4.1-mini",
    url: str = "https://api.openai.com/v1/responses",
) -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="upstream throttled",
        llm_provider="openai",
        model=model,
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", url),
        ),
    )


def _response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_123",
        created_at=0,
        model="gpt-4.1-mini",
        output=[],
    )


def _first_a_then_b(seq):
    if len(seq) == 1:
        return seq[0]
    for deployment in seq:
        if deployment["model_info"]["id"] == "A":
            return deployment
    return seq[0]


@pytest.mark.asyncio
async def test_retry_deployment_failover_disabled_preserves_affinity_target():
    router = _router(enable_retry_deployment_failover=False)
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        if deployment_id == "A":
            raise _rate_limit_error()
        return _response()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            side_effect=handler,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_a_then_b,
        ),
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aresponses(
                model="responses-model",
                input="hi",
                litellm_metadata={"user_api_key_hash": "user-1"},
            )

    assert seen_deployment_ids == ["A", "A"]


@pytest.mark.asyncio
async def test_retry_deployment_failover_excludes_failed_deployment_before_affinity():
    router = _router(enable_retry_deployment_failover=True)
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        if deployment_id == "A":
            raise _rate_limit_error()
        return _response()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            side_effect=handler,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_a_then_b,
        ),
    ):
        await router.aresponses(
            model="responses-model",
            input="hi",
            litellm_metadata={"user_api_key_hash": "user-1"},
        )

    assert seen_deployment_ids == ["A", "B"]


@pytest.mark.asyncio
async def test_retry_deployment_failover_ignores_caller_supplied_exclusion_state():
    router = _router(enable_retry_deployment_failover=True)
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        return _response()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            side_effect=handler,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_a_then_b,
        ),
    ):
        await router.aresponses(
            model="responses-model",
            input="hi",
            litellm_metadata={"user_api_key_hash": "user-1"},
            _retry_excluded_deployment_ids=["A"],
            _retry_last_failed_deployment_id="A",
        )

    assert seen_deployment_ids == ["A"]


@pytest.mark.asyncio
async def test_retry_deployment_failover_applies_to_anthropic_messages():
    router = _router(
        model_name="messages-model",
        litellm_model="anthropic/claude-sonnet-4-5-20250929",
        enable_retry_deployment_failover=True,
    )
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        if deployment_id == "A":
            raise _rate_limit_error(
                model="anthropic/claude-sonnet-4-5-20250929",
                url="https://api.anthropic.com/v1/messages",
            )
        return {"id": "msg_123", "content": [{"type": "text", "text": "ok"}]}

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
            side_effect=handler,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_a_then_b,
        ),
    ):
        await router.aanthropic_messages(
            model="messages-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            litellm_metadata={"user_api_key_hash": "user-1"},
        )

    assert seen_deployment_ids == ["A", "B"]


@pytest.mark.asyncio
async def test_retry_deployment_failover_avoids_last_failed_after_all_deployments_fail():
    router = Router(
        model_list=[
            {
                "model_name": "responses-model",
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": "test-key-a",
                    "num_retries": 3,
                    "cooldown_time": 0,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "responses-model",
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": "test-key-b",
                    "num_retries": 3,
                    "cooldown_time": 0,
                },
                "model_info": {"id": "B"},
            },
        ],
        num_retries=0,
        disable_cooldowns=True,
        optional_pre_call_checks=["deployment_affinity"],
        enable_retry_deployment_failover=True,
    )
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        raise _rate_limit_error()

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            side_effect=handler,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=_first_a_then_b,
        ),
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aresponses(
                model="responses-model",
                input="hi",
                litellm_metadata={"user_api_key_hash": "user-1"},
            )

    assert seen_deployment_ids == ["A", "B", "A", "B"]


@pytest.mark.asyncio
async def test_retry_deployment_failover_single_deployment_falls_back_to_original_list():
    router = Router(
        model_list=[
            {
                "model_name": "responses-model",
                "litellm_params": {
                    "model": "openai/gpt-4.1-mini",
                    "api_key": "test-key-a",
                    "num_retries": 1,
                    "cooldown_time": 0,
                },
                "model_info": {"id": "A"},
            }
        ],
        num_retries=0,
        disable_cooldowns=True,
        optional_pre_call_checks=["deployment_affinity"],
        enable_retry_deployment_failover=True,
    )
    seen_deployment_ids = []

    async def handler(*args, **kwargs):
        deployment_id = kwargs["litellm_params"]["model_info"]["id"]
        seen_deployment_ids.append(deployment_id)
        raise _rate_limit_error()

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
        new_callable=AsyncMock,
        side_effect=handler,
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aresponses(
                model="responses-model",
                input="hi",
                litellm_metadata={"user_api_key_hash": "user-1"},
            )

    assert seen_deployment_ids == ["A", "A"]


def test_retry_deployment_failover_setting_can_be_updated():
    router = _router()
    assert router.get_settings()["enable_retry_deployment_failover"] is False

    router.update_settings(enable_retry_deployment_failover=True)

    assert router.get_settings()["enable_retry_deployment_failover"] is True


def test_retry_deployment_failover_skips_previous_response_id_requests():
    router = _router(enable_retry_deployment_failover=True)
    deployments = router.get_model_list(model_name="responses-model")

    filtered_deployments = router._filter_retry_excluded_deployments(
        healthy_deployments=deployments or [],
        request_kwargs={
            "previous_response_id": "resp_123",
            "_retry_excluded_deployment_ids": ["A"],
            "_retry_last_failed_deployment_id": "A",
        },
    )

    assert [d["model_info"]["id"] for d in filtered_deployments] == ["A", "B"]
