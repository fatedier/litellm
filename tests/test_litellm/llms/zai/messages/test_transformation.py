import json

import httpx
import pytest

import litellm
from litellm.anthropic_interface import messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.zai.messages.transformation import (
    DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE,
    ZAIAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize(
    ("api_base", "expected"),
    (
        (None, DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE),
        ("https://api.z.ai/api/paas/v4", DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE),
        ("https://api.z.ai/api/coding/paas/v4", DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE),
        (
            "https://api.z.ai/api/paas/v4/chat/completions",
            DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE,
        ),
        (
            "https://open.bigmodel.cn/api/paas/v4",
            "https://open.bigmodel.cn/api/anthropic/v1/messages",
        ),
        (
            "https://open.bigmodel.cn/api/anthropic",
            "https://open.bigmodel.cn/api/anthropic/v1/messages",
        ),
        (
            "https://open.bigmodel.cn/api/anthropic/v1",
            "https://open.bigmodel.cn/api/anthropic/v1/messages",
        ),
        (
            "https://open.bigmodel.cn/api/anthropic/v1/messages",
            "https://open.bigmodel.cn/api/anthropic/v1/messages",
        ),
    ),
)
def test_should_map_zai_bases_to_anthropic_messages_endpoint(api_base: str | None, expected: str):
    config = ZAIAnthropicMessagesConfig()

    assert config.get_complete_url(api_base, None, "glm-5.3", {}, {}) == expected


def test_should_prepare_zai_anthropic_messages_headers():
    config = ZAIAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="zai-key",
        api_base="https://open.bigmodel.cn/api/paas/v4",
    )

    assert headers["x-api-key"] == "zai-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://open.bigmodel.cn/api/paas/v4"


def test_should_route_zai_provider_to_native_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.3",
        provider=LlmProviders.ZAI,
    )

    assert isinstance(config, ZAIAnthropicMessagesConfig)


def test_should_preserve_cache_control_in_native_zai_request():
    config = ZAIAnthropicMessagesConfig()
    system = [
        {
            "type": "text",
            "text": "stable prefix",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    request = config.transform_anthropic_messages_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 16, "system": system},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["system"] == system


def test_should_translate_glm_5_3_reasoning_effort_to_adaptive_thinking(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "zai/glm-5.3",
        {"supports_adaptive_thinking": True, "supports_reasoning": True},
    )
    config = ZAIAnthropicMessagesConfig()

    request = config.transform_anthropic_messages_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 32, "reasoning_effort": "low"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "low"}
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("model", ("glm-5.3", "glm-5.3-flash"))
def test_should_normalize_glm_5_3_family_medium_effort_without_model_metadata(model: str):
    config = ZAIAnthropicMessagesConfig()

    request = config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 4096, "reasoning_effort": "medium"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "reasoning_effort" not in request


@pytest.mark.parametrize(
    "optional_params",
    (
        {"reasoning_effort": "medium"},
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
    ),
)
def test_should_normalize_zai_medium_effort_to_high(
    monkeypatch: pytest.MonkeyPatch,
    optional_params: dict,
):
    monkeypatch.setitem(
        litellm.model_cost,
        "zai/glm-5.3",
        {"supports_adaptive_thinking": True, "supports_reasoning": True},
    )
    config = ZAIAnthropicMessagesConfig()

    request = config.transform_anthropic_messages_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 32, **optional_params},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}


def test_should_not_mutate_output_config_when_normalizing_medium_effort(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(
        litellm.model_cost,
        "zai/glm-5.3",
        {"supports_adaptive_thinking": True, "supports_reasoning": True},
    )
    config = ZAIAnthropicMessagesConfig()
    output_config = {
        "effort": "medium",
        "format": {"type": "json_schema"},
    }

    request = config.transform_anthropic_messages_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 32,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert output_config == {
        "effort": "medium",
        "format": {"type": "json_schema"},
    }
    assert request["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema"},
    }
    assert request["output_config"] is not output_config


@pytest.mark.asyncio
async def test_zai_anthropic_messages_uses_native_domestic_endpoint():
    captured_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json={
                "id": "msg_zai_native",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 8,
                },
            },
        )

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    try:
        response = await messages.acreate(
            model="zai/glm-5.3",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
            api_key="zai-key",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            extra_headers={"anthropic-beta": "context-management-2025-06-27"},
            client=client,
        )
    finally:
        await client.close()

    assert captured_request is not None
    assert str(captured_request.url) == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    assert captured_request.headers["x-api-key"] == "zai-key"
    assert captured_request.headers["anthropic-beta"] == "context-management-2025-06-27"
    assert json.loads(captured_request.content) == {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "model": "glm-5.3",
        "stream": False,
    }
    assert response["usage"]["cache_read_input_tokens"] == 8
