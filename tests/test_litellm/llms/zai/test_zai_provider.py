"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token
from litellm.llms.zai.chat.transformation import ZAIChatConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


def test_zai_models_in_model_cost():
    """Test that ZAI models are in the model cost map"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    zai_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
        "zai/glm-4.5-x",
        "zai/glm-4.5-air",
        "zai/glm-4.5-airx",
        "zai/glm-4-32b-0414-128k",
        "zai/glm-4.5-flash",
    ]

    for model in zai_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "zai"


def test_zai_glm46_cost_calculation():
    """Test the cost calculation for glm-4.6"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_flash_model_is_free():
    """Test that glm-4.5-flash has zero cost"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.5-flash"
    info = litellm.model_cost[key]

    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0


def test_glm47_supports_reasoning():
    """Test that GLM-4.7 supports reasoning"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.7"
    assert key in litellm.model_cost, f"Model {key} not found in model_cost"

    info = litellm.model_cost[key]
    assert info["supports_reasoning"] is True


def test_glm47_cost_calculation():
    """Test cost calculation for GLM-4.7"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.7",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.7: $0.6/M input, $2.2/M output (same as GLM-4.6)
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_glm53_supports_reasoning_params():
    supported_params = ZAIChatConfig().get_supported_openai_params(model="zai/glm-5.3-flash")

    assert "thinking" in supported_params
    assert "reasoning_effort" in supported_params


def test_zai_supports_reasoning_params_for_deployment_alias():
    supported_params = ZAIChatConfig().get_supported_openai_params(model="deployment-alias")

    assert "thinking" in supported_params
    assert "reasoning_effort" in supported_params


def test_zai_maps_thinking_to_extra_body_and_preserves_existing_values():
    existing_extra_body = {"foo": "bar"}
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"thinking": {"type": "enabled"}},
        optional_params={"extra_body": existing_extra_body},
        model="zai/glm-5.3-flash",
        drop_params=False,
    )

    assert "thinking" not in mapped_params
    assert mapped_params["extra_body"] == {"foo": "bar", "thinking": {"type": "enabled"}}
    assert existing_extra_body == {"foo": "bar"}


def test_zai_keeps_reasoning_effort_as_top_level_param():
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="zai/glm-5.3-flash",
        drop_params=False,
    )

    assert mapped_params["reasoning_effort"] == "high"


def test_zai_normalizes_structured_reasoning_effort_for_responses_bridge():
    bridge_request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="glm-5.3-flash",
        input="hello",
        responses_api_request={"reasoning": {"effort": "medium", "summary": "detailed"}},
        custom_llm_provider="zai",
    )
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": bridge_request["reasoning_effort"]},
        optional_params={},
        model="zai/glm-5.3-flash",
        drop_params=False,
    )

    assert mapped_params["reasoning_effort"] == "high"


def test_zai_preserves_reasoning_params_for_deployment_alias():
    mapped_params = litellm.get_optional_params(
        model="deployment-alias",
        custom_llm_provider="zai",
        base_model="zai/glm-5.3",
        reasoning_effort="medium",
        thinking={"type": "enabled"},
    )

    assert mapped_params["reasoning_effort"] == "high"
    assert mapped_params["extra_body"]["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize(
    ("requested_effort", "expected_effort"),
    (("minimal", "low"), ("low", "low"), ("medium", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")),
)
def test_zai_normalizes_reasoning_effort(requested_effort: str, expected_effort: str):
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": requested_effort},
        optional_params={},
        model="zai/glm-5.3-flash",
        drop_params=False,
    )

    assert mapped_params["reasoning_effort"] == expected_effort


def test_zai_preserves_none_reasoning_effort():
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="deployment-alias",
        drop_params=False,
    )

    assert mapped_params["reasoning_effort"] == "none"


@pytest.mark.parametrize("reasoning_effort", ("default", "ultra"))
def test_zai_preserves_unmapped_reasoning_effort(reasoning_effort: str):
    mapped_params = ZAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort},
        optional_params={},
        model="deployment-alias",
        drop_params=False,
    )

    assert mapped_params["reasoning_effort"] == reasoning_effort


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_zai_thinking_is_sent_as_top_level_wire_field(respx_mock, zai_response, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    await litellm.acompletion(
        model="zai/glm-5.3-flash",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "enabled"},
        extra_body={"foo": "bar"},
    )

    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert request_body["thinking"] == {"type": "enabled"}
    assert request_body["foo"] == "bar"
    assert "extra_body" not in request_body


@pytest.mark.asyncio
async def test_zai_reasoning_effort_is_sent_as_top_level_wire_field(respx_mock, zai_response, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    await litellm.acompletion(
        model="zai/glm-5.3-flash",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="high",
    )

    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert request_body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_zai_responses_bridge_sends_structured_reasoning_effort_as_string(
    respx_mock, zai_response, monkeypatch
):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    await litellm.aresponses(
        model="zai/glm-5.3",
        input="Hello",
        reasoning={"effort": "medium", "summary": "detailed"},
    )

    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert request_body["reasoning_effort"] == "high"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25
