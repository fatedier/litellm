import httpx
from unittest.mock import patch

from litellm.llms.nova.messages.transformation import NovaMessagesConfig
from litellm.types.router import GenericLiteLLMParams


def test_messages_url_preserves_native_anthropic_endpoint_shape():
    config = NovaMessagesConfig()
    assert config.get_complete_url(
        api_base="https://gateway.example/v1",
        api_key="deployment-key",
        model="logical-model",
        optional_params={},
        litellm_params={},
    ) == "https://gateway.example/v1/messages"
    assert config.get_complete_url(
        api_base="https://gateway.example/v1/messages",
        api_key="deployment-key",
        model="logical-model",
        optional_params={},
        litellm_params={},
    ) == "https://gateway.example/v1/messages"


def test_messages_request_and_response_keep_anthropic_shape():
    config = NovaMessagesConfig()
    messages = [{"role": "user", "content": "hello"}]
    optional_params = {"max_tokens": 32, "system": "Be concise", "stream": False}

    request = config.transform_anthropic_messages_request(
        model="logical-model",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request == {
        "model": "logical-model",
        "messages": messages,
        "max_tokens": 32,
        "system": "Be concise",
        "stream": False,
    }

    response = config.transform_anthropic_messages_response(
        model="logical-model",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "logical-model",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
            },
        ),
        logging_obj=None,
    )
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 5
    assert response["usage"]["cache_read_input_tokens"] == 3
    assert response["usage"]["cache_creation_input_tokens"] == 2


def test_messages_deployment_key_is_used_and_billing_metadata_is_filtered():
    config = NovaMessagesConfig()
    with patch.dict("os.environ", {"NOVA_API_KEY": "env-key"}):
        headers, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="logical-model",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="deployment-key",
            api_base="https://gateway.example/v1",
        )

    assert headers["authorization"] == "Bearer deployment-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert api_base == "https://gateway.example/v1"

    request = config.transform_anthropic_messages_request(
        model="logical-model",
        messages=[{"role": "user", "content": "hello"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 32,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: internal"},
                {"type": "text", "text": "keep this"},
            ],
        },
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )
    assert request["system"] == [{"type": "text", "text": "keep this"}]


def test_messages_environment_fallbacks_are_used_without_deployment_values():
    config = NovaMessagesConfig()
    with patch.dict(
        "os.environ",
        {"NOVA_API_BASE": "https://env.example/v1", "NOVA_API_KEY": "env-key"},
    ):
        headers, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="logical-model",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        request_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="logical-model",
            optional_params={},
            litellm_params={},
        )

    assert headers["authorization"] == "Bearer env-key"
    assert api_base == "https://env.example/v1"
    assert request_url == "https://env.example/v1/messages"
