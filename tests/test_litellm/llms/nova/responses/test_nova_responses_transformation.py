import httpx
from unittest.mock import patch

from litellm.llms.nova.responses.transformation import NovaResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def test_responses_url_and_environment_use_deployment_values_first():
    config = NovaResponsesAPIConfig()
    with patch.dict(
        "os.environ",
        {"NOVA_API_BASE": "https://env.example/v1", "NOVA_API_KEY": "env-key"},
    ):
        assert config.get_complete_url(
            api_base="https://deployment.example/v1",
            litellm_params={},
        ) == "https://deployment.example/v1/responses"
        assert config.get_complete_url(
            api_base="https://deployment.example/v1/responses",
            litellm_params={},
        ) == "https://deployment.example/v1/responses"
        headers = config.validate_environment(
            headers={},
            model="logical-model",
            litellm_params=GenericLiteLLMParams(api_key="deployment-key"),
        )

    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer deployment-key",
    }


def test_responses_response_usage_is_preserved_for_litellm_cost_calculation():
    config = NovaResponsesAPIConfig()
    response = config.transform_response_api_response(
        model="logical-model",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "id": "resp_123",
                "object": "response",
                "created_at": 1720000000,
                "status": "completed",
                "model": "logical-model",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "text": {"format": {"type": "text"}},
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 6,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 16,
                },
            },
        ),
        logging_obj=type("Logging", (), {"post_call": lambda *args, **kwargs: None})(),
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 6
    assert response.usage.total_tokens == 16
    assert response.usage.input_tokens_details.cached_tokens == 4
    assert response.usage.output_tokens_details.reasoning_tokens == 2


def test_responses_streaming_completion_preserves_usage_details():
    config = NovaResponsesAPIConfig()
    event = config.transform_streaming_response(
        model="logical-model",
        parsed_chunk={
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1720000000,
                "status": "completed",
                "model": "logical-model",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "text": {"format": {"type": "text"}},
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 6,
                    "output_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 16,
                },
            },
        },
        logging_obj=None,
    )

    assert event.response is not None
    assert event.response.usage is not None
    assert event.response.usage.input_tokens_details.cached_tokens == 4
    assert event.response.usage.output_tokens_details.reasoning_tokens == 2
