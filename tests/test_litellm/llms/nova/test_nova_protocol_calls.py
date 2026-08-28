import litellm
import pytest


def test_nova_chat_completion_calls_gateway_endpoint(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://gateway.example/v1/chat/completions").respond(
        json={
            "id": "chatcmpl_nova",
            "object": "chat.completion",
            "created": 1720000000,
            "model": "logical-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    response = litellm.completion(
        model="nova/logical-model",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://gateway.example/v1",
        api_key="deployment-key",
    )

    assert route.called
    assert response.choices[0].message.content == "ok"
    assert route.calls[0].request.headers["authorization"] == "Bearer deployment-key"


def test_nova_responses_calls_gateway_endpoint(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://gateway.example/v1/responses").respond(
        json={
            "id": "resp_nova",
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
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }
    )

    response = litellm.responses(
        model="nova/logical-model",
        input="hello",
        api_base="https://gateway.example/v1",
        api_key="deployment-key",
    )

    assert route.called
    assert response._hidden_params["custom_llm_provider"] == "nova"
    assert response._hidden_params["litellm_model_name"] == "nova/logical-model"
    assert route.calls[0].request.headers["authorization"] == "Bearer deployment-key"


@pytest.mark.asyncio
async def test_nova_messages_calls_gateway_endpoint(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://gateway.example/v1/messages").respond(
        json={
            "id": "msg_nova",
            "type": "message",
            "role": "assistant",
            "model": "logical-model",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    response = await litellm.anthropic.messages.acreate(
        model="nova/logical-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=16,
        api_base="https://gateway.example/v1",
        api_key="deployment-key",
    )

    assert route.called
    assert response["id"] == "msg_nova"
    assert route.calls[0].request.headers["authorization"] == "Bearer deployment-key"
