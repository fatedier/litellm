import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
    AIGatewayModeration,
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.proxy.guardrails.guardrail_hooks.aigateway_moderation.aigateway_moderation import (
    _compact_parallel_choice_progress,
    _ReviewProgress,
    StreamChunk,
    _StreamTextExtractor,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
    AIGatewayModerationConfigModel,
    AIGatewayModerationOptionalParams,
)
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIRequest,
)
from litellm.types.utils import GenericGuardrailAPIInputs


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    url: str
    headers: Mapping[str, str]
    payload: GenericGuardrailAPIRequest


class _RecordingHandler:
    def __init__(
        self,
        responses: Sequence[Mapping[str, object] | Exception] = ({"action": "NONE"},),
        gate: asyncio.Event | None = None,
    ) -> None:
        self.responses = tuple(responses)
        self.gate = gate
        self.started = asyncio.Event()
        self.calls: list[_RecordedCall] = []

    async def post(
        self,
        url: str,
        json: dict[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        payload = GenericGuardrailAPIRequest.model_validate(json or {})
        self.calls.append(_RecordedCall(url, headers or {}, payload))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        response_spec = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response_spec, Exception):
            raise response_spec
        request = httpx.Request("POST", url)
        return httpx.Response(200, json=dict(response_spec), request=request)


def _request_data(request_route: str = "/chat/completions") -> dict[str, object]:
    return {
        "model": "test-model",
        "litellm_call_id": "call-1",
        "metadata": {
            "guardrails": ["aigateway"],
            "user_api_key_request_route": request_route,
        },
    }


def _user(request_route: str = "/chat/completions") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="test", request_route=request_route)


def _guardrail(
    handler: _RecordingHandler,
    *,
    decision_timeout_seconds: float = 5.0,
    streaming_review_chunk_size: int = 256,
    streaming_review_context_size: int = 256,
    fail_on_error: bool = True,
) -> AIGatewayModeration:
    return AIGatewayModeration(
        api_base="http://adapter:9800",
        api_key="adapter-key",
        guardrail_name="aigateway",
        event_hook=GuardrailEventHooks.post_call,
        async_handler=cast(AsyncHTTPHandler, handler),
        decision_timeout_seconds=decision_timeout_seconds,
        streaming_review_chunk_size=streaming_review_chunk_size,
        streaming_review_context_size=streaming_review_context_size,
        fail_on_error=fail_on_error,
    )


async def _items(items: Sequence[StreamChunk]) -> AsyncIterator[StreamChunk]:
    for item in items:
        yield item


async def _collect(
    guardrail: AIGatewayModeration,
    source: Sequence[StreamChunk],
    *,
    request_route: str = "/chat/completions",
    request_data: dict[str, object] | None = None,
) -> list[StreamChunk]:
    return [
        item
        async for item in guardrail.async_post_call_streaming_iterator_hook(
            _user(request_route),
            _items(source),
            request_data or _request_data(request_route),
        )
    ]


def _chat_text(text: str) -> Mapping[str, object]:
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
            }
        ]
    }


def _parallel_chat_text(index: int, text: str, finish_reason: str | None = None) -> Mapping[str, object]:
    return {
        "choices": [
            {
                "index": index,
                "delta": {"content": text},
                "finish_reason": finish_reason,
            }
        ]
    }


def _chat_tool(arguments: str, choice_index: int = 0) -> Mapping[str, object]:
    return {
        "choices": [
            {
                "index": choice_index,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": arguments,
                            },
                        }
                    ]
                },
            }
        ]
    }


def _response_text(
    text: str,
    sequence_number: int,
    *,
    output_index: int = 0,
    item_id: str = "message-1",
) -> Mapping[str, object]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": 0,
        "item_id": item_id,
        "delta": text,
    }


def _response_text_done(
    text: str,
    sequence_number: int,
    *,
    output_index: int = 0,
    item_id: str = "message-1",
) -> Mapping[str, object]:
    return {
        "type": "response.output_text.done",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "content_index": 0,
        "item_id": item_id,
        "text": text,
    }


def _response_completed(
    texts: Sequence[tuple[str, str]],
    sequence_number: int,
) -> Mapping[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": {
            "output": [
                {
                    "id": item_id,
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
                for item_id, text in texts
            ]
        },
    }


def _response_tool(
    arguments: str,
    sequence_number: int,
    *,
    output_index: int = 0,
    item_id: str = "call-1",
) -> Mapping[str, object]:
    return {
        "type": "response.function_call_arguments.delta",
        "sequence_number": sequence_number,
        "output_index": output_index,
        "item_id": item_id,
        "delta": arguments,
    }


def _anthropic_event(event_type: str, payload: Mapping[str, object]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _reviewed_texts(handler: _RecordingHandler) -> list[str]:
    return [call.payload.texts[0] for call in handler.calls if call.payload.texts is not None and call.payload.texts]


def _reviewed_images(handler: _RecordingHandler) -> list[list[str]]:
    return [call.payload.images for call in handler.calls if call.payload.images]


@pytest.mark.asyncio
async def test_stream_waits_for_review_before_releasing_buffered_chunks() -> None:
    gate = asyncio.Event()
    handler = _RecordingHandler(gate=gate)
    guardrail = _guardrail(handler)
    source = [_chat_text("a" * 128), _chat_text("b" * 128)]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        _request_data(),
    )
    first_output = asyncio.create_task(anext(iterator))

    await asyncio.wait_for(handler.started.wait(), timeout=1)
    assert not first_output.done()

    gate.set()
    assert await asyncio.wait_for(first_output, timeout=1) == source[0]
    assert [item async for item in iterator] == source[1:]


@pytest.mark.asyncio
async def test_chat_stream_waits_for_image_review_before_release() -> None:
    gate = asyncio.Event()
    handler = _RecordingHandler(gate=gate)
    guardrail = _guardrail(handler)
    image = "data:image/png;base64,aGVsbG8="
    source: list[StreamChunk] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"images": [{"type": "image_url", "image_url": {"url": image}}]},
                }
            ]
        }
    ]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        _request_data(),
    )
    first_output = asyncio.create_task(anext(iterator))

    await asyncio.wait_for(handler.started.wait(), timeout=1)
    assert not first_output.done()
    assert _reviewed_images(handler) == [[image]]

    gate.set()
    assert await asyncio.wait_for(first_output, timeout=1) == source[0]
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_parallel_choices_are_reviewed_and_released_before_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _parallel_chat_text(0, "a" * 256),
        _parallel_chat_text(1, "b" * 256),
    ]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        {**_request_data(), "n": 2},
    )

    assert await anext(iterator) == source[0]
    assert _reviewed_texts(handler) == ["a" * 256]
    assert await anext(iterator) == source[1]
    assert _reviewed_texts(handler) == ["a" * 256, "b" * 256]
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_parallel_interleaved_choices_use_independent_review_progress() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _parallel_chat_text(0, "a" * 128),
        _parallel_chat_text(1, "b" * 256),
        _parallel_chat_text(0, "c" * 128),
    ]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        {**_request_data(), "n": 2},
    )

    assert await anext(iterator) == source[0]
    assert _reviewed_texts(handler) == ["b" * 256, "a" * 128]
    assert [item async for item in iterator] == source[1:]
    assert _reviewed_texts(handler) == ["b" * 256, "a" * 128, "a" * 128 + "c" * 128]


@pytest.mark.asyncio
async def test_parallel_reviewable_choice_flushes_another_choice_tail_before_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    stalled_tail = "a" * 100
    reviewable_text = "b" * 256
    source = [
        _parallel_chat_text(0, stalled_tail),
        *(_parallel_chat_text(1, "b" * 64) for _ in range(4)),
    ]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        {**_request_data(), "n": 2},
    )

    assert await anext(iterator) == source[0]
    assert _reviewed_texts(handler) == [reviewable_text, stalled_tail]
    assert [item async for item in iterator] == source[1:]


@pytest.mark.asyncio
async def test_parallel_inactive_choice_preserves_context_across_another_choice_release() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    choice_zero_prefix = "a" * 253 + "bad"
    choice_zero_suffix = "word" + "c" * 252
    choice_one_text = "b" * 256
    source = [
        _parallel_chat_text(0, choice_zero_prefix),
        _parallel_chat_text(1, choice_one_text),
        _parallel_chat_text(0, choice_zero_suffix),
    ]

    assert await _collect(guardrail, source, request_data={**_request_data(), "n": 2}) == source
    assert _reviewed_texts(handler) == [
        choice_zero_prefix,
        choice_one_text,
        choice_zero_prefix + choice_zero_suffix,
    ]


def test_parallel_choice_progress_is_compacted_after_release() -> None:
    context_a = "a" * 256
    context_b = "b" * 256
    progress = (
        (0, _ReviewProgress(approved_text="a" * 1024, previous_context=context_a)),
        (1, _ReviewProgress(approved_text="b" * 2048, previous_context=context_b)),
    )

    assert _compact_parallel_choice_progress(progress) == (
        (0, _ReviewProgress(previous_context=context_a)),
        (1, _ReviewProgress(previous_context=context_b)),
    )


@pytest.mark.asyncio
async def test_parallel_plain_string_fails_closed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _collect(
            guardrail,
            ["a" * 300],
            request_data={**_request_data(), "n": 2},
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "Streaming response returned text without a choice index for parallel chat choices"
    assert handler.calls == []


@pytest.mark.asyncio
async def test_parallel_plain_string_respects_fail_open() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler, fail_on_error=False)
    source = ["a" * 300]

    assert await _collect(guardrail, source, request_data={**_request_data(), "n": 2}) == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_parallel_tool_call_discards_all_pre_tool_choice_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _parallel_chat_text(0, "a" * 100),
        _chat_tool("tool-input", choice_index=1),
        _parallel_chat_text(0, "b" * 256),
    ]

    assert await _collect(guardrail, source, request_data={**_request_data(), "n": 2}) == source
    assert _reviewed_texts(handler) == ["b" * 256]


@pytest.mark.asyncio
async def test_parallel_choice_finish_reviews_tail_before_stream_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_parallel_chat_text(0, "a" * 100, finish_reason="stop")]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        {**_request_data(), "n": 2},
    )

    assert await anext(iterator) == source[0]
    assert _reviewed_texts(handler) == ["a" * 100]


@pytest.mark.asyncio
async def test_parallel_choice_tails_are_reviewed_at_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _parallel_chat_text(0, "a" * 100),
        _parallel_chat_text(1, "b" * 100),
    ]

    assert (
        await _collect(
            guardrail,
            source,
            request_data={**_request_data(), "n": 2},
        )
        == source
    )
    assert _reviewed_texts(handler) == ["a" * 100, "b" * 100]


@pytest.mark.asyncio
async def test_stream_preserves_many_fragmented_pending_chunks() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    payload = f"data: {json.dumps(_chat_text('a' * 4096))}\n\n".encode()
    source = [payload[index : index + 1] for index in range(len(payload))]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 4096]


@pytest.mark.asyncio
async def test_many_small_chat_deltas_keep_only_rolling_review_context() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_chat_text("a") for _ in range(4096)]

    assert await _collect(guardrail, source) == source
    assert len(handler.calls) == 16
    assert len(_reviewed_texts(handler)[0]) == 256
    assert all(len(text) == 512 for text in _reviewed_texts(handler)[1:])


def test_responses_snapshots_are_only_materialized_for_text_boundaries() -> None:
    extractor = _StreamTextExtractor()
    snapshots = tuple(
        extractor.extract(_response_text("a", sequence_number)).frames[0].response_text_snapshot
        for sequence_number in range(64)
    )
    first_tool_frame = extractor.extract(_response_tool("tool-input", 64)).frames[0]
    repeated_tool_frames = tuple(
        extractor.extract(_response_tool("tool-input", sequence_number)).frames[0] for sequence_number in range(65, 69)
    )
    next_tool_frame = extractor.extract(
        _response_tool("other-tool-input", 69, output_index=1, item_id="call-2")
    ).frames[0]

    assert snapshots == (None,) * 64
    assert first_tool_frame.has_tool_call
    assert first_tool_frame.response_text_snapshot == "a" * 64
    assert all(not frame.has_tool_call and frame.response_text_snapshot is None for frame in repeated_tool_frames)
    assert next_tool_frame.has_tool_call
    assert next_tool_frame.response_text_snapshot == "a" * 64


def test_responses_tool_event_requires_an_item_identity() -> None:
    extractor = _StreamTextExtractor()

    with pytest.raises(GuardrailRaisedException, match="did not include an item identity"):
        extractor.extract(
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 0,
                "output_index": 0,
                "delta": "tool-input",
            }
        )


@pytest.mark.asyncio
async def test_many_small_responses_deltas_keep_only_rolling_review_context() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        *(_response_text("a", sequence_number) for sequence_number in range(4096)),
        _response_text_done("a" * 4096, 4096),
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert len(handler.calls) == 16
    assert len(_reviewed_texts(handler)[0]) == 256
    assert all(len(text) == 512 for text in _reviewed_texts(handler)[1:])


@pytest.mark.asyncio
async def test_stream_reviews_with_previous_batch_context_and_reviews_tail_at_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _chat_text("a" * 128),
        _chat_text("b" * 128),
        _chat_text("c" * 256),
        _chat_text("d" * 10),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == [
        "a" * 128 + "b" * 128,
        "a" * 128 + "b" * 128 + "c" * 256,
        "c" * 256 + "d" * 10,
    ]
    assert all(call.url == "http://adapter:9800/beta/litellm_basic_guardrail_api" for call in handler.calls)
    assert all(
        call.payload.additional_provider_specific_params == {"api_type": "chat_completions"} for call in handler.calls
    )


@pytest.mark.asyncio
async def test_stream_reviews_oversized_pending_batches_before_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(
        handler,
        streaming_review_chunk_size=128,
        streaming_review_context_size=128,
    )
    source = [
        _chat_text("a" * 100),
        _chat_text("b" * 50),
        _chat_text("c" * 80),
        _chat_text("d" * 60),
        _chat_text("e" * 10),
    ]
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items(source),
        _request_data(),
    )

    assert await anext(iterator) == source[0]
    assert _reviewed_texts(handler) == ["a" * 100 + "b" * 50]
    assert await anext(iterator) == source[1]
    assert await anext(iterator) == source[2]
    assert _reviewed_texts(handler) == [
        "a" * 100 + "b" * 50,
        "a" * 78 + "b" * 50 + "c" * 80 + "d" * 60,
    ]
    assert await anext(iterator) == source[3]
    assert await anext(iterator) == source[4]
    assert _reviewed_texts(handler) == [
        "a" * 100 + "b" * 50,
        "a" * 78 + "b" * 50 + "c" * 80 + "d" * 60,
        "c" * 68 + "d" * 60 + "e" * 10,
    ]
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_stream_review_chunk_size_is_configurable() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(
        handler,
        streaming_review_chunk_size=4,
        streaming_review_context_size=2,
    )
    source = [
        _chat_text("a" * 3),
        _chat_text("b"),
        _chat_text("c" * 4),
        _chat_text("d" * 2),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == [
        "a" * 3 + "b",
        "ab" + "c" * 4,
        "c" * 2 + "d" * 2,
    ]


@pytest.mark.asyncio
async def test_stream_review_context_can_be_disabled() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(
        handler,
        streaming_review_chunk_size=4,
        streaming_review_context_size=0,
    )
    source = [
        _chat_text("a" * 4),
        _chat_text("b" * 4),
        _chat_text("c" * 2),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 4, "b" * 4, "c" * 2]


@pytest.mark.parametrize(
    ("request_route", "source", "expected_api_type"),
    (
        ("/chat/completions", (_chat_text("a" * 256),), "chat_completions"),
        (
            "/responses",
            (_response_text("a" * 256, 0), _response_text_done("a" * 256, 1)),
            "responses",
        ),
        (
            "/messages",
            (
                _anthropic_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "a" * 256},
                    },
                ),
            ),
            "anthropic_messages",
        ),
    ),
)
@pytest.mark.asyncio
async def test_stream_api_type_uses_authenticated_route_when_metadata_route_is_missing(
    request_route: str,
    source: tuple[StreamChunk, ...],
    expected_api_type: str,
) -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    request_data: dict[str, object] = {
        "model": "test-model",
        "litellm_call_id": "call-1",
        "metadata": {"guardrails": ["aigateway"]},
    }

    assert (
        tuple(
            await _collect(
                guardrail,
                source,
                request_route=request_route,
                request_data=request_data,
            )
        )
        == source
    )
    assert handler.calls[0].payload.additional_provider_specific_params == {"api_type": expected_api_type}


@pytest.mark.asyncio
async def test_stream_reviews_forward_logging_context() -> None:
    import time

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    request_data = _request_data()
    request_data["litellm_logging_obj"] = LiteLLMLoggingObj(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="call-123",
        function_id="function-123",
        litellm_trace_id="trace-456",
    )
    source = [_chat_text("a" * 256)]

    assert await _collect(guardrail, source, request_data=request_data) == source
    assert handler.calls[0].payload.litellm_call_id == "call-123"
    assert handler.calls[0].payload.litellm_trace_id == "trace-456"


@pytest.mark.asyncio
async def test_large_delta_is_reviewed_as_one_pending_batch() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text = "a" * 256 + "b" * 256 + "c" * 88
    source = [_chat_text(text)]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == [text]


@pytest.mark.asyncio
async def test_each_later_review_keeps_only_the_immediately_previous_batch() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _chat_text("a" * 256),
        _chat_text("b" * 256),
        _chat_text("c" * 256),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == [
        "a" * 256,
        "a" * 256 + "b" * 256,
        "b" * 256 + "c" * 256,
    ]


@pytest.mark.asyncio
async def test_tool_call_releases_short_prefix_and_reviews_later_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _chat_text("a" * 100),
        _chat_tool("secret-tool-input"),
        _chat_text("b" * 156),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["b" * 156]


@pytest.mark.asyncio
async def test_short_text_before_tool_call_is_released_without_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_chat_text("a" * 255), _chat_tool("tool-input")]

    assert await _collect(guardrail, source) == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_completed_batch_before_later_tool_call_remains_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_chat_text("a" * 256), _chat_tool("tool-input"), _chat_text("b" * 256)]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 256, "b" * 256]


@pytest.mark.asyncio
async def test_empty_chat_tool_calls_do_not_bypass_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    empty_tool_calls = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "a" * 100, "tool_calls": []},
                }
            ]
        }
    )
    remaining_text = json.dumps(_chat_text("b" * 156))
    source = [f"data: {empty_tool_calls}\n\n".encode(), f"data: {remaining_text}\n\n".encode()]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 100 + "b" * 156]


@pytest.mark.asyncio
async def test_coalesced_text_and_tool_sse_frames_bypass_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        (f"data: {json.dumps(_chat_text('a' * 256))}\n\ndata: {json.dumps(_chat_tool('tool-input'))}\n\n").encode()
    ]

    assert await _collect(guardrail, source) == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_coalesced_tool_and_text_sse_frames_review_post_tool_text() -> None:
    handler = _RecordingHandler(({"action": "BLOCKED", "blocked_reason": "blocked"},))
    guardrail = _guardrail(handler)
    source = [
        (f"data: {json.dumps(_chat_tool('tool-input'))}\n\ndata: {json.dumps(_chat_text('a' * 256))}\n\n").encode()
    ]

    with pytest.raises(GuardrailRaisedException):
        await _collect(guardrail, source)

    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_coalesced_post_tool_text_combines_with_later_chunk() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        (f"data: {json.dumps(_chat_tool('tool-input'))}\n\ndata: {json.dumps(_chat_text('a' * 100))}\n\n").encode(),
        f"data: {json.dumps(_chat_text('b' * 156))}\n\n".encode(),
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 100 + "b" * 156]


@pytest.mark.asyncio
async def test_coalesced_post_tool_tail_is_reviewed_at_eof() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        (f"data: {json.dumps(_chat_tool('tool-input'))}\n\ndata: {json.dumps(_chat_text('a' * 100))}\n\n").encode()
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 100]


@pytest.mark.asyncio
async def test_coalesced_text_tool_text_reviews_only_final_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        (
            f"data: {json.dumps(_chat_text('a' * 256))}\n\n"
            f"data: {json.dumps(_chat_tool('tool-input'))}\n\n"
            f"data: {json.dumps(_chat_text('b' * 256))}\n\n"
        ).encode()
    ]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["b" * 256]


@pytest.mark.asyncio
async def test_coalesced_responses_tool_and_text_reviews_post_tool_text() -> None:
    handler = _RecordingHandler(({"action": "BLOCKED", "blocked_reason": "blocked"},))
    guardrail = _guardrail(handler)
    source = [
        (
            f"data: {json.dumps(_response_tool('tool-input', 0))}\n\n"
            f"data: {json.dumps(_response_text('a' * 256, 1))}\n\n"
        ).encode()
    ]

    with pytest.raises(GuardrailRaisedException):
        await _collect(guardrail, source, request_route="/responses")

    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_tool_only_chat_stream_passes_through_without_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_chat_tool("first"), _chat_tool("second")]

    assert await _collect(guardrail, source) == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_responses_tool_calls_release_prefixes_and_review_later_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _response_tool("tool-secret", 0),
        _response_text("a" * 128, 1),
        _response_tool("more-tool-secret", 2, output_index=1, item_id="call-2"),
        _response_text("b" * 128, 3),
        _response_text_done("a" * 128 + "b" * 128, 4),
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["b" * 128]


@pytest.mark.asyncio
async def test_responses_same_tool_fragments_preserve_interleaved_visible_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text = "a" * 128 + "b" * 128
    source = [
        _response_tool("tool-secret", 0),
        _response_text("a" * 128, 1),
        _response_tool("more-tool-secret", 2),
        _response_text("b" * 128, 3),
        _response_text_done(text, 4),
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == [text]


@pytest.mark.asyncio
async def test_responses_tool_only_stream_passes_through_without_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_response_tool("tool-secret", 0)]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert handler.calls == []


@pytest.mark.parametrize(
    "item_type",
    (
        "apply_patch_call",
        "local_shell_call",
        "tool_search_call",
    ),
)
@pytest.mark.asyncio
async def test_responses_supported_tool_items_review_later_text(item_type: str) -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    tool_item: Mapping[str, object] = {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"id": "tool-1", "type": item_type},
    }
    source = [tool_item, _response_text("a" * 256, 1), _response_text_done("a" * 256, 2)]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.parametrize(
    "event_type",
    (
        "response.output_item.added",
        "response.output_item.done",
    ),
)
@pytest.mark.asyncio
async def test_responses_image_generation_item_does_not_bypass_text_review(event_type: str) -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    image_item: Mapping[str, object] = {
        "type": event_type,
        "sequence_number": 1,
        "output_index": 1,
        "item": {"id": "image-1", "type": "image_generation_call"},
    }
    source = [_response_text("a" * 100, 0), image_item, _response_text_done("a" * 100, 2)]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["a" * 100]


@pytest.mark.asyncio
async def test_responses_completed_image_is_reviewed_before_release() -> None:
    image = "aGVsbG8="
    handler = _RecordingHandler(({"action": "BLOCKED", "blocked_reason": "blocked image"},))
    guardrail = _guardrail(handler)
    source: list[StreamChunk] = [
        {
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "output": [
                    {
                        "id": "image-1",
                        "type": "image_generation_call",
                        "result": image,
                    }
                ]
            },
        }
    ]

    with pytest.raises(GuardrailRaisedException, match="blocked image"):
        await _collect(guardrail, source, request_route="/responses")

    assert _reviewed_images(handler) == [[image]]


@pytest.mark.parametrize(
    ("event_type", "image_field"),
    (
        ("response.image_generation_call.partial_image", "partial_image_b64"),
        ("image_generation.partial_image", "b64_json"),
    ),
)
@pytest.mark.asyncio
async def test_responses_partial_image_is_reviewed_before_release(event_type: str, image_field: str) -> None:
    image = "aGVsbG8="
    handler = _RecordingHandler(({"action": "BLOCKED", "blocked_reason": "blocked image"},))
    guardrail = _guardrail(handler)
    source: list[StreamChunk] = [
        {
            "type": event_type,
            "sequence_number": 0,
            "partial_image_index": 0,
            image_field: image,
        }
    ]

    with pytest.raises(GuardrailRaisedException, match="blocked image"):
        await _collect(guardrail, source, request_route="/responses")

    assert _reviewed_images(handler) == [[image]]


@pytest.mark.asyncio
async def test_responses_completed_mixed_text_and_image_reviews_both() -> None:
    text = "visible final answer"
    image = "aGVsbG8="
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source: list[StreamChunk] = [
        {
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "output": [
                    {
                        "id": "message-1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    },
                    {
                        "id": "image-1",
                        "type": "image_generation_call",
                        "result": image,
                    },
                ]
            },
        }
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == [text]
    assert _reviewed_images(handler) == [[image]]


@pytest.mark.asyncio
async def test_responses_image_is_reviewed_once_across_done_and_completed_events() -> None:
    image = "aGVsbG8="
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    image_item = {
        "id": "image-1",
        "type": "image_generation_call",
        "result": image,
    }
    source: list[StreamChunk] = [
        {
            "type": "response.output_item.done",
            "sequence_number": 0,
            "output_index": 0,
            "item": image_item,
        },
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {"output": [image_item]},
        },
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_images(handler) == [[image]]


@pytest.mark.asyncio
async def test_responses_terminal_aggregate_reviews_message_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    payload = json.dumps(
        {
            "type": "response.completed",
            "sequence_number": 0,
            "response": {
                "output": [
                    {
                        "id": "call-1",
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": "tool-secret",
                    },
                    {
                        "id": "message-1",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "a" * 256}],
                    },
                ]
            },
        }
    )
    source = [f"event: response.completed\ndata: {payload}\n\n".encode()]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_responses_final_text_without_tool_call_is_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text = "a" * 128 + "b" * 128
    source = [_response_text("a" * 128, 0), _response_text("b" * 128, 1), _response_text_done(text, 2)]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == [text]
    assert handler.calls[0].payload.additional_provider_specific_params == {"api_type": "responses"}


@pytest.mark.asyncio
async def test_responses_stream_ignores_parallel_chat_choice_count() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_response_text("a" * 256, 0), _response_text_done("a" * 256, 1)]

    assert (
        await _collect(
            guardrail,
            source,
            request_route="/responses",
            request_data={**_request_data("/responses"), "n": 2},
        )
        == source
    )
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_responses_plain_string_chunk_is_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = ["a" * 300]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["a" * 300]


@pytest.mark.asyncio
async def test_responses_plain_string_chunks_accumulate() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = ["a" * 100, "b" * 156]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == ["a" * 100 + "b" * 156]


@pytest.mark.asyncio
async def test_responses_done_event_does_not_duplicate_reviewed_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text = "a" * 256
    source: list[StreamChunk] = [
        _response_text(text, 0),
        _response_text_done(text, 1),
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == [text]


@pytest.mark.asyncio
async def test_responses_rejects_active_part_omitted_from_terminal_response() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text_a = "a" * 10
    text_b = "b" * 10
    source = [
        _response_text(text_a, 0, output_index=0, item_id="message-a"),
        _response_text(text_b, 1, output_index=1, item_id="message-b"),
        _response_text_done(text_a, 2, output_index=0, item_id="message-a"),
        _response_completed((("message-a", text_a),), 3),
    ]

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _collect(guardrail, source, request_route="/responses")

    assert exc_info.value.message == "Streaming response ended before a Responses API text part was finalized"
    assert handler.calls == []


@pytest.mark.asyncio
async def test_responses_terminal_response_finalizes_every_included_active_part() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    text_a = "a" * 10
    text_b = "b" * 10
    source = [
        _response_text(text_a, 0, output_index=0, item_id="message-a"),
        _response_text(text_b, 1, output_index=1, item_id="message-b"),
        _response_text_done(text_a, 2, output_index=0, item_id="message-a"),
        _response_completed((("message-a", text_a), ("message-b", text_b)), 3),
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert _reviewed_texts(handler) == [text_a + text_b]


@pytest.mark.asyncio
async def test_responses_reasoning_and_audio_are_not_reviewed_as_final_output_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source: list[StreamChunk] = [
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 0,
            "output_index": 0,
            "summary_index": 0,
            "item_id": "reasoning-1",
            "delta": "private reasoning",
        },
        {
            "type": "response.audio.transcript.delta",
            "sequence_number": 1,
            "delta": "audio transcript",
        },
    ]

    assert await _collect(guardrail, source, request_route="/responses") == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_chat_audio_transcript_is_not_reviewed_as_final_output_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source: list[StreamChunk] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"audio": {"transcript": "a" * 256}},
                }
            ]
        }
    ]

    assert await _collect(guardrail, source) == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_anthropic_tool_call_reviews_later_text() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"secret":"value"}'},
            },
        ),
        _anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "a" * 256},
            },
        ),
    ]

    assert await _collect(guardrail, source, request_route="/messages") == source
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_anthropic_final_text_without_tool_call_is_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [
        _anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "a" * 256},
            },
        )
    ]

    assert await _collect(guardrail, source, request_route="/messages") == source
    assert _reviewed_texts(handler) == ["a" * 256]
    assert handler.calls[0].payload.additional_provider_specific_params == {"api_type": "anthropic_messages"}


@pytest.mark.asyncio
async def test_split_sse_text_is_held_until_frame_and_review_complete() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    payload = json.dumps({"choices": [{"index": 0, "delta": {"content": "a" * 256}}]})
    source = [f"data: {payload[:80]}".encode(), f"{payload[80:]}\n\n".encode()]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_block_before_first_release_yields_no_content() -> None:
    handler = _RecordingHandler(({"action": "BLOCKED", "blocked_reason": "blocked"},))
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _collect(guardrail, [_chat_text("a" * 256)])

    assert exc_info.value.guardrail_name == "aigateway"
    assert exc_info.value.message == "blocked"


@pytest.mark.asyncio
async def test_later_block_keeps_unapproved_batch_and_emits_stream_error() -> None:
    handler = _RecordingHandler(
        (
            {"action": "NONE"},
            {"action": "BLOCKED", "blocked_reason": "blocked"},
        )
    )
    guardrail = _guardrail(handler)
    first_chunk = _chat_text("a" * 256)
    second_chunk = _chat_text("b" * 256)
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user(),
        _items([first_chunk, second_chunk]),
        _request_data(),
    )

    assert await anext(iterator) == first_chunk
    remainder = [item async for item in iterator]

    assert len(remainder) == 1
    assert isinstance(remainder[0], bytes)
    assert b'"type":"guardrail_violation"' in remainder[0]
    assert second_chunk not in remainder
    assert _reviewed_texts(handler) == ["a" * 256, "a" * 256 + "b" * 256]


@pytest.mark.asyncio
async def test_responses_block_error_uses_last_released_sequence_number() -> None:
    handler = _RecordingHandler(
        (
            {"action": "NONE"},
            {"action": "BLOCKED", "blocked_reason": "blocked"},
        )
    )
    guardrail = _guardrail(handler)
    first_chunk = _response_text("a" * 256, 4)
    second_chunk = _response_text("b" * 256, 5)
    iterator = guardrail.async_post_call_streaming_iterator_hook(
        _user("/responses"),
        _items([first_chunk, second_chunk]),
        _request_data("/responses"),
    )

    assert await anext(iterator) == first_chunk
    remainder = [item async for item in iterator]
    error_payload = TypeAdapter(dict[str, object]).validate_json(
        cast(bytes, remainder[0]).decode().split("data: ", 1)[1]
    )

    assert error_payload["sequence_number"] == 5
    assert second_chunk not in remainder


@pytest.mark.asyncio
async def test_fail_open_releases_buffered_content_on_adapter_error() -> None:
    handler = _RecordingHandler((httpx.ConnectError("offline"),))
    guardrail = _guardrail(handler, fail_on_error=False)
    source = [_chat_text("a" * 256), _chat_text("tail")]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 256, "a" * 256 + "tail"]


@pytest.mark.asyncio
async def test_fail_open_releases_buffered_content_on_review_timeout() -> None:
    gate = asyncio.Event()
    handler = _RecordingHandler(gate=gate)
    guardrail = _guardrail(handler, decision_timeout_seconds=0.01, fail_on_error=False)
    source = [_chat_text("a" * 256)]

    assert await _collect(guardrail, source) == source
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.asyncio
async def test_guardrail_intervention_blocks_streaming_response() -> None:
    handler = _RecordingHandler(
        ({"action": "GUARDRAIL_INTERVENED", "blocked_reason": "custom block", "texts": ["redacted"]},)
    )
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _collect(guardrail, [_chat_text("a" * 256)])

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "custom block"
    assert _reviewed_texts(handler) == ["a" * 256]


@pytest.mark.parametrize("response", ({}, {"action": "UNKNOWN_ACTION"}))
@pytest.mark.asyncio
async def test_invalid_streaming_action_fails_closed(response: Mapping[str, object]) -> None:
    handler = _RecordingHandler((response,))
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await _collect(guardrail, [_chat_text("a" * 256)])

    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "Response moderation failed"


@pytest.mark.asyncio
async def test_invalid_streaming_action_respects_fail_open() -> None:
    handler = _RecordingHandler(({"action": "UNKNOWN_ACTION"},))
    guardrail = _guardrail(handler, fail_on_error=False)
    source = [_chat_text("a" * 256)]

    assert await _collect(guardrail, source) == source


@pytest.mark.asyncio
async def test_unsupported_stream_route_passes_through_without_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    source = [_chat_text("a" * 256)]

    assert await _collect(guardrail, source, request_route="/embeddings") == source
    assert handler.calls == []


@pytest.mark.asyncio
async def test_non_streaming_mixed_response_reviews_text_and_preserves_tool_calls() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    inputs = cast(
        GenericGuardrailAPIInputs,
        {
            "texts": ["final answer"],
            "tool_calls": [tool_call],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=_request_data("/responses"),
        input_type="response",
    )

    assert result.get("tool_calls") == [tool_call]
    assert handler.calls[0].payload.texts == ["final answer"]


@pytest.mark.asyncio
async def test_guardrail_intervention_blocks_non_streaming_image_response() -> None:
    handler = _RecordingHandler(({"action": "GUARDRAIL_INTERVENED", "images": ["replacement-image"]},))
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"images": ["original-image"]},
            request_data=_request_data("/responses"),
            input_type="response",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Content violates policy"
    assert handler.calls[0].payload.images == ["original-image"]


@pytest.mark.asyncio
async def test_guardrail_intervention_blocks_non_streaming_image_request() -> None:
    handler = _RecordingHandler(
        ({"action": "GUARDRAIL_INTERVENED", "blocked_reason": "custom block", "images": ["replacement-image"]},)
    )
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"images": ["original-image"]},
            request_data=_request_data("/responses"),
            input_type="request",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "custom block"
    assert handler.calls[0].payload.images == ["original-image"]


@pytest.mark.asyncio
async def test_guardrail_intervention_blocks_text_only_request() -> None:
    handler = _RecordingHandler(({"action": "GUARDRAIL_INTERVENED", "texts": ["redacted"]},))
    guardrail = _guardrail(handler)

    with pytest.raises(GuardrailRaisedException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["original"]},
            request_data=_request_data("/responses"),
            input_type="request",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Content violates policy"


@pytest.mark.parametrize("response", ({}, {"action": "UNKNOWN_ACTION"}))
@pytest.mark.asyncio
async def test_invalid_text_only_request_action_fails_closed(response: Mapping[str, object]) -> None:
    handler = _RecordingHandler((response,))
    guardrail = _guardrail(handler)

    with pytest.raises(Exception, match="AIGateway moderation returned an invalid action"):
        await guardrail.apply_guardrail(
            inputs={"texts": ["original"]},
            request_data=_request_data("/responses"),
            input_type="request",
        )


@pytest.mark.asyncio
async def test_non_streaming_final_text_response_is_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    inputs: GenericGuardrailAPIInputs = {"texts": ["final answer"]}

    assert (
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=_request_data("/responses"),
            input_type="response",
        )
        == inputs
    )
    assert handler.calls[0].payload.texts == ["final answer"]
    assert handler.calls[0].payload.additional_provider_specific_params == {"api_type": "responses"}


@pytest.mark.asyncio
async def test_non_streaming_api_type_ignores_client_marker() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    inputs: GenericGuardrailAPIInputs = {"texts": ["final answer"]}
    request_data = {
        **_request_data("/chat/completions"),
        "_aigateway_api_type": "embeddings",
        "body": {"_aigateway_api_type": "embeddings"},
    }

    assert (
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
        == inputs
    )
    assert handler.calls[0].payload.additional_provider_specific_params == {"api_type": "chat_completions"}


@pytest.mark.asyncio
async def test_non_streaming_tool_only_response_skips_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    inputs = cast(
        GenericGuardrailAPIInputs,
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        },
    )

    assert (
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=_request_data("/responses"),
            input_type="response",
        )
        == inputs
    )
    assert handler.calls == []


@pytest.mark.asyncio
async def test_non_streaming_image_only_response_is_reviewed() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    inputs: GenericGuardrailAPIInputs = {"images": ["data:image/png;base64,aGVsbG8="]}

    assert (
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=_request_data("/responses"),
            input_type="response",
        )
        == inputs
    )
    assert handler.calls[0].payload.images == inputs["images"]
    assert handler.calls[0].payload.texts == []


@pytest.mark.asyncio
async def test_non_streaming_mixed_response_preserves_images_for_review() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)
    inputs: GenericGuardrailAPIInputs = {
        "texts": ["final answer"],
        "images": ["data:image/png;base64,aGVsbG8="],
        "model": "test-model",
    }

    assert (
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=_request_data("/responses"),
            input_type="response",
        )
        == inputs
    )
    assert handler.calls[0].payload.texts == inputs["texts"]
    assert handler.calls[0].payload.images == inputs["images"]
    assert handler.calls[0].payload.model == inputs["model"]


def test_config_model_exposes_basic_adapter_fields() -> None:
    fields = AIGatewayModerationConfigModel.model_fields

    assert set(fields) >= {"api_base", "api_key", "fail_on_error", "optional_params"}
    assert AIGatewayModerationOptionalParams().decision_timeout_seconds == 5.0
    assert AIGatewayModerationOptionalParams().streaming_review_chunk_size == 256
    assert AIGatewayModerationOptionalParams().streaming_review_context_size == 256


def test_config_model_rejects_non_positive_decision_timeout() -> None:
    with pytest.raises(ValidationError):
        AIGatewayModerationOptionalParams(decision_timeout_seconds=0)


def test_config_model_rejects_non_positive_streaming_review_chunk_size() -> None:
    with pytest.raises(ValidationError):
        AIGatewayModerationOptionalParams(streaming_review_chunk_size=0)


def test_config_model_rejects_negative_streaming_review_context_size() -> None:
    with pytest.raises(ValidationError):
        AIGatewayModerationOptionalParams(streaming_review_context_size=-1)


def test_missing_adapter_configuration_is_rejected() -> None:
    handler = _RecordingHandler()

    with pytest.raises(ValueError, match="api_base is required"):
        AIGatewayModeration(
            api_base="",
            api_key="adapter-key",
            async_handler=cast(AsyncHTTPHandler, handler),
        )
    with pytest.raises(ValueError, match="api_key is required"):
        AIGatewayModeration(
            api_base="http://adapter:9800",
            api_key="",
            async_handler=cast(AsyncHTTPHandler, handler),
        )


def test_completed_basic_adapter_url_with_trailing_slash_is_not_duplicated() -> None:
    handler = _RecordingHandler()
    guardrail = AIGatewayModeration(
        api_base="http://adapter:9800/beta/litellm_basic_guardrail_api/",
        api_key="adapter-key",
        async_handler=cast(AsyncHTTPHandler, handler),
    )

    assert guardrail.api_base == "http://adapter:9800/beta/litellm_basic_guardrail_api"


def test_in_memory_update_rebuilds_basic_adapter_state() -> None:
    handler = _RecordingHandler()
    guardrail = _guardrail(handler)

    guardrail.update_in_memory_litellm_params(
        LitellmParams(
            guardrail="aigateway_moderation",
            mode=GuardrailEventHooks.post_call,
            api_base="http://new-adapter:9900/beta/litellm_basic_guardrail_api/",
            api_key="new-key",
            default_on=True,
            fail_on_error=False,
            optional_params={
                "decision_timeout_seconds": 2.5,
                "streaming_review_chunk_size": 128,
                "streaming_review_context_size": 64,
            },
        )
    )

    assert guardrail.api_base == "http://new-adapter:9900/beta/litellm_basic_guardrail_api"
    assert guardrail.headers["x-api-key"] == "new-key"
    assert guardrail.fail_on_error is False
    assert guardrail.decision_timeout_seconds == 2.5
    assert guardrail.streaming_review_chunk_size == 128
    assert guardrail.streaming_review_context_size == 64


def test_integration_is_registered() -> None:
    assert guardrail_class_registry["aigateway_moderation"] is AIGatewayModeration
    assert "aigateway_moderation" in guardrail_initializer_registry
