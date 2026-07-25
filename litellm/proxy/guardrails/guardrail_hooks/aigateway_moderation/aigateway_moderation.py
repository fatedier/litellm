import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from io import StringIO
from typing import TYPE_CHECKING, Literal, Optional, Protocol, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.litellm_core_utils.cached_imports import get_litellm_logging_class
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils import callback_utils
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.secret_managers.main import get_secret
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, Mode
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
    AIGatewayModerationOptionalParams,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
        AIGatewayModerationConfigModel,
    )

_BASIC_PATH = "/beta/litellm_basic_guardrail_api"
_DEFAULT_STREAM_REVIEW_CHUNK_SIZE = 256
_DEFAULT_STREAM_REVIEW_CONTEXT_SIZE = 256
_MAX_SSE_BUFFER_BYTES = 8 * 1024 * 1024
_MAX_PENDING_STREAM_BYTES = 8 * 1024 * 1024
_STREAM_DECISION_TIMEOUT_SECONDS = 5.0
_SSE_FRAME_DELIMITER_PATTERN = re.compile(rb"\r\n\r\n|\n\n|\r\r")
_ANTHROPIC_CONTROL_EVENT_TYPES = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
    }
)
_RESPONSE_STREAM_EVENT_TYPES = frozenset(event.value for event in ResponsesAPIStreamEvents)
_RESPONSE_IMAGE_PARTIAL_EVENT_TYPE = "response.image_generation_call.partial_image"
_ADDITIONAL_RESPONSE_CONTROL_EVENT_TYPES = frozenset(
    {
        "response.queued",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call.completed",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        _RESPONSE_IMAGE_PARTIAL_EVENT_TYPE,
        "response.image_generation_call.completed",
    }
)
_RESPONSE_TEXT_DELTA_EVENT_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.refusal.delta",
    }
)
_RESPONSE_TEXT_DONE_EVENT_TYPES = frozenset(
    {
        "response.output_text.done",
        "response.refusal.done",
    }
)
_RESPONSE_TOOL_EVENT_TYPES = frozenset(
    {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    }
)
_RESPONSE_REASONING_TEXT_EVENT_TYPES = frozenset(
    {
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    }
)
_RESPONSE_AUDIO_CONTROL_EVENT_TYPES = frozenset(
    {
        "response.audio.done",
        "response.audio.transcript.done",
    }
)
_RESPONSE_AUDIO_DELTA_EVENT_TYPE = "response.audio.delta"
_RESPONSE_AUDIO_TRANSCRIPT_DELTA_EVENT_TYPE = "response.audio.transcript.delta"
_RESPONSE_TERMINAL_EVENT_TYPES = frozenset(
    {
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value,
        ResponsesAPIStreamEvents.RESPONSE_FAILED.value,
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value,
    }
)
StreamChunk: TypeAlias = BaseModel | bytes | bytearray | str | Mapping[str, object]
_StreamAPI: TypeAlias = Literal["chat_completions", "responses", "anthropic_messages"]
_AIGatewayAPIType: TypeAlias = Literal["chat_completions", "responses", "anthropic_messages", "embeddings", "other"]
_ResponsePartKey: TypeAlias = tuple[str, int | None, int | None, int | None, str | None]
_ResponseTextPart: TypeAlias = tuple[_ResponsePartKey, str]
_STRING_OBJECT_MAPPING_ADAPTER = TypeAdapter(Mapping[str, object])
_AIGATEWAY_API_TYPE_KEY = "_aigateway_api_type"


def _is_response_tool_item_type(item_type: str | None) -> bool:
    return item_type is not None and item_type.endswith("_call") and item_type != "image_generation_call"


class _GenericGuardrailInitializer(Protocol):
    def __call__(
        self,
        *,
        api_base: str | None,
        api_key: str | None,
        guardrail_name: str | None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None,
        default_on: bool,
        fail_on_error: bool,
    ) -> None: ...


class _GenericGuardrailApply(Protocol):
    def __call__(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> Awaitable[GenericGuardrailAPIInputs]: ...


class _AddAppliedGuardrail(Protocol):
    def __call__(self, request_data: dict[str, object], guardrail_name: str | None) -> None: ...


class _OpenPayload(BaseModel):
    model_config = ConfigDict(extra="allow", from_attributes=True)


class _TextDelta(BaseModel):
    type: str | None = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    content: str | None = None
    citation: _OpenPayload | None = None


class _ChatAudio(BaseModel):
    transcript: str | None = None


class _ChatImageURL(BaseModel):
    url: str | None = None


class _ChatImage(BaseModel):
    image_url: _ChatImageURL | str | None = None


class _ThinkingBlock(BaseModel):
    thinking: str | None = None


class _ChatDelta(BaseModel):
    content: str | None = None
    refusal: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    function_call: _OpenPayload | None = None
    tool_calls: tuple[_OpenPayload, ...] | None = None
    audio: _ChatAudio | None = None
    images: tuple[_ChatImage, ...] | None = None
    thinking_blocks: tuple[_ThinkingBlock, ...] | None = None
    reasoning_items: tuple[_OpenPayload, ...] | None = None
    annotations: tuple[_OpenPayload, ...] | None = None
    provider_specific_fields: _OpenPayload | None = None


class _ChatChoice(BaseModel):
    index: int | None = None
    delta: _ChatDelta = Field(default_factory=_ChatDelta)
    finish_reason: str | None = None


class _ResponseOutputContent(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: str | None = None
    text: str | None = None
    refusal: str | None = None
    reasoning: str | None = None
    logs: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    annotations: tuple[_OpenPayload, ...] = ()


class _ResponseOutputItem(BaseModel):
    model_config = ConfigDict(extra="allow", from_attributes=True)

    id: str | None = None
    type: str | None = None
    content: tuple[_ResponseOutputContent, ...] | None = None
    summary: tuple[_ResponseOutputContent, ...] = ()
    arguments: object | None = None
    code: str | None = None
    input: str | None = None
    output: object | None = None
    error: str | None = None
    outputs: tuple[_ResponseOutputContent, ...] | None = None
    queries: tuple[str, ...] = ()
    results: tuple[_ResponseOutputContent, ...] | None = None
    result: str | None = None


class _ResponseContainer(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    output: tuple[_ResponseOutputItem, ...] = ()


class _StreamPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    type: str | None = None
    sequence_number: int | None = Field(default=None, ge=0, strict=True)
    delta: str | _TextDelta | None = None
    text: str | None = None
    refusal: str | None = None
    arguments: str | None = None
    code: str | None = None
    input: str | None = None
    output: str | tuple[_ResponseOutputContent, ...] | None = None
    stdout: str | None = None
    stderr: str | None = None
    partial_image_b64: str | None = None
    b64_json: str | None = None
    output_index: int | None = None
    content_index: int | None = None
    summary_index: int | None = None
    annotation_index: int | None = None
    item_id: str | None = None
    item: _ResponseOutputItem | None = None
    part: _ResponseOutputContent | None = None
    annotation: _OpenPayload | None = None
    response: _ResponseContainer | None = None
    content_block: _OpenPayload | None = None
    choices: tuple[_ChatChoice, ...] = ()


@dataclass(frozen=True, slots=True)
class _ChatChoiceText:
    index: int | None
    text: str
    is_finished: bool = False


@dataclass(frozen=True, slots=True)
class _ResponsePartAccumulator:
    buffer: StringIO
    last_position: int
    interleaved: bool = False
    finalized: bool = False


class _ResponsesTextAccumulator:
    __slots__ = ("parts", "next_position")

    def __init__(self) -> None:
        self.parts: dict[_ResponsePartKey, _ResponsePartAccumulator] = {}
        self.next_position = 0

    @property
    def has_unfinalized_parts(self) -> bool:
        return any(not part.finalized for part in self.parts.values())

    def update_part(
        self,
        part_key: _ResponsePartKey,
        text: str,
        *,
        authoritative: bool,
        finalized: bool,
    ) -> tuple[str, bool]:
        previous_state = self.parts.get(part_key)
        reorders_existing_parts = previous_state is None and any(
            _response_part_order(existing_key) > _response_part_order(part_key) for existing_key in self.parts
        )
        position = self.next_position
        self.next_position = position + 1
        if not authoritative:
            buffer = previous_state.buffer if previous_state is not None else StringIO()
            buffer.write(text)
            updated_part = _ResponsePartAccumulator(
                buffer,
                position,
                previous_state is not None
                and (previous_state.interleaved or previous_state.last_position < position - 1),
                finalized,
            )
            self.parts[part_key] = updated_part
            return text, reorders_existing_parts or updated_part.interleaved
        previous = "" if previous_state is None else previous_state.buffer.getvalue()
        requires_full_text = previous_state is not None and (
            previous_state.interleaved or (previous_state.last_position < position - 1 and text != previous)
        )
        replaces_previous = previous_state is not None and not text.startswith(previous)
        unchecked_text = text if requires_full_text or replaces_previous else text[len(previous) :]
        self.parts[part_key] = _ResponsePartAccumulator(StringIO(text), position, finalized=finalized)
        return unchecked_text, reorders_existing_parts or requires_full_text or replaces_previous

    def snapshot(self) -> str:
        ordered_parts = sorted(
            self.parts.items(),
            key=lambda item: (
                item[0][1] if item[0][1] is not None else -1,
                item[0][2] if item[0][2] is not None else -1,
                item[0][3] if item[0][3] is not None else -1,
                item[1].last_position,
            ),
        )
        return "".join(part.buffer.getvalue() for _, part in ordered_parts)


@dataclass(frozen=True, slots=True)
class _ExtractedStreamText:
    text: str
    chat_choices: tuple[_ChatChoiceText, ...] = ()
    images: tuple[str, ...] = ()
    is_terminal: bool = False
    has_tool_call: bool = False
    response_tool_call_ids: frozenset[str] = frozenset()
    response_text_snapshot: str | None = None
    append_to_response_text: bool = False
    needs_response_text_snapshot: bool = False


@dataclass(frozen=True, slots=True)
class _ExtractedStreamBatch:
    frames: tuple[_ExtractedStreamText, ...] = ()


def _resolve_config_reference(value: str, field_name: str) -> str:
    if not value.startswith("os.environ/"):
        return value
    resolved = cast(object, get_secret(value))
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{field_name} environment reference could not be resolved")
    return resolved


class _StreamingModerationFailure(GuardrailRaisedException):
    pass


def _stream_failure(message: str) -> _StreamingModerationFailure:
    return _StreamingModerationFailure(
        guardrail_name="aigateway_moderation",
        message=message,
        should_wrap_with_default_message=False,
        status_code=502,
    )


def _configured_guardrail_exception(
    exception: GuardrailRaisedException,
    guardrail_name: str | None,
) -> GuardrailRaisedException:
    return GuardrailRaisedException(
        guardrail_name=guardrail_name or "aigateway_moderation",
        message=exception.message,
        should_wrap_with_default_message=False,
        status_code=exception.status_code,
    )


def _stream_api(request_route: str | None) -> _StreamAPI | None:
    route = (request_route or "").rstrip("/")
    if route.endswith("/chat/completions"):
        return "chat_completions"
    if route.endswith("/responses"):
        return "responses"
    if route.endswith("/messages"):
        return "anthropic_messages"
    return None


def _request_route(request_data: Mapping[str, object]) -> str | None:
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = request_data.get(metadata_key)
        if isinstance(metadata, Mapping):
            metadata_route = _optional_string(cast(Mapping[str, object], metadata), "user_api_key_request_route")
            if metadata_route is not None:
                return metadata_route
    return None


def _aigateway_api_type(request_route: str | None) -> _AIGatewayAPIType:
    route = (request_route or "").rstrip("/")
    stream_api = _stream_api(route)
    if stream_api is not None:
        return stream_api
    if route.endswith("/responses/{response_id}"):
        return "responses"
    if route.endswith("/embeddings"):
        return "embeddings"
    return "other"


def _request_data_with_api_type(
    request_data: Mapping[str, object],
    api_type: _AIGatewayAPIType,
    *,
    include_body: bool,
) -> dict[str, object]:
    if not include_body:
        return {**request_data, _AIGATEWAY_API_TYPE_KEY: api_type}
    raw_body = request_data.get("body")
    body = dict(cast(Mapping[str, object], raw_body)) if isinstance(raw_body, Mapping) else {}
    return {
        **request_data,
        _AIGATEWAY_API_TYPE_KEY: api_type,
        "body": {**body, _AIGATEWAY_API_TYPE_KEY: api_type},
    }


def _request_data_with_authenticated_route(
    request_data: Mapping[str, object], request_route: str | None
) -> dict[str, object]:
    raw_metadata = request_data.get("metadata")
    metadata = _STRING_OBJECT_MAPPING_ADAPTER.validate_python(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    return {
        **request_data,
        "metadata": {**metadata, "user_api_key_request_route": request_route},
    }


def _stream_error_frame(
    request_route: str | None,
    exception: GuardrailRaisedException,
    response_sequence_number: int | None,
) -> bytes | None:
    stream_api = _stream_api(request_route)
    if stream_api is None:
        return None
    is_block = exception.status_code < 500
    if stream_api == "anthropic_messages":
        payload: dict[str, object] = {
            "type": "error",
            "error": {
                "type": "invalid_request_error" if is_block else "api_error",
                "message": exception.message,
            },
        }
        event_name = "event: error\n"
    elif stream_api == "responses":
        payload = {
            "type": "error",
            "sequence_number": 0 if response_sequence_number is None else response_sequence_number + 1,
            "code": "guardrail_violation" if is_block else "moderation_unavailable",
            "message": exception.message,
            "param": None,
        }
        event_name = "event: error\n"
    else:
        payload = {
            "error": {
                "message": exception.message,
                "type": "guardrail_violation" if is_block else "moderation_unavailable",
                "param": None,
                "code": str(exception.status_code),
            }
        }
        event_name = ""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{event_name}data: {data}\n\n".encode()


def _stream_chunk_byte_size(item: StreamChunk) -> int:
    if isinstance(item, bytearray):
        return len(item)
    if isinstance(item, bytes):
        return len(item)
    if isinstance(item, str):
        return len(item.encode())
    if isinstance(item, BaseModel):
        return len(item.model_dump_json().encode())
    return len(_STRING_OBJECT_MAPPING_ADAPTER.dump_json(item))


def _response_sequence_number(item: StreamChunk) -> int | None:
    if isinstance(item, BaseModel):
        value = getattr(item, "sequence_number", None)
    elif isinstance(item, Mapping):
        value = item.get("sequence_number")
    else:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _optional_string(request_data: Mapping[str, object], key: str) -> str | None:
    value = request_data.get(key)
    return value if isinstance(value, str) else None


def _requested_choice_count(request_data: Mapping[str, object]) -> int:
    value = request_data.get("n")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _chat_delta_text(delta: _ChatDelta) -> str:
    return "".join((delta.content or "", delta.refusal or ""))


def _chat_image_url(image: _ChatImage) -> str | None:
    if isinstance(image.image_url, str):
        return image.image_url
    if isinstance(image.image_url, _ChatImageURL):
        return image.image_url.url
    return None


def _chat_delta_images(delta: _ChatDelta) -> tuple[str, ...]:
    return tuple(image_url for image in delta.images or () for image_url in (_chat_image_url(image),) if image_url)


def _response_part_key(event_type: str, payload: _StreamPayload) -> _ResponsePartKey:
    family = event_type.removesuffix(".delta").removesuffix(".done")
    if payload.output_index is None or payload.item_id is None:
        raise _stream_failure("Streaming response contained a Responses API text event without an item identity")
    if family in {"response.output_text", "response.refusal"} and payload.content_index is None:
        raise _stream_failure("Streaming response contained a Responses API text event without a content index")
    return (
        family,
        payload.output_index,
        payload.content_index,
        payload.summary_index,
        payload.item_id,
    )


def _response_done_text(event_type: str, payload: _StreamPayload) -> str | None:
    match event_type:
        case "response.output_text.done":
            return payload.text
        case "response.refusal.done":
            return payload.refusal
        case _:
            return None


def _response_part_order(part_key: _ResponsePartKey) -> tuple[int, int, int]:
    return (
        part_key[1] if part_key[1] is not None else -1,
        part_key[2] if part_key[2] is not None else -1,
        part_key[3] if part_key[3] is not None else -1,
    )


def _response_content_part(
    item: _ResponseOutputItem,
    output_index: int,
    content_index: int,
    content: _ResponseOutputContent,
) -> _ResponseTextPart | None:
    match content.type:
        case "output_text":
            family, text, part_content_index, summary_index = (
                "response.output_text",
                content.text,
                content_index,
                None,
            )
        case "refusal":
            family, text, part_content_index, summary_index = (
                "response.refusal",
                content.refusal or content.text,
                content_index,
                None,
            )
        case "reasoning_text" | "summary_text":
            return None
        case _:
            if content.text or content.refusal:
                raise _stream_failure("Streaming response contained unsupported terminal Responses text")
            return None
    if not text:
        return None
    return (
        (family, output_index, part_content_index, summary_index, item.id),
        text,
    )


def _response_output_parts(item: _ResponseOutputItem, output_index: int) -> tuple[_ResponseTextPart, ...]:
    if item.type != "message":
        return ()
    return tuple(
        part
        for part in (
            *(
                _response_content_part(item, output_index, content_index, content)
                for content_index, content in enumerate(item.content or ())
            ),
        )
        if part is not None
    )


def _response_output_images(item: _ResponseOutputItem) -> tuple[str, ...]:
    return (item.result,) if item.type == "image_generation_call" and item.result else ()


def _response_event_images(event_type: str | None, payload: _StreamPayload) -> tuple[str, ...]:
    if event_type == _RESPONSE_IMAGE_PARTIAL_EVENT_TYPE:
        if not payload.partial_image_b64:
            raise _stream_failure("Streaming response contained an invalid Responses partial image")
        return (payload.partial_image_b64,)
    if event_type == ResponsesAPIStreamEvents.IMAGE_GENERATION_PARTIAL_IMAGE.value:
        if not payload.b64_json:
            raise _stream_failure("Streaming response contained an invalid Responses partial image")
        return (payload.b64_json,)
    if event_type in {
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value,
    }:
        return () if payload.item is None else _response_output_images(payload.item)
    if event_type not in _RESPONSE_TERMINAL_EVENT_TYPES or payload.response is None:
        return ()
    return tuple(image for item in payload.response.output for image in _response_output_images(item))


def _terminal_response_parts(event_type: str | None, payload: _StreamPayload) -> tuple[_ResponseTextPart, ...]:
    if event_type == ResponsesAPIStreamEvents.SHELL_CALL_OUTPUT.value:
        if payload.output_index is None:
            raise _stream_failure("Streaming response contained shell output without an output index")
        nested_output = (
            *(payload.output if isinstance(payload.output, tuple) else ()),
            *(
                (_ResponseOutputContent(stdout=payload.stdout, stderr=payload.stderr),)
                if payload.stdout or payload.stderr
                else ()
            ),
        )
        item = _ResponseOutputItem(
            id=payload.item_id,
            type="shell_call_output",
            output=payload.output if isinstance(payload.output, str) else nested_output,
        )
        return _response_output_parts(item, payload.output_index)
    if event_type in {
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value,
    }:
        if payload.item is None or payload.output_index is None:
            raise _stream_failure("Streaming response contained an invalid Responses output item")
        return _response_output_parts(payload.item, payload.output_index)
    if event_type not in {
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value,
        ResponsesAPIStreamEvents.RESPONSE_FAILED.value,
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value,
    }:
        return ()
    if payload.response is None:
        return ()
    return tuple(
        part
        for output_index, item in enumerate(payload.response.output)
        for part in _response_output_parts(item, output_index)
    )


def _response_tool_call_id(item_id: str | None) -> str:
    if not item_id:
        raise _stream_failure("Streaming Responses tool event did not include an item identity")
    return item_id


def _terminal_response_tool_call_ids(event_type: str | None, payload: _StreamPayload) -> frozenset[str]:
    if event_type not in _RESPONSE_TERMINAL_EVENT_TYPES or payload.response is None:
        return frozenset()
    return frozenset(
        _response_tool_call_id(item.id) for item in payload.response.output if _is_response_tool_item_type(item.type)
    )


def _response_part_event_parts(
    event_type: str | None,
    payload: _StreamPayload,
) -> tuple[_ResponseTextPart, ...] | None:
    content_part_events = {
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED.value,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE.value,
    }
    summary_part_events = {
        ResponsesAPIStreamEvents.RESPONSE_PART_ADDED.value,
        ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE.value,
    }
    if event_type not in content_part_events | summary_part_events:
        return None
    if payload.part is None or payload.output_index is None or payload.item_id is None:
        raise _stream_failure("Streaming response contained an invalid Responses API part event")
    item = _ResponseOutputItem(id=payload.item_id)
    if event_type in content_part_events:
        if payload.content_index is None:
            raise _stream_failure("Streaming response contained an invalid Responses API content part event")
        content_part = _response_content_part(item, payload.output_index, payload.content_index, payload.part)
        return () if content_part is None else (content_part,)
    if payload.summary_index is None:
        raise _stream_failure("Streaming response contained an invalid Responses API summary part event")
    return ()


def _extract_authoritative_response_parts(
    parts: tuple[_ResponseTextPart, ...],
    response_text: _ResponsesTextAccumulator,
    *,
    is_terminal: bool,
    finalizes_parts: bool,
) -> _ExtractedStreamText:
    texts: list[str] = []
    needs_response_text_snapshot = False
    for part_key, text in parts:
        unchecked_text, part_needs_snapshot = response_text.update_part(
            part_key,
            text,
            authoritative=True,
            finalized=finalizes_parts,
        )
        texts.append(unchecked_text)
        needs_response_text_snapshot = needs_response_text_snapshot or part_needs_snapshot
    return _ExtractedStreamText(
        "".join(texts),
        is_terminal=is_terminal,
        append_to_response_text=True,
        needs_response_text_snapshot=needs_response_text_snapshot,
    )


def _extract_chat_payload(payload: _StreamPayload) -> _ExtractedStreamText:
    return _ExtractedStreamText(
        text="",
        chat_choices=tuple(
            _ChatChoiceText(
                choice.index,
                _chat_delta_text(choice.delta),
                choice.finish_reason is not None,
            )
            for choice in payload.choices
        ),
        images=tuple(image for choice in payload.choices for image in _chat_delta_images(choice.delta)),
        is_terminal=any(choice.finish_reason is not None for choice in payload.choices),
        has_tool_call=any(
            choice.delta.function_call is not None or bool(choice.delta.tool_calls) for choice in payload.choices
        ),
    )


def _extract_anthropic_payload(event_type: str | None, payload: _StreamPayload) -> _ExtractedStreamText | None:
    if event_type == "content_block_start":
        content_block = (
            None
            if payload.content_block is None
            else _STRING_OBJECT_MAPPING_ADAPTER.validate_python(payload.content_block.model_dump(mode="python"))
        )
        content_block_type = None if content_block is None else content_block.get("type")
        content_block_text = None if content_block is None else content_block.get("text")
        text = content_block_text if content_block_type == "text" and isinstance(content_block_text, str) else ""
        return _ExtractedStreamText(
            text,
            has_tool_call=content_block_type in {"tool_use", "server_tool_use", "mcp_tool_use"},
        )
    if event_type == "content_block_delta":
        if not isinstance(payload.delta, _TextDelta) or payload.delta.type is None:
            raise _stream_failure("Streaming response contained an invalid Anthropic content delta")
        match payload.delta.type:
            case "text_delta":
                text = payload.delta.text
            case "thinking_delta":
                text = ""
            case "input_json_delta":
                return _ExtractedStreamText("", has_tool_call=True)
            case "compaction_delta":
                return _ExtractedStreamText("")
            case "signature_delta":
                return _ExtractedStreamText("")
            case "citations" | "citations_delta":
                if payload.delta.citation is None:
                    raise _stream_failure("Streaming response contained an invalid Anthropic citation delta")
                return _ExtractedStreamText("")
            case _:
                raise _stream_failure("Streaming response contained an unsupported Anthropic content delta")
        if text is None:
            raise _stream_failure("Streaming response contained an invalid Anthropic content delta")
        return _ExtractedStreamText(text)
    if event_type in _ANTHROPIC_CONTROL_EVENT_TYPES:
        return _ExtractedStreamText("", is_terminal=event_type == "message_stop")
    return None


def _extract_responses_payload(
    event_type: str | None,
    payload: _StreamPayload,
    response_text: _ResponsesTextAccumulator,
) -> _ExtractedStreamText:
    if event_type == _RESPONSE_AUDIO_TRANSCRIPT_DELTA_EVENT_TYPE:
        if not isinstance(payload.delta, str):
            raise _stream_failure("Streaming response contained an invalid Responses audio transcript delta")
        # Audio and its transcript are intentionally outside the current moderation scope, so this
        # text is parsed but not appended to the response text review window.
        return _ExtractedStreamText(payload.delta)
    if event_type == _RESPONSE_AUDIO_DELTA_EVENT_TYPE:
        if not isinstance(payload.delta, str):
            raise _stream_failure("Streaming response contained an invalid Responses audio delta")
        return _ExtractedStreamText("")
    if event_type in _RESPONSE_AUDIO_CONTROL_EVENT_TYPES:
        return _ExtractedStreamText("")
    if event_type in _RESPONSE_REASONING_TEXT_EVENT_TYPES:
        return _ExtractedStreamText("")
    if event_type in _RESPONSE_TEXT_DELTA_EVENT_TYPES:
        if not isinstance(payload.delta, str):
            raise _stream_failure("Streaming response contained an invalid Responses API text delta")
        part_key = _response_part_key(event_type, payload)
        unchecked_text, needs_response_text_snapshot = response_text.update_part(
            part_key,
            payload.delta,
            authoritative=False,
            finalized=False,
        )
        return _ExtractedStreamText(
            unchecked_text,
            append_to_response_text=True,
            needs_response_text_snapshot=needs_response_text_snapshot,
        )
    if event_type in _RESPONSE_TEXT_DONE_EVENT_TYPES:
        part_key = _response_part_key(event_type, payload)
        text = _response_done_text(event_type, payload)
        if text is None:
            raise _stream_failure("Streaming response contained an invalid Responses API text completion")
        unchecked_text, needs_response_text_snapshot = response_text.update_part(
            part_key,
            text,
            authoritative=True,
            finalized=True,
        )
        return _ExtractedStreamText(
            unchecked_text,
            append_to_response_text=True,
            needs_response_text_snapshot=needs_response_text_snapshot,
        )
    if event_type in _RESPONSE_TOOL_EVENT_TYPES:
        return _ExtractedStreamText(
            "",
            has_tool_call=True,
            response_tool_call_ids=frozenset((_response_tool_call_id(payload.item_id),)),
        )
    if (
        event_type
        in {
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value,
        }
        and payload.item is not None
        and _is_response_tool_item_type(payload.item.type)
    ):
        return _ExtractedStreamText(
            "",
            has_tool_call=True,
            response_tool_call_ids=frozenset((_response_tool_call_id(payload.item.id),)),
        )
    if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED.value:
        return _ExtractedStreamText("")
    response_part_event_parts = _response_part_event_parts(event_type, payload)
    if response_part_event_parts is not None:
        return _extract_authoritative_response_parts(
            response_part_event_parts,
            response_text,
            is_terminal=False,
            finalizes_parts=event_type == ResponsesAPIStreamEvents.CONTENT_PART_DONE.value,
        )
    if event_type == ResponsesAPIStreamEvents.ERROR.value:
        raise _stream_failure("Streaming response contained an upstream error event")
    response_images = _response_event_images(event_type, payload)
    terminal_parts = _terminal_response_parts(event_type, payload)
    if terminal_parts:
        return replace(
            _extract_authoritative_response_parts(
                terminal_parts,
                response_text,
                is_terminal=event_type in _RESPONSE_TERMINAL_EVENT_TYPES,
                finalizes_parts=(
                    event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value
                    or event_type in _RESPONSE_TERMINAL_EVENT_TYPES
                ),
            ),
            images=response_images,
        )
    if response_images:
        return _ExtractedStreamText(
            "",
            images=response_images,
            is_terminal=event_type in _RESPONSE_TERMINAL_EVENT_TYPES,
        )
    terminal_tool_call_ids = _terminal_response_tool_call_ids(event_type, payload)
    if terminal_tool_call_ids:
        return _ExtractedStreamText(
            "",
            is_terminal=True,
            has_tool_call=True,
            response_tool_call_ids=terminal_tool_call_ids,
        )
    if event_type in _RESPONSE_TERMINAL_EVENT_TYPES:
        return _ExtractedStreamText("", is_terminal=True)
    if event_type in _RESPONSE_STREAM_EVENT_TYPES:
        return _ExtractedStreamText("")
    if event_type in _ADDITIONAL_RESPONSE_CONTROL_EVENT_TYPES:
        return _ExtractedStreamText("")
    raise _stream_failure("Streaming response contained an unsupported chunk")


def _extract_payload_text(
    payload: _StreamPayload,
    response_text: _ResponsesTextAccumulator,
    event_name: str | None = None,
) -> _ExtractedStreamText:
    if event_name is not None and payload.type is not None and event_name != payload.type:
        raise _stream_failure("Streaming response contained mismatched SSE event types")
    if "choices" in payload.model_fields_set:
        return _extract_chat_payload(payload)
    event_type = event_name or payload.type
    anthropic_text = _extract_anthropic_payload(event_type, payload)
    return (
        anthropic_text if anthropic_text is not None else _extract_responses_payload(event_type, payload, response_text)
    )


def _scan_sse_frames(raw: bytes | bytearray, scan_offset: int = 0) -> tuple[tuple[bytes, ...], int]:
    frames: list[bytes] = []
    offset = 0
    for delimiter in _SSE_FRAME_DELIMITER_PATTERN.finditer(raw, scan_offset):
        frames.append(bytes(raw[offset : delimiter.start()]))
        offset = delimiter.end()
    return tuple(frames), offset


class _ResponseSequenceTracker:
    def __init__(self) -> None:
        self.sse_buffer = bytearray()
        self.last_sequence_number: int | None = None

    def observe(self, item: StreamChunk) -> None:
        item_sequence_number = _response_sequence_number(item)
        if item_sequence_number is not None:
            self.last_sequence_number = item_sequence_number
            return
        if isinstance(item, bytearray):
            self._observe_sse_bytes(bytes(item))
        elif isinstance(item, bytes):
            self._observe_sse_bytes(item)

    def _observe_sse_bytes(self, raw: bytes) -> None:
        previous_length = len(self.sse_buffer)
        self.sse_buffer.extend(raw)
        scan_offset = max(0, previous_length - 3)
        frames, consumed_offset = _scan_sse_frames(self.sse_buffer, scan_offset)
        if len(self.sse_buffer) - consumed_offset > _MAX_SSE_BUFFER_BYTES or any(
            len(frame) > _MAX_SSE_BUFFER_BYTES for frame in frames
        ):
            raise _stream_failure("Streaming response exceeded the maximum SSE frame size")
        if consumed_offset:
            self.sse_buffer = self.sse_buffer[consumed_offset:]
        for frame in frames:
            self._observe_sse_frame(frame)

    def _observe_sse_frame(self, raw: bytes) -> None:
        try:
            block = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _stream_failure("Streaming response was not valid UTF-8") from exc
        data_lines = tuple(line[5:].strip() for line in block.splitlines() if line.startswith("data:"))
        if not data_lines:
            return
        data = "\n".join(data_lines)
        if data == "[DONE]":
            return
        try:
            payload = _StreamPayload.model_validate_json(data)
        except ValidationError as exc:
            raise _stream_failure("Streaming response contained an invalid SSE event") from exc
        if payload.sequence_number is not None:
            self.last_sequence_number = payload.sequence_number


def _extract_sse_block(
    raw: bytes,
    response_text: _ResponsesTextAccumulator,
) -> _ExtractedStreamText:
    try:
        block = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _stream_failure("Streaming response was not valid UTF-8") from exc
    lines = block.splitlines()
    data_lines = tuple(line[5:].strip() for line in lines if line.startswith("data:"))
    if not data_lines:
        if any(line.startswith("event:") for line in lines):
            raise _stream_failure("Streaming response contained an SSE event without data")
        if any(line and not line.startswith((":", "id:", "retry:")) for line in lines):
            raise _stream_failure("Streaming response contained a non-SSE byte frame")
        return _ExtractedStreamText("")
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return _ExtractedStreamText("", is_terminal=True)
    event_name = next((line[6:].strip() for line in lines if line.startswith("event:")), None)
    try:
        payload = _StreamPayload.model_validate_json(data)
    except ValidationError as exc:
        raise _stream_failure("Streaming response contained an invalid SSE event") from exc
    return _extract_payload_text(payload, response_text, event_name)


class _StreamTextExtractor:
    def __init__(self) -> None:
        self.sse_buffer = bytearray()
        self.response_text = _ResponsesTextAccumulator()
        self.seen_response_tool_call_ids = frozenset[str]()
        self.seen_image_digests = frozenset[bytes]()

    @property
    def has_incomplete_sse_frame(self) -> bool:
        return bool(self.sse_buffer)

    @property
    def has_unfinalized_response_part(self) -> bool:
        return self.response_text.has_unfinalized_parts

    def extract(self, item: StreamChunk) -> _ExtractedStreamBatch:
        if isinstance(item, bytearray):
            return self._extract_sse_bytes(bytes(item))
        if isinstance(item, bytes):
            return self._extract_sse_bytes(item)
        if isinstance(item, str):
            if self.sse_buffer:
                return self._extract_sse_bytes(item.encode())
            return _ExtractedStreamBatch((_ExtractedStreamText(item, append_to_response_text=True),))
        if self.sse_buffer:
            raise _stream_failure("Streaming response ended an SSE frame with an unsupported chunk")
        try:
            payload = (
                _StreamPayload.model_validate_json(item.model_dump_json())
                if isinstance(item, BaseModel)
                else _StreamPayload.model_validate(item)
            )
        except ValidationError as exc:
            raise _stream_failure("Streaming response contained an unsupported chunk") from exc
        extracted = _extract_payload_text(payload, self.response_text)
        return _ExtractedStreamBatch((self._prepare_extracted(extracted),))

    def finish(self) -> _ExtractedStreamBatch:
        if not self.sse_buffer.strip():
            self.sse_buffer = bytearray()
            return _ExtractedStreamBatch()
        extracted = _extract_sse_block(bytes(self.sse_buffer), self.response_text)
        self.sse_buffer = bytearray()
        return _ExtractedStreamBatch((self._prepare_extracted(extracted),))

    def _extract_sse_bytes(self, raw: bytes) -> _ExtractedStreamBatch:
        previous_length = len(self.sse_buffer)
        self.sse_buffer.extend(raw)
        scan_offset = max(0, previous_length - 3)
        frames, consumed_offset = _scan_sse_frames(self.sse_buffer, scan_offset)
        if len(self.sse_buffer) - consumed_offset > _MAX_SSE_BUFFER_BYTES or any(
            len(frame) > _MAX_SSE_BUFFER_BYTES for frame in frames
        ):
            raise _stream_failure("Streaming response exceeded the maximum SSE frame size")
        if consumed_offset:
            self.sse_buffer = self.sse_buffer[consumed_offset:]
        return _ExtractedStreamBatch(tuple(self._extract_sse_frames(frames)))

    def _extract_sse_frames(self, frames: tuple[bytes, ...]) -> Iterator[_ExtractedStreamText]:
        for frame in frames:
            if not frame.strip():
                continue
            extracted = _extract_sse_block(frame, self.response_text)
            yield self._prepare_extracted(extracted)

    def _prepare_extracted(self, extracted: _ExtractedStreamText) -> _ExtractedStreamText:
        image_digests = tuple((image, hashlib.sha256(image.encode()).digest()) for image in extracted.images)
        new_images = tuple(image for image, digest in image_digests if digest not in self.seen_image_digests)
        self.seen_image_digests = self.seen_image_digests | frozenset(digest for _, digest in image_digests)
        extracted_with_images = replace(extracted, images=new_images)
        tool_call_ids = extracted.response_tool_call_ids
        if not tool_call_ids:
            return _with_response_text_snapshot(extracted_with_images, self.response_text)
        has_new_tool_call = not tool_call_ids.issubset(self.seen_response_tool_call_ids)
        self.seen_response_tool_call_ids = self.seen_response_tool_call_ids | tool_call_ids
        return _with_response_text_snapshot(
            extracted_with_images if has_new_tool_call else replace(extracted_with_images, has_tool_call=False),
            self.response_text,
        )


def _with_response_text_snapshot(
    extracted: _ExtractedStreamText,
    response_text: _ResponsesTextAccumulator,
) -> _ExtractedStreamText:
    return _ExtractedStreamText(
        text=extracted.text,
        chat_choices=extracted.chat_choices,
        images=extracted.images,
        is_terminal=extracted.is_terminal,
        has_tool_call=extracted.has_tool_call,
        response_tool_call_ids=extracted.response_tool_call_ids,
        response_text_snapshot=(
            response_text.snapshot() if extracted.has_tool_call or extracted.needs_response_text_snapshot else None
        ),
        append_to_response_text=extracted.append_to_response_text,
        needs_response_text_snapshot=extracted.needs_response_text_snapshot,
    )


@dataclass(frozen=True, slots=True)
class _ReviewProgress:
    approved_text: str = ""
    previous_context: str = ""


@dataclass(frozen=True, slots=True)
class _StreamReviewState:
    current_text: str = ""
    review_progress: _ReviewProgress = _ReviewProgress()
    parallel_choice_text: Mapping[int, str] = field(default_factory=dict)
    parallel_choice_progress: tuple[tuple[int, _ReviewProgress], ...] = ()
    finished_choice_indices: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class _ReviewPlan:
    base_approved_text: str
    previous_context: str
    segments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingStreamChunk:
    item: StreamChunk
    previous: Optional["_PendingStreamChunk"] = None


def _review_plan(
    current_text: str,
    progress: _ReviewProgress,
    *,
    review_tail: bool,
    chunk_size: int,
) -> _ReviewPlan:
    if current_text == progress.approved_text:
        return _ReviewPlan(progress.approved_text, progress.previous_context, ())
    if current_text.startswith(progress.approved_text):
        base_approved_text = progress.approved_text
        previous_context = progress.previous_context
        pending_text = current_text[len(progress.approved_text) :]
    else:
        base_approved_text = ""
        previous_context = ""
        pending_text = current_text
    segments = (pending_text,) if review_tail or len(pending_text) >= chunk_size else ()
    return _ReviewPlan(base_approved_text, previous_context, segments)


def _review_window_after_release(
    requested_choices: int,
    current_text: str,
    review_progress: _ReviewProgress,
) -> tuple[str, _ReviewProgress]:
    if requested_choices != 1:
        return current_text, review_progress
    return "", _ReviewProgress(previous_context=review_progress.previous_context)


def _compact_parallel_choice_progress(
    choice_progress: tuple[tuple[int, _ReviewProgress], ...],
) -> tuple[tuple[int, _ReviewProgress], ...]:
    return tuple(
        (choice_index, _ReviewProgress(previous_context=progress.previous_context))
        for choice_index, progress in choice_progress
    )


def _release_stream_chunks(
    latest_chunk: _PendingStreamChunk | None,
    response_sequence_tracker: _ResponseSequenceTracker | None,
) -> Iterator[StreamChunk]:
    newest_first: tuple[StreamChunk, ...] = tuple(node.item for node in _pending_stream_chunk_nodes(latest_chunk))
    for chunk in reversed(newest_first):
        if response_sequence_tracker is not None:
            response_sequence_tracker.observe(chunk)
        yield chunk


def _pending_stream_chunk_nodes(latest_chunk: _PendingStreamChunk | None) -> Iterator[_PendingStreamChunk]:
    current = latest_chunk
    while current is not None:
        yield current
        current = current.previous


def _pending_stream_buffer_exceeded(
    pending_bytes: int,
    extractor: _StreamTextExtractor,
    text_is_approved: bool,
) -> bool:
    return pending_bytes > _MAX_PENDING_STREAM_BYTES and (extractor.has_incomplete_sse_frame or not text_is_approved)


def _parallel_choices_are_approved(
    choice_text: Mapping[int, str],
    choice_progress: tuple[tuple[int, _ReviewProgress], ...],
) -> bool:
    return all(
        text == _parallel_choice_progress(choice_progress, index).approved_text for index, text in choice_text.items()
    )


def _parallel_choice_progress(
    choice_progress: tuple[tuple[int, _ReviewProgress], ...],
    choice_index: int,
) -> _ReviewProgress:
    return next((progress for index, progress in choice_progress if index == choice_index), _ReviewProgress())


def _last_tool_call_frame_index(frames: tuple[_ExtractedStreamText, ...]) -> int | None:
    return next((index for index in range(len(frames) - 1, -1, -1) if frames[index].has_tool_call), None)


def _last_response_snapshot_frame_index(frames: tuple[_ExtractedStreamText, ...]) -> int | None:
    return next(
        (index for index in range(len(frames) - 1, -1, -1) if frames[index].response_text_snapshot is not None),
        None,
    )


def _response_batch_review_state(
    frames: tuple[_ExtractedStreamText, ...],
    current_text: str,
    review_progress: _ReviewProgress,
) -> tuple[str, _ReviewProgress]:
    last_snapshot_index = _last_response_snapshot_frame_index(frames)
    if last_snapshot_index is None:
        base_text = current_text
        remaining_frames = frames
    else:
        base_text = frames[last_snapshot_index].response_text_snapshot or ""
        remaining_frames = frames[last_snapshot_index + 1 :]
    updated_text = base_text + "".join(frame.text for frame in remaining_frames if frame.append_to_response_text)
    last_tool_call_index = _last_tool_call_frame_index(frames)
    if last_tool_call_index is None:
        return updated_text, review_progress
    tool_snapshot = frames[last_tool_call_index].response_text_snapshot
    if tool_snapshot is None:
        raise _stream_failure("Streaming Responses tool event did not preserve its text boundary")
    # A Responses tool call is the same response-level boundary as Chat. Intentionally treat the
    # pre-tool snapshot as approved without another review; only subsequent final text starts a new window.
    return updated_text, _ReviewProgress(approved_text=tool_snapshot)


def _finish_stream_extraction(extractor: _StreamTextExtractor) -> _ExtractedStreamBatch:
    trailing = extractor.finish()
    if extractor.has_unfinalized_response_part:
        raise _stream_failure("Streaming response ended before a Responses API text part was finalized")
    return trailing


def _response_text_inputs(inputs: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
    texts = inputs.get("texts")
    images = inputs.get("images")
    if not texts and not images:
        return {}
    response_inputs: GenericGuardrailAPIInputs = {}
    if texts:
        response_inputs["texts"] = texts
    if images:
        response_inputs["images"] = images
    model = inputs.get("model")
    if model is not None:
        response_inputs["model"] = model
    return response_inputs


def _request_logging_obj(request_data: Mapping[str, object]) -> Optional["LiteLLMLoggingObj"]:
    logging_obj = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return None
    logging_class = get_litellm_logging_class()
    return logging_obj if isinstance(logging_obj, logging_class) else None


class AIGatewayModeration(GenericGuardrailAPI):
    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        async_handler: AsyncHTTPHandler | None = None,
        decision_timeout_seconds: float = _STREAM_DECISION_TIMEOUT_SECONDS,
        streaming_review_chunk_size: int = _DEFAULT_STREAM_REVIEW_CHUNK_SIZE,
        streaming_review_context_size: int = _DEFAULT_STREAM_REVIEW_CONTEXT_SIZE,
        fail_on_error: bool = True,
    ) -> None:
        if api_base is None or not api_base.strip():
            raise ValueError("api_base is required for AIGateway moderation")
        if api_key is None or not api_key.strip():
            raise ValueError("api_key is required for AIGateway moderation")
        if decision_timeout_seconds <= 0:
            raise ValueError("decision_timeout_seconds must be positive")
        if streaming_review_chunk_size <= 0:
            raise ValueError("streaming_review_chunk_size must be positive")
        if streaming_review_context_size < 0:
            raise ValueError("streaming_review_context_size must not be negative")
        parent_initializer = cast(  # cast-ok: legacy base kwargs are untyped; only documented parameters are passed
            _GenericGuardrailInitializer, super().__init__
        )
        parent_initializer(
            api_base=api_base,
            api_key=api_key,
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            fail_on_error=fail_on_error,
        )
        if async_handler is not None:
            self.async_handler = async_handler
        self.decision_timeout_seconds = decision_timeout_seconds
        self.streaming_review_chunk_size = streaming_review_chunk_size
        self.streaming_review_context_size = streaming_review_context_size

    @staticmethod
    def get_config_model() -> type["AIGatewayModerationConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
            AIGatewayModerationConfigModel,
        )

        return AIGatewayModerationConfigModel

    def update_in_memory_litellm_params(
        self,
        litellm_params: LitellmParams | Mapping[str, object],
    ) -> None:
        params = (
            litellm_params
            if isinstance(litellm_params, LitellmParams)
            else LitellmParams.model_validate(dict(litellm_params))
        )
        api_base = params.api_base
        api_key = params.api_key
        if api_base is None or not api_base.strip():
            raise ValueError("api_base is required for AIGateway moderation")
        if api_key is None or not api_key.strip():
            raise ValueError("api_key is required for AIGateway moderation")
        resolved_api_base = _resolve_config_reference(api_base, "api_base")
        resolved_api_key = _resolve_config_reference(api_key, "api_key")

        raw_optional_params = params.optional_params
        if isinstance(raw_optional_params, Mapping):
            optional_params = AIGatewayModerationOptionalParams.model_validate(raw_optional_params)
        elif isinstance(raw_optional_params, BaseModel):
            if "fail_on_error" in raw_optional_params.model_fields_set:
                raise ValueError("fail_on_error must be configured at litellm_params.fail_on_error")
            configured_fields = raw_optional_params.model_fields_set.intersection(
                AIGatewayModerationOptionalParams.model_fields
            )
            optional_params = AIGatewayModerationOptionalParams.model_validate(
                {
                    **raw_optional_params.model_dump(include=configured_fields),
                    **(raw_optional_params.model_extra or {}),
                }
            )
        else:
            optional_params = AIGatewayModerationOptionalParams()

        mode = params.mode
        if isinstance(mode, Mode):
            event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode = mode
        elif isinstance(mode, list):
            event_hook = [GuardrailEventHooks(item) for item in mode]
        else:
            event_hook = GuardrailEventHooks(mode)
        if self.supported_event_hooks is not None:
            self._validate_event_hook(event_hook, self.supported_event_hooks)

        normalized_base_url = resolved_api_base.rstrip("/")
        normalized_api_base = (
            normalized_base_url if normalized_base_url.endswith(_BASIC_PATH) else f"{normalized_base_url}{_BASIC_PATH}"
        )
        headers = {**self.headers, "x-api-key": resolved_api_key}

        super().update_in_memory_litellm_params(params)
        self.api_base = normalized_api_base
        self.headers = headers
        self.event_hook = event_hook
        self.default_on = params.default_on or False
        self.fail_on_error = True if params.fail_on_error is None else params.fail_on_error
        self.decision_timeout_seconds = optional_params.decision_timeout_seconds
        self.streaming_review_chunk_size = optional_params.streaming_review_chunk_size
        self.streaming_review_context_size = optional_params.streaming_review_context_size

    def _validate_guardrail_response(
        self,
        response_json: object,
        input_type: Literal["request", "response"],
        request_data: Mapping[str, object],
        inputs: GenericGuardrailAPIInputs,
    ) -> None:
        super()._validate_guardrail_response(response_json, input_type, request_data, inputs)
        if not isinstance(response_json, Mapping):
            raise ValueError("AIGateway moderation returned a non-object response")
        action = response_json.get("action")
        if action in {"NONE", "BLOCKED"}:
            return
        if action == "GUARDRAIL_INTERVENED":
            blocked_reason = response_json.get("blocked_reason")
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=blocked_reason
                if isinstance(blocked_reason, str) and blocked_reason
                else "Content violates policy",
                should_wrap_with_default_message=False,
            )
        raise ValueError("AIGateway moderation returned an invalid action")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        parent_apply = cast(  # cast-ok: legacy request_data is untyped; this override narrows it to string keys
            _GenericGuardrailApply, super().apply_guardrail
        )
        api_type = _aigateway_api_type(_request_route(request_data))
        request_data_with_api_type = _request_data_with_api_type(request_data, api_type, include_body=True)
        if input_type == "response":
            response_inputs = _response_text_inputs(inputs)
            if not response_inputs:
                return inputs
            try:
                reviewed_inputs = await parent_apply(
                    inputs=response_inputs,
                    request_data=request_data_with_api_type,
                    input_type=input_type,
                    logging_obj=logging_obj,
                )
            except GuardrailRaisedException as exc:
                raise _configured_guardrail_exception(exc, self.guardrail_name) from exc
            reviewed_response_inputs = {
                key: value
                for key, value in reviewed_inputs.items()
                if key in response_inputs or key == "stream_holdback_chars"
            }
            return {**inputs, **reviewed_response_inputs}
        try:
            return await parent_apply(
                inputs=inputs,
                request_data=request_data_with_api_type,
                input_type=input_type,
                logging_obj=logging_obj,
            )
        except GuardrailRaisedException as exc:
            raise _configured_guardrail_exception(exc, self.guardrail_name) from exc

    def get_guardrail_dynamic_request_body_params(self, request_data: dict[str, object]) -> dict[str, object]:
        parent_dynamic_params = cast(
            Callable[[dict[str, object]], dict[str, object]],
            super().get_guardrail_dynamic_request_body_params,
        )
        dynamic_params = parent_dynamic_params(request_data)
        api_type = request_data.get(_AIGATEWAY_API_TYPE_KEY)
        if not isinstance(api_type, str):
            return dynamic_params
        return {**dynamic_params, "api_type": api_type}

    async def async_post_call_streaming_iterator_hook(  # pyright: ignore[reportIncompatibleMethodOverride]  # the base hook only types chat chunks, but proxy hooks also receive native SSE bytes
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterable[StreamChunk],
        request_data: dict[str, object],
    ) -> AsyncGenerator[StreamChunk, None]:
        stream_api = _stream_api(user_api_key_dict.request_route)
        if stream_api is None:
            async for item in response:
                yield item
            return

        add_applied_guardrail = cast(  # cast-ok: shared helper uses legacy unparameterized Dict annotations
            _AddAppliedGuardrail, callback_utils.add_guardrail_to_applied_guardrails_header
        )
        add_applied_guardrail(request_data, self.guardrail_name)
        request_data_with_route = _request_data_with_authenticated_route(request_data, user_api_key_dict.request_route)
        request_data_with_api_type = _request_data_with_api_type(request_data_with_route, stream_api, include_body=True)
        stream_started = False
        response_sequence_tracker = _ResponseSequenceTracker() if stream_api == "responses" else None

        try:
            async for buffered_item in self._moderated_stream(
                response,
                request_data,
                request_data_with_api_type,
                stream_api,
                response_sequence_tracker,
            ):
                stream_started = True
                yield buffered_item
        except GuardrailRaisedException as exc:
            configured_exception = _configured_guardrail_exception(exc, self.guardrail_name)
            error_frame = _stream_error_frame(
                user_api_key_dict.request_route,
                configured_exception,
                None if response_sequence_tracker is None else response_sequence_tracker.last_sequence_number,
            )
            if error_frame is None or (not stream_started and stream_api != "anthropic_messages"):
                raise configured_exception from exc
            yield error_frame

    async def _moderated_stream(
        self,
        response: AsyncIterable[StreamChunk],
        request_data: dict[str, object],
        request_data_with_api_type: dict[str, object],
        stream_api: _StreamAPI,
        response_sequence_tracker: _ResponseSequenceTracker | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        response_iterator = response.__aiter__()
        extractor = _StreamTextExtractor()
        requested_choices = _requested_choice_count(request_data) if stream_api == "chat_completions" else 1
        review_state = _StreamReviewState()
        pending_chunks: _PendingStreamChunk | None = None
        pending_bytes = 0

        try:
            async for item in response_iterator:
                pending_chunks = _PendingStreamChunk(item, pending_chunks)
                pending_bytes += _stream_chunk_byte_size(item)
                extracted_batch = extractor.extract(item)
                review_state, text_is_approved = await self._review_extracted_batch(
                    extracted_batch,
                    review_state,
                    stream_api,
                    requested_choices,
                    request_data_with_api_type,
                    review_tail=False,
                )

                if _pending_stream_buffer_exceeded(pending_bytes, extractor, text_is_approved):
                    raise _stream_failure("Streaming response exceeded the maximum pending buffer size")

                if extractor.has_incomplete_sse_frame or not text_is_approved:
                    continue

                for buffered_item in _release_stream_chunks(pending_chunks, response_sequence_tracker):
                    yield buffered_item
                pending_chunks = None
                pending_bytes = 0
                review_state = self._review_state_after_release(review_state, requested_choices)

            trailing_batch = _finish_stream_extraction(extractor)
            review_state, text_is_approved = await self._review_extracted_batch(
                trailing_batch,
                review_state,
                stream_api,
                requested_choices,
                request_data_with_api_type,
                review_tail=True,
            )
            if not text_is_approved:
                raise _stream_failure("Response moderation did not cover the complete text response")
            for buffered_item in _release_stream_chunks(pending_chunks, response_sequence_tracker):
                yield buffered_item
        except _StreamingModerationFailure as exc:
            if self.fail_on_error:
                raise
            verbose_proxy_logger.warning(
                "AIGateway response moderation failed open; releasing buffered and remaining stream content",
                exc_info=exc,
            )
            for buffered_item in _release_stream_chunks(pending_chunks, None):
                yield buffered_item
            async for remaining_item in response_iterator:
                yield remaining_item

    async def _review_extracted_batch(
        self,
        batch: _ExtractedStreamBatch,
        state: _StreamReviewState,
        stream_api: _StreamAPI,
        requested_choices: int,
        request_data: dict[str, object],
        *,
        review_tail: bool,
    ) -> tuple[_StreamReviewState, bool]:
        updated_state = self._update_stream_review_state(batch, state, stream_api, requested_choices)
        await self._review_response_images(
            tuple(image for extracted in batch.frames for image in extracted.images),
            request_data,
        )
        if requested_choices == 1:
            reviewed_progress = await self._review_scheduled_text(
                updated_state.current_text,
                updated_state.review_progress,
                request_data,
                review_tail=review_tail,
            )
            reviewed_state = replace(updated_state, review_progress=reviewed_progress)
            return reviewed_state, reviewed_state.current_text == reviewed_progress.approved_text
        reviewed_choice_progress = await self._review_parallel_choices(
            updated_state.parallel_choice_text,
            updated_state.parallel_choice_progress,
            updated_state.finished_choice_indices,
            request_data,
            review_all_tails=review_tail,
        )
        text_is_approved = _parallel_choices_are_approved(
            updated_state.parallel_choice_text,
            reviewed_choice_progress,
        )
        review_advanced = any(
            progress.approved_text
            != _parallel_choice_progress(updated_state.parallel_choice_progress, choice_index).approved_text
            for choice_index, progress in reviewed_choice_progress
        )
        if review_advanced and not text_is_approved:
            reviewed_choice_progress = await self._review_parallel_choices(
                updated_state.parallel_choice_text,
                reviewed_choice_progress,
                updated_state.finished_choice_indices,
                request_data,
                review_all_tails=True,
            )
            text_is_approved = _parallel_choices_are_approved(
                updated_state.parallel_choice_text,
                reviewed_choice_progress,
            )
        reviewed_state = replace(updated_state, parallel_choice_progress=reviewed_choice_progress)
        return reviewed_state, text_is_approved

    def _update_stream_review_state(
        self,
        batch: _ExtractedStreamBatch,
        state: _StreamReviewState,
        stream_api: _StreamAPI,
        requested_choices: int,
    ) -> _StreamReviewState:
        if stream_api == "responses":
            current_text, review_progress = _response_batch_review_state(
                batch.frames,
                state.current_text,
                state.review_progress,
            )
            return replace(state, current_text=current_text, review_progress=review_progress)
        updated_state = state
        last_tool_call_frame_index = _last_tool_call_frame_index(batch.frames)
        for frame_index, extracted in enumerate(batch.frames):
            current_text, parallel_choice_text, finished_choice_indices = self._update_stream_text(
                extracted,
                requested_choices,
                updated_state.current_text,
                updated_state.parallel_choice_text,
                updated_state.finished_choice_indices,
            )
            updated_state = replace(
                updated_state,
                current_text=current_text,
                parallel_choice_text=parallel_choice_text,
                finished_choice_indices=finished_choice_indices,
            )
            if frame_index == last_tool_call_frame_index:
                updated_state = _StreamReviewState()
        return updated_state

    @staticmethod
    def _review_state_after_release(state: _StreamReviewState, requested_choices: int) -> _StreamReviewState:
        if requested_choices == 1:
            current_text, review_progress = _review_window_after_release(
                requested_choices,
                state.current_text,
                state.review_progress,
            )
            return replace(state, current_text=current_text, review_progress=review_progress)
        return replace(
            state,
            parallel_choice_text={},
            parallel_choice_progress=_compact_parallel_choice_progress(state.parallel_choice_progress),
        )

    @staticmethod
    def _update_stream_text(
        extracted: _ExtractedStreamText,
        requested_choices: int,
        current_text: str,
        parallel_choice_text: Mapping[int, str],
        finished_choice_indices: frozenset[int],
    ) -> tuple[str, dict[int, str], frozenset[int]]:
        if requested_choices == 1:
            if any(choice.index not in {None, 0} for choice in extracted.chat_choices):
                raise _stream_failure("Streaming response returned an unexpected parallel chat choice")
            return (
                current_text + extracted.text + "".join(choice.text for choice in extracted.chat_choices),
                dict(parallel_choice_text),
                finished_choice_indices,
            )
        if extracted.text:
            if extracted.chat_choices:
                raise _stream_failure("Streaming response mixed parallel chat choices with another stream schema")
            raise _stream_failure("Streaming response returned text without a choice index for parallel chat choices")
        updated_choice_text = dict(parallel_choice_text)
        updated_finished_choice_indices = finished_choice_indices
        for choice in extracted.chat_choices:
            if choice.index is None or choice.index < 0 or choice.index >= requested_choices:
                raise _stream_failure("Streaming response returned an invalid parallel chat choice index")
            updated_choice_text[choice.index] = updated_choice_text.get(choice.index, "") + choice.text
            if choice.is_finished:
                updated_finished_choice_indices = updated_finished_choice_indices | {choice.index}
        return current_text, updated_choice_text, updated_finished_choice_indices

    async def _review_parallel_choices(
        self,
        choice_text: Mapping[int, str],
        choice_progress: tuple[tuple[int, _ReviewProgress], ...],
        finished_choice_indices: frozenset[int],
        request_data: dict[str, object],
        *,
        review_all_tails: bool,
    ) -> tuple[tuple[int, _ReviewProgress], ...]:
        choice_items = tuple(sorted(choice_text.items()))
        reviewed_progress = await asyncio.gather(
            *(
                self._review_scheduled_text(
                    text,
                    _parallel_choice_progress(choice_progress, index),
                    request_data,
                    review_tail=review_all_tails or index in finished_choice_indices,
                )
                for index, text in choice_items
            )
        )
        inactive_progress = tuple((index, progress) for index, progress in choice_progress if index not in choice_text)
        active_progress = tuple(
            (choice_item[0], progress) for choice_item, progress in zip(choice_items, reviewed_progress)
        )
        return tuple(sorted((*inactive_progress, *active_progress)))

    async def _review_scheduled_text(
        self,
        current_text: str,
        progress: _ReviewProgress,
        request_data: dict[str, object],
        *,
        review_tail: bool,
    ) -> _ReviewProgress:
        plan = _review_plan(
            current_text,
            progress,
            review_tail=review_tail,
            chunk_size=self.streaming_review_chunk_size,
        )
        approved_text = plan.base_approved_text
        previous_context = plan.previous_context
        for segment in plan.segments:
            await self._review_response_text(previous_context + segment, request_data)
            approved_text += segment
            reviewed_text = previous_context + segment
            previous_context = (
                reviewed_text[-self.streaming_review_context_size :] if self.streaming_review_context_size else ""
            )
        return _ReviewProgress(approved_text, previous_context)

    async def _review_response_text(
        self,
        text: str,
        request_data: dict[str, object],
    ) -> None:
        await self._review_response_inputs({"texts": [text]}, request_data)

    async def _review_response_images(
        self,
        images: tuple[str, ...],
        request_data: dict[str, object],
    ) -> None:
        if images:
            await self._review_response_inputs({"images": list(images)}, request_data)

    async def _review_response_inputs(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
    ) -> None:
        model = request_data.get("model")
        if isinstance(model, str):
            inputs = {**inputs, "model": model}
        try:
            await asyncio.wait_for(
                self.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="response",
                    logging_obj=_request_logging_obj(request_data),
                ),
                timeout=self.decision_timeout_seconds,
            )
        except GuardrailRaisedException:
            raise
        except asyncio.TimeoutError as exc:
            if not self.fail_on_error:
                verbose_proxy_logger.warning(
                    "AIGateway response moderation timed out; allowing content because fail_on_error is false"
                )
                return
            raise _stream_failure("Response moderation timed out") from exc
        except Exception as exc:
            if not self.fail_on_error:
                verbose_proxy_logger.warning(
                    "AIGateway response moderation failed; allowing content because fail_on_error is false",
                    exc_info=exc,
                )
                return
            raise _stream_failure("Response moderation failed") from exc
