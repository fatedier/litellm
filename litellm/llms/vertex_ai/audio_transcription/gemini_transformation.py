import base64
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

from httpx import Headers, Request, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import ContentPolicyViolationError, UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import (
    normalize_transcription_language_to_bcp47,
    process_audio_file,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.vertex_ai.common_utils import (
    VertexAIError,
    get_vertex_base_url,
    validate_vertex_location,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)

_SUPPORTED_RESPONSE_FORMATS: Final = ("json", "text", "verbose_json")
_URL_UNSAFE_PROJECT_CHARS: Final = ("/", "?", "#", "\\", ":", " ", "\t", "\n", "\r")
_BOOLEAN_STRINGS: Final = frozenset(("true", "false"))
_TRANSCRIPTION_MODES: Final = frozenset(("VERBATIM", "SMART"))
_CONTENT_POLICY_FINISH_REASONS: Final = frozenset(
    (
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
    )
)
_STRING_SEQUENCE_ADAPTER: Final = TypeAdapter(tuple[str, ...])
_TIMESTAMP_GRANULARITIES_ADAPTER: Final = TypeAdapter(tuple[Literal["word", "segment"], ...])


class _VertexGeminiAudioTranscriptionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    language_codes: tuple[str, ...] | None = Field(default=None, serialization_alias="languageCodes")
    custom_vocabulary: tuple[str, ...] | None = Field(default=None, serialization_alias="customVocabulary")
    word_timestamp: bool | None = Field(default=None, serialization_alias="wordTimestamp")
    diarization: bool | None = None
    mode: Literal["VERBATIM", "SMART"] | None = None


class _VertexGeminiGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_transcription_config: _VertexGeminiAudioTranscriptionConfig = Field(
        serialization_alias="audioTranscriptionConfig"
    )


class _VertexGeminiInlineData(BaseModel):
    model_config = ConfigDict(frozen=True)

    mime_type: str = Field(serialization_alias="mimeType")
    data: str


class _VertexGeminiInputPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    inline_data: _VertexGeminiInlineData = Field(serialization_alias="inlineData")


class _VertexGeminiInputContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    parts: tuple[_VertexGeminiInputPart, ...]


class _VertexGeminiGenerateContentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contents: tuple[_VertexGeminiInputContent, ...]
    generation_config: _VertexGeminiGenerationConfig = Field(serialization_alias="generationConfig")


class _VertexGeminiWord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    word: str | None = None
    start_offset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("startOffset", "start_offset"),
    )
    end_offset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("endOffset", "end_offset"),
    )


class _VertexGeminiTranscription(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str | None = None
    language_code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("languageCode", "language_code"),
    )
    speaker_label: str | None = Field(
        default=None,
        validation_alias=AliasChoices("speakerLabel", "speaker_label"),
    )
    words: tuple[_VertexGeminiWord, ...] = ()


class _VertexGeminiOutputPart(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str | None = None
    audio_transcription: _VertexGeminiTranscription | None = Field(
        default=None,
        validation_alias=AliasChoices("audioTranscription", "audio_transcription"),
    )


class _VertexGeminiOutputContent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    parts: tuple[_VertexGeminiOutputPart, ...] = ()


class _VertexGeminiCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: _VertexGeminiOutputContent | None = None
    finish_reason: str | None = Field(
        default=None,
        validation_alias=AliasChoices("finishReason", "finish_reason"),
    )
    finish_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices("finishMessage", "finish_message"),
    )


class _VertexGeminiUsageMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_token_count: int = Field(
        default=0,
        validation_alias=AliasChoices("promptTokenCount", "prompt_token_count"),
    )
    candidates_token_count: int = Field(
        default=0,
        validation_alias=AliasChoices("candidatesTokenCount", "candidates_token_count"),
    )
    total_token_count: int = Field(
        default=0,
        validation_alias=AliasChoices("totalTokenCount", "total_token_count"),
    )


class _VertexGeminiPromptFeedback(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    block_reason: str | None = Field(
        default=None,
        validation_alias=AliasChoices("blockReason", "block_reason"),
    )
    block_reason_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices("blockReasonMessage", "block_reason_message"),
    )


class _VertexGeminiGenerateContentResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    candidates: tuple[_VertexGeminiCandidate, ...] = ()
    model_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelVersion", "model_version"),
    )
    prompt_feedback: _VertexGeminiPromptFeedback | None = Field(
        default=None,
        validation_alias=AliasChoices("promptFeedback", "prompt_feedback"),
    )
    usage_metadata: _VertexGeminiUsageMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("usageMetadata", "usage_metadata"),
    )


class _OpenAITranscriptionWord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


class VertexGeminiAudioTranscriptionConfig(BaseAudioTranscriptionConfig, VertexBase):
    def __init__(self) -> None:
        BaseAudioTranscriptionConfig.__init__(self)
        VertexBase.__init__(self)

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: abstract provider contract returns list
        return ["language", "response_format", "timestamp_granularities"]  # mutable-ok: provider contract

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: abstract provider contract returns request params dict
        self._raise_if_live_model(model)
        response_format: Final = non_default_params.get("response_format")
        unsupported_response_format: Final = (
            response_format is not None and response_format not in _SUPPORTED_RESPONSE_FORMATS
        )
        if unsupported_response_format and not (drop_params or litellm.drop_params):
            raise UnsupportedParamsError(
                message=(
                    f"Vertex Gemini audio transcription does not support response_format={response_format!r}. "
                    f"Supported values: {', '.join(_SUPPORTED_RESPONSE_FORMATS)}"
                ),
                model=model,
                llm_provider="vertex_ai",
            )
        params_after_response_format: Final[Mapping[str, object]] = (
            MappingProxyType({key: value for key, value in non_default_params.items() if key != "response_format"})
            if unsupported_response_format
            else non_default_params
        )

        granularities: Final = self._parse_timestamp_granularities(
            params_after_response_format.get("timestamp_granularities")
        )
        if "segment" in granularities and not (drop_params or litellm.drop_params):
            raise UnsupportedParamsError(
                message=(
                    "Vertex Gemini file transcription supports word timestamps, not segment timestamps. "
                    "Use timestamp_granularities=['word']"
                ),
                model=model,
                llm_provider="vertex_ai",
            )
        params_without_granularities: Final = MappingProxyType(
            {key: value for key, value in params_after_response_format.items() if key != "timestamp_granularities"}
        )
        mapped_params: Final[Mapping[str, object]] = (
            MappingProxyType({**params_without_granularities, "timestamp_granularities": ("word",)})
            if "segment" in granularities and "word" in granularities and (drop_params or litellm.drop_params)
            else params_without_granularities
            if "segment" in granularities and (drop_params or litellm.drop_params)
            else params_after_response_format
        )

        return {**optional_params, **mapped_params}  # mutable-ok: abstract provider contract returns dict

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | Headers,  # mutable-ok: abstract provider error contract
    ) -> BaseLLMException:
        return VertexAIError(status_code=status_code, message=error_message, headers=headers)

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: dict[str, object],  # mutable-ok: VertexBase helper has a legacy dict contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: HTTP handler mutates provider headers
        access_token, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(  # pyright: ignore[reportUnknownMemberType]  # legacy helper
                litellm_params
            ),
            project_id=self.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # legacy helper
                litellm_params
            ),
            custom_llm_provider="vertex_ai",
        )
        return {  # mutable-ok: HTTP handler requires mutable headers
            **headers,
            "Authorization": f"Bearer {access_token}",
            "x-goog-user-project": project_id,
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: dict[str, object],  # mutable-ok: VertexBase helper has a legacy dict contract
        stream: bool | None = None,
    ) -> str:
        normalized_model: Final = self._normalize_model(model)
        self._raise_if_live_model(normalized_model)
        configured_location: Final = self.safe_get_vertex_ai_location(  # pyright: ignore[reportUnknownMemberType]  # legacy helper
            litellm_params
        )
        default_location: Final = (
            "global" if "transcribe" in normalized_model.lower() else self.get_default_vertex_location()
        )
        try:
            location: Final = validate_vertex_location(configured_location or default_location)
        except ValueError as error:
            raise VertexAIError(status_code=400, message=str(error)) from error
        project_id: Final = self._validate_project_id(
            self.safe_get_vertex_ai_project(  # pyright: ignore[reportUnknownMemberType]  # legacy helper
                litellm_params
            )
            or self._resolve_project_id_from_credentials(litellm_params)
        )
        base_url: Final = (api_base or get_vertex_base_url(location)).rstrip("/")
        versioned_base_url: Final = base_url if base_url.endswith("/v1beta1") else f"{base_url}/v1beta1"
        encoded_model: Final = encode_url_path_segment(normalized_model, field_name="Vertex Gemini model")
        return (
            f"{versioned_base_url}/projects/{project_id}/locations/{location}/publishers/google/models/"
            f"{encoded_model}:generateContent"
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        self._raise_if_live_model(model)
        processed_audio: Final = process_audio_file(audio_file)
        language: Final = optional_params.get("language")
        language_codes: Final = (
            (normalize_transcription_language_to_bcp47(language),) if isinstance(language, str) and language else None
        )
        granularities: Final = self._parse_timestamp_granularities(optional_params.get("timestamp_granularities"))
        word_timestamp: Final = "word" in granularities
        diarization: Final = self._parse_optional_bool(optional_params.get("diarization"), "diarization")
        custom_vocabulary: Final = self._parse_optional_string_sequence(optional_params.get("custom_vocabulary"))
        mode: Final = self._parse_mode(optional_params.get("mode"))
        if mode == "SMART" and (word_timestamp or diarization):
            raise UnsupportedParamsError(
                message="Vertex Gemini SMART transcription mode cannot be combined with timestamps or diarization",
                model=model,
                llm_provider="vertex_ai",
            )

        request: Final = _VertexGeminiGenerateContentRequest(
            contents=(
                _VertexGeminiInputContent(
                    parts=(
                        _VertexGeminiInputPart(
                            inline_data=_VertexGeminiInlineData(
                                mime_type=processed_audio.content_type,
                                data=base64.b64encode(processed_audio.file_content).decode("ascii"),
                            )
                        ),
                    )
                ),
            ),
            generation_config=_VertexGeminiGenerationConfig(
                audio_transcription_config=_VertexGeminiAudioTranscriptionConfig(
                    language_codes=language_codes,
                    custom_vocabulary=custom_vocabulary,
                    word_timestamp=True if word_timestamp else None,
                    diarization=diarization,
                    mode=mode,
                )
            ),
        )
        return AudioTranscriptionRequestData(
            data=request.model_dump(mode="json", by_alias=True, exclude_none=True),
            content_type="application/json",
        )

    def transform_audio_transcription_response(self, raw_response: Response) -> TranscriptionResponse:
        try:
            parsed: Final = _VertexGeminiGenerateContentResponse.model_validate_json(raw_response.text)
        except ValidationError as error:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Invalid Vertex Gemini transcription response: {error}",
                headers=raw_response.headers,
            ) from error

        self._raise_if_blocked(
            reason=(parsed.prompt_feedback.block_reason if parsed.prompt_feedback is not None else None),
            provider_message=(
                parsed.prompt_feedback.block_reason_message if parsed.prompt_feedback is not None else None
            ),
            model=parsed.model_version or "gemini-transcription",
            raw_response=raw_response,
        )
        candidate: Final = parsed.candidates[0] if parsed.candidates else None
        if candidate is not None and candidate.finish_reason in _CONTENT_POLICY_FINISH_REASONS:
            self._raise_if_blocked(
                reason=candidate.finish_reason,
                provider_message=candidate.finish_message,
                model=parsed.model_version or "gemini-transcription",
                raw_response=raw_response,
            )
        parts: Final = candidate.content.parts if candidate is not None and candidate.content else ()
        text_fragments: Final[tuple[str, ...]] = tuple(
            text
            for part in parts
            if (text := part.text or (part.audio_transcription.text if part.audio_transcription else None)) is not None
        )
        transcriptions: Final = tuple(
            part.audio_transcription for part in parts if part.audio_transcription is not None
        )
        languages: Final = frozenset(
            transcription.language_code for transcription in transcriptions if transcription.language_code
        )
        words: Final = tuple(
            _OpenAITranscriptionWord(
                word=word.word or "",
                start=self._parse_duration_seconds(word.start_offset),
                end=self._parse_duration_seconds(word.end_offset),
                speaker=transcription.speaker_label,
            ).model_dump(exclude_none=True)
            for transcription in transcriptions
            for word in transcription.words
        )
        response: Final = TranscriptionResponse(text="".join(text_fragments))
        response["task"] = "transcribe"
        usage: Final = self._transform_usage_metadata(parsed.usage_metadata)
        if usage is not None:
            response["usage"] = usage
        if len(languages) == 1:
            response["language"] = next(iter(languages))
        if words:
            response["words"] = words
        setattr(  # noqa: B010  # preserve provider metadata without exposing a private attribute access
            response, "_hidden_params", raw_response.json()
        )
        return response

    @staticmethod
    def _transform_usage_metadata(
        usage_metadata: _VertexGeminiUsageMetadata | None,
    ) -> TranscriptionUsageTokensObject | None:
        if usage_metadata is None:
            return None
        total_tokens: Final = usage_metadata.total_token_count or (
            usage_metadata.prompt_token_count + usage_metadata.candidates_token_count
        )
        return TranscriptionUsageTokensObject(
            type="tokens",
            input_tokens=usage_metadata.prompt_token_count,
            output_tokens=usage_metadata.candidates_token_count,
            total_tokens=total_tokens,
            input_token_details=TranscriptionUsageInputTokenDetailsObject(
                audio_tokens=usage_metadata.prompt_token_count,
                text_tokens=0,
            ),
        )

    @staticmethod
    def _raise_if_blocked(
        reason: str | None,
        provider_message: str | None,
        model: str,
        raw_response: Response,
    ) -> None:
        if reason is None:
            return
        message: Final = (
            f"Vertex Gemini transcription blocked with {reason}: {provider_message}"
            if provider_message
            else f"Vertex Gemini transcription blocked with {reason}"
        )
        error_response: Final = Response(
            status_code=400,
            headers=raw_response.headers,
            content=raw_response.content,
            request=Request(method="POST", url="https://aiplatform.googleapis.com/"),
        )
        raise ContentPolicyViolationError(
            message=message,
            model=model,
            llm_provider="vertex_ai",
            response=error_response,
            provider_specific_fields={  # mutable-ok: exception contract requires dict
                "block_reason": reason,
                "block_reason_message": provider_message,
            },
            body={  # mutable-ok: OpenAI exception contract requires dict
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "content_policy_violation",
            },
        )

    @staticmethod
    def _normalize_model(model: str) -> str:
        return model.removeprefix("vertex_ai/")

    @classmethod
    def _raise_if_live_model(cls, model: str) -> None:
        normalized_model: Final = cls._normalize_model(model).lower()
        if "-live" in normalized_model:
            raise UnsupportedParamsError(
                message=f"Vertex Gemini Live model {normalized_model!r} cannot be used with /audio/transcriptions",
                model=normalized_model,
                llm_provider="vertex_ai",
            )

    @staticmethod
    def _validate_project_id(project_id: str) -> str:
        if not project_id or ".." in project_id or any(char in project_id for char in _URL_UNSAFE_PROJECT_CHARS):
            raise VertexAIError(status_code=400, message=f"Invalid vertex_project format: {project_id!r}")
        return project_id

    def _resolve_project_id_from_credentials(
        self,
        litellm_params: dict[str, object],  # mutable-ok: VertexBase helper has a legacy dict contract
    ) -> str:
        _, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(  # pyright: ignore[reportUnknownMemberType]  # legacy helper
                litellm_params
            ),
            project_id=None,
            custom_llm_provider="vertex_ai",
        )
        return project_id

    @staticmethod
    def _parse_timestamp_granularities(value: object) -> tuple[Literal["word", "segment"], ...]:
        if value is None:
            return ()
        try:
            return _TIMESTAMP_GRANULARITIES_ADAPTER.validate_python(value)
        except ValidationError as error:
            raise UnsupportedParamsError(
                message="timestamp_granularities must contain only 'word' or 'segment'",
                llm_provider="vertex_ai",
            ) from error

    @staticmethod
    def _parse_optional_bool(value: object, field_name: str) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in _BOOLEAN_STRINGS:
            return value.lower() == "true"
        raise UnsupportedParamsError(
            message=f"{field_name} must be a boolean",
            llm_provider="vertex_ai",
        )

    @staticmethod
    def _parse_optional_string_sequence(value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                try:
                    return _STRING_SEQUENCE_ADAPTER.validate_json(value)
                except ValidationError as error:
                    raise UnsupportedParamsError(
                        message="custom_vocabulary must be a string or a list of strings",
                        llm_provider="vertex_ai",
                    ) from error
            return (value,)
        try:
            return _STRING_SEQUENCE_ADAPTER.validate_python(value)
        except ValidationError as error:
            raise UnsupportedParamsError(
                message="custom_vocabulary must be a string or a list of strings",
                llm_provider="vertex_ai",
            ) from error

    @staticmethod
    def _parse_mode(value: object) -> Literal["VERBATIM", "SMART"] | None:
        if value is None:
            return None
        if isinstance(value, str) and value.upper() in _TRANSCRIPTION_MODES:
            return "SMART" if value.upper() == "SMART" else "VERBATIM"
        raise UnsupportedParamsError(
            message="mode must be 'VERBATIM' or 'SMART'",
            llm_provider="vertex_ai",
        )

    @staticmethod
    def _parse_duration_seconds(value: str | None) -> float | None:
        if value is None or not value.endswith("s"):
            return None
        try:
            return float(value[:-1])
        except ValueError:
            return None
