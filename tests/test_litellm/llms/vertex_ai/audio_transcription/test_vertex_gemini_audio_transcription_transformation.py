import base64
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from pytest_mock import MockerFixture

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.audio_transcription.gemini_transformation import (
    VertexGeminiAudioTranscriptionConfig,
)
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    VertexAIAudioTranscriptionConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params_transcription


@pytest.fixture
def config() -> VertexGeminiAudioTranscriptionConfig:
    return VertexGeminiAudioTranscriptionConfig()


class TestProviderRouting:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.5-transcribe-preview",
            "gemini-4-transcribe-preview",
        ],
    )
    def test_vertex_gemini_transcription_models_use_generate_content_config(self, model):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexGeminiAudioTranscriptionConfig)

    def test_vertex_non_transcription_gemini_models_keep_speech_v2_config(self):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="gemini-3.5-flash",
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexAIAudioTranscriptionConfig)

    def test_vertex_non_gemini_models_keep_speech_v2_config(self):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="chirp_3",
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexAIAudioTranscriptionConfig)

    def test_vertex_ai_beta_is_not_supported(self):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="gemini-3.5-transcribe-preview",
            provider=LlmProviders.VERTEX_AI_BETA,
        )
        assert provider_config is None

    def test_optional_params_do_not_forward_model_as_provider_param(self):
        optional_params = get_optional_params_transcription(
            model="gemini-3.5-transcribe-preview",
            custom_llm_provider="vertex_ai",
            language="zh",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            diarization=True,
            custom_vocabulary=["LiteLLM"],
        )
        assert optional_params["language"] == "zh"
        assert optional_params["response_format"] == "verbose_json"
        assert optional_params["timestamp_granularities"] == ["word"]
        assert optional_params["diarization"] is True
        assert optional_params["custom_vocabulary"] == ["LiteLLM"]


class TestGetCompleteUrl:
    def test_transcribe_model_defaults_to_global(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert url == (
            "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/"
            "publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
        )

    def test_configured_location_is_preserved(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3.5-flash",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": "us-central1"},
        )
        assert url.startswith("https://us-central1-aiplatform.googleapis.com/v1beta1/")
        assert "/locations/us-central1/" in url

    def test_custom_api_base_with_version_is_not_duplicated(self, config):
        url = config.get_complete_url(
            api_base="https://vertex.example.test/v1beta1",
            api_key=None,
            model="gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": "global"},
        )
        assert url.startswith("https://vertex.example.test/v1beta1/projects/test-project/")

    def test_live_model_is_rejected(self, config):
        with pytest.raises(litellm.UnsupportedParamsError, match="cannot be used with /audio/transcriptions"):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="gemini-3.5-transcribe-live-preview",
                optional_params={},
                litellm_params={"vertex_project": "test-project", "vertex_location": "global"},
            )


class TestTransformRequest:
    def test_openai_params_and_extensions_map_to_generate_content(self, config):
        audio_bytes = b"fake-audio-bytes"
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=("meeting.wav", audio_bytes, "audio/wav"),
            optional_params={
                "language": "zh",
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
                "diarization": True,
                "custom_vocabulary": ["LiteLLM", "Vertex AI"],
            },
            litellm_params={},
        )
        assert request_data.files is None
        assert request_data.content_type == "application/json"
        assert request_data.data == {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        }
                    ],
                }
            ],
            "generationConfig": {
                "audioTranscriptionConfig": {
                    "languageCodes": ["zh-CN"],
                    "customVocabulary": ["LiteLLM", "Vertex AI"],
                    "wordTimestamp": True,
                    "diarization": True,
                }
            },
        }

    def test_default_request_has_no_transcription_prompt(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=b"fake-audio-bytes",
            optional_params={},
            litellm_params={},
        )
        assert request_data.data["contents"][0]["parts"] == [
            {
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": base64.b64encode(b"fake-audio-bytes").decode("ascii"),
                }
            }
        ]
        assert request_data.data["generationConfig"] == {"audioTranscriptionConfig": {}}

    def test_smart_mode_rejects_word_timestamps(self, config):
        with pytest.raises(litellm.UnsupportedParamsError, match="SMART"):
            config.transform_audio_transcription_request(
                model="gemini-3.5-transcribe-preview",
                audio_file=b"fake-audio-bytes",
                optional_params={"mode": "smart", "timestamp_granularities": ["word"]},
                litellm_params={},
            )


class TestOptionalParams:
    @pytest.mark.parametrize("response_format", ["json", "text", "verbose_json"])
    def test_supported_response_formats(self, response_format):
        optional_params = get_optional_params_transcription(
            model="gemini-3.5-transcribe-preview",
            custom_llm_provider="vertex_ai",
            response_format=response_format,
        )
        assert optional_params["response_format"] == response_format

    @pytest.mark.parametrize("response_format", ["srt", "vtt"])
    def test_unsupported_response_formats(self, response_format):
        with pytest.raises(litellm.UnsupportedParamsError, match="response_format"):
            get_optional_params_transcription(
                model="gemini-3.5-transcribe-preview",
                custom_llm_provider="vertex_ai",
                response_format=response_format,
            )

    def test_segment_timestamps_are_rejected(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="segment timestamps"):
            get_optional_params_transcription(
                model="gemini-3.5-transcribe-preview",
                custom_llm_provider="vertex_ai",
                timestamp_granularities=["segment"],
            )

    def test_prompt_is_rejected(self):
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params_transcription(
                model="gemini-3.5-transcribe-preview",
                custom_llm_provider="vertex_ai",
                prompt="Transcribe this audio",
            )


class TestTransformResponse:
    def test_text_transcription_metadata_and_words(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Hello ",
                                    "audioTranscription": {
                                        "text": "Hello ",
                                        "languageCode": "en-US",
                                        "speakerLabel": "spk_1",
                                        "words": [{"word": "Hello", "startOffset": "0s", "endOffset": "0.5s"}],
                                    },
                                },
                                {
                                    "audioTranscription": {
                                        "text": "world.",
                                        "languageCode": "en-US",
                                        "speakerLabel": "spk_2",
                                        "words": [{"word": "world", "startOffset": "0.5s", "endOffset": "1.2s"}],
                                    }
                                },
                            ]
                        }
                    }
                ]
            },
        )
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == "Hello world."
        assert response["task"] == "transcribe"
        assert response["language"] == "en-US"
        assert response["words"] == (
            {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "spk_1"},
            {"word": "world", "start": 0.5, "end": 1.2, "speaker": "spk_2"},
        )

    def test_successful_response_preserves_raw_vertex_metadata(self, config):
        raw_payload = {
            "responseId": "response-123",
            "modelVersion": "gemini-3.5-transcribe-preview",
            "candidates": [{"content": {"parts": [{"audioTranscription": {"text": "Hello world."}}]}}],
            "usageMetadata": {
                "promptTokenCount": 70,
                "candidatesTokenCount": 12,
                "totalTokenCount": 82,
                "providerExtension": {"billingClass": "audio"},
            },
        }

        response = config.transform_audio_transcription_response(httpx.Response(status_code=200, json=raw_payload))

        hidden_params: Final = getattr(response, "_hidden_params", None)
        assert hidden_params == raw_payload

    def test_usage_metadata_maps_audio_and_output_tokens(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Async transcript."}]}}],
                "usageMetadata": {
                    "promptTokenCount": 70,
                    "candidatesTokenCount": 12,
                    "totalTokenCount": 82,
                    "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 70}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 12}],
                },
            },
        )

        response = config.transform_audio_transcription_response(raw_response)

        assert response.usage is not None
        assert response.usage.model_dump() == {
            "type": "tokens",
            "input_tokens": 70,
            "output_tokens": 12,
            "total_tokens": 82,
            "input_token_details": {"audio_tokens": 70, "text_tokens": 0},
        }

        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/gemini-3.5-transcribe-preview": {
                    "input_cost_per_audio_token": 2.5e-06,
                    "input_cost_per_token": 2.5e-06,
                    "litellm_provider": "vertex_ai",
                    "mode": "audio_transcription",
                    "output_cost_per_token": 1.2e-05,
                }
            },
            clear=False,
        ):
            cost = litellm.completion_cost(
                completion_response=response,
                model="vertex_ai/gemini-3.5-transcribe-preview",
                custom_llm_provider="vertex_ai",
                call_type="atranscription",
            )

        assert cost == pytest.approx(70 * 2.5e-06 + 12 * 1.2e-05)

    def test_empty_candidates_return_empty_text(self, config):
        response = config.transform_audio_transcription_response(httpx.Response(status_code=200, json={}))
        assert response.text == ""
        assert response.usage is None

    def test_prompt_feedback_block_raises_content_policy_violation(self, config):
        raw_response = httpx.Response(
            status_code=200,
            request=httpx.Request(
                method="POST",
                url=(
                    "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/"
                    "publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
                ),
            ),
            headers={"x-request-id": "blocked-request"},
            json={
                "promptFeedback": {
                    "blockReason": "PROHIBITED_CONTENT",
                    "blockReasonMessage": "The audio is blocked due to prohibited content",
                },
                "modelVersion": "gemini-3.5-transcribe-preview",
                "usageMetadata": {
                    "promptTokenCount": 70,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 70,
                },
            },
        )

        with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
            config.transform_audio_transcription_response(raw_response)

        error = exc_info.value
        assert error.status_code == 400
        assert error.model == "gemini-3.5-transcribe-preview"
        assert error.llm_provider == "vertex_ai"
        assert "PROHIBITED_CONTENT" in str(error)
        assert "The audio is blocked due to prohibited content" in str(error)
        assert error.response.headers["x-request-id"] == "blocked-request"
        assert error.provider_specific_fields == {
            "block_reason": "PROHIBITED_CONTENT",
            "block_reason_message": "The audio is blocked due to prohibited content",
        }
        assert error.body == {
            "message": (
                "Vertex Gemini transcription blocked with PROHIBITED_CONTENT: "
                "The audio is blocked due to prohibited content"
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": "content_policy_violation",
        }

    def test_candidate_finish_reason_block_raises_content_policy_violation(self, config):
        raw_response = httpx.Response(
            status_code=200,
            request=httpx.Request(
                method="POST",
                url="https://aiplatform.googleapis.com/v1beta1/generateContent",
            ),
            json={
                "candidates": [
                    {
                        "finishReason": "SAFETY",
                        "finishMessage": "The output was blocked by safety filters",
                    }
                ],
                "modelVersion": "gemini-3.5-transcribe-preview",
            },
        )

        with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
            config.transform_audio_transcription_response(raw_response)

        error = exc_info.value
        assert error.status_code == 400
        assert error.model == "gemini-3.5-transcribe-preview"
        assert error.llm_provider == "vertex_ai"
        assert "SAFETY" in str(error)
        assert "The output was blocked by safety filters" in str(error)
        assert error.provider_specific_fields == {
            "block_reason": "SAFETY",
            "block_reason_message": "The output was blocked by safety filters",
        }


@patch.object(VertexGeminiAudioTranscriptionConfig, "_ensure_access_token")
def test_litellm_transcription_sends_vertex_generate_content(mock_ensure_token):
    mock_ensure_token.return_value = ("mock-token", "test-project")
    client = HTTPHandler()
    raw_response = httpx.Response(
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"audioTranscription": {"text": "Hello world."}}]}}]},
    )

    with patch.object(client, "post", return_value=raw_response) as mock_post:
        response = litellm.transcription(
            model="vertex_ai/gemini-3.5-transcribe-preview",
            file=("meeting.wav", b"fake-audio-bytes", "audio/wav"),
            language="en",
            vertex_project="test-project",
            vertex_location="global",
            client=client,
        )

    assert response.text == "Hello world."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == (
        "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/"
        "publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
    )
    assert call_kwargs["headers"]["Authorization"] == "Bearer mock-token"
    assert call_kwargs["json"]["generationConfig"]["audioTranscriptionConfig"]["languageCodes"] == ["en-US"]
    assert "text" not in call_kwargs["json"]["contents"][0]["parts"][0]


@pytest.mark.asyncio
async def test_litellm_atranscription_sends_vertex_generate_content(mocker: MockerFixture):
    mocker.patch.object(
        VertexGeminiAudioTranscriptionConfig,
        "_ensure_access_token",
        return_value=("mock-token", "test-project"),
    )
    client = AsyncHTTPHandler()
    raw_response = httpx.Response(
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"audioTranscription": {"text": "Async transcript."}}]}}]},
    )
    mock_post = mocker.patch.object(client, "post", return_value=raw_response)

    try:
        response = await litellm.atranscription(
            model="vertex_ai/gemini-3.5-transcribe-preview",
            file=("meeting.wav", b"fake-audio-bytes", "audio/wav"),
            vertex_project="test-project",
            vertex_location="global",
            client=client,
        )
    finally:
        await client.close()

    assert response.text == "Async transcript."
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["generationConfig"] == {"audioTranscriptionConfig": {}}
