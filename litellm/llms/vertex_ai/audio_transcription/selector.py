from typing import Final

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)


def get_vertex_audio_transcription_config(model: str) -> BaseAudioTranscriptionConfig:
    normalized_model: Final = model.removeprefix("vertex_ai/").lower()
    if normalized_model.startswith("gemini-") and "transcribe" in normalized_model:
        from litellm.llms.vertex_ai.audio_transcription.gemini_transformation import (
            VertexGeminiAudioTranscriptionConfig,
        )

        return VertexGeminiAudioTranscriptionConfig()

    from litellm.llms.vertex_ai.audio_transcription.transformation import (
        VertexAIAudioTranscriptionConfig,
    )

    return VertexAIAudioTranscriptionConfig()
