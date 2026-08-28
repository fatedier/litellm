"""Responses API configuration for the Nova downstream gateway."""

from __future__ import annotations

from typing import Final

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class NovaResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Forward Responses API requests to a Nova gateway."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.NOVA

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:
        params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = params.api_key or get_secret_str("NOVA_API_KEY")
        headers.setdefault("Content-Type", "application/json")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("NOVA_API_BASE")
        if resolved_api_base is None:
            raise ValueError(
                "api_base not set for Nova Responses API. Set via api_base parameter or NOVA_API_BASE environment variable"
            )
        normalized_api_base: Final = resolved_api_base.rstrip("/")
        if normalized_api_base.endswith("/responses"):
            return normalized_api_base
        return f"{normalized_api_base}/responses"

    def supports_native_websocket(self) -> bool:
        return False
