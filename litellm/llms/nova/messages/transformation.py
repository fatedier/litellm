"""Anthropic Messages configuration for the Nova downstream gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders


class NovaMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    """Forward Anthropic Messages requests to a Nova gateway without translation."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.NOVA

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[dict[str, object]],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:
        parent_validate: Final = cast(
            Callable[..., tuple[dict[str, str], str | None]],
            OpenAILikeAnthropicMessagesConfig.validate_anthropic_messages_environment,
        )
        return parent_validate(
            self,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key or get_secret_str("NOVA_API_KEY"),
            api_base=api_base or get_secret_str("NOVA_API_BASE"),
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        parent_get_complete_url: Final = cast(
            Callable[..., str],
            OpenAILikeAnthropicMessagesConfig.get_complete_url,
        )
        return parent_get_complete_url(
            self,
            api_base=api_base or get_secret_str("NOVA_API_BASE"),
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def should_strip_billing_metadata(self) -> bool:
        return True
