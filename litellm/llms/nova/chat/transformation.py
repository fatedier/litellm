"""Chat Completions configuration for the Nova downstream gateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, cast

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class NovaChatConfig(OpenAIGPTConfig):
    """Forward Chat Completions requests to a Nova gateway."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.NOVA

    def get_supported_openai_params(self, model: str) -> list[str]:
        parent_get_supported_params: Final = cast(
            Callable[[OpenAIGPTConfig, str], list[str]],
            OpenAIGPTConfig.get_supported_openai_params,
        )
        params_list: Final[list[str]] = [
            *parent_get_supported_params(self, model),
            *OPENAI_CHAT_COMPLETION_PARAMS,
        ]
        return params_list

    def _map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_openai_params: Final = self.get_supported_openai_params(model)
        mapped_params: Final[dict[str, object]] = {
            **optional_params,
            **{
                param: value
                for param, value in non_default_params.items()
                if param in supported_openai_params and param != "thinking"
            },
        }
        thinking: Final[object | None] = non_default_params.get("thinking")
        if thinking is None:
            return mapped_params
        existing_extra_body: Final[object] = optional_params.get("extra_body")
        extra_body_source: Final[dict[str, object]] = (
            cast(dict[str, object], existing_extra_body) if isinstance(existing_extra_body, dict) else {}
        )
        extra_body: Final[dict[str, object]] = {
            **extra_body_source,
            "thinking": thinking,
        }
        return {**mapped_params, "extra_body": extra_body}

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("NOVA_API_BASE")
        resolved_api_key: Final = api_key or get_secret_str("NOVA_API_KEY")
        return resolved_api_base, resolved_api_key

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("NOVA_API_KEY")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("NOVA_API_BASE")
        if resolved_api_base is None:
            raise ValueError(
                "api_base not set for Nova Chat Completions API. "
                "Set via api_base parameter or NOVA_API_BASE environment variable"
            )
        parent_get_complete_url: Final = cast(
            Callable[..., str],
            OpenAIGPTConfig.get_complete_url,
        )
        return parent_get_complete_url(
            self,
            api_base=resolved_api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            **optional_params,
        }
