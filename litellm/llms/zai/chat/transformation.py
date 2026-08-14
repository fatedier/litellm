from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"
_ZAI_REASONING_EFFORT_MAP: Final = MappingProxyType(
    {
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
)
_ZAI_REASONING_PARAMS: Final = frozenset(("thinking", "reasoning_effort"))


def _normalize_zai_reasoning_effort(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _ZAI_REASONING_EFFORT_MAP.get(value, value)
    if isinstance(value, Mapping):
        mapping_value: Final = cast(Mapping[str, object], value)
        effort: Final = mapping_value.get("effort")
        if isinstance(effort, str):
            return _ZAI_REASONING_EFFORT_MAP.get(effort, effort)
    return None


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list[str]:
        base_params: Final = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        base_params.extend(("thinking", "reasoning_effort"))

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: matches BaseConfig.map_openai_params request contract
        optional_params: dict[str, object],  # mutable-ok: matches BaseConfig.map_openai_params request contract
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: returns mutable request params to the LiteLLM handler
        mapped_params: Final[dict[str, object]] = cast(
            dict[str, object],
            super().map_openai_params(  # pyright: ignore[reportUnknownMemberType]  # inherited base contract is untyped
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            ),
        )
        thinking_value: Final[object | None] = mapped_params.get("thinking")
        reasoning_effort: Final = (
            _normalize_zai_reasoning_effort(mapped_params["reasoning_effort"])
            if "reasoning_effort" in mapped_params
            else None
        )
        params_without_reasoning: Final[dict[str, object]] = {  # mutable-ok: shared optional-params pipeline mutates request params
            key: value for key, value in mapped_params.items() if key not in _ZAI_REASONING_PARAMS
        }
        params_with_reasoning_effort: Final[dict[str, object]] = (
            {  # mutable-ok: shared optional-params pipeline mutates request params
                **params_without_reasoning,
                "reasoning_effort": reasoning_effort,
            }
            if reasoning_effort is not None
            else params_without_reasoning
        )
        if not isinstance(thinking_value, dict):
            return params_with_reasoning_effort

        existing_extra_body: Final[object | None] = params_with_reasoning_effort.get("extra_body")
        extra_body: Final[dict[str, object]] = (
            cast(dict[str, object], existing_extra_body) if isinstance(existing_extra_body, dict) else {}
        )
        thinking_extra_body: Final[dict[str, object]] = {  # mutable-ok: JSON request body
            "thinking": cast(dict[str, object], thinking_value)
        }
        merged_extra_body: Final[dict[str, object]] = {**extra_body, **thinking_extra_body}  # mutable-ok: JSON request body
        return {  # mutable-ok: JSON request body returned to the LiteLLM handler
            **params_with_reasoning_effort,
            "extra_body": merged_extra_body,
        }
