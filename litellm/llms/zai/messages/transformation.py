from typing import Optional, Protocol, cast

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AnthropicMessagesRequest,
    AnthropicMessagesRequestOptionalParams,
)
from litellm.types.router import GenericLiteLLMParams

DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE = "https://api.z.ai/api/anthropic/v1/messages"
_ANTHROPIC_SUFFIX = "/api/anthropic"
_MESSAGES_SUFFIX = "/v1/messages"
_ZAI_CHAT_SUFFIXES = (
    "/api/coding/paas/v4/chat/completions",
    "/api/paas/v4/chat/completions",
    "/api/coding/paas/v4",
    "/api/paas/v4",
)


class _TransformAnthropicMessagesRequest(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[AllAnthropicMessageValues],
        anthropic_messages_optional_request_params: AnthropicMessagesRequestOptionalParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> AnthropicMessagesRequest: ...


class _ValidateAnthropicMessagesEnvironment(Protocol):
    def __call__(
        self,
        *,
        headers: dict[str, str],
        model: str,
        messages: list[object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: Optional[str],
        api_base: Optional[str],
    ) -> tuple[dict[str, str], Optional[str]]: ...


class ZAIAnthropicMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def should_filter_anthropic_beta_headers(self) -> bool:
        return False

    def transform_anthropic_messages_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # bare base types
        self,
        model: str,
        messages: list[AllAnthropicMessageValues],
        anthropic_messages_optional_request_params: AnthropicMessagesRequestOptionalParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> AnthropicMessagesRequest:
        is_adaptive_thinking_model = (
            AnthropicModelInfo._is_adaptive_thinking_model  # pyright: ignore[reportPrivateUsage]  # shared resolver
        )
        if (
            model.lower().removeprefix("zai/") in {"glm-5.3", "glm-5.3-flash"}
            or is_adaptive_thinking_model(model, self._resolved_provider)
        ):
            reasoning_effort = (
                anthropic_messages_optional_request_params["reasoning_effort"]
                if "reasoning_effort" in anthropic_messages_optional_request_params
                else None
            )
            if reasoning_effort == "medium":
                anthropic_messages_optional_request_params["reasoning_effort"] = "high"

            output_config = (
                anthropic_messages_optional_request_params["output_config"]
                if "output_config" in anthropic_messages_optional_request_params
                else None
            )
            if isinstance(output_config, dict) and output_config.get("effort") == "medium":
                anthropic_messages_optional_request_params["output_config"] = {
                    **output_config,
                    "effort": "high",
                }

        parent_transform = cast(
            _TransformAnthropicMessagesRequest,
            super().transform_anthropic_messages_request,
        )
        return parent_transform(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("ZAI_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("ZAI_ANTHROPIC_API_BASE")
            or get_secret_str("ZAI_API_BASE")
            or DEFAULT_ZAI_ANTHROPIC_MESSAGES_API_BASE
        )

    @classmethod
    def _get_anthropic_messages_api_base(cls, api_base: Optional[str] = None) -> str:
        base_url = cls.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith(_MESSAGES_SUFFIX):
            return base_url
        if base_url.endswith(f"{_ANTHROPIC_SUFFIX}/v1"):
            return f"{base_url}/messages"
        if base_url.endswith(_ANTHROPIC_SUFFIX):
            return f"{base_url}{_MESSAGES_SUFFIX}"
        for suffix in _ZAI_CHAT_SUFFIXES:
            if base_url.endswith(suffix):
                return f"{base_url[: -len(suffix)]}{_ANTHROPIC_SUFFIX}{_MESSAGES_SUFFIX}"
        if base_url.endswith("/v1"):
            return f"{base_url}/messages"
        return f"{base_url}{_ANTHROPIC_SUFFIX}{_MESSAGES_SUFFIX}"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: Optional[bool] = None,
    ) -> str:
        return self._get_anthropic_messages_api_base(api_base=api_base)

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> tuple[dict[str, str], Optional[str]]:
        parent_validate = cast(
            _ValidateAnthropicMessagesEnvironment,
            super().validate_anthropic_messages_environment,
        )
        return parent_validate(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self.get_api_key(api_key=api_key),
            api_base=api_base,
        )
