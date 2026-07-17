from typing import Dict, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE = "https://api.moonshot.ai/anthropic/v1/messages"
_ANTHROPIC_SUFFIX = "/anthropic"
_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"
_MESSAGES_SUFFIX = "/v1/messages"
_V1_SUFFIX = "/v1"


class MoonshotAnthropicMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "moonshot"

    @staticmethod
    def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("MOONSHOT_API_KEY")

    @staticmethod
    def _get_anthropic_messages_api_base(api_base: Optional[str] = None) -> str:
        base_url = (
            api_base
            or get_secret_str("MOONSHOT_ANTHROPIC_API_BASE")
            or get_secret_str("MOONSHOT_API_BASE")
            or DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
        ).rstrip("/")

        if base_url.endswith(_MESSAGES_SUFFIX):
            if _ANTHROPIC_SUFFIX in base_url:
                return base_url
            base_url = base_url[: -len(_MESSAGES_SUFFIX)]
            if base_url.endswith(_V1_SUFFIX):
                base_url = base_url[: -len(_V1_SUFFIX)]
        if base_url.endswith(f"{_ANTHROPIC_SUFFIX}{_V1_SUFFIX}"):
            return f"{base_url}/messages"
        if base_url.endswith(_ANTHROPIC_SUFFIX):
            return f"{base_url}{_MESSAGES_SUFFIX}"
        if base_url.endswith(_CHAT_COMPLETIONS_SUFFIX):
            base_url = base_url[: -len(_CHAT_COMPLETIONS_SUFFIX)]
        elif base_url.endswith(_V1_SUFFIX):
            base_url = base_url[: -len(_V1_SUFFIX)]

        return f"{base_url}{_ANTHROPIC_SUFFIX}{_MESSAGES_SUFFIX}"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict[str, object],
        litellm_params: object,
        stream: Optional[bool] = None,
    ) -> str:
        return self._get_anthropic_messages_api_base(api_base=api_base)

    def validate_anthropic_messages_environment(
        self,
        headers: Dict[str, str],
        model: str,
        messages: List[Dict[str, object]],
        optional_params: Dict[str, object],
        litellm_params: object,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Optional[str]]:
        dynamic_api_key = self._get_api_key(api_key=api_key)
        if dynamic_api_key and "authorization" not in headers and "x-api-key" not in headers:
            authorization = dynamic_api_key
            if not authorization.lower().startswith("bearer "):
                authorization = f"Bearer {authorization}"
            headers["authorization"] = authorization
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers, api_base
