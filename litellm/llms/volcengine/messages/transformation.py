from typing import Dict, List, Optional, Tuple

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str

DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE = "https://ark.cn-beijing.volces.com/api/compatible/v1/messages"
_API_COMPATIBLE_SUFFIX = "/api/compatible"
_API_V3_SUFFIX = "/api/v3"
_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
_MESSAGES_SUFFIX = "/v1/messages"
_V1_SUFFIX = "/v1"


class VolcEngineAnthropicMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "volcengine"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("ARK_API_KEY") or get_secret_str("VOLCENGINE_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("VOLCENGINE_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE
        )

    @classmethod
    def _get_anthropic_messages_api_base(cls, api_base: Optional[str] = None) -> str:
        base_url = cls.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith(_MESSAGES_SUFFIX):
            return base_url
        if base_url.endswith(f"{_API_COMPATIBLE_SUFFIX}{_V1_SUFFIX}"):
            return f"{base_url}/messages"
        if base_url.endswith(_API_COMPATIBLE_SUFFIX):
            return f"{base_url}{_MESSAGES_SUFFIX}"

        replaceable_suffixes = (
            f"{_API_V3_SUFFIX}{_CHAT_COMPLETIONS_SUFFIX}",
            _API_V3_SUFFIX,
            f"{_V1_SUFFIX}{_CHAT_COMPLETIONS_SUFFIX}",
            _CHAT_COMPLETIONS_SUFFIX,
            _V1_SUFFIX,
        )
        for suffix in replaceable_suffixes:
            if base_url.endswith(suffix):
                return f"{base_url[: -len(suffix)]}{_API_COMPATIBLE_SUFFIX}{_MESSAGES_SUFFIX}"

        return f"{base_url}{_API_COMPATIBLE_SUFFIX}{_MESSAGES_SUFFIX}"

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
        dynamic_api_key = self.get_api_key(api_key=api_key)
        if dynamic_api_key and not any(key.lower() == "authorization" for key in headers):
            authorization = dynamic_api_key
            if not authorization.lower().startswith("bearer "):
                authorization = f"Bearer {authorization}"
            headers["authorization"] = authorization
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers, api_base
