from litellm.llms.volcengine.messages.transformation import (
    DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE,
    VolcEngineAnthropicMessagesConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_should_map_volcengine_bases_to_anthropic_messages_endpoint():
    config = VolcEngineAnthropicMessagesConfig()

    assert config.get_complete_url(None, None, "doubao-seed", {}, {}) == DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE
    for api_base in (
        "https://ark.cn-beijing.volces.com/api/compatible",
        "https://ark.cn-beijing.volces.com/api/compatible/v1",
        "https://ark.cn-beijing.volces.com/api/v3",
        "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE,
    ):
        assert (
            config.get_complete_url(api_base, None, "doubao-seed", {}, {})
            == DEFAULT_VOLCENGINE_ANTHROPIC_MESSAGES_API_BASE
        )


def test_should_prepare_volcengine_anthropic_messages_headers():
    config = VolcEngineAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="doubao-seed",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="volcengine-key",
        api_base="https://ark.cn-beijing.volces.com/api/compatible",
    )

    assert headers["authorization"] == "Bearer volcengine-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://ark.cn-beijing.volces.com/api/compatible"


def test_should_preserve_existing_volcengine_auth_header():
    config = VolcEngineAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"Authorization": "Bearer existing-key"},
        model="doubao-seed",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="volcengine-key",
    )

    assert headers["Authorization"] == "Bearer existing-key"
    assert "authorization" not in headers


def test_should_route_volcengine_provider_to_native_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="doubao-seed-2-1-pro-260628",
        provider=LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineAnthropicMessagesConfig)
