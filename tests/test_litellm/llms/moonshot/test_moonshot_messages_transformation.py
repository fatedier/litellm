from litellm.llms.moonshot.messages.transformation import (
    DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE,
    MoonshotAnthropicMessagesConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_should_map_moonshot_base_to_anthropic_messages_endpoint():
    config = MoonshotAnthropicMessagesConfig()

    assert (
        config.get_complete_url("https://api.moonshot.cn/v1", None, "kimi-k2.6", {}, {})
        == "https://api.moonshot.cn/anthropic/v1/messages"
    )
    assert config.get_complete_url(None, None, "kimi-k2.6", {}, {}) == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    assert (
        config.get_complete_url("https://api.moonshot.ai/anthropic", None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url("https://api.moonshot.ai/anthropic/v1", None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url("https://api.moonshot.ai/v1", None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url("https://api.moonshot.ai/v1/chat/completions", None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url("https://api.moonshot.ai/v1/messages", None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )
    assert (
        config.get_complete_url(DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE, None, "kimi-k2.6", {}, {})
        == DEFAULT_MOONSHOT_ANTHROPIC_MESSAGES_API_BASE
    )


def test_should_prepare_moonshot_anthropic_messages_headers():
    config = MoonshotAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="kimi-k2.6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        api_key="moonshot-key",
        api_base="https://api.moonshot.ai/anthropic",
    )

    assert headers["authorization"] == "Bearer moonshot-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://api.moonshot.ai/anthropic"


def test_should_preserve_existing_moonshot_auth_header():
    config = MoonshotAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"authorization": "Bearer existing-key"},
        model="kimi-k2.6",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="moonshot-key",
    )

    assert headers["authorization"] == "Bearer existing-key"


def test_should_route_moonshot_provider_to_native_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="kimi-k2.6",
        provider=LlmProviders.MOONSHOT,
    )

    assert isinstance(config, MoonshotAnthropicMessagesConfig)
