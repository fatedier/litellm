import litellm

from litellm.llms.nova.chat.transformation import NovaChatConfig
from litellm.llms.nova.messages.transformation import NovaMessagesConfig
from litellm.llms.nova.responses.transformation import NovaResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_nova_provider_is_registered_without_shadowing_existing_nova_providers():
    assert LlmProviders.NOVA in litellm.provider_list

    assert litellm.get_llm_provider("nova/logical-model", api_base="https://gateway.example/v1", api_key="sk") == (
        "logical-model",
        "nova",
        "sk",
        "https://gateway.example/v1",
    )
    assert litellm.get_llm_provider("amazon_nova/model", api_base="https://gateway.example/v1", api_key="sk")[1] == (
        "amazon_nova"
    )
    assert litellm.get_llm_provider(
        "sagemaker_nova/model", api_base="https://gateway.example/v1", api_key="sk"
    )[1] == "sagemaker_nova"


def test_nova_provider_configs_are_selected_for_each_supported_protocol():
    chat_config = ProviderConfigManager.get_provider_chat_config(
        model="logical-model",
        provider=LlmProviders.NOVA,
    )
    responses_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=LlmProviders.NOVA,
        model="logical-model",
    )
    messages_config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="logical-model",
        provider=LlmProviders.NOVA,
    )

    assert isinstance(chat_config, NovaChatConfig)
    assert isinstance(responses_config, NovaResponsesAPIConfig)
    assert isinstance(messages_config, NovaMessagesConfig)
    assert chat_config.custom_llm_provider == LlmProviders.NOVA
    assert responses_config.custom_llm_provider == LlmProviders.NOVA
    assert messages_config.custom_llm_provider == LlmProviders.NOVA


def test_nova_deployment_pricing_uses_standard_token_usage_fields(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "nova/logical-model",
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 0.1e-6,
            "cache_creation_input_token_cost": 0.5e-6,
            "output_cost_per_reasoning_token": 3e-6,
            "litellm_provider": "nova",
            "mode": "chat",
        },
    )

    response = {
        "model": "logical-model",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }

    assert litellm.completion_cost(
        completion_response=response,
        model="logical-model",
        custom_llm_provider="nova",
    ) == 24.4e-6
