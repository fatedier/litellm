from unittest.mock import patch

from litellm.llms.nova.chat.transformation import NovaChatConfig


def test_chat_request_is_forwarded_without_provider_specific_transformation():
    config = NovaChatConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    optional_params = {
        "stream": True,
        "response_format": {"type": "json_object"},
        "extra_body": {"provider_option": "preserve"},
    }

    result = config.transform_request(
        model="logical-model",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert result == {"model": "logical-model", "messages": messages, **optional_params}


def test_chat_url_and_deployment_credentials_take_precedence_over_environment():
    config = NovaChatConfig()
    with patch.dict(
        "os.environ",
        {"NOVA_API_BASE": "https://env.example/v1", "NOVA_API_KEY": "env-key"},
    ):
        assert config._get_openai_compatible_provider_info(
            api_base="https://deployment.example/v1",
            api_key="deployment-key",
        ) == ("https://deployment.example/v1", "deployment-key")
        assert config.get_complete_url(
            api_base="https://deployment.example/v1",
            api_key="deployment-key",
            model="logical-model",
            optional_params={},
            litellm_params={},
        ) == "https://deployment.example/v1/chat/completions"


def test_chat_environment_fallbacks_are_used_when_deployment_values_are_missing():
    config = NovaChatConfig()
    with patch.dict(
        "os.environ",
        {"NOVA_API_BASE": "https://env.example/v1", "NOVA_API_KEY": "env-key"},
    ):
        assert config._get_openai_compatible_provider_info(api_base=None, api_key=None) == (
            "https://env.example/v1",
            "env-key",
        )


def test_chat_requires_a_nova_api_base_instead_of_defaulting_to_openai():
    config = NovaChatConfig()
    with patch.dict("os.environ", {}, clear=True):
        try:
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="logical-model",
                optional_params={},
                litellm_params={},
            )
        except ValueError as exc:
            assert "NOVA_API_BASE" in str(exc)
        else:
            raise AssertionError("Nova Chat Completions must not default to the OpenAI endpoint")
