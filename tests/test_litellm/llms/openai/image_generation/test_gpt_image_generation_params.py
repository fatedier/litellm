from collections.abc import Mapping
from queue import SimpleQueue
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from openai import OpenAI
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.llms.azure.image_generation import AzureGPTImageGenerationConfig
from litellm.llms.fal_ai.image_generation.stable_diffusion_transformation import (
    FalAIStableDiffusionConfig,
)
from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.llms.stability.image_generation.transformation import (
    StabilityImageGenerationConfig,
)
from litellm.types.utils import ImageResponse
from litellm.utils import get_optional_params_image_gen


class _PromptRequest(TypedDict):
    prompt: ReadOnly[str]


class _OutputFormatParams(TypedDict):
    output_format: ReadOnly[str]


class _EmptyParams(TypedDict):
    pass


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def test_azure_gpt_image_request_options_are_mapped():
    optional_params: Final = get_optional_params_image_gen(
        model="gpt-image-2",
        background="transparent",
        moderation="low",
        output_compression=80,
        output_format="webp",
        custom_llm_provider="azure",
        provider_config=AzureGPTImageGenerationConfig(),
    )

    assert optional_params["background"] == "transparent"
    assert optional_params["moderation"] == "low"
    assert optional_params["output_compression"] == 80
    assert optional_params["output_format"] == "webp"


def test_gpt_image_options_preserve_provider_specific_output_format_passthrough():
    stability_params: Final = get_optional_params_image_gen(
        model="stable-image-core",
        custom_llm_provider="stability",
        provider_config=StabilityImageGenerationConfig(),
        output_format="webp",
    )
    fal_params: Final = get_optional_params_image_gen(
        model="fal-ai/stable-diffusion-v35-medium",
        custom_llm_provider="fal_ai",
        provider_config=FalAIStableDiffusionConfig(),
        output_format="png",
    )

    assert stability_params["output_format"] == "webp"
    assert fal_params["output_format"] == "png"


@pytest.mark.parametrize("model", ("gpt-image-1", "gpt-image-1.5", "gpt-image-2"))
def test_gpt_image_request_options_reach_openai_provider(model: str):
    captured_requests: Final[SimpleQueue[Mapping[str, object]]] = SimpleQueue()

    def capture_request(request: httpx.Request) -> httpx.Response:
        captured_requests.put(_JSON_OBJECT_ADAPTER.validate_json(request.content))
        return httpx.Response(
            200,
            request=request,
            content=b'{"created":1,"data":[{"b64_json":"eA=="}]}',
        )

    http_client: Final = httpx.Client(transport=httpx.MockTransport(capture_request))
    client: Final = OpenAI(
        api_key="test-key",
        base_url="https://capture.invalid/v1",
        http_client=http_client,
    )
    try:
        litellm.image_generation(
            model=f"openai/{model}",
            prompt="A transparent icon",
            background="transparent",
            moderation="low",
            output_compression=80,
            output_format="webp",
            client=client,
        )
    finally:
        http_client.close()

    captured_request: Final = captured_requests.get_nowait()
    assert captured_request["background"] == "transparent"
    assert captured_request["moderation"] == "low"
    assert captured_request["output_compression"] == 80
    assert captured_request["output_format"] == "webp"


def test_response_preserves_requested_output_format():
    config: Final = GPTImageGenerationConfig()
    raw_response: Final = httpx.Response(
        status_code=200,
        content=b'{"created":1,"data":[{"b64_json":"image-data"}]}',
    )
    request_data: Final[_PromptRequest] = {"prompt": "A transparent icon"}
    optional_params: Final[_OutputFormatParams] = {"output_format": "webp"}
    litellm_params: Final[_EmptyParams] = {}

    result: Final = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data=request_data,
        optional_params=optional_params,
        litellm_params=litellm_params,
        encoding=None,
    )

    assert result.output_format == "webp"
