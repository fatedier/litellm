from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel

from litellm.types.guardrails import (
    GuardrailEventHooks,
    Mode,
    SupportedGuardrailIntegrations,
)
from litellm.types.proxy.guardrails.guardrail_hooks.aigateway_moderation import (
    AIGatewayModerationOptionalParams,
)

from .aigateway_moderation import AIGatewayModeration

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


class _CallbackManager(Protocol):
    def add_litellm_callback(self, callback: AIGatewayModeration) -> None: ...


def _coerce_event_hook(
    mode: str | list[str] | Mode,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> AIGatewayModeration:
    import litellm

    raw_optional_params = getattr(litellm_params, "optional_params", None)
    if isinstance(raw_optional_params, Mapping):
        if "fail_on_error" in raw_optional_params:
            raise ValueError("fail_on_error must be configured at litellm_params.fail_on_error")
        optional_params = AIGatewayModerationOptionalParams.model_validate(raw_optional_params)
    elif isinstance(raw_optional_params, BaseModel):
        if "fail_on_error" in raw_optional_params.model_fields_set:
            raise ValueError("fail_on_error must be configured at litellm_params.fail_on_error")
        configured_fields = raw_optional_params.model_fields_set.intersection(
            AIGatewayModerationOptionalParams.model_fields
        )
        optional_params = AIGatewayModerationOptionalParams.model_validate(
            {
                **raw_optional_params.model_dump(include=configured_fields),
                **(raw_optional_params.model_extra or {}),
            }
        )
    else:
        optional_params = AIGatewayModerationOptionalParams()
    callback = AIGatewayModeration(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
        fail_on_error=True if litellm_params.fail_on_error is None else litellm_params.fail_on_error,
        decision_timeout_seconds=optional_params.decision_timeout_seconds,
        streaming_review_chunk_size=optional_params.streaming_review_chunk_size,
        streaming_review_context_size=optional_params.streaming_review_context_size,
    )
    callback_manager = cast(  # cast-ok: package global is a concrete callback manager with imprecise typing
        _CallbackManager, litellm.logging_callback_manager
    )
    callback_manager.add_litellm_callback(callback)
    return callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AIGATEWAY_MODERATION.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.AIGATEWAY_MODERATION.value: AIGatewayModeration,
}
