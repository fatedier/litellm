from pydantic import BaseModel, ConfigDict, Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class AIGatewayModerationOptionalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Maximum time to submit streamed content and wait for the final moderation decision",
    )
    streaming_review_chunk_size: int = Field(
        default=256,
        gt=0,
        description="Minimum accumulated streamed response characters that trigger a moderation review",
    )
    streaming_review_context_size: int = Field(
        default=256,
        ge=0,
        description="Number of preceding streamed response characters included with each moderation batch",
    )


class AIGatewayModerationConfigModel(GuardrailConfigModel[AIGatewayModerationOptionalParams]):
    api_base: str = Field(description="Base URL of the AIGateway moderation adapter")
    api_key: str = Field(description="API key for the AIGateway moderation adapter")
    fail_on_error: bool = Field(
        default=True,
        description="Whether moderation service errors should block the request",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "AIGateway Moderation"
